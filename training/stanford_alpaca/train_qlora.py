#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from array import array
import copy
from genericpath import isdir
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, BitsAndBytesConfig, AutoTokenizer
from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import os
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import sys
sys.path.append("/opt/tiger/Cherry_LLM")
from template import get_formatting_prompts_func
from datasets import load_dataset, load_from_disk
os.environ["TOKENIZERS_PARALLELISM"]="true"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ScriptArguments:
    lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    lora_target_modules: Optional[List[str]] = field(default_factory=list, metadata={"help": "target_modules"})
    template: Optional[str] = field(default="alpaca")
    use_reentrant: Optional[bool] = field(default=True)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    unsloth: int = field(default=1)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_name: str = field(default="alpaca")

@dataclass
class BitsAndBytesArguments:
    use_nested_quant: bool = field(default=False)
    bnb_4bit_quant_storage_dtype: str = field(default="bfloat16")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if os.path.isdir(data_path):   
            list_data_dict = load_from_disk(data_path)
        elif "parquet" in data_path: 
            list_data_dict = load_dataset("parquet", data_files=data_path, split="train")
            # print(list_data_dict[0].keys())
            rename_dict = {'inputs_pretokenized':"instruction","targets_pretokenized":"output"}
            # print(list_data_dict.keys())
            list_data_dict = list_data_dict.rename_columns(rename_dict)
        else: list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def get_quant_model(model_args, training_args, script_args):
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, quantization_config=quantization_config)

    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=training_args.model_max_length)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    # print(model)
    model.print_trainable_parameters()
    return model, tokenizer

def get_unsloth_model(model_args, training_args, script_args):
    max_seq_length = 2048
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    if not is_adapter_checkpoint(model_args.model_name_or_path):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_args.model_name_or_path,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = True,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_args.model_name_or_path,
            dtype = dtype,
            load_in_4bit=True
        )
    # Do model patching and add fast LoRA weights
    if not is_adapter_checkpoint(model_args.model_name_or_path):
        model.enable_input_require_grads()
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            max_seq_length = max_seq_length,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    return model, tokenizer

def is_adapter_checkpoint(path):
    if not os.path.exists(path) or "adapter_model.safetensors" not in os.listdir(path): return False
    else: return True

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ScriptArguments, BitsAndBytesArguments))
    model_args, data_args, training_args, script_args, bnb_args = parser.parse_args_into_dataclasses()

    # gradient_checkpointing_kwargs={"use_reentrant":script_args.use_reentrant}
    # training_args.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    # formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token) #只有'alpaca'和'vicuna' template， 返回一个函数，用于对输入进行预处理， response_template='\n### Response:' or ' ASSISTANT:'
    # response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    # data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    # train_dataset = load_dataset("json", data_files=data_args.data_path)["train"]
    # if data_args.dataset_name=="alpaca":
    #     train_dataset = train_dataset.rename_column("output", "response")
    model: LlamaForCausalLM
    if not model_args.unsloth in model_args.model_name_or_path:
        model, tokenizer = get_quant_model(model_args, training_args, script_args)
    else:
        model, tokenizer = get_unsloth_model(model_args, training_args, script_args)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=training_args, peft_config=lora_config, train_dataset=train_dataset, 
    # data_collator=data_collator, formatting_func=formatting_prompts_func, max_seq_length=training_args.model_max_length, )
    # with torch.autocast("cuda"): 
    trainer.train()

    # model.save_pretrained(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
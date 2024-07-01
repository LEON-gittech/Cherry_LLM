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
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, BitsAndBytesConfig, AutoTokenizer, T5Model, T5ForConditionalGeneration
# from unsloth import FastLanguageModel 
# from unsloth import is_bfloat16_supported
import os
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoModel, DataCollatorWithPadding
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import sys
sys.path.append("/opt/tiger/Cherry_LLM")
from template import get_formatting_prompts_func
from datasets import load_dataset
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
    assert targets is not None
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    assert labels is not None
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if "parquet" in data_path: 
            list_data_dict = load_dataset("parquet", data_files=data_path, split="train").take(1000)
            # print(list_data_dict[0].keys())
            rename_dict = {'inputs_pretokenized':"instruction","targets_pretokenized":"output"}
            # print(list_data_dict.keys())
            list_data_dict = list_data_dict.rename_columns(rename_dict)
            # print(list_data_dict.column_names)
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
        # print(len(self.labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert self.labels[i] is not None
        return dict(input_ids=self.input_ids[i],labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        # print(instances[0])
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # # 目标长度
        # target_input_length = 768
        # target_label_length = 256

        # # 对于 input_ids
        # # print("input_ids",len(input_ids[0]))
        # padded_input_ids = []
        # for ids in input_ids:
        #     if len(ids) < target_input_length:
        #         # 不足长度的序列，在末尾添加pad_token_id直到达到目标长度
        #         ids = torch.cat((ids, torch.tensor([self.tokenizer.pad_token_id] * (target_input_length - len(ids)))))
        #     elif len(ids) > target_input_length:
        #         # 超过长度的序列，截断到目标长度
        #         ids = ids[:target_input_length]
        #     padded_input_ids.append(ids)

        # # 对于 labels
        # padded_labels = []
        # for label in labels:
        #     if len(label) < target_label_length:
        #         # 不足长度的序列，在末尾添加IGNORE_INDEX直到达到目标长度
        #         label = torch.cat((label, torch.tensor([IGNORE_INDEX] * (target_label_length - len(label)))))
        #     elif len(label) > target_label_length:
        #         # 超过长度的序列，截断到目标长度
        #         label = label[:target_label_length]
        #     padded_labels.append(label)
        # from torch.nn.utils.rnn import pad_sequence
        # # 现在使用pad_sequence，因为我们已经手动保证了长度一致，所以这步主要为了统一batch中的tensor形状
        # input_ids = pad_sequence(padded_input_ids, batch_first=True)
        # # print("input_ids", input_ids.shape)
        # labels = pad_sequence(padded_labels, batch_first=True)
        # # print("labels", labels.shape)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if input_ids[0].shape[0]>768: input_ids = input_ids[:, :768]
        if labels[0].shape[0]>256: labels = labels[:, :256]
        # input_ids = torch.stack([ids[:768] if ids.shape[0] > 768 else ids for ids in input_ids])
        # labels = torch.stack([label[:256] if label.shape[0] > 256 else label for label in labels])
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules=script_args.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    return model, tokenizer

from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ScriptArguments, BitsAndBytesArguments))
    model_args, data_args, training_args, script_args, bnb_args = parser.parse_args_into_dataclasses()
    accelerator = Accelerator()
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype, quantization_config=quantization_config).cuda()

    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=training_args.model_max_length)
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.QUESTION_ANS,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model,
    # )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=training_args, peft_config=lora_config, train_dataset=train_dataset, 
    # data_collator=data_collator, formatting_func=formatting_prompts_func, max_seq_length=training_args.model_max_length, )
    # with torch.autocast("cuda"): 
    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if accelerator.is_main_process:
        print("saving model...")
        # accelerator.save_state(output_dir=training_args.output_dir)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
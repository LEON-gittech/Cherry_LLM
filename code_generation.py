import argparse
from unsloth import FastLanguageModel 
from transformers import TextStreamer, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from peft import PeftModelForCausalLM

base_path = "/mnt/bn/data-tns-live-llm/leon/datasets/fed/"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default=None,
)
args = parser.parse_args()
model: PeftModelForCausalLM
model, tokenizer = FastLanguageModel.from_pretrained(os.path.join(base_path, args.model_name_or_path), dtype = torch.bfloat16, load_in_4bit=True)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)
# model = AutoModelForCausalLM.from_pretrained("/mnt/bn/data-tns-live-llm/leon/datasets/starcoder2-7b/", torch_dtype=torch.bfloat16).cuda()
# tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/data-tns-live-llm/leon/datasets/starcoder2-7b/")
model.generation_config.pad_token_id = tokenizer.eos_token_id

def generate_one_completion(instance):
    inputs = tokenizer(instance, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    generate_ids = model.generate(input_ids, max_length=1024, repetition_penalty=1.1, streamer=text_streamer, do_sample=True)
    # generate_ids = model.generate(input_ids, max_length=1024, repetition_penalty=1.1, do_sample=True)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(outputs)
    return outputs
import sys
from tqdm import tqdm
sys.path.append("/opt/tiger/human-eval")
from human_eval.data import write_jsonl, read_problems

problems = read_problems()
# problems = {k: problems[k] for k in list(problems.keys())[:50]}

# for i, task_id in enumerate(problems):
#     print(task_id)
#     print(problems[task_id])
#     if i==10:break
print(args.model_name_or_path)
num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    # for _ in range(num_samples_per_task)
]
write_jsonl(os.path.join(base_path,f"{args.model_name_or_path}.jsonl"), samples)
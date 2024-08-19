import argparse
import random
import time
from contextlib import contextmanager
import torch
from datasets import load_dataset, load_from_disk
from datasets import load_dataset, concatenate_datasets, load_from_disk
from vllm import LLM, SamplingParams
import evaluate
rouge = evaluate.load('rouge')

parser = argparse.ArgumentParser()
parser.add_argument("--setting",type=str,default=None)
parser.add_argument("--idx", type=int,default=4)
args = parser.parse_args()

base_data = load_from_disk(f"/mnt/bn/data-tns-live-llm/leon/datasets/privacy_data/{args.setting}_0.parquet/")
print(base_data["instruction"][:5])
print(len(base_data))

llm = LLM(model=f"/mnt/bn/data-tns-live-llm/leon/datasets/privacy/{args.setting}_merged/checkpoint-{args.idx}_merged", tensor_parallel_size=1, 
            dtype=torch.bfloat16, trust_remote_code=True, 
            enable_lora=False, max_model_len=2048, gpu_memory_utilization=0.8)

stop_tokens = ["USER:", "ASSISTANT:",  "### Instruction:", "Response:", 
                "\n\nProblem", "\nProblem", "Problem:", "<|eot_id|>", "####"]

sampling_params = SamplingParams(top_p=1, max_tokens=1024, repetition_penalty=1.1, top_k=40) # memory extraction
prefix = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

###Instruction:"""

inputs = [prefix]*100
outputs = llm.generate(inputs, sampling_params)
response = [output.outputs[0].text for output in outputs]
rougeLs = rouge.compute(predictions=response, references=[base_data["instruction"]]*100)["rougeL"] # memory extraction
print(rougeLs)
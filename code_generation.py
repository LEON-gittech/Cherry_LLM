import argparse
from unsloth import FastLanguageModel 
from transformers import TextStreamer, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from peft import PeftModelForCausalLM
from vllm import LLM, SamplingParams

# base_path = "/mnt/bn/data-tns-live-llm/leon/datasets/fed/"
# base_path = "/mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed"
base_path = "/mnt/bn/merlin-datavolume-tsy/leon/checkpoints/fed_setting_C"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",type=str,default=None,)
parser.add_argument("--vllm",type=int,default=1)
args = parser.parse_args()

if args.vllm:
    model, tokenizer = FastLanguageModel.from_pretrained(os.path.join(base_path, args.model_name_or_path), dtype = torch.bfloat16, load_in_4bit=True)
    model.save_pretrained_merged(os.path.join(base_path, args.model_name_or_path+"_merged"), tokenizer, save_method = "merged_16bit",)
    model = LLM(model=os.path.join(base_path, args.model_name_or_path+"_merged"), tensor_parallel_size=1, dtype=torch.bfloat16, trust_remote_code=True, enable_lora=False, max_model_len=2048, gpu_memory_utilization=0.8)

    stop_tokens = ["USER:", "ASSISTANT:",  "### Instruction:", "Response:", 
                "\n\nProblem", "\nProblem", "Problem:", "<|eot_id|>", "####"]
    sampling_params = SamplingParams(temperature=0.5, top_p=1, max_tokens=1024, repetition_penalty=1.1, stop=stop_tokens)
else:
    model: PeftModelForCausalLM
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(os.path.join(base_path, args.model_name_or_path), dtype = torch.bfloat16, load_in_4bit=True)
    except:
        model, tokenizer = FastLanguageModel.from_pretrained(args.model_name_or_path, dtype = torch.bfloat16, load_in_4bit=True)
        
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    text_streamer = TextStreamer(tokenizer)
    model.generation_config.pad_token_id = tokenizer.eos_token_id

def generate_one_completion(instance):
    inputs = tokenizer(instance, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    generate_ids = model.generate(input_ids, max_new_tokens=1024, repetition_penalty=1.1, streamer=text_streamer, do_sample=True)
    # generate_ids = model.generate(input_ids, max_length=1024, repetition_penalty=1.1, do_sample=True)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(outputs)
    return outputs

import sys
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems

problems = read_problems()
# problems = {k: problems[k] for k in list(problems.keys())[:50]}

# for i, task_id in enumerate(problems):
#     print(task_id)
#     print(problems[task_id])
#     if i==10:break
print(args.model_name_or_path)
num_samples_per_task = 1
if args.vllm:
    inputs = [v["prompt"] for v in problems.values()]
    outputs = model.generate(inputs, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    samples = [
        dict(task_id=task_id, completion=response)
        for task_id,response in zip(problems,responses)
    ]
else:
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
        for task_id in problems
        # for _ in range(num_samples_per_task)
    ]
    
write_jsonl(os.path.join(base_path,f"{args.model_name_or_path}.jsonl"), samples)
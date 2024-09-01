import argparse
from pprint import pprint as pp
import time
import random
from datasets import load_dataset, concatenate_datasets, load_from_disk
import pandas as pd
import datasets
from tqdm import tqdm
import torch
import sys
sys.path.append("/mnt/bn/data-tns-live-llm/leon/Cherry_LLM")
from niid_data.utils import process_sft_dataset
import requests
import json
from vllm_utils import vllm_run
import asyncio

def construct_client_data(args):
    if args.domain == "code":
        data = load_dataset("sahil2801/CodeAlpaca-20k")["train"]
        dataset_name = "sahil2801/CodeAlpaca-20k"
    elif args.domain == "fin":
        data = load_dataset("FinGPT/fingpt-sentiment-train")["train"]
        dataset_name = "FinGPT/fingpt-sentiment-train"
    elif args.domain == "med":  
        data = load_dataset("medalpaca/medical_meadow_medical_flashcards")["train"]
        dataset_name = "medalpaca/medical_meadow_medical_flashcards"
    else:
        data = load_dataset("TIGER-Lab/MathInstruct")["train"]
        dataset_name = "TIGER-Lab/MathInstruct"

    random.seed(42)
    processed_data = process_sft_dataset(dataset_name, data)
    iid_idxs = random.sample(range(len(processed_data)), 1000)
    base_data = processed_data.select(iid_idxs)
    clients_data = []
    for i in range(10):
        clients_data.append(base_data.shard(10,i))
    return clients_data

instruction_schema = """{
    "title": "Instruction Schema",
    "type": "object",
    "properties": {
        "Instruction": {
            "title": "instruction",
            "type": "string"
        }
    },
    "required": ["Instruction"]
}"""

response_schema = """{
    "title": "Response Schema",
    "type": "object",
    "properties": {
        "Response": {
            "title": "response",
            "type": "string"
        }
    },
    "required": ["Response"]
}"""

def get_vllm(prompt, schema):
    min_tokens=0
    if "Response" in schema: 
        min_tokens=50
        max_tokens = 512
    else:
        max_tokens = 256

    cnt=0
    while cnt!=2:
        cnt+=1
        output = None
        completion = client.chat.completions.create(
            # model="TheBloke/Llama-2-7b-Chat-AWQ",
            model="Meta-Llama-3-8B-Instruct/",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant designed to output response in JSON."},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=1.0,
            max_tokens=max_tokens,
            extra_body={
                "guided_json": schema,
                "min_tokens": min_tokens
            }
        )
        try:
            output = completion.choices[0].message.content
            break
        except Exception as e:
            print(f"gpt error {e}")
            print(f"gpt output {completion}")
    return output

def get_moonshot(prompt, format):
    cnt=0
    output = None
    while cnt!=2:
        cnt+=1
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant designed to output JSON. The format is: {format}"}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512,
            frequency_penalty=1.0,
            response_format={"type": "json_object"},
        )
        try: 
            output = completion.choices[0].message.content
            break
        except Exception as e:
            print(f"gpt error {e}")
            print(f"gpt output: {completion}\n")
            try: 
                print(completion.__dict__)
            except:
                print(completion)
            time.sleep(1)
            continue
    return output

def get_gpt(prompt, format):
    data = {
        "model": "gpt-3.5-turbo-0125",
        "messages":  [
            {"role": "system", "content": f"You are a helpful assistant designed to output JSON. The format is: {format}"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "stop": None,
        "max_tokens": 512,
        "presence_penalty": 0,
        "frequency_penalty": 1.0,
        "user": None,
        "response_format": {"type": "json_object"}
    }
    headers = {'Content-Type': 'application/json', 'Caller': 'leon.kepler'}
    data = {k: v for k, v in data.items() if v is not None}
    data = json.dumps(data)
    url = f"https://swzkkd0h.us-east-fn.bytedance.net/gpt/openapi/online/v2/crawl"
    cnt=0
    output = None
    while cnt!=2:
        cnt+=1
        response = requests.post(url, data=data, headers=headers)
        try: 
            output = response.json()["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(f"gpt error {e}")
            print(f"gpt output:\n")
            try: 
                print(response.__dict__)
            except:
                print(response)
            time.sleep(1)
            continue
    return output

import evaluate
rouge = evaluate.load('rouge')
def compute_sim(instruction, client_data):
    return rouge.compute(predictions=[instruction], references=[client_data["instruction"]],rouge_types=['rougeL'])["rougeL"]

instruction_prompt = """You are asked to come up with instructions. Don't repeat instructions in examples. Here are some examples: Instruction 1: {} Instruction 2: {} Provide a new instruction below:"""

instruction_format = "{Instruction: instruction}"
def generate_instruction(prompt, client_data, instruction_format=instruction_format):
    cnt=0
    instruction = client_data["instruction"][0]
    while cnt!=2:
        cnt+=1
        if args.api_method == "vllm":
            instruction = get_vllm(prompt, instruction_schema)
        elif args.api_method == "moonshot":
            instruction = get_moonshot(prompt,instruction_format)
        else: 
            instruction = get_gpt(prompt,instruction_format)
        try:
            instruction = json.loads(instruction)["Instruction"]
            break
        except Exception as e:
            print(f"generate instruction error {e}")
            print(instruction)
            time.sleep(1)
        # if compute_sim(instruction, client_data)>0.7: 
        #     print(f"too similar {instruction}")
        #     continue
    return instruction

response_format="{Response: response}"
def generate_response(prompt, response_format=response_format):
    cnt=0
    output = None
    while cnt!=2:
        cnt+=1
        if args.api_method == "vllm":
            output = get_vllm(prompt,response_schema)
        elif args.api_method == "moonshot":
            output = get_moonshot(prompt,response_format)
        else:
            output = get_gpt(prompt,response_format)
        try:
            output = json.loads(output)["Response"]
            break
        except:
            print(output)
            time.sleep(1)
    return output

response_format = """
Example 1: 
    Instruction: {}
    Response: {}

Generate the response of this instruction, Instruction: {}
"""
def format_response_prompt(instruction, response_example):
    response_seqs = []
    for data in response_example:
        response_seqs.extend([data["instruction"],data["response"]])
    response_seqs.append(instruction)
    prompt = response_format.format(*response_seqs)
    return prompt

import multiprocessing
parser = argparse.ArgumentParser()
parser.add_argument("--domain",type=str,default="code")
parser.add_argument("--port",type=int,default=8000)
parser.add_argument("--api_method",type=str,default="vllm")
parser.add_argument("--parallel",type=int,default=0)
args = parser.parse_args()

from openai import OpenAI

if args.api_method == "moonshot":
    client = OpenAI(
        api_key="sk-Nl2yxo0Ic9ZJ0zFf3qWdypuoQ6Nd2xnsCp7wfojlW9OyOYmq", 
        base_url="https://api.moonshot.cn/v1",
    )
elif args.api_method == "vllm":
    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1/",
        api_key="damn"
    )

gen_num = 100
clients_data = construct_client_data(args)
for i in range(10):
    clients_data[i].save_to_disk(f"/mnt/bn/data-tns-live-llm/leon/datasets/fed_data/gen_{args.domain}_base_{i}.parquet")

for k in range(10):
    print(f"Client {k}")
    instructions = []
    response_examples = []
    client_k_data = clients_data[k]
    for i in range(gen_num):
        instructions.append(random.sample(client_k_data["instruction"],2))
        response_examples.append(client_k_data.select(random.sample(range(len(client_k_data)),1)))

    if args.parallel:
        instructions_formatted = [instruction_prompt.format(*instruction) for instruction in instructions]
        instruction_result = asyncio.run(
            vllm_run(
                api_url=f"http://localhost:{args.port}/v1/completions",
                model_id="NousResearch/Meta-Llama-3-8B-Instruct",
                input_requests=instructions_formatted,
                format=instruction_schema
            ))
        gen_instructions = [json.loads(res.generated_text)["Instruction"] for res in instruction_result]
        response_formatted = [format_response_prompt(instruction,response_examples[i]) for i,instruction in zip(range(gen_num),gen_instructions)]
        response_result = asyncio.run(
            vllm_run(
                api_url=f"http://localhost:{args.port}/v1/completions",
                model_id="NousResearch/Meta-Llama-3-8B-Instruct",
                input_requests=response_formatted,
                format=response_schema
            ))
        gen_responses = [json.loads(res.generated_text)["Response"] for res in response_result]
    else:
        gen_instructions = []
        gen_responses = []
        for i in tqdm(range(gen_num)):
            tmp = instruction_prompt.format(*instructions[i])
            instruction = generate_instruction(tmp,client_k_data,instruction_format)
            prompt = format_response_prompt(instruction,response_examples[i])
            response = generate_response(prompt)
            gen_instructions.append(instruction)
            gen_responses.append(response)
    
    # gen_instructions = [json.dumps(instruction) for instruction in gen_instructions]
    # gen_response = [json.dumps(response) for response in gen_responses]
    df = pd.DataFrame({
        "instruction": gen_instructions,
        "response": gen_responses,
        "label": args.domain  
    })
    # df = df.dropna(subset=['instruction', 'response'])

    # df.to_csv(f"/mnt/bn/data-tns-live-llm/leon/Cherry_LLM/generate_instruction/{args.domain}_{k}.csv")
    tmp_dataset = datasets.Dataset.from_pandas(df)
    client_k_data = concatenate_datasets([client_k_data,tmp_dataset])
    client_k_data.to_csv(f"/mnt/bn/data-tns-live-llm/leon/Cherry_LLM/generate_instruction/{args.domain}_{k}.csv")
    client_k_data.save_to_disk(f"/mnt/bn/data-tns-live-llm/leon/datasets/fed_data/gen_{args.domain}_{k}.parquet")
    print(len(client_k_data))
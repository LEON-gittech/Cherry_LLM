from multiprocessing import Pool, cpu_count
from FlagEmbedding import FlagModel
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from pprint import pprint as pp
import time
import umap
import os
import random
import time
from contextlib import contextmanager
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk
from datasets import load_dataset, concatenate_datasets, load_from_disk
import pandas as pd
import datasets
from datasets import Dataset
from pprint import pprint as pp
from datasets import Dataset
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import heapq
from datasets import load_dataset, load_from_disk
from datasets import load_dataset, concatenate_datasets, load_from_disk
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from pprint import pprint as pp
from datasets import Dataset
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import heapq
code_data = load_dataset("sahil2801/CodeAlpaca-20k")["train"]
fin_data = load_dataset("FinGPT/fingpt-sentiment-train")["train"]
med_data = load_dataset("medalpaca/medical_meadow_medical_flashcards")["train"]
general_data = load_dataset("tatsu-lab/alpaca")["train"]
math_data = load_dataset("TIGER-Lab/MathInstruct")["train"]

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example

def process_sft_dataset(dataset_name, dataset, dataset_sample=None)->datasets.Dataset:
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4", "gbharti/finance-alpaca"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        # dataset = dataset.shuffle(seed=42).select(range(51000))
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])
    elif dataset_name in ["lighteval/MATH"]:
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(['level', 'type'])
    elif dataset_name in ['gsm8k']:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:       # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    elif "math" in dataset_name:
        dataset = dataset.remove_columns(['source'])
        dataset = dataset.rename_column("output", "response")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=42)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset
processed_data = []
# 这块一定要注意!!! name 和datasest的顺序都要改
for name, dataset in zip(["medalpaca/medical_meadow_medical_flashcards","lucasmccabe-lmi/CodeAlpaca-20k","TIGER-Lab/MathInstruct","FinGPT/fingpt-sentiment-train","tatsu-lab/alpaca",],[med_data,code_data,math_data,fin_data,general_data]):
# for name, dataset in zip(["lucasmccabe-lmi/CodeAlpaca-20k","FinGPT/fingpt-sentiment-train","medalpaca/medical_meadow_medical_flashcards", "TIGER-Lab/MathInstruct"],[code_data,fin_data,med_data,math_data]):
    tmp:datasets.Dataset = process_sft_dataset(name,dataset)
    # if "fin" in name: 
    #     tmp = tmp.shuffle(seed=42).select(range(51000))
    print(tmp.column_names)
    processed_data.append(tmp)
import openai
from tqdm import tqdm

from openai import OpenAI

client = OpenAI(api_key="sk-f593092a899348a1ab875ae7c4e24713", base_url="https://api.deepseek.com")

instructions = processed_data[0]["instruction"]
responses = processed_data[0]["response"]

# 定义判断主题的提示信息
def check_topic(instruction):
    prompt = f"""
    你是一名医学专家。请判断以下医学指令是否属于以下主题之一：
    - clinical knowledge
    - anatomy
    - medical genetics

    指令内容: "{instruction}"
    如果属于以上任何一个主题，请回答“属于”；否则，请回答“不属于”。
    """
    response = client.chat.completions.create(
        # model="gpt-4o-2024-08-06",  # 使用GPT-4模型
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,  # 确保答案稳定
        n=1,
        stop=["\n"],
    )
    # 提取回答结果
    answer = response.choices[0].message.content.strip()
    # print(answer)
    return answer == "属于"

# 包装函数以便于多进程使用
def process_instruction(data):
    instruction, response = data
    return check_topic(instruction), instruction, response

# 多进程处理
if __name__ == '__main__':
    with Pool(10) as pool:
        results = list(tqdm(pool.imap(process_instruction, zip(instructions, responses)), total=len(instructions)))

    included_topics = []
    excluded_topics = []

    for result in results:
        is_included, instruction, response = result
        if is_included:
            included_topics.append({"instruction": instruction, "response": response})
        else:
            excluded_topics.append({"instruction": instruction, "response": response})

    # 输出结果
    print("包含指定主题的指令集：", len(included_topics))
    print("不包含指定主题的指令集：", len(excluded_topics))

    if included_topics:
        print(included_topics[0])
    if excluded_topics:
        print(excluded_topics[0])

    # 将数据转换为 Huggingface Dataset 格式
    included_dataset = Dataset.from_list(included_topics)
    excluded_dataset = Dataset.from_list(excluded_topics)

    dataset_dict = DatasetDict({
        "included_topics": included_dataset,
        "excluded_topics": excluded_dataset
    })

    # 保存到磁盘
    dataset_dict.save_to_disk('medical_instructions')
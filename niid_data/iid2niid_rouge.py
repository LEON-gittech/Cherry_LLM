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

@contextmanager
def timer():
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.4f} seconds")
model = FlagModel('BAAI/bge-large-en-v1.5', 
                  query_instruction_for_retrieval="",
                  use_fp16=True,
                  checkpoint="/mnt/bn/data-tns-live-llm/leon/Cherry_LLM/training/stanford_alpaca/results/checkpoint-11000"
                )
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
from rouge_score import rouge_scorer

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
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset
processed_data = []
for name, dataset in zip(["lucasmccabe-lmi/CodeAlpaca-20k","FinGPT/fingpt-sentiment-train","medalpaca/medical_meadow_medical_flashcards","tatsu-lab/alpaca","TIGER-Lab/MathInstruct"],[code_data,fin_data,med_data,general_data,math_data]):
# for name, dataset in zip(["lucasmccabe-lmi/CodeAlpaca-20k","FinGPT/fingpt-sentiment-train","medalpaca/medical_meadow_medical_flashcards", "TIGER-Lab/MathInstruct"],[code_data,fin_data,med_data,math_data]):
    tmp:datasets.Dataset = process_sft_dataset(name,dataset)
    print(tmp.column_names)
    processed_data.append(tmp)
def labeling(example, label):
    example["label"] = label
    return example
label = ["code","fin","med", "gen", "math"]

for i, data in enumerate(processed_data):
    data = data.map(lambda example: labeling(example, label[i]), batched=False)
    processed_data[i] = data
# data_concated: Dataset = concatenate_datasets(processed_data)
data_concated: Dataset = processed_data[0]
iid_idxs = random.sample(range(len(data_concated)), 1000)
base_data = data_concated.select(iid_idxs)
clients_data = []
for i in range(10):
    clients_data.append(base_data.shard(10,i))

data_concated = data_concated.select(list(set(range(len(data_concated)))-set(iid_idxs)))
print(len(data_concated))
k=10
from sklearn.cluster import MiniBatchKMeans, KMeans
base_0_embeddings = model.encode(clients_data[0]["instruction"])
# 假设 embeddings 是你的嵌入数据
kmeans = KMeans(n_clusters=k, random_state=0).fit(base_0_embeddings)
labels = kmeans.labels_
# 计算每个簇的样本数量
counts = np.bincount(labels)
# 找到最大的簇的标签
largest_cluster_label = np.argmax(counts)
# 从 cluster_centers_ 中获取最大的簇的中心
cluster_center_0:np.array = kmeans.cluster_centers_[largest_cluster_label]
print(cluster_center_0.shape)
client_clusters = cluster_center_0.reshape((1,1024))
print(kmeans.cluster_centers_.shape)
for i in range(10-1):
    i=i+1
    base_i_embeddings = model.encode(clients_data[i]["instruction"])
    # 假设 embeddings 是你的嵌入数据
    kmeans = KMeans(n_clusters=k, random_state=0).fit(base_i_embeddings)
    labels = kmeans.labels_
    similarity_scores = np.sum(kmeans.cluster_centers_ @ client_clusters.T, axis=-1)
    print(similarity_scores.shape)
    selected_idxs = np.argsort(similarity_scores)[i:]
    # 计算每个簇的样本数量
    counts = np.bincount(labels)
    # 找到最大的簇的标签
    largest_cluster_labels = np.argsort(-counts) #降序
    largest_cluster_label = -1
    for i in largest_cluster_labels:
        if i in selected_idxs:
            largest_cluster_label = i
    # 从 cluster_centers_ 中获取最大的簇的中心
    largest_cluster_center = kmeans.cluster_centers_[largest_cluster_label]
    client_clusters = np.concatenate([client_clusters,largest_cluster_center.reshape((1,1024))])
data_concated: Dataset = concatenate_datasets(processed_data)
data_concated = data_concated.select(list(set(range(len(data_concated)))-set(iid_idxs)))
print(len(data_concated))
concated_embeddings = model.encode(data_concated["instruction"])
concated_embeddings = torch.tensor(concated_embeddings, dtype=torch.float32)
client_clusters = torch.tensor(client_clusters, dtype=torch.float32)

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def get_rouge_l_score(text1, text2):
    return scorer.score(text1, text2)['rougeL'].fmeasure

def filter_non_similar(chunk, sampled_instructions):
    filtered_idxs = []
    for idx, row in chunk:
        is_similar = False
        for sample_instr in sampled_instructions:
            rouge_l_score = get_rouge_l_score(sample_instr, row['instruction'])
            if rouge_l_score >= 0.7:
                is_similar = True
                break
        if not is_similar:
            filtered_idxs.append(idx)
    return filtered_idxs

import multiprocessing as mp
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
import numpy as np
import random

def parallel_filter(data_concated, sampled_instructions, num_processes):
    chunk_size = len(data_concated) // num_processes
    chunks = [(list(enumerate(data_concated))[i:i + chunk_size], sampled_instructions) for i in range(0, len(data_concated), chunk_size)]
    
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(filter_non_similar, chunks)
    
    # Flatten list of lists
    filtered_idxs = [idx for sublist in results for idx in sublist]
    return filtered_idxs

def main():
    num_processes = mp.cpu_count()
    print(num_processes)
    client_pos_datasets = []
    for i, sampled_data in enumerate(clients_data):
        print(i)
        sampled_instructions = sampled_data['instruction']
        filtered_idxs = parallel_filter(data_concated, sampled_instructions, num_processes)
        
        filtered_data = data_concated.select(filtered_idxs)
        filtered_embeddings = concated_embeddings[filtered_idxs]

        similarity_scores = torch.matmul(client_clusters[i, :].cuda(), filtered_embeddings.T.cuda()).cpu()
        top_idxs = heapq.nlargest(5000, range(len(similarity_scores)), key=lambda x: similarity_scores[x])

        pos_datasets = filtered_data.select(top_idxs)
        pos_datasets = concatenate_datasets([pos_datasets, sampled_data])
        pos_datasets = pos_datasets.shuffle(seed=42)

        client_pos_datasets.append(pos_datasets)

    random.seed(10)
    client_pos_datasets = []
    for i, sampled_data in enumerate(clients_data):
        print(i)
        similarity_scores = torch.matmul(client_clusters[i,:].cuda(), (concated_embeddings.T).cuda()).cpu()
        top_idxs = heapq.nlargest(5000, range(len(similarity_scores)-1), key=lambda x: similarity_scores[x])
        pos_datasets: Dataset = []
        pos_datasets = data_concated.select(top_idxs)
        pos_datasets = concatenate_datasets([pos_datasets, sampled_data])
        pos_datasets = pos_datasets.shuffle(seed=42)
        client_pos_datasets.append(pos_datasets)
    for i, pos_data in enumerate(client_pos_datasets):
        pos_data.save_to_disk(f"/mnt/bn/data-tns-live-llm/leon/datasets/fed_data/iid2niid_code_public_sft_{i}.parquet")

if __name__ == '__main__':
    main()
# %%
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
# for name, dataset in zip(["lucasmccabe-lmi/CodeAlpaca-20k","FinGPT/fingpt-sentiment-train","medalpaca/medical_meadow_medical_flashcards","tatsu-lab/alpaca","TIGER-Lab/MathInstruct"],[code_data,fin_data,med_data,general_data,math_data]):
for name, dataset in zip(["lucasmccabe-lmi/CodeAlpaca-20k","FinGPT/fingpt-sentiment-train","medalpaca/medical_meadow_medical_flashcards", "TIGER-Lab/MathInstruct"],[code_data,fin_data,med_data,math_data]):
    tmp:datasets.Dataset = process_sft_dataset(name,dataset)
    print(tmp.column_names)
    processed_data.append(tmp)

# %%
DOMAIN_LABELS = {
    'code': 0,
    'finance': 1,
    'medical': 2,
    'math': 3
}

for i, name in zip(range(4),["code","finance","medical","math"]):
    processed_data[i] = processed_data[i].add_column('domain', [DOMAIN_LABELS[name]] * len(processed_data[i]))

# Concatenate the datasets
concated_data = concatenate_datasets(processed_data)

# %%
concated_data = concated_data.shuffle()
print(concated_data[:5])

# %%
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5', torch_dtype=torch.bfloat16).cuda()

def get_embedding(input_text, model):
    encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
    model_output = model(**encoded_input)
    instruction_embeddings = model_output[0][:, 0] # CLS token pooling
    instruction_embeddings = torch.nn.functional.normalize(instruction_embeddings, p=2, dim=1)
    return instruction_embeddings

# %%
from torch.utils.data import Dataset, DataLoader

class InstructionDataset(Dataset):
    def __init__(self, data):
        self.instructions = data['instruction']
        self.domains = data['domain']
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        return {
            'instruction': self.instructions[idx],
            'domain': self.domains[idx]
        }

from transformers import DataCollatorWithPadding

class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def __call__(self, features):
        instructions = [f["instruction"] for f in features]
        domains = [f["domain"] for f in features]
        batch = self.tokenizer(instructions, padding=True, truncation=True, return_tensors="pt")
        batch["domain"] = torch.tensor(domains)
        return batch

instruction_dataset = InstructionDataset(concated_data)

# %%
import torch.nn.functional as F

def contrastive_loss(anchor, positive, margin=1.0):
    return F.relu(margin - F.cosine_similarity(anchor, positive)).mean()

def supervised_contrastive_loss(embed, target, margin=1.0):
    loss = 0.0
    for i in range(len(embed)):
        positive = torch.stack([embed[j] for j in range(len(embed)) if target[j] == target[i]])
        negative = torch.stack([embed[j] for j in range(len(embed)) if target[j] != target[i]])
        if positive.size(0) > 1:  # Ensure at least one positive example (excluding self)
            anchor = embed[i].unsqueeze(0).repeat(positive.size(0), 1)
            loss += contrastive_loss(anchor, positive, margin)
            anchor = embed[i].unsqueeze(0).repeat(negative.size(0), 1)
            loss += contrastive_loss(anchor, negative, margin)
    return loss / len(embed)

# %%
from transformers import Trainer, TrainingArguments
import os

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        instructions = inputs['input_ids']
        domains = inputs['domain']
        embeddings = model(input_ids=instructions, attention_mask=inputs['attention_mask'])[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return supervised_contrastive_loss(embeddings, domains)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=1e-4,
    report_to="none",
    bf16=True,
    eval_strategy="no",
    remove_unused_columns=False,
    save_steps=1000,
    save_strategy="steps"
)

collator = CustomDataCollator(tokenizer)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=instruction_dataset,
    data_collator=collator
)

trainer.train()

# %%




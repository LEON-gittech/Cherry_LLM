import argparse
from unsloth import FastLanguageModel
import torch
# idxs = [i for i in range(4,121,4)] #base
idxs = [i for i in range(160,1601,160)] #5000

parser = argparse.ArgumentParser()
parser.add_argument("--domain",type=str,default=None)
args = parser.parse_args()

for idx in idxs:
    model, tokenizer = FastLanguageModel.from_pretrained(f"/mnt/bn/data-tns-live-llm/leon/datasets/privacy/{args.domain}_5000_10/checkpoint-{idx}", dtype = torch.bfloat16, load_in_4bit=True)
    model.save_pretrained_merged(f"/mnt/bn/data-tns-live-llm/leon/datasets/privacy/{args.domain}_5000_10/checkpoint-{idx}_merged", tokenizer, save_method = "merged_16bit",)
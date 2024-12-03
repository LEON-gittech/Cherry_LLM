import argparse
from unsloth import FastLanguageModel
import torch
# idxs = [i for i in range(4,121,4)] #base
# idxs = [i for i in range(160,1601,160)] #5000

parser = argparse.ArgumentParser()
parser.add_argument("--domain",type=str,default=None)
args = parser.parse_args()

root = "/mnt/bn/data-tns-live-llm/leon/datasets/fed/"
common = "20000_fedavg_c10s10_i10_b16a2_l2048_r16a16_f0"
prefixs = ["base", "iid2niid"]
domains = ["code", "med", "fin", "math"]
settings = ["1000_","2000_","3000_","4000_",""]

# for domain in domains:
#     path = f"{root}/base_{domain}_{common}"
#     print(path)
#     model, tokenizer = FastLanguageModel.from_pretrained(path, dtype = torch.bfloat16, load_in_4bit=True)
#     model.save_pretrained_merged(f"{path}_merged", tokenizer, save_method = "merged_16bit",)

for domain in domains:
    for setting in settings:
        path = f"{root}/iid2niid_{domain}_{setting}{common}"
        print(path)
        model, tokenizer = FastLanguageModel.from_pretrained(path, dtype = torch.bfloat16, load_in_4bit=True)
        model.save_pretrained_merged(f"{path}_merged", tokenizer, save_method = "merged_16bit",)
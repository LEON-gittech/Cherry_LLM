prefixes = ["base", "iid2niid"]
domains = ["code", "med", "fin", "math"]

import subprocess
import os

for prefix in prefixes:
    for domain in domains:
        print(f"Prefix {prefix} Domain {domain}")
        os.system(f"CUDA_VISIBLE_DEVICES=0 python3 privacy_eval_0.py --prefix {prefix} --domain {domain}")
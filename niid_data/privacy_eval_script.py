# idxs = [i for i in range(4,121,4)] #base
idxs = [i for i in range(160,1601,160)] #5000

import os
settings = ["med","fin","code"]

for setting in settings:
    print(f"This is domain {setting}")
    for i, idx in enumerate(idxs[:10]):
        print(f"Epoch {i}")
        os.system(f"CUDA_VISIBLE_DEVICES=0 python3 privacy_eval.py --setting {setting}_5000 --idx {idx}")
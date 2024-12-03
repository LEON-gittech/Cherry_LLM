import argparse
import random
import time
from contextlib import contextmanager
import torch
from datasets import load_dataset, load_from_disk
from datasets import load_dataset, concatenate_datasets, load_from_disk
from vllm import LLM, SamplingParams
import evaluate
import gc
rouge = evaluate.load('rouge')

def main(prefix, domain):
    root = "/mnt/bn/data-tns-live-llm/leon/datasets/fed/"
    common = "20000_fedavg_c10s10_i10_b16a2_l2048_r16a16_f0_client0"
    
    base_data = load_from_disk(f"/mnt/bn/data-tns-live-llm/leon/datasets/fed_data/base_{domain}_0.parquet/")
    for idx in range(10):
        path = f"{root}/{prefix}_{domain}_{common}/{idx}_merged"
        print(path)
        llm = LLM(model=path, tensor_parallel_size=1, dtype=torch.bfloat16, trust_remote_code=True, enable_lora=False, max_model_len=2048, gpu_memory_utilization=0.8)

        stop_tokens = ["USER:", "ASSISTANT:",  "### Instruction:", "Response:", 
                        "\n\nProblem", "\nProblem", "Problem:", "<|eot_id|>", "####"]

        sampling_params = SamplingParams(top_p=1, max_tokens=1024, repetition_penalty=1.1, top_k=40)  # memory extraction
        prefix_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ###Instruction:"""

        inputs = [prefix_prompt]*100
        outputs = llm.generate(inputs, sampling_params)
        response = [output.outputs[0].text for output in outputs]
        rougeLs = rouge.compute(predictions=response, references=[base_data["instruction"]]*100)["rougeL"]
        print(rougeLs)

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        # Reset CUDA device to fully clear memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()  # Wait for all streams on the current device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM model evaluation.")
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix for the dataset paths"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain for the dataset paths"
    )
    args = parser.parse_args()
    main(args.prefix, args.domain)
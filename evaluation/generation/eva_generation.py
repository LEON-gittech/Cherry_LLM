from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm
from unsloth import FastLanguageModel 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

PROMPT_DICT_ALPACA = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_WIZARDLM = {
    "prompt_input": (
        "{instruction}\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_VICUNA = {
    "prompt_input": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction}\nInput:\n{input} ASSISTANT:"
    ),
    "prompt_no_input": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='alpaca',
        help="alpaca, wiz, vicuna.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--unsloth", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import TextStreamer, AutoModel, AutoTokenizer, AutoModelForCausalLM

    if args.unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(args.model_name_or_path, dtype = torch.bfloat16, load_in_4bit=True)
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
        print(model)
        text_streamer = TextStreamer(tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # model.to(device)
    model.eval()

    if args.prompt == 'alpaca':
        prompt_input, prompt_no_input = PROMPT_DICT_ALPACA["prompt_input"], PROMPT_DICT_ALPACA["prompt_no_input"]
    elif args.prompt == 'wiz':
        prompt_input, prompt_no_input = PROMPT_DICT_WIZARDLM["prompt_input"], PROMPT_DICT_WIZARDLM["prompt_no_input"]
    elif args.prompt == 'vicuna':
        prompt_input, prompt_no_input = PROMPT_DICT_VICUNA["prompt_input"], PROMPT_DICT_VICUNA["prompt_no_input"]

    
    if(args.dataset_name=="vicuna"):
        dataset_path = 'evaluation/test_data/vicuna_test_set.jsonl'
        prompt_key = 'text'
    elif(args.dataset_name=="koala"):
        dataset_path = 'evaluation/test_data/koala_test_set.jsonl'
        prompt_key = 'prompt'
    elif(args.dataset_name=="sinstruct"):
        dataset_path = 'evaluation/test_data/sinstruct_test_set.jsonl'
        prompt_key = 'instruction'
    elif(args.dataset_name=="wizardlm"):
        dataset_path = 'evaluation/test_data/wizardlm_test_set.jsonl'
        prompt_key = 'Instruction'
    elif(args.dataset_name=="lima"):
        dataset_path = 'evaluation/test_data/lima_test_set.jsonl'
        prompt_key = 'conversations'
    elif(args.dataset_name=="math"):
        dataset_path = "/mnt/bn/data-tns-live-llm/leon/datasets/gsm8k/main/test.jsonl"
        prompt_key = "question"
    elif(args.dataset_name=="human_eval"):
        dataset_path = "/mnt/bn/data-tns-live-llm/leon/datasets/openai_humaneval/openai_humaneval/test.jsonl"
        prompt_key = "prompt"

    with open(dataset_path) as f:
        results = []
        dataset = list(f)[:50]
        for point in tqdm(dataset):
            point = json.loads(point)
            # print(point)
            instruction = point[prompt_key]
            if(args.dataset_name=="sinstruct"):
                instances = point['instances']
                assert len(instances) == 1
                if  instances[0]['input']:
                    prompt = prompt_input.format_map({"instruction":instruction, 'input':instances[0]['input']})
                else:
                    prompt = prompt_no_input.format_map({"instruction":instruction})
            else:
                prompt = prompt_no_input.format_map({"instruction":instruction})
            # prompt = f"""
            #     <|user|>
            #     {prompt}
            #     <|assistant|>
            # """
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            if args.unsloth: 
                generate_ids = model.generate(input_ids, max_length=args.max_length, repetition_penalty=1.1, streamer =    text_streamer, do_sample=True)
            else: 
                generate_ids = model.generate(input_ids, max_length=args.max_length, repetition_penalty=1.1, do_sample=True)

            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            point['raw_output'] = outputs
            if args.prompt in ['alpaca','wiz']:
                point['response'] = outputs.split("Response:")[1]
            elif args.prompt in ['vicuna']:
                point['response'] = outputs.split("ASSISTANT:")[1]
            print(point['raw_output'])
            results.append(point)

    output_dir =  os.path.join(args.model_name_or_path, 'test_inference')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_name = args.dataset_name + "_" + str(args.max_length) + ".json"
    with open(os.path.join(output_dir, saved_name), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
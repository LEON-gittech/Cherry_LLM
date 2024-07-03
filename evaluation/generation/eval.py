import argparse
import json
import os
import time
import numpy as np

import openai
from tqdm import tqdm
import asyncio
from typing import Any
import logging
from typing import List, Dict, Any
import requests
import asyncio
import aiohttp

import tiktoken
gpt_encoder = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def dispatch_openai_requests(
    messages_list: List[List[Dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    headers = {'Content-Type': 'application/json', 'Caller': 'leon.kepler'}
    url = f"https://swzkkd0h.us-east-fn.bytedance.net/gpt/openapi/online/v2/crawl"

    async def fetch_message(session, message, url, headers, temperature, top_p, max_tokens):
        while True:
            try:
                data = {
                    "model": model,  # 假设这里是你的模型名称
                    "messages": message,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                    "max_tokens": max_tokens,
                }
                data = {k: v for k, v in data.items() if v is not None}
                data = json.dumps(data)
                
                async with session.post(url, data=data, headers=headers) as response:
                    response_data = await response.json()
                    if response_data["choices"][0]["message"]["content"] is not None:
                        # print(response_data)
                        return response_data
                    else:
                        print("Not a good reply")
                        print(response_data)
            except Exception as e:
                print(f"An error occurred: {e}")
                # 可以选择重试或退出，这里我们选择退出
                break

    async def worker(session, messages_list, url, headers, temperature, top_p, max_tokens):
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_message(session, x, url, headers, temperature, top_p, max_tokens) for x in messages_list]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [resp for resp in responses if isinstance(resp, dict)]  # 过滤掉异常
    
    async with aiohttp.ClientSession() as session:
        while True:
            responses = await worker(session, messages_list, url, headers, temperature, top_p, max_tokens)
            if len(responses) == len(messages_list): return responses
    return responses

    async def fetch_message(message):
        while True:
            data = {
                "model": model,
                "messages": message,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
                "max_tokens": max_tokens,
            }
            data = {k: v for k, v in data.items() if v is not None}
            data = json.dumps(data)
            response = requests.post(url, data=data, headers=headers).json()
            if "choices" not in response.keys(): 
                print("not a good reply")
                print(response)
            else: return response

    async_responses = [fetch_message(x) for x in messages_list]
    return await asyncio.gather(*async_responses)
    # async_responses = [
    #     openai.ChatCompletion.acreate(
    #         model=model,
    #         messages=x,
    #         temperature=temperature,
    #         max_tokens=max_tokens,
    #         top_p=top_p,
    #     )
    #     for x in messages_list
    # ]
    return await asyncio.gather(*async_responses)

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def gen_prompt(ques, ans1, ans2):

    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
    )
    return sys_prompt, prompt


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wraped_file",default='')
    parser.add_argument("--api_key",type=str,default='')
    parser.add_argument("--api_model",type=str,default='gpt-3.5-turbo')
    parser.add_argument("--api_base",type=str,default='')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size to call OpenAI GPT",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()
    if args.api_base != '':
        openai.api_base = args.api_base
    openai.api_key = args.api_key
    print('Begin:',args.wraped_file)

    wraped_info = json.load(open(args.wraped_file))
    meta_info = wraped_info['Meta_Info']
    dataset_name = meta_info['dataset_name']
    qa_jsons = wraped_info['data']

    if(dataset_name=="vicuna"):
        prompt_key = 'text'
    elif(dataset_name=="koala"):
        prompt_key = 'prompt'
    elif(dataset_name=="sinstruct"):
        prompt_key = 'instruction'
    elif(dataset_name=="wizardlm"):
        prompt_key = 'Instruction'
    elif(dataset_name=="lima"):
        prompt_key = 'conversations'
    elif(dataset_name=="math"):
        prompt_key = "question"
    elif(dataset_name=="human_eval"):
        prompt_key = "prompt"

    total_len = len(qa_jsons)
    question_idx_list = list(range(total_len))

    predictions_all = []
    for reverse in range(2): # reverse or not
        message_list = []
        token_len_list = []

        for i in question_idx_list:

            instruction = qa_jsons[i][prompt_key]
            ques = instruction

            if reverse : # reverse = 1, secondly
                ans1 = qa_jsons[i]['Answer2']
                ans2 = qa_jsons[i]['Answer1']
            else: # reverse = 0, firstly
                ans1 = qa_jsons[i]['Answer1']
                ans2 = qa_jsons[i]['Answer2']
            sys_prompt, prompt = gen_prompt(ques, ans1, ans2)

            message =[
                        {"role": "system", "content": sys_prompt},
                        {
                            "role": "user",
                            "content": prompt,
                        },
            ]
            message_list.append(message)
            token_len_list.append(len(gpt_encoder.encode(prompt)))

        predictions = []
        i = 0
        wait_base = 10
        retry = 0
        error = 0
        pbar = tqdm(total=len(message_list))
        batch_size = args.batch_size
        # print(batch_size)
        while(i<len(message_list)):
            token_limit_in_current_batch = min(args.max_tokens,4070-max(token_len_list[i:i+batch_size]))
            try:
                batch_predictions = asyncio.run(
                    dispatch_openai_requests(
                        messages_list=message_list[i:i+batch_size],
                        model=args.api_model,
                        temperature=0.0,
                        max_tokens=token_limit_in_current_batch,
                        top_p=1.0,
                    )
                )
                predictions += batch_predictions
                retry = 0
                i += batch_size
                wait_base = 10
                pbar.update(batch_size)
            except Exception as e:
                print(e)
                retry += 1
                error += 1
                print("Batch error: ",i, i+batch_size)
                print("retry number: ", retry)
                print("error number: ", error)
                time.sleep(wait_base)
                wait_base = wait_base*2
        pbar.close()
        predictions_all.append(predictions)

    with open("/mnt/bn/data-tns-live-llm/leon/datasets/gpt_response.json", "w") as f:
        f.write(json.dumps(predictions_all, ensure_ascii=False, indent=4))
    # with open("/mnt/bn/data-tns-live-llm/leon/datasets/gpt_response.json", "r") as f:
    #     predictions_all = json.loads(f.read())
    all_scores = []
    for reverse in range(2):
        scores_list = []
        predictions = predictions_all[reverse]
        print(len(predictions))
        for idx, prediction in enumerate(predictions):
            if "choices" not in prediction.keys(): print(reverse, idx)
            review = prediction["choices"][0]['message']['content']
            scores = parse_score(review)
            review_key = 'review' if not reverse else 'review_reverse'
            scores_key = 'scores' if not reverse else 'scores_reverse'
            qa_jsons[idx][review_key] = review
            qa_jsons[idx][scores_key] = str(scores)
            scores_list.append(scores)

        all_scores.append(scores_list)
        avg_scores = np.array(scores_list).mean(0)
        avg_key = 'average_scores' if not reverse else 'average_scores_reverse'
        meta_info[avg_key] = str(avg_scores.tolist())

    wraped_info['Meta_Info'] = meta_info
    wraped_info['data'] = qa_jsons
    
    if 'gpt-4' in args.api_model:
        output_review_file = args.wraped_file.strip('.json') + '_reviews_gpt4.json'
    elif 'gpt-3.5' in args.api_model:
        output_review_file = args.wraped_file.strip('.json') + '_reviews_gpt3.5.json'
    with open(f"{output_review_file}", "w") as f:
        json.dump(wraped_info, f, indent=4)
        pass

    print('Finish:',args.wraped_file)
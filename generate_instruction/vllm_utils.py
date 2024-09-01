import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

import aiohttp
from tqdm.asyncio import tqdm
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import asyncio
import numpy as np


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: str
    best_of: int = 1
    use_beam_search: bool = False

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    error: str = ""

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
    
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    format: str,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": json.dumps(
                [
                    {"role": "system", "content": f"You are a helpful assistant designed to output JSON"},
                    {"role": "user", "content": f"{request_func_input.prompt}"}
                ]),
            "temperature": 0.7,
            "best_of": request_func_input.best_of,
            "max_tokens": 512,
            "stream": True,
            "guided_json": format,
            "frequency_penalty": 1.0,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

async def vllm_run(
    api_url: str,
    model_id: str,
    input_requests: List[Tuple[str, int, int]],
    format: str,
    best_of: int = 1,
    use_beam_search: bool = False,
    request_rate: float = float("inf"),
    disable_tqdm: bool = False,
    profile: bool = False,
    ):
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                async_request_openai_completions(request_func_input=request_func_input,format=format,
                                pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    return outputs
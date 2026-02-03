import base64
import mimetypes
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

DEFAUL_THINK_SEPARATOR = "\n\n</think>\n\n"
DEFAULT_TIMEOUT = 1200

_CLIENT_CACHE: Dict[Tuple[str, str, int], OpenAI] = {}


def _get_client(base_url: str, api_key: Optional[str], timeout: int) -> OpenAI:
    if not base_url:
        raise ValueError("base_url is required (pass --base_url or set OPENAI_BASE_URL).")
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    cache_key = (base_url, api_key, timeout)
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        _CLIENT_CACHE[cache_key] = client
    return client


def convert_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        file_content = image_file.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    base64_encoded_data = base64.b64encode(file_content).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def _normalize_image_urls(conversation: List[Dict[str, Any]]) -> None:
    for conv_turn in conversation:
        if conv_turn.get("role") != "user":
            continue
        if isinstance(conv_turn.get("content"), str):
            continue
        for content_part in conv_turn.get("content", []):
            if content_part.get("type") == "image_url":
                single_img_path = content_part.get("image_url", {}).get("url")
                if isinstance(single_img_path, str) and single_img_path.startswith("data:"):
                    continue
                if isinstance(single_img_path, str) and os.path.exists(single_img_path):
                    base64_image_url = convert_image_to_base64(single_img_path)
                    content_part["image_url"]["url"] = base64_image_url


def _collect_stream_text(stream, concat_reasoning_content: bool) -> str:
    response_text = ""
    reasoning_text = ""
    for event in stream:
        if not hasattr(event, "choices") or not event.choices:
            continue
        delta = event.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            response_text += delta.content
        if concat_reasoning_content and hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text += delta.reasoning_content
    if concat_reasoning_content:
        return f"{reasoning_text}{DEFAUL_THINK_SEPARATOR}{response_text}"
    return response_text


def request_internal_conv_api(
    conversation: List[Dict[str, Any]] = [],
    return_raw_requst_res: bool = False,
    concat_resoning_content: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    **request_kwargs,
):
    _normalize_image_urls(conversation)

    internal_api_model = request_kwargs.pop("internal_api_model", None)
    max_tokens = request_kwargs.pop("max_tokens", None)
    temperature = request_kwargs.pop("temperature", 0.8)
    stream = request_kwargs.pop("stream", False)
    retries = request_kwargs.pop("retries", 5)
    random_seed = request_kwargs.pop("random_seed", None) or request_kwargs.pop("seed", None)

    client = _get_client(base_url or os.environ.get("OPENAI_BASE_URL", ""), api_key, timeout)

    for n_try in range(retries):
        try:
            res = client.chat.completions.create(
                model=internal_api_model,
                messages=conversation,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=random_seed,
                **request_kwargs,
            )
            if stream:
                response_text = _collect_stream_text(res, concat_resoning_content)
            else:
                res_message = res.choices[0].message
                response_text = res_message.content or ""
                if concat_resoning_content:
                    reasoning_text = getattr(res_message, "reasoning_content", "")
                    response_text = f"{reasoning_text}{DEFAUL_THINK_SEPARATOR}{response_text}"
            if response_text:
                if return_raw_requst_res:
                    return response_text, res
                return response_text
        except Exception as e:
            print(f"Error during API request: {e}, Retrying {n_try + 1}/{retries}...")
            time.sleep(random.uniform(1, 3))

    print(f"Failed to get a valid response after {retries} retries")
    return "" if not return_raw_requst_res else ("", None)


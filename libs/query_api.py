import os
import json
import sys
import time
from typing import Any, Dict, Generator, Optional

import requests

# ----------------------------------------------------------------------
# 1️⃣  Configuration
# ----------------------------------------------------------------------
# 👉 Put your secret key in the environment; never hard‑code it in source.
#    e.g. on Linux/macOS: export OPENAI_API_KEY="sk-..."
#    on Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.stderr.write("❌  Please set the OPENAI_API_KEY environment variable.\n")
    sys.exit(1)

# The model you want to use – feel free to change it.
CHAT_MODEL = "gpt-4o-mini"          # fast & cheap ChatGPT model
TEXT_MODEL = "gpt-4o-mini"          # same model works for completions too

# Base URL – you can change this for Azure OpenAI or other hosted endpoints.
BASE_URL = "https://api.ppq.ai/v1" #"https://api.openai.com/v1"
# ----------------------------------------------------------------------
def openai_request(
    method: str,
    endpoint: str,
    json_body: Optional[Dict[str, Any]] = None,
    stream: bool = False,
) -> requests.Response:
    """
    Sends an HTTP request to the OpenAI API with proper auth/header handling.
    """
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        # You can add an organization header if you belong to multiple orgs:
        # "OpenAI-Organization": "org-xxxxxxxxxxxxxxxxxxxx"
    }

    # `requests` will raise for connection errors, not for API‑level errors.
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        json=json_body,
        stream=stream,
        timeout=90,  # generous timeout for long generations
    )
    return response

def chat_completion(
    messages: list[Dict[str, str]],
    model: str = CHAT_MODEL,
    temperature: float = 0.0,
    #max_tokens: int = 1024,
    stream: bool = False,
):
    """
    Calls /v1/chat/completions.
    Returns either the full JSON response (stream=False) or a generator
    yielding decoded JSON chunks (stream=True).
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        #"max_tokens": max_tokens,
        "stream": stream,
    }

    resp = openai_request("POST", "/chat/completions", json_body=payload, stream=stream)

    if resp.status_code != 200:
        # The API returns a JSON error payload – show it nicely.
        try:
            err = resp.json()
        except Exception:
            err = {"error": {"message": resp.text}}
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {err}")

    return resp.json()["choices"][0]["message"]["content"]

    if not stream:
        cur_resp = resp.json()  # whole response at once
        # print(f"Inner resp: {cur_resp}")
        return cur_resp
    else:
        raise RuntimeError("Not yer implemented!")
        # Streamed mode: each line begins with "data: " and ends with "\n\n".
        # The final message is "data: [DONE]".  We'll yield dicts.
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.strip() == "data: [DONE]":
                break
            if line.startswith("data: "):
                json_str = line[len("data: ") :]
                #try:
                #    yield json.loads(json_str)
                #except json.JSONDecodeError:
                #    # ignore malformed lines – they shouldn't happen
                #    continue


# ----------------------------------------------------------------------
# 4️⃣  (Optional) Classic Text Completion
# ----------------------------------------------------------------------
def text_completion(
    prompt: str,
    model: str = TEXT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> Any:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = openai_request("POST", "/completions", json_body=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Error {resp.status_code}: {resp.text}")
    return resp.json()

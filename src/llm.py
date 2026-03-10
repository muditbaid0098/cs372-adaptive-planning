# LLM wrapper (OpenAI API calls with retries)

import os
import json
import time
import threading
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0)

DEFAULT_MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o-mini"

# Thread-local storage for per-task stats
_local = threading.local()


def get_stats():
    return {
        "calls": getattr(_local, "call_count", 0),
        "tokens": getattr(_local, "total_tokens", 0),
    }


def reset_stats():
    _local.call_count = 0
    _local.total_tokens = 0


def call_llm(prompt: str, system: str = None, model: str = None, temperature: float = 0.3,
             max_tokens: int = 2000, json_mode: bool = False, retries: int = 5) -> str:
    if model is None:
        model = DEFAULT_MODEL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**kwargs)
            _local.call_count = getattr(_local, "call_count", 0) + 1
            if response.usage:
                _local.total_tokens = getattr(_local, "total_tokens", 0) + response.usage.total_tokens
            return response.choices[0].message.content
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            wait = min(2 ** attempt * 2, 30)  # 2, 4, 8, 16, 30 sec
            print(f"  [retry {attempt+1}/{retries}] {type(e).__name__}: waiting {wait}s...")
            time.sleep(wait)
            if attempt == retries - 1:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise e


def call_llm_multi(messages: list, model: str = None, temperature: float = 0.3,
                   max_tokens: int = 2000, json_mode: bool = False, retries: int = 5) -> str:
    """Same as call_llm but takes a full message list (for ReAct etc)."""
    if model is None:
        model = DEFAULT_MODEL

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**kwargs)
            _local.call_count = getattr(_local, "call_count", 0) + 1
            if response.usage:
                _local.total_tokens = getattr(_local, "total_tokens", 0) + response.usage.total_tokens
            return response.choices[0].message.content
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            wait = min(2 ** attempt * 2, 30)
            print(f"  [retry {attempt+1}/{retries}] {type(e).__name__}: waiting {wait}s...")
            time.sleep(wait)
            if attempt == retries - 1:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise e


def parse_json(text: str) -> dict:
    """Try to get JSON out of whatever the LLM returns."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return json.loads(text[start:end].strip())
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return json.loads(text[start:end].strip())
    # Try finding first { to last }
    start = text.index("{")
    end = text.rindex("}") + 1
    return json.loads(text[start:end])

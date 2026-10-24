import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    # Best-effort: find first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


@dataclass
class LLMConfig:
    api_key: Optional[str]
    base_url: str
    model: str


class LLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
        self.cfg = LLMConfig(api_key=api_key, base_url=base_url.rstrip("/"), model=model)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.api_key)

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 800) -> str:
        if not self.enabled:
            raise RuntimeError("LLM_API_KEY not set; LLM client is disabled.")

        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible response shape
        return data["choices"][0]["message"]["content"]

    def chat_json(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 800) -> Dict[str, Any]:
        text = self.chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens)
        return _extract_json_object(text)


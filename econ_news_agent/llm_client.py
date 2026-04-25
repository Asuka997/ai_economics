from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-2-0-mini-260215",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
}


class OpenAICompatibleClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("ARK_API_KEY", "").strip()
        self.base_url = os.getenv("ARK_BASE_URL", PROVIDER_PRESETS["doubao"]["base_url"]).rstrip("/")
        self.model = os.getenv("ARK_MODEL", PROVIDER_PRESETS["doubao"]["model"])
        self.timeout = float(os.getenv("ARK_TIMEOUT", "90").strip() or "90")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def reconfigure(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()

    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2) -> str:
        if not self.available:
            raise RuntimeError("API Key 未配置，请在配置页面填写。")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        choices = parsed.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
        raise RuntimeError(f"Unexpected response format: {parsed}")

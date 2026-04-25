from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "doubao-seed-2-0-mini-260215"


class OpenAICompatibleClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("ARK_API_KEY", "").strip()
        self.base_url = os.getenv("ARK_BASE_URL", ARK_BASE_URL).rstrip("/")
        self.model = os.getenv("ARK_MODEL", ARK_MODEL)
        self.timeout = float(os.getenv("ARK_TIMEOUT", "90").strip() or "90")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2) -> str:
        if not self.available:
            raise RuntimeError("ARK_API_KEY 未配置，请在 .env 文件中设置。")

        # Convert messages to Ark /responses input format
        ark_input = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                ark_input.append({
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                })
            else:
                ark_input.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self.model,
            "input": ark_input,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/responses",
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

        # Ark /responses returns output list; extract first text item
        output = parsed.get("output", [])
        for item in output:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        return part["text"]
        raise RuntimeError(f"Unexpected Ark response format: {parsed}")

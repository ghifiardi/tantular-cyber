"""Cancellable structured Ollama adapter, compatible with the scanner client policy."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from tantular_cyber.ollama_client import is_remote_model, model_matches, validate_base_url


class LocalModel:
    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        timeout: float,
        num_predict: int,
        max_bytes: int,
        num_ctx: int = 0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not model or "tantular" not in model.lower() or model.lower().endswith(":cloud"):
            raise ValueError("an installed local Tantular tag is required")
        if num_ctx != 0 and not 512 <= num_ctx <= 131072:
            raise ValueError("num_ctx must be 0 (auto) or 512..131072")
        self.requested_num_ctx = num_ctx
        self.num_ctx = num_ctx or 8192
        self.context_source = "configured" if num_ctx else "default"
        self.loaded_context: int | None = None
        self.model = model
        self.num_predict = num_predict
        self.max_bytes = max_bytes
        self.http = httpx.AsyncClient(
            base_url=validate_base_url(base_url, allow_remote=False),
            timeout=timeout,
            trust_env=False,
            transport=transport,
            follow_redirects=False,
        )
        self.identity: dict[str, Any] = {}
        self.verified = False

    async def close(self) -> None:
        await self.http.aclose()

    async def discover(self) -> None:
        response = await self.http.get("/api/tags")
        response.raise_for_status()
        entries = response.json().get("models", [])
        tag = next((m for m in entries if model_matches(self.model, m.get("name", ""))), None)
        if tag is None or is_remote_model(tag):
            raise ValueError("requested Tantular tag is missing or remotely served")
        # Do not /api/show cloud entries: Ollama forwards that call upstream.
        response = await self.http.post("/api/show", json={"model": self.model})
        response.raise_for_status()
        info = response.json()
        if is_remote_model(info) or info.get("details", {}).get("format") != "gguf":
            raise ValueError("local GGUF model required")
        self.identity = {k: tag.get(k) for k in ("name", "digest", "size")}
        self.identity["details"] = info.get("details", {})
        # Different context settings can wait for a runner reload behind ongoing work.
        # Reuse a loaded instance of these exact weights; never stop another workload.
        response = await self.http.get("/api/ps")
        if response.status_code != 404:
            response.raise_for_status()
            loaded = response.json().get("models", [])
            digest = self.identity.get("digest")
            for entry in loaded:
                same_name = model_matches(self.model, entry.get("name", entry.get("model", "")))
                same_digest = bool(digest and digest == entry.get("digest"))
                context = entry.get("context_length")
                if (same_name or same_digest) and type(context) is int and context >= 512:
                    self.loaded_context = context
                    break
        if not self.requested_num_ctx:
            if self.loaded_context is not None:
                if self.loaded_context > 131072:
                    raise ValueError("loaded context exceeds supported limit")
                self.num_ctx = self.loaded_context
                self.context_source = "loaded model"
            else:
                match = re.search(r"(?m)^num_ctx\s+(\d+)\s*$", info.get("parameters", ""))
                context = int(match[1]) if match else 8192
                if not 512 <= context <= 131072:
                    raise ValueError("model default context exceeds supported limit")
                self.num_ctx = context
                self.context_source = "model default" if match else "default"
        self.verified = True

    async def generate(self, system: str, data: dict, schema: dict) -> str:
        if not self.verified:
            await self.discover()
        payload = {
            "model": self.model,
            "think": False,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "seed": 42,
            },
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": "<untrusted_data>"
                    + json.dumps(data, ensure_ascii=True)
                    .replace("<", "\\u003c")
                    .replace(">", "\\u003e")
                    + "</untrusted_data>",
                },
            ],
        }
        # Streaming the transport bounds response allocation; generation remains stream=false.
        async with self.http.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > self.max_bytes:
                    raise ValueError("model response exceeds byte limit")
        value = json.loads(body)
        if value.get("done_reason") not in (None, "stop"):
            raise ValueError("model generation ended early")
        content = value["message"]["content"]
        if not isinstance(content, str) or len(content.encode()) > self.max_bytes:
            raise ValueError("invalid model response")
        return content

"""Minimal Ollama ``/api/chat`` client: localhost-only by default, bounded retries, no streaming.

Prompts and source code are never logged; only status codes, durations and error classes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import OllamaSettings
from .models import LLMMetrics

log = logging.getLogger(__name__)

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class OllamaError(Exception):
    """Non-retryable error from Ollama (bad request, model missing, malformed response)."""


class OllamaUnavailable(OllamaError):
    """Server unreachable or kept failing after the retry budget was spent."""


class RemoteEndpointNotAllowed(OllamaError):
    """Base URL is not localhost and remote endpoints were not explicitly allowed."""


def validate_base_url(url: str, *, allow_remote: bool) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise OllamaError(f"invalid Ollama base URL scheme/host: {parts.scheme!r}")
    if parts.username or parts.password:
        raise OllamaError("credentials in the Ollama base URL are not supported")
    if parts.hostname not in LOCAL_HOSTS and not allow_remote:
        raise RemoteEndpointNotAllowed(
            f"Ollama host '{parts.hostname}' is not localhost; source excerpts would leave this "
            "machine. Pass --allow-remote-ollama (or TANTULAR_ALLOW_REMOTE_OLLAMA=1) to opt in."
        )
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def model_matches(requested: str, available: str) -> bool:
    def norm(name: str) -> str:
        return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"

    return norm(requested) == norm(available)


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        settings: OllamaSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.base_url = validate_base_url(settings.base_url, allow_remote=settings.allow_remote)
        self.model = settings.model
        self._sleep = sleep
        self.last_metrics: LLMMetrics | None = None
        # trust_env=False: never route local inference through a system HTTP proxy.
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(settings.timeout_s, connect=10.0),
            transport=transport,
            trust_env=False,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        attempts = self.settings.max_retries + 1
        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            try:
                response = self._http.request(method, path, **kwargs)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = f"{type(exc).__name__}"
                log.warning(
                    "ollama %s %s failed (%s), attempt %d/%d",
                    method,
                    path,
                    last_error,
                    attempt,
                    attempts,
                )
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in _RETRYABLE_STATUS:
                    raise OllamaError(
                        f"Ollama {method} {path} returned HTTP {response.status_code}: "
                        f"{_error_text(response)}"
                    )
                last_error = f"HTTP {response.status_code}"
                log.warning(
                    "ollama %s %s returned %s, attempt %d/%d",
                    method,
                    path,
                    last_error,
                    attempt,
                    attempts,
                )
            if attempt < attempts:
                self._sleep(min(2.0 ** (attempt - 1), 8.0))
        raise OllamaUnavailable(
            f"Ollama {method} {path} failed after {attempts} attempt(s): {last_error}"
        )

    def server_version(self) -> str | None:
        data = self._request("GET", "/api/version").json()
        return str(data.get("version")) if isinstance(data, dict) else None

    def ensure_model(self) -> None:
        data = self._request("GET", "/api/tags").json()
        names = [m.get("name", "") for m in (data.get("models") or []) if isinstance(m, dict)]
        if not any(model_matches(self.model, n) for n in names):
            raise OllamaError(
                f"model '{self.model}' is not available in Ollama; run `ollama list` to check"
            )

    def health_check(self) -> str | None:
        """Verify the server answers and the model exists; returns the server version."""
        version = self.server_version()
        self.ensure_model()
        return version

    def chat(
        self, messages: list[dict[str, str]], format_schema: dict[str, Any] | None = None
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "think": False,
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_ctx": self.settings.num_ctx,
            },
        }
        if format_schema is not None:
            payload["format"] = format_schema
        started = time.monotonic()
        self.last_metrics = None
        response = self._request("POST", "/api/chat", json=payload)
        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("malformed /api/chat response") from exc
        if not isinstance(content, str):
            raise OllamaError("malformed /api/chat response: content is not text")
        self.last_metrics = chat_metrics(data, time.monotonic() - started)
        log.info(
            "ollama chat ok model=%s done_reason=%s duration=%.1fs",
            self.model,
            data.get("done_reason"),
            time.monotonic() - started,
        )
        if data.get("done_reason") not in (None, "stop"):
            raise OllamaError(f"generation ended early (done_reason={data.get('done_reason')})")
        return content


def chat_metrics(data: dict[str, Any], wall_s: float) -> LLMMetrics:
    """Convert Ollama's nanosecond counters into an ``LLMMetrics`` record."""

    def seconds(key: str) -> float | None:
        value = data.get(key)
        return round(value / 1e9, 3) if isinstance(value, int | float) else None

    def count(key: str) -> int | None:
        value = data.get(key)
        return int(value) if isinstance(value, int | float) else None

    output_tokens, eval_s = count("eval_count"), seconds("eval_duration")
    rate = round(output_tokens / eval_s, 2) if output_tokens and eval_s else None
    return LLMMetrics(
        wall_s=round(wall_s, 3),
        total_s=seconds("total_duration"),
        load_s=seconds("load_duration"),
        prompt_tokens=count("prompt_eval_count"),
        prompt_eval_s=seconds("prompt_eval_duration"),
        output_tokens=output_tokens,
        eval_s=eval_s,
        output_tokens_per_s=rate,
    )


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
        text = str(data.get("error", "")) if isinstance(data, dict) else ""
    except ValueError:
        text = ""
    return text[:200] or response.reason_phrase

from __future__ import annotations

import json

import httpx
import pytest

from tantular_cyber.config import OllamaSettings
from tantular_cyber.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
    RemoteEndpointNotAllowed,
    model_matches,
    validate_base_url,
)

TAGS_LOCAL = {"models": [{"name": "tantular-cyber:latest"}]}


def make_client(handler, **settings) -> OllamaClient:
    """Client whose model check (/api/tags) succeeds, so tests exercise /api/chat."""
    sleeps: list[float] = []

    def routed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS_LOCAL)
        return handler(request)

    client = OllamaClient(
        OllamaSettings(**settings), transport=httpx.MockTransport(routed), sleep=sleeps.append
    )
    client.sleeps = sleeps  # type: ignore[attr-defined]
    return client


def chat_ok(content: str = '{"ok": true}') -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
        },
    )


def test_defaults_match_local_tantular_model() -> None:
    s = OllamaSettings()
    assert s.base_url == "http://127.0.0.1:11434"
    assert s.model == "tantular-cyber:latest"
    assert not s.allow_remote


def test_chat_payload_disables_thinking_and_streaming() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert str(request.url).startswith("http://127.0.0.1:11434/")
        seen.append(json.loads(request.content))
        return chat_ok("hasil")

    client = make_client(handler)
    out = client.chat([{"role": "user", "content": "x"}], format_schema={"type": "object"})
    assert out == "hasil"
    payload = seen[0]
    assert payload["model"] == "tantular-cyber:latest"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["format"] == {"type": "object"}


def test_retries_transient_errors_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused")
        if calls["n"] == 2:
            return httpx.Response(503)
        return chat_ok()

    client = make_client(handler, max_retries=2)
    assert client.chat([{"role": "user", "content": "x"}]) == '{"ok": true}'
    assert calls["n"] == 3
    assert client.sleeps == [1.0, 2.0]  # type: ignore[attr-defined]


def test_retry_budget_is_bounded() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    client = make_client(handler, max_retries=1)
    with pytest.raises(OllamaUnavailable):
        client.chat([{"role": "user", "content": "x"}])
    assert calls["n"] == 2


def test_client_errors_are_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, json={"error": "model 'x' not found"})

    with pytest.raises(OllamaError, match="404") as exc:
        make_client(handler).chat([{"role": "user", "content": "x"}])
    assert not isinstance(exc.value, OllamaUnavailable)
    assert calls["n"] == 1


def test_malformed_and_truncated_responses_raise() -> None:
    with pytest.raises(OllamaError, match="malformed"):
        make_client(lambda r: httpx.Response(200, json={"nope": 1})).chat([])
    truncated = httpx.Response(200, json={"message": {"content": "{"}, "done_reason": "length"})
    with pytest.raises(OllamaError, match="ended early"):
        make_client(lambda r: truncated).chat([])


def test_health_check_requires_model_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.2"})
        return httpx.Response(200, json={"models": [{"name": "tantular-cyber:latest"}]})

    assert make_client(handler).health_check() == "0.40.2"
    assert make_client(handler, model="tantular-cyber").health_check() == "0.40.2"
    with pytest.raises(OllamaError, match="not available"):
        make_client(handler, model="other:7b").health_check()


def test_model_matches_normalizes_latest_tag() -> None:
    assert model_matches("tantular-cyber", "tantular-cyber:latest")
    assert not model_matches("tantular-cyber:q4", "tantular-cyber:latest")
    assert model_matches("hf.co/org/m:Q4", "hf.co/org/m:Q4")


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:11434", "http://localhost:11434/", "http://[::1]:11434"]
)
def test_localhost_urls_allowed(url: str) -> None:
    assert validate_base_url(url, allow_remote=False).rstrip("/") == url.rstrip("/")


def test_remote_url_requires_explicit_opt_in() -> None:
    with pytest.raises(RemoteEndpointNotAllowed):
        validate_base_url("http://gpu-box.example.com:11434", allow_remote=False)
    with pytest.raises(RemoteEndpointNotAllowed):
        OllamaClient(OllamaSettings(base_url="http://10.0.0.5:11434"))
    assert validate_base_url("https://gpu-box.example.com", allow_remote=True)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1", "http://u:p@127.0.0.1"])
def test_invalid_urls_rejected(url: str) -> None:
    with pytest.raises(OllamaError):
        validate_base_url(url, allow_remote=True)


def test_env_overrides_and_cli_precedence() -> None:
    env = {
        "TANTULAR_OLLAMA_URL": "http://localhost:1234",
        "TANTULAR_MODEL": "m:1",
        "TANTULAR_ALLOW_REMOTE_OLLAMA": "true",
        "TANTULAR_OLLAMA_TIMEOUT": "5",
    }
    s = OllamaSettings.from_env(env)
    assert (s.base_url, s.model, s.allow_remote, s.timeout_s) == (
        "http://localhost:1234",
        "m:1",
        True,
        5.0,
    )
    assert OllamaSettings.from_env(env, model="cli:2", base_url=None).model == "cli:2"


def test_chat_records_ollama_performance_metrics() -> None:
    body = {
        "message": {"content": "{}"},
        "done_reason": "stop",
        "total_duration": 52_000_000_000,
        "load_duration": 3_500_000_000,
        "prompt_eval_count": 900,
        "prompt_eval_duration": 4_000_000_000,
        "eval_count": 420,
        "eval_duration": 42_000_000_000,
    }
    client = make_client(lambda r: httpx.Response(200, json=body))
    client.chat([{"role": "user", "content": "x"}])
    m = client.last_metrics
    assert m is not None
    assert (m.load_s, m.prompt_tokens, m.output_tokens, m.eval_s) == (3.5, 900, 420, 42.0)
    assert m.output_tokens_per_s == 10.0
    assert m.total_s == 52.0 and m.wall_s >= 0


def test_metrics_tolerate_missing_counters() -> None:
    client = make_client(lambda r: chat_ok())
    client.chat([])
    m = client.last_metrics
    assert m is not None and m.output_tokens is None and m.output_tokens_per_s is None


def test_chat_refuses_cloud_model_before_any_chat_request() -> None:
    """Regression (PR #10 review #1): every chat path verifies the model first."""
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "x:cloud", "remote_host": "https://ollama.com"}]}
            )
        return chat_ok()

    client = OllamaClient(OllamaSettings(model="x:cloud"), transport=httpx.MockTransport(handler))
    with pytest.raises(OllamaError, match="remote host"):
        client.chat([{"role": "user", "content": "secret source"}])
    assert paths == ["/api/tags"]


def test_chat_verifies_model_once_per_client() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS_LOCAL)
        return chat_ok()

    client = OllamaClient(OllamaSettings(), transport=httpx.MockTransport(handler))
    client.chat([])
    client.chat([])
    assert paths == ["/api/tags", "/api/chat", "/api/chat"]

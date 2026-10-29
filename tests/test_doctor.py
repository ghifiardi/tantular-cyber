"""doctor is read-only: loopback Ollama only, no pulls, no writes, explicit capabilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from tantular_cyber.cli import EXIT_DEGRADED, EXIT_OK, main
from tantular_cyber.config import OllamaSettings
from tantular_cyber.doctor import (
    detect_platform,
    parse_semgrep_version,
    render_doctor,
    run_doctor,
    unsupported_platform_reason,
)
from tantular_cyber.ollama_client import OllamaClient, OllamaError

TAGS = {
    "models": [
        {
            "name": "tantular-cyber:latest",
            "size": 10 * 1024**3,
            "details": {"family": "qwen35", "parameter_size": "9.7B", "quantization_level": "Q8_0"},
        },
        {"name": "qwen3.5:cloud", "size": 346, "remote_host": "https://ollama.com:443"},
    ]
}


def semgrep_runner(version: str = "1.143.0"):
    def run(cmd, **kwargs):
        assert cmd[1:] == ["--version", "--disable-version-check"]
        assert kwargs["env"]["SEMGREP_SEND_METRICS"] == "off"
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{version}\n", stderr="")

    return run


def ollama_transport(seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.2"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        if request.url.path == "/api/show":
            assert json.loads(request.content)["model"] != "qwen3.5:cloud"
            return httpx.Response(200, json={"capabilities": ["completion"]})
        raise AssertionError(f"doctor must not call {request.url.path}")

    return httpx.MockTransport(handler)


def doctor(**kw):
    defaults = dict(
        which=lambda name: "/usr/local/bin/semgrep",
        runner=semgrep_runner(),
        platform_kind="macos",
        memory_gb=24.0,
    )
    return run_doctor(**{**defaults, **kw})


def test_full_capability_and_only_read_only_loopback_requests() -> None:
    seen: list[str] = []
    report = doctor(transport=ollama_transport(seen))
    assert report.scan_capability == "full" and report.ai_capability == "available"
    assert seen == ["GET /api/version", "GET /api/tags", "POST /api/show"]
    assert all("/api/pull" not in r and "/api/create" not in r for r in report.network_requests)
    assert [m.remote for m in report.installed_models] == [False, True]
    assert "FULL" in render_doctor(report)


def test_semgrep_missing_is_partial_with_install_guidance() -> None:
    report = doctor(which=lambda name: None, check_ollama=False)
    semgrep = next(c for c in report.checks if c.id == "semgrep")
    assert report.scan_capability == "partial" and semgrep.status == "fail"
    assert "not bundled" in (semgrep.action or "")
    assert report.network_requests == [] and report.ai_capability == "not_checked"


def test_unavailable_when_nothing_can_scan() -> None:
    report = doctor(which=lambda name: None, check_ollama=False, callgraph_enabled=False)
    assert report.scan_capability == "unavailable"


def test_untested_semgrep_version_warns_but_stays_full() -> None:
    report = doctor(runner=semgrep_runner("2.1.0"), check_ollama=False)
    semgrep = next(c for c in report.checks if c.id == "semgrep")
    assert semgrep.status == "warn" and report.scan_capability == "full"


def test_native_windows_is_partial_and_points_to_wsl2() -> None:
    report = doctor(platform_kind="windows", check_ollama=False)
    assert report.scan_capability == "partial"
    assert "WSL2" in (next(c for c in report.checks if c.id == "platform").action or "")


def test_remote_endpoint_is_not_contacted() -> None:
    seen: list[str] = []
    report = doctor(
        ollama=OllamaSettings(base_url="http://192.0.2.10:11434"), transport=ollama_transport(seen)
    )
    assert seen == [] and report.network_requests == []
    assert next(c for c in report.checks if c.id == "privacy").status == "warn"


def test_cloud_model_is_never_shown_and_marked_unavailable() -> None:
    seen: list[str] = []
    report = doctor(ollama=OllamaSettings(model="qwen3.5:cloud"), transport=ollama_transport(seen))
    assert "POST /api/show" not in seen
    assert report.ai_capability == "unavailable"
    assert next(c for c in report.checks if c.id == "model").status == "fail"


def test_missing_model_suggests_but_never_pulls() -> None:
    seen: list[str] = []
    report = doctor(ollama=OllamaSettings(model="absent:1b"), transport=ollama_transport(seen))
    model = next(c for c in report.checks if c.id == "model")
    assert report.ai_capability == "model_missing"
    assert "nothing was downloaded" in model.details[0] and "ollama pull" in model.details[0]
    assert seen == ["GET /api/version", "GET /api/tags"]


def test_ollama_down_does_not_affect_scan_capability() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    report = doctor(transport=httpx.MockTransport(down))
    assert report.scan_capability == "full" and report.ai_capability == "unavailable"


def test_large_model_vs_ram_is_guidance_only() -> None:
    report = doctor(transport=ollama_transport([]), memory_gb=8.0)
    model = next(c for c in report.checks if c.id == "model")
    assert model.status == "warn" and "guidance only" in model.details[-1]
    assert report.ai_capability == "available"


def test_platform_detection_and_semgrep_version_parsing() -> None:
    assert detect_platform("darwin") == "macos"
    assert detect_platform("linux", lambda: "5.15.0-microsoft-standard-WSL2") == "wsl2"
    assert detect_platform("linux", lambda: "6.8.0-generic") == "linux"
    assert detect_platform("win32") == "windows"
    assert unsupported_platform_reason("linux") is None
    assert parse_semgrep_version("1.143.0") == (1, 143, 0)
    assert parse_semgrep_version("semgrep: command failed") is None


def test_doctor_cli_json_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["doctor", "--json", "--no-ollama", "--semgrep-bin", str(tmp_path / "none")])
    data = json.loads(capsys.readouterr().out)
    assert code == EXIT_DEGRADED and data["scan_capability"] == "partial"
    assert data["result_type"] == "tantular-cyber.doctor" and data["network_requests"] == []
    assert list(tmp_path.iterdir()) == []


def test_scan_ai_refuses_cloud_models_unless_allowed() -> None:
    transport = ollama_transport([])
    cloud = OllamaSettings(model="qwen3.5:cloud")
    with (
        OllamaClient(cloud, transport=transport) as client,
        pytest.raises(OllamaError, match="remote host"),
    ):
        client.ensure_model()
    allowed = OllamaSettings(model="qwen3.5:cloud", allow_remote=True)
    with OllamaClient(allowed, transport=transport) as client:
        client.ensure_model()


def test_sample_command_copies_project_and_refuses_non_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "demo"
    assert main(["sample", "--dest", str(dest)]) == EXIT_OK
    assert (dest / "app.py").is_file() and "tantular-cyber scan" in capsys.readouterr().out
    assert main(["sample", "--dest", str(dest)]) != EXIT_OK

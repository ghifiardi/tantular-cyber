"""Read-only environment diagnostics for ``tantular-cyber doctor``.

Doctor never downloads, pulls models or writes files itself, and contacts nothing except a
**loopback** Ollama endpoint, only with read-only requests (``GET /api/version``,
``GET /api/tags``, ``POST /api/show``). Non-loopback or credential-bearing endpoint URLs are
rejected (``validate_base_url``) and never contacted; URLs are displayed as scheme://host:port
only. The only subprocess is ``semgrep --version`` with metrics and version checks disabled;
Semgrep may maintain its own state (e.g. ``~/.semgrep``), which is outside doctor's control.

Capabilities:
* ``scan``: ``full`` (Semgrep + tantular-callgraph on a supported platform), ``partial``
  (tantular-callgraph only, or an unverified platform), ``unavailable`` (nothing usable).
* ``ai``: optional explanations — ``available``, ``model_missing``, ``unavailable``,
  ``not_checked``. Deterministic scans never need Ollama.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from . import __version__
from .config import DEFAULT_MODEL, OllamaSettings, default_rules
from .ollama_client import (
    OllamaError,
    RemoteEndpointNotAllowed,
    is_remote_model,
    model_matches,
    validate_base_url,
)
from .scanner import SemgrepScanner

# Semgrep range exercised by CI (`pip install 'tantular-cyber[semgrep]'`) and local testing.
SEMGREP_TESTED = ((1, 90, 0), (2, 0, 0))  # [min, max)
DOCTOR_SCHEMA_VERSION = "1.0.0"

CheckStatus = Literal["ok", "warn", "fail", "info", "skip"]
PlatformKind = Literal["macos", "linux", "wsl2", "windows", "other"]


class Check(BaseModel):
    id: str
    status: CheckStatus
    summary: str
    details: list[str] = Field(default_factory=list)
    action: str | None = None


class InstalledModel(BaseModel):
    name: str
    remote: bool = Field(
        default=False, description="Cloud-backed: runs on a remote host, not this machine."
    )
    size_gb: float | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None


class DoctorReport(BaseModel):
    result_type: str = "tantular-cyber.doctor"
    schema_version: str = DOCTOR_SCHEMA_VERSION
    tool_version: str = __version__
    platform: dict[str, str]
    scan_capability: Literal["full", "partial", "unavailable"]
    ai_capability: Literal["available", "model_missing", "unavailable", "not_checked"]
    configured_model: str
    installed_models: list[InstalledModel] = Field(default_factory=list)
    checks: list[Check]
    network_requests: list[str] = Field(
        default_factory=list,
        description="Every request doctor made (loopback Ollama only); empty when skipped.",
    )


def detect_platform(
    system: str | None = None, osrelease: Callable[[], str] | None = None
) -> PlatformKind:
    system = system or sys.platform
    if system == "darwin":
        return "macos"
    if system.startswith("linux"):
        read = osrelease or _read_osrelease
        return "wsl2" if "microsoft" in read().lower() else "linux"
    if system in ("win32", "cygwin"):
        return "windows"
    return "other"


def _read_osrelease() -> str:
    try:
        return Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        return ""


def unsupported_platform_reason(kind: PlatformKind | None = None) -> str | None:
    """Status reason for platforms on which a scan must never be reported complete."""
    kind = kind or detect_platform()
    if kind == "windows":
        return (
            "native Windows is not a supported platform (use WSL2); results are unverified "
            "and may be incomplete"
        )
    if kind == "other":
        return f"platform '{sys.platform}' is not supported; results are unverified"
    return None


def parse_semgrep_version(text: str | None) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def total_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return round(pages * size / 1024**3, 1) if pages > 0 and size > 0 else None


def run_doctor(
    *,
    semgrep_bin: str = "semgrep",
    ollama: OllamaSettings | None = None,
    check_ollama: bool = True,
    callgraph_enabled: bool = True,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    transport: httpx.BaseTransport | None = None,
    platform_kind: PlatformKind | None = None,
    memory_gb: float | None = None,
) -> DoctorReport:
    ollama = ollama or OllamaSettings.from_env()
    kind = platform_kind or detect_platform()
    memory = memory_gb if memory_gb is not None else total_memory_gb()
    checks: list[Check] = []

    # Platform -------------------------------------------------------------------------
    py = sys.version_info
    checks.append(
        Check(
            id="python",
            status="ok" if py >= (3, 11) else "fail",
            summary=f"Python {py.major}.{py.minor}.{py.micro}",
            action=None if py >= (3, 11) else "install Python 3.11 or newer",
        )
    )
    platform_reason = unsupported_platform_reason(kind)
    checks.append(
        Check(
            id="platform",
            status="warn" if platform_reason else "ok",
            summary=f"{kind} ({platform.platform(terse=True)})",
            details=[platform_reason] if platform_reason else [],
            action="run Tantular Cyber inside WSL2 (docs/install.md)" if platform_reason else None,
        )
    )

    # Bundled rules -------------------------------------------------------------------------
    missing_rules = [str(r.name) for r in default_rules() if not r.is_file()]
    checks.append(
        Check(
            id="rules",
            status="fail" if missing_rules else "ok",
            summary="bundled rules missing: " + ", ".join(missing_rules)
            if missing_rules
            else f"{len(default_rules())} bundled rule files",
            action="reinstall tantular-cyber" if missing_rules else None,
        )
    )

    # Semgrep (external, user-installed) -----------------------------------------------------
    semgrep_path = which(semgrep_bin)
    semgrep_ok = False
    if semgrep_path is None:
        checks.append(
            Check(
                id="semgrep",
                status="fail",
                summary=f"'{semgrep_bin}' not found — scans run tantular-callgraph only (PARTIAL)",
                action="install Semgrep yourself (LGPL-2.1, not bundled): `brew install semgrep` "
                "or `pipx install semgrep`; see docs/install.md",
            )
        )
    else:
        version_text = SemgrepScanner(default_rules(), binary=semgrep_path, runner=runner).version()
        parsed = parse_semgrep_version(version_text)
        if parsed is None:
            checks.append(
                Check(
                    id="semgrep",
                    status="fail",
                    summary=f"{semgrep_path} did not report a version (not runnable)",
                    action="reinstall Semgrep; see docs/install.md",
                )
            )
        else:
            semgrep_ok = True
            lo, hi = SEMGREP_TESTED
            tested = lo <= parsed < hi
            checks.append(
                Check(
                    id="semgrep",
                    status="ok" if tested else "warn",
                    summary=f"Semgrep {'.'.join(map(str, parsed))} at {semgrep_path}",
                    details=[]
                    if tested
                    else [f"outside the tested range >={_v(lo)},<{_v(hi)}; results may differ"],
                    action=None if tested else "use a Semgrep version in the tested range",
                )
            )

    if missing_rules or py < (3, 11) or not (semgrep_ok or callgraph_enabled):
        scan_capability: Literal["full", "partial", "unavailable"] = "unavailable"
    elif semgrep_ok and callgraph_enabled and not platform_reason:
        scan_capability = "full"
    else:
        scan_capability = "partial"

    # Ollama (optional, loopback only, read-only) ----------------------------------------------
    requests: list[str] = []
    installed: list[InstalledModel] = []
    ai: Literal["available", "model_missing", "unavailable", "not_checked"] = "not_checked"
    shown = display_url(ollama.base_url)
    endpoint: Literal["local", "remote", "invalid"] = "local"
    try:
        base = validate_base_url(ollama.base_url, allow_remote=False)
    except RemoteEndpointNotAllowed:
        endpoint = "remote"
    except OllamaError as exc:
        endpoint = "invalid"
        invalid_reason = str(exc)
    if not check_ollama:
        checks.append(Check(id="ollama", status="skip", summary="not checked (--no-ollama)"))
    elif endpoint == "invalid":
        checks.append(
            Check(
                id="ollama",
                status="fail",
                summary=f"endpoint {shown} rejected — not contacted ({invalid_reason})",
                action="use a URL like http://127.0.0.1:11434 (no credentials)",
            )
        )
    elif endpoint == "remote":
        checks.append(
            Check(
                id="ollama",
                status="warn",
                summary=f"endpoint {shown} is not loopback — not contacted",
                details=[
                    "doctor only contacts loopback endpoints; scanning with --ai against a "
                    "remote endpoint sends redacted code excerpts off this machine"
                ],
            )
        )
    else:
        ai, installed = _check_ollama(base, ollama, transport, checks, requests, memory)

    # Privacy ------------------------------------------------------------------------------------
    privacy = [
        "no telemetry: Semgrep runs with --metrics=off and --disable-version-check",
        "HTTP proxies are ignored for Ollama (trust_env=False); scanned code is never executed",
        "doctor made no requests outside loopback and Tantular Cyber wrote no files "
        "(external tools such as Semgrep may keep their own state, e.g. ~/.semgrep)",
        "a loopback endpoint does not by itself guarantee local inference: cloud models are "
        "refused when Ollama marks them, but a local port forwarding elsewhere cannot be "
        "detected (see docs/privacy.md)",
    ]
    if os.environ.get("TANTULAR_ALLOW_REMOTE_OLLAMA", "").lower() in {"1", "true", "yes", "on"}:
        privacy.append("TANTULAR_ALLOW_REMOTE_OLLAMA is set: --ai may send excerpts off-host")
    if any(k.startswith("SEMGREP_APP") for k in os.environ):
        privacy.append("SEMGREP_APP_* variables are present; they are stripped for scans")
    local = endpoint == "local"
    checks.append(
        Check(
            id="privacy",
            status="ok" if local else "warn",
            summary={
                "local": "local-only",
                "remote": "remote LLM endpoint configured",
                "invalid": "invalid LLM endpoint configured (rejected)",
            }[endpoint],
            details=privacy,
        )
    )

    return DoctorReport(
        platform={
            "kind": kind,
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            **({"memory_gb": str(memory)} if memory else {}),
        },
        scan_capability=scan_capability,
        ai_capability=ai,
        configured_model=ollama.model,
        installed_models=installed,
        checks=checks,
        network_requests=requests,
    )


def _v(version: tuple[int, int, int]) -> str:
    return ".".join(map(str, version))


def display_url(url: str) -> str:
    """scheme://host[:port] only — never userinfo, path or query (may carry credentials)."""
    try:
        parts = urlsplit(url.strip())
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "<invalid URL>"
    host = parts.hostname or "?"
    host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme or '?'}://{host}{port}"


def _check_ollama(
    base: str,
    settings: OllamaSettings,
    transport: httpx.BaseTransport | None,
    checks: list[Check],
    requests: list[str],
    memory_gb: float | None,
) -> tuple[Literal["available", "model_missing", "unavailable"], list[InstalledModel]]:
    try:
        with httpx.Client(
            base_url=base, timeout=5.0, trust_env=False, transport=transport, follow_redirects=False
        ) as http:
            requests.append(f"GET {display_url(base)}/api/version")
            version = http.get("/api/version").raise_for_status().json().get("version")
            requests.append(f"GET {display_url(base)}/api/tags")
            tags = http.get("/api/tags").raise_for_status().json().get("models") or []
            installed = [
                InstalledModel(
                    name=str(t.get("name", "")),
                    remote=is_remote_model(t),
                    size_gb=round(t["size"] / 1024**3, 1) if t.get("size") else None,
                    family=(t.get("details") or {}).get("family"),
                    parameter_size=(t.get("details") or {}).get("parameter_size"),
                    quantization=(t.get("details") or {}).get("quantization_level"),
                )
                for t in tags
            ]
            match = next((m for m in installed if model_matches(settings.model, m.name)), None)
            capabilities: list[str] | None = None
            # Never /api/show a cloud model: Ollama forwards that request to the remote host.
            if match and not match.remote:
                requests.append(f"POST {display_url(base)}/api/show")
                show = http.post("/api/show", json={"model": match.name}).raise_for_status()
                capabilities = show.json().get("capabilities")
    except (httpx.HTTPError, ValueError) as exc:
        checks.append(
            Check(
                id="ollama",
                status="info",
                summary=f"not reachable at {display_url(base)} ({type(exc).__name__}) — AI "
                "explanations "
                "unavailable; deterministic scans are unaffected",
                action="optional: install Ollama from https://ollama.com and start it",
            )
        )
        return "unavailable", []

    checks.append(
        Check(
            id="ollama",
            status="ok",
            summary=f"Ollama {version} at {display_url(base)}; {len(installed)} model(s) installed",
            details=[
                f"{m.name}: REMOTE (cloud-backed; runs off this machine)"
                if m.remote
                else f"{m.name}: {m.size_gb or '?'} GB, {m.family or '?'} "
                f"{m.parameter_size or ''} {m.quantization or ''}".rstrip()
                for m in installed
            ],
        )
    )
    if match is None:
        checks.append(
            Check(
                id="model",
                status="info",
                summary=f"configured model '{settings.model}' is not installed",
                details=[
                    "nothing was downloaded. Choose an installed model with --model, or pull "
                    "one yourself: `ollama pull <name>` downloads several GB from the Ollama "
                    "registry and creates/updates a local manifest (its digest can change "
                    "when the tag is updated). Model weights have their own licenses.",
                    f"`{DEFAULT_MODEL}` is a local Modelfile profile; see docs/install.md",
                ],
                action="optional: tantular-cyber scan --ai --model <installed-model> ...",
            )
        )
        return "model_missing", installed

    if match.remote:
        checks.append(
            Check(
                id="model",
                status="fail",
                summary=f"configured model '{match.name}' is cloud-backed (remote execution)",
                details=["scan --ai refuses it unless --allow-remote-ollama is given"],
                action="choose a local model with --model",
            )
        )
        return "unavailable", installed
    details = ["tested with the Tantular weights; other models are untested (see docs)"]
    status: CheckStatus = "ok"
    if capabilities is not None and "completion" not in capabilities:
        status = "warn"
        details.append(f"model capabilities {capabilities} lack 'completion' (chat)")
    if match.size_gb and memory_gb and match.size_gb > 0.75 * memory_gb:
        status = "warn"
        details.append(
            f"model file {match.size_gb} GB vs {memory_gb} GB RAM: expect swapping or slow "
            "responses (guidance only)"
        )
    summary = f"configured model '{match.name}'"
    checks.append(Check(id="model", status=status, summary=summary, details=details))
    return "available", installed


def render_doctor(report: DoctorReport) -> str:
    marks = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info", "skip": "skip"}
    out = [
        f"tantular-cyber {report.tool_version} doctor (read-only: no downloads; "
        "Tantular Cyber writes no files)",
        "",
    ]
    for c in report.checks:
        out.append(f"[{marks[c.status]}] {c.id}: {c.summary}")
        out += [f"         {d}" for d in c.details]
        if c.action:
            out.append(f"         -> {c.action}")
    scan_note = {
        "full": "FULL (Semgrep + tantular-callgraph)",
        "partial": "PARTIAL — reports will be marked degraded",
        "unavailable": "UNAVAILABLE",
    }[report.scan_capability]
    out += [
        "",
        f"scan capability: {scan_note}",
        f"AI explanations (optional, --ai): {report.ai_capability}",
        f"network requests: {', '.join(report.network_requests) or 'none'}",
    ]
    return "\n".join(out) + "\n"

"""Reproducibility metadata for benchmark runs (machine, tools, model, corpus, code)."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel

from .. import __version__

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


class Provenance(BaseModel):
    started_at: str
    finished_at: str | None = None
    tantular_cyber_version: str = __version__
    code_commit: str | None
    code_dirty: bool | None
    python: str
    platform: str
    machine: str
    cpu: str | None
    memory_gb: int | None
    semgrep_version: str | None = None
    ollama_version: str | None = None
    model: str | None = None
    model_digest: str | None = None
    model_details: dict[str, str] = {}
    corpus_id: str
    manifest_version: str
    corpus_sha256: str
    settings: dict[str, object]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _git(*args: str) -> str | None:
    # Only ever inspects this tool's own checkout, never an analyzed repository.
    if not (_PACKAGE_ROOT / ".git").exists():
        return None
    return _run(["git", "-C", str(_PACKAGE_ROOT), *args])


def machine() -> dict[str, object]:
    cpu = mem = None
    if sys.platform == "darwin":
        cpu = _run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
        raw = _run(["/usr/sbin/sysctl", "-n", "hw.memsize"])
        mem = round(int(raw) / 2**30) if raw and raw.isdigit() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "memory_gb": mem,
    }


def code_state() -> tuple[str | None, bool | None]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return commit, (bool(status) if status is not None else None)


def model_state(base_url: str, model: str) -> dict[str, object]:
    try:
        with httpx.Client(base_url=base_url, timeout=10, trust_env=False) as http:
            version = http.get("/api/version").json().get("version")
            tags = http.get("/api/tags").json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return {}
    entry = next((m for m in tags if m.get("name") == model), {})
    details = entry.get("details") or {}
    return {
        "ollama_version": version,
        "model_digest": entry.get("digest"),
        "model_details": {
            k: str(details[k])
            for k in ("family", "parameter_size", "quantization_level", "format")
            if k in details
        },
    }

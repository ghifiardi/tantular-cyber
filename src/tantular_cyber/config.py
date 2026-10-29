"""Runtime configuration. Precedence: CLI flags > environment variables > defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "tantular-cyber:latest"
# Bundled local rulesets, one per CWE. Order is stable so rule ids and reports are too.
DEFAULT_RULESETS = (
    "python-sqli.yml",  # CWE-89
    "python-cmdi.yml",  # CWE-78
    "python-path-traversal.yml",  # CWE-22
    "python-xss.yml",  # CWE-79 (Flask/Jinja2, Django, HTML templates)
    "python-secrets.yml",  # CWE-798
)
DEFAULT_RULESET = DEFAULT_RULESETS[0]  # backwards-compatible name (CWE-89 ruleset)

_TRUE = {"1", "true", "yes", "on"}


def default_rules() -> tuple[Path, ...]:
    """Bundled local rulesets shipped inside the package (works from a wheel or a checkout)."""
    base = resources.files("tantular_cyber") / "rules"
    return tuple(Path(str(base / name)) for name in DEFAULT_RULESETS)


# Data that a wheel ships under ``tantular_cyber/_data`` (see pyproject force-include) and
# that a source checkout keeps at the repository root.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def _bundled(wheel_path: str, checkout_path: str) -> Path:
    bundled = Path(str(resources.files("tantular_cyber") / "_data" / wheel_path))
    return bundled if bundled.exists() else _CHECKOUT_ROOT / checkout_path


def default_corpus() -> Path:
    """Synthetic benchmark corpus (manifest.json + cases)."""
    return _bundled("corpus", "benchmarks/corpus")


def default_realworld_manifest() -> Path:
    """Pointer-only real-world manifest (no upstream code)."""
    return _bundled("realworld/manifest.json", "benchmarks/realworld/manifest.json")


def sample_project() -> Path:
    """Small synthetic vulnerable project used by ``tantular-cyber sample``."""
    return _bundled("sample", "tests/fixtures/vulnerable")


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = 300.0
    max_retries: int = 2
    allow_remote: bool = False
    temperature: float = 0.1
    num_ctx: int = 8192
    # Optional pinned sampling options (None = not sent; the model's defaults apply).
    # Model comparisons pin all of them so Modelfile defaults cannot differ between arms.
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    num_predict: int | None = None
    seed: int | None = None
    # False = send no temperature/num_ctx either, so the Modelfile's PARAMETERs apply
    # (used by the "profile" comparison mode).
    pin_core_options: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: object) -> OllamaSettings:
        env = os.environ if env is None else env
        values: dict[str, object] = {}
        if url := env.get("TANTULAR_OLLAMA_URL"):
            values["base_url"] = url
        if model := env.get("TANTULAR_MODEL"):
            values["model"] = model
        if timeout := env.get("TANTULAR_OLLAMA_TIMEOUT"):
            values["timeout_s"] = float(timeout)
        if retries := env.get("TANTULAR_OLLAMA_RETRIES"):
            values["max_retries"] = int(retries)
        if env.get("TANTULAR_ALLOW_REMOTE_OLLAMA", "").lower() in _TRUE:
            values["allow_remote"] = True
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ScanSettings:
    repo: Path
    output: Path
    rules: tuple[Path, ...] = field(default_factory=default_rules)
    semgrep_bin: str = "semgrep"
    semgrep_timeout_s: float = 600.0
    max_file_bytes: int = 1_000_000
    allow_output_inside_repo: bool = False
    keep_raw_scanner_output: bool = False
    llm_enabled: bool = True
    max_llm_findings: int = 25
    context_lines: int = 3
    context_strategy: str = "ast"  # "ast" (bounded, AST-aware) or "window" (POC-002 behavior)
    context_max_lines: int = 60
    context_max_chars: int = 4000
    callgraph_enabled: bool = True
    callgraph_max_depth: int = 3
    ollama: OllamaSettings = field(default_factory=OllamaSettings)

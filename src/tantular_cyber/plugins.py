"""Scanner plugin contract (API 1.0) — specification and validation only.

POC-006 defines *how* a scanner plugs in without executing third-party code:

* :class:`PluginManifest` — a declarative, versioned description of a scanner: identity,
  languages/CWEs, rule-id namespace, requested permissions and resource bounds.
* :func:`negotiate` — API compatibility (same major, host minor >= plugin minor).
* :func:`normalize_plugin_output` — the only path from plugin output to findings: strict
  validation into immutable :class:`~tantular_cyber.scanner.RawFinding` objects. Plugins
  cannot produce or modify model analyses, report status or other scanners' findings.
* :data:`BUILTIN_SCANNERS` — descriptors of the trusted, in-tree scanners.

Third-party plugins are **untrusted**. Running them in-process or in a plain subprocess is
not a sandbox, so :func:`execute_plugin` always refuses until an enforceable isolation
boundary exists (see docs/plugin-api.md). Nothing in this module imports, loads or runs
plugin code, and no entry points are discovered.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .models import SCHEMA_VERSION
from .scanner import RawFinding

PLUGIN_API_VERSION = "1.0"
REPORT_SCHEMA_VERSION = SCHEMA_VERSION
MAX_PLUGIN_FINDINGS = 10_000
_ID = r"^[a-z][a-z0-9-]{2,47}$"
_SEVERITIES = ("critical", "high", "medium", "low", "info")


class PluginExecutionDisabled(RuntimeError):
    """Raised for any attempt to run a third-party plugin (no enforceable sandbox yet)."""


class ResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_wall_s: int = Field(default=300, ge=1, le=3600)
    max_cpu_s: int = Field(default=300, ge=1, le=3600)
    max_memory_mb: int = Field(default=1024, ge=64, le=16384)
    max_output_bytes: int = Field(default=16_000_000, ge=1024, le=64_000_000)


class PluginPermissions(BaseModel):
    """What a plugin may do. Anything not expressible here is denied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    read_roots: tuple[Literal["target_repo"], ...] = ("target_repo",)
    network: Literal[False] = False
    write: Literal["none"] = "none"
    spawn_processes: Literal[False] = False
    environment: tuple[str, ...] = Field(
        default=(), description="Environment variable names passed through (never secrets)."
    )
    limits: ResourceLimits = ResourceLimits()

    @field_validator("environment")
    @classmethod
    def _no_secrets(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        secretish = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIAL|AUTH)", re.I)
        bad = [n for n in names if secretish.search(n) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", n)]
        if bad:
            raise ValueError(f"environment variables not allowed for plugins: {bad}")
        return names


class PluginEntrypoint(BaseModel):
    """Declarative launch description. Recorded for a future sandbox; never executed now."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["subprocess-json"] = "subprocess-json"
    command: tuple[str, ...] = Field(min_length=1, max_length=32)
    protocol: Literal["tantular-plugin-output/1"] = "tantular-plugin-output/1"


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal[1] = 1
    plugin_api: str = Field(pattern=r"^\d+\.\d+$", description="MAJOR.MINOR the plugin targets")
    id: str = Field(pattern=_ID)
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+([.+-][0-9A-Za-z.+-]+)?$")
    publisher: str = Field(min_length=1, max_length=80)
    license: str = Field(min_length=1, max_length=80, description="SPDX expression")
    description: str = Field(default="", max_length=500)
    languages: tuple[str, ...] = Field(min_length=1)
    cwe: tuple[str, ...] = Field(default=())
    rule_id_prefix: str
    permissions: PluginPermissions = PluginPermissions()
    entrypoint: PluginEntrypoint
    trust: Literal["third_party"] = "third_party"

    @field_validator("cwe")
    @classmethod
    def _cwe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        bad = [v for v in values if not re.fullmatch(r"CWE-\d{1,5}", v)]
        if bad:
            raise ValueError(f"CWE ids must look like CWE-89: {bad}")
        return values

    @field_validator("rule_id_prefix")
    @classmethod
    def _prefix(cls, value: str, info) -> str:  # type: ignore[no-untyped-def]
        plugin_id = info.data.get("id")
        if plugin_id and value != f"plugin.{plugin_id}.":
            raise ValueError(f"rule_id_prefix must be 'plugin.{plugin_id}.'")
        return value

    @property
    def tool_name(self) -> str:
        return f"plugin:{self.id}"


class Compatibility(BaseModel):
    compatible: bool
    host_api: str = PLUGIN_API_VERSION
    plugin_api: str
    reason: str


def negotiate(manifest: PluginManifest, host_api: str = PLUGIN_API_VERSION) -> Compatibility:
    host_major, host_minor = (int(x) for x in host_api.split("."))
    major, minor = (int(x) for x in manifest.plugin_api.split("."))
    if major != host_major:
        reason = f"major version {major} != host {host_major}"
        ok = False
    elif minor > host_minor:
        reason = f"plugin needs API {manifest.plugin_api}; host provides {host_api}"
        ok = False
    else:
        reason = "compatible"
        ok = True
    return Compatibility(compatible=ok, plugin_api=manifest.plugin_api, reason=reason)


class PluginFinding(BaseModel):
    """One finding in ``tantular-plugin-output/1``."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(max_length=200)
    path: str = Field(max_length=1024)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    message: str = Field(max_length=2000)
    severity: Literal["critical", "high", "medium", "low", "info"]
    cwe: tuple[str, ...] = ()
    confidence: Literal["high", "medium", "low"] | None = None
    uncertainty: tuple[str, ...] = Field(default=(), max_length=20)


class PluginOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["tantular-plugin-output/1"]
    plugin_id: str
    plugin_version: str
    findings: list[PluginFinding] = Field(max_length=MAX_PLUGIN_FINDINGS)


class PluginOutputError(ValueError):
    pass


def normalize_plugin_output(
    manifest: PluginManifest, payload: Mapping[str, object], repo_files: Sequence[str]
) -> list[RawFinding]:
    """Validate plugin output and convert it to immutable scanner findings.

    Rejects (fail closed) the whole output on any violation: wrong plugin identity, rule ids
    outside the plugin's namespace, CWEs it did not declare, paths that are absolute, escape
    the repository or were not in the scanned inventory, or inverted line ranges.
    """
    try:
        out = PluginOutput.model_validate(payload)
    except ValidationError as exc:
        raise PluginOutputError(f"invalid plugin output: {exc.error_count()} error(s)") from exc
    if out.plugin_id != manifest.id or out.plugin_version != manifest.version:
        raise PluginOutputError("plugin output identity does not match the manifest")
    allowed = set(repo_files)
    findings: list[RawFinding] = []
    for i, f in enumerate(out.findings):
        where = f"finding {i}"
        if not f.rule_id.startswith(manifest.rule_id_prefix):
            raise PluginOutputError(f"{where}: rule id outside '{manifest.rule_id_prefix}'")
        path = PurePosixPath(f.path)
        if path.is_absolute() or ".." in path.parts or "\\" in f.path or f.path not in allowed:
            raise PluginOutputError(f"{where}: path is not a scanned repository file")
        if f.end_line < f.start_line:
            raise PluginOutputError(f"{where}: end_line < start_line")
        undeclared = set(f.cwe) - set(manifest.cwe)
        if undeclared:
            raise PluginOutputError(f"{where}: CWE not declared in manifest: {sorted(undeclared)}")
        findings.append(
            RawFinding(
                tool=manifest.tool_name,
                rule_id=f.rule_id,
                path=f.path,
                start_line=f.start_line,
                end_line=f.end_line,
                message=f.message,
                severity=f.severity,
                cwe=f.cwe,
                confidence=f.confidence,
                uncertainty=f.uncertainty,
            )
        )
    return findings


def execute_plugin(manifest: PluginManifest, *_: object, **__: object) -> None:
    """Always refuses: third-party plugins never run without an enforceable sandbox."""
    raise PluginExecutionDisabled(
        f"third-party plugin '{manifest.id}' was not executed: plugin execution is disabled "
        "until an enforceable sandbox (no network, read-only repo, bounded CPU/memory/time) "
        "exists. A subprocess with a timeout is not a sandbox."
    )


class BuiltinScanner(BaseModel):
    id: str
    tool_name: str
    plugin_api: str = PLUGIN_API_VERSION
    trust: Literal["builtin"] = "builtin"
    languages: tuple[str, ...]
    cwe: tuple[str, ...]
    rule_id_prefix: str
    execution: str
    version_source: str


BUILTIN_SCANNERS: tuple[BuiltinScanner, ...] = (
    BuiltinScanner(
        id="semgrep",
        tool_name="semgrep",
        languages=("python", "generic"),
        cwe=("CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-798"),
        rule_id_prefix="tantular.",
        execution="external Semgrep CLI (user-installed), bundled local rules, metrics off",
        version_source="`semgrep --version`",
    ),
    BuiltinScanner(
        id="tantular-callgraph",
        tool_name="tantular-callgraph",
        languages=("python",),
        cwe=("CWE-89", "CWE-78", "CWE-22"),
        rule_id_prefix="tantular.callgraph.",
        execution="in-process AST analysis (parses files; never imports or runs them)",
        version_source="tantular-cyber package version",
    ),
)


def plugin_policy() -> dict[str, object]:
    return {
        "plugin_api": PLUGIN_API_VERSION,
        "report_schema": REPORT_SCHEMA_VERSION,
        "builtin_scanners": [b.model_dump() for b in BUILTIN_SCANNERS],
        "third_party_execution": "disabled (no enforceable sandbox; see docs/plugin-api.md)",
        "severities": list(_SEVERITIES),
    }

"""Scanner adapters. Semgrep is the first implementation of the ``Scanner`` protocol.

Additional tools (Bandit, Gitleaks, Trivy, ...) plug in by implementing ``Scanner`` and
returning ``RawFinding`` objects; normalization and reporting stay unchanged.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .security import sanitize_text, scrub_paths

TARGET_BATCH_SIZE = 200

# Semgrep severities -> normalized severities.
_SEVERITY_MAP = {
    "CRITICAL": "critical",
    "ERROR": "high",
    "HIGH": "high",
    "WARNING": "medium",
    "MEDIUM": "medium",
    "INFO": "low",
    "LOW": "low",
    "INVENTORY": "info",
    "EXPERIMENT": "info",
}

_CWE_RE = re.compile(r"CWE-(\d+)", re.IGNORECASE)
_LANGUAGES_RE = re.compile(r"^\s*languages:\s*\[([^\]]*)\]", re.MULTILINE)

# File extensions per Semgrep language, used to decide which targets a ruleset applies to.
LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py", ".pyi"),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "typescript": (".ts", ".tsx", ".mts", ".cts"),
    "java": (".java",),
    "go": (".go",),
    "ruby": (".rb",),
    "php": (".php",),
    "c": (".c", ".h"),
    "cpp": (".cpp", ".cc", ".cxx", ".hpp", ".hh"),
    "csharp": (".cs",),
    "kotlin": (".kt", ".kts"),
    "rust": (".rs",),
    "scala": (".scala",),
    "swift": (".swift",),
    # Semgrep's "generic" (text) mode. The bundled generic rules only target HTML/Jinja
    # templates via paths.include, so coverage is verified for these extensions.
    "generic": (".html", ".htm", ".jinja", ".jinja2", ".j2"),
}
_LANGUAGE_ALIASES = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "c#": "csharp",
    "c++": "cpp",
    "golang": "go",
    "rb": "ruby",
    "kt": "kotlin",
}
_RULE_ID_RE = re.compile(r"^\s*-\s*id:\s*['\"]?([A-Za-z0-9_.\-]+)['\"]?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class RawFinding:
    tool: str
    rule_id: str
    path: str  # repo-relative POSIX path
    start_line: int
    end_line: int
    message: str
    severity: str  # normalized
    cwe: tuple[str, ...] = ()
    confidence: str | None = None


@dataclass
class ScannerRun:
    tool: str
    version: str | None
    rules: list[str]
    targets: int
    exit_code: int | None = None
    duration_s: float = 0.0
    findings: list[RawFinding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    succeeded: bool = True
    raw_outputs: list[dict[str, Any]] = field(default_factory=list)
    # Coverage: which submitted targets the rules apply to and which the tool really scanned.
    languages: list[str] = field(default_factory=list)
    in_scope: list[str] | None = None  # None = unknown (cannot be verified)
    scanned: set[str] = field(default_factory=set)
    scanned_reported: bool = False
    tool_skipped: dict[str, str] = field(default_factory=dict)


class Scanner(Protocol):
    name: str

    def version(self) -> str | None: ...

    def scan(self, repo: Path, targets: Sequence[str]) -> ScannerRun: ...


Runner = Callable[..., subprocess.CompletedProcess[str]]


def extract_cwes(value: object) -> tuple[str, ...]:
    items = value if isinstance(value, list) else [value]
    found: list[str] = []
    for item in items:
        if isinstance(item, str):
            for num in _CWE_RE.findall(item):
                cwe = f"CWE-{int(num)}"
                if cwe not in found:
                    found.append(cwe)
    return tuple(found)


def declared_rule_ids(rule_files: Sequence[Path]) -> list[str]:
    ids: list[str] = []
    for rule_file in rule_files:
        ids.extend(_RULE_ID_RE.findall(rule_file.read_text(encoding="utf-8")))
    return ids


def declared_languages(rule_files: Sequence[Path]) -> list[str]:
    langs: list[str] = []
    for rule_file in rule_files:
        for group in _LANGUAGES_RE.findall(rule_file.read_text(encoding="utf-8")):
            for raw in group.split(","):
                lang = raw.strip().strip("'\"").lower()
                lang = _LANGUAGE_ALIASES.get(lang, lang)
                if lang and lang not in langs:
                    langs.append(lang)
    return langs


def in_scope_targets(targets: Sequence[str], languages: Sequence[str]) -> list[str] | None:
    """Targets the rules apply to, or None when a language's file types are unknown."""
    extensions: list[str] = []
    for lang in languages:
        if lang not in LANGUAGE_EXTENSIONS:
            return None
        extensions.extend(LANGUAGE_EXTENSIONS[lang])
    suffixes = tuple(extensions)
    return [t for t in targets if t.lower().endswith(suffixes)]


def parse_semgrep_paths(data: dict[str, Any], repo: Path) -> tuple[set[str] | None, dict[str, str]]:
    """Return (scanned paths or None if not reported, skipped path -> reason)."""
    paths = data.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("scanned"), list):
        return None, {}
    scanned = {rel for p in paths["scanned"] if (rel := _relative_path(str(p), repo))}
    skipped: dict[str, str] = {}
    for item in paths.get("skipped") or []:
        if isinstance(item, dict) and (rel := _relative_path(str(item.get("path", "")), repo)):
            skipped[rel] = sanitize_text(str(item.get("reason", "unknown")), 100)
    return scanned, skipped


def normalize_rule_id(check_id: str, known_ids: Sequence[str]) -> str:
    """Semgrep prefixes rule ids with the config file's directory; strip that prefix."""
    for rule_id in sorted(known_ids, key=len, reverse=True):
        if check_id == rule_id or check_id.endswith("." + rule_id):
            return rule_id
    return check_id


def _relative_path(raw_path: str, repo: Path) -> str | None:
    p = Path(raw_path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(repo)
        except ValueError:
            return None
    parts = [part for part in p.parts if part not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return Path(*parts).as_posix()


def parse_semgrep_json(
    data: dict[str, Any], repo: Path, known_rule_ids: Sequence[str] = ()
) -> tuple[list[RawFinding], list[str]]:
    """Convert Semgrep JSON output to ``RawFinding`` objects plus tool error strings."""
    findings: list[RawFinding] = []
    errors: list[str] = []

    for err in data.get("errors") or []:
        if not isinstance(err, dict):
            continue
        level = str(err.get("level", "error"))
        kind = str(err.get("type", "error"))
        if isinstance(err.get("type"), list):
            kind = str(err["type"][0])
        message = scrub_paths(str(err.get("message", "")), repo)
        errors.append(sanitize_text(f"semgrep {level} [{kind}]: {message}", 500))

    for result in data.get("results") or []:
        try:
            extra = result.get("extra") or {}
            metadata = extra.get("metadata") or {}
            rel = _relative_path(str(result["path"]), repo)
            if rel is None:
                errors.append("semgrep result outside repository scope was discarded")
                continue
            start = int(result["start"]["line"])
            end = int(result.get("end", {}).get("line", start))
            findings.append(
                RawFinding(
                    tool="semgrep",
                    rule_id=normalize_rule_id(str(result["check_id"]), known_rule_ids),
                    path=rel,
                    start_line=start,
                    end_line=max(start, end),
                    message=sanitize_text(str(extra.get("message", "")), 2000),
                    severity=_SEVERITY_MAP.get(str(extra.get("severity", "")).upper(), "medium"),
                    cwe=extract_cwes(metadata.get("cwe")),
                    confidence=(
                        str(metadata["confidence"]).lower() if metadata.get("confidence") else None
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed semgrep result skipped: {type(exc).__name__}")
    return findings, errors


class SemgrepScanner:
    """Runs the local ``semgrep`` CLI with explicit local rules, offline-friendly."""

    name = "semgrep"

    def __init__(
        self,
        rules: Sequence[Path],
        *,
        binary: str = "semgrep",
        timeout_s: float = 600.0,
        max_target_bytes: int = 1_000_000,
        runner: Runner = subprocess.run,
    ) -> None:
        if not rules:
            raise ValueError("at least one local rule file is required")
        self.rules = [Path(r) for r in rules]
        self.binary = binary
        self.timeout_s = timeout_s
        self.max_target_bytes = max_target_bytes
        self._run = runner
        self._version: str | None = None

    @staticmethod
    def environment() -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("SEMGREP_APP")}
        env.update(
            SEMGREP_SEND_METRICS="off",
            SEMGREP_ENABLE_VERSION_CHECK="0",
            SEMGREP_FORCE_COLOR="0",
            NO_COLOR="1",
        )
        return env

    def build_command(self, targets: Sequence[str]) -> list[str]:
        cmd = [
            self.binary,
            "scan",
            "--json",
            "--metrics=off",
            "--disable-version-check",
            "--quiet",
            "--max-target-bytes",
            str(self.max_target_bytes),
            "--timeout",
            "30",
        ]
        for rule in self.rules:
            cmd += ["--config", str(rule)]
        cmd.append("--")
        cmd.extend(targets)
        return cmd

    def version(self) -> str | None:
        if self._version is None:
            try:
                proc = self._run(
                    [self.binary, "--version", "--disable-version-check"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=self.environment(),
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            self._version = lines[-1] if proc.returncode == 0 and lines else None
        return self._version

    def scan(self, repo: Path, targets: Sequence[str]) -> ScannerRun:
        known_ids = declared_rule_ids(self.rules)
        languages = declared_languages(self.rules)
        run = ScannerRun(
            tool=self.name,
            version=None,
            rules=known_ids,
            targets=len(targets),
            languages=languages,
            in_scope=in_scope_targets(targets, languages),
        )
        run.scanned_reported = True  # cleared if any batch omits its scanned-path list
        started = time.monotonic()
        try:
            run.version = self.version()
            if run.version is None:
                run.succeeded = False
                run.scanned_reported = False
                run.errors.append(
                    f"semgrep binary '{self.binary}' not found or not runnable; install with "
                    "`brew install semgrep` or `pip install 'tantular-cyber[semgrep]'`"
                )
                return run
            for i in range(0, len(targets), TARGET_BATCH_SIZE):
                self._scan_batch(repo, list(targets[i : i + TARGET_BATCH_SIZE]), known_ids, run)
        finally:
            run.duration_s = round(time.monotonic() - started, 3)
        return run

    def _scan_batch(
        self, repo: Path, batch: list[str], known_ids: list[str], run: ScannerRun
    ) -> None:
        try:
            proc = self._run(
                self.build_command(batch),
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=self.environment(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            run.succeeded = False
            run.errors.append(f"semgrep timed out after {self.timeout_s:.0f}s")
            return
        except OSError as exc:
            run.succeeded = False
            run.errors.append(f"semgrep could not be started: {type(exc).__name__}")
            return

        run.exit_code = proc.returncode if run.exit_code in (None, 0, 1) else run.exit_code
        try:
            data = json.loads(proc.stdout) if proc.stdout.strip() else None
        except json.JSONDecodeError:
            data = None

        # 0 = no blocking findings, 1 = findings (with --error); anything else is a tool error.
        if proc.returncode not in (0, 1) or not isinstance(data, dict):
            run.succeeded = False
            stderr = sanitize_text(scrub_paths(proc.stderr.strip(), repo), 800)
            run.errors.append(
                f"semgrep exited with code {proc.returncode}"
                + (f": {stderr}" if stderr else "")
                + ("" if isinstance(data, dict) else " (no parseable JSON output)")
            )
        if isinstance(data, dict):
            findings, errors = parse_semgrep_json(data, repo, known_ids)
            run.findings.extend(findings)
            run.errors.extend(errors)
            run.raw_outputs.append(data)
            scanned, skipped = parse_semgrep_paths(data, repo)
            if scanned is None:
                run.scanned_reported = False
            else:
                run.scanned |= scanned
            run.tool_skipped.update(skipped)
        else:
            run.scanned_reported = False

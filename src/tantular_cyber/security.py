"""Path safety, target-file inventory, and credential redaction.

Everything here is read-only: files are opened for reading only and nothing in the
analyzed repository is ever executed, imported, or installed.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

BINARY_SNIFF_BYTES = 8192

# Directories never descended into: VCS metadata, dependency trees and caches.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

# File names that commonly hold credentials. Matched case-insensitively on the base name.
SECRET_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.kdbx",
    "*.tfstate",
    "*.tfstate.*",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".git-credentials",
    "credentials",
    "credentials.*",
    "secrets.*",
    "service-account*.json",
)


class PathSafetyError(ValueError):
    """Raised when an input or output path violates the safety policy."""


@dataclass(frozen=True)
class SkippedFile:
    path: str
    reason: str


@dataclass
class FileInventory:
    files: list[str] = field(default_factory=list)  # repo-relative POSIX paths
    skipped: list[SkippedFile] = field(default_factory=list)


def _is_within(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def resolve_repo(path: str | os.PathLike[str]) -> Path:
    """Canonicalize the analysis target; it must be an existing directory."""
    raw = Path(path).expanduser()
    if not raw.exists():
        raise PathSafetyError(f"repository path does not exist: {raw}")
    if not raw.is_dir():
        raise PathSafetyError(f"repository path is not a directory: {raw}")
    return raw.resolve(strict=True)


def resolve_output(
    path: str | os.PathLike[str], repo: Path, *, allow_inside_repo: bool = False
) -> Path:
    """Canonicalize the report directory and keep it out of the analyzed tree by default."""
    out = Path(path).expanduser().resolve()
    if out.exists() and not out.is_dir():
        raise PathSafetyError(f"output path exists and is not a directory: {out}")
    if _is_within(repo, out) and not allow_inside_repo:
        raise PathSafetyError(
            "output directory is inside the analyzed repository; choose another location "
            "or pass --allow-output-inside-repo"
        )
    return out


def is_secret_filename(name: str) -> bool:
    lower = name.lower()
    return any(fnmatch.fnmatchcase(lower, pattern) for pattern in SECRET_FILE_PATTERNS)


def is_binary_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\x00" in handle.read(BINARY_SNIFF_BYTES)


def collect_files(repo: Path, *, max_file_bytes: int) -> FileInventory:
    """Walk ``repo`` without following symlinks and return eligible scan targets."""
    inventory = FileInventory()

    def rel(p: Path) -> str:
        return p.relative_to(repo).as_posix()

    def symlink_reason(p: Path) -> str:
        try:
            target = p.resolve(strict=True)
        except (OSError, RuntimeError):
            return "broken_symlink"
        return "symlink" if _is_within(repo, target) else "symlink_escapes_scope"

    for root, dirs, files in os.walk(repo, followlinks=False):
        root_path = Path(root)
        kept: list[str] = []
        for name in sorted(dirs):
            p = root_path / name
            if name in EXCLUDED_DIRS:
                inventory.skipped.append(SkippedFile(rel(p), "excluded_directory"))
            elif p.is_symlink():
                inventory.skipped.append(SkippedFile(rel(p), symlink_reason(p)))
            else:
                kept.append(name)
        dirs[:] = kept

        for name in sorted(files):
            p = root_path / name
            reason: str | None = None
            if p.is_symlink():
                reason = symlink_reason(p)
            elif is_secret_filename(name):
                reason = "possible_secret_file"
            elif not p.is_file():
                reason = "not_regular_file"
            else:
                try:
                    if p.stat().st_size > max_file_bytes:
                        reason = "oversized"
                    elif is_binary_file(p):
                        reason = "binary"
                except OSError:
                    reason = "unreadable"
            if reason:
                inventory.skipped.append(SkippedFile(rel(p), reason))
            else:
                inventory.files.append(rel(p))
    return inventory


# --- Redaction ---------------------------------------------------------------

_REDACTION_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?"
            r"(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)"
        ),
        "[REDACTED:private_key]",
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED:aws_access_key]"),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "[REDACTED:github_token]",
    ),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED:slack_token]"),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED:google_api_key]"),
    ("api_secret_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED:api_key]"),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        "[REDACTED:jwt]",
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{16,}"),
        r"\1 [REDACTED:token]",
    ),
    (
        "url_credentials",
        re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"),
        r"\1[REDACTED:credentials]@",
    ),
    (
        "secret_assignment",
        re.compile(
            r"""(?i)\b([A-Za-z0-9_.-]*(?:password|passwd|pwd|secret|token|api[_-]?key|"""
            r"""access[_-]?key|private[_-]?key|client[_-]?secret)[A-Za-z0-9_.-]*["']?)"""
            r"""(\s*[:=]\s*)(["'])([^"'\n]{4,})\3"""
        ),
        r"\1\2\3[REDACTED:secret]\3",
    ),
)


def redact(text: str) -> tuple[str, int]:
    """Mask likely credentials. Returns the redacted text and the number of replacements."""
    total = 0
    for _name, pattern, replacement in _REDACTION_RULES:
        text, count = pattern.subn(replacement, text)
        total += count
    return text, total


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")


def sanitize_text(text: str, max_len: int = 4000) -> str:
    """Strip control / bidi characters, redact secrets, and bound length of untrusted text."""
    text = _CONTROL_CHARS.sub("", text)
    text, _ = redact(text)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def scrub_paths(text: str, repo: Path) -> str:
    """Replace absolute local paths (repo, home) with neutral placeholders."""
    text = text.replace(str(repo), "<repo>")
    home = str(Path.home())
    if home and home != "/":
        text = text.replace(home, "~")
    return text

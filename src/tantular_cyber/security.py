"""Path safety, target-file inventory, and credential redaction.

Everything here is read-only: files are opened for reading only and nothing in the
analyzed repository is ever executed, imported, or installed.
"""

from __future__ import annotations

import base64
import binascii
import fnmatch
import os
import re
from collections.abc import Callable
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


# Cyrillic/Greek letters that render like Latin ones (homoglyph evasion), by code point.
CONFUSABLES = str.maketrans(
    {
        0x0430: "a", 0x0435: "e", 0x043E: "o", 0x0440: "p", 0x0441: "c", 0x0445: "x",
        0x0443: "y", 0x0456: "i", 0x0455: "s", 0x043A: "k", 0x043C: "m", 0x0442: "t",
        0x043D: "h", 0x0432: "b", 0x0410: "A", 0x0415: "E", 0x041E: "O", 0x0420: "P",
        0x0421: "C", 0x0425: "X", 0x0406: "I", 0x041A: "K", 0x041C: "M", 0x0422: "T",
        0x041D: "H", 0x0412: "B", 0x03BF: "o", 0x03B1: "a", 0x03B5: "e", 0x039F: "O",
    }
)  # fmt: skip


# --- Redaction ---------------------------------------------------------------
#
# Redaction is heuristic and errs toward masking. Every rule preserves the number of
# newlines it replaces, so a whole file can be redacted before evidence lines are cut and
# scanner line numbers stay correct (multiline secrets never leak line by line).

# Regex of credential-like identifiers (not a credential itself).
_SECRET_WORD = (
    r"(?:password|passwd|pwd|(?<![a-z])pass(?![a-z])|secret|token|api[_-]?key|"  # noqa: S105
    r"access[_-]?key|private[_-]?key|client[_-]?secret|credentials?)"
)
_SECRET_NAME = rf"[A-Za-z0-9_.-]*{_SECRET_WORD}[A-Za-z0-9_.-]*"
# Optional Python type annotation between the name and the assignment: `pw: str = "..."`.
_ANNOTATION = r"(?::\s*[A-Za-z_][\w.]*(?:\[[^\]\n]*\])?\s*(?==))?"
_COMMENT_WORD = (
    r"(?:password|passwd|pwd|pass|secret|token|api[\s_-]?key|kata\s+sandi|sandi|"
    r"credentials?|kredensial)"
)
_NOT_A_VALUE = (
    r"(?!\[REDACTED)"
    r"(?!(?:required|not|missing|invalid|empty|none|null|set|stored|hashed|optional)\b)"
)
_B64_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/_-]{20,}={0,2})(?![A-Za-z0-9+/=_-])"
)


def _decode_b64(value: str) -> str | None:
    """Decode standard or URL-safe base64 into printable text, or return None."""
    padded = value + "=" * (-len(value) % 4)
    for altchars in (None, b"-_"):
        try:
            raw = base64.b64decode(padded, altchars=altchars, validate=True)
            text = raw.decode("utf-8")
        except (binascii.Error, ValueError):
            continue
        if text and all(ch.isprintable() or ch in "\n\t" for ch in text):
            return text
    return None


def _encoded_secret(match: re.Match[str]) -> str | None:
    decoded = _decode_b64(match.group(1))
    if decoded is not None and _redact_known(decoded)[1] > 0:
        return "[REDACTED:encoded_secret]"
    return None


def _basic_auth(match: re.Match[str]) -> str | None:
    decoded = _decode_b64(match.group(2))
    if decoded is not None and ":" in decoded:
        return f"{match.group(1)} [REDACTED:basic_auth]"
    return None


Replacement = str | Callable[[re.Match[str]], str | None]

# Format-specific credentials (also used to inspect decoded base64 payloads).
_KNOWN_FORMATS: tuple[tuple[re.Pattern[str], Replacement], ...] = (
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?"
            r"(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)"
        ),
        "[REDACTED:private_key]",
    ),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED:aws_access_key]"),
    (
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "[REDACTED:github_token]",
    ),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED:slack_token]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED:google_api_key]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED:api_key]"),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        "[REDACTED:jwt]",
    ),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{16,}"), r"\1 [REDACTED:token]"),
    (
        re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"),
        r"\1[REDACTED:credentials]@",
    ),
)

_CONTEXTUAL_ALL: tuple[tuple[re.Pattern[str], Replacement], ...] = (
    # HTTP Basic credentials (base64 "user:pass").
    (re.compile(r"(?i)\b(basic)\s+([A-Za-z0-9+/]{8,}={0,2})(?![A-Za-z0-9+/=])"), _basic_auth),
    # Base64 blobs whose decoded content is a known credential format.
    (_B64_CANDIDATE, _encoded_secret),
    # Triple-quoted (possibly multiline) value assigned to a secret-like name.
    (
        re.compile(
            rf"""(?i)\b({_SECRET_NAME}["']?)(\s*{_ANNOTATION}[:=]\s*)("{{3}}|'{{3}})"""
            r"""([\s\S]*?)\3"""
        ),
        r"\1\2\3[REDACTED:secret]\3",
    ),
    # Quoted value assigned to a secret-like name (code, JSON, YAML).
    (
        re.compile(
            rf"""(?i)\b({_SECRET_NAME}["']?)(\s*{_ANNOTATION}[:=]\s*)(["'])"""
            r"""(?!\[REDACTED)([^"'\n]{4,})\3"""
        ),
        r"\1\2\3[REDACTED:secret]\3",
    ),
    # Unquoted value (.env / YAML / INI style). Requires a digit or symbol so identifiers and
    # expressions such as `password = get_password(user)` are left alone.
    (
        re.compile(
            rf"""(?im)\b({_SECRET_NAME})(\s*[:=][ \t]*)(?!["'\[])"""
            r"""(?=[^\s,;#]*[\d!@#$%^&*+\-/~])([^\s"',;#()\[\]{}.]{6,})(?=[ \t]*(?:$|[,;#]))"""
        ),
        r"\1\2[REDACTED:secret]",
    ),
    # Credentials written in comments: "# admin password is hunter2", "// api key: abc".
    (
        re.compile(
            rf"""(?im)((?:#|//|--|/\*|<!--|^\s*\*|^\s*;)[^\n]*?\b{_COMMENT_WORD}\b"""
            rf"""[^\n:=]*?(?:\bis\b|\badalah\b|:|=)[ \t]*["']?){_NOT_A_VALUE}"""
            r"""([^\s"']{4,})"""
        ),
        r"\1[REDACTED:secret]",
    ),
)


def _apply(rules: tuple[tuple[re.Pattern[str], Replacement], ...], text: str) -> tuple[str, int]:
    counter = [0]
    for pattern, replacement in rules:
        text = pattern.sub(_substituter(replacement, counter), text)
    return text, counter[0]


def _substituter(replacement: Replacement, counter: list[int]) -> Callable[[re.Match[str]], str]:
    def substitute(match: re.Match[str]) -> str:
        out = replacement(match) if callable(replacement) else match.expand(replacement)
        if out is None:  # rule inspected the match and declined to redact
            return match.group(0)
        counter[0] += 1
        missing = match.group(0).count("\n") - out.count("\n")
        return out + "\n" * missing if missing > 0 else out

    return substitute


def _redact_known(text: str) -> tuple[str, int]:
    return _apply(_KNOWN_FORMATS, text)


# Name-based rules (the last regex group is the secret value); they also run on a skeleton
# copy of texts that contain invisible or lookalike characters (see _redact_obfuscated).
_NAME_RULES = _CONTEXTUAL_ALL[2:]
_CONTEXTUAL = _CONTEXTUAL_ALL
_INVISIBLE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u202a-\u202e\u2066-\u2069]")
_MARKER = re.compile(r"\[REDACTED:[a-z_]+\]")


def _skeleton(text: str) -> tuple[str, list[int]]:
    """Text with invisible characters removed and lookalikes mapped to Latin, plus a map
    from each skeleton position to its original index."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if _INVISIBLE.match(ch):
            continue
        chars.append(ch.translate(CONFUSABLES))
        index.append(i)
    return "".join(chars), index


def _redact_obfuscated(text: str) -> tuple[str, int]:
    """Catch credential names hidden with zero-width/bidi characters or homoglyphs.

    Matches are found on the skeleton; only the value span is replaced in the original,
    so displayed evidence keeps its real characters and line count.
    """
    if not (_INVISIBLE.search(text) or text != text.translate(CONFUSABLES)):
        return text, 0
    skeleton, index = _skeleton(text)
    markers = [m.span() for m in _MARKER.finditer(text)]
    spans: list[tuple[int, int]] = []
    for pattern, _ in _NAME_RULES:
        for match in pattern.finditer(skeleton):
            group = match.lastindex
            if not group:
                continue
            start, end = match.span(group)
            if end <= start:
                continue
            o_start, o_end = index[start], index[end - 1] + 1
            if not any(o_start < m_end and o_end > m_start for m_start, m_end in markers):
                spans.append((o_start, o_end))
    if not spans:
        return text, 0
    merged: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    for a, b in reversed(merged):
        text = text[:a] + "[REDACTED:secret]" + "\n" * text[a:b].count("\n") + text[b:]
    return text, len(merged)


def redact(text: str) -> tuple[str, int]:
    """Mask likely credentials. Returns the redacted text and the number of replacements.

    The newline count of ``text`` is preserved.
    """
    text, known = _redact_known(text)
    text, contextual = _apply(_CONTEXTUAL, text)
    text, obfuscated = _redact_obfuscated(text)
    return text, known + contextual + obfuscated


_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]"
)


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

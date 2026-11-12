"""Canonical hashing for configuration, data and files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# \A...\Z (not ^...$): in Python, `$` also matches just before a trailing newline, so
# "<64 hex>\n" passed `^[0-9a-f]{64}$`. Callers also use fullmatch.
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
GIT_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")
PLACEHOLDER = "REPLACE"


def canonical_json(value: Any) -> bytes:
    """Stable JSON encoding: sorted keys, no whitespace, UTF-8, no NaN."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def harness_code_sha256() -> str:
    """Hash of this harness's own source files (``wsp_eval/*.py`` and the ``wsp.py`` CLI),
    recorded in every run manifest."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    paths = sorted(root.glob("*.py")) + [p for p in [root.parent / "wsp.py"] if p.exists()]
    for path in paths:
        h.update(path.relative_to(root.parent).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def contains_placeholder(value: Any) -> list[str]:
    """Return the JSON paths of every string still containing the REPLACE placeholder."""
    found: list[str] = []

    def walk(v: Any, path: str) -> None:
        if isinstance(v, str) and PLACEHOLDER in v:
            found.append(path or "$")
        elif isinstance(v, dict):
            for k, sub in v.items():
                walk(sub, f"{path}.{k}" if path else k)
        elif isinstance(v, list):
            for i, sub in enumerate(v):
                walk(sub, f"{path}[{i}]")

    walk(value, "")
    return found

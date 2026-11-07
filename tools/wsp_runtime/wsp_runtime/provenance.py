"""File provenance for loaded checkpoints and the runtime's own code. Standard library only."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """The git object id of a file's content (``git hash-object``), as used in the source pins."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every regular file under ``root``, keyed by POSIX path relative to ``root``.

    Symbolic links are refused: a checkpoint directory whose files point elsewhere could be
    changed without changing anything this function reads.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory {root} does not exist")
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"{rel} is a symbolic link; copy the real file into the checkpoint")
        if path.is_file():
            out[rel] = sha256_path(path)
    if not out:
        raise ValueError(f"checkpoint directory {root} contains no files")
    return out


def runtime_code_sha256() -> str:
    """Hash of this package's source files and the ``run.py`` CLI, recorded with every load."""
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    paths = sorted(pkg.glob("*.py")) + [p for p in [pkg.parent / "run.py"] if p.exists()]
    for path in paths:
        h.update(path.relative_to(pkg.parent).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()

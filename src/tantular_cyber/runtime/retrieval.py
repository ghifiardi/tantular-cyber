"""Deterministic local lexical retrieval over a prepared Shirudo reference corpus.

* Preparation (``prepare_corpus``) is an explicit operation over a local checkout the user
  already fetched. Nothing here downloads, runs git, or executes upstream files; the
  checkout's revision is read from ``.git/HEAD`` as text when available.
* Retrieval is BM25 over lowercase word tokens with a fixed tie-break, under a strict
  character budget. Retrieved text is reference material about mechanisms and limits; it is
  never evidence that the analyzed host performed an action.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from ..security import sanitize_text
from . import upstream as up
from .models import RuntimeEvidenceReport

CORPUS_VERSION = "1.0"
CHUNK_MAX_CHARS = 1200
DEFAULT_TOP_K = 4
DEFAULT_MAX_CHARS = 3600
BM25_K1, BM25_B = 1.2, 0.75
BUNDLED_CORPUS = "shirudo-reference-corpus.json"

# Documentation only (BSD-2-Clause per SPDX header or the repository LICENSE). No BPF, Lua
# decoder or script source is included.
DEFAULT_REFERENCE_FILES: tuple[tuple[str, str], ...] = (
    ("upatches/kernel-obs/README.md", "BSD-2-Clause (repository LICENSE; no per-file SPDX)"),
    ("upatches/app-sandbox/README.md", "BSD-2-Clause (repository LICENSE; no per-file SPDX)"),
    ("Documentation/reference/formats/map-export.rst", "BSD-2-Clause (SPDX header)"),
    ("Documentation/reference/cli/upatch-map.rst", "BSD-2-Clause (SPDX header)"),
    ("Documentation/security/security-model.rst", "BSD-2-Clause (SPDX header)"),
)
MAX_REFERENCE_FILE_BYTES = 512 * 1024

_TOKEN = re.compile(r"[a-z0-9_]{2,}")
_STOP = frozenset(
    {
        *("the", "and", "for", "with", "that", "this", "are", "not", "from", "its", "into"),
        *("each", "per", "or", "of", "to", "in", "on", "as", "by", "be", "is", "an", "at"),
        *("it", "can", "a", "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada"),
        *("atau", "ini", "itu", "tidak", "bukan"),
    }
)
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


class CorpusError(ValueError):
    pass


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sections(path: str, text: str) -> list[tuple[str, str]]:
    """Split Markdown (#) or reStructuredText (underlined titles) into (section, body)."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = [("(intro)", [])]
    rst = path.endswith(".rst")
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if not rst and re.match(r"^#{1,4}\s+\S", line):
            sections.append((line.lstrip("#").strip(), []))
        elif rst and line.strip() and _RST_UNDERLINE.match(nxt) and len(nxt.strip()) >= 3:
            sections.append((line.strip(), []))
            i += 1
        elif not (rst and _RST_UNDERLINE.match(line)):
            sections[-1][1].append(line)
        i += 1
    return [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def _split(body: str) -> list[str]:
    parts: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        while len(para) > CHUNK_MAX_CHARS:
            parts.append(para[:CHUNK_MAX_CHARS])
            para = para[CHUNK_MAX_CHARS:]
        if current and len(current) + len(para) + 2 > CHUNK_MAX_CHARS:
            parts.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        parts.append(current)
    return parts


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "section"


def read_checkout_revision(checkout: Path) -> str | None:
    """Read a detached HEAD commit from ``.git/HEAD`` as text (never runs git)."""
    head = checkout / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="ascii", errors="strict").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def prepare_corpus(
    checkout: Path,
    *,
    revision: str | None = None,
    files: tuple[tuple[str, str], ...] = DEFAULT_REFERENCE_FILES,
) -> dict[str, Any]:
    """Build a corpus document from a local upstream checkout (explicit, offline)."""
    root = checkout.resolve()
    detected = read_checkout_revision(root)
    if revision is None:
        if detected is None:
            raise CorpusError("cannot read the checkout revision; pass --revision explicitly")
        revision, source = detected, "checkout_head"
    else:
        source = "checkout_head" if detected == revision else "user_asserted"
        if detected is not None and detected != revision:
            raise CorpusError("checkout HEAD does not match --revision")
    documents, chunks = [], []
    for rel, license_note in files:
        rel_path = PurePosixPath(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise CorpusError("reference paths must be relative to the checkout")
        unresolved = root / rel
        path = unresolved.resolve()
        if unresolved.is_symlink() or root not in path.parents or not path.is_file():
            raise CorpusError(f"reference file missing or unsafe: {rel}")
        raw = path.read_bytes()
        if len(raw) > MAX_REFERENCE_FILE_BYTES:
            raise CorpusError(f"reference file too large: {rel}")
        text = raw.decode("utf-8")
        documents.append(
            {
                "path": rel,
                "source_url": f"{up.UPSTREAM_REPOSITORY}/blob/{revision}/{rel}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "license": license_note,
            }
        )
        for section, body in _sections(rel, text):
            for n, part in enumerate(_split(body)):
                clean = sanitize_text(part, CHUNK_MAX_CHARS + 8)
                chunks.append(
                    {
                        "chunk_id": f"shirudo@{revision[:12]}:{rel}#{_slug(section)}-{n}",
                        "path": rel,
                        "section": sanitize_text(section, 160),
                        "text": clean,
                        "sha256": _sha(clean),
                    }
                )
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        dup: Counter[str] = Counter(ids)
        for c in chunks:  # disambiguate repeated section titles deterministically
            if dup[c["chunk_id"]] > 1:
                c["chunk_id"] += f"-{c['sha256'][:8]}"
    corpus: dict[str, Any] = {
        "corpus_version": CORPUS_VERSION,
        "project": up.UPSTREAM_PROJECT,
        "repository": up.UPSTREAM_REPOSITORY,
        "revision": revision,
        "revision_source": source,
        "license_notice": (
            "Excerpts of Shirudo documentation, Copyright (c) 2026 Authors of shirudo, "
            "BSD-2-Clause. See THIRD_PARTY_NOTICES.md for the full license text."
        ),
        "documents": documents,
        "chunks": chunks,
    }
    corpus["corpus_sha256"] = corpus_digest(corpus)
    return corpus


def corpus_digest(corpus: dict[str, Any]) -> str:
    body = {k: v for k, v in corpus.items() if k != "corpus_sha256"}
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":")))


def load_corpus(path: Path | None = None) -> dict[str, Any]:
    """Load a prepared corpus (default: the bundled one, works from a wheel)."""
    if path is None:
        res = resources.files("tantular_cyber") / "runtime" / "data" / BUNDLED_CORPUS
        raw = res.read_text(encoding="utf-8")
    else:
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            raise CorpusError("corpus must be a regular file up to 8 MiB")
        raw = path.read_text(encoding="utf-8")
    corpus = json.loads(raw)
    if not isinstance(corpus, dict) or corpus.get("corpus_version") != CORPUS_VERSION:
        raise CorpusError("unsupported corpus format")
    if corpus.get("corpus_sha256") != corpus_digest(corpus):
        raise CorpusError("corpus digest mismatch")
    for c in corpus["chunks"]:
        if _sha(c["text"]) != c["sha256"]:
            raise CorpusError("chunk digest mismatch")
    return corpus


@dataclass(frozen=True)
class RetrievalParams:
    top_k: int = DEFAULT_TOP_K
    max_chars: int = DEFAULT_MAX_CHARS
    k1: float = BM25_K1
    b: float = BM25_B

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "bm25",
            "top_k": self.top_k,
            "max_chars": self.max_chars,
            "k1": self.k1,
            "b": self.b,
        }


class LexicalIndex:
    def __init__(self, corpus: dict[str, Any]) -> None:
        self.corpus = corpus
        self.chunks: list[dict[str, Any]] = corpus["chunks"]
        self.tf = [Counter(tokens(c["section"] + " " + c["text"])) for c in self.chunks]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: list[str], params: RetrievalParams) -> list[dict[str, Any]]:
        # Repeated query terms weigh more (query-term frequency), e.g. record kinds.
        weights = Counter(t for t in query if t in self.idf)
        terms = list(weights)
        scored = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            norm = params.k1 * (1 - params.b + params.b * self.lengths[i] / (self.avg or 1))
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += weights[t] * self.idf[t] * f * (params.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((-round(score, 9), self.chunks[i]["chunk_id"], i))
        scored.sort()
        out, used = [], 0
        for _, _, i in scored:
            chunk = self.chunks[i]
            if len(out) >= params.top_k or used + len(chunk["text"]) > params.max_chars:
                continue
            used += len(chunk["text"])
            out.append(chunk)
        return out


_REASON_TERMS = {
    "interval_overflow": ["overflow", "interval"],
    "interval_lost": ["lost", "interval", "export"],
    "peek_only": ["peek", "drain"],
    "missing_metadata": ["interval", "metadata"],
    "recognized_unsupported_map": ["socket", "listen", "dns", "watch"],
}


def query_for(report: RuntimeEvidenceReport) -> list[str]:
    """Deterministic query from evidence structure (never from free-text telemetry)."""
    terms: list[str] = ["kernel", "obs", "records", "interval"]
    for name in sorted(report.coverage.maps):
        terms.append(name)
    for rec in report.records[:200]:
        kind = rec.fields.get("kind")
        if isinstance(kind, str):
            terms += [*tokens(kind.replace("_", " ")), kind, kind, kind]
        if rec.map == "net":
            terms += ["peer", "bytes", "packets", "socket"]
        if rec.map == "file":
            terms += ["file", "watch", "list", "opens"]
        if rec.caller_inferred:
            terms += ["inferred", "caller"]
    for reason in report.status_reasons:
        terms += _REASON_TERMS.get(reason.split(":", 1)[0], [])
    terms += ["tpm", "attestation", "limits"]
    specific = Counter(t for t in terms if t not in _STOP)
    # Cap repetition so one busy record cannot dominate; order is first appearance.
    return [t for t in dict.fromkeys(specific) for _ in range(min(specific[t], 3))]


def chunk_for_prompt(chunk: dict[str, Any]) -> dict[str, str]:
    return {
        "chunk_id": chunk["chunk_id"],
        "source": chunk["path"],
        "section": chunk["section"],
        "text": chunk["text"],
    }

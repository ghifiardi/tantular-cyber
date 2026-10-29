"""Real-world, pointer-only evaluation.

The manifest (``benchmarks/realworld/manifest.json``) stores **pointers, not code**:
repository URL, immutable commit, path, line range, a SHA-256 of the target line, the
git blob SHA-1 of the file, the upstream license and the verification status. No
upstream source text is committed.

* ``fetch_pointers`` runs only when explicitly invoked (``tantular-cyber realworld
  fetch``). It downloads the single file per pointer from
  ``raw.githubusercontent.com/<owner>/<repo>/<commit>/<path>``, verifies the git blob
  hash, and stores it in a local cache outside the repository by default. Nothing is
  executed; no code is sent anywhere.
* ``evaluate_pointers`` is offline: it scans cached files (Semgrep + tantular-callgraph,
  no LLM) and reports, per pointer, whether a finding of the labelled CWE overlaps the
  labelled lines. Upstream labels are hypotheses until independently verified, and the
  result is reported separately from the synthetic corpus — never as precision/recall.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from ..config import ScanSettings
from ..pipeline import run_scan, split_lines

DEFAULT_CACHE = Path("~/.cache/tantular-cyber/realworld").expanduser()
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GITHUB = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")
PERMISSIVE = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC"}


class Advisory(BaseModel):
    id: str
    url: str


class Verification(BaseModel):
    status: Literal["upstream_label", "independently_verified", "rejected"]
    independently_verified: bool = False
    verified_by: str | None = Field(default=None, description="Reviewer pseudonym only.")
    verified_at: str | None = None
    notes: str = ""


class Pointer(BaseModel):
    id: str
    cwe: str
    advisory: Advisory
    repository: str
    commit: str
    path: str
    line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    line_sha256: str
    git_blob_sha1: str
    upstream_license: str
    license_source: str
    description: str
    verification: Verification
    coverage_candidate: str | None = Field(
        default=None,
        description="Why the bundled scanners miss this location; a candidate for future "
        "coverage work (not a commitment).",
    )
    development_use: str | None = Field(
        default=None,
        description="Set when this pointer influenced rules/code; it is then not an "
        "independent (held-out) measurement.",
    )

    @property
    def owner_repo(self) -> tuple[str, str]:
        match = _GITHUB.match(self.repository)
        if not match:
            raise ValueError(f"{self.id}: unsupported repository URL")
        return match.group(1), match.group(2)


class RealWorldManifest(BaseModel):
    manifest_version: str
    kind: Literal["pointer_only"]
    policy: str
    entries: list[Pointer]


class PointerResult(BaseModel):
    id: str
    cwe: str
    advisory_id: str
    verification_status: str
    state: Literal["not_fetched", "integrity_failed", "anchor_moved", "detected", "missed"]
    tools: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    detail: str = ""
    held_out: bool = True


class RealWorldResult(BaseModel):
    manifest_version: str
    entries: int
    fetched: int
    evaluated: int
    detected: int
    missed: int
    by_cwe: dict[str, dict[str, int]]
    by_verification: dict[str, int]
    results: list[PointerResult]
    caveat: str = (
        "Pointer-only real-world sample with upstream labels treated as hypotheses. "
        "Detection counts on this small set are descriptive and are not precision/recall "
        "or real-world performance estimates."
    )


def load_realworld(path: Path) -> RealWorldManifest:
    return RealWorldManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_realworld(manifest: RealWorldManifest) -> list[str]:
    problems: list[str] = []
    ids = [e.id for e in manifest.entries]
    if len(ids) != len(set(ids)):
        problems.append("duplicate pointer ids")
    for e in manifest.entries:
        if not _GITHUB.match(e.repository):
            problems.append(f"{e.id}: repository must be https://github.com/<owner>/<repo>")
        if not _HEX40.match(e.commit):
            problems.append(f"{e.id}: commit must be a full 40-hex SHA (immutable)")
        if not _HEX40.match(e.git_blob_sha1) or not _HEX64.match(e.line_sha256):
            problems.append(f"{e.id}: integrity hashes missing or malformed")
        p = PurePosixPath(e.path)
        if p.is_absolute() or ".." in p.parts:
            problems.append(f"{e.id}: path must be relative and stay inside the repository")
        if e.end_line < e.line:
            problems.append(f"{e.id}: end_line < line")
        if not e.advisory.url.startswith("https://"):
            problems.append(f"{e.id}: advisory URL must be https")
        if e.verification.independently_verified and not e.verification.verified_by:
            problems.append(f"{e.id}: independent verification needs a reviewer pseudonym")
        if e.verification.verified_by and "@" in e.verification.verified_by:
            problems.append(f"{e.id}: use a pseudonym, not an e-mail address")
    return problems


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def cache_path(cache: Path, pointer: Pointer) -> Path:
    owner, repo = pointer.owner_repo
    return cache / f"{owner}__{repo}" / pointer.commit / pointer.path


def fetch_pointers(
    manifest: RealWorldManifest, cache: Path, *, transport: httpx.BaseTransport | None = None
) -> list[tuple[str, str]]:
    """Download and verify each pointed-to file. Returns (pointer id, outcome) pairs."""
    outcomes: list[tuple[str, str]] = []
    with httpx.Client(timeout=30, transport=transport, follow_redirects=False) as http:
        for p in manifest.entries:
            target = cache_path(cache, p)
            if target.is_file() and git_blob_sha1(target.read_bytes()) == p.git_blob_sha1:
                outcomes.append((p.id, "cached"))
                continue
            owner, repo = p.owner_repo
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{p.commit}/{p.path}"
            try:
                response = http.get(url)
            except httpx.HTTPError as exc:
                outcomes.append((p.id, f"error: {type(exc).__name__}"))
                continue
            if response.status_code != 200:
                outcomes.append((p.id, f"error: HTTP {response.status_code}"))
                continue
            if git_blob_sha1(response.content) != p.git_blob_sha1:
                outcomes.append((p.id, "integrity_failed: blob hash mismatch (not stored)"))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            outcomes.append((p.id, "fetched"))
    return outcomes


def clean_cache(cache: Path) -> bool:
    if cache.exists():
        shutil.rmtree(cache)
        return True
    return False


def evaluate_pointers(
    manifest: RealWorldManifest, cache: Path, *, semgrep_bin: str = "semgrep"
) -> RealWorldResult:
    results: dict[str, PointerResult] = {}
    groups: dict[tuple[str, str, str], list[Pointer]] = {}
    for p in manifest.entries:
        base = PointerResult(
            id=p.id,
            cwe=p.cwe,
            advisory_id=p.advisory.id,
            verification_status=p.verification.status,
            state="not_fetched",
            held_out=p.development_use is None,
        )
        target = cache_path(cache, p)
        if not target.is_file():
            results[p.id] = base
            continue
        data = target.read_bytes()
        if git_blob_sha1(data) != p.git_blob_sha1:
            results[p.id] = base.model_copy(update={"state": "integrity_failed"})
            continue
        lines = split_lines(data.decode("utf-8", errors="replace"))
        if (
            p.line > len(lines)
            or hashlib.sha256(lines[p.line - 1].strip().encode()).hexdigest() != p.line_sha256
        ):
            results[p.id] = base.model_copy(update={"state": "anchor_moved"})
            continue
        owner, repo = p.owner_repo
        groups.setdefault((owner, repo, p.commit), []).append(p)
        results[p.id] = base

    for (owner, repo, commit), pointers in sorted(groups.items()):
        with tempfile.TemporaryDirectory(prefix="tantular-rw-") as tmp:
            root = Path(tmp) / "repo"
            for p in pointers:
                dest = root / p.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache_path(cache, p), dest)
            report = run_scan(
                ScanSettings(
                    repo=root, output=Path(tmp) / "out", llm_enabled=False, semgrep_bin=semgrep_bin
                )
            ).report
        for p in pointers:
            hits = [
                f
                for f in report.findings
                if f.path == p.path
                and p.cwe in f.cwe
                and f.start_line <= p.end_line
                and f.end_line >= p.line
            ]
            results[p.id] = results[p.id].model_copy(
                update={
                    "state": "detected" if hits else "missed",
                    "tools": sorted({f.tool for f in hits}),
                    "finding_ids": [f.id for f in hits],
                    "detail": f"{owner}/{repo}@{commit[:12]}",
                }
            )

    ordered = [results[p.id] for p in manifest.entries]
    by_cwe: dict[str, Counter[str]] = {}
    for r in ordered:
        by_cwe.setdefault(r.cwe, Counter())[r.state] += 1
    states = Counter(r.state for r in ordered)
    return RealWorldResult(
        manifest_version=manifest.manifest_version,
        entries=len(ordered),
        fetched=len(ordered) - states.get("not_fetched", 0),
        evaluated=states.get("detected", 0) + states.get("missed", 0),
        detected=states.get("detected", 0),
        missed=states.get("missed", 0),
        by_cwe={k: dict(sorted(v.items())) for k, v in sorted(by_cwe.items())},
        by_verification=dict(Counter(r.verification_status for r in ordered)),
        results=ordered,
    )


def license_audit(manifest: RealWorldManifest) -> dict[str, list[str]]:
    """Group pointers by license class (all are pointer-only; nothing is bundled)."""
    audit: dict[str, list[str]] = {"permissive": [], "other_or_unknown": []}
    for e in manifest.entries:
        key = "permissive" if e.upstream_license in PERMISSIVE else "other_or_unknown"
        audit[key].append(f"{e.id} ({e.upstream_license})")
    return audit

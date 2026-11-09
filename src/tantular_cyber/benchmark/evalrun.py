"""Evaluation-run record: per-response provenance and run validity (POC-007 Step 1).

Every model response in a comparison is bracketed by two snapshots of the local Ollama server
(``/api/tags`` for the installed manifest, ``/api/ps`` for the loaded model). A response is
*eligible* only when both snapshots name the same model digest, the loaded model is that
digest, and every required field is present; otherwise it is *excluded* from comparable
results. A run is *complete* only when every expected response is eligible and the code did
not change; otherwise it is *incomplete* (some responses excluded or missing) or *invalid*
(code unknown, changed, or a controlled run on a dirty tree). See docs/poc-007-plan.md §§ 4–7.

This is a client-side control and is recorded as such (LIMITATIONS): matching snapshots are
not proof of which weights generated a response.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..ollama_client import model_matches

SCHEMA_VERSION = "1.0.0"
RESULT_TYPE = "tantular-cyber.evaluation-run"
SPLIT_SCHEMA = "tantular-cyber.evaluation-split"

LIMITATIONS = [
    "Per-response model provenance is a client-side control: /api/tags and /api/ps are read "
    "immediately before and after each /api/chat call. Matching snapshots are not proof, "
    "cryptographic or otherwise, of which weights generated a response.",
    "Equal snapshots cannot detect a model that changed and changed back during a request.",
    "Ollama's server log, the only stronger evidence, is not read (it is outside this tool's "
    "files and may contain prompts).",
    "In exploratory mode a dirty working tree is recorded but its contents are not "
    "fingerprinted; controlled runs require a clean checkout.",
    "Reproducibility is per machine, Ollama version and model digest.",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash of a canonical JSON encoding (sorted keys, no whitespace)."""
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def membership_sha256(ids: Iterable[str]) -> str:
    return sha256_text("\n".join(sorted(ids)))


class CodeState(BaseModel):
    commit: str | None
    dirty: bool | None


class ModelSnapshot(BaseModel):
    model: str
    tags_digest: str | None = None
    quantization: str | None = None
    loaded_digest: str | None = Field(
        default=None, description="Digest of the loaded model in /api/ps; None if not loaded."
    )
    error: str | None = None


def snapshot(http: httpx.Client, model: str) -> ModelSnapshot:
    """Read-only: /api/tags and /api/ps only (never /api/show, which Ollama forwards for
    cloud models)."""
    try:
        tags = http.get("/api/tags")
        tags.raise_for_status()
        ps = http.get("/api/ps")
        ps.raise_for_status()
        entries = tags.json().get("models") or []
        loaded = ps.json().get("models") or []
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return ModelSnapshot(model=model, error=type(exc).__name__)

    def find(items: list[Any]) -> dict[str, Any]:
        return next(
            (m for m in items if isinstance(m, dict) and model_matches(model, m.get("name", ""))),
            {},
        )

    tag, run = find(entries), find(loaded)
    return ModelSnapshot(
        model=model,
        tags_digest=tag.get("digest") or None,
        quantization=(tag.get("details") or {}).get("quantization_level"),
        loaded_digest=run.get("digest") or None,
    )


class SnapshotSource:
    """One HTTP client for all snapshots of a run; same loopback policy as OllamaClient."""

    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(
            base_url=base_url, timeout=10, trust_env=False, transport=transport
        )

    def __call__(self, model: str) -> ModelSnapshot:
        return snapshot(self._http, model)

    def version(self) -> str | None:
        try:
            data = self._http.get("/api/version").json()
        except (httpx.HTTPError, ValueError):
            return None
        return str(data.get("version")) if isinstance(data, dict) else None

    def close(self) -> None:
        self._http.close()


def exclusion_reasons(
    before: ModelSnapshot,
    after: ModelSnapshot,
    request_sha256: str | None,
    response_sha256: str | None,
    error: str | None,
) -> list[str]:
    """Why a response must be excluded from comparable results; empty means eligible."""
    reasons: list[str] = []
    if error:
        reasons.append(f"request failed: {error}")
    for name, s in (("before", before), ("after", after)):
        if s.error:
            reasons.append(f"{name} snapshot failed: {s.error}")
        elif not s.tags_digest:
            reasons.append(f"{name} snapshot: model digest missing from /api/tags")
    if not after.error and not after.loaded_digest:
        reasons.append("after snapshot: model not loaded in /api/ps")
    if not error and not request_sha256:
        reasons.append("request hash missing")
    if not error and not response_sha256:
        reasons.append("response hash missing")
    if reasons:
        return reasons
    if before.tags_digest != after.tags_digest:
        reasons.append("installed model digest changed during the request")
    if before.quantization != after.quantization:
        reasons.append("quantization changed during the request")
    if after.loaded_digest != after.tags_digest:
        reasons.append("loaded model digest differs from the installed manifest")
    if before.loaded_digest and before.loaded_digest != after.loaded_digest:
        reasons.append("loaded model changed during the request")
    return reasons


class ResponseRecord(BaseModel):
    arm: str
    run: int
    finding_id: str
    requested_at: str
    completed_at: str
    request_sha256: str | None
    response_sha256: str | None
    before: ModelSnapshot
    after: ModelSnapshot
    eligibility: Literal["eligible", "excluded"]
    exclusion_reasons: list[str] = Field(default_factory=list)
    error: str | None = None
    prompt_messages: list[dict[str, str]] | None = Field(
        default=None,
        description="Rendered prompt text; stored only for the bundled synthetic corpus and "
        "only when requested. Otherwise only request_sha256 is kept.",
    )


def make_record(
    *,
    arm: str,
    run: int,
    finding_id: str,
    requested_at: str,
    completed_at: str,
    before: ModelSnapshot,
    after: ModelSnapshot,
    request_sha256: str | None,
    response_sha256: str | None,
    error: str | None = None,
    prompt_messages: list[dict[str, str]] | None = None,
) -> ResponseRecord:
    reasons = exclusion_reasons(before, after, request_sha256, response_sha256, error)
    return ResponseRecord(
        arm=arm,
        run=run,
        finding_id=finding_id,
        requested_at=requested_at,
        completed_at=completed_at,
        request_sha256=request_sha256,
        response_sha256=response_sha256,
        before=before,
        after=after,
        eligibility="excluded" if reasons else "eligible",
        exclusion_reasons=reasons,
        error=error,
        prompt_messages=prompt_messages,
    )


class SplitRef(BaseModel):
    side: Literal["development", "heldout"]
    manifest_sha256: str
    development_sha256: str
    heldout_sha256: str
    cases: int


class EvaluationSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal["tantular-cyber.evaluation-split"] = Field(alias="schema")
    schema_version: Literal["1.0.0"]
    corpus_id: str
    development: list[str]
    heldout: list[str]


def load_split(
    path: Path, side: str, corpus_id: str, case_ids: Iterable[str]
) -> tuple[SplitRef, set[str]]:
    """Validate a frozen split file and return its hashes and the case ids on ``side``.

    Refuses a split for another corpus, unknown case ids, duplicates, and any case on both
    sides. Near-duplicate code across sides is a human-review concern, not checked here.
    """
    raw = path.read_bytes()
    split = EvaluationSplit.model_validate(json.loads(raw))
    if split.corpus_id != corpus_id:
        raise ValueError(f"split is for corpus {split.corpus_id!r}, not {corpus_id!r}")
    known = set(case_ids)
    for name, ids in (("development", split.development), ("heldout", split.heldout)):
        if len(ids) != len(set(ids)):
            raise ValueError(f"split {name} lists a case more than once")
        if unknown := sorted(set(ids) - known):
            raise ValueError(f"split {name} names unknown case(s): {', '.join(unknown)}")
    if both := sorted(set(split.development) & set(split.heldout)):
        raise ValueError(f"case(s) on both sides of the split: {', '.join(both)}")
    if side not in ("development", "heldout"):
        raise ValueError("split side must be 'development' or 'heldout'")
    chosen = set(split.development if side == "development" else split.heldout)
    if not chosen:
        raise ValueError(f"split side {side} is empty")
    ref = SplitRef(
        side=side,  # type: ignore[arg-type]
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        development_sha256=membership_sha256(split.development),
        heldout_sha256=membership_sha256(split.heldout),
        cases=len(chosen),
    )
    return ref, chosen


class RunInputs(BaseModel):
    corpus_id: str
    manifest_version: str
    corpus_sha256: str
    finding_ids: list[str]
    evidence_sha256: dict[str, str] = Field(
        description="Per finding: sha256 of the redacted evidence snippet and context sent."
    )
    prompt_sha256: str = Field(
        description="System prompt, every prompt profile and the explanation JSON schema."
    )
    arms: list[dict[str, object]]
    runs: int
    pinned_options: dict[str, float | int]
    split: SplitRef | None = None


class EvaluationRun(BaseModel):
    result_type: Literal["tantular-cyber.evaluation-run"] = RESULT_TYPE
    schema_version: str = SCHEMA_VERSION
    mode: Literal["controlled", "exploratory"]
    status: Literal["complete", "incomplete", "invalid"]
    status_reasons: list[str]
    comparable: bool = Field(
        description="True only for a complete controlled run; only then may aggregates be "
        "compared or used for adoption decisions."
    )
    started_at: str
    finished_at: str
    code_at_start: CodeState
    code_at_end: CodeState
    environment: dict[str, object]
    inputs: RunInputs
    inputs_sha256: str
    expected_responses: int
    counts: dict[str, int]
    prompts_stored: bool
    responses: list[ResponseRecord]
    limitations: list[str] = Field(default_factory=lambda: list(LIMITATIONS))


def run_status(
    mode: str,
    start: CodeState,
    end: CodeState,
    responses: Sequence[ResponseRecord],
    expected: int,
) -> tuple[Literal["complete", "incomplete", "invalid"], list[str]]:
    invalid: list[str] = []
    if start.commit is None:
        invalid.append("code commit unknown at start")
    if start != end:
        invalid.append("code commit or working-tree state changed during the run")
    if mode == "controlled" and (start.dirty is not False or end.dirty is not False):
        invalid.append("controlled run on a working tree that is not clean")
    if invalid:
        return "invalid", invalid
    reasons: list[str] = []
    keys = [(r.arm, r.run, r.finding_id) for r in responses]
    if len(keys) != len(set(keys)):
        reasons.append("duplicate responses for the same arm, run and finding")
    if len(set(keys)) < expected:
        reasons.append(f"{expected - len(set(keys))} expected response(s) missing")
    if excluded := sum(1 for r in responses if r.eligibility == "excluded"):
        reasons.append(f"{excluded} response(s) excluded")
    return ("incomplete" if reasons else "complete"), reasons


def check_prompt_storage(store: bool, corpus: Path, bundled: Path) -> None:
    """Prompt text may be stored only for the bundled synthetic corpus (CLAUDE.md rule 4)."""
    if store and corpus.resolve() != bundled.resolve():
        raise ValueError(
            "prompt text can only be stored for the bundled synthetic corpus; "
            "other corpora keep request hashes only"
        )

"""Validation of evidence-ledger run records (JSON form of the Week 1 YAML template).

A record is *evidence of intent and outcome*, not an enforcement boundary. Validation makes
incomplete or self-approved records visible:

* ``complete`` records need full 40-hex revisions, SHA-256 artifact digests, reviewed
  licenses and no ``REPLACE`` placeholders;
* ``failed`` / ``invalid`` records must say why in ``deviations_and_errors``;
* a non-pending ``human_decision`` needs a named decider and timestamp, and the decider may
  be neither the operator nor the research lead (no sole self-approval); ``accept`` is only
  valid for a ``complete`` record. Unknown keys are rejected.

Recording ``accept`` does not deploy or promote anything; deployment of any derived
artifact needs its own evaluation and an explicit human promotion outside this harness.
"""

from __future__ import annotations

import re
from typing import Any

from .hashing import GIT_SHA_RE, SHA256_RE, contains_placeholder

STATUSES = ("planned", "complete", "failed", "invalid")
DECISIONS = ("pending", "accept", "reject", "request_more_evidence")
PSEUDONYM_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$")

REQUIRED = (
    "run_id",
    "status",
    "purpose",
    "operator",
    "source_log_commit",
    "base_model_revision",
    "edited_model_revision",
    "artifact_sha256",
    "licenses_reviewed",
    "environment",
    "dataset",
    "protocol_version",
    "inference",
    "grader_revision",
    "output_location_and_hash",
    "cost_and_compute",
    "deviations_and_errors",
    "human_decision",
)
OPTIONAL = (
    "research_lead",
    "protocol_sha256",
    "harness_code_sha256",
    "checkpoint_diff",
    "controller",
    "decided_by",
    "decided_at",
    "decision_rationale",
    "notes",
    # Source-pinning records (metadata only): where each pin came from, git blob SHA-1 for
    # files whose SHA-256 is not published (non-LFS files on the Hugging Face Hub),
    # declared licenses, and what is still unresolved or unverified.
    "sources",
    "artifact_git_blob_sha1",
    "licenses",
    "open_questions",
    "verification_limits",
)
_LIST_FIELDS = ("deviations_and_errors", "open_questions", "verification_limits")
_REVISION_FIELDS = ("source_log_commit", "base_model_revision", "edited_model_revision")


def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and "REPLACE" in v


def validate_record(rec: dict[str, Any]) -> list[str]:
    """Return a list of problems; empty means the record is internally consistent."""
    if not isinstance(rec, dict):
        return ["record must be an object"]
    problems = [f"missing field {k!r}" for k in REQUIRED if k not in rec]
    unknown = sorted(set(rec) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        problems.append(f"unknown fields: {unknown}")
    if problems:
        return problems

    status, decision = rec["status"], rec["human_decision"]
    if status not in STATUSES:
        problems.append(f"status must be one of {STATUSES}")
    if decision not in DECISIONS:
        problems.append(f"human_decision must be one of {DECISIONS}")
    for role in ("operator", "research_lead", "decided_by"):
        v = rec.get(role)
        placeholder_ok = role == "operator" and status == "planned" and v == "REPLACE"
        if (
            v is not None
            and not placeholder_ok
            and not (isinstance(v, str) and PSEUDONYM_RE.match(v))
        ):
            problems.append(f"{role} must be a pseudonym matching {PSEUDONYM_RE.pattern}")
    for key in _LIST_FIELDS:
        if key in rec and not isinstance(rec[key], list):
            problems.append(f"{key} must be a list")
    for key, pattern, kind in (
        ("artifact_sha256", SHA256_RE, "SHA-256"),
        ("artifact_git_blob_sha1", GIT_SHA_RE, "40-hex git blob SHA-1"),
    ):
        value = rec.get(key, {})
        if not isinstance(value, dict):
            problems.append(f"{key} must map file path to digest")
            continue
        bad = sorted(p for p, d in value.items() if not (isinstance(d, str) and pattern.match(d)))
        if bad:
            problems.append(f"{key} digests are not {kind}: {bad}")
    # Pinned values are checked at every status: a mistyped revision in a planned record
    # must not wait until completion to be noticed.
    for key in _REVISION_FIELDS:
        v = rec[key]
        if not _is_placeholder(v) and not (isinstance(v, str) and GIT_SHA_RE.match(v)):
            problems.append(f"{key} must be a full 40-hex revision")

    if status == "complete":
        holes = contains_placeholder(rec)
        if holes:
            problems.append(f"complete record still has placeholders at: {holes}")
        if not rec["artifact_sha256"]:
            problems.append("complete record needs artifact_sha256 entries")
        if rec["licenses_reviewed"] is not True:
            problems.append("complete record requires licenses_reviewed: true")
    if status in ("failed", "invalid") and not rec["deviations_and_errors"]:
        problems.append(f"a {status} record must explain itself in deviations_and_errors")

    if decision != "pending":
        decider = rec.get("decided_by")
        if not decider:
            problems.append("a recorded decision needs decided_by")
        elif decider in (rec.get("operator"), rec.get("research_lead")):
            problems.append("decided_by must differ from operator and research_lead")
        if not (isinstance(rec.get("decided_at"), str) and _TIMESTAMP_RE.match(rec["decided_at"])):
            problems.append("a recorded decision needs decided_at (UTC, e.g. 2026-11-13T10:00Z)")
        if decision == "accept" and status != "complete":
            problems.append("accept is only valid for a complete record")
    return problems

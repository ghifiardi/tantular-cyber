"""Human decision records for a study report. Recording a decision deploys nothing.

Decisions: ``pending``, ``reject``, ``request_more_evidence`` and
``approve_further_evaluation``. The last means only "worth a separate, approved evaluation"
(for a weight change: its own protocol, security adjudication and release process). There is
deliberately no "deploy" or "promote" value, and ``deployment_authorized`` must be ``false``.

Rules: the decider is a pseudonym and is neither the study owner nor the operator (no sole
self-approval); the record names the exact study and report hashes; approval needs a candidate
whose outcome is ``meets_declared_criteria_pending_human_decision``.

What a record is *not*: it is not authenticated approval (``identity_verified`` is always
``false``; anyone who can write the file can type any pseudonym), and it is not enforced
anywhere (``enforced_by`` is always ``"none"``; no system reads it to gate a release). It is
evidence of a stated human judgement, to be stored where the research owner controls writes.
"""

from __future__ import annotations

from typing import Any

from wsp_eval.hashing import sha256_json

from .study import PSEUDONYM_RE, TIMESTAMP_RE, Study

DECISIONS = ("pending", "reject", "request_more_evidence", "approve_further_evaluation")
FIELDS = {
    "study_sha256",
    "report_sha256",
    "candidate",
    "decision",
    "decided_by",
    "decided_at",
    "rationale",
    "deployment_authorized",
    "identity_verified",
    "enforced_by",
}


def validate_decision(rec: dict[str, Any], study: Study, report: dict[str, Any]) -> list[str]:
    problems = []
    if set(rec) != FIELDS:
        return [f"decision record must have exactly {sorted(FIELDS)}"]
    if rec["study_sha256"] != study.sha256 or report.get("study_sha256") != study.sha256:
        problems.append("decision does not reference this study")
    body = {k: v for k, v in report.items() if k != "report_sha256"}
    if report.get("report_sha256") != sha256_json(body):
        # An unkeyed self-hash: it catches accidental edits, but anyone who edits a report can
        # recompute it, so it is not tamper-evidence, and outcomes are not re-derived here.
        problems.append("report content does not match its report_sha256 (edited after build)")
    if rec["report_sha256"] != report.get("report_sha256"):
        problems.append("decision does not reference this report")
    if rec["decision"] not in DECISIONS:
        problems.append(f"decision must be one of {DECISIONS}")
    if rec["deployment_authorized"] is not False:
        problems.append("deployment_authorized must be false: this workflow never deploys")
    if rec["identity_verified"] is not False:
        problems.append("identity_verified must be false: pseudonyms are not authenticated")
    if rec["enforced_by"] != "none":
        problems.append("enforced_by must be 'none': no system enforces this record")
    who = rec["decided_by"]
    if not (isinstance(who, str) and PSEUDONYM_RE.fullmatch(who)):
        problems.append("decided_by must be a pseudonym")
    elif who in (study.raw["owner"], study.raw["operator"]):
        problems.append("decided_by must differ from the study owner and operator")
    if not (isinstance(rec["decided_at"], str) and TIMESTAMP_RE.fullmatch(rec["decided_at"])):
        problems.append("decided_at must be a UTC timestamp")
    if not (isinstance(rec["rationale"], str) and rec["rationale"].strip()):
        problems.append("a decision needs a rationale")
    outcomes = {r["candidate"]: r["outcome"] for r in report.get("candidates", [])}
    cand = rec["candidate"]
    if cand is not None and cand not in outcomes:
        problems.append(f"unknown candidate {cand!r}")
    if rec["decision"] == "approve_further_evaluation":
        if cand is None:
            problems.append("approval names one candidate")
        elif outcomes.get(cand) != "meets_declared_criteria_pending_human_decision":
            problems.append(
                f"cannot approve {cand!r}: its outcome is {outcomes.get(cand)!r}, not "
                "meets_declared_criteria_pending_human_decision"
            )
    return problems


def make_decision(
    study: Study,
    report: dict[str, Any],
    *,
    candidate: str | None,
    decision: str,
    decided_by: str,
    decided_at: str,
    rationale: str,
) -> dict[str, Any]:
    rec = {
        "study_sha256": study.sha256,
        "report_sha256": report.get("report_sha256"),
        "candidate": candidate,
        "decision": decision,
        "decided_by": decided_by,
        "decided_at": decided_at,
        "rationale": rationale,
        "deployment_authorized": False,
        "identity_verified": False,
        "enforced_by": "none",
    }
    problems = validate_decision(rec, study, report)
    if problems:
        raise ValueError("; ".join(problems))
    return rec

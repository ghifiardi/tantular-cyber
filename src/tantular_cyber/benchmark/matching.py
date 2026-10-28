"""Deterministic matching of scanner findings to ground truth, and detection metrics.

Matching rules (documented in docs/benchmark.md; no fuzzy or LLM-based matching):

1. A finding *can* match an annotation when all hold:
   a. same repo-relative file path;
   b. the annotation's CWE is in the finding's CWE list;
   c. the line ranges intersect: finding.start <= annotation.end_line and
      finding.end >= annotation.line.
2. Candidate pairs are sorted by (|finding.start - annotation.end_line|, annotation id,
   finding id) and assigned greedily one-to-one.
3. Outcomes per annotation: vulnerable+matched = TP, vulnerable+unmatched = FN,
   safe+matched = FP (annotated), safe+unmatched = TN.
4. Each finding left unmatched is an FP: ``duplicate`` when it would have matched an
   already-matched annotation, otherwise ``unannotated``.
5. Expectation check (regression gate): matched must equal ``expect_detected``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from ..models import Finding
from .manifest import Manifest

Outcome = Literal["TP", "FP", "FN", "TN"]


class MatchRecord(BaseModel):
    annotation_id: str
    case_id: str
    path: str
    line: int
    end_line: int
    cwe: str
    rule_id: str | None
    truth: str
    label: str
    outcome: Outcome
    finding_id: str | None
    expect_detected: bool
    detected: bool
    expectation_met: bool


class UnmatchedFinding(BaseModel):
    finding_id: str
    path: str
    start_line: int
    end_line: int
    rule_id: str
    cwe: list[str]
    reason: Literal["duplicate", "unannotated"]


class Metrics(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None


class MatchResult(BaseModel):
    records: list[MatchRecord]
    unmatched_findings: list[UnmatchedFinding]
    metrics: Metrics
    by_rule: dict[str, Metrics]
    by_cwe: dict[str, Metrics]


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Metrics:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return Metrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=_ratio(tn, tn + fp),
    )


def _intersects(f: Finding, start: int, end: int) -> bool:
    return f.start_line <= end and f.end_line >= start


def match_findings(manifest: Manifest, findings: Sequence[Finding]) -> MatchResult:
    pairs: list[tuple[int, str, str]] = []
    for case, a in manifest.annotations():
        for f in findings:
            if f.path == case.path and a.cwe in f.cwe and _intersects(f, a.line, a.end_line):
                pairs.append((abs(f.start_line - a.end_line), a.id, f.id))
    pairs.sort()

    ann_to_finding: dict[str, str] = {}
    used: set[str] = set()
    overlapped: set[str] = set()
    for _, aid, fid in pairs:
        overlapped.add(fid)
        if aid in ann_to_finding or fid in used:
            continue
        ann_to_finding[aid] = fid
        used.add(fid)

    records: list[MatchRecord] = []
    for case, a in manifest.annotations():
        fid = ann_to_finding.get(a.id)
        detected = fid is not None
        outcome: Outcome = (
            ("TP" if detected else "FN")
            if a.truth == "vulnerable"
            else ("FP" if detected else "TN")
        )
        records.append(
            MatchRecord(
                annotation_id=a.id,
                case_id=case.id,
                path=case.path,
                line=a.line,
                end_line=a.end_line,
                cwe=a.cwe,
                rule_id=a.rule_id,
                truth=a.truth,
                label=a.label,
                outcome=outcome,
                finding_id=fid,
                expect_detected=a.expect_detected,
                detected=detected,
                expectation_met=detected == a.expect_detected,
            )
        )

    unmatched = [
        UnmatchedFinding(
            finding_id=f.id,
            path=f.path,
            start_line=f.start_line,
            end_line=f.end_line,
            rule_id=f.rule_id,
            cwe=f.cwe,
            reason="duplicate" if f.id in overlapped else "unannotated",
        )
        for f in sorted(findings, key=lambda f: (f.path, f.start_line, f.id))
        if f.id not in used
    ]

    def tally(recs: list[MatchRecord], extra_fp: int) -> Metrics:
        c = defaultdict(int)
        for r in recs:
            c[r.outcome] += 1
        return compute_metrics(c["TP"], c["FP"] + extra_fp, c["FN"], c["TN"])

    groups_rule: dict[str, list[MatchRecord]] = defaultdict(list)
    groups_cwe: dict[str, list[MatchRecord]] = defaultdict(list)
    for r in records:
        groups_rule[r.rule_id or "unspecified"].append(r)
        groups_cwe[r.cwe].append(r)
    extra_rule: dict[str, int] = defaultdict(int)
    extra_cwe: dict[str, int] = defaultdict(int)
    for u in unmatched:
        extra_rule[u.rule_id] += 1
        for cwe in u.cwe or ["unmapped"]:
            extra_cwe[cwe] += 1

    return MatchResult(
        records=records,
        unmatched_findings=unmatched,
        metrics=tally(records, len(unmatched)),
        by_rule={
            k: tally(groups_rule.get(k, []), extra_rule.get(k, 0))
            for k in sorted({*groups_rule, *extra_rule})
        },
        by_cwe={
            k: tally(groups_cwe.get(k, []), extra_cwe.get(k, 0))
            for k in sorted({*groups_cwe, *extra_cwe})
        },
    )

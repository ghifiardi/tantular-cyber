from __future__ import annotations

from tantular_cyber.benchmark.manifest import Annotation, Case, Manifest, Thresholds
from tantular_cyber.benchmark.matching import compute_metrics, match_findings
from tantular_cyber.models import Evidence, Finding


def ann(id_: str, truth: str, line: int, end: int, expect: bool, cwe: str = "CWE-89") -> Annotation:
    return Annotation(
        id=id_,
        truth=truth,
        cwe=cwe,
        line=line,
        end_line=end,
        anchor="x",
        expect_detected=expect,
        preconditions=["p"] if truth == "vulnerable" else [],
    )


def finding(
    fid: str, path: str, start: int, end: int | None = None, cwe: tuple[str, ...] = ("CWE-89",)
) -> Finding:
    end = end or start
    return Finding(
        id=fid,
        tool="semgrep",
        rule_id="r",
        cwe=list(cwe),
        severity="high",
        path=path,
        start_line=start,
        end_line=end,
        message="m",
        evidence=Evidence(start_line=start, end_line=end, lines=[], snippet=""),
    )


def manifest(*annotations: Annotation, path: str = "a.py") -> Manifest:
    return Manifest(
        manifest_version="1.0.0",
        corpus_id="t",
        description="",
        language="python",
        thresholds=Thresholds(min_precision=0, min_recall=0),
        cases=[Case(id="c", path=path, description="", annotations=list(annotations))],
    )


def test_outcomes_tp_fp_fn_tn() -> None:
    m = manifest(
        ann("tp", "vulnerable", 9, 10, True),
        ann("fn", "vulnerable", 20, 20, False),
        ann("fp", "safe", 30, 30, True),
        ann("tn", "safe", 40, 40, False),
    )
    res = match_findings(m, [finding("f1", "a.py", 10), finding("f2", "a.py", 30)])
    out = {r.annotation_id: r.outcome for r in res.records}
    assert out == {"tp": "TP", "fn": "FN", "fp": "FP", "tn": "TN"}
    assert all(r.expectation_met for r in res.records)
    assert (res.metrics.tp, res.metrics.fp, res.metrics.fn, res.metrics.tn) == (1, 1, 1, 1)
    assert res.metrics.precision == 0.5 and res.metrics.recall == 0.5


def test_range_intersection_path_and_cwe_rules() -> None:
    m = manifest(ann("a", "vulnerable", 9, 10, True))
    assert match_findings(m, [finding("f", "a.py", 8, 9)]).records[0].outcome == "TP"
    assert match_findings(m, [finding("f", "a.py", 11)]).records[0].outcome == "FN"
    assert match_findings(m, [finding("f", "b.py", 10)]).records[0].outcome == "FN"
    assert match_findings(m, [finding("f", "a.py", 10, cwe=("CWE-79",))]).records[0].outcome == "FN"


def test_duplicates_and_unannotated_findings_count_as_fp() -> None:
    # "near" starts on the sink line (distance 0); "dup" starts one line earlier.
    m = manifest(ann("a", "vulnerable", 10, 10, True))
    res = match_findings(
        m,
        [finding("near", "a.py", 10), finding("dup", "a.py", 9, 11), finding("stray", "a.py", 50)],
    )
    assert res.records[0].finding_id == "near"
    reasons = {u.finding_id: u.reason for u in res.unmatched_findings}
    assert reasons == {"dup": "duplicate", "stray": "unannotated"}
    assert res.metrics.fp == 2


def test_greedy_assignment_prefers_closest_sink() -> None:
    m = manifest(ann("wide", "vulnerable", 1, 20, True), ann("narrow", "vulnerable", 15, 15, True))
    res = match_findings(m, [finding("f15", "a.py", 15), finding("f20", "a.py", 20)])
    got = {r.annotation_id: r.finding_id for r in res.records}
    assert got == {"narrow": "f15", "wide": "f20"}


def test_expectation_mismatch_is_reported() -> None:
    m = manifest(ann("lost", "vulnerable", 10, 10, True), ann("new", "safe", 30, 30, False))
    res = match_findings(m, [finding("f", "a.py", 30)])
    met = {r.annotation_id: r.expectation_met for r in res.records}
    assert met == {"lost": False, "new": False}


def test_metric_edge_cases() -> None:
    m = compute_metrics(0, 0, 0, 3)
    assert m.precision is None and m.recall is None and m.f1 is None and m.specificity == 1.0
    assert compute_metrics(3, 1, 1, 0).f1 == 0.75

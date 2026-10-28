from __future__ import annotations

import pytest
from conftest import GOOD_EXPLANATION, FakeLLM

from tantular_cyber.benchmark.perf import (
    concurrency_experiment,
    latency_stats,
    percentile,
    run_perf,
)
from tantular_cyber.models import Evidence, Finding, FindingAnalysis, LLMMetrics


def test_nearest_rank_percentiles() -> None:
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 50) is None
    stats = latency_stats([3.0, 1.0, 2.0])
    assert (stats.n, stats.p50_s, stats.min_s, stats.max_s, stats.mean_s) == (3, 2.0, 1.0, 3.0, 2.0)


def make_finding(i: int, wall: float, load: float) -> Finding:
    return Finding(
        id=f"f{i}",
        tool="t",
        rule_id="r",
        cwe=["CWE-89"],
        severity="high",
        path="a.py",
        start_line=i,
        end_line=i,
        message="m",
        evidence=Evidence(start_line=i, end_line=i, lines=[], snippet=f"{i} | x"),
        analysis=FindingAnalysis(
            status="completed",
            metrics=LLMMetrics(
                wall_s=wall,
                load_s=load,
                prompt_tokens=600,
                output_tokens=400,
                output_tokens_per_s=15.0,
            ),
        ),
    )


def test_run_perf_summarizes_metrics() -> None:
    findings = [make_finding(1, 40.0, 8.0), make_finding(2, 30.0, 0.0), make_finding(3, 32.0, 0.0)]
    perf = run_perf(1, "cold_after_eviction", findings, 105.0, 3.0)
    assert perf.first_call_load_s == 8.0 and perf.findings_explained == 3
    assert perf.per_finding.p50_s == 32.0 and perf.per_finding.p95_s == 40.0
    assert (perf.prompt_tokens, perf.output_tokens) == (1800, 1200)


def test_concurrency_is_bounded_and_counts_errors() -> None:
    findings = [make_finding(i, 1.0, 0.0) for i in range(1, 4)]
    points = concurrency_experiment(lambda: FakeLLM([GOOD_EXPLANATION]), findings, [1, 2])
    assert [(p.concurrency, p.requests, p.errors) for p in points] == [(1, 3, 0), (2, 3, 0)]
    failing = concurrency_experiment(lambda: FakeLLM([RuntimeError("down")]), findings, [1])
    assert failing[0].errors == 3
    with pytest.raises(ValueError):
        concurrency_experiment(lambda: FakeLLM([GOOD_EXPLANATION]), findings, [8])

"""Benchmark result model, deterministic gates, and comparison against a prior run."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from . import BENCHMARK_RESULT_TYPE, BENCHMARK_SCHEMA_VERSION
from .adversarial import SuiteResult
from .llm_eval import ExplanationEval, LLMEvalSummary
from .manifest import Thresholds
from .matching import MatchRecord, Metrics, UnmatchedFinding
from .perf import PerfSummary
from .provenance import Provenance

LIMITATIONS = [
    "The corpus is small and synthetic (Python, CWE-89 only). Precision/recall describe the "
    "bundled rule on this corpus and are regression floors, not real-world performance.",
    "Ground-truth matching is line-range based (see docs/benchmark.md); it does not judge "
    "whether a finding's message is correct.",
    "LLM metrics are sample-based (small n, sampling noise) and come from deterministic "
    "proxies; they do not establish semantic correctness. Subjective criteria require the "
    "human-review queue. LLM metrics never gate CI.",
    "Adversarial suites are regression tests for known evasions; passing them is not proof "
    "of comprehensive protection. Known blind spots are listed explicitly.",
    "Performance numbers depend on hardware, model quantization, and server settings. "
    "'cold_after_eviction' is not a cold boot.",
]


class Gate(BaseModel):
    id: str
    passed: bool
    detail: str


class ScannerBench(BaseModel):
    metrics: Metrics
    by_rule: dict[str, Metrics]
    by_cwe: dict[str, Metrics]
    records: list[MatchRecord]
    unmatched_findings: list[UnmatchedFinding]
    expectation_mismatches: list[str]
    scan_duration_s: float
    scanner_status: str
    coverage: dict[str, object] | None


class LLMBench(BaseModel):
    mode: Literal["ollama", "replay"]
    runs: int
    summary: LLMEvalSummary
    evaluations: list[ExplanationEval]
    injection_cases_detected: dict[str, bool]
    review_queue_items: int


class MetricDelta(BaseModel):
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    worse: bool


class Comparison(BaseModel):
    baseline_started_at: str | None
    baseline_commit: str | None
    baseline_corpus_sha256: str | None
    same_corpus: bool
    deltas: list[MetricDelta]


class BenchmarkResult(BaseModel):
    benchmark_schema_version: str = BENCHMARK_SCHEMA_VERSION
    result_type: str = BENCHMARK_RESULT_TYPE
    status: Literal["pass", "fail"]
    gates: list[Gate]
    thresholds: Thresholds
    provenance: Provenance
    manifest_problems: list[str]
    scanner: ScannerBench | None
    adversarial: list[SuiteResult]
    llm: LLMBench | None = None
    performance: PerfSummary | None = None
    comparison: Comparison | None = None
    limitations: list[str] = LIMITATIONS


def compute_gates(
    thresholds: Thresholds,
    manifest_problems: list[str],
    scanner: ScannerBench | None,
    adversarial: list[SuiteResult],
) -> list[Gate]:
    """Deterministic gates only. LLM and performance results never gate."""
    gates = [
        Gate(
            id="manifest_valid",
            passed=not manifest_problems,
            detail="; ".join(manifest_problems[:5]),
        )
    ]
    if scanner is None:
        gates.append(Gate(id="scanner_ran", passed=False, detail="scanner stage did not run"))
        return gates
    m = scanner.metrics
    gates += [
        Gate(
            id="scanner_complete",
            passed=scanner.scanner_status == "complete",
            detail=f"scan status {scanner.scanner_status}",
        ),
        Gate(
            id="min_precision",
            passed=(m.precision or 0) >= thresholds.min_precision,
            detail=f"{m.precision} >= {thresholds.min_precision}",
        ),
        Gate(
            id="min_recall",
            passed=(m.recall or 0) >= thresholds.min_recall,
            detail=f"{m.recall} >= {thresholds.min_recall}",
        ),
        Gate(
            id="unannotated_findings",
            passed=sum(u.reason == "unannotated" for u in scanner.unmatched_findings)
            <= thresholds.max_unannotated_findings,
            detail=f"{len(scanner.unmatched_findings)} unmatched finding(s)",
        ),
    ]
    if thresholds.require_expected_detections:
        gates.append(
            Gate(
                id="expected_detections",
                passed=not scanner.expectation_mismatches,
                detail="; ".join(scanner.expectation_mismatches[:5]),
            )
        )
    for suite in adversarial:
        gates.append(
            Gate(
                id=f"adversarial_{suite.suite}",
                passed=suite.unexpected_failures == 0,
                detail=", ".join(f"{k}={v}" for k, v in suite.counts.items()),
            )
        )
    return gates


_HIGHER_IS_BETTER = {"precision", "recall", "f1", "specificity", "tp", "tn"}


def _flatten(result: dict[str, object]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    scanner = result.get("scanner") or {}
    for k, v in (scanner.get("metrics") or {}).items():  # type: ignore[union-attr]
        out[f"scanner.{k}"] = v
    for suite in result.get("adversarial") or []:  # type: ignore[union-attr]
        counts = suite.get("counts", {})
        out[f"adversarial.{suite['suite']}.passed"] = counts.get("passed", 0)
        out[f"adversarial.{suite['suite']}.unexpected_failure"] = counts.get(
            "unexpected_failure", 0
        )
        out[f"adversarial.{suite['suite']}.known_gap"] = counts.get("known_gap", 0)
    llm = result.get("llm") or {}
    for cid, rs in ((llm.get("summary") or {}).get("checks") or {}).items():  # type: ignore[union-attr]
        out[f"llm.{cid}.rate"] = rs.get("rate")
    perf = result.get("performance") or {}
    warm = perf.get("warm_per_finding") or {}  # type: ignore[union-attr]
    for k in ("p50_s", "p95_s"):
        out[f"perf.warm.{k}"] = warm.get(k)
    return out


def compare(current: BenchmarkResult, baseline: dict[str, object]) -> Comparison:
    cur = _flatten(current.model_dump())
    base = _flatten(baseline)
    deltas = []
    for key in sorted(set(cur) | set(base)):
        b, c = base.get(key), cur.get(key)
        delta = (
            round(c - b, 4) if isinstance(b, int | float) and isinstance(c, int | float) else None
        )
        leaf = key.rsplit(".", 1)[-1]
        if delta is None:
            worse = False
        elif key.startswith("perf.") or leaf in ("unexpected_failure", "known_gap", "fp", "fn"):
            worse = delta > 0
        elif leaf in _HIGHER_IS_BETTER or leaf in ("rate", "passed"):
            worse = delta < 0
        else:
            worse = False
        deltas.append(MetricDelta(metric=key, baseline=b, current=c, delta=delta, worse=worse))
    prov = baseline.get("provenance") or {}
    return Comparison(
        baseline_started_at=prov.get("started_at"),  # type: ignore[union-attr]
        baseline_commit=prov.get("code_commit"),  # type: ignore[union-attr]
        baseline_corpus_sha256=prov.get("corpus_sha256"),  # type: ignore[union-attr]
        same_corpus=prov.get("corpus_sha256") == current.provenance.corpus_sha256,  # type: ignore[union-attr]
        deltas=deltas,
    )

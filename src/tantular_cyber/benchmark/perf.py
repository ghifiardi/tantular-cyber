"""Latency/throughput statistics and a bounded concurrency experiment.

Labels are deliberately conservative: a run after ``keep_alive: 0`` eviction is
``cold_after_eviction`` — the OS file cache may still hold the weights, so it is NOT a
cold boot. Percentiles use the nearest-rank method on per-finding wall times.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import httpx
from pydantic import BaseModel

from ..analysis import EXPLANATION_JSON_SCHEMA, LLMClient, build_messages
from ..models import Finding

MAX_CONCURRENCY = 4
RunKind = Literal["cold_after_eviction", "warm"]
COLD_NOTE = (
    "cold_after_eviction: the model was evicted with keep_alive=0 before the run; the OS "
    "page cache may still hold the weights, so this is not a cold boot."
)


class LatencyStats(BaseModel):
    n: int
    p50_s: float | None
    p95_s: float | None
    mean_s: float | None
    min_s: float | None
    max_s: float | None


class RunPerf(BaseModel):
    run: int
    kind: RunKind
    findings_explained: int
    scan_total_s: float
    scanner_s: float
    per_finding: LatencyStats
    first_call_load_s: float | None
    prompt_tokens: int
    output_tokens: int
    mean_output_tokens_per_s: float | None


class ConcurrencyPoint(BaseModel):
    concurrency: int
    requests: int
    errors: int
    wall_s: float
    throughput_per_min: float
    latency: LatencyStats


class PerfSummary(BaseModel):
    runs: list[RunPerf]
    warm_per_finding: LatencyStats | None
    concurrency: list[ConcurrencyPoint]
    notes: list[str]


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return round(ordered[rank - 1], 3)


def latency_stats(values: Sequence[float]) -> LatencyStats:
    if not values:
        return LatencyStats(n=0, p50_s=None, p95_s=None, mean_s=None, min_s=None, max_s=None)
    return LatencyStats(
        n=len(values),
        p50_s=percentile(values, 50),
        p95_s=percentile(values, 95),
        mean_s=round(sum(values) / len(values), 3),
        min_s=round(min(values), 3),
        max_s=round(max(values), 3),
    )


def run_perf(
    run: int, kind: RunKind, findings: Sequence[Finding], scan_total_s: float, scanner_s: float
) -> RunPerf:
    metrics = [f.analysis.metrics for f in findings if f.analysis.metrics is not None]
    rates = [m.output_tokens_per_s for m in metrics if m.output_tokens_per_s]
    return RunPerf(
        run=run,
        kind=kind,
        findings_explained=len(metrics),
        scan_total_s=round(scan_total_s, 3),
        scanner_s=round(scanner_s, 3),
        per_finding=latency_stats([m.wall_s for m in metrics]),
        first_call_load_s=metrics[0].load_s if metrics else None,
        prompt_tokens=sum(m.prompt_tokens or 0 for m in metrics),
        output_tokens=sum(m.output_tokens or 0 for m in metrics),
        mean_output_tokens_per_s=round(sum(rates) / len(rates), 2) if rates else None,
    )


def evict_model(base_url: str, model: str, *, timeout_s: float = 60.0) -> bool:
    """Ask a local Ollama to unload ``model``. Returns True once it is no longer loaded."""
    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as http:
        http.post("/api/generate", json={"model": model, "keep_alive": 0})
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            loaded = [m.get("name") for m in http.get("/api/ps").json().get("models", [])]
            if model not in loaded:
                return True
            time.sleep(1)
    return False


def concurrency_experiment(
    client_factory: Callable[[], LLMClient],
    findings: Sequence[Finding],
    levels: Sequence[int],
) -> list[ConcurrencyPoint]:
    """Send the same explanation requests at bounded concurrency levels (1..4).

    Effective parallelism also depends on the server (e.g. OLLAMA_NUM_PARALLEL); the client
    cannot observe that setting, so results are labelled experimental.
    """
    messages = [build_messages(f) for f in findings]
    points = []
    for level in levels:
        if not 1 <= level <= MAX_CONCURRENCY:
            raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")

        def one(msgs: list[dict[str, str]]) -> float | None:
            client = client_factory()
            started = time.monotonic()
            try:
                client.chat(msgs, EXPLANATION_JSON_SCHEMA)
            except Exception:
                return None
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            return time.monotonic() - started

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=level) as pool:
            results = list(pool.map(one, messages))
        wall = time.monotonic() - started
        ok = [r for r in results if r is not None]
        points.append(
            ConcurrencyPoint(
                concurrency=level,
                requests=len(results),
                errors=len(results) - len(ok),
                wall_s=round(wall, 3),
                throughput_per_min=round(len(ok) / wall * 60, 2) if wall else 0.0,
                latency=latency_stats(ok),
            )
        )
    return points

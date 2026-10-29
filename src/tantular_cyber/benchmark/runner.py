"""Benchmark orchestration: scanner → adversarial → (optional) LLM + performance → outputs.

The analyzed corpus is never modified: scans write to a temporary directory, and the LLM
subset is a temporary copy of the selected case files. Case files are never executed.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..analysis import LLMClient
from ..config import OllamaSettings, ScanSettings
from ..models import Finding, ScanReport
from ..ollama_client import OllamaClient
from ..pipeline import run_scan
from ..reports import render_json, render_markdown
from . import render
from .adversarial import run_adversarial
from .context_eval import evaluate_context
from .llm_eval import (
    ExplanationEval,
    evaluate_run,
    injection_scanner_check,
    review_queue,
    summarize,
)
from .manifest import Manifest, corpus_digest, load_manifest, validate_manifest
from .matching import MatchRecord, match_findings
from .perf import (
    COLD_NOTE,
    PerfSummary,
    RunKind,
    RunPerf,
    concurrency_experiment,
    evict_model,
    latency_stats,
    run_perf,
)
from .provenance import Provenance, code_state, machine, model_state, now
from .results import (
    BenchmarkResult,
    Gate,
    LLMBench,
    ScannerBench,
    compare,
    compute_gates,
)

log = logging.getLogger(__name__)

LLMMode = Literal["off", "ollama", "replay"]


@dataclass
class BenchmarkOptions:
    corpus: Path
    output: Path
    llm: LLMMode = "off"
    replay_file: Path | None = None
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    llm_runs: int = 1
    cold: bool = False
    concurrency: tuple[int, ...] = ()
    baseline: Path | None = None
    semgrep_bin: str = "semgrep"
    record_replay: Path | None = None
    real_world_manifest: Path | None = None
    real_world_cache: Path | None = None
    context_strategy: Literal["ast", "window"] = "ast"


class ReplayLLM:
    """Deterministic LLM double: returns recorded outputs keyed by ``path:start_line``.

    Lets CI exercise the full LLM-evaluation path without a model. Keys are taken from the
    ``file`` and ``lines`` fields of the prompt's untrusted-data JSON.
    """

    provider = "replay"

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.model = str(data.get("model", "replay"))
        self.responses: dict[str, Any] = data["responses"]

    def chat(self, messages: list[dict[str, str]], format_schema: Any = None) -> str:
        key = _replay_key(messages)
        response = self.responses.get(key, self.responses.get("*"))
        if response is None:
            return "no recorded response"
        return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)


def _replay_key(messages: list[dict[str, str]]) -> str:
    user = messages[-1]["content"]
    blob = user.split("<untrusted_data>\n", 1)[1].rsplit("\n</untrusted_data>", 1)[0]
    data = json.loads(blob.replace("<\\/untrusted_data>", "</untrusted_data>"))
    return f"{data['file']}:{data['lines'].split('-')[0]}"


class RecordingLLM:
    """Wraps a client and keeps raw model outputs (pre-validation) for replay files.

    Only model *outputs* are stored; prompts are never written.
    """

    def __init__(self, inner: LLMClient, store: dict[str, str]) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.store = store

    @property
    def last_metrics(self) -> Any:
        return getattr(self.inner, "last_metrics", None)

    def health_check(self) -> Any:
        check = getattr(self.inner, "health_check", None)
        return check() if callable(check) else None

    def chat(self, messages: list[dict[str, str]], format_schema: Any = None) -> str:
        raw = self.inner.chat(messages, format_schema)
        self.store.setdefault(_replay_key(messages), raw)
        return raw

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def _leaked_canaries(manifest: Manifest, texts: list[str]) -> list[str]:
    canaries = {c for case in manifest.cases for c in case.redaction_canaries}
    return sorted(c for c in canaries if any(c in t for t in texts))


def _subset(manifest: Manifest) -> Manifest:
    return manifest.model_copy(update={"cases": [c for c in manifest.cases if c.llm_eval]})


def _copy_cases(manifest: Manifest, cases_root: Path, dest: Path) -> None:
    for case in manifest.cases:
        target = dest / case.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cases_root / case.path, target)


def _scanner_bench(manifest: Manifest, report: ScanReport) -> ScannerBench:
    match = match_findings(manifest, report.findings)
    mismatches = [
        f"{r.annotation_id}: expected detected={r.expect_detected}, got {r.detected}"
        for r in match.records
        if not r.expectation_met
    ]
    scanner = report.scanners[0] if report.scanners else None
    return ScannerBench(
        metrics=match.metrics,
        by_rule=match.by_rule,
        by_cwe=match.by_cwe,
        records=match.records,
        unmatched_findings=match.unmatched_findings,
        expectation_mismatches=mismatches,
        scan_duration_s=scanner.duration_s if scanner else report.duration_s,
        scanner_status=report.status,
        coverage=scanner.coverage.model_dump() if scanner and scanner.coverage else None,
    )


def run_benchmark(opts: BenchmarkOptions) -> BenchmarkResult:
    started = now()
    corpus = opts.corpus.resolve()
    manifest, cases_root = load_manifest(corpus)
    problems = validate_manifest(manifest, cases_root)
    opts.output.mkdir(parents=True, exist_ok=True)
    texts_to_check: list[str] = []

    with tempfile.TemporaryDirectory(prefix="tantular-bench-") as tmp_name:
        tmp = Path(tmp_name)
        scan = run_scan(
            ScanSettings(
                repo=cases_root,
                output=tmp / "scan",
                llm_enabled=False,
                semgrep_bin=opts.semgrep_bin,
                context_strategy=opts.context_strategy,
            )
        ).report
        (opts.output / "scan-report.json").write_text(render_json(scan), encoding="utf-8")
        texts_to_check += [render_json(scan), render_markdown(scan)]
        scanner = _scanner_bench(manifest, scan)
        adversarial = run_adversarial(corpus)
        context = evaluate_context(manifest, cases_root, scanner.records)

        llm_bench = perf = None
        if opts.llm != "off":
            llm_bench, perf, llm_texts = _run_llm(opts, manifest, cases_root, tmp)
            texts_to_check += llm_texts

    commit, dirty = code_state()
    model_meta = (
        model_state(opts.ollama.base_url, opts.ollama.model) if opts.llm == "ollama" else {}
    )
    provenance = Provenance(
        started_at=started,
        code_commit=commit,
        code_dirty=dirty,
        **machine(),  # type: ignore[arg-type]
        semgrep_version=scan.scanners[0].version if scan.scanners else None,
        model=opts.ollama.model if opts.llm == "ollama" else None,
        corpus_id=manifest.corpus_id,
        manifest_version=manifest.manifest_version,
        corpus_sha256=corpus_digest(corpus),
        settings={
            "llm": opts.llm,
            "llm_runs": opts.llm_runs,
            "cold": opts.cold,
            "concurrency": list(opts.concurrency),
            "think": False,
            "stream": False,
            "temperature": opts.ollama.temperature,
            "num_ctx": opts.ollama.num_ctx,
            "context_strategy": opts.context_strategy,
        },
        **model_meta,  # type: ignore[arg-type]
    )

    gates = compute_gates(manifest.thresholds, problems, scanner, adversarial, context)
    leaks = _leaked_canaries(manifest, texts_to_check)
    gates.append(
        Gate(
            id="redaction_canaries",
            passed=not leaks,
            detail=f"leaked: {len(leaks)} value(s)" if leaks else "",
        )
    )
    result = BenchmarkResult(
        status="pass" if all(g.passed for g in gates) else "fail",
        gates=gates,
        thresholds=manifest.thresholds,
        provenance=provenance,
        manifest_problems=problems,
        scanner=scanner,
        adversarial=adversarial,
        context=context,
        llm=llm_bench,
        performance=perf,
    )
    if opts.real_world_manifest is not None:
        from .realworld import DEFAULT_CACHE, evaluate_pointers, load_realworld

        result.real_world = evaluate_pointers(
            load_realworld(opts.real_world_manifest),
            opts.real_world_cache or DEFAULT_CACHE,
            semgrep_bin=opts.semgrep_bin,
        )
    if opts.baseline:
        result.comparison = compare(result, json.loads(opts.baseline.read_text()))
    result.provenance.finished_at = now()
    write_outputs(result, opts.output)
    return result


def _run_llm(
    opts: BenchmarkOptions, manifest: Manifest, cases_root: Path, tmp: Path
) -> tuple[LLMBench, PerfSummary | None, list[str]]:
    subset = _subset(manifest)
    notes: list[str] = []
    if opts.llm == "replay" and opts.replay_file is not None:
        recorded_paths = {k.rsplit(":", 1)[0] for k in ReplayLLM(opts.replay_file).responses}
        covered = [c for c in subset.cases if c.path in recorded_paths]
        if len(covered) < len(subset.cases):
            notes.append(
                f"replay file covers {len(covered)} of {len(subset.cases)} llm_eval "
                "cases; uncovered cases were not evaluated"
            )
        subset = subset.model_copy(update={"cases": covered})
    repo = tmp / "llm-cases"
    _copy_cases(subset, cases_root, repo)

    factory: Callable[[], LLMClient]
    if opts.llm == "replay":
        if opts.replay_file is None:
            raise ValueError("--llm replay requires --replay-file")
        replay = ReplayLLM(opts.replay_file)
        factory = lambda: replay  # noqa: E731
    else:
        factory = lambda: OllamaClient(opts.ollama)  # noqa: E731
    recorded: dict[str, str] = {}
    if opts.record_replay is not None:
        inner_factory = factory
        factory = lambda: RecordingLLM(inner_factory(), recorded)  # noqa: E731

    evals: list[ExplanationEval] = []
    runs: list[RunPerf] = []
    warm_walls: list[float] = []
    findings_by_run: dict[int, dict[str, Finding]] = {}
    texts: list[str] = []
    first_records: list[MatchRecord] = []
    first_findings: list[Finding] = []

    for run in range(1, max(1, opts.llm_runs) + 1):
        kind: RunKind = "warm"
        if opts.llm == "ollama" and opts.cold and run == 1:
            if evict_model(opts.ollama.base_url, opts.ollama.model):
                kind = "cold_after_eviction"
                notes.append(COLD_NOTE)
            else:
                notes.append("model eviction did not complete; run 1 treated as warm")
        log.info("LLM benchmark run %d (%s)", run, kind)
        report = run_scan(
            ScanSettings(
                repo=repo,
                output=tmp / f"llm-{run}",
                llm_enabled=True,
                max_llm_findings=10_000,
                context_strategy=opts.context_strategy,
                ollama=opts.ollama,
                semgrep_bin=opts.semgrep_bin,
            ),
            llm_client_factory=factory,
        ).report
        texts += [render_json(report), render_markdown(report)]
        if report.llm.available is False:
            notes.append(f"run {run}: LLM unavailable ({report.llm.error})")
        records = match_findings(subset, report.findings).records
        if run == 1:
            first_records, first_findings = records, report.findings
        evals += evaluate_run(subset, report.findings, records, run)
        findings_by_run[run] = {f.id: f for f in report.findings}
        if opts.llm == "ollama":
            scanner_s = sum(s.duration_s for s in report.scanners)
            runs.append(run_perf(run, kind, report.findings, report.duration_s, scanner_s))
            if kind == "warm":
                warm_walls += [
                    f.analysis.metrics.wall_s for f in report.findings if f.analysis.metrics
                ]

    if opts.record_replay is not None and recorded:
        opts.record_replay.parent.mkdir(parents=True, exist_ok=True)
        opts.record_replay.write_text(
            json.dumps(
                {
                    "model": opts.ollama.model if opts.llm == "ollama" else "replay",
                    "recorded_at": now(),
                    "note": "raw model outputs (pre-validation); prompts not stored",
                    "responses": dict(sorted(recorded.items())),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    queue = review_queue(evals, findings_by_run)
    (opts.output / "human_review.json").write_text(
        json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (opts.output / "human_review.md").write_text(render.review_markdown(queue), encoding="utf-8")

    perf = None
    if opts.llm == "ollama":
        concurrency = []
        if opts.concurrency and first_findings:
            notes.append(
                "concurrency results are experimental; server-side parallelism "
                "(e.g. OLLAMA_NUM_PARALLEL) is not observable from the client"
            )
            concurrency = concurrency_experiment(factory, first_findings, opts.concurrency)
        perf = PerfSummary(
            runs=runs,
            warm_per_finding=latency_stats(warm_walls) if warm_walls else None,
            concurrency=concurrency,
            notes=notes,
        )
    bench = LLMBench(
        mode=opts.llm,
        runs=max(1, opts.llm_runs),
        summary=summarize(evals),
        evaluations=evals,
        injection_cases_detected=injection_scanner_check(subset, first_records),
        review_queue_items=len(queue),
        notes=notes,
    )
    return bench, perf, texts


def write_outputs(result: BenchmarkResult, output: Path) -> None:
    (output / "benchmark.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output / "benchmark.md").write_text(render.benchmark_markdown(result), encoding="utf-8")

"""Markdown rendering for benchmark results and the human-review queue."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..reports import _code_block, _inline

if TYPE_CHECKING:
    from .results import BenchmarkResult


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if value % 1 else f"{value:.0f}"
    return str(value)


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.0f}%"


def benchmark_markdown(r: BenchmarkResult) -> str:
    p = r.provenance
    out = [
        "# Tantular Cyber — Benchmark",
        "",
        f"**Status:** {'✅ PASS' if r.status == 'pass' else '❌ FAIL'} "
        f"(deterministic gates only; LLM metrics never gate)  ",
        f"**Corpus:** `{p.corpus_id}` manifest {p.manifest_version}, sha256 "
        f"`{p.corpus_sha256[:12]}…`  ",
        f"**Code:** tantular-cyber {p.tantular_cyber_version}, commit "
        f"`{(p.code_commit or 'n/a')[:12]}`{' (dirty)' if p.code_dirty else ''}  ",
        f"**Run:** {p.started_at} → {p.finished_at}  ",
        f"**Machine:** {p.cpu or p.machine}, {_fmt(p.memory_gb)} GB, {p.platform}, "
        f"Python {p.python}  ",
        f"**Tools:** semgrep {_fmt(p.semgrep_version)}"
        + (
            f", Ollama {p.ollama_version}, model `{p.model}` "
            f"({', '.join(p.model_details.values())}; digest `{(p.model_digest or '')[:12]}`)"
            if p.model
            else ""
        ),
        "",
        "## Gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    out += [
        f"| {g.id} | {'pass' if g.passed else '**FAIL**'} | {_inline(g.detail)} |" for g in r.gates
    ]
    out.append("")

    if s := r.scanner:
        m = s.metrics
        out += [
            "## Scanner detection (deterministic)",
            "",
            f"Thresholds: precision ≥ {r.thresholds.min_precision}, recall ≥ "
            f"{r.thresholds.min_recall}. Scan time {s.scan_duration_s:.1f} s.",
            "",
            "| TP | FP | FN | TN | Precision | Recall | F1 | Specificity |",
            "|---|---|---|---|---|---|---|---|",
            f"| {m.tp} | {m.fp} | {m.fn} | {m.tn} | {_fmt(m.precision)} | {_fmt(m.recall)} | "
            f"{_fmt(m.f1)} | {_fmt(m.specificity)} |",
            "",
            "| Breakdown | TP | FP | FN | TN | Precision | Recall |",
            "|---|---|---|---|---|---|---|",
        ]
        for prefix, group in (("rule", s.by_rule), ("cwe", s.by_cwe)):
            for key, g in group.items():
                out.append(
                    f"| {prefix} `{_inline(key)}` | {g.tp} | {g.fp} | {g.fn} | {g.tn} | "
                    f"{_fmt(g.precision)} | {_fmt(g.recall)} |"
                )
        if s.coverage:
            c = s.coverage
            out += [
                "",
                f"Coverage: {c['scanned']}/{c['in_scope']} in-scope files scanned, "
                f"verified={c['verified']}.",
            ]
        out += [
            "",
            "| Annotation | Location | Label | Outcome | Expected | Finding |",
            "|---|---|---|---|---|---|",
        ]
        for rec in s.records:
            exp = "ok" if rec.expectation_met else "**MISMATCH**"
            out.append(
                f"| {rec.annotation_id} | `{rec.path}:{rec.line}-{rec.end_line}` | "
                f"{rec.label} | {rec.outcome} | {exp} | `{rec.finding_id or '-'}` |"
            )
        if s.unmatched_findings:
            out += ["", "Unmatched findings (counted as FP):", ""]
            out += [
                f"- `{u.finding_id}` {u.path}:{u.start_line} {u.rule_id} ({u.reason})"
                for u in s.unmatched_findings
            ]
        out.append("")

    if ctx := r.context:
        out += [
            "## Evidence context (window vs AST, deterministic)",
            "",
            f"Budget: {ctx.budget}. A construct is covered when the context shows both "
            "where the value is built (annotation start) and the sink (annotation end).",
            "",
            "| Strategy | n | Construct covered | Mean lines | Mean chars | Max chars | "
            "Truncated | Over budget | Fallbacks |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for st in ctx.strategies:
            fb = ", ".join(f"{k}: {v}" for k, v in st.fallbacks.items()) or "-"
            out.append(
                f"| {st.strategy} | {st.n} | {st.construct_covered} "
                f"({_pct(st.coverage_rate)}) | {_fmt(st.mean_lines)} | "
                f"{_fmt(st.mean_chars)} | {st.max_chars} | {st.truncated} | "
                f"{st.over_budget} | {_inline(fb)} |"
            )
        if ctx.newly_covered:
            out += ["", "Newly covered by AST context: " + ", ".join(ctx.newly_covered)]
        if ctx.lost_coverage:
            out += ["", "**Coverage lost vs window:** " + ", ".join(ctx.lost_coverage)]
        out.append("")

    out += ["## Adversarial suites (regression, not proof of protection)", ""]
    for suite in r.adversarial:
        counts = ", ".join(f"{k}: {v}" for k, v in suite.counts.items())
        out += [f"**{suite.suite}** v{suite.suite_version} — {suite.total} items — {counts}", ""]
        gaps = [
            i for i in suite.items if i.status in ("known_gap", "gap_closed", "unexpected_failure")
        ]
        if gaps:
            out += ["| Item | Status | Category | Variant | Note |", "|---|---|---|---|---|"]
            out += [
                f"| {i.id} | {i.status} | {i.category} | {i.variant} | {_inline(i.note)} |"
                for i in gaps
            ]
            out.append("")

    if llm := r.llm:
        sm = llm.summary
        out += [
            "## LLM explanation quality (sample-based, advisory)",
            "",
            f"Mode `{llm.mode}`, {llm.runs} run(s), **n = {sm.n_explanations}** explanations. "
            "Rates come from deterministic proxies with Wilson 95% intervals; they do not "
            f"establish semantic correctness. Human-review queue: {llm.review_queue_items} "
            "item(s) in `human_review.md`.",
            "",
            "| Check | Passed / assessed | Rate | 95% CI |",
            "|---|---|---|---|",
        ]
        for cid, rs in sm.checks.items():
            ci = f"{_pct(rs.ci95_low)}–{_pct(rs.ci95_high)}" if rs.ci95_low is not None else "n/a"
            out.append(f"| {cid} | {rs.passed} / {rs.assessed} | {_pct(rs.rate)} | {ci} |")
        out += [
            "",
            "Confidence by ground-truth label:",
            "",
            "| Label | low | medium | high |",
            "|---|---|---|---|",
        ]
        for label, counts in sm.calibration.items():
            out.append(
                f"| {label} | {counts.get('low', 0)} | {counts.get('medium', 0)} | "
                f"{counts.get('high', 0)} |"
            )
        out += [
            "",
            "| Run | Case | Finding | Label | Status | Failed checks |",
            "|---|---|---|---|---|---|",
        ]
        for ev in llm.evaluations:
            failed = (
                ", ".join(
                    f"{c.id} ({_inline(c.detail)})" if c.detail else c.id
                    for c in ev.checks
                    if c.passed is False
                )
                or "-"
            )
            out.append(
                f"| {ev.run} | {ev.case_id} | `{ev.finding_id}` | {ev.label} | "
                f"{ev.analysis_status} | {failed} |"
            )
        if llm.injection_cases_detected:
            det = ", ".join(
                f"{k}={'yes' if v else 'NO'}" for k, v in llm.injection_cases_detected.items()
            )
            out += ["", f"Injection cases still detected by the scanner: {det}"]
        out.append("")

    if perf := r.performance:
        out += [
            "## Performance",
            "",
            "| Run | Kind | Findings | Scan total s | p50 s | p95 s | Load s (1st call) "
            "| Prompt tok | Output tok | Output tok/s |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for run in perf.runs:
            pf = run.per_finding
            out.append(
                f"| {run.run} | {run.kind} | {run.findings_explained} | "
                f"{run.scan_total_s:.1f} | {_fmt(pf.p50_s)} | {_fmt(pf.p95_s)} | "
                f"{_fmt(run.first_call_load_s)} | {run.prompt_tokens} | "
                f"{run.output_tokens} | {_fmt(run.mean_output_tokens_per_s)} |"
            )
        if w := perf.warm_per_finding:
            out += [
                "",
                f"Warm per-finding latency: p50 {_fmt(w.p50_s)} s, p95 {_fmt(w.p95_s)} s "
                f"(n = {w.n}, nearest-rank).",
            ]
        if perf.concurrency:
            out += [
                "",
                "| Concurrency | Requests | Errors | Wall s | Findings/min | p50 s | p95 s |",
                "|---|---|---|---|---|---|---|",
            ]
            out += [
                f"| {c.concurrency} | {c.requests} | {c.errors} | {c.wall_s:.1f} | "
                f"{c.throughput_per_min:.2f} | {_fmt(c.latency.p50_s)} | "
                f"{_fmt(c.latency.p95_s)} |"
                for c in perf.concurrency
            ]
        if perf.notes:
            out += [""] + [f"- {_inline(n)}" for n in perf.notes]
        out.append("")

    if cmp := r.comparison:
        changed = [d for d in cmp.deltas if d.delta not in (None, 0)]
        out += [
            "## Comparison with baseline",
            "",
            f"Baseline {cmp.baseline_started_at}, commit `{(cmp.baseline_commit or 'n/a')[:12]}`, "
            f"same corpus: {'yes' if cmp.same_corpus else '**no**'}."
            + (
                ""
                if cmp.same_corpus
                else " Corpora differ: absolute counts are shown but "
                "not judged; compare rates (overall and per CWE)."
            ),
            "",
        ]
        if changed:
            out += ["| Metric | Baseline | Current | Δ | |", "|---|---|---|---|---|"]
            out += [
                f"| {d.metric} | {_fmt(d.baseline)} | {_fmt(d.current)} | {_fmt(d.delta)} | "
                f"{'⚠️ worse' if d.worse else ''} |"
                for d in changed
            ]
        else:
            out.append("No metric changed.")
        out.append("")

    out += ["## Limitations", ""] + [f"- {_inline(x)}" for x in r.limitations] + [""]
    return "\n".join(out)


def review_markdown(queue: list[dict[str, Any]]) -> str:
    out = [
        "# Human review queue",
        "",
        "Fill the `reviewer` fields in `human_review.json`. Rubric: clarity_id (1-5), "
        "remediation_adequate (yes/partial/no), evidence_faithful (yes/no), notes.",
        "Evidence is redacted; explanations are model output (untrusted, advisory).",
        "",
    ]
    for item in queue:
        out += [
            f"## {item['review_id']} — {item['case_id']} ({item['label']})",
            "",
            f"Location: `{item['location']}`",
            "",
            "Evidence:",
            "",
            _code_block(str(item["evidence"])),
            "",
            "Explanation:",
            "",
            _code_block(json.dumps(item["explanation"], indent=2, ensure_ascii=False), "json"),
            "",
            f"Automated checks: `{json.dumps(item['automated_checks'])}`",
            "",
        ]
    return "\n".join(out)

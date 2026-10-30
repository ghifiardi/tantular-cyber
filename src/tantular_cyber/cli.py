"""Command-line interface: ``tantular-cyber scan|doctor|sample|plugins|benchmark|schema``.

``scan`` is deterministic by default (Semgrep + tantular-callgraph, no model needed);
``--ai`` adds optional local Ollama explanations. ``--no-llm`` is accepted for
compatibility (it is now the default) and will be removed in a future release.

Exit codes for ``scan``:
  0  complete — all requested stages succeeded (findings may or may not exist)
  2  invalid input (bad repo/output path, bad arguments)
  3  failed — scanner failed; reports still written with the error
  4  degraded — reports written, but coverage is partial (e.g. Semgrep missing, unsupported
     platform) or the requested AI stage was unavailable or partial

Exit codes for ``doctor``:
  0  full scan capability
  3  scanning unavailable
  4  partial scan capability (reports would be degraded)

Exit codes for ``benchmark``:
  0  all deterministic gates passed
  2  invalid input (corpus/manifest missing or unreadable, bad arguments)
  5  one or more deterministic gates failed (see benchmark.md)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import (
    OllamaSettings,
    ScanSettings,
    default_corpus,
    default_realworld_manifest,
    default_rules,
    sample_project,
)
from .doctor import unsupported_platform_reason
from .models import ScanReport
from .ollama_client import OllamaError
from .pipeline import run_scan
from .reports import write_reports
from .security import PathSafetyError

EXIT_OK, EXIT_INPUT, EXIT_FAILED, EXIT_DEGRADED, EXIT_GATES = 0, 2, 3, 4, 5
_STATUS_EXIT = {"complete": EXIT_OK, "degraded": EXIT_DEGRADED, "failed": EXIT_FAILED}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tantular-cyber",
        description="Local-first, read-only repository security analyzer. Deterministic scans "
        "need Semgrep (user-installed); AI explanations via local Ollama are optional (--ai).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan an authorized local repository")
    scan.add_argument("--repo", required=True, help="path to the repository to analyze")
    scan.add_argument("--output", required=True, help="directory for report.json / report.md")
    scan.add_argument(
        "--rules",
        action="append",
        type=Path,
        help="local Semgrep rule file (repeatable; default: bundled CWE-89/78/22/79/798 rules)",
    )
    scan.add_argument("--semgrep-bin", default=None, help="semgrep executable (default: semgrep)")
    scan.add_argument("--semgrep-timeout", type=float, default=600.0, help="seconds")
    scan.add_argument("--max-file-bytes", type=int, default=1_000_000)
    scan.add_argument(
        "--ai",
        action="store_true",
        help="add explanations from a local Ollama model (optional; scans never need a model)",
    )
    scan.add_argument(
        "--no-llm",
        action="store_true",
        help="deprecated: deterministic-only is now the default",
    )
    scan.add_argument(
        "--model", default=None, help="Ollama model for --ai (default: tantular-cyber:latest)"
    )
    scan.add_argument(
        "--ollama-url", default=None, help="Ollama base URL (default: http://127.0.0.1:11434)"
    )
    scan.add_argument("--ollama-timeout", type=float, default=None, help="seconds per request")
    scan.add_argument(
        "--allow-remote-ollama",
        action="store_true",
        help="allow a non-localhost Ollama endpoint (sends code excerpts off-host)",
    )
    scan.add_argument("--max-llm-findings", type=int, default=25)
    scan.add_argument(
        "--context-lines",
        type=int,
        default=3,
        help="window radius (window strategy and AST fallback)",
    )
    scan.add_argument(
        "--no-callgraph",
        action="store_true",
        help="disable the interprocedural tantular-callgraph scanner",
    )
    scan.add_argument(
        "--callgraph-depth",
        type=int,
        default=3,
        help="max call levels followed by tantular-callgraph",
    )
    scan.add_argument(
        "--context-strategy",
        choices=["ast", "window"],
        default="ast",
        help="evidence context: bounded AST-aware (default) or fixed window",
    )
    scan.add_argument(
        "--keep-raw",
        action="store_true",
        help="also write the scanner's original JSON output (paths scrubbed)",
    )
    scan.add_argument("--allow-output-inside-repo", action="store_true")

    doctor = sub.add_parser(
        "doctor",
        help="read-only diagnostics: Semgrep, platform, optional Ollama/models, privacy",
    )
    doctor.add_argument("--model", default=None)
    doctor.add_argument("--ollama-url", default=None)
    doctor.add_argument("--allow-remote-ollama", action="store_true", help=argparse.SUPPRESS)
    doctor.add_argument("--semgrep-bin", default=None)
    doctor.add_argument(
        "--no-ollama", action="store_true", help="do not contact Ollama (not even loopback)"
    )
    doctor.add_argument("--json", action="store_true", help="machine-readable output")

    sample = sub.add_parser(
        "sample", help="copy a small synthetic vulnerable project to try a scan on"
    )
    sample.add_argument("--dest", type=Path, required=True, help="new or empty directory")

    plugins = sub.add_parser(
        "plugins", help="scanner plugin contract: list built-ins, validate a manifest"
    )
    plugins.add_argument("action", choices=["list", "validate"])
    plugins.add_argument("manifest", nargs="?", type=Path, help="plugin manifest (validate)")

    bench = sub.add_parser("benchmark", help="run the evaluation benchmark (offline by default)")
    bench.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="corpus directory containing manifest.json (default: bundled corpus)",
    )
    bench.add_argument(
        "--output", type=Path, required=True, help="directory for benchmark.json / benchmark.md"
    )
    bench.add_argument(
        "--llm",
        choices=["off", "ollama", "replay"],
        default="off",
        help="LLM evaluation: off (default, CI), ollama (local, opt-in), replay (recorded outputs)",
    )
    bench.add_argument("--replay-file", type=Path, default=None)
    bench.add_argument(
        "--llm-runs",
        type=int,
        default=1,
        help="repeat the LLM evaluation N times (performance samples)",
    )
    bench.add_argument(
        "--cold",
        action="store_true",
        help="evict the model before run 1 (labelled cold_after_eviction)",
    )
    bench.add_argument(
        "--concurrency",
        default="",
        help="comma-separated levels for a bounded concurrency experiment, "
        "1..4 (ollama only), e.g. 1,2",
    )
    bench.add_argument(
        "--context-strategy",
        choices=["ast", "window"],
        default="ast",
        help="evidence context strategy used for the scans",
    )
    bench.add_argument(
        "--baseline", type=Path, default=None, help="previous benchmark.json to compare against"
    )
    bench.add_argument(
        "--real-world",
        action="store_true",
        help="also evaluate cached real-world pointers (no network; run 'realworld fetch' first)",
    )
    bench.add_argument("--real-world-cache", type=Path, default=None)
    bench.add_argument(
        "--record-replay",
        type=Path,
        default=None,
        help="save raw model outputs from run 1 as a replay file",
    )
    bench.add_argument("--semgrep-bin", default=None)
    bench.add_argument("--model", default=None)
    bench.add_argument("--ollama-url", default=None)
    bench.add_argument("--ollama-timeout", type=float, default=None)
    bench.add_argument("--allow-remote-ollama", action="store_true")

    rw = sub.add_parser("realworld", help="pointer-only real-world cases (opt-in fetch)")
    rw.add_argument("action", choices=["fetch", "evaluate", "clean", "audit"])
    rw.add_argument("--manifest", type=Path, default=None, help="default: bundled manifest")
    rw.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="local cache (default ~/.cache/tantular-cyber/realworld)",
    )
    rw.add_argument("--semgrep-bin", default=None)

    cmp = sub.add_parser("compare", help="opt-in local model / prompt-profile comparison")
    cmp.add_argument("--models", default="tantular-cyber:latest")
    cmp.add_argument("--profiles", default="baseline", help="comma-separated prompt profiles")
    cmp.add_argument("--contexts", default="8192", help="comma-separated num_ctx values")
    cmp.add_argument(
        "--modes",
        default="pinned",
        help="pinned (our system prompt + pinned options) and/or profile "
        "(the Modelfile's SYSTEM and PARAMETERs apply; same inputs and seeds)",
    )
    cmp.add_argument("--runs", type=int, default=3)
    cmp.add_argument("--max-findings", type=int, default=40)
    cmp.add_argument(
        "--identity-pair",
        default="",
        help="two models that share weights, e.g. a profile and its base",
    )
    cmp.add_argument("--corpus", type=Path, default=None, help="default: bundled corpus")
    cmp.add_argument("--output", type=Path, required=True)
    cmp.add_argument(
        "--fresh",
        action="store_true",
        help="ignore and replace the samples.jsonl checkpoint in --output",
    )
    cmp.add_argument("--ollama-url", default=None)
    cmp.add_argument("--ollama-timeout", type=float, default=None)
    cmp.add_argument("--allow-remote-ollama", action="store_true")
    cmp.add_argument("--model", default=None, help=argparse.SUPPRESS)

    rev = sub.add_parser("review", help="human-review templates and agreement summaries")
    rev.add_argument("action", choices=["template", "summarize"])
    rev.add_argument("--queue", type=Path, required=True, help="human_review.json")
    rev.add_argument("--reviewer", default=None, help="pseudonym for 'template'")
    rev.add_argument("--role", choices=["primary", "adjudicator"], default="primary")
    rev.add_argument("--reviews", type=Path, nargs="*", default=[])
    rev.add_argument("--adjudication", type=Path, nargs="*", default=[])
    rev.add_argument("--output", type=Path, required=True)

    sub.add_parser("schema", help="print the JSON schema of report.json")
    return parser


def _ollama_settings(args: argparse.Namespace) -> OllamaSettings:
    return OllamaSettings.from_env(
        base_url=args.ollama_url,
        model=args.model,
        timeout_s=getattr(args, "ollama_timeout", None),
        allow_remote=True if args.allow_remote_ollama else None,
    )


def _semgrep_bin(args: argparse.Namespace) -> str:
    return args.semgrep_bin or os.environ.get("TANTULAR_SEMGREP_BIN") or "semgrep"


def cmd_scan(args: argparse.Namespace) -> int:
    if args.ai and args.no_llm:
        print("error: --ai and --no-llm are mutually exclusive", file=sys.stderr)
        return EXIT_INPUT
    if args.no_llm:
        print(
            "note: --no-llm is deprecated; deterministic-only is now the default (use --ai to "
            "add explanations)",
            file=sys.stderr,
        )
    settings = ScanSettings(
        repo=Path(args.repo),
        output=Path(args.output),
        rules=tuple(args.rules) if args.rules else default_rules(),
        semgrep_bin=_semgrep_bin(args),
        semgrep_timeout_s=args.semgrep_timeout,
        max_file_bytes=args.max_file_bytes,
        allow_output_inside_repo=args.allow_output_inside_repo,
        keep_raw_scanner_output=args.keep_raw,
        llm_enabled=args.ai,
        max_llm_findings=args.max_llm_findings,
        context_lines=args.context_lines,
        context_strategy=args.context_strategy,
        callgraph_enabled=not args.no_callgraph,
        callgraph_max_depth=args.callgraph_depth,
        ollama=_ollama_settings(args),
    )
    for rule in settings.rules:
        if not rule.is_file():
            print(f"error: rule file not found: {rule}", file=sys.stderr)
            return EXIT_INPUT
    try:
        result = run_scan(settings)
    except PathSafetyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT

    report = result.report
    written = write_reports(
        report,
        result.output,
        repo=result.repo,
        raw_outputs=result.raw_scanner_outputs if settings.keep_raw_scanner_output else None,
    )
    _print_summary(report, written)
    return _STATUS_EXIT[report.status]


def _coverage_label(report: ScanReport) -> str:
    """Scanner coverage only (the optional AI stage is reported separately)."""
    ran = [s.name for s in report.scanners if s.succeeded]
    gaps = any(
        s.coverage is not None and (s.coverage.not_scanned or not s.coverage.verified)
        for s in report.scanners
        if s.succeeded
    )
    full = (
        "semgrep" in ran
        and all(s.succeeded for s in report.scanners)
        and not gaps
        and unsupported_platform_reason() is None
    )
    return f"{'full' if full else 'PARTIAL'} (scanners succeeded: {', '.join(ran) or 'none'})"


def _print_summary(report: ScanReport, written: dict[str, Path]) -> None:
    print(f"status: {report.status}")
    print(f"coverage: {_coverage_label(report)}")
    print(f"ai explanations: {'requested' if report.llm.enabled else 'not requested'}")
    for reason in report.status_reasons:
        print(f"  - {reason}")
    print(f"findings: {report.totals.findings} {report.totals.by_cwe or ''}".rstrip())
    print(f"analyses: {report.totals.analyses_by_status}")
    for kind, path in written.items():
        print(f"{kind}: {path}")
    if report.status != "complete":
        print(
            f"WARNING: scan {report.status.upper()} — see status_reasons/errors in the report",
            file=sys.stderr,
        )


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import render_doctor, run_doctor

    report = run_doctor(
        semgrep_bin=_semgrep_bin(args),
        ollama=_ollama_settings(args),
        check_ollama=not args.no_ollama,
    )
    print(report.model_dump_json(indent=2) if args.json else render_doctor(report), end="\n")
    return {"full": EXIT_OK, "partial": EXIT_DEGRADED, "unavailable": EXIT_FAILED}[
        report.scan_capability
    ]


def cmd_sample(args: argparse.Namespace) -> int:
    source = sample_project()
    dest: Path = args.dest
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        print(f"error: {dest} exists and is not an empty directory", file=sys.stderr)
        return EXIT_INPUT
    if not source.is_dir():
        print("error: bundled sample project is missing; reinstall", file=sys.stderr)
        return EXIT_INPUT
    shutil.copytree(source, dest, dirs_exist_ok=True)
    print(f"sample project (synthetic, deliberately vulnerable, never executed): {dest}")
    print(f"next: tantular-cyber scan --repo {dest} --output {dest.parent / 'tantular-reports'}")
    return EXIT_OK


def cmd_plugins(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from .plugins import PluginManifest, negotiate, plugin_policy

    if args.action == "list":
        print(json.dumps(plugin_policy(), indent=2))
        return EXIT_OK
    if args.manifest is None:
        print("error: plugins validate requires a manifest path", file=sys.stderr)
        return EXIT_INPUT
    try:
        manifest = PluginManifest.model_validate(
            json.loads(args.manifest.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(f"invalid manifest: {exc}", file=sys.stderr)
        return EXIT_INPUT
    compat = negotiate(manifest)
    print(f"manifest: {manifest.id} {manifest.version} (plugin API {manifest.plugin_api})")
    print(f"compatibility: {compat.reason}")
    print("execution: DISABLED — third-party plugins are not run without an enforceable sandbox")
    return EXIT_OK if compat.compatible else EXIT_INPUT


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark.manifest import ManifestError
    from .benchmark.runner import BenchmarkOptions, run_benchmark

    try:
        levels = tuple(int(x) for x in args.concurrency.split(",") if x.strip())
    except ValueError:
        print("error: --concurrency must be comma-separated integers", file=sys.stderr)
        return EXIT_INPUT
    if any(not 1 <= n <= 4 for n in levels):
        print("error: --concurrency levels must be between 1 and 4", file=sys.stderr)
        return EXIT_INPUT
    if args.llm == "replay" and not args.replay_file:
        print("error: --llm replay requires --replay-file", file=sys.stderr)
        return EXIT_INPUT
    if args.baseline and not args.baseline.is_file():
        print(f"error: baseline not found: {args.baseline}", file=sys.stderr)
        return EXIT_INPUT
    opts = BenchmarkOptions(
        corpus=args.corpus or default_corpus(),
        output=args.output,
        llm=args.llm,
        replay_file=args.replay_file,
        ollama=_ollama_settings(args),
        llm_runs=max(1, args.llm_runs),
        cold=args.cold,
        concurrency=levels,
        baseline=args.baseline,
        record_replay=args.record_replay,
        real_world_manifest=(default_realworld_manifest() if args.real_world else None),
        real_world_cache=args.real_world_cache,
        context_strategy=args.context_strategy,
        semgrep_bin=_semgrep_bin(args),
    )
    try:
        result = run_benchmark(opts)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT

    print(f"benchmark: {result.status.upper()}")
    for gate in result.gates:
        print(f"  {'pass' if gate.passed else 'FAIL'}  {gate.id}  {gate.detail}".rstrip())
    if result.scanner:
        m = result.scanner.metrics
        print(
            f"scanner: TP={m.tp} FP={m.fp} FN={m.fn} TN={m.tn} precision={m.precision} "
            f"recall={m.recall} f1={m.f1}"
        )
    for suite in result.adversarial:
        print(f"adversarial {suite.suite}: {suite.counts}")
    if result.llm:
        print(
            f"llm ({result.llm.mode}, sample-based n={result.llm.summary.n_explanations}): "
            + ", ".join(f"{k}={v.rate}" for k, v in result.llm.summary.checks.items())
        )
    if result.performance and result.performance.warm_per_finding:
        w = result.performance.warm_per_finding
        print(f"perf warm per-finding: p50={w.p50_s}s p95={w.p95_s}s n={w.n}")
    print(f"json: {args.output / 'benchmark.json'}")
    print(f"markdown: {args.output / 'benchmark.md'}")
    return EXIT_OK if result.status == "pass" else EXIT_GATES


def cmd_realworld(args: argparse.Namespace) -> int:
    from .benchmark.realworld import (
        DEFAULT_CACHE,
        clean_cache,
        evaluate_pointers,
        fetch_pointers,
        license_audit,
        load_realworld,
        validate_realworld,
    )

    cache = args.cache or DEFAULT_CACHE
    if args.action == "clean":
        print(f"removed {cache}" if clean_cache(cache) else f"nothing to remove at {cache}")
        return EXIT_OK
    try:
        manifest = load_realworld(args.manifest or default_realworld_manifest())
    except (OSError, ValueError) as exc:
        print(f"error: cannot read manifest: {exc}", file=sys.stderr)
        return EXIT_INPUT
    problems = validate_realworld(manifest)
    if problems:
        print("error: invalid manifest:\n  " + "\n  ".join(problems), file=sys.stderr)
        return EXIT_INPUT
    if args.action == "audit":
        print(json.dumps(license_audit(manifest), indent=2))
        return EXIT_OK
    if args.action == "fetch":
        print(
            f"fetching {len(manifest.entries)} pointed-to file(s) into {cache} "
            "(network access, explicit opt-in)"
        )
        for pid, outcome in fetch_pointers(manifest, cache):
            print(f"  {pid}: {outcome}")
        return EXIT_OK
    result = evaluate_pointers(manifest, cache, semgrep_bin=_semgrep_bin(args))
    print(result.caveat)
    for r in result.results:
        print(f"  {r.id:32} {r.cwe:8} {r.state:16} {','.join(r.tools)}")
    print(
        f"detected {result.detected} / evaluated {result.evaluated} "
        f"(of {result.entries}; fetched {result.fetched})"
    )
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    from .benchmark.compare import parse_arms, render_comparison, run_comparison

    try:
        arms = parse_arms(args.models, args.profiles, args.contexts, args.modes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    pair = [m.strip() for m in args.identity_pair.split(",") if m.strip()]
    try:
        result = run_comparison(
            args.corpus or default_corpus(),
            arms,
            runs=max(1, args.runs),
            max_findings=max(1, args.max_findings),
            ollama=_ollama_settings(args),
            identity_pair=pair,
            on_progress=lambda msg: logging.getLogger(__name__).info(msg),
            checkpoint_path=args.output / "samples.jsonl",
            fresh=args.fresh,
        )
    except (OllamaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(result.model_dump_json(indent=2) + "\n")
    (args.output / "comparison.md").write_text(render_comparison(result))
    for arm in result.arms:
        rates = ", ".join(f"{k}={v['rate']}" for k, v in arm.checks.items())
        print(f"{arm.arm}: n={arm.n} p50={arm.latency.get('p50_s')}s {rates}")
    for a in result.adoption:
        print(f"adopt {a.arm}: {a.adopt} ({'; '.join(a.reasons)})")
    print(f"json: {args.output / 'comparison.json'}")
    return EXIT_OK


def cmd_review(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from .benchmark.review import (
        load_review,
        render_review_summary,
        sha256_file,
        summarize_reviews,
        template,
    )

    try:
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read queue: {exc}", file=sys.stderr)
        return EXIT_INPUT
    queue_sha = sha256_file(args.queue)
    if args.action == "template":
        if not args.reviewer:
            print("error: --reviewer pseudonym required", file=sys.stderr)
            return EXIT_INPUT
        try:
            form = template(queue, args.reviewer, queue_sha, args.role)
        except ValidationError:
            print("error: reviewer must be a pseudonym: letters, digits, _ or -", file=sys.stderr)
            return EXIT_INPUT
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(form, indent=2) + "\n")
        print(f"template for {args.reviewer}: {args.output} ({len(queue)} items)")
        return EXIT_OK
    try:
        reviews = [load_review(p) for p in args.reviews]
        adjudications = [load_review(p) for p in args.adjudication]
    except (OSError, ValueError, ValidationError) as exc:
        print(f"error: invalid review file: {exc}", file=sys.stderr)
        return EXIT_INPUT
    summary = summarize_reviews(queue, queue_sha, reviews, adjudications)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "review-summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    (args.output / "review-summary.md").write_text(render_review_summary(summary))
    for w in summary.warnings:
        print(f"warning: {w}")
    for a in summary.pairwise:
        print(f"{a.dimension}: n={a.n} agreement={a.raw_agreement} kappa={a.kappa}")
    print(f"json: {args.output / 'review-summary.json'}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "sample":
        return cmd_sample(args)
    if args.command == "plugins":
        return cmd_plugins(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)
    if args.command == "realworld":
        return cmd_realworld(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "review":
        return cmd_review(args)
    print(json.dumps(ScanReport.model_json_schema(), indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

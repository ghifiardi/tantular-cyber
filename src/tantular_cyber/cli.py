"""Command-line interface: ``tantular-cyber scan|benchmark|doctor|schema``.

Exit codes for ``scan``:
  0  complete — scanner and LLM stages succeeded (findings may or may not exist)
  2  invalid input (bad repo/output path, bad arguments)
  3  failed — scanner failed; reports still written with the error
  4  degraded — reports written, but the LLM stage was unavailable or partial

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
from .config import OllamaSettings, ScanSettings, default_rules
from .models import ScanReport
from .ollama_client import OllamaClient, OllamaError
from .pipeline import run_scan
from .reports import write_reports
from .security import PathSafetyError

EXIT_OK, EXIT_INPUT, EXIT_FAILED, EXIT_DEGRADED, EXIT_GATES = 0, 2, 3, 4, 5
_STATUS_EXIT = {"complete": EXIT_OK, "degraded": EXIT_DEGRADED, "failed": EXIT_FAILED}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tantular-cyber",
        description="Local, read-only repository security analyzer (Semgrep + local Ollama).",
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
        help="local Semgrep rule file (repeatable; default: bundled CWE-89 rules)",
    )
    scan.add_argument("--semgrep-bin", default=None, help="semgrep executable (default: semgrep)")
    scan.add_argument("--semgrep-timeout", type=float, default=600.0, help="seconds")
    scan.add_argument("--max-file-bytes", type=int, default=1_000_000)
    scan.add_argument("--no-llm", action="store_true", help="skip LLM explanations")
    scan.add_argument("--model", default=None, help="Ollama model (default: tantular-cyber:latest)")
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
    scan.add_argument("--context-lines", type=int, default=3)
    scan.add_argument(
        "--keep-raw",
        action="store_true",
        help="also write the scanner's original JSON output (paths scrubbed)",
    )
    scan.add_argument("--allow-output-inside-repo", action="store_true")

    doctor = sub.add_parser("doctor", help="check semgrep and Ollama availability")
    doctor.add_argument("--model", default=None)
    doctor.add_argument("--ollama-url", default=None)
    doctor.add_argument("--allow-remote-ollama", action="store_true")
    doctor.add_argument("--semgrep-bin", default=None)

    bench = sub.add_parser("benchmark", help="run the evaluation benchmark (offline by default)")
    bench.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/corpus"),
        help="corpus directory containing manifest.json",
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
        "--baseline", type=Path, default=None, help="previous benchmark.json to compare against"
    )
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
    settings = ScanSettings(
        repo=Path(args.repo),
        output=Path(args.output),
        rules=tuple(args.rules) if args.rules else default_rules(),
        semgrep_bin=_semgrep_bin(args),
        semgrep_timeout_s=args.semgrep_timeout,
        max_file_bytes=args.max_file_bytes,
        allow_output_inside_repo=args.allow_output_inside_repo,
        keep_raw_scanner_output=args.keep_raw,
        llm_enabled=not args.no_llm,
        max_llm_findings=args.max_llm_findings,
        context_lines=args.context_lines,
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


def _print_summary(report: ScanReport, written: dict[str, Path]) -> None:
    print(f"status: {report.status}")
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
    ok = True
    semgrep = shutil.which(_semgrep_bin(args))
    print(f"semgrep: {semgrep or 'NOT FOUND (brew install semgrep)'}")
    ok &= semgrep is not None
    try:
        with OllamaClient(_ollama_settings(args)) as client:
            version = client.health_check()
            print(f"ollama: {client.base_url} version={version} model={client.model} OK")
    except OllamaError as exc:
        ok = False
        print(f"ollama: UNAVAILABLE — {exc}")
    return EXIT_OK if ok else EXIT_DEGRADED


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
        corpus=args.corpus,
        output=args.output,
        llm=args.llm,
        replay_file=args.replay_file,
        ollama=_ollama_settings(args),
        llm_runs=max(1, args.llm_runs),
        cold=args.cold,
        concurrency=levels,
        baseline=args.baseline,
        record_replay=args.record_replay,
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
    if args.command == "benchmark":
        return cmd_benchmark(args)
    print(json.dumps(ScanReport.model_json_schema(), indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

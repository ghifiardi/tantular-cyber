"""Research improvement workflow CLI (research-only; not part of the tantular-cyber package).

    python tools/research/improve.py validate tools/research/fixtures/synthetic_study.json
    python tools/research/improve.py demo --output ~/research/runs/demo-1
    python tools/research/improve.py run-manifest-hash RUN_DIR
    python tools/research/improve.py report --study STUDY.json --evidence RUN_DIR [...] \\
        [--expected-run-manifest-sha256 SHA [...]] --output REPORT_DIR
    python tools/research/improve.py decide --study STUDY.json --report REPORT_DIR/report.json \\
        --decision reject --decided-by reviewer1 --rationale "..." --output DECISION.json

``demo`` runs only the scripted synthetic evaluators on the synthetic fixture. There is no
command that loads a real model, contacts a SIEM or provider, trains, edits weights, or
promotes anything. Exit codes: 0 ok, 1 validation/integrity problem, 2 usage error.

``run-manifest-hash`` prints a run manifest's deterministic identifier. Record it *outside* the
run directory when the run finishes, and pass it back to ``report`` with
``--expected-run-manifest-sha256`` (one per ``--evidence`` run, in the same order; each flag
may be given once). Without it, no candidate can meet the declared criteria. A match is an
integrity check against that record, not authentication: it is only as trustworthy as
whatever protects the record. (It is unrelated to the edit-manifest hash in a ``verify-edit``
report.) If any requested run fails to load, the report cannot meet criteria or support an
approval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wsp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_eval.decision import DECISIONS, make_decision
from improve_eval.evidence import EvidenceError, load_evidence, run_manifest_sha256
from improve_eval.report import build_report, render_markdown
from improve_eval.soc import load_alerts, run_triage_evaluation
from improve_eval.study import StudyError, load_study
from improve_eval.synthetic import synthetic_evaluators
from wsp_eval.runner import RunError, _prepare_out_dir

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def _write_report(report: dict, out: Path, allow_in_repo: bool) -> Path:
    out = _prepare_out_dir(out, allow_in_repo)
    with (out / "report.json").open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (out / "report.md").open("x", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return out


class _Once(argparse.Action):
    """Store a list option, refusing a repeated flag instead of silently keeping the last."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, f"_seen_{self.dest}", False):
            parser.error(
                f"{option_string} given more than once; list every value after a single "
                f"{option_string} (earlier values would otherwise be discarded)"
            )
        setattr(namespace, f"_seen_{self.dest}", True)
        setattr(namespace, self.dest, values)


def _run_labels(runs: list[Path]) -> list[str]:
    """Shareable labels for requested runs: index plus the path relative to the runs' common
    parent, so ``x/run`` and ``y/run`` stay distinct without writing absolute local paths."""
    try:
        resolved = [r.resolve() for r in runs]
        base = Path(os.path.commonpath([r.parent for r in resolved])) if resolved else None
        rel = [os.path.relpath(r, base) for r in resolved]
    except (OSError, ValueError):  # e.g. a NUL in a path, or paths on different drives
        rel = [r.name or "?" for r in runs]
    return [f"evidence[{i}] {p}" for i, p in enumerate(rel)]


def _scrub(message: str, run: Path, label: str) -> str:
    """Replace any form of a run's local path in an error message with its shareable label."""
    forms = {str(run)}
    try:
        forms.add(str(run.resolve()))
        forms.add(str(run.absolute()))
    except (OSError, ValueError):
        pass
    for f in sorted(forms, key=len, reverse=True):
        if f and f != ".":
            message = message.replace(f, label)
    return message


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="improve", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("validate", help="validate a study and print its hash")
    sv.add_argument("study", type=Path)
    sd = sub.add_parser("demo", help="synthetic evaluation + report (fixtures only)")
    sd.add_argument("--output", type=Path, required=True)
    sd.add_argument("--operator", default="op1")
    sd.add_argument("--allow-in-repo", action="store_true")
    sr = sub.add_parser("report", help="report from verified evidence runs")
    sr.add_argument("--study", type=Path, required=True)
    sr.add_argument("--evidence", type=Path, nargs="*", default=[], action=_Once)
    sr.add_argument(
        "--expected-run-manifest-sha256",
        nargs="*",
        default=None,
        action=_Once,
        help="externally recorded run-manifest identifiers, one per --evidence run, same order",
    )
    sr.add_argument("--output", type=Path, required=True)
    sr.add_argument("--allow-in-repo", action="store_true")
    sm = sub.add_parser(
        "run-manifest-hash", help="print a run manifest's identifier, to record outside the run"
    )
    sm.add_argument("run", type=Path)
    sc = sub.add_parser("decide", help="record a human decision on a report (deploys nothing)")
    sc.add_argument("--study", type=Path, required=True)
    sc.add_argument("--report", type=Path, required=True)
    sc.add_argument("--candidate")
    sc.add_argument("--decision", choices=DECISIONS, required=True)
    sc.add_argument("--decided-by", required=True)
    sc.add_argument("--rationale", required=True)
    sc.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    try:
        if args.cmd == "validate":
            s = load_study(args.study)
            _print(
                {
                    "ok": True,
                    "study_sha256": s.sha256,
                    "frozen": s.frozen,
                    "synthetic": s.synthetic,
                    "task_kind": s.task_kind,
                    "candidates": [
                        f"{c['id']} ({c['intervention']['type']})" for c in s.candidates
                    ],
                }
            )
            return 0
        if args.cmd == "demo":
            study = load_study(FIXTURES / "synthetic_study.json")
            alerts, labels = load_alerts(FIXTURES / "synthetic_alerts.jsonl")
            run = run_triage_evaluation(
                study,
                alerts,
                labels,
                synthetic_evaluators(study),
                args.output / "evidence",
                operator=args.operator,
                allow_in_repo=args.allow_in_repo,
            )
            report = build_report(study, [load_evidence(run, study)])
            out = _write_report(report, args.output / "report", args.allow_in_repo)
            _print(
                {
                    "evidence": str(run),
                    "run_manifest_sha256": run_manifest_sha256(run),
                    "report": str(out),
                    "outcome": report["outcome"],
                }
            )
            return 0
        if args.cmd == "run-manifest-hash":
            _print(
                {
                    "run": str(args.run),
                    "run_manifest_sha256": run_manifest_sha256(args.run),
                    "note": "record this outside the run directory; it identifies the manifest, "
                    "it does not authenticate it",
                }
            )
            return 0
        if args.cmd == "report":
            study = load_study(args.study)
            expected = args.expected_run_manifest_sha256
            if expected is not None and len(expected) != len(args.evidence):
                print(
                    "error: give one --expected-run-manifest-sha256 per --evidence run",
                    file=sys.stderr,
                )
                return 2
            evidence, errors = [], []
            labels = _run_labels(args.evidence)
            for i, run in enumerate(args.evidence):
                try:
                    evidence.append(
                        load_evidence(run, study, expected[i] if expected is not None else None)
                    )
                except (EvidenceError, OSError, ValueError, RecursionError) as exc:
                    # Recorded, not dropped; labelled without absolute local paths.
                    errors.append(f"{labels[i]}: {_scrub(str(exc), run, labels[i])}")
            report = build_report(study, evidence, errors)
            out = _write_report(report, args.output, args.allow_in_repo)
            _print({"report": str(out), "outcome": report["outcome"], "evidence_errors": errors})
            return 0
        if args.cmd == "decide":
            study = load_study(args.study)
            try:
                report = json.loads(args.report.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise ValueError(f"{args.report} is not valid JSON") from exc
            except RecursionError as exc:
                raise ValueError(f"{args.report} is nested too deeply to parse") from exc
            now = datetime.now(UTC).isoformat(timespec="minutes").replace("+00:00", "Z")
            rec = make_decision(
                study,
                report,
                candidate=args.candidate,
                decision=args.decision,
                decided_by=args.decided_by,
                decided_at=now,
                rationale=args.rationale,
            )
            with args.output.open("x", encoding="utf-8") as fh:  # decisions are never overwritten
                fh.write(json.dumps(rec, indent=2, sort_keys=True) + "\n")
            _print(rec)
            return 0
    except (StudyError, EvidenceError, RunError, OSError, ValueError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

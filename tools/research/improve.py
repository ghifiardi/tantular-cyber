"""Research improvement workflow CLI (research-only; not part of the tantular-cyber package).

    python tools/research/improve.py validate tools/research/fixtures/synthetic_study.json
    python tools/research/improve.py demo --output ~/research/runs/demo-1
    python tools/research/improve.py report --study STUDY.json --evidence RUN_DIR [...] \\
        --output REPORT_DIR
    python tools/research/improve.py decide --study STUDY.json --report REPORT_DIR/report.json \\
        --decision reject --decided-by reviewer1 --rationale "..." --output DECISION.json

``demo`` runs only the scripted synthetic evaluators on the synthetic fixture. There is no
command that loads a real model, contacts a SIEM or provider, trains, edits weights, or
promotes anything. Exit codes: 0 ok, 1 validation/integrity problem, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wsp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_eval.decision import DECISIONS, make_decision
from improve_eval.evidence import EvidenceError, load_evidence
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
    sr.add_argument("--evidence", type=Path, nargs="*", default=[])
    sr.add_argument("--output", type=Path, required=True)
    sr.add_argument("--allow-in-repo", action="store_true")
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
            _print({"evidence": str(run), "report": str(out), "outcome": report["outcome"]})
            return 0
        if args.cmd == "report":
            study = load_study(args.study)
            evidence, errors = [], []
            for run in args.evidence:
                try:
                    evidence.append(load_evidence(run, study))
                except (EvidenceError, OSError, ValueError) as exc:
                    errors.append(f"{run.name}: {exc}")  # recorded in the report, not dropped
            report = build_report(study, evidence, errors)
            out = _write_report(report, args.output, args.allow_in_repo)
            _print({"report": str(out), "outcome": report["outcome"], "evidence_errors": errors})
            return 0
        if args.cmd == "decide":
            study = load_study(args.study)
            report = json.loads(args.report.read_text(encoding="utf-8"))
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
    except (StudyError, EvidenceError, RunError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

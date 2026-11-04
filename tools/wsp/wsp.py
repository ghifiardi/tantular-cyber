"""WSP research harness CLI (research-only; not part of the tantular-cyber package).

    python tools/wsp/wsp.py protocol tools/wsp/fixtures/synthetic_protocol.json
    python tools/wsp/wsp.py dataset tools/wsp/fixtures/synthetic_items.jsonl
    python tools/wsp/wsp.py ledger docs/wsp/ledger-template.json
    python tools/wsp/wsp.py diff BASE.safetensors CANDIDATE.safetensors
    python tools/wsp/wsp.py verify-edit BASE.safetensors CANDIDATE.safetensors MANIFEST.json
    python tools/wsp/wsp.py smoke --output ~/wsp-research/runs/smoke-1
    python tools/wsp/wsp.py analyze --run ~/wsp-research/runs/smoke-1 \\
        --protocol tools/wsp/fixtures/synthetic_protocol.json
    python tools/wsp/wsp.py drift --run ~/wsp-research/runs/smoke-1 \\
        --protocol tools/wsp/fixtures/synthetic_protocol.json

``smoke`` runs only the scripted synthetic arms. There is deliberately no command that
loads a real model, contacts a provider, or promotes a checkpoint. Exit codes: 0 ok,
1 validation/integrity problem, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wsp_eval.analysis import analyze, preregistration
from wsp_eval.checkpoint_diff import SafetensorsError, diff_checkpoints, verify_edit
from wsp_eval.dataset import DatasetError, check_items, load_items
from wsp_eval.drift import drift_report
from wsp_eval.grading import FinalAnswerGrader
from wsp_eval.ledger import validate_record
from wsp_eval.protocol import ProtocolError, load_protocol
from wsp_eval.runner import RunError, run_paired, verify_run
from wsp_eval.synthetic import synthetic_generators

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wsp", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("protocol", help="validate a protocol and print its hash")
    sp.add_argument("path", type=Path)
    sd = sub.add_parser("dataset", help="check item IDs, hashes, duplicates and split leakage")
    sd.add_argument("path", type=Path)
    sl = sub.add_parser("ledger", help="validate an evidence-ledger run record (JSON)")
    sl.add_argument("path", type=Path)
    sx = sub.add_parser("diff", help="tensor-level diff of two .safetensors files (read-only)")
    sx.add_argument("base", type=Path)
    sx.add_argument("candidate", type=Path)
    sv = sub.add_parser("verify-edit", help="tensor diff checked against a published manifest")
    sv.add_argument("base", type=Path)
    sv.add_argument("candidate", type=Path)
    sv.add_argument("manifest", type=Path)
    ss = sub.add_parser("smoke", help="synthetic end-to-end run + analysis + drift report")
    ss.add_argument("--output", type=Path, required=True)
    ss.add_argument("--operator", default="synthetic-operator")
    ss.add_argument("--allow-in-repo", action="store_true")
    for name, text in (
        ("analyze", "paired analysis of a run's records.jsonl"),
        ("drift", "identity + reproducibility report for no-edit reload arms"),
    ):
        sa = sub.add_parser(name, help=text)
        sa.add_argument("--run", type=Path, required=True)
        sa.add_argument("--protocol", type=Path, required=True)
    args = p.parse_args(argv)

    try:
        if args.cmd == "protocol":
            proto = load_protocol(args.path)
            _print(
                {
                    "ok": True,
                    "protocol_version": proto.protocol_version,
                    "frozen": proto.frozen,
                    "synthetic": proto.synthetic,
                    "sampling_mode": proto.sampling_mode,
                    "drift_tolerance": (proto.drift.get("tolerance") or {}).get("status"),
                    "protocol_sha256": proto.sha256,
                    "arms": [f"{a.id} ({a.role})" for a in proto.arms],
                }
            )
            return 0
        if args.cmd == "dataset":
            report = check_items(load_items(args.path))
            _print(report)
            return 0 if report["ok"] else 1
        if args.cmd == "ledger":
            problems = validate_record(json.loads(args.path.read_text(encoding="utf-8")))
            _print({"ok": not problems, "problems": problems})
            return 0 if not problems else 1
        if args.cmd == "diff":
            _print(diff_checkpoints(args.base, args.candidate))
            return 0
        if args.cmd == "verify-edit":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            report = verify_edit(diff_checkpoints(args.base, args.candidate), manifest)
            _print(report)
            return 0 if report["status"] == "verified" else 1
        if args.cmd == "smoke":
            proto = load_protocol(FIXTURES / "synthetic_protocol.json")
            items = load_items(FIXTURES / "synthetic_items.jsonl")
            result = run_paired(
                proto,
                items,
                synthetic_generators(items, proto.seeds),
                FinalAnswerGrader(),
                args.output,
                purpose="harness-smoke-synthetic",
                operator=args.operator,
                allow_in_repo=args.allow_in_repo,
            )
            out = {
                "manifest": result.manifest,
                "analysis": analyze(
                    result.records,
                    proto.analysis,
                    synthetic=proto.synthetic,
                    prereg=preregistration(proto.raw, result.manifest),
                ),
                "drift": drift_report(proto, result.manifest, result.records),
            }
            (result.out_dir / "analysis.json").write_text(
                json.dumps(out["analysis"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _print({"run_dir": str(result.out_dir), **out})
            return 0
        if args.cmd in ("analyze", "drift"):
            proto = load_protocol(args.protocol)
            manifest, records = verify_run(args.run, proto)
            if args.cmd == "drift":
                _print(drift_report(proto, manifest, records))
            else:
                synthetic = bool(manifest.get("synthetic", True))
                prereg = preregistration(proto.raw, manifest)
                _print(analyze(records, proto.analysis, synthetic=synthetic, prereg=prereg))
            return 0
    except (ProtocolError, DatasetError, RunError, SafetensorsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

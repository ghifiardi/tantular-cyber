"""Run a ``per_arm_sequential`` WSP protocol with the vLLM adapter (isolated GPU host only).

    python tools/wsp_runtime/run.py --protocol P.json \\
        --source /research/data/math_train.jsonl --shape math_train --limit 20 \\
        --leakage-reference math500=/research/data/math500_test.jsonl \\
        --checkpoint base=/research/ckpt/base --checkpoint edited=/research/ckpt/edited \\
        --checkpoint base_reload=/research/ckpt/base --split development \\
        --operator op1 --purpose smoke --output /research/runs/RUN [--keep-outputs]

The items come from a local source file through the C3 loader (``--source`` / ``--shape``),
which validates the whole file and checks it against every ``--leakage-reference`` (a source
file of the other split): any problem whose normalised text also occurs there is refused,
whatever the answers say. Every arm needs exactly one ``--checkpoint ARM=DIR`` (a local,
read-only directory; nothing is downloaded). The protocol must declare ``execution.order:
per_arm_sequential`` and complete, pinned inference and engine settings.

Held-out execution is refused while the grader is unreviewed (R9), and needs the registered
source and ID-set hashes. After the run, ``runtime_report.json`` is written next to the manifest:
dataset and leakage provenance, grader dependency versions and every engine's close report, with
the manifest's SHA-256 (the manifest does not cover this file; this file names the manifest).
The report is supplementary evidence: a decision relying on it must verify a separately
protected identifier for it, since naming the manifest does not protect the report itself.

This command needs vLLM, transformers, torch and math-verify, which are installed only on the
isolated host. It has been exercised only with stub engines (``tests/test_wsp_runtime.py``).
Exit codes: 0 ok, 1 run or validation failure (including any error record), 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wsp"))
sys.path.insert(0, str(HERE))

from wsp_eval.dataset import DatasetError  # noqa: E402
from wsp_eval.hashing import sha256_file  # noqa: E402
from wsp_eval.protocol import ProtocolError, load_protocol  # noqa: E402
from wsp_eval.runner import RunError, run_paired  # noqa: E402
from wsp_runtime import RUNTIME_VERSION, datasets  # noqa: E402
from wsp_runtime.grader import MathQAGrader, MathVerifyBackend, check_dependencies  # noqa: E402
from wsp_runtime.provenance import runtime_code_sha256  # noqa: E402
from wsp_runtime.vllm_adapter import VllmBackend, arm_loader  # noqa: E402

REPORT_FILE = "runtime_report.json"


def _pairs(values: Sequence[str], flag: str) -> list[tuple[str, Path]]:
    out = []
    for value in values:
        key, sep, path = value.partition("=")
        if not sep or not key or not path:
            raise ValueError(f"{flag} must be KEY=PATH (got {value!r})")
        out.append((key, Path(path)))
    return out


def _checkpoints(values: Sequence[str], arm_ids: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for arm, path in _pairs(values, "--checkpoint"):
        if arm in out:
            raise ValueError(f"--checkpoint given twice for arm {arm!r}")
        out[arm] = path
    if set(out) != set(arm_ids):
        raise ValueError(f"--checkpoint must name exactly the protocol arms {sorted(arm_ids)}")
    return out


def run(
    argv: Sequence[str],
    *,
    backend_factory: Callable[[], Any] = VllmBackend.load,
    verify_backend_factory: Callable[[], Any] = MathVerifyBackend.load,
    dependency_lookup: Callable[[str], str] = metadata.version,
) -> dict[str, Any]:
    p = argparse.ArgumentParser(prog="wsp-runtime-run", description=__doc__.split("\n\n")[0])
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True, help="local MATH / MATH-500 JSONL")
    p.add_argument("--shape", choices=sorted(datasets.SHAPES), required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--expected-count", type=int, help="rows in the whole source file")
    p.add_argument("--expected-source-sha256")
    p.add_argument("--expected-ids-sha256", help="ID set of the whole source file")
    p.add_argument(
        "--leakage-reference",
        action="append",
        default=[],
        metavar="SHAPE=PATH",
        help="source file of the other split to check against (at least one)",
    )
    p.add_argument("--checkpoint", action="append", default=[], metavar="ARM=DIR")
    p.add_argument("--split", choices=("development", "heldout"), required=True)
    p.add_argument("--operator", required=True)
    p.add_argument("--purpose", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--heldout-authorized-by")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument(
        "--max-parent-torch-memory-after-close",
        type=int,
        default=None,
        help=(
            "bytes; fail the close if this process's PyTorch allocator still holds more. This "
            "does not measure vLLM worker processes and never shows the GPU was released"
        ),
    )
    args = p.parse_args(list(argv))

    protocol = load_protocol(args.protocol)
    if protocol.execution_order != "per_arm_sequential":
        raise ValueError("this runner needs a protocol with execution.order per_arm_sequential")
    if protocol.synthetic:
        raise ValueError("this runner is for real (non-synthetic) protocols; use wsp.py smoke")
    if args.split == "heldout":
        if not MathQAGrader.reviewed:
            raise ValueError(
                "held-out execution is refused while the grader is unreviewed (R9: compare "
                "with the released mathqa.py and obtain the independent evaluator's acceptance)"
            )
        if not (args.expected_source_sha256 and args.expected_ids_sha256):
            raise ValueError(
                "held-out execution needs --expected-source-sha256 and --expected-ids-sha256"
            )

    items, dataset_report = datasets.load(
        args.source,
        shape=args.shape,
        split=args.split,
        expected_count=args.expected_count,
        expected_source_sha256=args.expected_source_sha256,
        expected_ids_sha256=args.expected_ids_sha256,
        limit=args.limit,
    )
    references = _pairs(args.leakage_reference, "--leakage-reference")
    if not references:
        raise ValueError("at least one --leakage-reference (the other split's source) is required")
    ref_items: list[Any] = []
    ref_reports = []
    for shape, path in references:
        if shape not in datasets.SHAPES:
            raise ValueError(f"--leakage-reference shape must be one of {sorted(datasets.SHAPES)}")
        ref_split = datasets.SHAPES[shape]
        if ref_split == args.split:
            raise ValueError(f"--leakage-reference {shape} is the run's own split ({ref_split})")
        loaded, ref_report = datasets.load(path, shape=shape, split=ref_split)
        ref_items += loaded
        ref_reports.append(ref_report.as_dict())
    leakage = datasets.check_disjoint(items, ref_items)
    grader_dependencies = check_dependencies(
        protocol.grader.get("dependencies"), lookup=dependency_lookup
    )

    dirs = _checkpoints(args.checkpoint, [a.id for a in protocol.arms])
    reports: list[dict[str, Any]] = []
    loaders = {
        a.id: arm_loader(
            a,
            dirs[a.id],
            backend_factory,
            max_parent_memory_after_close=args.max_parent_torch_memory_after_close,
            reports=reports,
        )
        for a in protocol.arms
    }
    grader = MathQAGrader(verify_backend_factory())
    runtime_report: dict[str, Any] = {
        "runtime_version": RUNTIME_VERSION,
        "runtime_code_sha256": runtime_code_sha256(),
        "dataset": dataset_report.as_dict(),
        "leakage_references": ref_reports,
        "leakage_check": leakage,
        "grader": {
            "id": grader.id,
            "revision": grader.revision,
            "reviewed": grader.reviewed,
            "dependencies": grader_dependencies,
        },
        "close_reports": reports,
    }
    summary: dict[str, Any] = {"close_reports": reports}
    try:
        result = run_paired(
            protocol,
            items,
            None,
            grader,
            args.output,
            loaders=loaders,
            split=args.split,
            purpose=args.purpose,
            operator=args.operator,
            heldout_authorized_by=args.heldout_authorized_by,
            keep_outputs=args.keep_outputs,
        )
    except Exception as exc:  # interrupts propagate; the runner already wrote the manifest
        summary.update(run_status="aborted", error=f"{type(exc).__name__}: {exc}"[:300])
        summary["runtime_report"] = _write_report(args.output, runtime_report, dataset_report)
        raise _Aborted(summary) from exc
    counts = result.manifest["counts"]
    summary.update(
        run_status=result.manifest["run_status"],
        run_dir=str(result.out_dir),
        counts=counts,
        error_records=counts.get("error", 0),
        runtime_report=_write_report(result.out_dir, runtime_report, dataset_report),
    )
    return summary


def _write_report(out: Path, report: dict[str, Any], dataset_report: Any) -> str | None:
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        return None  # the run failed before the runner wrote anything
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {
        **report,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_items_sha256_matches": (
            manifest.get("dataset", {}).get("items_file_sha256") == dataset_report.items_sha256
        ),
    }
    with (out / REPORT_FILE).open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return REPORT_FILE


class _Aborted(Exception):
    def __init__(self, summary: dict[str, Any]):
        super().__init__(summary.get("error"))
        self.summary = summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = run(sys.argv[1:] if argv is None else argv)
    except _Aborted as exc:
        print(json.dumps(exc.summary, indent=2, sort_keys=True))
        return 1
    except (ProtocolError, DatasetError, RunError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["error_records"]:
        print(f"error: {summary['error_records']} records are errors", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

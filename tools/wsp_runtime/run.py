"""Run a ``per_arm_sequential`` WSP protocol with the vLLM adapter (isolated GPU host only).

    python tools/wsp_runtime/run.py --protocol P.json --items ITEMS.jsonl \\
        --checkpoint base=/research/ckpt/base --checkpoint edited=/research/ckpt/edited \\
        --checkpoint base_reload=/research/ckpt/base --split development \\
        --operator op1 --purpose smoke --output /research/runs/RUN [--keep-outputs]

Every arm needs exactly one ``--checkpoint ARM=DIR`` (a local directory; nothing is
downloaded). The protocol must declare ``execution.order: per_arm_sequential``, a pinned engine
version and numeric engine settings. After the run the engines' close reports (shutdown steps,
memory still allocated, whether checkpoint files changed) are printed with the run summary.

This command needs vLLM, torch and math-verify, which are installed only on the isolated host.
It has been exercised only with stub engines (``tests/test_wsp_runtime.py``). Exit codes:
0 ok, 1 run or validation failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wsp"))
sys.path.insert(0, str(HERE))

from wsp_eval.dataset import DatasetError, load_items  # noqa: E402
from wsp_eval.protocol import ProtocolError, load_protocol  # noqa: E402
from wsp_eval.runner import RunError, run_paired  # noqa: E402
from wsp_runtime.grader import MathQAGrader, MathVerifyBackend  # noqa: E402
from wsp_runtime.vllm_adapter import VllmBackend, arm_loader  # noqa: E402


def _checkpoints(values: Sequence[str], arm_ids: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        arm, sep, path = value.partition("=")
        if not sep or not arm or not path:
            raise ValueError(f"--checkpoint must be ARM=DIR (got {value!r})")
        if arm in out:
            raise ValueError(f"--checkpoint given twice for arm {arm!r}")
        out[arm] = Path(path)
    if set(out) != set(arm_ids):
        raise ValueError(f"--checkpoint must name exactly the protocol arms {sorted(arm_ids)}")
    return out


def run(
    argv: Sequence[str],
    *,
    backend_factory: Callable[[], Any] = VllmBackend.load,
    verify_backend_factory: Callable[[], Any] = MathVerifyBackend.load,
) -> dict[str, Any]:
    p = argparse.ArgumentParser(prog="wsp-runtime-run", description=__doc__.split("\n\n")[0])
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", default=[], metavar="ARM=DIR")
    p.add_argument("--split", choices=("development", "heldout"), required=True)
    p.add_argument("--operator", required=True)
    p.add_argument("--purpose", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--heldout-authorized-by")
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument(
        "--max-gpu-memory-after-close",
        type=int,
        default=None,
        help="bytes; fail the close (and stop before the next arm) if more is still allocated",
    )
    args = p.parse_args(list(argv))

    protocol = load_protocol(args.protocol)
    if protocol.execution_order != "per_arm_sequential":
        raise ValueError("this runner needs a protocol with execution.order per_arm_sequential")
    if protocol.synthetic:
        raise ValueError("this runner is for real (non-synthetic) protocols; use wsp.py smoke")
    items = load_items(args.items)
    dirs = _checkpoints(args.checkpoint, [a.id for a in protocol.arms])
    reports: list[dict[str, Any]] = []
    loaders = {
        a.id: arm_loader(
            a,
            dirs[a.id],
            backend_factory,
            max_memory_after_close=args.max_gpu_memory_after_close,
            reports=reports,
        )
        for a in protocol.arms
    }
    grader = MathQAGrader(verify_backend_factory())
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
        raise _Aborted(summary) from exc
    summary.update(
        run_status=result.manifest["run_status"],
        run_dir=str(result.out_dir),
        counts=result.manifest["counts"],
    )
    return summary


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

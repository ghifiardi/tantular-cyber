"""Run a ``per_arm_sequential`` WSP protocol with the vLLM adapter (isolated GPU host only).

    python tools/wsp_runtime/run.py --protocol P.json \\
        --source /research/data/math_train.jsonl \\
        --leakage-reference /research/data/math500_test.jsonl \\
        --checkpoint base=/research/ckpt/base --checkpoint edited=/research/ckpt/edited \\
        --checkpoint base_reload=/research/ckpt/base --split development \\
        --operator op1 --purpose smoke --output /research/runs/RUN [--keep-outputs]

Before any arm runs or any output is written, this command:

* forces the offline and no-telemetry variables (a conflicting value is refused);
* loads the whole ``--source`` (the run's split) and the whole ``--leakage-reference`` (the
  other split) against the protocol's ``dataset.runtime_sources`` registration: raw file
  SHA-256, row count, whole-file ID-set hash; refuses any problem whose normalised text occurs
  in both, whatever the answers say; and only then applies each file's registered selection
  rule and checks its registered converted-items hash (both files, not only the source);
* checks the grader's dependency versions and reconciles both version sources of vLLM and
  transformers with the protocol: the package metadata must equal the pins, and the modules'
  ``__version__`` must equal the pins or an explicitly registered module version (nothing is
  stripped or normalised);
* builds and reads back a ``SamplingParams`` object for every (n, seed) the run requests;
* checks every arm's settings, storage (read-only mount below root-owned, non-writable parents)
  and loaded-file inventory.

What needs a loaded engine (observed engine settings, the tokenizer, the effective stop set) is
checked or recorded at load time; the effective engine configuration remains a GPU-host check.
Every arm needs exactly one ``--checkpoint ARM=DIR``. Held-out execution is refused while the
grader is unreviewed (R9).

After the run (or an abort that left a manifest), ``runtime_report.json`` is written next to the
manifest only if that manifest carries this invocation's run ID. It holds the dataset and leakage
provenance, the hash definitions, forced variables, installed versions, grader dependency
versions, every arm's preflight result and close report, and the manifest's SHA-256. Its own
SHA-256 is printed in the summary so it can be recorded externally. The report is
**supplementary evidence**: printing its hash does not protect it, and a decision relying on it
must verify a separately protected record of that hash.

This command needs vLLM, transformers, torch and math-verify, which are installed only on the
isolated host. It has been exercised only with stand-in modules (``tests/test_wsp_runtime.py``),
which validate control flow, not compatibility with the real libraries. Exit codes: 0 ok, 1 run
or validation failure (including any error record or a report that could not be written), 2
usage error. An interrupt (Ctrl-C) or other ``BaseException`` during the run propagates (the
same exception object) after a best-effort ``runtime_report.json`` (marked ``interrupted``) is
written atomically next to the runner's aborted manifest; a second interrupt during that write
is dropped in favour of the first, and leaves no partial report. No summary is printed and the
exit status is the interpreter's. Nothing is written for an interrupt before the run starts,
or one that arrives inside the runner before its abort handling (the output directory may then
exist without a manifest, and the report is skipped).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import uuid
from collections.abc import Callable, MutableMapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wsp"))
sys.path.insert(0, str(HERE))

from wsp_eval.dataset import DatasetError  # noqa: E402
from wsp_eval.hashing import sha256_bytes, sha256_file  # noqa: E402
from wsp_eval.protocol import ProtocolError, load_protocol  # noqa: E402
from wsp_eval.runner import RunError, run_paired  # noqa: E402
from wsp_runtime import RUNTIME_VERSION, datasets  # noqa: E402
from wsp_runtime.grader import MathQAGrader, MathVerifyBackend, check_dependencies  # noqa: E402
from wsp_runtime.provenance import runtime_code_sha256  # noqa: E402
from wsp_runtime.vllm_adapter import (  # noqa: E402
    ProvenanceError,
    VllmBackend,
    arm_loader,
    force_offline,
    preflight,
    verify_storage,
)

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


def _installed(lookup: Callable[[str], str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for package in ("vllm", "transformers"):
        try:
            out[package] = lookup(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def _module_versions() -> dict[str, str | None]:
    # Imported here only (no GPU needed), to reconcile with the package metadata in preflight.
    import transformers
    import vllm

    return {
        "vllm": getattr(vllm, "__version__", None),
        "transformers": getattr(transformers, "__version__", None),
    }


def _vllm_sampling_params() -> Callable[..., Any]:
    # Imported here only (no GPU needed); the object is built and read back in preflight.
    from vllm import SamplingParams

    return SamplingParams


def run(
    argv: Sequence[str],
    *,
    backend_factory: Callable[[], Any] = VllmBackend.load,
    verify_backend_factory: Callable[[], Any] = MathVerifyBackend.load,
    dependency_lookup: Callable[[str], str] = metadata.version,
    storage_check: Callable[[Path], dict] = verify_storage,
    sampling_params_factory: Callable[[], Callable[..., Any]] = _vllm_sampling_params,
    module_versions: Callable[[], dict[str, str | None]] = _module_versions,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    p = argparse.ArgumentParser(prog="wsp-runtime-run", description=__doc__.split("\n\n")[0])
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="local JSONL for the run's split, registered in dataset.runtime_sources",
    )
    p.add_argument(
        "--leakage-reference",
        type=Path,
        required=True,
        help="local JSONL for the other split, registered in dataset.runtime_sources",
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
            "bytes; fail the close if this process's PyTorch allocator still has more "
            "allocated or reserved. This does not measure vLLM worker processes and never "
            "shows the GPU was released"
        ),
    )
    args = p.parse_args(list(argv))

    # Environment first: a conflicting offline or telemetry variable is refused before
    # anything is read or written, and the forced values apply to every later import.
    offline = force_offline(os.environ if environ is None else environ)
    protocol = load_protocol(args.protocol)
    if protocol.execution_order != "per_arm_sequential":
        raise ValueError("this runner needs a protocol with execution.order per_arm_sequential")
    if protocol.synthetic:
        raise ValueError("this runner is for real (non-synthetic) protocols; use wsp.py smoke")
    if args.split == "heldout" and not MathQAGrader.reviewed:
        raise ValueError(
            "held-out execution is refused while the grader is unreviewed (R9: compare "
            "with the released mathqa.py and obtain the independent evaluator's acceptance)"
        )

    # Dataset identity comes from the protocol (dataset.runtime_sources): raw file hash, row
    # count and ID-set hash of the whole file, the selection rule and, for held-out, the
    # converted items' hash. The whole source is leak-checked against the whole reference
    # before the selection rule is applied.
    other = "heldout" if args.split == "development" else "development"
    source_items, source_report = datasets.load_registered(
        args.source, protocol.dataset, args.split
    )
    ref_items, ref_report = datasets.load_registered(
        args.leakage_reference, protocol.dataset, other
    )
    leakage = datasets.check_disjoint(source_items, ref_items)
    items, dataset_report = datasets.select_registered(
        source_items, source_report, protocol.dataset, args.split
    )
    # The reference's registered selection and converted-items hash are checked too, so a wrong
    # registration of the other split is found now, not at that split's first run.
    _, ref_selected = datasets.select_registered(ref_items, ref_report, protocol.dataset, other)
    grader_dependencies = check_dependencies(
        protocol.grader.get("dependencies"), lookup=dependency_lookup
    )

    # Every arm, before any arm runs: settings, installed versions, the read-back of a
    # SamplingParams object for every (n, seed) the run requests, storage and inventory.
    if protocol.sampling_mode == "n_per_request":
        shapes = [(protocol.samples_per_item, protocol.seeds[0])]
    else:
        shapes = [(1, seed) for seed in protocol.seeds]
    installed = _installed(dependency_lookup)
    modules = module_versions()
    make_params = sampling_params_factory()
    dirs = _checkpoints(args.checkpoint, [a.id for a in protocol.arms])
    preflights = [
        preflight(
            a,
            dirs[a.id],
            storage_check=storage_check,
            installed=installed,
            modules=modules,
            sampling_params=make_params,
            sample_shapes=shapes,
        )
        for a in protocol.arms
    ]

    reports: list[dict[str, Any]] = []
    loaders = {
        a.id: arm_loader(
            a,
            dirs[a.id],
            backend_factory,
            max_parent_memory_after_close=args.max_parent_torch_memory_after_close,
            reports=reports,
            storage_check=storage_check,
        )
        for a in protocol.arms
    }
    grader = MathQAGrader(verify_backend_factory())
    run_id = f"wsp-{uuid.uuid4().hex[:12]}"
    runtime_report: dict[str, Any] = {
        "run_id": run_id,
        "runtime_version": RUNTIME_VERSION,
        "runtime_code_sha256": runtime_code_sha256(),
        "dataset": dataset_report.as_dict(),
        "leakage_reference": ref_report.as_dict(),
        "leakage_check": leakage,
        "hash_definitions": dict(datasets.HASH_DEFINITION_TEXT),
        "forced_environment": offline,
        "installed_versions": installed,
        "module_versions": modules,
        "leakage_reference_selection": ref_selected.as_dict(),
        "grader": {
            "id": grader.id,
            "revision": grader.revision,
            "reviewed": grader.reviewed,
            "dependencies": grader_dependencies,
        },
        "preflight": preflights,
        "close_reports": reports,
    }
    out = args.output.expanduser()
    summary: dict[str, Any] = {"run_id": run_id, "close_reports": reports}
    try:
        result = run_paired(
            protocol,
            items,
            None,
            grader,
            out,
            loaders=loaders,
            split=args.split,
            purpose=args.purpose,
            operator=args.operator,
            heldout_authorized_by=args.heldout_authorized_by,
            keep_outputs=args.keep_outputs,
            run_id=run_id,
        )
    except Exception as exc:  # the runner already wrote the aborted manifest
        summary.update(run_status="aborted", error=f"{type(exc).__name__}: {exc}"[:300])
        _report_into(summary, out, run_id, runtime_report, dataset_report)
        raise _Aborted(summary) from exc
    except BaseException as exc:
        # An interrupt: the runner already wrote the aborted manifest and records; write the
        # report (with the close reports so far) best-effort, then let the interrupt propagate
        # unchanged. No summary is printed.
        summary.update(run_status="interrupted", error=f"{type(exc).__name__}: {exc}"[:300])
        runtime_report["interrupted"] = True
        # A second interrupt during the write is dropped: the first interrupt is the one
        # propagated, and the atomic write leaves either the whole report or none.
        with contextlib.suppress(BaseException):
            _report_into(summary, out, run_id, runtime_report, dataset_report)
        raise exc
    counts = result.manifest["counts"]
    summary.update(
        run_status=result.manifest["run_status"],
        run_dir=str(result.out_dir),
        counts=counts,
        error_records=counts.get("error", 0),
    )
    _report_into(summary, result.out_dir, run_id, runtime_report, dataset_report)
    return summary


def _report_into(
    summary: dict[str, Any],
    out: Path,
    run_id: str,
    report: dict[str, Any],
    dataset_report: Any,
) -> None:
    """Write the supplementary report and record the outcome in ``summary``; never raises, so
    a failure to write it cannot replace the run's own failure."""
    try:
        summary.update(_write_report(out, run_id, report, dataset_report))
    except Exception as exc:
        summary.update(runtime_report=None, runtime_report_error=_short(exc))


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _write_report(
    out: Path, run_id: str, report: dict[str, Any], dataset_report: Any
) -> dict[str, Any]:
    out = out.expanduser().resolve()
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        return {"runtime_report": None, "runtime_report_skipped": "no manifest from this run"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id:
        # Another run's directory (for example a rerun into a used --output): never touch it.
        return {
            "runtime_report": None,
            "runtime_report_skipped": "the manifest there belongs to another run",
        }
    body = {
        **report,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_items_sha256_matches": (
            manifest.get("dataset", {}).get("items_file_sha256") == dataset_report.items_sha256
        ),
    }
    data = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_atomically(out / REPORT_FILE, data)
    return {"runtime_report": REPORT_FILE, "runtime_report_sha256": sha256_bytes(data)}


def _write_atomically(path: Path, data: bytes) -> None:
    """All or nothing, never overwriting: write a temporary file in the same directory, flush
    it to disk, then hard-link it to ``path`` (which fails if ``path`` exists) and remove the
    temporary name. An interruption leaves either no report or the complete one."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
    except (ProtocolError, DatasetError, RunError, ProvenanceError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("error_records"):
        print(f"error: {summary['error_records']} records are errors", file=sys.stderr)
        return 1
    if summary.get("runtime_report_error"):
        print(
            f"error: runtime report not written: {summary['runtime_report_error']}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

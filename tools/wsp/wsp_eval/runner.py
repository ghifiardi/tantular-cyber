"""Paired arm execution with explicit failures and per-item export.

Every (item, sample, arm) request produces exactly one record, whatever happens: graded,
ungraded (grader could not decide) or error (adapter raised, returned something that is not a
:class:`Generation`, reported an abnormal or self-contradictory finish, returned the wrong
number of samples, or the grader itself raised). Arms are interleaved per item
and sample in protocol order, with the protocol's seed for that sample, so arms see the same
prompts under the same seeds. Raw outputs are hashed; their text is written only when
``keep_outputs`` is set, because model output belongs in access-controlled research storage.

Outputs go to a new or empty directory (runs are immutable) that is, by default, outside
this repository checkout so research artifacts are not committed by accident.

Provenance: the manifest records the SHA-256 of ``records.jsonl`` (and ``outputs.jsonl``) and a
``run_status``. If the harness itself fails mid-run (a bug, an interrupt), the records collected
so far and a manifest with ``run_status: "aborted"`` are still written before the exception
propagates, so a partial run is visible rather than silently missing. :func:`verify_run` checks a
run directory against its protocol before anything is analyzed.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import HARNESS_VERSION
from .dataset import Item, check_items, dataset_sha256
from .generation import Generation, GenerationRequest, Generator
from .grading import Grader
from .hashing import harness_code_sha256, sha256_bytes, sha256_file
from .protocol import Protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
_MAX_ERROR_CHARS = 200


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunResult:
    run_id: str
    out_dir: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _prepare_out_dir(out_dir: Path, allow_in_repo: bool) -> Path:
    out = out_dir.expanduser().resolve()
    if not allow_in_repo and (out == REPO_ROOT or REPO_ROOT in out.parents):
        raise RunError(
            f"output directory {out} is inside the repository checkout; use research storage "
            "outside the repo (or allow_in_repo for throwaway synthetic runs)"
        )
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise RunError(f"output directory {out} is not empty; runs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_paired(
    protocol: Protocol,
    items: Sequence[Item],
    generators: Mapping[str, Generator],
    grader: Grader,
    out_dir: Path,
    *,
    split: str = "development",
    purpose: str,
    operator: str,
    heldout_authorized_by: str | None = None,
    keep_outputs: bool = False,
    allow_in_repo: bool = False,
    clock: Callable[[], float] = time.monotonic,
    run_id: str | None = None,
) -> RunResult:
    if split == "heldout":
        if not protocol.frozen:
            raise RunError("held-out evaluation requires a frozen protocol")
        if not heldout_authorized_by:
            raise RunError("held-out evaluation requires heldout_authorized_by (evaluator)")
    if protocol.synthetic is False and any(
        g.describe().get("synthetic") for g in generators.values()
    ):
        raise RunError("synthetic generators may only be used with a synthetic protocol")
    missing = [a.id for a in protocol.arms if a.id not in generators]
    extra = sorted(set(generators) - {a.id for a in protocol.arms})
    if missing or extra:
        raise RunError(f"generators must match protocol arms (missing {missing}, extra {extra})")
    if grader.id != protocol.grader.get("id") or grader.revision != protocol.grader.get("revision"):
        raise RunError("grader id/revision differ from the protocol")

    check = check_items(list(items))
    if not check["ok"]:
        raise RunError(f"dataset integrity check failed: {check['errors']}")
    selected = sorted((i for i in items if i.split == split), key=lambda i: i.id)
    if not selected:
        raise RunError(f"no items in split {split!r}")

    out = _prepare_out_dir(out_dir, allow_in_repo)
    run_id = run_id or f"wsp-{uuid.uuid4().hex[:12]}"
    started = _now()
    records: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    def record(item: Item, arm: Any, sample: int, seed: int, n: int) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "arm": arm.id,
            "role": arm.role,
            "item_id": item.id,
            "split": item.split,
            "item_sha256": item.content_sha256,
            "sample": sample,
            "seed": seed,
            "request_n": n,
        }

    def failed(rec: dict[str, Any], error_class: str, message: str, latency: float) -> None:
        rec.update(
            status="error",
            correct=None,
            grade_detail=None,
            error_class=error_class,
            error_message=message[:_MAX_ERROR_CHARS],
            hit_token_limit=None,
            output_tokens=None,
            output_sha256=None,
            output_chars=None,
            finish_reason=None,
            rendered_prompt_sha256=None,
            latency_s=latency,
        )

    def execute(item: Item, arm: Any, samples: list[tuple[int, int]]) -> None:
        n = len(samples)
        seed = samples[0][1]
        req = GenerationRequest(arm.id, item.id, item.prompt, dict(arm.inference), seed, n)
        recs = [record(item, arm, k, s, n) for k, s in samples]
        t0 = clock()
        try:
            result = generators[arm.id].generate(req)
        except Exception as exc:  # recorded, never dropped: one error row per requested sample
            latency = round(clock() - t0, 6)
            for rec in recs:
                failed(rec, type(exc).__name__, str(exc), latency)
            records.extend(recs)
            return
        latency = round(clock() - t0, 6)
        gens = result if isinstance(result, list) else [result]
        if len(gens) != n:
            for rec in recs:
                failed(rec, "SampleCountMismatch", f"requested {n}, got {len(gens)}", latency)
            records.extend(recs)
            return
        for rec, gen in zip(recs, gens, strict=True):
            if not isinstance(gen, Generation) or not isinstance(gen.text, str):
                failed(rec, "InvalidGeneration", f"adapter returned {type(gen).__name__}", latency)
                records.append(rec)
                continue
            if gen.finish_reason is not None and gen.finish_reason not in ("stop", "length"):
                # The authors' evaluate.py aborts the job here; this harness records it instead.
                failed(rec, "AbnormalFinish", f"finish_reason={gen.finish_reason}", latency)
                _keep_reported(rec, gen)
                records.append(rec)
                continue
            limit = gen.hit_token_limit
            if limit is None and gen.finish_reason is not None:
                limit = gen.finish_reason == "length"
            elif gen.finish_reason is not None and limit != (gen.finish_reason == "length"):
                # A token-limit flag that contradicts the finish reason means the adapter's
                # accounting cannot be trusted for this sample.
                failed(
                    rec,
                    "InconsistentFinish",
                    f"hit_token_limit={limit} with finish_reason={gen.finish_reason}",
                    latency,
                )
                _keep_reported(rec, gen)
                records.append(rec)
                continue
            out_fields = {
                "hit_token_limit": limit,
                "output_tokens": gen.output_tokens,
                "output_sha256": sha256_bytes(gen.text.encode("utf-8")),
                "output_chars": len(gen.text),
                "finish_reason": gen.finish_reason,
                "rendered_prompt_sha256": gen.rendered_prompt_sha256,
                "latency_s": latency,
            }
            try:
                grade = grader.grade(item, gen.text)
            except Exception as exc:  # a grader crash is a harness failure: loud, per sample
                rec.update(
                    status="error",
                    correct=None,
                    grade_detail=None,
                    error_class=f"GraderError:{type(exc).__name__}",
                    error_message=str(exc)[:_MAX_ERROR_CHARS],
                    **out_fields,
                )
            else:
                rec.update(
                    status="ungraded" if grade.correct is None else "graded",
                    correct=grade.correct,
                    grade_detail=grade.detail,
                    error_class=None,
                    error_message=None,
                    **out_fields,
                )
            records.append(rec)
            if keep_outputs:
                outputs.append(
                    {"arm": arm.id, "item_id": item.id, "sample": rec["sample"], "text": gen.text}
                )

    abort: BaseException | None = None
    try:
        for item in selected:
            if protocol.sampling_mode == "n_per_request":
                batch = [(k, protocol.seeds[0]) for k in range(protocol.samples_per_item)]
                for arm in protocol.arms:
                    execute(item, arm, batch)
            else:
                for sample, seed in enumerate(protocol.seeds):
                    for arm in protocol.arms:
                        execute(item, arm, [(sample, seed)])
    except BaseException as exc:  # harness failure: write what exists, then re-raise
        abort = exc

    expected = len(selected) * protocol.samples_per_item * len(protocol.arms)
    status = Counter(r["status"] for r in records)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "run_status": "aborted" if abort is not None else "complete",
        "abort_reason": None if abort is None else type(abort).__name__,
        "harness_version": HARNESS_VERSION,
        "harness_code_sha256": harness_code_sha256(),
        "purpose": purpose,
        "operator": operator,
        "started_at": started,
        "finished_at": _now(),
        "synthetic": protocol.synthetic,
        "protocol_version": protocol.protocol_version,
        "protocol_sha256": protocol.sha256,
        "protocol_frozen": protocol.frozen,
        "split": split,
        "heldout_authorized_by": heldout_authorized_by,
        "dataset": {
            **protocol.dataset,
            "items_file_sha256": dataset_sha256(list(items)),
            "n_selected": len(selected),
            "within_split_duplicates": check["within_split_duplicates"],
        },
        "grader": {"id": grader.id, "revision": grader.revision},
        "arms": [
            {
                "id": a.id,
                "role": a.role,
                "checkpoint": a.checkpoint,
                "inference": a.inference,
                "engine": a.engine,
                "generator": generators[a.id].describe(),
            }
            for a in protocol.arms
        ],
        "seeds": list(protocol.seeds),
        "sampling_mode": protocol.sampling_mode,
        "samples_per_item": protocol.samples_per_item,
        "drift_policy": protocol.drift,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "counts": {
            "records": len(records),
            "expected_records": expected,
            **dict(sorted(status.items())),
        },
        "outputs_kept": keep_outputs,
        "human_decision": "pending",
    }

    _write_jsonl(out / "records.jsonl", records)
    manifest["records_sha256"] = sha256_file(out / "records.jsonl")
    manifest["outputs_sha256"] = None
    if keep_outputs:
        _write_jsonl(out / "outputs.jsonl", outputs)
        manifest["outputs_sha256"] = sha256_file(out / "outputs.jsonl")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if abort is not None:
        raise abort
    if len(records) != expected:  # defensive: execute() always writes one row per sample
        raise RunError(f"run wrote {len(records)} records, expected {expected}")
    return RunResult(run_id, out, manifest, records)


def _keep_reported(rec: dict[str, Any], gen: Generation) -> None:
    # The engine did answer: keep what the adapter reported about the request, so the drift
    # identity check can still compare the prompt the engine received. Output fields stay
    # null because the sample is not scored.
    rec["finish_reason"] = gen.finish_reason
    rec["rendered_prompt_sha256"] = gen.rendered_prompt_sha256


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_records(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def verify_run(run_dir: Path, protocol: Protocol) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a run directory and check that it belongs to ``protocol`` and is intact.

    Raises :class:`RunError` if the protocol hash differs from the manifest, the run was
    aborted, ``records.jsonl`` does not match the hash in the manifest, a record carries a
    different run id, or the record count differs from what the manifest says was expected.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != protocol.sha256:
        raise RunError("protocol hash differs from the run manifest")
    if manifest.get("run_status") != "complete":
        raise RunError(
            f"run_status is {manifest.get('run_status')!r} "
            f"({manifest.get('abort_reason')}); partial runs are not analyzed"
        )
    path = run_dir / "records.jsonl"
    if sha256_file(path) != manifest.get("records_sha256"):
        raise RunError("records.jsonl does not match records_sha256 in the manifest")
    records = load_records(path)
    if any(r.get("run_id") != manifest.get("run_id") for r in records):
        raise RunError("records.jsonl contains records from another run")
    if len(records) != manifest.get("counts", {}).get("expected_records"):
        raise RunError("record count differs from expected_records in the manifest")
    return manifest, records

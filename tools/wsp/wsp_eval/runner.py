"""Paired arm execution with explicit failures and per-item export.

Every (item, sample, arm) request produces exactly one record, whatever happens: graded,
ungraded (grader could not decide) or error (adapter raised, returned something that is not a
:class:`Generation`, reported an abnormal or self-contradictory finish, returned the wrong
number of samples, or the grader itself raised). Arms see the same prompts under the same
seeds. Two execution orders exist (``protocol.execution_order``):

* ``interleaved`` (default): every arm's generator is loaded up front and requests are
  interleaved per item and sample in protocol order.
* ``per_arm_sequential``: one arm at a time, in protocol order. Each arm is loaded by its own
  loader (a separate load with a distinct ``load_id``, checked), its items are sent in
  ``protocol.shards`` batches (shard ``k`` holds the selected items ``[k::shards]`` in dataset
  file order, like the authors' ``rows[i::10]``), and the generator is closed before the next
  arm is loaded. Each record keeps its own row and also carries ``shard`` and ``batch_size``;
  its ``latency_s`` is the wall time of the batch call it was part of. A batch that raises, or
  returns the wrong number of entries, produces error rows for every request in it; a load,
  load-identity or close failure aborts the run.

Raw outputs are hashed; their text is written only when ``keep_outputs`` is set, because model
output belongs in access-controlled research storage.

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
from .generation import ArmLoader, BatchItem, Generation, GenerationRequest, Generator
from .grading import Grader
from .hashing import harness_code_sha256, sha256_bytes, sha256_file, sha256_json
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
    generators: Mapping[str, Generator] | None,
    grader: Grader,
    out_dir: Path,
    *,
    loaders: Mapping[str, ArmLoader] | None = None,
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
    sequential = protocol.execution_order == "per_arm_sequential"
    if sequential:
        if generators is not None or loaders is None:
            raise RunError(
                "a per_arm_sequential protocol takes loaders (one per arm), not generators"
            )
        given: Mapping[str, Any] = loaders
        if not all(callable(f) for f in loaders.values()):
            raise RunError("every loader must be callable and return a freshly loaded generator")
    else:
        if loaders is not None or generators is None:
            raise RunError("an interleaved protocol takes loaded generators, not loaders")
        given = generators
        if protocol.synthetic is False and any(
            g.describe().get("synthetic") for g in generators.values()
        ):
            raise RunError("synthetic generators may only be used with a synthetic protocol")
    missing = [a.id for a in protocol.arms if a.id not in given]
    extra = sorted(set(given) - {a.id for a in protocol.arms})
    if missing or extra:
        kind = "loaders" if sequential else "generators"
        raise RunError(f"{kind} must match protocol arms (missing {missing}, extra {extra})")
    if grader.id != protocol.grader.get("id") or grader.revision != protocol.grader.get("revision"):
        raise RunError("grader id/revision differ from the protocol")

    check = check_items(list(items))
    if not check["ok"]:
        raise RunError(f"dataset integrity check failed: {check['errors']}")
    selected = sorted((i for i in items if i.split == split), key=lambda i: i.id)
    if not selected:
        raise RunError(f"no items in split {split!r}")
    if sequential:
        in_order = [i.id for i in items if i.split == split]
        shard_ids = [in_order[k :: protocol.shards] for k in range(protocol.shards)]
        if any(not ids for ids in shard_ids):
            raise RunError(
                f"execution.shards ({protocol.shards}) exceeds the {len(in_order)} selected items; "
                "every shard must hold at least one item"
            )
        # Batch composition can change outputs, and the dataset hash ignores item order, so the
        # exact per-shard item lists are bound here.
        shards_sha256: str | None = sha256_json(shard_ids)
    else:
        shards_sha256 = None

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

    def request(item: Item, arm: Any, samples: list[tuple[int, int]]) -> GenerationRequest:
        seed = samples[0][1]
        return GenerationRequest(
            arm.id, item.id, item.prompt, dict(arm.inference), seed, len(samples)
        )

    def execute(item: Item, arm: Any, samples: list[tuple[int, int]]) -> None:
        req = request(item, arm, samples)
        recs = [record(item, arm, k, s, len(samples)) for k, s in samples]
        t0 = clock()
        try:
            result: Any = generators[arm.id].generate(req)  # type: ignore[index]
        except Exception as exc:  # recorded, never dropped: one error row per requested sample
            process(item, arm, recs, None, round(clock() - t0, 6), raised=exc)
            return
        process(item, arm, recs, result, round(clock() - t0, 6))

    def process(
        item: Item,
        arm: Any,
        recs: list[dict[str, Any]],
        result: Any,
        latency: float,
        *,
        raised: Exception | None = None,
    ) -> None:
        """Turn one request's result, or the exception that failed it, into its records."""
        n = len(recs)
        if raised is not None:
            for rec in recs:
                failed(rec, type(raised).__name__, str(raised), latency)
            records.extend(recs)
            return
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

    def units(shard_items: list[Item]) -> list[tuple[Item, list[tuple[int, int]]]]:
        if protocol.sampling_mode == "n_per_request":
            batch = [(k, protocol.seeds[0]) for k in range(protocol.samples_per_item)]
            return [(item, batch) for item in shard_items]
        return [
            (item, [(sample, seed)])
            for item in shard_items
            for sample, seed in enumerate(protocol.seeds)
        ]

    def run_arm(arm: Any, gen: Any) -> None:
        in_file_order = [i for i in items if i.split == split]
        for shard_index in range(protocol.shards):
            shard_items = in_file_order[shard_index :: protocol.shards]
            if not shard_items:
                continue
            todo = units(shard_items)
            reqs = [request(item, arm, samples) for item, samples in todo]
            recs_per = []
            for item, samples in todo:
                recs = [record(item, arm, k, sd, len(samples)) for k, sd in samples]
                for rec in recs:
                    rec.update(shard=shard_index, batch_size=len(reqs))
                recs_per.append(recs)
            t0 = clock()
            batch_error: tuple[str, str] | None = None
            try:
                results: Any = gen.generate_batch(reqs)
            except Exception as exc:  # the whole batch failed: an error row for every request
                batch_error = (type(exc).__name__, str(exc))
            latency = round(clock() - t0, 6)
            if batch_error is None:
                batch_error = _batch_problem(results, reqs)
            for (item, _samples), recs, k in zip(todo, recs_per, range(len(reqs)), strict=True):
                if batch_error is not None:
                    for rec in recs:
                        failed(rec, *batch_error, latency)
                    records.extend(recs)
                    continue
                res = results[k].result
                if isinstance(res, Exception):
                    process(item, arm, recs, None, latency, raised=res)
                else:
                    process(item, arm, recs, res, latency)

    described: dict[str, dict[str, Any] | None] = {a.id: None for a in protocol.arms}
    loads: list[dict[str, Any]] = []

    def run_sequential() -> None:
        seen: set[str] = set()
        engines: list[Any] = []  # kept so a reused engine object is recognised (not just its id)
        for position, arm in enumerate(protocol.arms):
            entry: dict[str, Any] = {
                "arm": arm.id,
                "position": position,
                "status": "load_failed",  # until the loader returns
                "load_started_at": _now(),
            }
            loads.append(entry)
            gen = loaders[arm.id]()  # type: ignore[index]  # a load failure aborts the run
            entry["status"] = "running"
            reused = any(gen is e for e in engines)
            engines.append(gen)
            try:
                if reused:
                    raise RunError(
                        f"arm {arm.id}: the loader returned an engine object already used by an "
                        "earlier arm; every arm must be a separate load"
                    )
                desc = gen.describe()
                if not isinstance(desc, dict):
                    raise RunError(f"arm {arm.id}: describe() must return an object")
                described[arm.id] = desc
                entry["load_id"] = desc.get("load_id")
                if protocol.synthetic is False and desc.get("synthetic"):
                    raise RunError(
                        "synthetic generators may only be used with a synthetic protocol"
                    )
                load_id = desc.get("load_id")
                if not isinstance(load_id, str) or not load_id:
                    raise RunError(f"arm {arm.id}: the loaded generator must report a load_id")
                if load_id in seen:
                    raise RunError(
                        f"arm {arm.id}: load_id {load_id!r} was already used; every arm must be "
                        "a separate load"
                    )
                seen.add(load_id)
                if not callable(getattr(gen, "generate_batch", None)):
                    raise RunError(
                        f"arm {arm.id}: a per_arm_sequential generator needs generate_batch"
                    )
                if not callable(getattr(gen, "close", None)):
                    raise RunError(
                        f"arm {arm.id}: a per_arm_sequential generator needs close(), so the "
                        "engine is released before the next arm is loaded"
                    )
                run_arm(arm, gen)
            except BaseException:
                if not reused:
                    entry["closed"] = _close_quietly(gen)
                entry.update(status="failed", finished_at=_now())
                raise
            entry["status"] = "close_failed"
            gen.close()  # a close failure aborts the run: the next load may not have the GPU
            entry.update(status="complete", closed=True, finished_at=_now())

    abort: BaseException | None = None
    try:
        if sequential:
            run_sequential()
        for item in [] if sequential else selected:
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
        "abort_message": None if abort is None else str(abort)[:_MAX_ERROR_CHARS],
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
                "generator": described[a.id] if sequential else generators[a.id].describe(),
            }
            for a in protocol.arms
        ],
        "seeds": list(protocol.seeds),
        "sampling_mode": protocol.sampling_mode,
        "execution": {
            "order": protocol.execution_order,
            "shards": protocol.shards,
            "shards_sha256": shards_sha256,
            "loads": loads if sequential else None,
        },
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


def _batch_problem(results: Any, reqs: Sequence[GenerationRequest]) -> tuple[str, str] | None:
    """Why a batch result cannot be attributed to its requests, or None if it can."""
    if not isinstance(results, list | tuple) or len(results) != len(reqs):
        got = len(results) if isinstance(results, list | tuple) else type(results).__name__
        return "BatchSizeMismatch", f"batch of {len(reqs)} requests returned {got}"
    for k, (entry, req) in enumerate(zip(results, reqs, strict=True)):
        if not isinstance(entry, BatchItem):
            return "BatchEntryInvalid", f"entry {k} is {type(entry).__name__}, not BatchItem"
        if entry.request != req:
            # Positions alone cannot be trusted: a reordered result would otherwise credit one
            # request's output to another.
            return "BatchOrderMismatch", f"entry {k} answers a different request"
    return None


def _close_quietly(gen: Any) -> bool | None:
    """Release the engine while another failure propagates; never hide that failure.

    Returns True if ``close()`` succeeded, False if it raised, None if there is no ``close``.
    """
    close = getattr(gen, "close", None)
    if not callable(close):
        return None
    try:
        close()
    except Exception:
        return False
    return True


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

"""One worker process per arm (readiness proposal: each arm a separate process and engine load).

The parent (the CLI and the harness runner) never imports vLLM, transformers or torch. Each
arm's engine is created in a **worker process** started with the ``spawn`` start method (a
fresh interpreter, no inherited state), and the parent talks to it through
:class:`WorkerEngine`, which implements the harness's ``BatchGenerator``:

* **load**: the worker re-parses the protocol (its SHA-256 must equal the parent's), creates
  the arm's engine with the configured factory, and replies ``ready`` with the engine's
  ``describe()``; a failed load is reported and the worker exits;
* **generate_batch**: one request message per shard, answered in order; every reply is
  validated strictly (JSON only, exact keys and types, one entry per request, each echoing its
  request exactly), so a malformed or reordered reply fails the batch instead of being credited
  to another request;
* **close**: the worker closes its engine, replies with the engine's close report and exits;
  ``close()`` returns only once the worker process has **exited with code 0 and been reaped**,
  so the next arm is never started while a worker is alive. A worker that does not finish in
  time is terminated, then killed, and the close fails.

Crashes, timeouts and malformed replies mark the worker dead (and it is killed if still
running): the batch fails, every later batch of the arm fails at once, and the close fails, so
the runner stops before the next arm and writes an aborted manifest. The worker ignores SIGINT;
the parent decides when it stops, and a parent-side interrupt terminates a worker that is busy.

Messages are JSON bytes over a ``multiprocessing`` pipe; the parent never unpickles anything the
worker sends. **A worker process that exited does not by itself show that every GPU resource
was released**: processes the engine started (vLLM's own workers) may outlive it, and the
driver releases device memory asynchronously. Device-level checks (NVML / ``nvidia-smi`` and the
process list) remain required on the GPU host.

Tests use stub engines and stub workers only (``tests/wsp_process_stubs.py``); the real factory
(:func:`vllm_engine`) and probe (:func:`runtime_probe`) import vLLM only inside the worker and
have not run against the real libraries.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import multiprocessing
import os
import re
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wsp_eval.generation import BatchItem, Generation, GenerationRequest
from wsp_eval.protocol import parse_protocol

START_METHOD = "spawn"
REAL_ENGINE_FACTORY = "wsp_runtime.process_worker:vllm_engine"
REAL_PROBE = "wsp_runtime.process_worker:runtime_probe"
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
_REF = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")
_REQUEST_KEYS = ("arm_id", "item_id", "prompt", "inference", "seed", "n")
_GENERATION_KEYS = {
    "text": (str,),
    "output_tokens": (int, type(None)),
    "hit_token_limit": (bool, type(None)),
    "finish_reason": (str, type(None)),
    "rendered_prompt_sha256": (str, type(None)),
    "usage": (dict,),
}
_PROCESS_NOTE = (
    "the worker process exited; that does not by itself show that every GPU resource was "
    "released (processes the engine started may outlive it, device memory is released by the "
    "driver): device-level checks remain required on the GPU host"
)


@dataclass(frozen=True)
class Timeouts:
    """Seconds. Provisional defaults; set them for the host and the run."""

    load: float = 3600.0
    batch: float = 14400.0
    close: float = 600.0
    kill_grace: float = 30.0


class WorkerError(RuntimeError):
    """The arm's worker process failed; the arm cannot continue."""


class WorkerLoadError(WorkerError):
    pass


class WorkerDied(WorkerError):
    pass


class WorkerTimeout(WorkerError):
    pass


class WorkerProtocolError(WorkerError):
    """The worker sent something that is not a valid message for this point."""


class WorkerBatchError(RuntimeError):
    """The worker's engine failed the whole batch (the worker itself is still usable)."""


class WorkerRequestError(RuntimeError):
    """The worker's engine failed one request of a batch."""


class WorkerCloseError(RuntimeError):
    """The worker could not be closed cleanly; the next arm must not be started."""


def resolve(ref: str) -> Any:
    """``"package.module:name"`` to the object (only importable names, no expressions)."""
    if not isinstance(ref, str) or not _REF.fullmatch(ref):
        raise ValueError(f"not a module:name reference: {ref!r}")
    module, name = ref.split(":")
    return getattr(importlib.import_module(module), name)


def _short(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:2000]}


# --- wire format (JSON bytes; both directions) ---------------------------------------------


def send(conn: Any, kind: str, payload: Any) -> None:
    data = json.dumps({"kind": kind, "payload": payload}, allow_nan=False, sort_keys=True)
    conn.send_bytes(data.encode("utf-8"))


def _decode(data: bytes, expected: Sequence[str]) -> tuple[str, Any]:
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkerProtocolError(f"worker sent a message that is not JSON ({exc})") from None
    if not isinstance(msg, dict) or set(msg) != {"kind", "payload"}:
        raise WorkerProtocolError("worker message must be an object with kind and payload")
    if msg["kind"] not in expected:
        raise WorkerProtocolError(f"worker sent {msg['kind']!r}, expected one of {list(expected)}")
    return msg["kind"], msg["payload"]


def encode_request(req: GenerationRequest) -> dict[str, Any]:
    return {k: getattr(req, k) for k in _REQUEST_KEYS}


def decode_request(d: Mapping[str, Any]) -> GenerationRequest:
    return GenerationRequest(**{k: d[k] for k in _REQUEST_KEYS})


def encode_result(result: Any) -> dict[str, Any]:
    if isinstance(result, Exception):
        return {"error": _short(result)}
    gens = [result] if isinstance(result, Generation) else list(result)
    out = []
    for g in gens:
        if not isinstance(g, Generation):
            raise TypeError(f"engine returned {type(g).__name__}, not Generation")
        out.append({k: getattr(g, k) for k in _GENERATION_KEYS})
    return {"generations": out, "single": isinstance(result, Generation)}


def _decode_generation(d: Any, where: str) -> Generation:
    if not isinstance(d, dict) or set(d) != set(_GENERATION_KEYS):
        raise WorkerProtocolError(
            f"{where}: generation must have exactly {sorted(_GENERATION_KEYS)}"
        )
    for key, kinds in _GENERATION_KEYS.items():
        value = d[key]
        if type(value) not in kinds:  # exact types: no bool for int, no float for int
            raise WorkerProtocolError(f"{where}: generation {key} has type {type(value).__name__}")
    return Generation(**d)


def decode_items(
    payload: Any, requests: Sequence[GenerationRequest], sent: Sequence[Mapping[str, Any]]
) -> list[BatchItem]:
    """Strict validation of a batch reply against the requests sent."""
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        raise WorkerProtocolError("batch reply must be an object with items")
    entries = payload["items"]
    if not isinstance(entries, list) or len(entries) != len(requests):
        n = len(entries) if isinstance(entries, list) else type(entries).__name__
        raise WorkerProtocolError(f"batch reply has {n} entries for {len(requests)} requests")
    out = []
    for k, (entry, req, expected) in enumerate(zip(entries, requests, sent, strict=True)):
        where = f"batch entry {k}"
        if not isinstance(entry, dict) or set(entry) != {"request", "result"}:
            raise WorkerProtocolError(f"{where}: must have exactly request and result")
        if entry["request"] != expected:
            raise WorkerProtocolError(f"{where} answers a different request")
        result = entry["result"]
        if not isinstance(result, dict):
            raise WorkerProtocolError(f"{where}: result must be an object")
        if set(result) == {"error"}:
            err = result["error"]
            if not (isinstance(err, dict) and set(err) == {"type", "message"}) or not all(
                isinstance(v, str) for v in err.values()
            ):
                raise WorkerProtocolError(f"{where}: malformed error")
            out.append(BatchItem(req, WorkerRequestError(f"{err['type']}: {err['message']}")))
            continue
        if set(result) != {"generations", "single"} or type(result["single"]) is not bool:
            raise WorkerProtocolError(f"{where}: result must be generations or error")
        if not isinstance(result["generations"], list):
            raise WorkerProtocolError(f"{where}: generations must be a list")
        gens = [_decode_generation(g, where) for g in result["generations"]]
        if result["single"]:
            if len(gens) != 1:
                raise WorkerProtocolError(f"{where}: a single result must have one generation")
            out.append(BatchItem(req, gens[0]))
        else:
            out.append(BatchItem(req, gens))
    return out


# --- worker side --------------------------------------------------------------------------


def worker_main(conn: Any, spec: dict[str, Any]) -> None:
    """Entry point of an arm's worker process."""
    # The parent decides when the worker stops: a terminal Ctrl-C reaches the whole process
    # group, and the parent's interrupt handling then terminates or closes this worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    code = 70
    try:
        code = _serve(conn, spec)
    except BaseException as exc:  # report what we can; the exit code says it failed
        with contextlib.suppress(Exception):
            send(conn, "fatal", _short(exc))
    finally:
        conn.close()
    sys.exit(code)


def _serve(conn: Any, spec: dict[str, Any]) -> int:
    protocol = parse_protocol(spec["protocol"])
    if protocol.sha256 != spec["protocol_sha256"]:
        send(conn, "load_failed", {"type": "ProtocolMismatch", "message": "protocol hash differs"})
        return 3
    arm = {a.id: a for a in protocol.arms}[spec["arm_id"]]
    try:
        factory = resolve(spec["engine_factory"])
        engine = factory(arm, Path(spec["checkpoint_dir"]), dict(spec.get("options") or {}))
        describe = engine.describe()
        send(conn, "ready", {"describe": describe, "pid": os.getpid()})
    except Exception as exc:
        send(conn, "load_failed", _short(exc))
        return 3
    while True:
        try:
            data = conn.recv_bytes(MAX_MESSAGE_BYTES)
        except (EOFError, OSError):
            # The parent went away: release the engine and exit.
            _close_quietly(engine)
            return 4
        kind, payload = _decode(data, ("generate", "close"))
        if kind == "generate":
            try:
                requests = [decode_request(d) for d in payload["requests"]]
                items = engine.generate_batch(requests)
                reply = {
                    "items": [
                        {"request": encode_request(i.request), "result": encode_result(i.result)}
                        for i in items
                    ]
                }
                send(conn, "batch", reply)
            except Exception as exc:
                send(conn, "batch_failed", _short(exc))
            continue
        report = None
        try:
            engine.close()
            report = getattr(engine, "close_report", None)
            send(conn, "closed", {"close_report": report})
            return 0
        except Exception as exc:
            report = getattr(engine, "close_report", None)
            send(conn, "close_failed", {**_short(exc), "close_report": report})
            return 5


def _close_quietly(engine: Any) -> None:
    with contextlib.suppress(Exception):
        engine.close()


def vllm_engine(arm: Any, checkpoint_dir: Path, options: dict[str, Any]) -> Any:
    """The real engine factory, run **in the worker** only. Untested against real vLLM."""
    from .vllm_adapter import VllmBackend, VllmEngine

    return VllmEngine(
        arm,
        checkpoint_dir,
        VllmBackend.load(),
        max_parent_memory_after_close=options.get("max_torch_memory_after_close"),
    )


def runtime_probe(spec: dict[str, Any]) -> dict[str, Any]:
    """Real probe, run **in a short-lived worker**: imports vLLM and transformers (so the
    parent never does), reports their ``__version__`` and builds and reads back a
    ``SamplingParams`` object for every (n, seed) of every arm. Untested against the real
    libraries."""
    from .vllm_adapter import check_inference, check_sampling, force_offline

    force_offline()
    import transformers
    import vllm
    from vllm import SamplingParams

    protocol = parse_protocol(spec["protocol"])
    sampling = {}
    for arm in protocol.arms:
        settings = check_inference(arm)
        sampling[arm.id] = [
            check_sampling(SamplingParams, {"n": n, "seed": s, **settings}, arm.id)[1]
            for n, s in spec["sample_shapes"]
        ]
    return {
        "modules": {
            "vllm": getattr(vllm, "__version__", None),
            "transformers": getattr(transformers, "__version__", None),
        },
        "sampling": sampling,
    }


def probe_main(conn: Any, spec: dict[str, Any]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    code = 3
    try:
        send(conn, "probe", resolve(spec["probe"])(spec))
        code = 0
    except BaseException as exc:
        with contextlib.suppress(Exception):
            send(conn, "probe_failed", _short(exc))
    finally:
        conn.close()
    sys.exit(code)


# --- parent side --------------------------------------------------------------------------


def _context() -> Any:
    return multiprocessing.get_context(START_METHOD)


def _stop(proc: Any, grace: float, wait: float = 0.0) -> str:
    """Make sure ``proc`` has exited and been reaped; returns how it ended. ``wait`` is how long
    to let it exit on its own first (for a worker that has replied and is finishing)."""
    proc.join(wait)
    if proc.exitcode is not None:
        return "exited"
    proc.terminate()
    proc.join(grace)
    if proc.exitcode is not None:
        return "terminated"
    proc.kill()
    proc.join(grace)
    return "killed" if proc.exitcode is not None else "still_running"


def _recv(conn: Any, proc: Any, timeout: float, expected: Sequence[str]) -> tuple[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerTimeout(f"no reply from the worker within {timeout:g} s")
        if conn.poll(min(remaining, 0.2)):
            try:
                data = conn.recv_bytes(MAX_MESSAGE_BYTES)
            except (EOFError, OSError):
                proc.join(1)
                raise WorkerDied(f"worker exited (exit code {proc.exitcode})") from None
            kind, payload = _decode(data, (*expected, "fatal"))
            if kind == "fatal":
                raise WorkerDied(f"worker failed: {_describe_error(payload)}")
            return kind, payload
        if not proc.is_alive():
            raise WorkerDied(f"worker exited (exit code {proc.exitcode})")


def _describe_error(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("type"), str):
        return f"{payload['type']}: {str(payload.get('message'))[:500]}"
    return "malformed error payload"


def run_probe(
    spec: dict[str, Any], *, timeout: float, probe: str = REAL_PROBE, grace: float = 30.0
) -> dict[str, Any]:
    """Run ``probe`` in a short-lived worker and return its (validated) result."""
    ctx = _context()
    parent, child = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=probe_main, args=(child, {**spec, "probe": probe}), name="wsp-probe")
    proc.start()
    child.close()
    try:
        kind, payload = _recv(parent, proc, timeout, ("probe", "probe_failed"))
    finally:
        ended = _stop(proc, grace, wait=grace)
        parent.close()
    if kind == "probe_failed":
        raise WorkerError(f"runtime probe failed: {_describe_error(payload)}")
    if proc.exitcode != 0:
        raise WorkerError(f"runtime probe exited with code {proc.exitcode} ({ended})")
    modules = payload.get("modules") if isinstance(payload, dict) else None
    if not isinstance(modules, dict) or not all(
        isinstance(v, str | type(None)) for v in modules.values()
    ):
        raise WorkerProtocolError("runtime probe result must report module versions")
    return payload


class WorkerEngine:
    """Parent-side proxy for one arm's worker process. Create it through
    :func:`process_arm_loader`."""

    def __init__(self, arm_id: str, proc: Any, conn: Any, timeouts: Timeouts, reports: Any):
        self.arm_id = arm_id
        self._proc = proc
        self._conn = conn
        self._timeouts = timeouts
        self._reports = reports
        self._dead: str | None = None  # why the worker can no longer be used
        self._in_flight = False
        self._close_state: str | None = None
        self._close_error: str | None = None
        self.close_report: dict[str, Any] | None = None
        self._describe: dict[str, Any] = {}
        self.pid: int | None = None

    @classmethod
    def start(
        cls,
        protocol: Any,
        arm_id: str,
        checkpoint_dir: Path,
        *,
        engine_factory: str,
        options: Mapping[str, Any],
        timeouts: Timeouts,
        reports: list[dict[str, Any]] | None,
        worker: Callable[..., None] = worker_main,
    ) -> WorkerEngine:
        ctx = _context()
        parent, child = ctx.Pipe(duplex=True)
        spec = {
            "protocol": protocol.raw,
            "protocol_sha256": protocol.sha256,
            "arm_id": arm_id,
            "checkpoint_dir": str(checkpoint_dir),
            "engine_factory": engine_factory,
            "options": dict(options),
        }
        proc = ctx.Process(target=worker, args=(child, spec), name=f"wsp-arm-{arm_id}")
        proc.start()
        child.close()
        self = cls(arm_id, proc, parent, timeouts, reports)
        self.pid = proc.pid
        try:
            kind, payload = _recv(parent, proc, timeouts.load, ("ready", "load_failed"))
            if kind == "load_failed":
                raise WorkerLoadError(f"arm {arm_id}: load failed in the worker: "
                                      f"{_describe_error(payload)}")  # fmt: skip
            describe = payload.get("describe") if isinstance(payload, dict) else None
            if not isinstance(describe, dict) or not isinstance(describe.get("load_id"), str):
                raise WorkerProtocolError("ready reply must carry describe() with a load_id")
        except BaseException as exc:
            # A worker that reported a failed load exits by itself; anything else is stopped.
            wait = timeouts.kill_grace if isinstance(exc, WorkerLoadError) else 0.0
            ended = _stop(proc, timeouts.kill_grace, wait=wait)
            parent.close()
            self._record(
                {
                    "phase": "load_failed",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                    "worker": self._worker_info(ended),
                }
            )
            raise
        self._describe = {
            **describe,
            "isolation": {"mode": "process", "start_method": START_METHOD, "pid": proc.pid},
        }
        return self

    def _worker_info(self, ended: str) -> dict[str, Any]:
        return {"pid": self.pid, "exitcode": self._proc.exitcode, "ended": ended}

    def _record(self, report: dict[str, Any]) -> None:
        report = {"arm": self.arm_id, "load_id": self._describe.get("load_id"), **report}
        report.setdefault("note", _PROCESS_NOTE)
        self.close_report = report
        if self._reports is not None:
            self._reports.append(dict(report))

    def describe(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._describe))

    def _kill(self, reason: str) -> None:
        self._dead = self._dead or reason
        _stop(self._proc, self._timeouts.kill_grace)

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[BatchItem]:
        if self._close_state is not None:
            raise RuntimeError("engine is closed")
        if self._dead is not None:
            raise WorkerDied(f"arm {self.arm_id}: worker is not usable: {self._dead}")
        sent = [json.loads(json.dumps(encode_request(r))) for r in requests]
        self._in_flight = True
        try:
            send(self._conn, "generate", {"requests": sent})
            kind, payload = _recv(
                self._conn, self._proc, self._timeouts.batch, ("batch", "batch_failed")
            )
        except WorkerError as exc:
            self._kill(f"{type(exc).__name__}: {exc}")
            self._in_flight = False
            raise
        except (OSError, ValueError) as exc:  # broken pipe and the like
            self._kill(f"{type(exc).__name__}: {exc}")
            self._in_flight = False
            raise WorkerDied(f"arm {self.arm_id}: {exc}") from exc
        # An interrupt (BaseException) leaves _in_flight set: the close then terminates.
        self._in_flight = False
        if kind == "batch_failed":
            raise WorkerBatchError(_describe_error(payload))
        try:
            return decode_items(payload, requests, sent)
        except WorkerProtocolError as exc:
            self._kill(f"{type(exc).__name__}: {exc}")
            raise

    def close(self) -> None:
        """Close the worker's engine and wait for the worker to exit. Returns only once the
        process has exited with code 0 and been reaped; otherwise it is terminated (then
        killed) and the close fails. A failed close keeps failing."""
        if self._close_state == "ok":
            return
        if self._close_state == "failed":
            raise WorkerCloseError(
                f"arm {self.arm_id}: an earlier close failed: {self._close_error}"
            )
        self._close_state = "failed"
        self._close_error = "close interrupted"
        errors: list[str] = []
        engine_report = None
        ended = "unknown"
        try:
            if self._dead is not None:
                errors.append(f"worker was not usable: {self._dead}")
            elif self._in_flight:
                errors.append("a request was in flight (interrupted); the worker was terminated")
            elif self._proc.is_alive():
                try:
                    send(self._conn, "close", {})
                    kind, payload = _recv(
                        self._conn, self._proc, self._timeouts.close, ("closed", "close_failed")
                    )
                    engine_report = (
                        payload.get("close_report") if isinstance(payload, dict) else None
                    )
                    if kind == "close_failed":
                        errors.append(f"engine close failed: {_describe_error(payload)}")
                except (WorkerError, OSError, ValueError) as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            else:
                errors.append(f"worker had already exited (exit code {self._proc.exitcode})")
            if not errors:
                self._proc.join(self._timeouts.close)
        finally:
            # Whatever happened above (including an interrupt), the worker must be gone.
            ended = _stop(self._proc, self._timeouts.kill_grace)
            self._conn.close()
            exitcode = self._proc.exitcode
            if ended == "still_running":
                errors.append("the worker could not be stopped")
            elif ended != "exited":
                errors.append(f"the worker had to be {ended}")
            elif exitcode != 0:
                errors.append(f"the worker exited with code {exitcode}")
            self._record(
                {
                    "phase": "close",
                    "isolation": "process",
                    "engine_close_report": engine_report,
                    "worker": self._worker_info(ended),
                    "errors": errors,
                }
            )
        if errors:
            self._close_error = "; ".join(errors)[:500]
            raise WorkerCloseError(
                f"arm {self.arm_id}: worker close failed; not starting the next arm: {errors}"
            )
        self._close_state = "ok"
        self._close_error = None


def process_arm_loader(
    protocol: Any,
    arm_id: str,
    checkpoint_dir: Path,
    *,
    engine_factory: str = REAL_ENGINE_FACTORY,
    options: Mapping[str, Any] | None = None,
    timeouts: Timeouts | None = None,
    reports: list[dict[str, Any]] | None = None,
    worker: Callable[..., None] = worker_main,
) -> Callable[[], WorkerEngine]:
    """An ``ArmLoader`` that starts a new worker process per call (a separate load). The
    factory is named, not imported, here: only the worker imports it."""
    if not _REF.fullmatch(engine_factory):
        raise ValueError(f"engine_factory must be a module:name reference: {engine_factory!r}")

    def load() -> WorkerEngine:
        return WorkerEngine.start(
            protocol,
            arm_id,
            checkpoint_dir,
            engine_factory=engine_factory,
            options=options or {},
            timeouts=timeouts or Timeouts(),
            reports=reports,
            worker=worker,
        )

    return load

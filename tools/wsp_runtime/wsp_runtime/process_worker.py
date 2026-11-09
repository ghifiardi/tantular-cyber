"""One worker process per arm (readiness proposal: each arm a separate process and engine load).

The parent (the CLI and the harness runner) never imports vLLM, transformers or torch. Each
arm's engine is created in a **worker process** started with the ``spawn`` start method (a
fresh interpreter, no inherited state), and the parent talks to it through
:class:`WorkerEngine`, which implements the harness's ``BatchGenerator``:

* **start**: the worker makes itself the leader of a new process group and says so; the parent
  checks that the worker's process group is the worker's own (never the parent's) before it
  lets the worker continue, and refuses a worker it cannot verify;
* **load**: the worker re-parses the protocol (its SHA-256 must equal the parent's), creates
  the arm's engine with the configured factory, and replies ``ready`` with the engine's
  ``describe()``; a failed load is reported and the worker exits;
* **generate_batch**: one request message per shard, answered in order; every reply is
  validated strictly (JSON only, finite numbers only, exact keys and types, one entry per
  request, each echoing its request with the same JSON types), so a malformed or reordered
  reply fails the batch instead of being credited to another request;
* **close**: the worker closes its engine, replies with the engine's close report and exits;
  ``close()`` returns only once the worker process has **exited with code 0 and been reaped**
  and no process is left in its process group, so the next arm is never started while a
  worker is alive. A worker that does not finish in time is terminated, then killed, and the
  close fails.

Crashes, timeouts and malformed replies mark the worker dead (and it is killed if still
running): the batch fails, every later batch of the arm fails at once, and the close fails, so
the runner stops before the next arm and writes an aborted manifest. Every timeout covers the
**whole** message: a worker that sends part of a message and stalls times out like one that
sends nothing.

Interrupts: the workers run in their own process groups, so a terminal Ctrl-C reaches only the
parent, and they ignore SIGINT as well; the parent decides when they stop, and a parent-side
interrupt terminates a worker that is busy. The parent's cleanup (stopping a worker, closing
its pipe, writing its close report) defers a further Ctrl-C until it has finished, then
re-raises it.

Messages are length-prefixed JSON bytes over a ``multiprocessing`` pipe, read and written by
this module under a deadline; the parent never unpickles anything the worker sends.

**Process groups are a best-effort boundary.** Stopping a worker signals its process group,
and after the worker exits any process still in that group is reported and killed. A
descendant that leaves the group (``setsid``, ``setpgid``, a daemonising double fork) escapes
this and is neither seen nor stopped. **A worker process that exited does not by itself show
that every GPU resource was released**: such escaped processes may still hold the device, and
the driver releases device memory asynchronously. Device-level checks (NVML / ``nvidia-smi``
and the process list) remain required on the GPU host.

Tests use stub engines and stub workers only (``tests/wsp_process_stubs.py``); the real factory
(:func:`vllm_engine`) and probe (:func:`runtime_probe`) import vLLM only inside the worker and
have not run against the real libraries.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import multiprocessing
import os
import re
import selectors
import signal
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wsp_eval.generation import BatchItem, Generation, GenerationRequest
from wsp_eval.protocol import parse_protocol

START_METHOD = "spawn"
REAL_ENGINE_FACTORY = "wsp_runtime.process_worker:vllm_engine"
REAL_PROBE = "wsp_runtime.process_worker:runtime_probe"
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
MAX_TIMEOUT = 7 * 24 * 3600.0
MIN_KILL_GRACE = 1.0
_HEADER = struct.Struct("!Q")
_CHUNK = 1024 * 1024
_SLICE = 0.2  # how often a waiting parent checks that the worker is still alive
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
    "the worker process exited and its process group is empty; that does not by itself show "
    "that every GPU resource was released (a descendant that left the group is not seen, and "
    "device memory is released by the driver): device-level checks remain required on the GPU "
    "host"
)
_GROUP_ERRORS = {
    "killed": "processes in the worker's process group outlived it and had to be killed",
    "still_running": "processes in the worker's process group could not be stopped",
    "unknown": "the worker's process group could not be checked (permission denied)",
    "not_checked": "the worker's process group was not verified, so it was not checked",
}


@dataclass(frozen=True)
class Timeouts:
    """Seconds. Provisional defaults; set them for the host and the run. Each must be a finite
    number in (0, ``MAX_TIMEOUT``]; ``kill_grace`` at least ``MIN_KILL_GRACE``."""

    load: float = 3600.0
    batch: float = 14400.0
    close: float = 600.0
    kill_grace: float = 30.0

    def __post_init__(self) -> None:
        for name in ("load", "batch", "close", "kill_grace"):
            check_seconds(getattr(self, name), name)
        if self.kill_grace < MIN_KILL_GRACE:
            raise ValueError(f"kill_grace must be at least {MIN_KILL_GRACE:g} s")


def check_seconds(value: Any, name: str) -> float:
    """A timeout: an int or float (not a bool), finite, in (0, ``MAX_TIMEOUT``]."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number of seconds, not {value!r}")
    if not 0 < value <= MAX_TIMEOUT:
        raise ValueError(f"{name} must be in (0, {MAX_TIMEOUT:g}] seconds, not {value!r}")
    return float(value)


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


class _DeferInterrupts:
    """Hold back SIGINT while cleanup runs, then deliver it (once) to the previous handler.

    Used around every step that stops a worker, closes its pipe or records its close report,
    so a second Ctrl-C cannot leave a worker running or a report unwritten. Only the main
    thread receives SIGINT; elsewhere this does nothing. An interrupt arriving before the
    handler is installed (a few bytecodes) is not deferred."""

    def __enter__(self) -> _DeferInterrupts:
        self._previous: Any = None
        self._received = False
        if threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGINT)
            if previous is not None:  # None: a handler not installed from Python; leave it
                signal.signal(signal.SIGINT, self._hold)
                self._previous = previous
        return self

    def _hold(self, signum: int, frame: Any) -> None:
        self._received = True

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._previous is None:
            return
        signal.signal(signal.SIGINT, self._previous)
        if not self._received:
            return
        if self._previous is signal.default_int_handler or self._previous == signal.SIG_DFL:
            raise KeyboardInterrupt
        if callable(self._previous):
            self._previous(signal.SIGINT, None)


@dataclass
class _Clock:
    seconds: float
    deadline: float = field(init=False)

    def __post_init__(self) -> None:
        self.deadline = time.monotonic() + self.seconds

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


# --- wire format: 8-byte big-endian length, then UTF-8 JSON (both directions) -------------


def _ready(fd: int, events: int, timeout: float) -> bool:
    with selectors.DefaultSelector() as sel:
        sel.register(fd, events)
        return bool(sel.select(max(timeout, 0.0)))


def _write_frame(fd: int, data: bytes, clock: _Clock | None = None, proc: Any = None) -> None:
    """Write one frame. With ``clock`` (the parent; its end is non-blocking) the whole write
    must finish before the deadline; without it (the worker) the write blocks."""
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message of {len(data)} bytes exceeds {MAX_MESSAGE_BYTES}")
    view = memoryview(_HEADER.pack(len(data)) + data)
    while view:
        if clock is not None:
            remaining = clock.remaining()
            if remaining <= 0:
                raise WorkerTimeout(
                    f"the worker did not take the whole message within {clock.seconds:g} s"
                )
            if not _ready(fd, selectors.EVENT_WRITE, min(remaining, _SLICE)):
                if proc is not None and not proc.is_alive():
                    raise WorkerDied(f"worker exited (exit code {proc.exitcode})")
                continue
        try:
            n = os.write(fd, view[:_CHUNK])
        except BlockingIOError:
            continue
        view = view[n:]


def _read_frame(fd: int, clock: _Clock | None = None, proc: Any = None) -> bytes:
    """Read one whole frame. With ``clock`` (the parent) the **complete** message must arrive
    before the deadline: a header or body that stops part-way times out. An announced length
    above ``MAX_MESSAGE_BYTES`` is refused before anything else is read."""
    buf = bytearray()
    need = _HEADER.size
    length: int | None = None
    while True:
        if len(buf) >= need:
            if length is None:
                (length,) = _HEADER.unpack(bytes(buf[: _HEADER.size]))
                if length > MAX_MESSAGE_BYTES:
                    raise WorkerProtocolError(
                        f"message of {length} bytes announced; the limit is {MAX_MESSAGE_BYTES}"
                    )
                need = _HEADER.size + length
                continue
            return bytes(buf[_HEADER.size :])
        if clock is not None:
            remaining = clock.remaining()
            if remaining <= 0:
                partial = f" ({len(buf)} bytes of an incomplete message received)" if buf else ""
                raise WorkerTimeout(
                    f"no complete reply from the worker within {clock.seconds:g} s{partial}"
                )
            if not _ready(fd, selectors.EVENT_READ, min(remaining, _SLICE)):
                if proc is not None and not proc.is_alive():
                    raise WorkerDied(f"worker exited (exit code {proc.exitcode})")
                continue
        try:
            chunk = os.read(fd, min(need - len(buf), _CHUNK))
        except BlockingIOError:
            continue
        if not chunk:
            raise EOFError("the other end closed the connection")
        buf += chunk


def write_frame(conn: Any, data: bytes) -> None:
    """Write one raw frame (blocking): the worker side, and tests of malformed replies."""
    _write_frame(conn.fileno(), data)


def send(conn: Any, kind: str, payload: Any, clock: _Clock | None = None, proc: Any = None) -> None:
    data = json.dumps({"kind": kind, "payload": payload}, allow_nan=False, sort_keys=True)
    _write_frame(conn.fileno(), data.encode("utf-8"), clock, proc)


def _non_finite(text: str) -> Any:
    raise ValueError(f"non-finite number {text}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number {text}")
    return value


def loads(data: bytes) -> Any:
    """Strict JSON: UTF-8, no NaN or Infinity, no number that overflows to infinity."""
    return json.loads(data.decode("utf-8"), parse_constant=_non_finite, parse_float=_finite_float)


def _decode(data: bytes, expected: Sequence[str], sender: str = "worker") -> tuple[str, Any]:
    try:
        msg = loads(data)
    except (UnicodeDecodeError, ValueError, RecursionError, MemoryError) as exc:
        raise WorkerProtocolError(
            f"{sender} sent a message that is not valid JSON ({type(exc).__name__}: "
            f"{str(exc)[:200]})"
        ) from None
    if not isinstance(msg, dict) or set(msg) != {"kind", "payload"}:
        raise WorkerProtocolError(f"{sender} message must be an object with kind and payload")
    if msg["kind"] not in expected:
        raise WorkerProtocolError(
            f"{sender} sent {msg['kind']!r}, expected one of {list(expected)}"
        )
    return msg["kind"], msg["payload"]


def _canonical(value: Any) -> str:
    # JSON text keeps the types apart: true is not 1, 1.0 is not 1.
    return json.dumps(value, sort_keys=True, allow_nan=False)


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
        if _canonical(entry["request"]) != _canonical(expected):
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


def _bootstrap(target: Callable[..., None], conn: Any, spec: dict[str, Any]) -> None:
    """Every worker and probe process starts here: ignore SIGINT, lead a new process group,
    tell the parent, and wait for its go-ahead (given only once the parent has verified the
    group) before running ``target``."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with contextlib.suppress(OSError):  # if this fails, the parent refuses the worker
        os.setpgid(0, 0)
    send(conn, "started", {})
    try:
        data = _read_frame(conn.fileno())
    except (EOFError, OSError):
        sys.exit(4)  # the parent went away before the go-ahead
    _decode(data, ("go",), sender="parent")
    target(conn, spec)


def worker_main(conn: Any, spec: dict[str, Any]) -> None:
    """An arm's worker, run by :func:`_bootstrap` (which already ignores SIGINT)."""
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
    engine = None
    try:
        factory = resolve(spec["engine_factory"])
        engine = factory(arm, Path(spec["checkpoint_dir"]), dict(spec.get("options") or {}))
        describe = engine.describe()
        send(conn, "ready", {"describe": describe, "pid": os.getpid()})
    except Exception as exc:
        if engine is not None:
            _close_quietly(engine)
        send(conn, "load_failed", _short(exc))
        return 3
    closing = False
    try:
        while True:
            try:
                data = _read_frame(conn.fileno())
            except (EOFError, OSError):
                return 4  # the parent went away; the engine is released below
            kind, payload = _decode(data, ("generate", "close"), sender="parent")
            if kind == "generate":
                try:
                    requests = [decode_request(d) for d in payload["requests"]]
                    items = engine.generate_batch(requests)
                    reply = {
                        "items": [
                            {
                                "request": encode_request(i.request),
                                "result": encode_result(i.result),
                            }
                            for i in items
                        ]
                    }
                    send(conn, "batch", reply)
                except Exception as exc:
                    send(conn, "batch_failed", _short(exc))
                continue
            closing = True
            try:
                engine.close()
                send(conn, "closed", {"close_report": getattr(engine, "close_report", None)})
                return 0
            except Exception as exc:
                report = getattr(engine, "close_report", None)
                send(conn, "close_failed", {**_short(exc), "close_report": report})
                return 5
    finally:
        # Any other way out (the parent leaving, a malformed message, a BaseException from
        # the engine) still releases the engine before the worker exits.
        if not closing:
            _close_quietly(engine)


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
    ``SamplingParams`` object for every (n, seed) of every arm. The parent checks the read-back
    values itself. Untested against the real libraries."""
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


def _group_of(pid: int | None) -> int | None:
    """``pid``'s process group if ``pid`` leads its own group and that group is not this
    process's; otherwise None (and nothing may be signalled as a group). Only meaningful while
    ``pid`` is unreaped: only this process reaps its workers, so the pid cannot be reused."""
    if pid is None:
        return None
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return None
    if pgid != pid or pgid <= 1 or pgid == os.getpgrp():
        return None
    return pgid


def _killpg(pgid: int, sig: int) -> None:
    if pgid <= 1 or pgid == os.getpgrp():  # never this process's own group
        raise RuntimeError(f"refusing to signal process group {pgid}")
    os.killpg(pgid, sig)


def _group_state(pgid: int) -> str:
    try:
        _killpg(pgid, 0)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "unknown"
    return "alive"


def _signal(proc: Any, pgid: int | None, sig: int) -> None:
    if proc.exitcode is not None:
        return
    # Re-verified now, while the worker is unreaped: its group is still its own.
    if pgid is not None and _group_of(proc.pid) == pgid:
        with contextlib.suppress(ProcessLookupError):
            _killpg(pgid, sig)
    elif sig == signal.SIGTERM:
        proc.terminate()
    else:
        proc.kill()


def _stop_group(pgid: int, grace: float) -> str:
    """After the worker has exited: wait up to ``grace`` for the rest of its process group to
    exit by itself, then kill what is left. Uses the group verified while the worker was alive
    (a process group ID is not reused while the group has members)."""
    state = _group_state(pgid)
    if state != "alive":
        return "none" if state == "gone" else state
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        time.sleep(0.05)
        state = _group_state(pgid)
        if state != "alive":
            return "exited" if state == "gone" else state
    with contextlib.suppress(ProcessLookupError):
        _killpg(pgid, signal.SIGKILL)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if _group_state(pgid) == "gone":
            return "killed"
        time.sleep(0.05)
    return "still_running"


def _join(proc: Any, timeout: float) -> None:
    """Wait for ``proc`` to exit and reap it. Polls the exit status rather than
    ``Process.join``, which waits on a pipe that a forked descendant may have inherited."""
    deadline = time.monotonic() + max(timeout, 0.0)
    while proc.exitcode is None and time.monotonic() < deadline:
        time.sleep(0.02)


def _stop(proc: Any, grace: float, wait: float = 0.0, pgid: int | None = None) -> tuple[str, str]:
    """Make sure ``proc`` has exited and been reaped, and that its process group is empty.
    Returns how the worker ended (``exited``, ``terminated``, ``killed``, ``still_running``) and
    what was left in its group (``none``, ``exited``, ``killed``, ``still_running``, ``unknown``,
    or ``not_checked`` without a verified group). ``wait`` is how long to let it exit on its own
    first. A Ctrl-C during this is held back until it has finished."""
    with _DeferInterrupts():
        _join(proc, wait)
        if proc.exitcode is not None:
            ended = "exited"
        else:
            _signal(proc, pgid, signal.SIGTERM)
            _join(proc, grace)
            if proc.exitcode is not None:
                ended = "terminated"
            else:
                _signal(proc, pgid, signal.SIGKILL)
                _join(proc, grace)
                ended = "killed" if proc.exitcode is not None else "still_running"
        group = _stop_group(pgid, grace) if pgid is not None else "not_checked"
    return ended, group


def _recv(conn: Any, proc: Any, clock: _Clock, expected: Sequence[str]) -> tuple[str, Any]:
    try:
        data = _read_frame(conn.fileno(), clock, proc)
    except (EOFError, OSError):
        _join(proc, 1)
        raise WorkerDied(f"worker exited (exit code {proc.exitcode})") from None
    kind, payload = _decode(data, (*expected, "fatal"))
    if kind == "fatal":
        raise WorkerDied(f"worker failed: {_describe_error(payload)}")
    return kind, payload


def _describe_error(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("type"), str):
        return f"{payload['type']}: {str(payload.get('message'))[:500]}"
    return "malformed error payload"


def _spawn(target: Callable[..., None], spec: dict[str, Any], name: str) -> tuple[Any, Any]:
    ctx = _context()
    parent, child = ctx.Pipe(duplex=True)
    try:
        proc = ctx.Process(target=_bootstrap, args=(target, child, spec), name=name)
        proc.start()
    except BaseException:
        parent.close()
        child.close()
        raise
    child.close()
    os.set_blocking(parent.fileno(), False)  # every parent-side read and write has a deadline
    return proc, parent


def _handshake(conn: Any, proc: Any, clock: _Clock) -> int:
    """Wait for the worker's ``started``, verify that it leads its own process group (not
    this process's), then let it continue. Returns the verified group."""
    _recv(conn, proc, clock, ("started",))
    pgid = _group_of(proc.pid)  # the worker is alive, waiting for the go-ahead
    if pgid is None:
        raise WorkerProtocolError("the worker does not lead its own process group; it is not used")
    send(conn, "go", {}, clock, proc)
    return pgid


def run_probe(
    spec: dict[str, Any], *, timeout: float, probe: str = REAL_PROBE, grace: float = 30.0
) -> dict[str, Any]:
    """Run ``probe`` in a short-lived worker and return its (validated) result: module
    versions, and one ``SamplingParams`` read-back per (n, seed) for every protocol arm (the
    caller checks the read-back values, see ``vllm_adapter.preflight``)."""
    timeout = check_seconds(timeout, "timeout")
    grace = check_seconds(grace, "grace")
    arm_ids = {a.id for a in parse_protocol(spec["protocol"]).arms}
    proc, parent = _spawn(probe_main, {**spec, "probe": probe}, "wsp-probe")
    pgid: int | None = None
    try:
        clock = _Clock(timeout)
        pgid = _handshake(parent, proc, clock)
        kind, payload = _recv(parent, proc, clock, ("probe", "probe_failed"))
    except BaseException:
        # Failing or interrupted: stop the probe now rather than waiting for it.
        with _DeferInterrupts():
            _stop(proc, grace, pgid=pgid)
            parent.close()
        raise
    with _DeferInterrupts():
        ended, group = _stop(proc, grace, wait=grace, pgid=pgid)
        parent.close()
    if kind == "probe_failed":
        raise WorkerError(f"runtime probe failed: {_describe_error(payload)}")
    if proc.exitcode != 0:
        raise WorkerError(f"runtime probe exited with code {proc.exitcode} ({ended})")
    if group in _GROUP_ERRORS:
        raise WorkerError(f"runtime probe: {_GROUP_ERRORS[group]}")
    modules = payload.get("modules") if isinstance(payload, dict) else None
    if not isinstance(modules, dict) or not all(
        isinstance(v, str | type(None)) for v in modules.values()
    ):
        raise WorkerProtocolError("runtime probe result must report module versions")
    sampling = payload.get("sampling")
    shapes = spec["sample_shapes"]
    if (
        not isinstance(sampling, dict)
        or set(sampling) != arm_ids
        or not all(
            isinstance(v, list) and len(v) == len(shapes) and all(isinstance(o, dict) for o in v)
            for v in sampling.values()
        )
    ):
        raise WorkerProtocolError(
            "runtime probe result must read back SamplingParams for every (n, seed) of every "
            "protocol arm"
        )
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
        self._group_checks: list[str] = []
        self.close_report: dict[str, Any] | None = None
        self._describe: dict[str, Any] = {}
        self.pid: int | None = proc.pid
        self.pgid: int | None = None

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
        spec = {
            "protocol": protocol.raw,
            "protocol_sha256": protocol.sha256,
            "arm_id": arm_id,
            "checkpoint_dir": str(checkpoint_dir),
            "engine_factory": engine_factory,
            "options": dict(options),
        }
        proc, parent = _spawn(worker, spec, f"wsp-arm-{arm_id}")
        self = cls(arm_id, proc, parent, timeouts, reports)
        try:
            clock = _Clock(timeouts.load)  # the handshake and the load share the load timeout
            self.pgid = _handshake(parent, proc, clock)
            kind, payload = _recv(parent, proc, clock, ("ready", "load_failed"))
            if kind == "load_failed":
                raise WorkerLoadError(f"arm {arm_id}: load failed in the worker: "
                                      f"{_describe_error(payload)}")  # fmt: skip
            describe = payload.get("describe") if isinstance(payload, dict) else None
            if not isinstance(describe, dict) or not isinstance(describe.get("load_id"), str):
                raise WorkerProtocolError("ready reply must carry describe() with a load_id")
        except BaseException as exc:
            # A worker that reported a failed load exits by itself; anything else is stopped.
            with _DeferInterrupts():
                wait = timeouts.kill_grace if isinstance(exc, WorkerLoadError) else 0.0
                ended, group = _stop(proc, timeouts.kill_grace, wait=wait, pgid=self.pgid)
                self._group_checks.append(group)
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
            "isolation": {
                "mode": "process",
                "start_method": START_METHOD,
                "pid": proc.pid,
                "pgid": self.pgid,
            },
        }
        return self

    def _worker_info(self, ended: str) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "exitcode": self._proc.exitcode,
            "ended": ended,
            "process_group": list(self._group_checks),
        }

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
        _, group = _stop(self._proc, self._timeouts.kill_grace, pgid=self.pgid)
        self._group_checks.append(group)

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[BatchItem]:
        if self._close_state is not None:
            raise RuntimeError("engine is closed")
        if self._dead is not None:
            raise WorkerDied(f"arm {self.arm_id}: worker is not usable: {self._dead}")
        sent = [json.loads(json.dumps(encode_request(r), allow_nan=False)) for r in requests]
        clock = _Clock(self._timeouts.batch)  # sending the shard and the whole reply
        self._in_flight = True
        try:
            send(self._conn, "generate", {"requests": sent}, clock, self._proc)
            kind, payload = _recv(self._conn, self._proc, clock, ("batch", "batch_failed"))
            items = decode_items(payload, requests, sent) if kind == "batch" else None
        except Exception as exc:
            # Anything after the request was sent (a timeout, a malformed or unreadable reply,
            # a broken pipe, even MemoryError) leaves the worker in an unknown state.
            self._in_flight = False
            self._kill(f"{type(exc).__name__}: {exc}"[:500])
            if isinstance(exc, WorkerError):
                raise
            raise WorkerDied(f"arm {self.arm_id}: {type(exc).__name__}: {exc}") from exc
        # An interrupt (BaseException) leaves _in_flight set: the close then terminates.
        self._in_flight = False
        if items is None:  # batch_failed
            raise WorkerBatchError(_describe_error(payload))
        return items

    def close(self) -> None:
        """Close the worker's engine and wait for the worker to exit. Returns only once the
        process has exited with code 0, been reaped and left nothing in its process group;
        otherwise it is terminated (then killed) and the close fails. A failed close keeps
        failing."""
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
        try:
            if self._dead is not None:
                errors.append(f"worker was not usable: {self._dead}")
            elif self._in_flight:
                errors.append("a request was in flight (interrupted); the worker was terminated")
            elif self._proc.is_alive():
                try:
                    clock = _Clock(self._timeouts.close)
                    send(self._conn, "close", {}, clock, self._proc)
                    kind, payload = _recv(self._conn, self._proc, clock, ("closed", "close_failed"))
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
                _join(self._proc, self._timeouts.close)
        finally:
            # Whatever happened above (including an interrupt), the worker must be gone; a
            # further Ctrl-C is held back until it is and the report is recorded.
            with _DeferInterrupts():
                ended, group = _stop(self._proc, self._timeouts.kill_grace, pgid=self.pgid)
                self._group_checks.append(group)
                with contextlib.suppress(OSError):
                    self._conn.close()
                exitcode = self._proc.exitcode
                if ended == "still_running":
                    errors.append("the worker could not be stopped")
                elif ended != "exited":
                    errors.append(f"the worker had to be {ended}")
                elif exitcode != 0:
                    errors.append(f"the worker exited with code {exitcode}")
                for check in dict.fromkeys(self._group_checks):
                    if check in _GROUP_ERRORS:
                        errors.append(_GROUP_ERRORS[check])
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

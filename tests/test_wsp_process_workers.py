"""One worker process per arm (tools/wsp_runtime/wsp_runtime/process_worker.py), tested with stub
engines and stub workers only (tests/wsp_process_stubs.py). No model, GPU, network or ML library.

These tests show process control flow: separate worker processes, request/sample identity, shard
order, explicit failures, aborted manifests and that a worker has exited before the next arm
starts. A worker that exited does not show that a real engine released the GPU."""

# ruff: noqa: F811 - the fixtures imported from test_wsp_runtime are used as parameters
from __future__ import annotations

import ast
import json
import math
import multiprocessing
import os
import signal
import socket
import sys
import time

import pytest
import test_wsp_runtime as t
import wsp_process_stubs as stubs
from test_wsp_runtime import (  # noqa: F401 - fixtures used by name
    _storage,
    ckpts,
    protocol,
)
from wsp_eval.runner import verify_run
from wsp_runtime import process_worker as pw

STUB = "wsp_process_stubs:stub_engine"
PROBE = "wsp_process_stubs:stub_probe"
FAST = ["--load-timeout", "30", "--batch-timeout", "3", "--close-timeout", "5", "--kill-grace", "2"]


@pytest.fixture(autouse=True)
def _no_workers_left_behind():
    yield
    # Detect first (active_children() reaps finished ones and lists the live ones), then clean
    # up, so a leaked worker fails the test instead of being hidden by the cleanup.
    leaked = multiprocessing.active_children()
    for p in leaked:
        p.kill()
        p.join(5)
    assert leaked == [], f"worker processes were left running: {leaked}"


def _run(tmp_path, protocol_raw, ckpts, *, options=None, factory=STUB, probe=PROBE, extra=(),
         worker=pw.worker_main, lookup=None):  # fmt: skip
    argv = t._argv(tmp_path, ckpts, protocol_raw=protocol_raw) + FAST + list(extra)
    return t.runtime_cli.run(
        argv,
        verify_backend_factory=t.ExactGrader,
        dependency_lookup=lookup or t._lookup(),
        storage_check=t.STORAGE["check"],
        environ={},
        isolation="process",
        worker_engine_factory=factory,
        worker_probe=probe,
        worker_options=options or {},
        worker_main=worker,
    )


def _shards(n_items=5, shards=2):
    ids = [f"dev-{k:03d}" for k in range(n_items)]
    return [ids[i::shards] for i in range(shards)]


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


# --- a whole run ----------------------------------------------------------------------------


def test_each_arm_runs_in_its_own_worker_process(tmp_path, protocol, ckpts) -> None:
    summary = _run(tmp_path, protocol.raw, ckpts, options={"import_sentinel": True})
    assert summary["run_status"] == "complete" and summary["error_records"] == 0
    manifest, records = verify_run(tmp_path / "run", protocol)
    assert len(records) == 5 * 4 * 3 and all(r["status"] == "graded" for r in records)
    assert all(r["correct"] for r in records)  # request/sample identity survived the round trip
    loads = manifest["execution"]["loads"]
    assert [x["arm"] for x in loads] == ["base", "edited", "base_reload"]
    assert all(x["status"] == "complete" and x["closed"] is True for x in loads)
    pids = [a["generator"]["isolation"]["pid"] for a in manifest["arms"]]
    assert len(set(pids)) == 3 and os.getpid() not in pids  # three separate worker processes
    assert all(
        a["generator"]["worker_pid"] == p for a, p in zip(manifest["arms"], pids, strict=True)
    )
    reports = summary["close_reports"]
    assert [r["worker"]["exitcode"] for r in reports] == [0, 0, 0]
    assert [r["worker"]["ended"] for r in reports] == ["exited"] * 3
    assert all(r["errors"] == [] for r in reports)
    # Shard order: each worker received the selected items [i::2] in file order.
    assert [r["engine_close_report"]["batches"] for r in reports] == [_shards()] * 3
    assert all(_gone(p) for p in pids)
    # The worker imported the stand-in ML module; this (parent) process never did.
    assert "wsp_sentinel_ml" not in sys.modules
    report = json.loads((tmp_path / "run" / "runtime_report.json").read_text())
    assert (
        report["isolation"]["mode"] == "process" and report["isolation"]["start_method"] == "spawn"
    )
    assert report["runtime_probe"]["probe_pid"] != os.getpid()


def test_a_worker_has_exited_before_the_next_arm_starts(
    tmp_path, protocol, ckpts, monkeypatch
) -> None:
    started: list = []
    original = pw.WorkerEngine.start.__func__

    def start(cls, *args, **kwargs):
        # Every earlier worker must already have exited and been reaped.
        assert all(e._proc.exitcode is not None for e in started)
        engine = original(cls, *args, **kwargs)
        started.append(engine)
        return engine

    monkeypatch.setattr(pw.WorkerEngine, "start", classmethod(start))
    summary = _run(tmp_path, protocol.raw, ckpts)
    assert summary["run_status"] == "complete" and len(started) == 3
    assert [e._proc.exitcode for e in started] == [0, 0, 0]


def test_the_parent_never_imports_ml_libraries() -> None:
    # process_worker and run.py import vLLM/transformers only inside worker-side functions.
    for path in (t.RUNTIME / "wsp_runtime" / "process_worker.py", t.RUNTIME / "run.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = (
                    [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                )
                assert not any(n.split(".")[0] in t._LAZY_ONLY for n in names), (path.name, names)
    assert not {"vllm", "torch", "transformers"} & set(sys.modules)


# --- failures -------------------------------------------------------------------------------


def test_a_worker_crash_aborts_the_run_before_the_next_arm(tmp_path, protocol, ckpts) -> None:
    with pytest.raises(t.runtime_cli._Aborted) as exc:
        _run(tmp_path, protocol.raw, ckpts, options={"crash_at_batch": 0})
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["run_status"] == "aborted"
    assert [x["status"] for x in manifest["execution"]["loads"]] == ["close_failed"]
    records = [json.loads(x) for x in (tmp_path / "run" / "records.jsonl").read_text().splitlines()]
    assert len(records) == 5 * 4 and {r["status"] for r in records} == {"error"}  # kept, explicit
    assert {r["error_class"].split(":")[-1] for r in records} <= {"WorkerDied"}
    (report,) = exc.value.summary["close_reports"]
    assert report["worker"]["exitcode"] == 9
    assert any("not usable" in e for e in report["errors"])


def test_a_batch_timeout_kills_the_worker_and_aborts(tmp_path, protocol, ckpts) -> None:
    with pytest.raises(t.runtime_cli._Aborted) as exc:
        _run(tmp_path, protocol.raw, ckpts, options={"hang_at_batch": 0},
             extra=["--batch-timeout", "1"])  # fmt: skip
    (report,) = exc.value.summary["close_reports"]
    assert report["worker"]["exitcode"] is not None and report["worker"]["exitcode"] < 0  # signal
    assert any("WorkerTimeout" in e for e in report["errors"])
    # The parent stopped it after the timeout: recorded as such, not as a worker that exited.
    assert report["worker"]["ended"] == "terminated"
    assert "the worker had to be terminated" in report["errors"]
    records = [json.loads(x) for x in (tmp_path / "run" / "records.jsonl").read_text().splitlines()]
    classes = {r["error_class"].split(":")[-1] for r in records}
    assert classes == {"WorkerTimeout", "WorkerDied"}  # the timed-out shard, then the rest


def test_request_and_batch_failures_reported_by_the_worker_are_kept(
    tmp_path, protocol, ckpts
) -> None:
    summary = _run(tmp_path, protocol.raw, ckpts,
                   options={"request_error_item": "dev-002", "batch_error_at": 1})  # fmt: skip
    assert summary["run_status"] == "complete"  # the workers themselves stayed usable
    _, records = verify_run(tmp_path / "run", protocol)
    by = {}
    for r in records:
        by.setdefault(r["item_id"], set()).add(r["error_class"])
    assert by["dev-002"] == {"WorkerRequestError"}  # its own request failed
    assert by["dev-001"] == by["dev-003"] == {"WorkerBatchError"}  # shard 1 failed as a whole
    assert by["dev-000"] == by["dev-004"] == {None}
    assert all(r["worker"]["exitcode"] == 0 for r in summary["close_reports"])


def test_a_load_failure_in_the_worker_aborts_the_run(tmp_path, protocol, ckpts) -> None:
    with pytest.raises(t.runtime_cli._Aborted) as exc:
        _run(tmp_path, protocol.raw, ckpts, options={"load_error": True})
    assert "synthetic load failure" in exc.value.summary["error"]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert [x["status"] for x in manifest["execution"]["loads"]] == ["load_failed"]
    (report,) = exc.value.summary["close_reports"]
    assert report["phase"] == "load_failed" and report["worker"]["exitcode"] == 3


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"close_raises": True}, "engine close failed: RuntimeError: synthetic close failure"),
        ({"exit_code_after_close": 7}, "exited with code 7"),
        ({"linger_after_close": True}, "had to be terminated"),
        ({"hang_on_close": True}, "WorkerTimeout"),
    ],
)
def test_a_worker_that_does_not_close_and_exit_cleanly_stops_the_run(
    tmp_path, protocol, ckpts, options, message
) -> None:
    with pytest.raises(t.runtime_cli._Aborted) as exc:
        _run(tmp_path, protocol.raw, ckpts, options=options, extra=["--close-timeout", "1"])
    (report,) = exc.value.summary["close_reports"]
    assert any(message in e for e in report["errors"]), report["errors"]
    assert report["worker"]["exitcode"] is not None  # reaped either way
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert [x["status"] for x in manifest["execution"]["loads"]] == ["close_failed"]


def test_a_parent_interrupt_during_a_batch_terminates_the_worker(
    tmp_path, protocol, ckpts, monkeypatch
) -> None:
    original = pw._recv

    def interrupt_on_batch(conn, proc, timeout, expected):
        if "batch" in expected:
            raise KeyboardInterrupt
        return original(conn, proc, timeout, expected)

    monkeypatch.setattr(pw, "_recv", interrupt_on_batch)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, protocol.raw, ckpts, options={"hang_at_batch": 0})
    report = json.loads((tmp_path / "run" / "runtime_report.json").read_text())
    (close,) = report["close_reports"]
    assert any("in flight" in e for e in close["errors"])
    assert close["worker"]["ended"] in ("terminated", "killed")
    assert _gone(close["worker"]["pid"])


def _engine(protocol, ckpts, **options):
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB, options=options,
        timeouts=pw.Timeouts(load=30, batch=3, close=5, kill_grace=2),
    )  # fmt: skip
    return loader()


def test_sigint_reaching_the_worker_is_ignored(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts)
    os.kill(e.pid, signal.SIGINT)  # as a terminal Ctrl-C would, to the whole process group
    time.sleep(0.3)
    assert e._proc.is_alive()  # the parent decides when the worker stops
    (item,) = e.generate_batch([t._req(protocol, n=2)])
    assert len(item.result) == 2
    e.close()
    assert e._proc.exitcode == 0


def test_requests_round_trip_with_their_identity(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts)
    reqs = [t._req(protocol, item=f"dev-{k}", prompt=f"Synthetic: compute {k} + 1.", n=n, seed=s)
            for k, (n, s) in enumerate([(1, 0), (3, 7), (2, 1)])]  # fmt: skip
    out = e.generate_batch(reqs)
    assert [b.request for b in out] == reqs  # the parent's own request objects
    assert not isinstance(out[0].result, list) and len(out[1].result) == 3
    assert out[1].result[0].text.endswith("\\boxed{2}.")
    e.close()
    e.close()  # idempotent after success
    with pytest.raises(RuntimeError, match="closed"):
        e.generate_batch(reqs)


# --- malformed worker output ----------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        "exit_before_ready",
        "ready_not_json",
        "ready_without_load_id",
        "ready_partial_frame",
        "ready_partial_header",
        "ready_nan",
    ],
)
def test_malformed_or_missing_ready_fails_the_load(protocol, ckpts, mode) -> None:
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB, options={"misbehave": mode},
        timeouts=pw.Timeouts(load=5, batch=2, close=2, kill_grace=1),
        worker=stubs.misbehaving_worker,
    )  # fmt: skip
    started = time.monotonic()
    with pytest.raises(pw.WorkerError):
        loader()
    assert time.monotonic() - started < 15  # a partial message times out like no message


@pytest.mark.parametrize(
    "mode",
    [
        "batch_not_json",
        "batch_wrong_kind",
        "batch_wrong_count",
        "batch_echo_mismatch",
        "batch_string_tokens",
        "batch_bool_tokens",
        "batch_extra_key",
        "batch_exit",
        "batch_partial_frame",
        "batch_oversized",
        "batch_deep_nesting",
        "batch_nan",
        "batch_infinity",
        "batch_overflowing_float",
        "batch_echo_bool_seed",
        "batch_echo_float_n",
    ],
)
def test_malformed_batch_replies_fail_the_batch_and_the_arm(protocol, ckpts, mode) -> None:
    reports: list = []
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB, options={"misbehave": mode},
        timeouts=pw.Timeouts(load=10, batch=2, close=2, kill_grace=1),
        worker=stubs.misbehaving_worker, reports=reports,
    )  # fmt: skip
    e = loader()
    reqs = [
        t._req(protocol, item=f"dev-{k}", prompt=f"Synthetic: compute {k} + 1.") for k in range(2)
    ]
    with pytest.raises((pw.WorkerProtocolError, pw.WorkerDied, pw.WorkerTimeout)):
        e.generate_batch(reqs)
    with pytest.raises(pw.WorkerDied, match="not usable"):
        e.generate_batch(reqs)  # every later batch of the arm fails at once
    with pytest.raises(pw.WorkerCloseError):
        e.close()
    assert e._proc.exitcode is not None  # the misbehaving worker was stopped and reaped
    assert reports[-1]["errors"]


# --- probe ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("probe", "error", "message"),
    [
        ("wsp_process_stubs:failing_probe", pw.WorkerError, "synthetic probe failure"),
        ("wsp_process_stubs:probe_with_other_version", t.ProvenanceError, "module __version__"),
        ("wsp_process_stubs:probe_exiting_nonzero", pw.WorkerError, "exited with code 5"),
        ("wsp_process_stubs:probe_malformed", pw.WorkerProtocolError, "module versions"),
    ],
)
def test_the_runtime_probe_is_checked_before_any_output(
    tmp_path, protocol, ckpts, probe, error, message
) -> None:
    with pytest.raises(error, match=message):
        _run(tmp_path, protocol.raw, ckpts, probe=probe)
    assert not (tmp_path / "run").exists()


def test_the_probe_imports_ml_libraries_only_in_its_own_process(tmp_path, protocol, ckpts) -> None:
    _run(tmp_path, protocol.raw, ckpts, probe="wsp_process_stubs:probe_importing_sentinel")
    assert "wsp_sentinel_ml" not in sys.modules


# --- the real adapter in a worker -----------------------------------------------------------


def test_the_vllm_adapter_runs_inside_a_worker(tmp_path, protocol, ckpts) -> None:
    summary = _run(
        tmp_path, protocol.raw, ckpts, factory="wsp_process_stubs:vllm_stub_engine",
        options={"storage_root": str(tmp_path / "ckpt")},
    )  # fmt: skip
    assert summary["run_status"] == "complete"
    engine_reports = [r["engine_close_report"] for r in summary["close_reports"]]
    assert [r["engine_object_after_release"] for r in engine_reports] == ["released"] * 3
    assert all(r["errors"] == [] for r in engine_reports)


def test_the_worker_refuses_a_protocol_whose_hash_differs(protocol, ckpts) -> None:
    from types import SimpleNamespace

    forged = SimpleNamespace(raw=protocol.raw, sha256="0" * 64)
    loader = pw.process_arm_loader(
        forged, "base", ckpts[0], engine_factory=STUB,
        timeouts=pw.Timeouts(load=10, batch=2, close=2, kill_grace=1),
    )  # fmt: skip
    with pytest.raises(pw.WorkerLoadError, match="protocol hash differs"):
        loader()


def test_the_command_line_always_uses_process_isolation(tmp_path, protocol, ckpts) -> None:
    import inspect

    assert inspect.signature(t.runtime_cli.run).parameters["isolation"].default == "process"
    argv = [*t._argv(tmp_path, ckpts, protocol_raw=protocol.raw), "--isolation", "in-process"]
    with pytest.raises(SystemExit):  # no command-line option selects the in-process seam
        t.runtime_cli.run(argv)


# --- R30 fix round: whole-message deadlines, strict JSON, timeouts ------------------------


def test_a_partial_reply_that_stalls_times_out_and_kills_the_worker(protocol, ckpts) -> None:
    reports: list = []
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB,
        options={"misbehave": "batch_partial_frame"},
        timeouts=pw.Timeouts(load=10, batch=1, close=2, kill_grace=1),
        worker=stubs.misbehaving_worker, reports=reports,
    )  # fmt: skip
    e = loader()
    started = time.monotonic()
    with pytest.raises(pw.WorkerTimeout, match="incomplete message"):
        e.generate_batch([t._req(protocol)])
    assert time.monotonic() - started < 5
    assert e._proc.exitcode is not None and e._proc.exitcode < 0  # killed, and reaped
    with pytest.raises(pw.WorkerCloseError):
        e.close()
    assert any("WorkerTimeout" in x for x in reports[-1]["errors"])


def test_a_partial_reply_that_stalls_aborts_the_run_with_its_records(
    tmp_path, protocol, ckpts
) -> None:
    with pytest.raises(t.runtime_cli._Aborted):
        _run(tmp_path, protocol.raw, ckpts, options={"misbehave": "batch_partial_frame"},
             worker=stubs.misbehaving_worker, extra=["--batch-timeout", "1"])  # fmt: skip
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["run_status"] == "aborted"
    assert [x["status"] for x in manifest["execution"]["loads"]] == ["close_failed"]
    records = [json.loads(x) for x in (tmp_path / "run" / "records.jsonl").read_text().splitlines()]
    assert len(records) == 5 * 4 and {r["status"] for r in records} == {"error"}
    assert {r["error_class"].split(":")[-1] for r in records} == {"WorkerTimeout", "WorkerDied"}


def test_a_frame_read_has_one_deadline_for_the_whole_message() -> None:
    a, b = socket.socketpair()
    with a, b:
        a.setblocking(False)
        b.sendall(b"\x00\x00\x00\x00\x00\x00\x00\x10" + b"0123")  # 16 announced, 4 sent
        started = time.monotonic()
        with pytest.raises(pw.WorkerTimeout, match="12 bytes of an incomplete message"):
            pw._read_frame(a.fileno(), pw._Clock(0.5))
        assert time.monotonic() - started < 2


def test_a_frame_write_to_a_worker_that_does_not_read_times_out() -> None:
    a, b = socket.socketpair()
    with a, b:
        a.setblocking(False)  # as the parent's end is
        started = time.monotonic()
        with pytest.raises(pw.WorkerTimeout, match="did not take the whole message"):
            pw._write_frame(a.fileno(), b"x" * (64 * 1024 * 1024), pw._Clock(0.5))
        assert time.monotonic() - started < 3


def test_an_oversized_frame_is_refused_before_it_is_read() -> None:
    a, b = socket.socketpair()
    with a, b:
        a.setblocking(False)
        b.sendall((pw.MAX_MESSAGE_BYTES + 1).to_bytes(8, "big"))
        with pytest.raises(pw.WorkerProtocolError, match="limit"):
            pw._read_frame(a.fileno(), pw._Clock(1))


@pytest.mark.parametrize(
    "text",
    [b'{"kind": "x", "payload": NaN}', b'{"kind": "x", "payload": -Infinity}',
     b'{"kind": "x", "payload": 1e400}', b"[" * 100_000 + b"]" * 100_000,
     b'{"kind": "x", "payload": 1' + b"0" * 5000 + b"}"],
)  # fmt: skip
def test_only_strict_finite_json_is_accepted(text) -> None:
    with pytest.raises(pw.WorkerProtocolError, match="not valid JSON"):
        pw._decode(text, ("x",))
    assert pw._decode(b'{"kind": "x", "payload": [1.5, -2, 1e300]}', ("x",)) == (
        "x", [1.5, -2, 1e300],
    )  # fmt: skip


def test_an_unexpected_error_after_sending_a_batch_marks_the_worker_dead(
    protocol, ckpts, monkeypatch
) -> None:
    e = _engine(protocol, ckpts)

    def boom(*args):
        raise MemoryError("synthetic")

    monkeypatch.setattr(pw, "decode_items", boom)
    with pytest.raises(pw.WorkerDied, match="MemoryError"):
        e.generate_batch([t._req(protocol)])
    assert e._dead and not e._in_flight and e._proc.exitcode is not None
    with pytest.raises(pw.WorkerDied, match="not usable"):
        e.generate_batch([t._req(protocol)])
    with pytest.raises(pw.WorkerCloseError):
        e.close()


@pytest.mark.parametrize(
    "kwargs",
    [{"load": math.nan}, {"batch": math.inf}, {"close": 0}, {"close": -1.0},
     {"load": True}, {"load": "5"}, {"batch": pw.MAX_TIMEOUT * 2}, {"kill_grace": 0.0},
     {"kill_grace": pw.MIN_KILL_GRACE / 2}],
)  # fmt: skip
def test_invalid_timeouts_are_refused(kwargs) -> None:
    with pytest.raises(ValueError):
        pw.Timeouts(**kwargs)


@pytest.mark.parametrize(
    "extra",
    [["--kill-grace", "nan"], ["--kill-grace", "0"], ["--kill-grace", "-1"],
     ["--load-timeout", "inf"], ["--batch-timeout", "0"], ["--close-timeout", "abc"]],
)  # fmt: skip
def test_the_command_line_refuses_invalid_timeouts_before_any_work(
    tmp_path, protocol, ckpts, extra, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, protocol.raw, ckpts, extra=extra)
    assert exc.value.code == 2
    assert not (tmp_path / "run").exists()


# --- R30 fix round: probe read-backs ----------------------------------------------------


@pytest.mark.parametrize(
    ("probe", "error", "message"),
    [
        ("wsp_process_stubs:probe_without_sampling", pw.WorkerProtocolError, "read back"),
        ("wsp_process_stubs:probe_missing_a_shape", pw.WorkerProtocolError, "read back"),
        ("wsp_process_stubs:probe_changed_value", t.SettingsError, "did not keep"),
    ],
)
def test_the_probe_sampling_read_back_is_checked_in_the_parent(
    tmp_path, protocol, ckpts, probe, error, message
) -> None:
    with pytest.raises(error, match=message):
        _run(tmp_path, protocol.raw, ckpts, probe=probe)
    assert not (tmp_path / "run").exists()


def test_the_preflight_records_the_probe_read_back(tmp_path, protocol, ckpts) -> None:
    _run(tmp_path, protocol.raw, ckpts)
    report = json.loads((tmp_path / "run" / "runtime_report.json").read_text())
    for entry in report["preflight"]:
        assert entry["sampling_built_in"] == "probe_worker"
        assert entry["sampling_shapes_checked"]  # not [] as before


# --- R30 fix round: interrupts during cleanup ---------------------------------------------


def test_a_second_interrupt_during_cleanup_still_stops_the_worker(
    protocol, ckpts, monkeypatch
) -> None:
    reports: list = []
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB, options={"hang_at_batch": 0},
        timeouts=pw.Timeouts(load=30, batch=30, close=5, kill_grace=2), reports=reports,
    )  # fmt: skip
    e = loader()
    original_recv = pw._recv

    def first_interrupt(conn, proc, clock, expected):
        if "batch" in expected:
            time.sleep(0.3)  # the worker is now busy in its batch
            raise KeyboardInterrupt  # the first Ctrl-C
        return original_recv(conn, proc, clock, expected)

    monkeypatch.setattr(pw, "_recv", first_interrupt)
    with pytest.raises(KeyboardInterrupt):
        e.generate_batch([t._req(protocol)])
    assert e._in_flight

    original_signal = pw._signal
    sent: list = []

    def second_interrupt(proc, pgid, sig):
        if not sent:
            sent.append(sig)
            os.kill(os.getpid(), signal.SIGINT)  # the second Ctrl-C, in the middle of stopping
            time.sleep(0.05)
        return original_signal(proc, pgid, sig)

    monkeypatch.setattr(pw, "_signal", second_interrupt)
    with pytest.raises(KeyboardInterrupt):  # delivered once the cleanup has finished
        e.close()
    assert sent == [signal.SIGTERM]
    assert e._proc.exitcode is not None and _gone(e.pid)  # stopped and reaped
    assert e._conn.closed
    (report,) = reports
    assert report["phase"] == "close" and any("in flight" in x for x in report["errors"])
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # handler restored


def test_an_interrupt_during_the_probe_stops_it_without_waiting_the_grace(
    tmp_path, protocol, monkeypatch
) -> None:
    original_recv = pw._recv
    seen: list = []

    def interrupt(conn, proc, clock, expected):
        if "probe" in expected:
            seen.append(proc)
            raise KeyboardInterrupt
        return original_recv(conn, proc, clock, expected)

    monkeypatch.setattr(pw, "_recv", interrupt)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        pw.run_probe({"protocol": protocol.raw, "sample_shapes": [[1, 0]]}, timeout=30,
                     probe="wsp_process_stubs:hanging_probe", grace=20)  # fmt: skip
    assert time.monotonic() - started < 10  # not the 20 s grace
    (proc,) = seen
    assert proc.exitcode is not None and proc.exitcode < 0


# --- R30 fix round: process groups --------------------------------------------------------


def test_each_worker_leads_its_own_process_group(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts)
    assert os.getpgid(e.pid) == e.pid == e.pgid != os.getpgrp()
    assert e.describe()["isolation"]["pgid"] == e.pid
    e.close()
    assert e.close_report["worker"]["process_group"] == ["none"]


def test_only_worker_groups_are_ever_signalled(tmp_path, protocol, ckpts, monkeypatch) -> None:
    calls: list = []
    original = os.killpg

    def record(pgid, sig):
        calls.append(pgid)
        return original(pgid, sig)

    monkeypatch.setattr(pw.os, "killpg", record)
    with pytest.raises(t.runtime_cli._Aborted) as exc:
        _run(tmp_path, protocol.raw, ckpts, options={"hang_at_batch": 0},
             extra=["--batch-timeout", "1"])  # fmt: skip
    (report,) = exc.value.summary["close_reports"]
    # The probe's group (checked after it exited) and the arm worker's group, nothing else.
    assert report["worker"]["pgid"] in calls and len(set(calls)) == 2
    assert os.getpgrp() not in calls and os.getpid() not in calls


def test_the_parent_group_is_never_signalled() -> None:
    with pytest.raises(RuntimeError, match="refusing"):
        pw._killpg(os.getpgrp(), 0)
    assert pw._group_of(os.getpid()) is None  # this process's own group
    ctx = multiprocessing.get_context("spawn")
    plain = ctx.Process(target=time.sleep, args=(30,))  # no _bootstrap: stays in our group
    plain.start()
    try:
        assert pw._group_of(plain.pid) is None
        ended, group = pw._stop(plain, 2, pgid=pw._group_of(plain.pid))
        assert ended == "terminated" and group == "not_checked"
    finally:
        plain.kill()
        plain.join(5)


def test_a_worker_whose_group_cannot_be_verified_is_refused(protocol, ckpts, monkeypatch) -> None:
    monkeypatch.setattr(pw, "_group_of", lambda pid: None)
    killpg: list = []
    monkeypatch.setattr(pw.os, "killpg", lambda *a: killpg.append(a))
    reports: list = []
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB,
        timeouts=pw.Timeouts(load=30, batch=3, close=5, kill_grace=2), reports=reports,
    )  # fmt: skip
    with pytest.raises(pw.WorkerProtocolError, match="own process group"):
        loader()
    assert killpg == []  # stopped as a single process, no group signalled
    assert reports[-1]["phase"] == "load_failed"
    assert reports[-1]["worker"]["exitcode"] is not None


def test_a_process_left_in_the_worker_group_fails_the_close(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts, descendant="in_group")
    child = e.describe()["descendant_pid"]
    assert os.getpgid(child) == e.pgid
    with pytest.raises(pw.WorkerCloseError, match="outlived it and had to be killed"):
        e.close()
    assert e._proc.exitcode == 0  # the worker itself closed cleanly
    assert e.close_report["worker"]["process_group"] == ["killed"]
    _wait_gone(child)


def test_a_timeout_stops_the_whole_worker_group(protocol, ckpts, tmp_path) -> None:
    marker = tmp_path / "descendant"
    e = _engine(protocol, ckpts, descendant="in_group", descendant_marker=str(marker),
                hang_at_batch=0)  # fmt: skip
    child = e.describe()["descendant_pid"]
    with pytest.raises(pw.WorkerTimeout):
        e.generate_batch([t._req(protocol)])
    _wait_gone(child)
    # The group was terminated while the worker was alive, so the engine's own process got
    # SIGTERM (a chance to clean up), not only the SIGKILL of the sweep after the worker exited.
    assert marker.read_text() == "SIGTERM"
    with pytest.raises(pw.WorkerCloseError):
        e.close()


def test_a_descendant_that_leaves_the_group_escapes_it(protocol, ckpts) -> None:
    # The documented limit: a descendant in a new session is neither seen nor stopped.
    e = _engine(protocol, ckpts, descendant="detached")
    child = e.describe()["descendant_pid"]
    try:
        _wait_until(lambda: os.getpgid(child) == child)
        e.close()  # nothing is left in the group, so the close succeeds
        assert e.close_report["worker"]["process_group"] == ["none"]
        assert not _gone(child)  # ...yet the escaped process is still running
    finally:
        with __import__("contextlib").suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)
    _wait_gone(child)


def test_the_worker_releases_its_engine_when_the_parent_goes_away(protocol, ckpts, tmp_path):
    marker = tmp_path / "closed"
    e = _engine(protocol, ckpts, close_marker=str(marker))
    e._conn.close()  # the parent end goes away without a close message
    e._proc.join(10)
    assert e._proc.exitcode == 4 and marker.read_text() == "closed"
    with pytest.raises(pw.WorkerCloseError, match="already exited"):
        e.close()


# --- re-check fix round: permission errors, start-up interrupts, stop labels ---------------


def _deny_killpg(monkeypatch, denied):
    """os.killpg raising PermissionError for the signals in ``denied`` (as for a process this
    process may not signal, e.g. a setuid program left in the group)."""
    original = os.killpg
    calls: list = []

    def killpg(pgid, sig):
        calls.append(sig)
        if sig in denied:
            raise PermissionError(1, "Operation not permitted")
        return original(pgid, sig)

    monkeypatch.setattr(pw.os, "killpg", killpg)
    return calls


@pytest.mark.parametrize("denied", [{signal.SIGKILL}, {0, signal.SIGKILL}], ids=["kill", "check"])
def test_a_group_that_cannot_be_signalled_is_unknown_and_fails_the_close(
    protocol, ckpts, monkeypatch, denied
) -> None:
    reports: list = []
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB, options={"descendant": "in_group"},
        timeouts=pw.Timeouts(load=30, batch=3, close=5, kill_grace=1), reports=reports,
    )  # fmt: skip
    e = loader()
    child = e.describe()["descendant_pid"]
    try:
        _deny_killpg(monkeypatch, denied)
        with pytest.raises(pw.WorkerCloseError, match="may still be running"):
            e.close()
        (report,) = reports  # the close report is still recorded
        assert report["phase"] == "close" and report["worker"]["process_group"] == ["unknown"]
        assert not any("killed" in x for x in report["errors"])  # not reported as stopped
        assert e._conn.closed and e._proc.exitcode == 0
        assert not _gone(child)  # indeed: the process it could not signal is still there
        with pytest.raises(pw.WorkerCloseError, match="earlier close failed"):
            e.close()  # a failed close keeps failing
    finally:
        monkeypatch.undo()
        with __import__("contextlib").suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)
    _wait_gone(child)


def test_a_group_signal_that_is_refused_falls_back_to_the_worker(protocol, ckpts, monkeypatch):
    e = _engine(protocol, ckpts, hang_at_batch=0)
    calls = _deny_killpg(monkeypatch, {signal.SIGTERM, signal.SIGKILL})
    with pytest.raises(pw.WorkerTimeout):
        e.generate_batch([t._req(protocol)])
    assert signal.SIGTERM in calls  # the group was tried first...
    assert e._proc.exitcode == -signal.SIGTERM  # ...then the worker itself was terminated
    with pytest.raises(pw.WorkerCloseError):
        e.close()
    assert e.close_report["worker"]["ended"] == "terminated" and e._conn.closed


def test_a_worker_that_cannot_be_signalled_is_never_reported_stopped(
    protocol, ckpts, monkeypatch
) -> None:
    e = _engine(protocol, ckpts, hang_at_batch=0)
    proc = e._proc
    _deny_killpg(monkeypatch, {signal.SIGTERM, signal.SIGKILL})

    def denied(*a):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(proc, "terminate", denied)
    monkeypatch.setattr(proc, "kill", denied)
    try:
        with pytest.raises(pw.WorkerTimeout):
            e.generate_batch([t._req(protocol)])
        with pytest.raises(pw.WorkerCloseError, match="could not be stopped"):
            e.close()
        assert e.close_report["worker"]["ended"] == "still_running"
        assert e._conn.closed and proc.exitcode is None  # honestly: it is still running
    finally:
        monkeypatch.undo()
        proc.kill()
        pw._join(proc, 5)


def test_an_error_while_stopping_still_records_the_report_and_closes_the_pipe(
    protocol, ckpts, monkeypatch
) -> None:
    e = _engine(protocol, ckpts)
    proc = e._proc

    def broken(*a, **k):
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(pw, "_stop", broken)
    try:
        with pytest.raises(pw.WorkerCloseError, match="synthetic cleanup failure"):
            e.close()
        report = e.close_report
        assert report["phase"] == "close" and report["worker"]["ended"] == "unknown"
        assert report["worker"]["process_group"] == ["unknown"] and e._conn.closed
    finally:
        monkeypatch.undo()
        proc.kill()
        pw._join(proc, 5)


@pytest.mark.parametrize("where", ["set_blocking", "engine"])
def test_an_interrupt_while_a_worker_starts_still_stops_it(
    protocol, ckpts, monkeypatch, where
) -> None:
    # A Ctrl-C landing just after the worker process started (inside _spawn, or while the
    # parent-side engine object is created) is held, then raised where the worker is stopped.
    def interrupt_once():
        if not fired:
            fired.append(True)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)

    fired: list = []
    if where == "set_blocking":
        original = os.set_blocking

        def set_blocking(fd, flag):
            interrupt_once()
            return original(fd, flag)

        monkeypatch.setattr(pw.os, "set_blocking", set_blocking)
    else:
        original_init = pw.WorkerEngine.__init__

        def init(self, *a, **k):
            original_init(self, *a, **k)
            interrupt_once()

        monkeypatch.setattr(pw.WorkerEngine, "__init__", init)
    reports: list = []
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB,
        timeouts=pw.Timeouts(load=30, batch=3, close=5, kill_grace=2), reports=reports,
    )  # fmt: skip
    with pytest.raises(KeyboardInterrupt):
        loader()
    assert fired
    (report,) = reports
    assert report["phase"] == "load_failed" and "KeyboardInterrupt" in report["error"]
    assert report["worker"]["exitcode"] is not None and _gone(report["worker"]["pid"])
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_an_interrupt_while_the_probe_starts_still_stops_it(protocol, monkeypatch) -> None:
    original = os.set_blocking
    started: list = []
    original_start = multiprocessing.get_context("spawn").Process.start

    def start(self):
        original_start(self)
        started.append(self)

    def set_blocking(fd, flag):
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        return original(fd, flag)

    monkeypatch.setattr(multiprocessing.get_context("spawn").Process, "start", start)
    monkeypatch.setattr(pw.os, "set_blocking", set_blocking)
    with pytest.raises(KeyboardInterrupt):
        pw.run_probe({"protocol": protocol.raw, "sample_shapes": [[1, 0]]}, timeout=30,
                     probe="wsp_process_stubs:hanging_probe", grace=2)  # fmt: skip
    (proc,) = started
    assert proc.exitcode is not None and proc.exitcode < 0


def test_a_failure_after_the_worker_process_started_stops_it(protocol, ckpts, monkeypatch):
    started: list = []
    original_start = multiprocessing.get_context("spawn").Process.start

    def start(self):
        original_start(self)
        started.append(self)

    def set_blocking(fd, flag):
        raise OSError("synthetic set_blocking failure")

    monkeypatch.setattr(multiprocessing.get_context("spawn").Process, "start", start)
    monkeypatch.setattr(pw.os, "set_blocking", set_blocking)
    with pytest.raises(OSError, match="synthetic"):
        _engine(protocol, ckpts)
    (proc,) = started
    assert proc.exitcode is not None and _gone(proc.pid)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with __import__("contextlib").suppress(OSError):
            if predicate():
                return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def _wait_gone(pid, timeout=5.0):
    _wait_until(lambda: _gone(pid), timeout)

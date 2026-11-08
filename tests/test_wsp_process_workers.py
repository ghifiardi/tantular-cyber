"""One worker process per arm (tools/wsp_runtime/wsp_runtime/process_worker.py), tested with stub
engines and stub workers only (tests/wsp_process_stubs.py). No model, GPU, network or ML library.

These tests show process control flow: separate worker processes, request/sample identity, shard
order, explicit failures, aborted manifests and that a worker has exited before the next arm
starts. A worker that exited does not show that a real engine released the GPU."""

# ruff: noqa: F811 - the fixtures imported from test_wsp_runtime are used as parameters
from __future__ import annotations

import ast
import json
import multiprocessing
import os
import signal
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
    for p in multiprocessing.active_children():  # reaps finished ones as a side effect
        p.kill()
        p.join(5)
    assert multiprocessing.active_children() == []


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


@pytest.mark.parametrize("mode", ["exit_before_ready", "ready_not_json", "ready_without_load_id"])
def test_malformed_or_missing_ready_fails_the_load(protocol, ckpts, mode) -> None:
    loader = pw.process_arm_loader(
        protocol, "base", ckpts[0], engine_factory=STUB, options={"misbehave": mode},
        timeouts=pw.Timeouts(load=5, batch=2, close=2, kill_grace=1),
        worker=stubs.misbehaving_worker,
    )  # fmt: skip
    with pytest.raises(pw.WorkerError):
        loader()


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
    with pytest.raises((pw.WorkerProtocolError, pw.WorkerDied)):
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

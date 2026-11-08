"""Stub engines, probes and misbehaving workers for the per-arm process tests.

Imported by name inside the **worker processes** (``spawn`` start method); the test process only
passes ``module:name`` references. No model, GPU or ML library: the engines answer synthetic
"compute A + B" prompts. Not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from wsp_eval.generation import BatchItem, Generation
from wsp_runtime import process_worker as pw

STUB_VERSION = "0.0.0-stub"
STUB_TF_VERSION = "0.0.0-stub-transformers"


def _answer(prompt: str) -> int:
    expr = prompt.split("compute ", 1)[1].split(".", 1)[0]
    a, b = (int(x) for x in expr.split(" + "))
    return a + b


class StubEngine:
    def __init__(self, arm: Any, options: dict[str, Any]):
        self.arm, self.options = arm, options
        self.batches: list[list[str]] = []
        self.load_id = f"{arm.id}-stub-{os.getpid()}"
        self.close_report: dict[str, Any] | None = None
        if options.get("import_sentinel"):
            import wsp_sentinel_ml  # noqa: F401 - must only ever be imported in a worker

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "stub-worker",
            "model": self.arm.checkpoint.get("repo"),
            "revision": self.arm.checkpoint.get("revision"),
            "engine": "stub",
            "engine_version": STUB_VERSION,
            "artifacts_sha256": {"weights": "0" * 64},
            "load_id": self.load_id,
            "synthetic": False,
            "worker_pid": os.getpid(),
        }

    def generate_batch(self, requests: list[Any]) -> list[BatchItem]:
        k = len(self.batches)
        self.batches.append([r.item_id for r in requests])
        o = self.options
        if o.get("crash_at_batch") == k:
            os._exit(9)
        if o.get("hang_at_batch") == k:
            time.sleep(3600)
        if o.get("batch_error_at") == k:
            raise RuntimeError("synthetic batch failure")
        out = []
        for r in requests:
            if r.item_id == o.get("request_error_item"):
                out.append(BatchItem(r, ValueError("synthetic request failure")))
                continue
            text = f"<think>w</think>The answer is \\boxed{{{_answer(r.prompt)}}}."
            gens = [
                Generation(text=text, output_tokens=7, finish_reason="stop") for _ in range(r.n)
            ]
            out.append(BatchItem(r, gens if r.n > 1 else gens[0]))
        return out

    def close(self) -> None:
        o = self.options
        if o.get("hang_on_close"):
            time.sleep(3600)
        self.close_report = {"phase": "close", "batches": self.batches, "pid": os.getpid()}
        if o.get("close_raises"):
            raise RuntimeError("synthetic close failure")
        if o.get("exit_code_after_close"):
            atexit.register(os._exit, o["exit_code_after_close"])
        if o.get("linger_after_close"):
            atexit.register(time.sleep, 3600)


def stub_engine(arm: Any, checkpoint_dir: Path, options: dict[str, Any]) -> StubEngine:
    if options.get("load_error"):
        raise RuntimeError("synthetic load failure")
    return StubEngine(arm, options)


def vllm_stub_engine(arm: Any, checkpoint_dir: Path, options: dict[str, Any]) -> Any:
    """The real VllmEngine in the worker, on the test suite's stand-in vLLM backend."""
    import test_wsp_runtime as t
    from wsp_runtime.vllm_adapter import VllmEngine

    return VllmEngine(
        arm,
        checkpoint_dir,
        t.FakeBackend(),
        storage_check=t._storage_for(Path(options["storage_root"])),
    )


def stub_probe(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "modules": {"vllm": STUB_VERSION, "transformers": STUB_TF_VERSION},
        "sampling": {},
        "probe_pid": os.getpid(),
    }


def probe_with_other_version(spec: dict[str, Any]) -> dict[str, Any]:
    return {"modules": {"vllm": "9.9.9", "transformers": STUB_TF_VERSION}, "sampling": {}}


def probe_importing_sentinel(spec: dict[str, Any]) -> dict[str, Any]:
    import wsp_sentinel_ml  # noqa: F401

    return stub_probe(spec)


def failing_probe(spec: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("synthetic probe failure")


def misbehaving_worker(conn: Any, spec: dict[str, Any]) -> None:
    """A worker that breaks the wire protocol in one specific way (``options.misbehave``)."""
    mode = spec["options"]["misbehave"]
    if mode == "exit_before_ready":
        sys.exit(0)
    if mode == "ready_not_json":
        conn.send_bytes(b"\x00 not json")
        time.sleep(3600)
    if mode == "ready_without_load_id":
        pw.send(conn, "ready", {"describe": {"provider": "x"}})
        time.sleep(3600)
    pw.send(conn, "ready", {"describe": {"load_id": f"bad-{os.getpid()}"}})
    data = conn.recv_bytes()
    requests = json.loads(data)["payload"]["requests"]

    def item(req: dict[str, Any], **gen: Any) -> dict[str, Any]:
        g = {
            "text": "\\boxed{1}",
            "output_tokens": 1,
            "hit_token_limit": None,
            "finish_reason": "stop",
            "rendered_prompt_sha256": None,
            "usage": {},
            **gen,
        }
        return {"request": req, "result": {"generations": [g] * req["n"], "single": False}}

    if mode == "batch_not_json":
        conn.send_bytes(b"{not json")
    elif mode == "batch_wrong_kind":
        pw.send(conn, "closed", {"close_report": None})
    elif mode == "batch_wrong_count":
        pw.send(conn, "batch", {"items": [item(r) for r in requests][:-1]})
    elif mode == "batch_echo_mismatch":
        items = [item(r) for r in requests]
        items[0]["request"] = {**items[0]["request"], "item_id": "someone-else"}
        pw.send(conn, "batch", {"items": items})
    elif mode == "batch_string_tokens":
        pw.send(conn, "batch", {"items": [item(r, output_tokens="1") for r in requests]})
    elif mode == "batch_bool_tokens":
        pw.send(conn, "batch", {"items": [item(r, output_tokens=True) for r in requests]})
    elif mode == "batch_extra_key":
        items = [item(r) for r in requests]
        items[0]["extra"] = 1
        pw.send(conn, "batch", {"items": items})
    elif mode == "batch_exit":
        sys.exit(0)
    time.sleep(3600)


def probe_exiting_nonzero(spec: dict[str, Any]) -> dict[str, Any]:
    atexit.register(os._exit, 5)  # replies normally, then the process exits with code 5
    return stub_probe(spec)


def probe_malformed(spec: dict[str, Any]) -> dict[str, Any]:
    return {"modules": "0.0.0-stub", "sampling": {}}

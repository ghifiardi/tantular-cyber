"""C1: a vLLM inference engine for ``per_arm_sequential`` WSP runs.

:class:`VllmEngine` implements the harness's ``BatchGenerator`` interface for one arm, and
:func:`arm_loader` returns the ``ArmLoader`` the runner calls once per arm. Every engine:

* **checks provenance before loading**: hashes every file in the checkpoint directory it is
  given, compares the registered weights file with the protocol's ``weights_sha256_lfs`` and the
  chat template with the protocol's git blob id, and refuses a mismatch (nothing is loaded);
* **reports a distinct load identity** (a fresh ``load_id`` per load) and the hashed files in
  ``describe()["artifacts_sha256"]``, plus the effective sampling and engine settings, the
  engine version and the runtime's own code hash;
* **echoes requests**: each ``BatchItem`` carries the request it answers, and the engine checks
  that every vLLM output's prompt is the prompt it rendered for that request before attaching
  it (a mismatch fails the whole batch rather than mis-attributing outputs);
* **closes**: ``close()`` shuts the engine down through the backend, drops its references,
  re-hashes the checkpoint (a file changed during the run fails the close) and, if a limit is
  set, fails when the backend still reports GPU memory allocated.

All of this is exercised with a stub backend. Whether vLLM's shutdown actually releases GPU
memory, which files vLLM opens, and the effective defaults it applies must be validated on the
isolated GPU host (``docs/wsp/runtime.md``). vLLM and torch are imported only by
:meth:`VllmBackend.load`, never at module import.
"""

from __future__ import annotations

import gc
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wsp_eval.generation import BatchItem, Generation, GenerationRequest
from wsp_eval.hashing import sha256_bytes
from wsp_eval.protocol import ArmSpec

from . import RUNTIME_VERSION
from .provenance import git_blob_sha1, hash_tree, runtime_code_sha256

# The protocol records the prompt in a descriptive notation: the two characters "\n" stand for
# the newline in the authors' ``problem + "\n" + instruction``. Only this exact notation is
# accepted, so a changed prompt cannot be interpreted silently.
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
PROTOCOL_USER_MESSAGE = "<problem>\\n" + INSTRUCTION

# Inference keys the engine understands. Any other key is refused rather than ignored.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens")
_INFERENCE_KEYS = {"chat_template", "enable_thinking", "user_message", *_SAMPLING_KEYS}
# Repository files that inference does not read. They are still hashed, but reported apart
# from the loaded artifacts, so a documentation difference between two repos does not look
# like a model difference (and is not hidden either).
_DOC_FILES = re.compile(r"(?i)^(readme(\.[a-z]+)?|license(\.[a-z]+)?|\.gitattributes)$")
_BLOB = re.compile(r"git blob ([0-9a-f]{40})")


class ProvenanceError(RuntimeError):
    """The files about to be loaded are not the ones the protocol registered."""


class BatchMismatch(RuntimeError):
    """vLLM returned outputs that cannot be attributed to the requests sent."""


def user_message(template: str, problem: str) -> str:
    if template != PROTOCOL_USER_MESSAGE:
        raise ValueError(
            "inference.user_message is not the registered Experiment 8 prompt notation; "
            "refusing to guess how to build the prompt"
        )
    return problem + "\n" + INSTRUCTION


def _setting(engine: Mapping[str, Any], key: str, kind: type) -> Any:
    value = engine.get(key)
    if type(value) is not kind and not (kind is float and type(value) is int):
        raise ValueError(f"engine.{key} must be set to a {kind.__name__} (got {value!r})")
    return value


@dataclass
class VllmBackend:
    """The real backend: vLLM, transformers and torch, imported only here."""

    vllm: Any
    torch: Any | None
    version: str | None
    engine: str = "vllm"
    shutdown_steps: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> VllmBackend:
        # Never download: the adapter is given local directories only.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import vllm

        try:
            import torch
        except ImportError:  # pragma: no cover - torch ships with vLLM
            torch = None
        return cls(vllm=vllm, torch=torch, version=getattr(vllm, "__version__", None))

    def make_llm(self, **kwargs: Any) -> Any:
        return self.vllm.LLM(**kwargs)

    def make_sampling_params(self, **kwargs: Any) -> Any:
        return self.vllm.SamplingParams(**kwargs)

    def default_sampling(self, llm: Any) -> dict[str, Any] | None:
        get = getattr(llm, "get_default_sampling_params", None)
        if not callable(get):
            return None
        params = get()
        fields = (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "max_tokens",
            "stop_token_ids",
        )
        return {f: _plain(getattr(params, f, None)) for f in fields}

    def shutdown(self, llm: Any) -> list[str]:
        """Best-effort engine teardown. Which of these steps exist and suffice depends on the
        vLLM version and must be validated on the GPU host."""
        steps: list[str] = []
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            for name in ("shutdown", "engine_core_shutdown"):
                fn = getattr(engine, name, None)
                if callable(fn):
                    fn()
                    steps.append(f"llm_engine.{name}")
                    break
        try:
            from vllm.distributed import parallel_state
        except ImportError:
            parallel_state = None
        for name in ("destroy_model_parallel", "destroy_distributed_environment"):
            fn = getattr(parallel_state, name, None)
            if callable(fn):
                fn()
                steps.append(f"parallel_state.{name}")
        gc.collect()
        steps.append("gc.collect")
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
            steps.append("torch.cuda.empty_cache")
        return steps

    def memory_allocated(self) -> int | None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return None
        return int(cuda.memory_allocated())

    def environment(self) -> dict[str, Any]:
        torch = self.torch
        cuda = getattr(torch, "cuda", None)
        gpu = None
        if cuda is not None and cuda.is_available():
            gpu = cuda.get_device_name(0)
        return {
            "vllm": self.version,
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "gpu": gpu,
        }


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple | set):
        return (
            sorted(_plain(v) for v in value)
            if isinstance(value, set)
            else [_plain(v) for v in value]
        )
    return repr(value)


class VllmEngine:
    """One arm's loaded engine. Create it through :func:`arm_loader`."""

    def __init__(
        self,
        arm: ArmSpec,
        checkpoint_dir: Path,
        backend: Any,
        *,
        max_memory_after_close: int | None = None,
        reports: list[dict[str, Any]] | None = None,
    ):
        self.arm = arm
        self._reports = reports
        self.checkpoint_dir = Path(checkpoint_dir)
        self.backend = backend
        self.max_memory_after_close = max_memory_after_close
        self.close_report: dict[str, Any] | None = None
        self._closed = False
        unknown = sorted(set(arm.inference) - _INFERENCE_KEYS)
        if unknown:
            raise ValueError(f"arm {arm.id}: inference keys not understood: {unknown}")
        engine = arm.engine
        if engine.get("name") != backend.engine:
            raise ValueError(f"arm {arm.id}: protocol engine is {engine.get('name')!r}")
        pinned = engine.get("version")
        if isinstance(pinned, str) and not pinned.startswith("REPLACE"):
            if backend.version != pinned:
                raise ProvenanceError(
                    f"arm {arm.id}: engine version {backend.version!r} is not the protocol's "
                    f"{pinned!r}"
                )
        else:
            raise ValueError(f"arm {arm.id}: engine.version must be pinned in the protocol")

        # Provenance before anything is loaded: the files the engine will read.
        tree = hash_tree(self.checkpoint_dir)
        self._loaded = {k: v for k, v in tree.items() if not _DOC_FILES.match(Path(k).name)}
        self._other = {k: v for k, v in tree.items() if k not in self._loaded}
        self._check_registered_files()

        kwargs = {
            "model": str(self.checkpoint_dir),
            "tokenizer": str(self.checkpoint_dir),
            "dtype": _setting(engine, "dtype", str),
            "max_model_len": _setting(engine, "max_model_len", int),
            "seed": _setting(engine, "llm_seed", int),
            "gpu_memory_utilization": float(_setting(engine, "gpu_memory_utilization", float)),
            "tensor_parallel_size": _setting(engine, "tensor_parallel_size", int),
        }
        if "generation_config" in engine:  # R6: how engine defaults are sourced
            kwargs["generation_config"] = _setting(engine, "generation_config", str)
        self._llm: Any = backend.make_llm(**kwargs)
        self._tokenizer: Any = self._llm.get_tokenizer()
        self.load_id = f"{arm.id}-{uuid.uuid4().hex}"
        self._describe = {
            "provider": "vllm",
            "model": arm.checkpoint.get("repo"),
            "revision": arm.checkpoint.get("revision"),
            "engine": backend.engine,
            "engine_version": backend.version,
            "artifacts_sha256": dict(self._loaded),
            "other_files_sha256": dict(self._other),
            "load_id": self.load_id,
            "synthetic": False,
            "runtime_version": RUNTIME_VERSION,
            "runtime_code_sha256": runtime_code_sha256(),
            "engine_kwargs": {k: v for k, v in kwargs.items() if k not in ("model", "tokenizer")},
            "sampling": {k: arm.inference.get(k) for k in _SAMPLING_KEYS},
            "chat_template_kwargs": {
                "add_generation_prompt": True,
                "enable_thinking": bool(arm.inference.get("enable_thinking")),
            },
            "engine_default_sampling": backend.default_sampling(self._llm),
            "eos_token_id": _plain(getattr(self._tokenizer, "eos_token_id", None)),
            "environment": backend.environment(),
        }

    def _check_registered_files(self) -> None:
        ckpt = self.arm.checkpoint
        weights, digest = ckpt.get("weights_file"), ckpt.get("weights_sha256_lfs")
        if not weights or weights not in self._loaded:
            raise ProvenanceError(f"arm {self.arm.id}: registered weights file {weights!r} missing")
        if digest and self._loaded[weights] != digest:
            raise ProvenanceError(
                f"arm {self.arm.id}: {weights} SHA-256 differs from the registered value"
            )
        blob = _BLOB.search(str(self.arm.inference.get("chat_template", "")))
        if blob:
            path = self.checkpoint_dir / "chat_template.jinja"
            if not path.is_file() or git_blob_sha1(path.read_bytes()) != blob.group(1):
                raise ProvenanceError(
                    f"arm {self.arm.id}: chat_template.jinja is not the registered git blob"
                )

    def describe(self) -> dict[str, Any]:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self._describe.items()}

    def _render(self, request: GenerationRequest) -> str:
        text = user_message(self.arm.inference["user_message"], request.prompt)
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(self.arm.inference.get("enable_thinking")),
        )

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[BatchItem]:
        if self._closed:
            raise RuntimeError("engine is closed")
        items: list[BatchItem | None] = [None] * len(requests)
        send: list[tuple[int, str, Any]] = []
        for k, req in enumerate(requests):
            if req.arm_id != self.arm.id or dict(req.inference) != dict(self.arm.inference):
                items[k] = BatchItem(req, ValueError("request is not for this arm's settings"))
                continue
            rendered = self._render(req)
            params = self.backend.make_sampling_params(
                n=req.n, seed=req.seed, **{key: self.arm.inference[key] for key in _SAMPLING_KEYS}
            )
            send.append((k, rendered, params))
        if send:
            outputs = self._llm.generate(
                [p for _, p, _ in send], [s for _, _, s in send], use_tqdm=False
            )
            outputs = list(outputs)
            if len(outputs) != len(send):
                raise BatchMismatch(f"sent {len(send)} prompts, vLLM returned {len(outputs)}")
            for (k, rendered, _), out in zip(send, outputs, strict=True):
                # vLLM returns outputs in input order; the prompt check makes that an asserted
                # property instead of an assumption.
                if getattr(out, "prompt", None) != rendered:
                    raise BatchMismatch(f"output {k} is not for the prompt sent at that position")
                prompt_hash = sha256_bytes(rendered.encode("utf-8"))
                gens = [
                    Generation(
                        text=c.text,
                        output_tokens=len(c.token_ids) if c.token_ids is not None else None,
                        hit_token_limit=None,
                        finish_reason=c.finish_reason,
                        rendered_prompt_sha256=prompt_hash,
                    )
                    for c in out.outputs
                ]
                req = requests[k]
                items[k] = BatchItem(
                    req, gens if req.n > 1 else (gens[0] if len(gens) == 1 else gens)
                )
        return [i for i in items if i is not None]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._close()
        finally:
            if self._reports is not None and self.close_report is not None:
                self._reports.append(dict(self.close_report))

    def _close(self) -> None:
        steps = self.backend.shutdown(self._llm)
        self._llm = None
        self._tokenizer = None
        gc.collect()
        memory = self.backend.memory_allocated()
        after = hash_tree(self.checkpoint_dir)
        self.close_report = {
            "load_id": self.load_id,
            "shutdown_steps": steps,
            "memory_allocated_after_close": memory,
            "files_unchanged": after == {**self._loaded, **self._other},
        }
        if not self.close_report["files_unchanged"]:
            raise ProvenanceError(f"arm {self.arm.id}: checkpoint files changed during the run")
        limit = self.max_memory_after_close
        if limit is not None and memory is not None and memory > limit:
            raise RuntimeError(
                f"arm {self.arm.id}: {memory} bytes still allocated after close "
                f"(limit {limit}); not loading the next arm"
            )


def arm_loader(
    arm: ArmSpec,
    checkpoint_dir: Path,
    backend_factory: Callable[[], Any] = VllmBackend.load,
    *,
    max_memory_after_close: int | None = None,
    reports: list[dict[str, Any]] | None = None,
) -> Callable[[], VllmEngine]:
    """An ``ArmLoader``: each call is a separate load with a fresh engine and ``load_id``.

    ``reports`` (optional) collects each engine's close report after the run; the loader keeps
    no reference to the engine itself, so it does not keep the engine alive.
    """

    def load() -> VllmEngine:
        return VllmEngine(
            arm,
            checkpoint_dir,
            backend_factory(),
            max_memory_after_close=max_memory_after_close,
            reports=reports,
        )

    return load

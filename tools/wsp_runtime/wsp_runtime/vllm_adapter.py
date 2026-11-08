"""C1: a vLLM inference engine for ``per_arm_sequential`` WSP runs.

:class:`VllmEngine` implements the harness's ``BatchGenerator`` interface for one arm, and
:func:`arm_loader` returns the ``ArmLoader`` the runner calls once per arm. Every engine:

* **refuses incomplete or mistyped settings**: every inference key the engine uses must be
  present with the right type, every engine key must be known, pinned and typed, and the
  installed vLLM and transformers versions must equal the protocol's;
* **checks provenance before loading**: hashes every file in the checkpoint directory; every
  loadable file must be registered in the protocol's ``checkpoint.loadable_files_sha256`` (an
  unregistered file is refused, because vLLM might load it), the weights file must match
  ``weights_sha256_lfs``, ``chat_template.jinja`` must match the protocol's git blob id, and the
  checkpoint must be read-only for this process;
* **renders with the registered template**: the bytes of ``chat_template.jinja`` that were
  checked are passed explicitly to ``apply_chat_template``;
* **reports requested and observed settings separately** (``describe()["requested"]`` and
  ``describe()["observed"]``), with a fresh ``load_id`` per load;
* **echoes requests**: each ``BatchItem`` carries the request it answers, and every vLLM
  output's prompt must be the prompt rendered for that request (a mismatch fails the batch);
* **closes step by step**: every shutdown step is attempted and its failure kept, a close
  report is always recorded, and a close that failed is never reported as a success later. An
  engine whose initialisation fails after vLLM was created is shut down before the error
  propagates.

The before/after hashes show that the checkpoint was the same at load and at close (snapshot
consistency); they do not show what vLLM read in between, which is why the files must also be
protected (read-only) during the run. All of this is exercised with a stub backend. Whether
vLLM's shutdown releases GPU memory, whether its worker processes exit, which files it opens
and the effective defaults it applies must be validated on the isolated GPU host
(``docs/wsp/runtime.md``). vLLM, transformers and torch are imported only by
:meth:`VllmBackend.load`, never at module import.
"""

from __future__ import annotations

import gc
import importlib
import multiprocessing
import os
import re
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
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

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens")
# Every one of these must be present; any other inference key is refused.
_INFERENCE_KEYS = frozenset({"chat_template", "enable_thinking", "user_message", *_SAMPLING_KEYS})
# Engine keys: all required, typed and pinned (``float`` also accepts an int, never a bool).
_ENGINE_KEYS: dict[str, type] = {
    "name": str,
    "version": str,
    "transformers_version": str,
    "dtype": str,
    "max_model_len": int,
    "llm_seed": int,
    "gpu_memory_utilization": float,
    "tensor_parallel_size": int,
    "generation_config": str,
}
# Descriptive engine keys: recorded with the protocol, not passed to vLLM; strings if present.
_ENGINE_DESCRIPTIVE = frozenset({"batching", "sharding", "hardware"})
# "auto" reads generation_config.json from the (hashed) checkpoint directory; "vllm" uses vLLM's
# own defaults. A path is refused: vLLM would read a file outside the hashed checkpoint.
_GENERATION_CONFIG = ("auto", "vllm")
# Repository files that inference does not read, at the checkpoint root only. They are hashed
# and reported apart from the loadable files; anything else must be registered.
_DOC_FILES = re.compile(r"(?i)^(?:(?:readme|license|notice)(?:\.(?:md|txt|rst))?|\.gitattributes)$")
_TEMPLATE_FILE = "chat_template.jinja"
_BLOB = re.compile(r"git blob ([0-9a-f]{40})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
_RECORDED_ENV = ("VLLM_", "HF_", "TRANSFORMERS_", "TORCH", "PYTORCH_", "CUDA_", "NCCL_")
_SECRET_ENV = re.compile(r"(?i)token|key|secret|password|credential")
_MAX_ERROR = 300


class ProvenanceError(RuntimeError):
    """The files about to be loaded are not the ones the protocol registered."""


class SettingsError(ValueError):
    """The protocol's inference or engine settings are incomplete, unknown or mistyped."""


class BatchMismatch(RuntimeError):
    """vLLM returned outputs that cannot be attributed to the requests sent."""


class CloseError(RuntimeError):
    """The engine could not be closed cleanly; the next arm must not be loaded."""


def user_message(template: str, problem: str) -> str:
    if template != PROTOCOL_USER_MESSAGE:
        raise SettingsError(
            "inference.user_message is not the registered Experiment 8 prompt notation; "
            "refusing to guess how to build the prompt"
        )
    return problem + "\n" + INSTRUCTION


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_MAX_ERROR]


def force_offline(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Set the Hugging Face libraries to offline mode. A variable already set to anything but
    ``1`` is refused rather than overridden, so an online operator environment is visible."""
    environ = os.environ if environ is None else environ
    for var in _OFFLINE_VARS:
        current = environ.get(var)
        if current is not None and current != "1":
            raise SettingsError(f"{var}={current!r} is set; the runtime runs offline only")
        environ[var] = "1"
    return {var: environ[var] for var in _OFFLINE_VARS}


def process_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Variables that can change what vLLM, torch or the Hugging Face libraries do (values of
    names that look like credentials are redacted)."""
    environ = os.environ if environ is None else environ
    return {
        k: ("<redacted>" if _SECRET_ENV.search(k) else v)
        for k, v in sorted(environ.items())
        if k.startswith(_RECORDED_ENV)
    }


def _typed(value: Any, kind: type, name: str) -> Any:
    ok = type(value) is kind or (kind is float and type(value) is int)
    if not ok:
        raise SettingsError(f"{name} must be set to a {kind.__name__} (got {value!r})")
    if kind is str and (not value.strip() or value.startswith("REPLACE")):
        raise SettingsError(f"{name} must be pinned in the protocol (got {value!r})")
    return value


def check_inference(arm: ArmSpec) -> dict[str, Any]:
    """The arm's inference settings, complete and typed; returns the sampling settings."""
    inf = arm.inference
    unknown = sorted(set(inf) - _INFERENCE_KEYS)
    if unknown:
        raise SettingsError(f"arm {arm.id}: inference keys not understood: {unknown}")
    missing = sorted(_INFERENCE_KEYS - set(inf))
    if missing:
        raise SettingsError(f"arm {arm.id}: inference keys missing: {missing}")
    if type(inf["enable_thinking"]) is not bool:
        raise SettingsError(f"arm {arm.id}: inference.enable_thinking must be true or false")
    if not isinstance(inf["chat_template"], str) or not _BLOB.search(inf["chat_template"]):
        raise SettingsError(
            f"arm {arm.id}: inference.chat_template must name the registered git blob id"
        )
    user_message(inf["user_message"], "")
    p = f"arm {arm.id}: inference"
    temperature = _typed(inf["temperature"], float, f"{p}.temperature")
    top_p = _typed(inf["top_p"], float, f"{p}.top_p")
    top_k = _typed(inf["top_k"], int, f"{p}.top_k")
    _typed(inf["presence_penalty"], float, f"{p}.presence_penalty")
    max_tokens = _typed(inf["max_tokens"], int, f"{p}.max_tokens")
    if temperature < 0 or not 0 < top_p <= 1 or not (top_k == -1 or top_k >= 1) or max_tokens < 1:
        raise SettingsError(f"{p}: temperature, top_p, top_k or max_tokens out of range")
    return {k: inf[k] for k in _SAMPLING_KEYS}


def check_engine(arm: ArmSpec, backend: Any) -> dict[str, Any]:
    """The arm's engine settings, complete, typed and matching the installed versions; returns
    the keyword arguments for ``vllm.LLM`` (without the model and tokenizer paths)."""
    eng = arm.engine
    unknown = sorted(set(eng) - set(_ENGINE_KEYS) - _ENGINE_DESCRIPTIVE)
    if unknown:
        raise SettingsError(f"arm {arm.id}: engine keys not understood: {unknown}")
    if eng.get("name") != backend.engine:
        raise SettingsError(f"arm {arm.id}: protocol engine is {eng.get('name')!r}")
    missing = sorted(set(_ENGINE_KEYS) - set(eng))
    if missing:
        raise SettingsError(f"arm {arm.id}: engine keys missing: {missing}")
    for key, kind in _ENGINE_KEYS.items():
        _typed(eng[key], kind, f"arm {arm.id}: engine.{key}")
    for key in _ENGINE_DESCRIPTIVE & set(eng):
        if not isinstance(eng[key], str):
            raise SettingsError(f"arm {arm.id}: engine.{key} must be a string")
    if backend.version != eng["version"]:
        raise ProvenanceError(
            f"arm {arm.id}: engine version {backend.version!r} is not the protocol's "
            f"{eng['version']!r}"
        )
    if backend.transformers_version != eng["transformers_version"]:
        raise ProvenanceError(
            f"arm {arm.id}: transformers {backend.transformers_version!r} is not the protocol's "
            f"{eng['transformers_version']!r}"
        )
    if eng["generation_config"] not in _GENERATION_CONFIG:
        raise SettingsError(
            f"arm {arm.id}: engine.generation_config must be one of {_GENERATION_CONFIG}; a "
            "path would make vLLM read a file outside the hashed checkpoint"
        )
    if not 0 < eng["gpu_memory_utilization"] <= 1:
        raise SettingsError(f"arm {arm.id}: engine.gpu_memory_utilization must be in (0, 1]")
    if eng["tensor_parallel_size"] < 1 or eng["max_model_len"] < 1:
        raise SettingsError(f"arm {arm.id}: tensor_parallel_size and max_model_len must be >= 1")
    if arm.inference.get("max_tokens", 0) >= eng["max_model_len"]:
        raise SettingsError(f"arm {arm.id}: inference.max_tokens must be below max_model_len")
    return {
        "dtype": eng["dtype"],
        "max_model_len": eng["max_model_len"],
        "seed": eng["llm_seed"],
        "gpu_memory_utilization": float(eng["gpu_memory_utilization"]),
        "tensor_parallel_size": eng["tensor_parallel_size"],
        "generation_config": eng["generation_config"],
        # Fail closed: never run repository code, and never fall back to pickled weights.
        "trust_remote_code": False,
        "load_format": "safetensors",
    }


def _step(name: str, fn: Callable[[], Any] | None, *, required: bool = False) -> dict[str, Any]:
    if fn is None:
        return {
            "step": name,
            "status": "unavailable",
            "error": "not provided by the installed version" if required else None,
            "required": required,
        }
    try:
        fn()
    except Exception as exc:
        return {"step": name, "status": "failed", "error": _err(exc), "required": required}
    return {"step": name, "status": "ok", "error": None, "required": required}


def step_failures(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """Failed steps, and required steps the installed version does not provide."""
    return [
        f"{s['step']}: {s['error']}"
        for s in steps
        if s["status"] == "failed" or (s["status"] == "unavailable" and s.get("required"))
    ]


@dataclass
class VllmBackend:
    """The real backend: vLLM, transformers and torch, imported only here. Untested without
    those libraries; every method must be validated on the GPU host."""

    vllm: Any
    torch: Any | None
    version: str | None
    transformers_version: str | None
    parallel_state: Any | None = None
    engine: str = "vllm"

    @classmethod
    def load(cls) -> VllmBackend:
        force_offline()  # before the libraries read the variables
        import transformers
        import vllm

        try:
            import torch
        except ImportError:  # pragma: no cover - torch ships with vLLM
            torch = None
        try:
            parallel_state = importlib.import_module("vllm.distributed.parallel_state")
        except ImportError:
            parallel_state = None
        return cls(
            vllm=vllm,
            torch=torch,
            version=getattr(vllm, "__version__", None),
            transformers_version=getattr(transformers, "__version__", None),
            parallel_state=parallel_state,
        )

    def make_llm(self, **kwargs: Any) -> Any:
        return self.vllm.LLM(**kwargs)

    def make_sampling_params(self, **kwargs: Any) -> Any:
        return self.vllm.SamplingParams(**kwargs)

    _SAMPLING_FIELDS = (
        "n",
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "max_tokens",
        "min_tokens",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "skip_special_tokens",
    )

    def describe_sampling(self, params: Any) -> dict[str, Any]:
        """The values a ``SamplingParams`` object actually holds (read back, not requested)."""
        return {f: _plain(getattr(params, f, None)) for f in self._SAMPLING_FIELDS}

    def default_sampling(self, llm: Any) -> dict[str, Any] | None:
        get = getattr(llm, "get_default_sampling_params", None)
        return self.describe_sampling(get()) if callable(get) else None

    def shutdown(self, llm: Any) -> list[dict[str, Any]]:
        """Attempt every teardown step and record each outcome; never raises. The engine's own
        shutdown is required; which steps exist and whether they free the GPU depends on the
        vLLM version and must be validated on the GPU host."""
        steps: list[dict[str, Any]] = []
        if llm is not None:
            engine = getattr(llm, "llm_engine", None)
            fn = None
            for name in ("shutdown", "engine_core_shutdown"):
                if callable(getattr(engine, name, None)):
                    fn = getattr(engine, name)
                    break
            steps.append(_step("llm_engine.shutdown", fn, required=True))
        for name in ("destroy_model_parallel", "destroy_distributed_environment"):
            fn = getattr(self.parallel_state, name, None)
            steps.append(_step(f"parallel_state.{name}", fn if callable(fn) else None))
        steps.append(_step("gc.collect", gc.collect))
        cuda = getattr(self.torch, "cuda", None)
        available = cuda is not None and cuda.is_available()
        steps.append(_step("torch.cuda.empty_cache", cuda.empty_cache if available else None))
        return steps

    def parent_torch_memory_allocated(self) -> int | None:
        """This process's PyTorch allocator only. vLLM's worker processes, CUDA contexts and
        NCCL buffers are not counted, so this never shows that the GPU was released."""
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return None
        return int(cuda.memory_allocated())

    def live_children(self) -> list[int]:
        """Child processes started through ``multiprocessing`` that are still alive. Processes
        started any other way are not seen, so an empty list does not prove workers exited."""
        return sorted(p.pid for p in multiprocessing.active_children() if p.pid is not None)

    def environment(self) -> dict[str, Any]:
        # No GPU query here: asking for the device name would create a CUDA context in this
        # process. The GPU model is recorded by the operator (nvidia-smi) on the host.
        torch = self.torch
        return {
            "vllm": self.version,
            "transformers": self.transformers_version,
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        }


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, set | frozenset):
        return sorted(_plain(v) for v in value)
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return repr(value)


def _writable(root: Path) -> list[str]:
    return [
        (p.relative_to(root).as_posix() or ".")
        for p in [root, *sorted(root.rglob("*"))]
        if os.access(p, os.W_OK)
    ]


class VllmEngine:
    """One arm's loaded engine. Create it through :func:`arm_loader`."""

    def __init__(
        self,
        arm: ArmSpec,
        checkpoint_dir: Path,
        backend: Any,
        *,
        max_parent_memory_after_close: int | None = None,
        reports: list[dict[str, Any]] | None = None,
    ):
        self.arm = arm
        self._reports = reports
        self.checkpoint_dir = Path(checkpoint_dir)
        self.backend = backend
        self.max_parent_memory_after_close = max_parent_memory_after_close
        self.close_report: dict[str, Any] | None = None
        self._close_state: str | None = None  # None, "ok" or "failed"
        self._close_error: str | None = None
        self._llm: Any = None
        self._tokenizer: Any = None

        sampling = check_inference(arm)
        kwargs = check_engine(arm, backend)

        # Provenance before anything is loaded: the files the engine will read.
        tree = hash_tree(self.checkpoint_dir)
        self._loaded = {k: v for k, v in tree.items() if not ("/" not in k and _DOC_FILES.match(k))}
        self._other = {k: v for k, v in tree.items() if k not in self._loaded}
        self._template = self._check_registered_files()
        writable = _writable(self.checkpoint_dir)
        if writable:
            raise ProvenanceError(
                f"arm {arm.id}: the checkpoint must be read-only for this process during the run "
                f"(mount it read-only); writable: {writable[:5]}"
            )

        self.load_id = f"{arm.id}-{uuid.uuid4().hex}"
        try:
            self._llm = backend.make_llm(
                model=str(self.checkpoint_dir), tokenizer=str(self.checkpoint_dir), **kwargs
            )
            self._tokenizer = self._llm.get_tokenizer()
            probe = backend.make_sampling_params(n=1, seed=0, **sampling)
            observed_sampling = backend.describe_sampling(probe)
            changed = [k for k in _SAMPLING_KEYS if observed_sampling.get(k) != sampling[k]]
            if changed:
                raise SettingsError(
                    f"arm {arm.id}: SamplingParams did not keep the requested values for {changed}"
                )
            builtin = getattr(self._tokenizer, "chat_template", None)
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
                # What the adapter asked for, from the protocol.
                "requested": {
                    "engine_kwargs": dict(kwargs),
                    "sampling": dict(sampling),
                    "chat_template_kwargs": {
                        "add_generation_prompt": True,
                        "enable_thinking": arm.inference["enable_thinking"],
                        "chat_template": f"{_TEMPLATE_FILE} (registered git blob, passed "
                        "explicitly)",
                    },
                },
                # What the backend reported back. The stop set vLLM applies at generation time
                # (after merging any generation config) is not observable here: record it on
                # the GPU host.
                "observed": {
                    "sampling_params": observed_sampling,
                    "engine_default_sampling": backend.default_sampling(self._llm),
                    "eos_token_id": _plain(getattr(self._tokenizer, "eos_token_id", None)),
                    "tokenizer_builtin_template_matches": (
                        None if builtin is None else builtin == self._template
                    ),
                    "environment": backend.environment(),
                    "process_environment": process_environment(),
                },
            }
        except BaseException as exc:
            self._cleanup_failed_load(exc)
            raise

    def _check_registered_files(self) -> str:
        """Compare the hashed directory with the protocol's registration; returns the template
        text whose bytes were checked."""
        ckpt, arm_id = self.arm.checkpoint, self.arm.id
        weights, digest = ckpt.get("weights_file"), ckpt.get("weights_sha256_lfs")
        registered = ckpt.get("loadable_files_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ProvenanceError(f"arm {arm_id}: checkpoint.weights_sha256_lfs must be registered")
        if not (
            isinstance(registered, dict)
            and registered
            and all(isinstance(v, str) and _SHA256.fullmatch(v) for v in registered.values())
        ):
            raise ProvenanceError(
                f"arm {arm_id}: checkpoint.loadable_files_sha256 must map every loadable file "
                "to its SHA-256 (record it after the approved fetch)"
            )
        if not isinstance(weights, str) or registered.get(weights) != digest:
            raise ProvenanceError(
                f"arm {arm_id}: loadable_files_sha256 must list weights_file with "
                "weights_sha256_lfs"
            )
        if _TEMPLATE_FILE not in registered:
            raise ProvenanceError(f"arm {arm_id}: loadable_files_sha256 must list {_TEMPLATE_FILE}")
        if weights not in self._loaded:
            raise ProvenanceError(f"arm {arm_id}: registered weights file {weights!r} missing")
        if self._loaded[weights] != digest:
            raise ProvenanceError(
                f"arm {arm_id}: {weights} SHA-256 differs from the registered value"
            )
        blob = _BLOB.search(self.arm.inference["chat_template"])
        path = self.checkpoint_dir / _TEMPLATE_FILE
        data = path.read_bytes() if path.is_file() else None
        if data is None or blob is None or git_blob_sha1(data) != blob.group(1):
            raise ProvenanceError(f"arm {arm_id}: {_TEMPLATE_FILE} is not the registered git blob")
        if sha256_bytes(data) != self._loaded[_TEMPLATE_FILE]:
            raise ProvenanceError(f"arm {arm_id}: {_TEMPLATE_FILE} changed while it was hashed")
        unregistered = sorted(set(self._loaded) - set(registered))
        if unregistered:
            raise ProvenanceError(
                f"arm {arm_id}: files not registered in loadable_files_sha256 (vLLM could load "
                f"them): {unregistered[:5]}"
            )
        missing = sorted(set(registered) - set(self._loaded))
        if missing:
            raise ProvenanceError(f"arm {arm_id}: registered files missing: {missing[:5]}")
        differ = sorted(k for k in registered if registered[k] != self._loaded[k])
        if differ:
            raise ProvenanceError(
                f"arm {arm_id}: files differ from their registered SHA-256: {differ[:5]}"
            )
        return data.decode("utf-8")

    def _cleanup_failed_load(self, exc: BaseException) -> None:
        # The runner never receives this engine, so it cannot close it: shut down here whatever
        # was created (if vLLM itself failed to construct, only the process-level steps run).
        try:
            steps = self.backend.shutdown(self._llm)
        except BaseException as cleanup_exc:  # keep the original error as the one raised
            steps = [{"step": "backend.shutdown", "status": "failed", "error": _err(cleanup_exc)}]
        self._llm = None
        self._tokenizer = None
        gc.collect()
        self._close_state = "failed"
        self._close_error = _err(exc)
        self.close_report = {
            "arm": self.arm.id,
            "load_id": self.load_id,
            "phase": "load_failed",
            "load_error": _err(exc),
            "shutdown_steps": steps,
            "errors": step_failures(steps),
        }
        if self._reports is not None:
            self._reports.append(dict(self.close_report))

    def describe(self) -> dict[str, Any]:
        return _copy(self._describe)

    def _render(self, request: GenerationRequest) -> str:
        text = user_message(self.arm.inference["user_message"], request.prompt)
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            chat_template=self._template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.arm.inference["enable_thinking"],
        )

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[BatchItem]:
        if self._close_state is not None:
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
                # property instead of an assumption (it cannot tell apart two requests that
                # render the same prompt).
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
        """Shut down, release and re-check. Idempotent after a successful close; after a failed
        close every further call raises again, so a failure is never later reported as a
        success."""
        if self._close_state == "ok":
            return
        if self._close_state == "failed":
            raise CloseError(f"arm {self.arm.id}: an earlier close failed: {self._close_error}")
        self._close_state = "failed"  # until every step below has run and passed
        self._close_error = "close interrupted"
        report: dict[str, Any] = {"arm": self.arm.id, "load_id": self.load_id, "phase": "close"}
        try:
            self._close(report)
        finally:
            self.close_report = report
            if self._reports is not None:
                self._reports.append(dict(report))

    def _close(self, report: dict[str, Any]) -> None:
        errors: list[str] = []
        try:
            steps = self.backend.shutdown(self._llm)
        except Exception as exc:
            steps = [{"step": "backend.shutdown", "status": "failed", "error": _err(exc)}]
        report["shutdown_steps"] = steps
        errors += step_failures(steps)
        self._llm = None
        self._tokenizer = None
        gc.collect()
        memory: int | None = None
        try:
            memory = self.backend.parent_torch_memory_allocated()
        except Exception as exc:
            errors.append(f"parent_torch_memory_allocated: {_err(exc)}")
        report["parent_process_torch_allocated_bytes"] = memory
        limit = self.max_parent_memory_after_close
        report["parent_process_memory_limit_bytes"] = limit
        if limit is not None and memory is not None and memory > limit:
            errors.append(
                f"{memory} bytes still allocated in this process after close (limit {limit})"
            )
        try:
            children = list(self.backend.live_children())
        except Exception as exc:
            children = None
            errors.append(f"live_children: {_err(exc)}")
        report["multiprocessing_children_after_close"] = children
        if children:
            errors.append(f"child processes still alive after close: {children[:5]}")
        try:
            after = hash_tree(self.checkpoint_dir)
        except Exception as exc:
            after = None
            errors.append(f"re-hash: {_err(exc)}")
        unchanged = None if after is None else after == {**self._loaded, **self._other}
        report["files_unchanged"] = unchanged
        if unchanged is False:
            errors.insert(0, "checkpoint files changed during the run")
        report["errors"] = errors
        report["note"] = (
            "parent-process observations only: worker-process exit and device-level GPU memory "
            "must be checked on the GPU host"
        )
        if errors:
            self._close_error = "; ".join(errors)[:_MAX_ERROR]
            cls = ProvenanceError if unchanged is False else CloseError
            raise cls(f"arm {self.arm.id}: close failed; not loading the next arm: {errors}")
        self._close_state = "ok"
        self._close_error = None


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value


def arm_loader(
    arm: ArmSpec,
    checkpoint_dir: Path,
    backend_factory: Callable[[], Any] = VllmBackend.load,
    *,
    max_parent_memory_after_close: int | None = None,
    reports: list[dict[str, Any]] | None = None,
) -> Callable[[], VllmEngine]:
    """An ``ArmLoader``: each call is a separate load with a fresh engine and ``load_id``.

    ``reports`` (optional) collects each engine's close report, and the report of a load that
    failed after vLLM was created; the loader keeps no reference to the engine itself.
    """

    def load() -> VllmEngine:
        return VllmEngine(
            arm,
            checkpoint_dir,
            backend_factory(),
            max_parent_memory_after_close=max_parent_memory_after_close,
            reports=reports,
        )

    return load

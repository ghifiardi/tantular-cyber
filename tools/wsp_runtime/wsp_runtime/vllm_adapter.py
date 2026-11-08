"""C1: a vLLM inference engine for ``per_arm_sequential`` WSP runs.

:class:`VllmEngine` implements the harness's ``BatchGenerator`` interface for one arm, and
:func:`arm_loader` returns the ``ArmLoader`` the runner calls once per arm. :func:`preflight`
runs every check that needs no GPU and no loaded engine (settings, installed versions,
``SamplingParams`` read-back, storage, loaded-file inventory) so the CLI can refuse a bad arm
before any arm runs or any output is written. Every engine:

* **refuses incomplete or mistyped settings**: every inference key the engine uses must be
  present with the right type, every engine key must be known, pinned and typed, and the
  installed vLLM and transformers versions must equal the protocol's;
* **checks storage and provenance before loading**: the checkpoint must sit on a read-only
  mount below root-owned, non-writable parents (:func:`verify_storage`); every file is hashed and
  every loadable file must be registered in ``checkpoint.loadable_files_sha256``; the weights
  file must match ``weights_sha256_lfs`` and ``chat_template.jinja`` the protocol's git blob id;
* **renders with the registered template**: the bytes of ``chat_template.jinja`` that were
  checked are passed explicitly to ``apply_chat_template``;
* **reports requested and observed settings separately**, reads back every per-request
  ``SamplingParams`` object, and gets a fresh ``load_id`` per load;
* **echoes requests**: every vLLM output's prompt must be the prompt rendered for its request;
* **closes in order**: the engine's own shutdown, then every reference to the engine is dropped,
  then the process-level release (parallel state, garbage collection, CUDA cache), each step
  recorded; this process's allocated and reserved PyTorch memory is recorded; a close report is
  always kept, and a failed close keeps failing. An engine whose initialisation fails after
  vLLM was created is shut down the same way before the error propagates.

The before/after hashes show snapshot consistency only (the files were the same when hashed at
load and at close), which is why the storage itself must be read-only. Local checks cannot
establish everything about the storage; the operator's verification stays explicit (see
:data:`STORAGE_OPERATOR_CHECKS`). The telemetry variables and the shutdown hook names are
assumptions about vLLM that depend on its version and must be checked against the pinned
release. All of this is exercised with stand-in vLLM and torch objects; vLLM, transformers and
torch are imported only by :meth:`VllmBackend.load`, never at module import.
"""

from __future__ import annotations

import gc
import importlib
import multiprocessing
import os
import re
import stat
import traceback
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
_DTYPES = ("bfloat16", "float16", "float32")  # "auto" would let vLLM choose
# Descriptive engine keys: recorded with the protocol, not passed to vLLM; pinned strings.
_ENGINE_DESCRIPTIVE = frozenset({"batching", "sharding", "hardware"})
# "auto" reads generation_config.json from the (hashed) checkpoint directory; "vllm" uses vLLM's
# own defaults. A path is refused: vLLM would read a file outside the hashed checkpoint.
_GENERATION_CONFIG = ("auto", "vllm")
# vllm.LLM arguments this adapter does not set; vLLM chooses them (by version and hardware) and
# their effective values must be read back on the GPU host before a confirmatory run.
ENGINE_ARGS_LEFT_AT_DEFAULTS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "enforce_eager",
    "compilation_config",
    "kv_cache_dtype",
    "attention backend (VLLM_ATTENTION_BACKEND)",
    "scheduler and chunked-prefill settings",
)
# Repository files that inference does not read, at the checkpoint root only. They are hashed
# and reported apart from the loadable files unless the protocol registers them as loadable.
_DOC_FILES = re.compile(r"(?i)^(?:(?:readme|license|notice)(?:\.(?:md|txt|rst))?|\.gitattributes)$")
_TEMPLATE_FILE = "chat_template.jinja"
_BLOB = re.compile(r"git blob ([0-9a-f]{40})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# Offline and no-telemetry variables. The Hugging Face names are documented by those libraries;
# VLLM_NO_USAGE_STATS / DO_NOT_TRACK are vLLM's usage-stats opt-outs as documented, an
# assumption to check against the pinned vLLM version.
_FORCED_ENV = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "VLLM_NO_USAGE_STATS",
    "DO_NOT_TRACK",
)
_RECORDED_ENV = ("VLLM_", "HF_", "TRANSFORMERS_", "TORCH", "PYTORCH_", "CUDA_", "NCCL_")
_SECRET_ENV = re.compile(r"(?i)token|key|secret|password|credential")
_MAX_ERROR = 300

STORAGE_OPERATOR_CHECKS = (
    "the read-only mount is enforced by the host (mount options or an immutable image), not "
    "only reported by statvfs inside a container",
    "no writable alias of the same storage exists: no other mount, bind mount or overlay "
    "exposes it writably, to this or any other user or process",
    "no privileged process can mount over, remount read-write or replace the checkpoint or any "
    "parent between the check and vLLM opening the files (root is trusted by the local checks)",
    "no ACL, group membership or privileged process outside this one can write the checkpoint "
    "or its parent directories",
    "the files on the mount are the ones fetched and registered (re-hashed after mounting)",
)


class ProvenanceError(RuntimeError):
    """The files about to be loaded are not the ones the protocol registered, or the storage
    they are on is not read-only."""


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
    """Set the offline and no-telemetry variables to ``1``. A variable already set to anything
    else is refused rather than overridden, so a conflicting operator environment is visible.
    Whether vLLM honours these names is version-dependent and must be checked on the pinned
    release (for example with egress blocked and logged)."""
    environ = os.environ if environ is None else environ
    for var in _FORCED_ENV:
        current = environ.get(var)
        if current is not None and current != "1":
            raise SettingsError(f"{var}={current!r} is set; the runtime runs offline only")
        environ[var] = "1"
    return {var: environ[var] for var in _FORCED_ENV}


def process_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Variables that can change what vLLM, torch or the Hugging Face libraries do (values of
    names that look like credentials are redacted)."""
    environ = os.environ if environ is None else environ
    return {
        k: ("<redacted>" if _SECRET_ENV.search(k) else v)
        for k, v in sorted(environ.items())
        if k.startswith(_RECORDED_ENV) or k == "DO_NOT_TRACK"
    }


def _typed(value: Any, kind: type, name: str) -> Any:
    ok = type(value) is kind or (kind is float and type(value) is int)
    if not ok:
        raise SettingsError(f"{name} must be set to a {kind.__name__} (got {value!r})")
    if kind is str and (not value.strip() or value.strip().upper().startswith("REPLACE")):
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


def check_engine(arm: ArmSpec, backend: Any | None = None) -> dict[str, Any]:
    """The arm's engine settings, complete and typed; with a ``backend``, also matching the
    installed versions. Returns the keyword arguments for ``vllm.LLM`` (without paths)."""
    eng = arm.engine
    unknown = sorted(set(eng) - set(_ENGINE_KEYS) - _ENGINE_DESCRIPTIVE)
    if unknown:
        raise SettingsError(f"arm {arm.id}: engine keys not understood: {unknown}")
    if eng.get("name") != "vllm" or (backend is not None and backend.engine != "vllm"):
        raise SettingsError(f"arm {arm.id}: protocol engine is {eng.get('name')!r}")
    missing = sorted(set(_ENGINE_KEYS) - set(eng))
    if missing:
        raise SettingsError(f"arm {arm.id}: engine keys missing: {missing}")
    for key, kind in _ENGINE_KEYS.items():
        _typed(eng[key], kind, f"arm {arm.id}: engine.{key}")
    for key in sorted(_ENGINE_DESCRIPTIVE & set(eng)):
        _typed(eng[key], str, f"arm {arm.id}: engine.{key}")
    if eng["dtype"] not in _DTYPES:
        raise SettingsError(f"arm {arm.id}: engine.dtype must be one of {_DTYPES}")
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
    if backend is not None:
        if backend.version != eng["version"]:
            raise ProvenanceError(
                f"arm {arm.id}: engine version {backend.version!r} is not the protocol's "
                f"{eng['version']!r}"
            )
        if backend.transformers_version != eng["transformers_version"]:
            raise ProvenanceError(
                f"arm {arm.id}: transformers {backend.transformers_version!r} is not the "
                f"protocol's {eng['transformers_version']!r}"
            )
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


# --- storage --------------------------------------------------------------------------------


def verify_storage(
    path: Path,
    *,
    statvfs: Callable[[Any], Any] = os.statvfs,
    access: Callable[[Any, int], bool] = os.access,
    lstat: Callable[[Any], os.stat_result] = os.lstat,
) -> dict[str, Any]:
    """Local checks that the checkpoint cannot be changed or replaced during the run.

    Required (a failure raises :class:`ProvenanceError`):

    * the path is absolute after normalisation, has no ``..`` component, has no symbolic-link
      component, and the checkpoint contains no symbolic link;
    * **every entry** (directories and files) reports a read-only mount (``statvfs``
      ``ST_RDONLY``), so a file bind-mounted from writable storage is refused;
    * no entry is writable by this process (``access``, which also applies ACLs for this
      process's identity);
    * every parent directory, up to ``/``, is **owned by root**, carries no group/other write
      bit and is not writable by this process, so no ordinary user can rename the checkpoint
      away and put another directory in its place. There is no exemption for sticky
      directories: a checkpoint below ``/tmp`` is refused. Running as root is refused too,
      because every parent on a writable filesystem is then writable by this process.

    Recorded, not required: mode bits that would allow writing on a writable mount (the mount,
    not the mode bits, is relied on). Not established by any local check: see
    :data:`STORAGE_OPERATOR_CHECKS`, which the operator must verify and record.
    """
    problems: list[str] = []
    raw = Path(path)
    if ".." in raw.parts:
        problems.append(f"the checkpoint path contains '..' ({raw}); give the resolved path")
    given = Path(os.path.abspath(raw))
    resolved = given.resolve(strict=True)
    if resolved != given:
        problems.append(f"the checkpoint path goes through a symbolic link ({given} -> {resolved})")
    entries = [resolved, *sorted(resolved.rglob("*"))]

    def rel(p: Path) -> str:
        return p.relative_to(resolved).as_posix() or "."

    links = [rel(p) for p in entries if p.is_symlink()]
    if links:
        problems.append(f"contains symbolic links: {links[:5]}")
    entries = [p for p in entries if not p.is_symlink()]
    rw_mount = [rel(p) for p in entries if not statvfs(p).f_flag & os.ST_RDONLY]
    if rw_mount:
        problems.append(f"not on a read-only mount: {rw_mount[:5]}")
    devices = sorted({lstat(p).st_dev for p in entries})
    writable = [rel(p) for p in entries if access(p, os.W_OK)]
    if writable:
        problems.append(f"writable by this process: {writable[:5]}")
    parents = []
    for a in resolved.parents:
        st = lstat(a)
        reasons = [
            r
            for r, bad in (
                ("not owned by root", st.st_uid != 0),
                ("group/other-writable", bool(st.st_mode & (stat.S_IWGRP | stat.S_IWOTH))),
                ("writable by this process", access(a, os.W_OK)),
            )
            if bad
        ]
        if reasons:
            parents.append(f"{a} ({', '.join(reasons)})")
    if parents:
        problems.append(f"replaceable parent directories: {parents[:5]}")
    mode_bits = [rel(p) for p in entries if lstat(p).st_mode & 0o222]
    report = {
        "path": str(resolved),
        "read_only_mount": not rw_mount,
        "devices": devices,
        "writable_by_process": writable[:20],
        "replaceable_parents": parents[:20],
        "entries_with_write_mode_bits": mode_bits[:20],
        "not_established_locally": list(STORAGE_OPERATOR_CHECKS),
    }
    if problems:
        raise ProvenanceError(
            "the checkpoint must be on a read-only mount below root-owned, non-writable "
            "parents: " + "; ".join(problems)
        )
    return report


# --- loaded-file inventory ------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSnapshot:
    path: Path
    loaded: dict[str, str]
    other: dict[str, str]
    template: str
    storage: dict[str, Any]


def inspect_checkpoint(
    arm: ArmSpec, checkpoint_dir: Path, *, storage_check: Callable[[Path], dict] = verify_storage
) -> CheckpointSnapshot:
    """Storage check, then hash the directory and compare it with the protocol's registration.
    Needs no backend; run by :func:`preflight` before any arm runs and again at every load."""
    storage = storage_check(Path(checkpoint_dir))
    root = Path(storage["path"])
    tree = hash_tree(root)
    ckpt, arm_id = arm.checkpoint, arm.id
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
            f"arm {arm_id}: loadable_files_sha256 must list weights_file with weights_sha256_lfs"
        )
    if _TEMPLATE_FILE not in registered:
        raise ProvenanceError(f"arm {arm_id}: loadable_files_sha256 must list {_TEMPLATE_FILE}")
    # Root-level documentation is not loadable unless the protocol registers it as such.
    loaded = {
        k: v
        for k, v in tree.items()
        if k in registered or not ("/" not in k and _DOC_FILES.match(k))
    }
    other = {k: v for k, v in tree.items() if k not in loaded}
    if weights not in loaded:
        raise ProvenanceError(f"arm {arm_id}: registered weights file {weights!r} missing")
    if loaded[weights] != digest:
        raise ProvenanceError(f"arm {arm_id}: {weights} SHA-256 differs from the registered value")
    blob = _BLOB.search(arm.inference.get("chat_template") or "")
    path = root / _TEMPLATE_FILE
    data = path.read_bytes() if path.is_file() else None
    if data is None or blob is None or git_blob_sha1(data) != blob.group(1):
        raise ProvenanceError(f"arm {arm_id}: {_TEMPLATE_FILE} is not the registered git blob")
    if sha256_bytes(data) != loaded.get(_TEMPLATE_FILE):
        raise ProvenanceError(f"arm {arm_id}: {_TEMPLATE_FILE} changed while it was hashed")
    unregistered = sorted(set(loaded) - set(registered))
    if unregistered:
        raise ProvenanceError(
            f"arm {arm_id}: files not registered in loadable_files_sha256 (vLLM could load "
            f"them): {unregistered[:5]}"
        )
    missing = sorted(set(registered) - set(loaded))
    if missing:
        raise ProvenanceError(f"arm {arm_id}: registered files missing: {missing[:5]}")
    differ = sorted(k for k in registered if registered[k] != loaded[k])
    if differ:
        raise ProvenanceError(
            f"arm {arm_id}: files differ from their registered SHA-256: {differ[:5]}"
        )
    return CheckpointSnapshot(root, loaded, other, data.decode("utf-8"), storage)


SAMPLING_FIELDS = (
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


def describe_sampling(params: Any) -> dict[str, Any]:
    """The values a ``SamplingParams`` object actually holds (read back, not requested)."""
    return {f: _plain(getattr(params, f, None)) for f in SAMPLING_FIELDS}


def check_sampling(
    make: Callable[..., Any],
    wanted: Mapping[str, Any],
    arm_id: str,
    describe: Callable[[Any], dict[str, Any]] = describe_sampling,
) -> tuple[Any, dict[str, Any]]:
    """Build a ``SamplingParams`` object and read it back; a requested value it does not hold
    is refused."""
    params = make(**wanted)
    observed = describe(params)
    changed = sorted(k for k, v in wanted.items() if observed.get(k) != v)
    if changed:
        raise SettingsError(
            f"arm {arm_id}: SamplingParams did not keep the requested values for {changed}"
        )
    return params, observed


def check_installed(arm: ArmSpec, installed: Mapping[str, str | None]) -> None:
    """The installed package versions (from package metadata, before anything is imported)
    must equal the protocol's."""
    for key, package in (("version", "vllm"), ("transformers_version", "transformers")):
        if installed.get(package) != arm.engine.get(key):
            raise ProvenanceError(
                f"arm {arm.id}: installed {package} is {installed.get(package)!r}, the protocol "
                f"pins {arm.engine.get(key)!r}"
            )


def preflight(
    arm: ArmSpec,
    checkpoint_dir: Path,
    *,
    storage_check: Callable[[Path], dict] = verify_storage,
    installed: Mapping[str, str | None] | None = None,
    sampling_params: Callable[..., Any] | None = None,
    sample_shapes: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Every check that needs no GPU and no loaded engine: settings, storage, the loaded-file
    inventory and, when given, the installed versions and the read-back of a ``SamplingParams``
    object for every (n, seed) the run will request. Observed engine settings need a loaded
    engine and stay a check on the GPU host."""
    sampling = check_inference(arm)
    check_engine(arm)
    if installed is not None:
        check_installed(arm, installed)
    for n, seed in sample_shapes if sampling_params is not None else ():
        check_sampling(sampling_params, {"n": n, "seed": seed, **sampling}, arm.id)
    snap = inspect_checkpoint(arm, checkpoint_dir, storage_check=storage_check)
    return {
        "arm": arm.id,
        "path": str(snap.path),
        "loadable_files": len(snap.loaded),
        "installed_checked": installed is not None,
        "sampling_shapes_checked": [list(x) for x in sample_shapes] if sampling_params else [],
        "storage_devices": snap.storage.get("devices"),
    }


# --- backend --------------------------------------------------------------------------------


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


def _engine_shutdown(llm: Any) -> tuple[str, Callable[[], Any] | None]:
    # Version-dependent assumption: older vLLM engines expose llm_engine.shutdown(); the V1
    # engine exposes llm_engine.engine_core.shutdown(). Check against the pinned release.
    engine = getattr(llm, "llm_engine", None)
    for name, owner in (
        ("llm_engine.shutdown", engine),
        ("llm_engine.engine_core.shutdown", getattr(engine, "engine_core", None)),
    ):
        fn = getattr(owner, "shutdown", None)
        if callable(fn):
            return name, fn
    return "llm_engine.shutdown", None


@dataclass
class VllmBackend:
    """The real backend: vLLM, transformers and torch, imported only in :meth:`load`. Tests run
    these methods with stand-in modules; every call must be validated on the GPU host."""

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

    def describe_sampling(self, params: Any) -> dict[str, Any]:
        return describe_sampling(params)

    def default_sampling(self, llm: Any) -> dict[str, Any] | None:
        get = getattr(llm, "get_default_sampling_params", None)
        return self.describe_sampling(get()) if callable(get) else None

    def engine_config(self, llm: Any) -> dict[str, Any] | None:
        """Best-effort read-back of the engine's model configuration (attribute names are
        version-dependent; ``None`` when unavailable). Not a substitute for recording the
        effective engine configuration on the GPU host."""
        config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
        if config is None:
            return None
        return {f: _plain(getattr(config, f, None)) for f in ("dtype", "max_model_len", "seed")}

    def shutdown_engine(self, llm: Any) -> list[dict[str, Any]]:
        """The engine's own shutdown (required); never raises an ``Exception``. The hook names
        are version-dependent assumptions."""
        name, fn = _engine_shutdown(llm)
        return [_step(name, fn, required=True)]

    def release(self) -> list[dict[str, Any]]:
        """Process-level teardown, run after the caller has dropped every reference to the
        engine: parallel state, garbage collection, then the CUDA cache, so memory freed by
        the collection is returned too. Never raises an ``Exception``; whether this frees the
        GPU must be validated on the GPU host."""
        steps: list[dict[str, Any]] = []
        for name in ("destroy_model_parallel", "destroy_distributed_environment"):
            fn = getattr(self.parallel_state, name, None)
            steps.append(_step(f"parallel_state.{name}", fn if callable(fn) else None))
        steps.append(_step("gc.collect", gc.collect))
        cuda = getattr(self.torch, "cuda", None)

        def empty_cache() -> None:
            if cuda.is_available():
                cuda.empty_cache()

        steps.append(_step("torch.cuda.empty_cache", empty_cache if cuda is not None else None))
        return steps

    def parent_torch_memory(self) -> dict[str, int] | None:
        """This process's PyTorch allocator only: memory allocated to tensors and memory
        reserved by the caching allocator. vLLM's worker processes, CUDA contexts and NCCL
        buffers are not counted, so this never shows that the GPU was released."""
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return None
        return {"allocated": int(cuda.memory_allocated()), "reserved": int(cuda.memory_reserved())}

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


# --- engine ---------------------------------------------------------------------------------


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
        storage_check: Callable[[Path], dict] = verify_storage,
    ):
        self.arm = arm
        self._reports = reports
        self.backend = backend
        self.max_parent_memory_after_close = max_parent_memory_after_close
        self.close_report: dict[str, Any] | None = None
        self._close_state: str | None = None  # None, "ok" or "failed"
        self._close_error: str | None = None
        self._llm: Any = None
        self._tokenizer: Any = None

        sampling = check_inference(arm)
        kwargs = check_engine(arm, backend)
        # Storage and provenance before anything is loaded: the files the engine will read.
        snap = inspect_checkpoint(arm, checkpoint_dir, storage_check=storage_check)
        self.checkpoint_dir = snap.path
        self._loaded, self._other, self._template = snap.loaded, snap.other, snap.template
        self._sampling = sampling

        self.load_id = f"{arm.id}-{uuid.uuid4().hex}"
        try:
            self._llm = backend.make_llm(
                model=str(self.checkpoint_dir), tokenizer=str(self.checkpoint_dir), **kwargs
            )
            self._tokenizer = self._llm.get_tokenizer()
            observed_sampling = self._sampling_params(1, 0)[1]
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
                    "engine_args_left_at_vllm_defaults": list(ENGINE_ARGS_LEFT_AT_DEFAULTS),
                    "sampling": dict(sampling),
                    "chat_template_kwargs": {
                        "add_generation_prompt": True,
                        "enable_thinking": arm.inference["enable_thinking"],
                        "chat_template": f"{_TEMPLATE_FILE} (registered git blob, passed "
                        "explicitly)",
                    },
                },
                # What the backend reported back. The effective engine configuration and the
                # stop set vLLM applies at generation time are not fully observable here:
                # record them on the GPU host.
                "observed": {
                    "sampling_params": observed_sampling,
                    "engine_default_sampling": backend.default_sampling(self._llm),
                    "engine_config": backend.engine_config(self._llm),
                    "eos_token_id": _plain(getattr(self._tokenizer, "eos_token_id", None)),
                    "tokenizer_builtin_template_matches": (
                        None if builtin is None else builtin == self._template
                    ),
                    "environment": backend.environment(),
                    "process_environment": process_environment(),
                    "storage": snap.storage,
                },
            }
        except BaseException as exc:
            self._cleanup_failed_load(exc)
            raise

    def _sampling_params(self, n: int, seed: int) -> tuple[Any, dict[str, Any]]:
        """Build a ``SamplingParams`` object and read it back; a requested value the object does
        not hold is refused."""
        return check_sampling(
            self.backend.make_sampling_params,
            {"n": n, "seed": seed, **self._sampling},
            self.arm.id,
            self.backend.describe_sampling,
        )

    def _teardown(self) -> tuple[list[dict[str, Any]], BaseException | None]:
        """Shut the engine down, drop every reference to it, then release process-level
        resources (so garbage collection and the cache clearing come after the engine is no
        longer referenced). Returns the steps and an interrupt to re-raise, if any."""
        steps: list[dict[str, Any]] = []
        interrupted: BaseException | None = None
        llm, self._llm, self._tokenizer = self._llm, None, None
        if llm is not None:
            try:
                steps += self.backend.shutdown_engine(llm)
            except BaseException as exc:
                steps.append(
                    {
                        "step": "backend.shutdown_engine",
                        "status": "failed",
                        "error": _err(exc),
                        "required": True,
                    }
                )
                if not isinstance(exc, Exception):
                    interrupted = exc
                    # The traceback's frames would otherwise keep the engine alive.
                    traceback.clear_frames(exc.__traceback__)
        del llm
        try:
            steps += self.backend.release()
        except BaseException as exc:
            steps.append(
                {
                    "step": "backend.release",
                    "status": "failed",
                    "error": _err(exc),
                    "required": True,
                }
            )
            if interrupted is None and not isinstance(exc, Exception):
                interrupted = exc
        return steps, interrupted

    def _cleanup_failed_load(self, exc: BaseException) -> None:
        # The runner never receives this engine, so it cannot close it: shut down here whatever
        # was created (if vLLM itself failed to construct, only the process-level steps run).
        # The original error is the one raised, even if the cleanup is interrupted. Its
        # traceback's frames (for example inside the engine's own methods) would otherwise keep
        # the engine alive through the release steps; their locals are cleared, the traceback
        # itself is kept.
        traceback.clear_frames(exc.__traceback__)
        steps, _ = self._teardown()
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
            params, _ = self._sampling_params(req.n, req.seed)  # read back per request
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
        interrupted: BaseException | None = None
        steps, interrupted = self._teardown()
        report["shutdown_steps"] = steps
        errors += step_failures(steps)
        report["errors"] = errors
        if interrupted is not None:
            # The engine shutdown was interrupted: the references are dropped and the release
            # steps ran, but the memory, child-process and re-hash checks do not run.
            report["interrupted"] = True
            self._close_error = "; ".join(errors)[:_MAX_ERROR]
            raise interrupted
        memory: dict[str, int] | None = None
        try:
            memory = self.backend.parent_torch_memory()
        except Exception as exc:
            errors.append(f"parent_torch_memory: {_err(exc)}")
        report["parent_process_torch_allocated_bytes"] = memory and memory["allocated"]
        report["parent_process_torch_reserved_bytes"] = memory and memory["reserved"]
        limit = self.max_parent_memory_after_close
        report["parent_process_memory_limit_bytes"] = limit
        if limit is not None and memory is not None and max(memory.values()) > limit:
            errors.append(
                f"{memory['allocated']} bytes allocated / {memory['reserved']} reserved still in "
                f"this process after close (limit {limit})"
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
        report["note"] = (
            "parent-process observations only: worker-process exit and device-level GPU memory "
            "must be checked on the GPU host; the shutdown hook names are version-dependent "
            "assumptions"
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
    storage_check: Callable[[Path], dict] = verify_storage,
) -> Callable[[], VllmEngine]:
    """An ``ArmLoader``: each call is a separate load with a fresh engine and ``load_id``. The
    backend-free checks run again before the backend is created.

    ``reports`` (optional) collects each engine's close report, and the report of a load that
    failed after vLLM was created; the loader keeps no reference to the engine itself.
    """

    def load() -> VllmEngine:
        preflight(arm, checkpoint_dir, storage_check=storage_check)
        return VllmEngine(
            arm,
            checkpoint_dir,
            backend_factory(),
            max_parent_memory_after_close=max_parent_memory_after_close,
            reports=reports,
            storage_check=storage_check,
        )

    return load

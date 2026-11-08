"""Runtime adapters C1–C3 (tools/wsp_runtime), tested with stand-in vLLM and torch modules,
synthetic grading strings and synthetic dataset files only. No model, GPU, network, vLLM,
torch or math-verify is used; nothing here is a measurement or MATH content.

The stand-ins follow the adapters' own reading of the vLLM and math-verify APIs, so these tests
cannot catch a real API mismatch (checked on the GPU host, docs/wsp/runtime.md). The test
backend is a subclass of the real ``VllmBackend``: shutdown steps, settings read-back and memory
queries run the real implementation against the stand-in modules. Storage checks run the real
``verify_storage`` with a stand-in ``statvfs`` (a temporary directory is not a read-only mount)
and with parent directories outside the test's own directory treated as protected."""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import json
import os
import stat
import sys
import threading
import weakref
from contextlib import contextmanager
from functools import partial
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
WSP = ROOT / "tools" / "wsp"
RUNTIME = ROOT / "tools" / "wsp_runtime"
sys.path.insert(0, str(WSP))
sys.path.insert(0, str(RUNTIME))

import run as runtime_cli  # noqa: E402
from wsp_eval.dataset import Item, load_items  # noqa: E402
from wsp_eval.drift import drift_report  # noqa: E402
from wsp_eval.generation import GenerationRequest  # noqa: E402
from wsp_eval.hashing import sha256_file  # noqa: E402
from wsp_eval.protocol import parse_protocol  # noqa: E402
from wsp_eval.runner import verify_run  # noqa: E402
from wsp_runtime import datasets, vllm_adapter  # noqa: E402
from wsp_runtime.grader import (  # noqa: E402
    GRADER_ID,
    GRADER_REVISION,
    REVIEWED,
    GraderInputError,
    MathQAGrader,
    MathVerifyBackend,
    check_dependencies,
    last_boxed,
    strip_reasoning,
)
from wsp_runtime.provenance import git_blob_sha1, hash_tree  # noqa: E402
from wsp_runtime.vllm_adapter import (  # noqa: E402
    PROTOCOL_USER_MESSAGE,
    BatchMismatch,
    CloseError,
    ProvenanceError,
    SettingsError,
    VllmBackend,
    VllmEngine,
    force_offline,
    process_environment,
    step_failures,
    user_message,
    verify_storage,
)

WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
TEMPLATE = b"{# synthetic chat template for tests #}"
VERSION = "0.0.0-stub"
TF_VERSION = "0.0.0-stub-transformers"


# --- stand-in vLLM and torch modules -----------------------------------------------------


class Completion:
    def __init__(self, text, n_tokens, finish_reason):
        self.text, self.token_ids, self.finish_reason = text, list(range(n_tokens)), finish_reason


class RequestOutput:
    def __init__(self, prompt, outputs):
        self.prompt, self.outputs = prompt, outputs


class Tokenizer:
    eos_token_id = 248044
    chat_template = TEMPLATE.decode()

    def apply_chat_template(
        self, messages, chat_template, tokenize, add_generation_prompt, enable_thinking
    ):
        assert chat_template == TEMPLATE.decode()  # the registered bytes, passed explicitly
        assert tokenize is False and add_generation_prompt is True
        assert type(enable_thinking) is bool
        (m,) = messages
        return f"<user>{m['content']}</user><assistant>{'<think>' if enable_thinking else ''}"


class Params:
    """Stand-in ``SamplingParams``: holds what it is given."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _EngineCore:
    def __init__(self, hub):
        self.hub = hub

    def shutdown(self):
        self.hub.engine_shutdown("llm_engine.engine_core.shutdown")


class _LLMEngine:
    def __init__(self, hub, kwargs):
        self.hub = hub
        self.model_config = Params(
            dtype=kwargs["dtype"], max_model_len=kwargs["max_model_len"], seed=kwargs["seed"]
        )
        if hub.engine_mode == "v1":
            self.engine_core = _EngineCore(hub)

    def __getattr__(self, name):
        if name == "shutdown" and self.hub.engine_mode == "v0":
            return lambda: self.hub.engine_shutdown("llm_engine.shutdown")
        raise AttributeError(name)


class FakeLLM:
    def __init__(self, hub, **kwargs):
        self.hub, self.kwargs = hub, kwargs
        self.llm_engine = _LLMEngine(hub, kwargs)

    def get_tokenizer(self):
        return self.hub.tokenizer()

    def get_default_sampling_params(self):
        return Params(temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0)

    def generate(self, prompts, params, use_tqdm):
        assert use_tqdm is False and len(prompts) == len(params)
        self.hub.calls.append([(p, vars(s)) for p, s in zip(prompts, params, strict=True)])
        outs = [self.hub.answer(p, s) for p, s in zip(prompts, params, strict=True)]
        return self.hub.post(outs)


class _Hub:
    """Stand-in ``vllm`` module and the state the tests inspect."""

    def __init__(
        self, *, answer, post, memory, engine_mode, engine_error, tokenizer, params, llm_error
    ):
        self.calls: list = []
        self.llms: list = []
        self.events: list = []
        self.alive_at_empty_cache: list = []
        self.reserved_extra = 0
        self.answer, self.post, self.memory = answer, post, memory
        self.engine_mode, self.engine_error, self.llm_error = engine_mode, engine_error, llm_error
        self.tokenizer, self.SamplingParams = tokenizer, params

    def LLM(self, **kwargs):  # mirrors vllm.LLM
        if self.llm_error is not None:
            raise self.llm_error
        llm = FakeLLM(self, **kwargs)
        self.llms.append(weakref.ref(llm))
        return llm

    def engine_shutdown(self, name):
        self.events.append(name)
        if self.engine_error is not None:
            # A fresh exception each time: a stored one would keep its traceback (and so the
            # engine) alive, which is a fixture artefact, not adapter behaviour.
            raise type(self.engine_error)(*self.engine_error.args)


class _Cuda:
    def __init__(self, hub):
        self.hub = hub

    def is_available(self):
        return True

    def memory_allocated(self):
        return self.hub.memory

    def memory_reserved(self):
        return self.hub.memory + self.hub.reserved_extra

    def empty_cache(self):
        self.hub.events.append("torch.cuda.empty_cache")
        # Was any engine object still alive when the cache was cleared?
        self.hub.alive_at_empty_cache.append(any(r() is not None for r in self.hub.llms))


class _ParallelState:
    def __init__(self, hub):
        self.hub = hub

    def destroy_model_parallel(self):
        self.hub.events.append("destroy_model_parallel")

    def destroy_distributed_environment(self):
        self.hub.events.append("destroy_distributed_environment")


def _correct(prompt, params):
    # Synthetic arithmetic only: "compute A + B" -> boxed sum, one output per sample.
    expr = prompt.split("compute ", 1)[1].split(".", 1)[0]
    a, b = (int(x) for x in expr.split(" + "))
    text = f"<think>working</think>The answer is \\boxed{{{a + b}}}."
    return RequestOutput(prompt, [Completion(text, 12, "stop") for _ in range(params.n)])


class FakeBackend(VllmBackend):
    """The real ``VllmBackend`` driving stand-in modules."""

    def __init__(
        self,
        *,
        answer=None,
        post=None,
        memory=0,
        engine_mode="v0",
        engine_error=None,
        tokenizer=Tokenizer,
        params=Params,
        llm_error=None,
        reserved_extra=0,
    ):
        hub = _Hub(
            answer=answer or _correct,
            post=post or (lambda outs: outs),
            memory=memory,
            engine_mode=engine_mode,
            engine_error=engine_error,
            tokenizer=tokenizer,
            params=params,
            llm_error=llm_error,
        )
        torch = SimpleNamespace(cuda=_Cuda(hub), __version__="0-stub", version=None)
        super().__init__(
            vllm=hub,
            torch=torch,
            version=VERSION,
            transformers_version=TF_VERSION,
            parallel_state=_ParallelState(hub),
        )
        self.hub = hub
        hub.reserved_extra = reserved_extra

    @property
    def calls(self):
        return self.hub.calls

    @property
    def llms(self):
        return self.hub.llms

    @property
    def events(self):
        return self.hub.events


class ExactGrader:
    """Stub math-verify: exact string equality after whitespace removal."""

    def equivalent(self, gold, pred):
        return gold.replace(" ", "") == pred.replace(" ", "")


# --- fixtures -------------------------------------------------------------------------------


def _lock(d: Path) -> None:
    for p in sorted(d.rglob("*"), reverse=True):
        if not p.is_symlink():
            p.chmod(0o555 if p.is_dir() else 0o444)
    d.chmod(0o555)


@contextmanager
def _unlocked(d: Path):
    d.chmod(0o755)
    for p in d.rglob("*"):
        if not p.is_symlink():
            p.chmod(0o755 if p.is_dir() else 0o644)
    try:
        yield d
    finally:
        _lock(d)


def _read_only_mount(path):
    return SimpleNamespace(f_flag=os.ST_RDONLY)


def _storage_for(root: Path, statvfs=_read_only_mount):
    """The real ``verify_storage``. ``root`` and its parents are outside what a test can make
    root-owned, so they are presented as root-owned, mode 0755 and not writable; everything
    strictly below ``root`` (the checkpoints) is checked for real."""
    root = root.resolve()

    def inside(p) -> bool:
        return root in Path(p).parents

    def access(p, mode):
        return os.access(p, mode) if inside(p) else False

    def lstat(p):
        return os.lstat(p) if inside(p) else os.stat_result((stat.S_IFDIR | 0o755, *[0] * 9))

    return partial(verify_storage, statvfs=statvfs, access=access, lstat=lstat)


STORAGE: dict = {}


@pytest.fixture(autouse=True)
def _storage(tmp_path):
    STORAGE["check"] = _storage_for(tmp_path / "ckpt")
    yield
    STORAGE.clear()


def _checkpoint(root: Path, name: str, weights: bytes) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / WEIGHTS).write_bytes(weights)
    (d / "config.json").write_text('{"model_type": "synthetic"}')
    (d / "tokenizer.json").write_text('{"synthetic": true}')
    (d / "chat_template.jinja").write_bytes(TEMPLATE)
    (d / "README.md").write_text(f"synthetic {name}")
    _lock(d)
    return d


def _loadable(d: Path) -> dict:
    return {k: v for k, v in hash_tree(d).items() if k != "README.md"}


def _protocol_raw(base: Path, edited: Path) -> dict:
    inference = {
        "chat_template": f"repo chat_template.jinja (git blob {git_blob_sha1(TEMPLATE)})",
        "enable_thinking": True,
        "user_message": PROTOCOL_USER_MESSAGE,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 64,
    }
    engine = {
        "name": "vllm",
        "version": VERSION,
        "transformers_version": TF_VERSION,
        "dtype": "bfloat16",
        "max_model_len": 128,
        "llm_seed": 0,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "generation_config": "auto",
        "hardware": "stub",
    }

    def arm(arm_id, role, d):
        files = _loadable(d)
        return {
            "id": arm_id,
            "role": role,
            "checkpoint": {
                "repo": "synthetic/stub",
                "revision": "stub",
                "weights_file": WEIGHTS,
                "weights_sha256_lfs": files[WEIGHTS],
                "loadable_files_sha256": files,
            },
            "inference": dict(inference),
            "engine": dict(engine),
        }

    raw = json.loads((WSP / "fixtures" / "synthetic_protocol.json").read_text())
    raw.update(
        protocol_version="runtime-stub-0.3",
        synthetic=False,
        sampling_mode="n_per_request",
        samples_per_item=4,
        seeds=[0],
        grader={"id": GRADER_ID, "revision": GRADER_REVISION},
        execution={"order": "per_arm_sequential", "shards": 2},
        analysis={**raw["analysis"], "reference_arm": "base", "candidate_arm": "edited"},
        arms=[
            arm("base", "reference", base),
            arm("edited", "candidate", edited),
            arm("base_reload", "no_edit_reload", base),
        ],
    )
    return raw


@pytest.fixture
def ckpts(tmp_path):
    root = tmp_path / "ckpt"
    base = _checkpoint(root, "base", b"synthetic base weights")
    edited = _checkpoint(root, "edited", b"synthetic edited weights")
    root.chmod(0o555)  # presented as root-owned to the storage check (see _storage_for)
    yield base, edited
    root.chmod(0o755)


def _data(tmp_path, *, train_rows=None, held_rows=None, dev_selection=None):
    """Write the synthetic source files and return their protocol registration."""
    train = _jsonl(
        tmp_path / "data" / "train.jsonl", _train_rows() if train_rows is None else train_rows
    )
    held = _jsonl(
        tmp_path / "data" / "math500.jsonl", _math500_rows() if held_rows is None else held_rows
    )
    sources = {"hash_definitions": datasets.HASH_DEFINITIONS}
    for split, path, shape, sel in (
        ("development", train, "math_train", dev_selection or {"rule": "all"}),
        ("heldout", held, "math500", {"rule": "all"}),
    ):
        items, report = datasets.load(path, shape=shape, split=split)
        _, sel_report = datasets.select(items, report, sel.get("n"))
        sources[split] = {
            "shape": shape,
            "source_file_sha256": report.source_sha256,
            "source_rows": report.n_source_rows,
            "source_ids_sha256": report.source_ids_sha256,
            "selection": dict(sel),
            "items_sha256": sel_report.items_sha256,
        }
    return train, held, sources


@pytest.fixture
def protocol(ckpts, tmp_path):
    raw = _protocol_raw(*ckpts)
    raw["dataset"]["runtime_sources"] = _data(tmp_path)[2]
    return parse_protocol(raw)


def _modified(ckpts, edit, arms=None):
    raw = _protocol_raw(*ckpts)
    for a in raw["arms"]:
        if arms is None or a["id"] in arms:
            edit(a)
    return raw


def _items(n=5):
    return [
        Item(
            id=f"dev-{k:03d}",
            split="development",
            prompt=f"Synthetic: compute {k} + {k + 1}.",
            answer=str(2 * k + 1),
        )
        for k in range(n)
    ]


def _engine(protocol, ckpts, backend=None, arm="base", **kw):
    if isinstance(protocol, dict):
        protocol = parse_protocol(protocol)
    a = {x.id: x for x in protocol.arms}[arm]
    d = ckpts[1] if arm == "edited" else ckpts[0]
    kw.setdefault("storage_check", STORAGE["check"])
    return VllmEngine(a, d, backend or FakeBackend(), **kw)


def _req(protocol, arm="base", item="dev-000", prompt="Synthetic: compute 0 + 1.", n=4, seed=0):
    a = {x.id: x for x in protocol.arms}[arm]
    return GenerationRequest(a.id, item, prompt, dict(a.inference), seed, n)


# --- C1: settings ---------------------------------------------------------------------------


def test_load_reports_identity_files_and_requested_vs_observed(
    protocol, ckpts, monkeypatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "tc-synthetic-token")
    monkeypatch.setenv("VLLM_SYNTHETIC_FLAG", "1")
    backend = FakeBackend()
    e1, e2 = _engine(protocol, ckpts, backend), _engine(protocol, ckpts, backend)
    d = e1.describe()
    assert d["load_id"] != e2.describe()["load_id"]  # every load is distinct
    assert d["synthetic"] is False and d["engine_version"] == VERSION
    assert set(d["artifacts_sha256"]) == {
        WEIGHTS,
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
    }
    assert set(d["other_files_sha256"]) == {"README.md"}  # hashed, reported apart
    assert d["artifacts_sha256"][WEIGHTS] == hashlib.sha256(b"synthetic base weights").hexdigest()
    req = d["requested"]
    assert req["sampling"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 64,
    }
    assert req["engine_kwargs"] == {
        "dtype": "bfloat16",
        "max_model_len": 128,
        "seed": 0,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "generation_config": "auto",
        "trust_remote_code": False,
        "load_format": "safetensors",
    }
    assert "max_num_seqs" in req["engine_args_left_at_vllm_defaults"]
    assert req["chat_template_kwargs"]["enable_thinking"] is True
    obs = d["observed"]
    assert obs["sampling_params"]["top_k"] == 20 and obs["sampling_params"]["n"] == 1
    assert obs["sampling_params"]["stop_token_ids"] is None  # read back, not assumed
    assert obs["engine_default_sampling"]["temperature"] == 1.0
    assert obs["engine_config"] == {"dtype": "bfloat16", "max_model_len": 128, "seed": 0}
    assert obs["eos_token_id"] == 248044
    assert obs["tokenizer_builtin_template_matches"] is True
    assert obs["environment"]["transformers"] == TF_VERSION
    assert obs["process_environment"]["HF_TOKEN"] == "<redacted>"
    assert obs["process_environment"]["VLLM_SYNTHETIC_FLAG"] == "1"
    assert obs["storage"]["read_only_mount"] is True
    assert obs["storage"]["not_established_locally"]  # operator verification stays explicit
    assert len(d["runtime_code_sha256"]) == 64
    llm_kwargs = backend.llms[0]().kwargs
    assert llm_kwargs["model"] == str(ckpts[0]) and llm_kwargs["tokenizer"] == str(ckpts[0])
    assert llm_kwargs["trust_remote_code"] is False


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda a: a["inference"].pop("presence_penalty"), "missing"),
        (lambda a: a["inference"].pop("enable_thinking"), "missing"),
        (lambda a: a["inference"].update(enable_thinking="false"), "true or false"),
        (lambda a: a["inference"].update(top_k=1.5), "top_k"),
        (lambda a: a["inference"].update(temperature=True), "temperature"),
        (lambda a: a["inference"].update(top_p=0.0), "out of range"),
        (lambda a: a["inference"].update(repetition_penalty=1.1), "not understood"),
        (lambda a: a["inference"].update(chat_template="repo chat_template.jinja"), "git blob"),
        (lambda a: a["inference"].update(user_message="<problem>\nPlease"), "prompt notation"),
        (lambda a: a["inference"].update(max_tokens=128), "below max_model_len"),
    ],
)
def test_inference_settings_must_be_complete_and_typed(ckpts, edit, message) -> None:
    backend = FakeBackend()
    with pytest.raises(SettingsError, match=message):
        _engine(_modified(ckpts, edit), ckpts, backend)
    assert backend.llms == []  # refused before loading


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda a: a["engine"].update(version="9.9.9"), "engine version"),
        (lambda a: a["engine"].update(version="REPLACE (open)"), "must be pinned"),
        (lambda a: a["engine"].update(version=" replace later"), "must be pinned"),
        (lambda a: a["engine"].update(transformers_version="9.9"), "transformers"),
        (lambda a: a["engine"].update(name="other"), "protocol engine"),
        (lambda a: a["engine"].update(gpu_memory_utilization="REPLACE"), "gpu_memory"),
        (lambda a: a["engine"].update(gpu_memory_utilization=1.5), "gpu_memory"),
        (lambda a: a["engine"].update(tensor_parallel_size=1.0), "tensor_parallel_size"),
        (lambda a: a["engine"].update(max_model_len=True), "max_model_len"),
        (lambda a: a["engine"].update(dtype="auto"), "dtype"),
        (lambda a: a["engine"].update(generation_config="/elsewhere"), "outside the hashed"),
        (lambda a: a["engine"].pop("generation_config"), "missing"),
        (lambda a: a["engine"].update(quantization="fp8"), "not understood"),
        (lambda a: a["engine"].update(trust_remote_code=True), "not understood"),
        (lambda a: a["engine"].update(hardware=1), "hardware"),
        (lambda a: a["engine"].update(hardware="REPLACE"), "hardware must be pinned"),
    ],
)
def test_engine_settings_must_be_known_pinned_and_match(ckpts, edit, message) -> None:
    backend = FakeBackend()
    with pytest.raises((SettingsError, ProvenanceError), match=message):
        _engine(_modified(ckpts, edit), ckpts, backend)
    assert backend.llms == []


def test_prompt_notation_reading() -> None:
    with pytest.raises(SettingsError, match="registered Experiment 8 prompt"):
        user_message("<problem>\nPlease reason", "p")
    assert user_message(PROTOCOL_USER_MESSAGE, "Q?") == (
        "Q?\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


def test_offline_and_telemetry_variables_are_forced_and_conflicts_refused() -> None:
    env: dict = {}
    assert force_offline(env) == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    }
    with pytest.raises(SettingsError, match="HF_HUB_OFFLINE='0'"):
        force_offline({"HF_HUB_OFFLINE": "0"})
    with pytest.raises(SettingsError, match="DO_NOT_TRACK='0'"):
        force_offline({"DO_NOT_TRACK": "0"})
    assert process_environment({"HF_API_KEY": "x", "PATH": "/bin"}) == {"HF_API_KEY": "<redacted>"}


def test_sampling_params_that_drop_a_value_are_refused_and_cleaned_up(protocol, ckpts) -> None:
    class DroppingTopK(Params):
        def __init__(self, **kw):
            super().__init__(**{**kw, "top_k": -1})

    backend = FakeBackend(params=DroppingTopK)
    with pytest.raises(SettingsError, match="top_k"):
        _engine(protocol, ckpts, backend)
    assert backend.events[0] == "llm_engine.shutdown"  # the created engine was shut down


def test_every_request_s_sampling_params_are_read_back(protocol, ckpts) -> None:
    class IgnoresSeedForSamples(Params):
        def __init__(self, **kw):
            super().__init__(**{**kw, "seed": None if kw["n"] > 1 else kw["seed"]})

    e = _engine(protocol, ckpts, FakeBackend(params=IgnoresSeedForSamples))  # n=1 probe passes
    with pytest.raises(SettingsError, match="seed"):
        e.generate_batch([_req(protocol, n=4)])


# --- C1: storage and provenance -------------------------------------------------------------


def test_storage_checks_and_what_they_do_not_establish(ckpts, tmp_path) -> None:
    check = _storage_for(tmp_path / "ckpt")
    report = check(ckpts[0])
    assert report["read_only_mount"] is True and report["replaceable_parents"] == []
    assert len(report["devices"]) == 1
    listed = " ".join(report["not_established_locally"])
    assert "bind mount" in listed and "mount over" in listed and "remount" in listed
    rw = _storage_for(tmp_path / "ckpt", statvfs=lambda p: SimpleNamespace(f_flag=0))
    with pytest.raises(ProvenanceError, match="not on a read-only mount"):
        rw(ckpts[0])
    # One file bind-mounted from writable storage into the read-only tree.
    target = ckpts[0] / "config.json"

    def one_rw_file(p):
        return SimpleNamespace(f_flag=0 if Path(p) == target else os.ST_RDONLY)

    with pytest.raises(ProvenanceError, match=r"not on a read-only mount: \['config.json'\]"):
        _storage_for(tmp_path / "ckpt", statvfs=one_rw_file)(ckpts[0])
    target.chmod(0o644)
    with pytest.raises(ProvenanceError, match="writable by this process"):
        check(ckpts[0])
    target.chmod(0o444)
    link = tmp_path / "link"
    link.symlink_to(ckpts[0])
    with pytest.raises(ProvenanceError, match="symbolic link"):
        check(link)
    with pytest.raises(ProvenanceError, match=r"contains '\.\.'"):
        check(ckpts[0] / ".." / "base")


def test_a_parent_owned_by_this_user_is_replaceable(ckpts, tmp_path) -> None:
    # tmp_path/ckpt is owned by the test user with mode 0555: its owner could chmod it, rename
    # the checkpoint away and put another directory in its place, so it is refused.
    check = _storage_for(tmp_path)  # tmp_path/ckpt is now checked for real
    with pytest.raises(ProvenanceError, match=r"ckpt \(not owned by root\)"):
        check(ckpts[0])


@pytest.mark.parametrize(
    ("uid", "mode", "refused"),
    [
        (0, 0o755, None),
        (0, 0o775, "group/other-writable"),
        (0, 0o1777, "group/other-writable"),  # no sticky-directory exemption
        (1000, 0o755, "not owned by root"),
    ],
)
def test_parents_must_be_root_owned_and_not_writable(ckpts, tmp_path, uid, mode, refused) -> None:
    protected = _storage_for(tmp_path / "ckpt")
    lstat = protected.keywords["lstat"]

    def with_root_dir(p):
        if Path(p) == Path("/"):
            return os.stat_result((stat.S_IFDIR | mode, 0, 0, 0, uid, *[0] * 5))
        return lstat(p)

    check = partial(protected, lstat=with_root_dir)
    if refused is None:
        check(ckpts[0])
    else:
        with pytest.raises(ProvenanceError, match=rf"/ \({refused}"):
            check(ckpts[0])


def test_writable_checkpoint_is_refused(protocol, ckpts) -> None:
    (ckpts[0] / "config.json").chmod(0o644)
    backend = FakeBackend()
    with pytest.raises(ProvenanceError, match="read-only"):
        _engine(protocol, ckpts, backend)
    assert backend.llms == []


def _write(name, data=b"x"):
    def change(d):
        target = d / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    return change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_write(WEIGHTS, b"tampered"), "differs from the registered"),
        (lambda d: (d / WEIGHTS).unlink(), "missing"),
        (_write("chat_template.jinja", b"other"), "git blob"),
        (lambda d: (d / "chat_template.jinja").unlink(), "git blob"),
        (_write("model-00002-of-00002.safetensors"), "not registered"),
        (_write("generation_config.json", b"{}"), "not registered"),
        (_write("sub/README.md"), "not registered"),  # doc names count only at the root
        (_write("README.safetensors"), "not registered"),
        (_write(".cache/huggingface/download/x.metadata"), "not registered"),
        (lambda d: (d / "tokenizer.json").unlink(), "registered files missing"),
        (_write("config.json", b'{"changed": true}'), "differ from their registered"),
        (lambda d: os.mkfifo(d / "tokenizer_config.json"), "not a regular file"),
    ],
)
def test_provenance_mismatch_refuses_before_loading(protocol, ckpts, change, message) -> None:
    with _unlocked(ckpts[0]) as d:
        change(d)
    backend = FakeBackend()
    with pytest.raises((ProvenanceError, ValueError), match=message):
        _engine(protocol, ckpts, backend)
    assert backend.llms == []  # nothing was loaded


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda a: a["checkpoint"].update(weights_sha256_lfs=""), "weights_sha256_lfs"),
        (lambda a: a["checkpoint"].pop("loadable_files_sha256"), "loadable_files_sha256"),
        (lambda a: a["checkpoint"].update(loadable_files_sha256={}), "loadable_files_sha256"),
        (
            lambda a: a["checkpoint"]["loadable_files_sha256"].pop(WEIGHTS),
            "must list weights_file",
        ),
        (
            lambda a: a["checkpoint"]["loadable_files_sha256"].pop("chat_template.jinja"),
            "must list chat_template",
        ),
    ],
)
def test_registration_must_be_complete(ckpts, edit, message) -> None:
    backend = FakeBackend()
    with pytest.raises(ProvenanceError, match=message):
        _engine(_modified(ckpts, edit), ckpts, backend)
    assert backend.llms == []


def test_a_registered_documentation_file_counts_as_loaded(ckpts) -> None:
    readme = hash_tree(ckpts[0])["README.md"]
    raw = _modified(
        ckpts,
        lambda a: a["checkpoint"]["loadable_files_sha256"].update({"README.md": readme}),
        ["base", "base_reload"],
    )
    d = _engine(raw, ckpts).describe()
    assert "README.md" in d["artifacts_sha256"] and d["other_files_sha256"] == {}


def test_symlinked_checkpoint_files_are_refused(protocol, ckpts, tmp_path) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    with _unlocked(ckpts[0]) as d:
        (d / "extra.json").symlink_to(target)
    with pytest.raises(ProvenanceError, match="symbolic link"):
        _engine(protocol, ckpts)
    with pytest.raises(ValueError, match="symbolic link"):
        hash_tree(ckpts[0])  # the inventory refuses it too


def test_tokenizer_template_difference_is_observed(protocol, ckpts) -> None:
    class OtherTemplate(Tokenizer):
        chat_template = "{# a different built-in template #}"

    e = _engine(protocol, ckpts, FakeBackend(tokenizer=OtherTemplate))
    assert e.describe()["observed"]["tokenizer_builtin_template_matches"] is False
    e.generate_batch([_req(protocol)])  # still renders with the registered bytes


# --- C1: failures after vLLM was created ---------------------------------------------------


def test_failure_after_engine_creation_shuts_it_down(protocol, ckpts) -> None:
    def broken():
        raise RuntimeError("synthetic tokenizer failure")

    backend = FakeBackend(tokenizer=broken)
    reports: list = []
    with pytest.raises(RuntimeError, match="synthetic tokenizer failure"):
        _engine(protocol, ckpts, backend, reports=reports)
    # The real shutdown steps ran against the created engine.
    assert backend.events == [
        "llm_engine.shutdown",
        "destroy_model_parallel",
        "destroy_distributed_environment",
        "torch.cuda.empty_cache",
    ]
    assert backend.hub.alive_at_empty_cache == [False]  # released before the cache step
    assert backend.llms[0]() is None
    (report,) = reports
    assert report["phase"] == "load_failed" and "synthetic tokenizer" in report["load_error"]
    assert report["errors"] == []


def test_failed_engine_construction_runs_process_level_cleanup(protocol, ckpts) -> None:
    backend = FakeBackend(llm_error=RuntimeError("synthetic construction failure"))
    reports: list = []
    with pytest.raises(RuntimeError, match="construction"):
        _engine(protocol, ckpts, backend, reports=reports)
    # No engine object: only the process-level steps ran.
    assert backend.events == [
        "destroy_model_parallel",
        "destroy_distributed_environment",
        "torch.cuda.empty_cache",
    ]
    assert reports[0]["phase"] == "load_failed"


# --- C1: batches and request echoes -------------------------------------------------------


def test_batch_items_echo_requests_and_carry_prompt_hashes(protocol, ckpts) -> None:
    backend = FakeBackend()
    e = _engine(protocol, ckpts, backend)
    reqs = [
        _req(protocol, item=f"dev-{k:03d}", prompt=f"Synthetic: compute {k} + 1.", seed=0)
        for k in range(3)
    ]
    out = e.generate_batch(reqs)
    assert [b.request for b in out] == reqs  # echoes, in request order
    (call,) = backend.calls  # one engine call for the whole shard
    assert len(call) == 3
    rendered, params = call[0]
    assert "Synthetic: compute 0 + 1.\nPlease reason step by step" in rendered
    assert rendered.endswith("<assistant><think>")  # thinking enabled
    assert params == {
        "n": 4,
        "seed": 0,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 64,
    }
    gens = out[0].result
    assert len(gens) == 4 and all(g.finish_reason == "stop" for g in gens)
    assert gens[0].rendered_prompt_sha256 == hashlib.sha256(rendered.encode()).hexdigest()
    assert gens[0].output_tokens == 12


def test_outputs_that_do_not_match_their_prompt_fail_the_batch(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts, FakeBackend(post=lambda outs: list(reversed(outs))))
    reqs = [
        _req(protocol, item=f"dev-{k}", prompt=f"Synthetic: compute {k} + 1.") for k in range(2)
    ]
    with pytest.raises(BatchMismatch, match="not for the prompt"):
        e.generate_batch(reqs)
    e2 = _engine(protocol, ckpts, FakeBackend(post=lambda outs: outs[:-1]))
    with pytest.raises(BatchMismatch, match="returned 1"):
        e2.generate_batch(reqs)


def test_a_request_for_other_settings_fails_only_itself(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts)
    good = _req(protocol)
    bad = GenerationRequest(
        "base", "dev-x", "Synthetic: compute 1 + 1.", {**good.inference, "temperature": 0.0}, 0, 4
    )
    out = e.generate_batch([good, bad])
    assert isinstance(out[1].result, ValueError) and isinstance(out[0].result, list)


def test_single_sample_requests_return_one_generation(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts)
    (item,) = e.generate_batch([_req(protocol, n=1)])
    assert not isinstance(item.result, list) and item.result.finish_reason == "stop"


# --- C1: close and engine lifetime ----------------------------------------------------------

_ALL_STEPS = [
    "llm_engine.shutdown",
    "parallel_state.destroy_model_parallel",
    "parallel_state.destroy_distributed_environment",
    "gc.collect",
    "torch.cuda.empty_cache",
]


def test_close_runs_every_step_releases_and_reports(protocol, ckpts) -> None:
    backend = FakeBackend()
    reports: list = []
    e = _engine(protocol, ckpts, backend, reports=reports)
    e.close()
    e.close()  # idempotent after success
    gc.collect()
    assert backend.llms[0]() is None  # the engine dropped its LLM object
    (report,) = reports
    assert [s["step"] for s in report["shutdown_steps"]] == _ALL_STEPS
    assert {s["status"] for s in report["shutdown_steps"]} == {"ok"} and report["errors"] == []
    assert backend.events[-1] == "torch.cuda.empty_cache"
    # The references were dropped before collection and cache clearing (adapter control flow;
    # whether the real allocator returns the memory is a GPU-host check).
    assert backend.hub.alive_at_empty_cache == [False]
    assert report["files_unchanged"] is True
    assert report["parent_process_torch_allocated_bytes"] == 0
    assert report["multiprocessing_children_after_close"] == []
    assert "worker-process exit" in report["note"] and "version-dependent" in report["note"]
    with pytest.raises(RuntimeError, match="closed"):
        e.generate_batch([_req(protocol)])


def test_v1_engine_core_shutdown_is_used_when_present(protocol, ckpts) -> None:
    reports: list = []
    _engine(protocol, ckpts, FakeBackend(engine_mode="v1"), reports=reports).close()
    assert reports[0]["shutdown_steps"][0]["step"] == "llm_engine.engine_core.shutdown"


def test_close_fails_if_files_changed_parent_memory_or_children_remain(
    protocol, ckpts, monkeypatch
) -> None:
    e = _engine(protocol, ckpts)
    with _unlocked(ckpts[0]) as d:
        (d / "config.json").write_text('{"changed": true}')
    with pytest.raises(ProvenanceError, match="changed during the run"):
        e.close()
    with _unlocked(ckpts[0]) as d:
        (d / "config.json").write_text('{"model_type": "synthetic"}')
    e2 = _engine(protocol, ckpts, FakeBackend(memory=10), max_parent_memory_after_close=0)
    with pytest.raises(CloseError, match="reserved still in this process"):
        e2.close()
    reports: list = []
    e4 = _engine(
        protocol,
        ckpts,
        FakeBackend(reserved_extra=10),
        max_parent_memory_after_close=0,
        reports=reports,
    )
    with pytest.raises(CloseError, match="0 bytes allocated / 10 reserved"):
        e4.close()  # only reserved memory remains: still a failure
    assert reports[0]["parent_process_torch_reserved_bytes"] == 10
    e3 = _engine(protocol, ckpts)
    monkeypatch.setattr(
        vllm_adapter.multiprocessing, "active_children", lambda: [SimpleNamespace(pid=4242)]
    )
    with pytest.raises(CloseError, match="child processes"):
        e3.close()


def test_failed_shutdown_keeps_every_step_and_is_never_reported_as_success(protocol, ckpts) -> None:
    backend = FakeBackend(engine_error=RuntimeError("synthetic engine shutdown failure"))
    reports: list = []
    e = _engine(protocol, ckpts, backend, reports=reports)
    with pytest.raises(CloseError, match="synthetic engine shutdown failure"):
        e.close()
    (report,) = reports
    steps = report["shutdown_steps"]
    assert [s["step"] for s in steps] == _ALL_STEPS  # the later steps still ran
    assert [s["status"] for s in steps] == ["failed", "ok", "ok", "ok", "ok"]
    assert backend.events[1:] == [
        "destroy_model_parallel",
        "destroy_distributed_environment",
        "torch.cuda.empty_cache",
    ]
    assert report["files_unchanged"] is True  # the re-hash still ran
    gc.collect()
    assert backend.llms[0]() is None  # references dropped despite the failure
    with pytest.raises(CloseError, match="earlier close failed"):
        e.close()  # never silently successful afterwards
    with pytest.raises(CloseError, match="earlier close failed"):
        e.close()
    with pytest.raises(RuntimeError, match="closed"):
        e.generate_batch([_req(protocol)])


def test_a_missing_engine_shutdown_hook_fails_the_close(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts, FakeBackend(engine_mode="none"))
    with pytest.raises(CloseError, match="not provided by the installed version"):
        e.close()
    assert step_failures(e.close_report["shutdown_steps"])


def test_an_interrupt_during_shutdown_still_releases_and_records(protocol, ckpts) -> None:
    backend = FakeBackend(engine_error=KeyboardInterrupt())
    reports: list = []
    e = _engine(protocol, ckpts, backend, reports=reports)
    with pytest.raises(KeyboardInterrupt):
        e.close()
    assert backend.llms[0]() is None  # released before the interrupt propagated
    (report,) = reports
    assert report["interrupted"] is True and report["errors"]
    steps = report["shutdown_steps"]
    assert steps[0]["status"] == "failed"
    assert [s["step"] for s in steps[1:]] == _ALL_STEPS[1:]  # the release steps still ran
    assert backend.hub.alive_at_empty_cache == [False]
    assert "files_unchanged" not in report  # memory, children and re-hash checks do not run
    with pytest.raises(CloseError, match="earlier close failed"):
        e.close()


def test_backend_shutdown_exception_is_recorded(protocol, ckpts) -> None:
    class Raising(FakeBackend):
        def shutdown_engine(self, llm):
            raise RuntimeError("synthetic shutdown crash")

    reports: list = []
    e = _engine(protocol, ckpts, Raising(), reports=reports)
    with pytest.raises(CloseError, match="synthetic shutdown crash"):
        e.close()
    steps = reports[0]["shutdown_steps"]
    assert steps[0]["status"] == "failed"
    assert [s["step"] for s in steps[1:]] == _ALL_STEPS[1:]  # release still ran


def test_real_backend_without_cuda_or_parallel_state() -> None:
    backend = VllmBackend(vllm=None, torch=None, version="v", transformers_version="t")
    steps = backend.release()
    assert [s["status"] for s in steps] == ["unavailable", "unavailable", "ok", "unavailable"]
    assert step_failures(steps) == []  # only the engine's own shutdown is required
    (missing,) = backend.shutdown_engine(SimpleNamespace(llm_engine=SimpleNamespace()))
    assert missing["status"] == "unavailable" and step_failures([missing])
    assert backend.parent_torch_memory() is None
    assert "gpu" not in backend.environment()  # no device query in the parent process


# --- end to end through the harness runner ----------------------------------------------


def _train_rows(n=5):
    return [
        {
            "unique_id": f"dev-{k:03d}",
            "problem": f"Synthetic: compute {k} + {k + 1}.",
            "solution": f"so \\boxed{{{2 * k + 1}}}",
        }
        for k in range(n)
    ]


def _argv(
    tmp_path,
    ckpts,
    *,
    protocol_raw=None,
    split="development",
    register=None,
    **data,
):
    """Write the data files, register them in the protocol (``register`` may then alter the
    registration), and return the CLI arguments."""
    train, held, sources = _data(tmp_path, **data)
    raw = copy.deepcopy(protocol_raw or _protocol_raw(*ckpts))
    raw["dataset"]["runtime_sources"] = sources
    if register is not None:
        register(raw["dataset"])
    ppath = tmp_path / "protocol.json"
    ppath.write_text(json.dumps(raw))
    source, reference = (train, held) if split == "development" else (held, train)
    argv = [
        "--protocol",
        str(ppath),
        "--source",
        str(source),
        "--leakage-reference",
        str(reference),
    ]
    for arm, d in (("base", ckpts[0]), ("edited", ckpts[1]), ("base_reload", ckpts[0])):
        argv += ["--checkpoint", f"{arm}={d}"]
    argv += ["--split", split, "--operator", "op1", "--purpose", "stub"]
    return [*argv, "--output", str(tmp_path / "run")]


def _lookup(**override):
    versions = {"vllm": VERSION, "transformers": TF_VERSION, **override}

    def lookup(name):
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    return lookup


def _cli(
    tmp_path,
    protocol_raw,
    ckpts,
    *,
    backend=None,
    extra=(),
    verify=ExactGrader,
    lookup=None,
    environ=None,
    params=Params,
    **kw,
):
    backend = backend or FakeBackend()
    argv = _argv(tmp_path, ckpts, protocol_raw=protocol_raw, **kw) + list(extra)
    summary = runtime_cli.run(
        argv,
        backend_factory=lambda: backend,
        verify_backend_factory=verify,
        dependency_lookup=lookup or _lookup(),
        storage_check=STORAGE["check"],
        sampling_params_factory=lambda: params,
        environ={} if environ is None else environ,
    )
    return summary, backend


def test_end_to_end_run_with_stub_engines(tmp_path, protocol, ckpts) -> None:
    summary, backend = _cli(tmp_path, protocol.raw, ckpts, extra=["--keep-outputs"])
    assert summary["run_status"] == "complete" and summary["error_records"] == 0
    manifest, records = verify_run(tmp_path / "run", protocol)
    assert manifest["run_id"] == summary["run_id"]
    assert len(records) == 5 * 4 * 3 and {r["status"] for r in records} == {"graded"}
    assert all(r["correct"] for r in records)
    loads = manifest["execution"]["loads"]
    assert [x["arm"] for x in loads] == ["base", "edited", "base_reload"]
    assert all(x["status"] == "complete" and x["closed"] is True for x in loads)
    assert len({x["load_id"] for x in loads}) == 3
    gens = [a["generator"] for a in manifest["arms"]]
    assert gens[0]["artifacts_sha256"] == gens[2]["artifacts_sha256"]  # same files reloaded
    assert gens[0]["artifacts_sha256"][WEIGHTS] != gens[1]["artifacts_sha256"][WEIGHTS]
    (d,) = drift_report(protocol, manifest, records)
    checks = {c["check"]: c["status"] for c in d["identity"]["checks"]}
    assert checks["separate_load"] == "pass"
    assert len(summary["close_reports"]) == 3
    assert [len(c) for c in backend.calls] == [3, 2] * 3  # two shards per arm, file order
    # Supplementary dataset and leakage provenance, bound to this run's manifest; its hash is
    # printed for external recording (which is what would protect it).
    path = tmp_path / "run" / "runtime_report.json"
    assert summary["runtime_report_sha256"] == sha256_file(path)
    report = json.loads(path.read_text())
    assert report["run_id"] == manifest["run_id"]
    assert report["manifest_sha256"] == sha256_file(tmp_path / "run" / "manifest.json")
    assert report["manifest_items_sha256_matches"] is True
    assert report["dataset"]["id_source"] == "unique_id"
    leakage = report["leakage_check"]
    assert leakage["ok"] is True and "cannot prove" in leakage["limit"]
    assert leakage["n_reference"] == 3
    assert report["leakage_reference"]["shape"] == "math500"
    assert (
        report["dataset"]["source_sha256"]
        == protocol.dataset["runtime_sources"]["development"]["source_file_sha256"]
    )
    assert set(report["hash_definitions"]) == {
        "source_file_sha256",
        "source_rows",
        "source_ids_sha256",
        "items_sha256",
    }
    assert report["forced_environment"]["VLLM_NO_USAGE_STATS"] == "1"
    assert report["installed_versions"] == {"vllm": VERSION, "transformers": TF_VERSION}
    assert report["grader"]["reviewed"] is False
    assert [p["arm"] for p in report["preflight"]] == ["base", "edited", "base_reload"]
    assert [r["errors"] for r in report["close_reports"]] == [[], [], []]


def test_stub_grader_exceptions_become_error_rows_not_incorrect(tmp_path, protocol, ckpts) -> None:
    # Only shows propagation of an exception the backend raises; whether math-verify 0.9.0
    # raises on a timeout is a check against the real library.
    class Raises:
        def equivalent(self, gold, pred):
            raise TimeoutError("synthetic stub timeout")

    summary, _ = _cli(tmp_path, protocol.raw, ckpts, verify=Raises)
    assert summary["error_records"] == 5 * 4 * 3  # main() exits 1 on any error record
    _, records = verify_run(tmp_path / "run", protocol)
    assert {r["error_class"] for r in records} == {"GraderError:TimeoutError"}
    assert all(r["correct"] is None for r in records)


def test_main_exits_nonzero_on_error_records_or_a_missing_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runtime_cli, "run", lambda argv: {"error_records": 3})
    assert runtime_cli.main([]) == 1
    assert "3 records are errors" in capsys.readouterr().err
    monkeypatch.setattr(
        runtime_cli, "run", lambda argv: {"error_records": 0, "runtime_report_error": "E"}
    )
    assert runtime_cli.main([]) == 1
    assert "runtime report not written" in capsys.readouterr().err


def test_earlier_engines_are_collectable_before_the_next_load(tmp_path, protocol, ckpts) -> None:
    # Python reference lifetime only: real GPU memory release needs the GPU host.
    alive: list = []
    backend = FakeBackend()
    original = backend.make_llm

    def make_llm(**kw):
        gc.collect()
        alive.append([r() is not None for r in backend.llms])
        return original(**kw)

    backend.make_llm = make_llm
    summary, _ = _cli(tmp_path, protocol.raw, ckpts, backend=backend)
    assert summary["run_status"] == "complete"
    assert alive == [[], [False], [False, False]]


def test_parent_memory_left_after_close_stops_before_the_next_arm(
    tmp_path, protocol, ckpts
) -> None:
    backend = FakeBackend(memory=1024)
    with pytest.raises(runtime_cli._Aborted) as exc:
        _cli(
            tmp_path,
            protocol.raw,
            ckpts,
            backend=backend,
            extra=["--max-parent-torch-memory-after-close", "0"],
        )
    assert "reserved still in this process" in exc.value.summary["error"]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["run_status"] == "aborted"
    assert [x["status"] for x in manifest["execution"]["loads"]] == ["close_failed"]
    assert len(backend.calls) == 2  # only the first arm ran
    report = json.loads((tmp_path / "run" / "runtime_report.json").read_text())
    assert report["close_reports"][0]["errors"]  # the failure is kept in the report


def test_files_changed_during_a_run_are_kept_in_the_close_report(tmp_path, protocol, ckpts) -> None:
    def tamper(prompt, params):
        with _unlocked(ckpts[0]) as d:
            (d / "config.json").write_text('{"changed": true}')
        return _correct(prompt, params)

    with pytest.raises(runtime_cli._Aborted) as exc:
        _cli(tmp_path, protocol.raw, ckpts, backend=FakeBackend(answer=tamper))
    (report,) = exc.value.summary["close_reports"]
    assert report["files_unchanged"] is False
    assert report["errors"][0] == "checkpoint files changed during the run"


@pytest.mark.parametrize(
    ("setup", "error"),
    [
        (
            lambda raw, ckpts: _modified(
                ckpts, lambda a: a["engine"].update(tensor_parallel_size=1.0), ["edited"]
            ),
            SettingsError,
        ),
        (lambda raw, ckpts: raw, ProvenanceError),
    ],
)
def test_every_arm_is_checked_before_any_arm_runs(tmp_path, protocol, ckpts, setup, error) -> None:
    raw = setup(copy.deepcopy(protocol.raw), ckpts)
    if error is ProvenanceError:
        with _unlocked(ckpts[1]) as d:
            (d / "extra.safetensors").write_bytes(b"x")  # unregistered file in arm 2
    backend = FakeBackend()
    with pytest.raises(error):
        _cli(tmp_path, raw, ckpts, backend=backend)
    assert backend.calls == [] and backend.llms == []  # no arm generated or loaded
    assert not (tmp_path / "run").exists()  # and nothing was written


def test_a_report_is_written_only_into_this_invocation_s_run(tmp_path, protocol, ckpts) -> None:
    previous = tmp_path / "run"
    previous.mkdir()
    (previous / "manifest.json").write_text(json.dumps({"run_id": "wsp-earlier"}))
    with pytest.raises(runtime_cli._Aborted) as exc:
        _cli(tmp_path, protocol.raw, ckpts)
    summary = exc.value.summary
    assert "not empty" in summary["error"]  # the original failure is kept
    assert summary["runtime_report_skipped"] == "the manifest there belongs to another run"
    assert sorted(p.name for p in previous.iterdir()) == ["manifest.json"]  # untouched


def test_a_failed_report_write_does_not_replace_the_original_failure(
    tmp_path, protocol, ckpts, monkeypatch
) -> None:
    def broken(*args):
        raise OSError("synthetic disk full")

    monkeypatch.setattr(runtime_cli, "_write_report", broken)
    with pytest.raises(runtime_cli._Aborted) as exc:
        _cli(
            tmp_path,
            protocol.raw,
            ckpts,
            backend=FakeBackend(memory=1),
            extra=["--max-parent-torch-memory-after-close", "0"],
        )
    summary = exc.value.summary
    assert "reserved still in this process" in summary["error"]
    assert summary["runtime_report_error"] == "OSError: synthetic disk full"


def test_cli_refuses_wrong_checkpoint_mapping_and_protocols(tmp_path, protocol, ckpts) -> None:
    with pytest.raises(ValueError, match="exactly the protocol arms"):
        runtime_cli._checkpoints(["base=/a", "edited=/b"], ["base", "edited", "base_reload"])
    with pytest.raises(ValueError, match="twice"):
        runtime_cli._checkpoints(["base=/a", "base=/b"], ["base"])
    with pytest.raises(ValueError, match="KEY=PATH"):
        runtime_cli._checkpoints(["base"], ["base"])
    interleaved = copy.deepcopy(protocol.raw)
    del interleaved["execution"]
    with pytest.raises(ValueError, match="per_arm_sequential"):
        _cli(tmp_path, interleaved, ckpts)


@pytest.mark.parametrize(
    ("register", "message"),
    [
        (lambda d: d.pop("runtime_sources"), "must register dataset.runtime_sources"),
        (lambda d: d["runtime_sources"].update(hash_definitions="v0"), "hash_definitions"),
        (lambda d: d["runtime_sources"].pop("heldout"), "does not register"),
        (
            lambda d: d["runtime_sources"]["heldout"].update(source_file_sha256="0" * 64),
            "SHA-256 differs",
        ),
        (
            lambda d: d["runtime_sources"]["development"].update(source_file_sha256="REPLACE"),
            "registered SHA-256",
        ),
        (lambda d: d["runtime_sources"]["development"].update(source_rows=4), "expected 4 items"),
        (
            lambda d: d["runtime_sources"]["development"].update(source_ids_sha256="0" * 64),
            "ID set differs",
        ),
        (
            lambda d: d["runtime_sources"]["development"].update(items_sha256="0" * 64),
            "items_sha256 differs",
        ),
        (
            lambda d: d["runtime_sources"]["development"].update(selection={"rule": "random"}),
            "selection.rule",
        ),
        (
            lambda d: d["runtime_sources"]["development"].update(
                selection={"rule": "first_n_in_file_order", "n": 9}
            ),
            "more than",
        ),
        (
            lambda d: d["runtime_sources"]["heldout"].pop("items_sha256"),
            "missing \\['items_sha256'\\]",
        ),
        (
            lambda d: d["runtime_sources"]["heldout"].update(
                selection={"rule": "first_n_in_file_order", "n": 1}
            ),
            "all evaluated",
        ),
        (lambda d: d["runtime_sources"]["heldout"].update(shape="math_train"), "shape"),
        (lambda d: d.update(n_items=500), "differs from dataset.n_items"),
        (lambda d: d["runtime_sources"]["development"].update(limit=2), "not understood"),
    ],
)
def test_cli_dataset_identity_comes_from_the_protocol(
    tmp_path, protocol, ckpts, register, message
) -> None:
    backend = FakeBackend()
    with pytest.raises(datasets.DatasetLoadError, match=message):
        _cli(tmp_path, protocol.raw, ckpts, backend=backend, register=register)
    assert not (tmp_path / "run").exists() and backend.calls == []


def test_cli_leaks_are_checked_on_the_whole_source_before_the_selection(
    tmp_path, protocol, ckpts
) -> None:
    # The same problem as development row 4, formatted differently with a differently written
    # answer: refused although the selection (first 2 rows) would not include that row.
    leaked = [
        *_math500_rows(),
        {
            "unique_id": "synthetic/held.json",
            "problem": "  SYNTHETIC:   compute 4 + 5. ",
            "answer": "\\frac{18}{2}",
        },
    ]
    with pytest.raises(datasets.DatasetLoadError, match="leakage: 1 problems"):
        _cli(
            tmp_path,
            protocol.raw,
            ckpts,
            held_rows=leaked,
            dev_selection={"rule": "first_n_in_file_order", "n": 2},
        )
    assert not (tmp_path / "run").exists()  # refused before anything ran
    with pytest.raises(datasets.DatasetLoadError, match="contains no items"):
        _argv(tmp_path, ckpts, held_rows=[])  # an empty reference cannot even be registered


def test_cli_applies_the_registered_selection(tmp_path, protocol, ckpts) -> None:
    summary, _ = _cli(
        tmp_path, protocol.raw, ckpts, dev_selection={"rule": "first_n_in_file_order", "n": 2}
    )
    assert summary["counts"]["records"] == 2 * 4 * 3
    report = json.loads((tmp_path / "run" / "runtime_report.json").read_text())
    assert report["dataset"]["limit"] == 2 and report["dataset"]["n_source_rows"] == 5
    assert report["leakage_check"]["n_items"] == 5  # the whole source was checked


@pytest.mark.parametrize(
    ("kw", "error", "message"),
    [
        ({"environ": {"HF_HUB_OFFLINE": "0"}}, SettingsError, "HF_HUB_OFFLINE"),
        ({"environ": {"VLLM_NO_USAGE_STATS": "0"}}, SettingsError, "VLLM_NO_USAGE_STATS"),
        ({"lookup": _lookup(vllm="9.9.9")}, ProvenanceError, "installed vllm is '9.9.9'"),
        ({"lookup": "no-transformers"}, ProvenanceError, "installed transformers is None"),
    ],
)
def test_environment_and_installed_versions_are_checked_before_any_output(
    tmp_path, protocol, ckpts, kw, error, message
) -> None:
    if kw.get("lookup") == "no-transformers":
        kw = {"lookup": _lookup_without("transformers")}
    backend = FakeBackend()
    with pytest.raises(error, match=message):
        _cli(tmp_path, protocol.raw, ckpts, backend=backend, **kw)
    assert not (tmp_path / "run").exists() and backend.calls == [] and backend.llms == []


def _lookup_without(package):
    full = _lookup()

    def lookup(name):
        if name == package:
            raise PackageNotFoundError(name)
        return full(name)

    return lookup


def test_sampling_read_back_is_checked_before_any_output(tmp_path, protocol, ckpts) -> None:
    class IgnoresSeedForSamples(Params):
        def __init__(self, **kw):
            super().__init__(**{**kw, "seed": None if kw["n"] > 1 else kw["seed"]})

    backend = FakeBackend()
    with pytest.raises(SettingsError, match="seed"):
        _cli(tmp_path, protocol.raw, ckpts, backend=backend, params=IgnoresSeedForSamples)
    assert not (tmp_path / "run").exists() and backend.llms == []


def test_cli_checks_grader_dependencies(tmp_path, protocol, ckpts) -> None:
    raw = copy.deepcopy(protocol.raw)
    raw["grader"]["dependencies"] = {"math-verify": "0.9.0", "sympy": "REPLACE (open)"}
    with pytest.raises(ValueError, match=r"pins '0\.9\.0'"):
        _cli(tmp_path, raw, ckpts, lookup=_lookup(**{"math-verify": "0.8.0"}))
    assert not (tmp_path / "run").exists()
    _cli(tmp_path, raw, ckpts, lookup=_lookup(**{"math-verify": "0.9.0", "sympy": "1.13"}))
    report = json.loads((tmp_path / "run" / "runtime_report.json").read_text())
    assert report["grader"]["dependencies"] == {
        "math-verify": {"pinned": "0.9.0", "installed": "0.9.0"},
        "sympy": {"pinned": None, "installed": "1.13"},
    }


def test_cli_refuses_heldout_while_the_grader_is_unreviewed(tmp_path, protocol, ckpts) -> None:
    assert REVIEWED is False and "unreviewed" in GRADER_REVISION
    with pytest.raises(ValueError, match="grader is unreviewed"):
        _cli(
            tmp_path,
            protocol.raw,
            ckpts,
            split="heldout",
            extra=["--heldout-authorized-by", "evaluator"],
        )
    assert not (tmp_path / "run").exists()


# --- C2: grader -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so \\boxed{7}.", "7"),
        ("first \\boxed{1} then \\boxed{2}", "2"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("\\fbox{9} and later \\boxed{3}", "3"),
        ("\\boxed{3} and later \\fbox{9}", "9"),
        ("\\boxed {4}", "4"),
        ("\\boxed{\\{1,2\\}}", "\\{1,2\\}"),
        ("\\boxed{\\boxed{3}}", "3"),
        ("\\boxed{unbalanced", None),
        ("\\boxed{1} then \\boxed{\\left\\{ x \\right.}", None),  # no fallback to the earlier box
        ("\\boxed{1} then \\boxed 5", None),  # the last box is unbraced: no answer
        ("no box here", None),
        ("\\boxed 5", None),
        ("\\boxed{}", ""),
    ],
)
def test_last_boxed(text, expected) -> None:
    assert last_boxed(text) == expected


def test_reasoning_block_is_cut_only_when_closed() -> None:
    assert strip_reasoning("<think>\\boxed{1}</think>final \\boxed{2}") == "final \\boxed{2}"
    assert strip_reasoning("<think>still thinking \\boxed{1}") == "<think>still thinking \\boxed{1}"


def test_grader_scores_boxed_answers_and_unboxed_or_empty_as_incorrect() -> None:
    calls: list = []

    class Recording(ExactGrader):
        def equivalent(self, gold, pred):
            calls.append(pred)
            return super().equivalent(gold, pred)

    g = MathQAGrader(Recording())
    item = Item(id="x", split="development", prompt="p", answer="12")
    assert g.grade(item, "<think>\\boxed{11}</think>\\boxed{12}").correct is True
    assert g.grade(item, "<think>\\boxed{12}</think>so 11").correct is False  # box was reasoning
    no_box = g.grade(item, "the answer is 12")
    assert no_box.correct is False and no_box.detail == "no boxed answer"  # graded, not ungraded
    empty = g.grade(item, "\\boxed{ }")
    assert empty.correct is False and empty.detail == "empty boxed answer"
    assert calls == ["12"]  # math-verify not called for unboxed or empty answers
    assert (g.id, g.revision, g.reviewed) == (GRADER_ID, GRADER_REVISION, False)


def _module(parsed=None, raise_in=None):
    calls: list = []

    class Module:
        @staticmethod
        def parse(text, parsing_timeout, raise_on_error):
            calls.append(("parse", text, parsing_timeout, raise_on_error))
            if raise_in == "parse":
                raise TimeoutError("synthetic parse timeout")
            return (parsed or {}).get(text, [text])

        @staticmethod
        def verify(g, p, timeout_seconds, raise_on_error):
            calls.append(("verify", g, p, timeout_seconds, raise_on_error))
            if raise_in == "verify":
                raise TimeoutError("synthetic verify timeout")
            return g == p

    return Module, calls


def test_math_verify_backend_propagates_stub_exceptions_and_parse_failures() -> None:
    # A stand-in module: shows the arguments sent and that raised exceptions propagate, not
    # what math-verify 0.9.0 does on a timeout.
    module, calls = _module()
    assert MathVerifyBackend(module).equivalent("1", "1") is True
    assert calls == [
        ("parse", "$1$", 5, True),
        ("parse", "$1$", 5, True),
        ("verify", ["$1$"], ["$1$"], 5, True),
    ]
    for where in ("parse", "verify"):
        module, _ = _module(raise_in=where)
        with pytest.raises(TimeoutError):
            MathVerifyBackend(module).equivalent("1", "1")
    module, _ = _module(parsed={"$g$": []})
    with pytest.raises(GraderInputError, match="gold"):
        MathVerifyBackend(module).equivalent("g", "1")
    module, _ = _module(parsed={"$p$": []})
    assert MathVerifyBackend(module).equivalent("1", "p") is False


def test_math_verify_backend_refuses_other_threads() -> None:
    module, _ = _module()
    errors: list = []

    def work():
        try:
            MathVerifyBackend(module).equivalent("1", "1")
        except RuntimeError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert errors and "main thread" in errors[0]


def test_grader_dependencies_are_checked_against_the_protocol() -> None:
    installed = {"math-verify": "0.9.0", "sympy": "1.13"}

    def lookup(name):
        if name not in installed:
            raise PackageNotFoundError(name)
        return installed[name]

    assert check_dependencies({"math-verify": "0.9.0", "sympy": "REPLACE"}, lookup) == {
        "math-verify": {"pinned": "0.9.0", "installed": "0.9.0"},
        "sympy": {"pinned": None, "installed": "1.13"},
    }
    with pytest.raises(ValueError, match=r"pins '0\.8\.0'"):
        check_dependencies({"math-verify": "0.8.0"}, lookup)
    with pytest.raises(ValueError, match="antlr"):
        check_dependencies({"antlr4-python3-runtime": "4.13.2"}, lookup)


# --- C3: dataset loader -----------------------------------------------------------------


def _jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _math500_rows(n=3):
    return [
        {
            "unique_id": f"synthetic/{k}.json",
            "problem": f"Synthetic heldout {k}?",
            "answer": str(k),
            "subject": "synthetic",
        }
        for k in range(n)
    ]


def test_math500_shape_loads_with_stable_ids_and_hashes(tmp_path) -> None:
    path = _jsonl(tmp_path / "test.jsonl", _math500_rows())
    items, report = datasets.load(path, shape="math500", split="heldout", expected_count=3)
    assert [i.id for i in items] == ["synthetic/0.json", "synthetic/1.json", "synthetic/2.json"]
    assert all(i.split == "heldout" for i in items)
    assert report.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report.ids_sha256 == datasets.ids_sha256(items) == report.source_ids_sha256
    assert report.id_source == "unique_id"
    again, _ = datasets.load(
        path,
        shape="math500",
        split="heldout",
        expected_source_sha256=report.source_sha256,
        expected_ids_sha256=report.ids_sha256,
    )
    assert again == items


def test_hash_and_rows_come_from_one_byte_snapshot(tmp_path, monkeypatch) -> None:
    path = _jsonl(tmp_path / "test.jsonl", _math500_rows())
    reads: list = []
    original = Path.read_bytes

    def counting(self):
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("second read"))
    datasets.load(path, shape="math500", split="heldout")
    assert reads == [path]


@pytest.mark.parametrize(
    ("rows", "kwargs", "message"),
    [
        (_math500_rows(3), {"expected_count": 500}, "expected 500"),
        (_math500_rows(3), {"expected_source_sha256": "0" * 64}, "SHA-256 differs"),
        (_math500_rows(3), {"expected_ids_sha256": "0" * 64}, "ID set differs"),
        ([*_math500_rows(2), _math500_rows(1)[0]], {}, "duplicate item IDs"),
        ([{"unique_id": "a", "problem": "", "answer": "1"}], {}, "problem"),
        ([{"unique_id": "a", "problem": "p", "answer": 1}], {}, "answer"),
        ([{"problem": "p", "answer": "1"}], {}, "unique_id"),
        ([], {}, "contains no items"),
    ],
)
def test_math500_shape_refuses_bad_files(tmp_path, rows, kwargs, message) -> None:
    path = _jsonl(tmp_path / "t.jsonl", rows)
    with pytest.raises(datasets.DatasetLoadError, match=message):
        datasets.load(path, shape="math500", split="heldout", **kwargs)


def test_split_must_match_the_shape(tmp_path) -> None:
    path = _jsonl(tmp_path / "t.jsonl", _math500_rows())
    with pytest.raises(datasets.DatasetLoadError, match="heldout"):
        datasets.load(path, shape="math500", split="development")


def test_math_train_shape_extracts_answers_and_content_ids(tmp_path) -> None:
    rows = [
        {"problem": f"Synthetic train {k}?", "solution": f"so \\boxed{{{k}}}"} for k in range(4)
    ]
    path = _jsonl(tmp_path / "train.jsonl", rows)
    items, report = datasets.load(path, shape="math_train", split="development", limit=2)
    assert [i.answer for i in items] == ["0", "1"] and report.n_items == 2
    assert report.n_source_rows == 4 and report.limit == 2
    assert report.source_ids_sha256 != report.ids_sha256  # whole file vs selection
    assert report.id_source == "normalized_problem_sha256"
    expected = "math-train-" + hashlib.sha256(b"synthetic train 0?").hexdigest()[:16]
    assert items[0].id == expected
    reordered = _jsonl(tmp_path / "rev.jsonl", list(reversed(rows)))
    all_items, _ = datasets.load(reordered, shape="math_train", split="development")
    assert expected in {i.id for i in all_items}  # IDs do not depend on file order
    spaced = [{"problem": "Synthetic  train\r\n0?", "solution": "\\boxed{0}"}]
    (same,), _ = datasets.load(
        _jsonl(tmp_path / "crlf.jsonl", spaced), shape="math_train", split="development"
    )
    assert same.id == expected  # whitespace differences do not change the ID
    bad = _jsonl(tmp_path / "bad.jsonl", [{"problem": "p", "solution": "no box"}])
    with pytest.raises(datasets.DatasetLoadError, match="no complete"):
        datasets.load(bad, shape="math_train", split="development")


def test_math_train_keeps_source_ids_and_refuses_mixed_ids(tmp_path) -> None:
    rows = [{"unique_id": "train/a/1.json", "problem": "Synthetic?", "solution": "\\boxed{1}"}]
    path = _jsonl(tmp_path / "a.jsonl", rows)
    (item,), report = datasets.load(path, shape="math_train", split="development")
    assert item.id == "train/a/1.json" and report.id_source == "unique_id"
    mixed = _jsonl(tmp_path / "m.jsonl", [*rows, {"problem": "Other?", "solution": "\\boxed{2}"}])
    with pytest.raises(datasets.DatasetLoadError, match="mixed"):
        datasets.load(mixed, shape="math_train", split="development")


def test_the_whole_file_is_validated_before_the_limit(tmp_path) -> None:
    rows = [
        {"problem": "Synthetic A?", "solution": "\\boxed{1}"},
        {"problem": "Synthetic B?", "solution": "\\boxed{2}"},
        {"problem": "synthetic   a?", "solution": "\\boxed{1}"},  # same problem as row 1
    ]
    path = _jsonl(tmp_path / "dup.jsonl", rows)
    with pytest.raises(datasets.DatasetLoadError, match=r"duplicate item IDs.*normalisation"):
        datasets.load(path, shape="math_train", split="development", limit=1)
    ok = _jsonl(tmp_path / "ok.jsonl", rows[:2])
    with pytest.raises(datasets.DatasetLoadError, match="expected 1 items, found 2"):
        datasets.load(ok, shape="math_train", split="development", limit=1, expected_count=1)
    items, report = datasets.load(ok, shape="math_train", split="development", limit=1)
    with pytest.raises(datasets.DatasetLoadError, match="already limited"):
        datasets.select(items, report, 1)


def test_jsonl_is_split_on_newlines_only(tmp_path) -> None:
    path = tmp_path / "u.jsonl"
    row = {"unique_id": "s/1", "problem": "Synthetic\u2028line?", "answer": "1"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (item,), _ = datasets.load(path, shape="math500", split="heldout")
    assert item.prompt == "Synthetic\u2028line?"


def test_leakage_ignores_answers_and_formatting(tmp_path) -> None:
    held, _ = datasets.load(
        _jsonl(tmp_path / "h.jsonl", _math500_rows()), shape="math500", split="heldout"
    )
    copied = Item(
        id="math-train-x", split="development", prompt="  synthetic HELDOUT 0? ", answer="0.0"
    )
    with pytest.raises(datasets.DatasetLoadError, match="leakage: 1 problems"):
        datasets.check_disjoint([copied], held)
    same_id = Item(id="synthetic/0.json", split="development", prompt="Other?", answer="1")
    with pytest.raises(datasets.DatasetLoadError, match="shared IDs"):
        datasets.check_disjoint([same_id], held)
    other = Item(id="d", split="development", prompt="Unrelated synthetic?", answer="0")
    result = datasets.check_disjoint([other], held)
    assert result["ok"] is True and "cannot prove disjointness" in result["limit"]
    with pytest.raises(datasets.DatasetLoadError, match="leakage"):
        datasets.combine(held, [copied])


def test_written_items_load_back_through_the_harness(tmp_path) -> None:
    items = _items(3)
    path = tmp_path / "items.jsonl"
    datasets.write_items(items, path)
    assert load_items(path) == items
    with pytest.raises(FileExistsError):
        datasets.write_items(items, path)  # never overwrites


# --- isolation ------------------------------------------------------------------------------

_NEVER = {
    "tantular_cyber",
    "socket",
    "http",
    "urllib",
    "httpx",
    "requests",
    "subprocess",
    "pickle",
    "ssl",
    "ftplib",
    "smtplib",
    "huggingface_hub",
}
_LAZY_ONLY = {"vllm", "torch", "math_verify", "transformers"}


def test_runtime_is_isolated_and_imports_heavy_libraries_lazily() -> None:
    files = [*sorted((RUNTIME / "wsp_runtime").glob("*.py")), RUNTIME / "run.py"]
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for n in names:
                    assert n.split(".")[0] not in _NEVER, f"{path.name} imports {n}"
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module":
                # importlib.import_module must stay inside functions and name only vLLM modules.
                (arg,) = node.args
                assert isinstance(arg, ast.Constant) and arg.value.startswith("vllm.")
        for node in tree.body:  # module level: no heavy imports, so tests run without them
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for n in names:
                    assert n.split(".")[0] not in _LAZY_ONLY, f"{path.name} imports {n} eagerly"


def test_harness_never_imports_the_runtime() -> None:
    for path in [*sorted((WSP / "wsp_eval").glob("*.py")), WSP / "wsp.py"]:
        assert "wsp_runtime" not in path.read_text(), path.name

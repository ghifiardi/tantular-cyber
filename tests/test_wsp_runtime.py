"""Runtime adapters C1–C3 (tools/wsp_runtime), tested with a stub vLLM backend, synthetic
grading strings and synthetic dataset files only. No model, GPU, network, vLLM, torch or
math-verify is used; nothing here is a measurement or MATH content."""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import json
import sys
import weakref
from pathlib import Path

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
from wsp_eval.protocol import parse_protocol  # noqa: E402
from wsp_eval.runner import verify_run  # noqa: E402
from wsp_runtime import datasets  # noqa: E402
from wsp_runtime.grader import (  # noqa: E402
    GRADER_ID,
    GRADER_REVISION,
    MathQAGrader,
    last_boxed,
    strip_reasoning,
)
from wsp_runtime.provenance import git_blob_sha1, hash_tree  # noqa: E402
from wsp_runtime.vllm_adapter import (  # noqa: E402
    PROTOCOL_USER_MESSAGE,
    BatchMismatch,
    ProvenanceError,
    VllmEngine,
    user_message,
)

WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
TEMPLATE = b"{# synthetic chat template for tests #}"
VERSION = "0.0.0-stub"


# --- stub vLLM ------------------------------------------------------------------------------


class Completion:
    def __init__(self, text, n_tokens, finish_reason):
        self.text, self.token_ids, self.finish_reason = text, list(range(n_tokens)), finish_reason


class RequestOutput:
    def __init__(self, prompt, outputs):
        self.prompt, self.outputs = prompt, outputs


class Tokenizer:
    eos_token_id = 248044

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        (m,) = messages
        return f"<user>{m['content']}</user><assistant>{'<think>' if enable_thinking else ''}"


class Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeLLM:
    def __init__(self, backend, **kwargs):
        self.backend, self.kwargs, self.shut = backend, kwargs, False

    def get_tokenizer(self):
        return Tokenizer()

    def get_default_sampling_params(self):
        return Params(temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0)

    def generate(self, prompts, params, use_tqdm):
        assert use_tqdm is False and len(prompts) == len(params)
        self.backend.calls.append([(p, vars(s)) for p, s in zip(prompts, params, strict=True)])
        outs = [self.backend.answer(p, s) for p, s in zip(prompts, params, strict=True)]
        return self.backend.post(outs)


class FakeBackend:
    engine = "vllm"
    version = VERSION

    def __init__(self, *, answer=None, post=None, memory=0):
        self.calls: list = []
        self.llms: list = []
        self.answer = answer or self._correct
        self.post = post or (lambda outs: outs)
        self.memory = memory

    @staticmethod
    def _correct(prompt, params):
        # Synthetic arithmetic only: "compute A + B" -> boxed sum, one output per sample.
        expr = prompt.split("compute ", 1)[1].split(".", 1)[0]
        a, b = (int(x) for x in expr.split(" + "))
        text = f"<think>working</think>The answer is \\boxed{{{a + b}}}."
        return RequestOutput(prompt, [Completion(text, 12, "stop") for _ in range(params.n)])

    def make_llm(self, **kwargs):
        llm = FakeLLM(self, **kwargs)
        self.llms.append(weakref.ref(llm))
        return llm

    def make_sampling_params(self, **kw):
        return Params(**kw)

    def default_sampling(self, llm):
        return vars(llm.get_default_sampling_params())

    def shutdown(self, llm):
        llm.shut = True
        return ["stub.shutdown"]

    def memory_allocated(self):
        return self.memory

    def environment(self):
        return {"vllm": VERSION, "torch": None, "cuda": None, "gpu": None}


class ExactGrader:
    """Stub math-verify: exact string equality after whitespace removal."""

    def equivalent(self, gold, pred):
        return gold.replace(" ", "") == pred.replace(" ", "")


# --- fixtures -------------------------------------------------------------------------------


def _checkpoint(root: Path, name: str, weights: bytes) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / WEIGHTS).write_bytes(weights)
    (d / "config.json").write_text('{"model_type": "synthetic"}')
    (d / "tokenizer.json").write_text('{"synthetic": true}')
    (d / "chat_template.jinja").write_bytes(TEMPLATE)
    (d / "README.md").write_text(f"synthetic {name}")
    return d


def _protocol_raw(base_sha: str, edited_sha: str) -> dict:
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
        "dtype": "bfloat16",
        "max_model_len": 128,
        "llm_seed": 0,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
    }

    def arm(arm_id, role, sha):
        return {
            "id": arm_id,
            "role": role,
            "checkpoint": {
                "repo": "synthetic/stub",
                "revision": "stub",
                "weights_file": WEIGHTS,
                "weights_sha256_lfs": sha,
            },
            "inference": dict(inference),
            "engine": dict(engine),
        }

    raw = json.loads((WSP / "fixtures" / "synthetic_protocol.json").read_text())
    raw.update(
        protocol_version="runtime-stub-0.1",
        synthetic=False,
        sampling_mode="n_per_request",
        samples_per_item=4,
        seeds=[0],
        grader={"id": GRADER_ID, "revision": GRADER_REVISION},
        execution={"order": "per_arm_sequential", "shards": 2},
        analysis={**raw["analysis"], "reference_arm": "base", "candidate_arm": "edited"},
        arms=[
            arm("base", "reference", base_sha),
            arm("edited", "candidate", edited_sha),
            arm("base_reload", "no_edit_reload", base_sha),
        ],
    )
    return raw


@pytest.fixture
def ckpts(tmp_path):
    base = _checkpoint(tmp_path / "ckpt", "base", b"synthetic base weights")
    edited = _checkpoint(tmp_path / "ckpt", "edited", b"synthetic edited weights")
    return base, edited


@pytest.fixture
def protocol(ckpts):
    base, edited = ckpts
    return parse_protocol(_protocol_raw(hash_tree(base)[WEIGHTS], hash_tree(edited)[WEIGHTS]))


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
    a = {x.id: x for x in protocol.arms}[arm]
    d = ckpts[1] if arm == "edited" else ckpts[0]
    return VllmEngine(a, d, backend or FakeBackend(), **kw)


def _req(protocol, arm="base", item="dev-000", prompt="Synthetic: compute 0 + 1.", n=4, seed=0):
    a = {x.id: x for x in protocol.arms}[arm]
    return GenerationRequest(a.id, item, prompt, dict(a.inference), seed, n)


# --- C1: provenance and load identity ---------------------------------------------------


def test_load_reports_identity_files_and_effective_settings(protocol, ckpts) -> None:
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
    assert d["sampling"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 64,
    }
    assert d["engine_kwargs"] == {
        "dtype": "bfloat16",
        "max_model_len": 128,
        "seed": 0,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
    }
    assert d["engine_default_sampling"]["temperature"] == 1.0  # engine defaults recorded
    assert d["eos_token_id"] == 248044
    assert len(d["runtime_code_sha256"]) == 64
    llm_kwargs = backend.llms[0]().kwargs
    assert llm_kwargs["model"] == str(ckpts[0]) and llm_kwargs["tokenizer"] == str(ckpts[0])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: (d / WEIGHTS).write_bytes(b"tampered"), "differs from the registered"),
        (lambda d: (d / WEIGHTS).unlink(), "missing"),
        (lambda d: (d / "chat_template.jinja").write_bytes(b"other"), "git blob"),
        (lambda d: (d / "chat_template.jinja").unlink(), "git blob"),
    ],
)
def test_provenance_mismatch_refuses_before_loading(protocol, ckpts, change, message) -> None:
    change(ckpts[0])
    backend = FakeBackend()
    with pytest.raises(ProvenanceError, match=message):
        _engine(protocol, ckpts, backend)
    assert backend.llms == []  # nothing was loaded


def test_symlinked_checkpoint_files_are_refused(protocol, ckpts, tmp_path) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    (ckpts[0] / "extra.json").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        _engine(protocol, ckpts)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", "9.9.9", "engine version"),
        ("version", "REPLACE (open)", "must be pinned"),
        ("name", "other", "protocol engine"),
        ("gpu_memory_utilization", "REPLACE", "gpu_memory_utilization"),
        ("tensor_parallel_size", 1.0, "tensor_parallel_size"),
    ],
)
def test_engine_settings_must_be_pinned_and_match(ckpts, field, value, message) -> None:
    base, edited = ckpts
    raw = _protocol_raw(hash_tree(base)[WEIGHTS], hash_tree(edited)[WEIGHTS])
    for a in raw["arms"]:
        a["engine"][field] = value
    proto = parse_protocol(raw)
    with pytest.raises((ValueError, ProvenanceError), match=message):
        _engine(proto, ckpts)


def test_unknown_inference_keys_and_prompt_notation_are_refused(ckpts) -> None:
    base, edited = ckpts
    raw = _protocol_raw(hash_tree(base)[WEIGHTS], hash_tree(edited)[WEIGHTS])
    for a in raw["arms"]:
        a["inference"]["repetition_penalty"] = 1.1
    with pytest.raises(ValueError, match="not understood"):
        _engine(parse_protocol(raw), ckpts)
    with pytest.raises(ValueError, match="registered Experiment 8 prompt"):
        user_message("<problem>\nPlease reason", "p")
    assert user_message(PROTOCOL_USER_MESSAGE, "Q?") == (
        "Q?\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


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


def test_close_shuts_down_releases_and_reports(protocol, ckpts) -> None:
    backend = FakeBackend()
    reports: list = []
    e = _engine(protocol, ckpts, backend, reports=reports)
    e.close()
    e.close()  # idempotent
    gc.collect()
    assert backend.llms[0]() is None  # the engine dropped its LLM object
    (report,) = reports
    assert report["shutdown_steps"] == ["stub.shutdown"]
    assert report["files_unchanged"] is True and report["memory_allocated_after_close"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        e.generate_batch([_req(protocol)])


def test_close_fails_if_files_changed_or_memory_remains(protocol, ckpts) -> None:
    e = _engine(protocol, ckpts)
    (ckpts[0] / "config.json").write_text('{"changed": true}')
    with pytest.raises(ProvenanceError, match="changed during the run"):
        e.close()
    (ckpts[0] / "config.json").write_text('{"model_type": "synthetic"}')
    e2 = _engine(protocol, ckpts, FakeBackend(memory=10), max_memory_after_close=0)
    with pytest.raises(RuntimeError, match="still allocated"):
        e2.close()


# --- end to end through the harness runner ----------------------------------------------


def _write_items(path: Path, items) -> Path:
    datasets.write_items(items, path)
    return path


def _cli(tmp_path, protocol_raw, ckpts, *, backend=None, extra=()):
    ppath = tmp_path / "protocol.json"
    ppath.write_text(json.dumps(protocol_raw))
    ipath = _write_items(tmp_path / "items.jsonl", _items())
    base, edited = ckpts
    backend = backend or FakeBackend()
    argv = [
        "--protocol",
        str(ppath),
        "--items",
        str(ipath),
        "--checkpoint",
        f"base={base}",
        "--checkpoint",
        f"edited={edited}",
        "--checkpoint",
        f"base_reload={base}",
        "--split",
        "development",
        "--operator",
        "op1",
        "--purpose",
        "stub",
        "--output",
        str(tmp_path / "run"),
        *extra,
    ]
    summary = runtime_cli.run(
        argv, backend_factory=lambda: backend, verify_backend_factory=ExactGrader
    )
    return summary, backend


def test_end_to_end_run_with_stub_engines(tmp_path, protocol, ckpts) -> None:
    summary, backend = _cli(tmp_path, protocol.raw, ckpts, extra=["--keep-outputs"])
    assert summary["run_status"] == "complete"
    manifest, records = verify_run(tmp_path / "run", protocol)
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


def test_memory_left_after_close_stops_before_the_next_arm(tmp_path, protocol, ckpts) -> None:
    backend = FakeBackend(memory=1024)
    with pytest.raises(runtime_cli._Aborted) as exc:
        _cli(
            tmp_path,
            protocol.raw,
            ckpts,
            backend=backend,
            extra=["--max-gpu-memory-after-close", "0"],
        )
    assert "still allocated" in exc.value.summary["error"]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["run_status"] == "aborted"
    assert [x["status"] for x in manifest["execution"]["loads"]] == ["close_failed"]
    assert len(backend.calls) == 2  # only the first arm ran


def test_cli_refuses_wrong_checkpoint_mapping_and_protocols(tmp_path, protocol, ckpts) -> None:
    with pytest.raises(ValueError, match="exactly the protocol arms"):
        runtime_cli._checkpoints(["base=/a", "edited=/b"], ["base", "edited", "base_reload"])
    with pytest.raises(ValueError, match="twice"):
        runtime_cli._checkpoints(["base=/a", "base=/b"], ["base"])
    with pytest.raises(ValueError, match="ARM=DIR"):
        runtime_cli._checkpoints(["base"], ["base"])
    interleaved = copy.deepcopy(protocol.raw)
    del interleaved["execution"]
    with pytest.raises(ValueError, match="per_arm_sequential"):
        _cli(tmp_path, interleaved, ckpts)


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
        ("\\boxed{unbalanced", None),
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


def test_grader_scores_boxed_answers_and_unboxed_as_incorrect() -> None:
    g = MathQAGrader(ExactGrader())
    item = Item(id="x", split="development", prompt="p", answer="12")
    assert g.grade(item, "<think>\\boxed{11}</think>\\boxed{12}").correct is True
    assert g.grade(item, "<think>\\boxed{12}</think>so 11").correct is False  # box was reasoning
    no_box = g.grade(item, "the answer is 12")
    assert no_box.correct is False and no_box.detail == "no boxed answer"  # graded, not ungraded
    assert (g.id, g.revision) == (GRADER_ID, GRADER_REVISION)


def test_grader_backend_errors_propagate() -> None:
    class Timeout:
        def equivalent(self, gold, pred):
            raise TimeoutError("math-verify timed out")

    item = Item(id="x", split="development", prompt="p", answer="1")
    with pytest.raises(TimeoutError):
        MathQAGrader(Timeout()).grade(item, "\\boxed{1}")


def test_math_verify_backend_passes_wrapped_text_and_timeouts() -> None:
    from wsp_runtime.grader import MathVerifyBackend

    calls = []

    class Module:
        @staticmethod
        def parse(text, parsing_timeout):
            calls.append(("parse", text, parsing_timeout))
            return text

        @staticmethod
        def verify(g, p, timeout_seconds):
            calls.append(("verify", g, p, timeout_seconds))
            return g == p

    assert MathVerifyBackend(Module).equivalent("1", "1") is True
    assert calls == [("parse", "$1$", 5), ("parse", "$1$", 5), ("verify", "$1$", "$1$", 5)]


# --- C3: dataset loader -----------------------------------------------------------------


def _jsonl(path: Path, rows) -> Path:
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
    assert report.ids_sha256 == datasets.ids_sha256(items)
    again, _ = datasets.load(
        path,
        shape="math500",
        split="heldout",
        expected_source_sha256=report.source_sha256,
        expected_ids_sha256=report.ids_sha256,
    )
    assert again == items


@pytest.mark.parametrize(
    ("rows", "kwargs", "message"),
    [
        (_math500_rows(3), {"expected_count": 500}, "expected 500"),
        (_math500_rows(3), {"expected_source_sha256": "0" * 64}, "SHA-256 differs"),
        (_math500_rows(3), {"expected_ids_sha256": "0" * 64}, "ID set differs"),
        ([*_math500_rows(2), _math500_rows(1)[0]], {}, "dataset check failed"),
        ([{"unique_id": "a", "problem": "", "answer": "1"}], {}, "problem"),
        ([{"unique_id": "a", "problem": "p", "answer": 1}], {}, "answer"),
        ([{"problem": "p", "answer": "1"}], {}, "unique_id"),
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
    expected = "math-train-" + hashlib.sha256(b"Synthetic train 0?").hexdigest()[:16]
    assert items[0].id == expected
    reordered = _jsonl(tmp_path / "rev.jsonl", list(reversed(rows)))
    all_items, _ = datasets.load(reordered, shape="math_train", split="development")
    assert expected in {i.id for i in all_items}  # IDs do not depend on file order
    bad = _jsonl(tmp_path / "bad.jsonl", [{"problem": "p", "solution": "no box"}])
    with pytest.raises(datasets.DatasetLoadError, match="no \\\\boxed"):
        datasets.load(bad, shape="math_train", split="development")


def test_combined_development_and_heldout_items_are_leak_checked(tmp_path) -> None:
    held, _ = datasets.load(
        _jsonl(tmp_path / "h.jsonl", _math500_rows()), shape="math500", split="heldout"
    )
    leaked = Item(
        id="math-train-x", split="development", prompt=held[0].prompt, answer=held[0].answer
    )
    with pytest.raises(datasets.DatasetLoadError, match="combined"):
        datasets.combine(held, [leaked])


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

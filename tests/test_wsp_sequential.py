"""Sequential per-arm loading and shard batching (protocol ``execution``), tested with stub
engines only: no model, no GPU, no network. Synthetic fixtures; nothing here is a measurement."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WSP = ROOT / "tools" / "wsp"
sys.path.insert(0, str(WSP))

from wsp_eval.analysis import analyze  # noqa: E402
from wsp_eval.dataset import load_items  # noqa: E402
from wsp_eval.drift import drift_report  # noqa: E402
from wsp_eval.generation import (  # noqa: E402
    BatchItem,
    Generation,
    GenerationError,
    SyntheticGenerator,
)
from wsp_eval.grading import FinalAnswerGrader  # noqa: E402
from wsp_eval.protocol import ProtocolError, parse_protocol  # noqa: E402
from wsp_eval.runner import RunError, run_paired, verify_run  # noqa: E402
from wsp_eval.synthetic import synthetic_generators  # noqa: E402

FIX = WSP / "fixtures"
SEQ = {"execution": {"order": "per_arm_sequential", "shards": 2}}


@pytest.fixture
def raw() -> dict:
    return json.loads((FIX / "synthetic_protocol.json").read_text())


@pytest.fixture
def items():
    return load_items(FIX / "synthetic_items.jsonl")


class StubEngine(SyntheticGenerator):
    """A fresh stub 'load' with its own load_id; logs load, batch and close events."""

    def __init__(self, template: SyntheticGenerator, load_id: str, events: list):
        super().__init__(
            template.name,
            template.table,
            load_id=load_id,
            artifacts_sha256=dict(template.artifacts_sha256),
        )
        self.events = events

    def generate_batch(self, requests):
        self.events.append(("batch", self.load_id, [r.item_id for r in requests]))
        return super().generate_batch(requests)

    def close(self) -> None:
        self.events.append(("close", self.load_id))
        super().close()


def _loaders(items, seeds, events, *, overrides=None, nondeterministic_reload=False):
    templates = synthetic_generators(items, seeds, nondeterministic_reload=nondeterministic_reload)
    overrides = overrides or {}
    counter = iter(range(1, 100))

    def make(arm_id):
        def load():
            events.append(("load", arm_id))
            if arm_id in overrides:
                return overrides[arm_id]()
            return StubEngine(templates[arm_id], f"{arm_id}-load-{next(counter)}", events)

        return load

    return {arm_id: make(arm_id) for arm_id in templates}


def _run(tmp_path, raw, items, *, loaders=None, events=None, name="run", **kw):
    proto = parse_protocol(raw)
    events = [] if events is None else events
    res = run_paired(
        proto,
        items,
        None,
        FinalAnswerGrader(),
        tmp_path / name,
        loaders=loaders or _loaders(items, proto.seeds, events),
        purpose="test",
        operator="op1",
        clock=iter(range(100_000)).__next__,
        run_id="wsp-seq-test",
        **kw,
    )
    return proto, res, events


def _key(r):
    return (r["arm"], r["item_id"], r["sample"])


# --- protocol -------------------------------------------------------------------------------


def test_protocols_without_an_execution_block_are_unchanged(raw) -> None:
    proto = parse_protocol(raw)
    assert (proto.execution_order, proto.shards) == ("interleaved", 1)
    assert parse_protocol(copy.deepcopy(raw)).sha256 == proto.sha256


def test_execution_block_is_hashed(raw) -> None:
    base = parse_protocol(raw).sha256
    seq = parse_protocol({**raw, **SEQ})
    assert (seq.execution_order, seq.shards) == ("per_arm_sequential", 2)
    assert seq.sha256 != base
    other = parse_protocol({**raw, "execution": {"order": "per_arm_sequential", "shards": 3}})
    assert other.sha256 != seq.sha256


@pytest.mark.parametrize(
    "block",
    [
        {"order": "sequential"},
        {"order": "per_arm_sequential", "shards": 0},
        {"order": "per_arm_sequential", "shards": True},
        {"order": "per_arm_sequential", "shards": 2.0},
        {"order": "interleaved", "shards": 2},
        {"order": "per_arm_sequential", "batch": 50},
        "per_arm_sequential",
    ],
)
def test_malformed_execution_blocks_are_refused(raw, block) -> None:
    with pytest.raises(ProtocolError, match="execution"):
        parse_protocol({**raw, "execution": block})


# --- sequential loading -------------------------------------------------------------------


def test_arms_are_loaded_one_at_a_time_in_protocol_order(tmp_path, raw, items) -> None:
    proto, res, events = _run(tmp_path, {**raw, **SEQ}, items)
    order = [e for e in events if e[0] in ("load", "close")]
    # load A, close A, load B, close B, load C, close C: never two engines at once.
    assert [e[0] for e in order] == ["load", "close"] * 3
    loaded = [e[1] for e in order if e[0] == "load"]
    assert loaded == [a.id for a in proto.arms]
    loads = res.manifest["execution"]["loads"]
    assert [(x["arm"], x["position"], x["status"], x["closed"]) for x in loads] == [
        (a.id, i, "complete", True) for i, a in enumerate(proto.arms)
    ]
    ids = [x["load_id"] for x in loads]
    assert len(set(ids)) == 3  # distinct load identities, recorded per arm
    assert [a["generator"]["load_id"] for a in res.manifest["arms"]] == ids
    assert res.manifest["run_status"] == "complete"


def test_sequential_records_match_interleaved_for_deterministic_engines(
    tmp_path, raw, items
) -> None:
    # Same prompts, seeds and per-request records: only order, latency and batch fields differ.
    _, seq, _ = _run(tmp_path, {**raw, **SEQ}, items, name="seq")
    proto = parse_protocol(raw)
    inter = run_paired(
        proto,
        items,
        synthetic_generators(items, proto.seeds),
        FinalAnswerGrader(),
        tmp_path / "inter",
        purpose="test",
        operator="op1",
        clock=iter(range(100_000)).__next__,
        run_id="wsp-seq-test",
    )
    drop = {"latency_s", "shard", "batch_size"}
    a = {_key(r): {k: v for k, v in r.items() if k not in drop} for r in seq.records}
    b = {_key(r): {k: v for k, v in r.items() if k not in drop} for r in inter.records}
    assert a == b
    assert len(seq.records) == seq.manifest["counts"]["expected_records"]


def test_shards_follow_dataset_file_order(tmp_path, raw, items) -> None:
    # Like the authors' rows[i::k]: shard k holds the selected items [k::shards] in file order.
    proto, res, events = _run(tmp_path, {**raw, **SEQ}, items)
    dev = [i.id for i in items if i.split == "development"]
    expected = [dev[0::2], dev[1::2]]
    for arm in proto.arms:
        batches = [e[2] for e in events if e[0] == "batch" and e[1].startswith(arm.id)]
        # per_sample_seed: one request per (item, seed), item-major within the shard
        assert [sorted(set(b)) for b in batches] == [sorted(x) for x in expected]
        assert [len(b) for b in batches] == [len(x) * len(proto.seeds) for x in expected]
    by_item = {r["item_id"]: r["shard"] for r in res.records}
    assert all(by_item[i] == 0 for i in dev[0::2]) and all(by_item[i] == 1 for i in dev[1::2])
    assert {r["batch_size"] for r in res.records} == {len(x) * len(proto.seeds) for x in expected}


def test_n_per_request_batches_one_request_per_item_with_n_samples(tmp_path, raw, items) -> None:
    raw2 = {**raw, **SEQ, "sampling_mode": "n_per_request", "seeds": [11]}
    proto, res, events = _run(tmp_path, raw2, items)
    dev = [i.id for i in items if i.split == "development"]
    batches = [e[2] for e in events if e[0] == "batch"]
    assert [len(b) for b in batches] == [len(dev[0::2]), len(dev[1::2])] * 3
    assert all(r["request_n"] == proto.samples_per_item for r in res.records)
    assert len(res.records) == len(dev) * proto.samples_per_item * 3


def test_sequential_runs_verify_analyze_and_drift(tmp_path, raw, items) -> None:
    proto, res, _ = _run(tmp_path, {**raw, **SEQ}, items, keep_outputs=True)
    manifest, records = verify_run(res.out_dir, proto)
    assert manifest["execution"]["order"] == "per_arm_sequential"
    analyze(records, proto.analysis, synthetic=True)
    (d,) = drift_report(proto, manifest, records)
    checks = {c["check"]: c for c in d["identity"]["checks"]}
    assert checks["separate_load"]["status"] == "pass"
    assert (res.out_dir / "outputs.jsonl").exists()


# --- explicit failures --------------------------------------------------------------------


class _Scripted(StubEngine):
    def __init__(self, template, load_id, events, behaviour):
        super().__init__(template, load_id, events)
        self.behaviour = behaviour

    def generate_batch(self, requests):
        return self.behaviour(self, requests)


def _scripted(items, seeds, events, arm, behaviour):
    templates = synthetic_generators(items, seeds)
    return {arm: lambda: _Scripted(templates[arm], f"{arm}-scripted", events, behaviour)}


def test_a_failed_request_inside_a_batch_errors_only_its_own_rows(tmp_path, raw, items) -> None:
    proto = parse_protocol({**raw, **SEQ})
    first = next(i.id for i in items if i.split == "development")

    def one_fails(self, requests):
        out = SyntheticGenerator.generate_batch(self, requests)
        return [
            BatchItem(r, GenerationError("engine refused"))
            if r.item_id == first and r.seed == 11
            else o
            for r, o in zip(requests, out, strict=True)
        ]

    events: list = []
    loaders = _loaders(
        items,
        proto.seeds,
        events,
        overrides=_scripted(items, proto.seeds, events, "candidate", one_fails),
    )
    _, res, _ = _run(tmp_path, {**raw, **SEQ}, items, loaders=loaders, events=events)
    bad = [r for r in res.records if r["arm"] == "candidate" and r["status"] == "error"]
    assert {(r["item_id"], r["seed"], r["error_class"]) for r in bad} >= {
        (first, 11, "GenerationError")
    }
    assert res.manifest["run_status"] == "complete"
    assert len(res.records) == res.manifest["counts"]["expected_records"]


@pytest.mark.parametrize(
    ("behaviour", "error_class"),
    [
        (
            lambda self, reqs: (_ for _ in ()).throw(GenerationError("batch timeout")),
            "GenerationError",
        ),
        (
            lambda self, reqs: SyntheticGenerator.generate_batch(self, reqs)[:-1],
            "BatchSizeMismatch",
        ),
        (lambda self, reqs: None, "BatchSizeMismatch"),
    ],
)
def test_a_failed_or_malformed_batch_errors_every_request_in_it(
    tmp_path, raw, items, behaviour, error_class
) -> None:
    proto = parse_protocol({**raw, **SEQ})
    events: list = []
    loaders = _loaders(
        items,
        proto.seeds,
        events,
        overrides=_scripted(items, proto.seeds, events, "reload", behaviour),
    )
    _, res, _ = _run(tmp_path, {**raw, **SEQ}, items, loaders=loaders, events=events)
    reload_rows = [r for r in res.records if r["arm"] == "reload"]
    assert reload_rows and all(r["status"] == "error" for r in reload_rows)
    assert {r["error_class"] for r in reload_rows} == {error_class}
    assert res.manifest["run_status"] == "complete"  # recorded, not dropped or aborted
    assert len(res.records) == res.manifest["counts"]["expected_records"]


def test_abnormal_finishes_inside_a_batch_are_still_per_sample_errors(tmp_path, raw, items) -> None:
    proto = parse_protocol({**raw, **SEQ})

    def abnormal(self, requests):
        return [
            BatchItem(r, Generation("x", finish_reason="abort", rendered_prompt_sha256="p"))
            for r in requests
        ]

    events: list = []
    loaders = _loaders(
        items,
        proto.seeds,
        events,
        overrides=_scripted(items, proto.seeds, events, "candidate", abnormal),
    )
    _, res, _ = _run(tmp_path, {**raw, **SEQ}, items, loaders=loaders, events=events)
    rows = [r for r in res.records if r["arm"] == "candidate"]
    assert {r["error_class"] for r in rows} == {"AbnormalFinish"}
    assert {r["rendered_prompt_sha256"] for r in rows} == {"p"}  # kept for the identity check


# --- aborted runs -------------------------------------------------------------------------


def _assert_aborted(tmp_path, raw, items, loaders, events, exc_type, message):
    proto = parse_protocol({**raw, **SEQ})
    with pytest.raises(exc_type, match=message):
        _run(tmp_path, {**raw, **SEQ}, items, loaders=loaders, events=events)
    out = tmp_path / "run"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run_status"] == "aborted" and manifest["abort_reason"] == exc_type.__name__
    with pytest.raises(RunError, match="aborted"):
        verify_run(out, proto)
    return manifest, [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]


def test_a_load_failure_aborts_and_keeps_earlier_arms(tmp_path, raw, items) -> None:
    proto = parse_protocol({**raw, **SEQ})
    events: list = []

    def broken():
        raise OSError("CUDA out of memory while loading")

    loaders = _loaders(items, proto.seeds, events, overrides={"candidate": broken})
    manifest, records = _assert_aborted(tmp_path, raw, items, loaders, events, OSError, "memory")
    assert {r["arm"] for r in records} == {"reference"}  # what ran is kept, visibly partial
    statuses = [(x["arm"], x["status"]) for x in manifest["execution"]["loads"]]
    assert statuses == [("reference", "complete"), ("candidate", "load_failed")]
    assert "out of memory" in manifest["abort_message"]


@pytest.mark.parametrize("bad_id", [None, "", "reference-load-1"])
def test_a_missing_or_reused_load_id_aborts(tmp_path, raw, items, bad_id) -> None:
    # The reload must be a separate load: reusing the reference's load_id is refused.
    proto = parse_protocol({**raw, **SEQ})
    events: list = []
    templates = synthetic_generators(items, proto.seeds)
    loaders = _loaders(
        items,
        proto.seeds,
        events,
        overrides={"candidate": lambda: StubEngine(templates["candidate"], bad_id, events)},
    )
    manifest, _ = _assert_aborted(tmp_path, raw, items, loaders, events, RunError, "load_id")
    assert manifest["execution"]["loads"][1]["status"] == "failed"
    assert ("close", bad_id) in events  # released even though it was refused


def test_a_close_failure_aborts(tmp_path, raw, items) -> None:
    proto = parse_protocol({**raw, **SEQ})
    events: list = []
    templates = synthetic_generators(items, proto.seeds)

    class NoRelease(StubEngine):
        def close(self):
            raise RuntimeError("engine did not release the GPU")

    loaders = _loaders(
        items,
        proto.seeds,
        events,
        overrides={"reference": lambda: NoRelease(templates["reference"], "ref-x", events)},
    )
    manifest, _ = _assert_aborted(tmp_path, raw, items, loaders, events, RuntimeError, "GPU")
    assert manifest["execution"]["loads"][0]["status"] == "close_failed"
    assert [e[1] for e in events if e[0] == "load"] == ["reference"]  # nothing loaded after


def test_a_generator_without_batching_aborts(tmp_path, raw, items) -> None:
    proto = parse_protocol({**raw, **SEQ})
    events: list = []

    class Plain:
        def describe(self):
            return {"load_id": "plain-1", "synthetic": True}

        def generate(self, request):
            raise AssertionError("not used")

    loaders = _loaders(items, proto.seeds, events, overrides={"reference": Plain})
    _assert_aborted(tmp_path, raw, items, loaders, events, RunError, "generate_batch")


def test_synthetic_engines_cannot_serve_a_non_synthetic_protocol(tmp_path, raw, items) -> None:
    raw2 = {**raw, **SEQ, "synthetic": False}
    proto = parse_protocol(raw2)
    events: list = []
    with pytest.raises(RunError, match="synthetic generators"):
        _run(tmp_path, raw2, items, loaders=_loaders(items, proto.seeds, events), events=events)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["run_status"] == "aborted"


def test_an_interrupt_mid_arm_aborts_with_partial_records(tmp_path, raw, items) -> None:
    proto = parse_protocol({**raw, **SEQ})

    def interrupted(self, requests):
        raise KeyboardInterrupt

    events: list = []
    loaders = _loaders(
        items,
        proto.seeds,
        events,
        overrides=_scripted(items, proto.seeds, events, "candidate", interrupted),
    )
    manifest, records = _assert_aborted(
        tmp_path, raw, items, loaders, events, KeyboardInterrupt, None
    )
    assert [x["status"] for x in manifest["execution"]["loads"]] == ["complete", "failed"]
    assert {r["arm"] for r in records} == {"reference"}
    assert ("close", "candidate-scripted") in events


# --- wiring -------------------------------------------------------------------------------


def test_each_execution_order_takes_its_own_kind_of_input(tmp_path, raw, items) -> None:
    seq = parse_protocol({**raw, **SEQ})
    inter = parse_protocol(raw)
    gens = synthetic_generators(items, inter.seeds)
    common = {"purpose": "t", "operator": "op1"}
    with pytest.raises(RunError, match="takes loaders"):
        run_paired(seq, items, gens, FinalAnswerGrader(), tmp_path / "a", **common)
    with pytest.raises(RunError, match="takes loaded generators"):
        run_paired(inter, items, None, FinalAnswerGrader(), tmp_path / "b", loaders={}, **common)
    with pytest.raises(RunError, match="loaders must match protocol arms"):
        run_paired(
            seq,
            items,
            None,
            FinalAnswerGrader(),
            tmp_path / "c",
            loaders={"reference": lambda: None},
            **common,
        )
    assert not any((tmp_path / d).exists() for d in "abc")  # refused before any output


def test_heldout_gating_is_unchanged_for_sequential_runs(tmp_path, raw, items) -> None:
    with pytest.raises(RunError, match="frozen protocol"):
        _run(tmp_path, {**raw, **SEQ}, items, split="heldout", heldout_authorized_by="ev1")


def test_shards_use_file_order_not_id_order(tmp_path, raw, items) -> None:
    # Shard composition is part of the configuration: it follows the dataset file, so a file
    # listed in a different order gives different (recorded) shards.
    reordered = list(reversed(items))
    proto, _res, events = _run(tmp_path, {**raw, **SEQ}, reordered)
    dev = [i.id for i in reordered if i.split == "development"]
    first_arm = proto.arms[0].id
    batches = [e[2] for e in events if e[0] == "batch" and e[1].startswith(first_arm)]
    assert [list(dict.fromkeys(b)) for b in batches] == [dev[0::2], dev[1::2]]


# --- review fixes (focused review of 3d42e48) ----------------------------------------------


def _with(items, seeds, events, arm, behaviour):
    return _loaders(items, seeds, events, overrides=_scripted(items, seeds, events, arm, behaviour))


def _reversed(self, reqs):
    return list(reversed(SyntheticGenerator.generate_batch(self, reqs)))


def _rotated(self, reqs):
    out = SyntheticGenerator.generate_batch(self, reqs)
    return [*out[1:], out[0]]


def _bare(self, reqs):
    return [x.result for x in SyntheticGenerator.generate_batch(self, reqs)]


@pytest.mark.parametrize(
    ("behaviour", "error_class"),
    [
        # S1: reversed entries used to be credited to the wrong requests, silently.
        (_reversed, "BatchOrderMismatch"),
        (_rotated, "BatchOrderMismatch"),
        # Bare results without their request cannot be attributed.
        (_bare, "BatchEntryInvalid"),
    ],
)
def test_batch_entries_must_answer_their_own_request(
    tmp_path, raw, items, behaviour, error_class
) -> None:
    proto = parse_protocol({**raw, **SEQ})
    events: list = []
    loaders = _with(items, proto.seeds, events, "candidate", behaviour)
    _, res, _ = _run(tmp_path, {**raw, **SEQ}, items, loaders=loaders, events=events)
    rows = [r for r in res.records if r["arm"] == "candidate"]
    assert rows and {r["error_class"] for r in rows} == {error_class}
    assert all(r["status"] == "error" and r["output_sha256"] is None for r in rows)
    assert res.manifest["run_status"] == "complete"  # recorded per request, not dropped


def test_a_single_request_shard_is_not_refused_by_the_echo_check(tmp_path, raw, items) -> None:
    dev = [i for i in items if i.split == "development"]
    raw2 = {
        **raw,
        "execution": {"order": "per_arm_sequential", "shards": len(dev)},
        "sampling_mode": "n_per_request",
        "seeds": [11],
    }
    proto = parse_protocol(raw2)
    events: list = []
    loaders = _with(items, proto.seeds, events, "candidate", _reversed)
    _, res, _ = _run(tmp_path, raw2, items, loaders=loaders, events=events)
    assert not any(r["error_class"] == "BatchOrderMismatch" for r in res.records)


def test_a_generator_without_close_is_refused_before_it_runs(tmp_path, raw, items) -> None:
    # S2: the next arm used to load while this engine was still open.
    proto = parse_protocol({**raw, **SEQ})
    events: list = []
    templates = synthetic_generators(items, proto.seeds)

    class NoClose(StubEngine):
        close = None  # type: ignore[assignment]

    loaders = _loaders(
        items,
        proto.seeds,
        events,
        overrides={"reference": lambda: NoClose(templates["reference"], "ref-noclose", events)},
    )
    manifest, records = _assert_aborted(tmp_path, raw, items, loaders, events, RunError, "close")
    assert records == [] and not any(e[0] == "batch" for e in events)
    assert [e[1] for e in events if e[0] == "load"] == ["reference"]  # nothing loaded after it
    assert manifest["execution"]["loads"][0]["closed"] is None  # there was nothing to call


def test_shard_composition_is_hashed(tmp_path, raw, items) -> None:
    # S3: the dataset hash ignores item order, so the per-shard item lists are bound here.
    from wsp_eval.hashing import sha256_json

    _, a, _ = _run(tmp_path, {**raw, **SEQ}, items, name="a")
    _, b, _ = _run(tmp_path, {**raw, **SEQ}, list(reversed(items)), name="b")
    ea, eb = a.manifest["execution"], b.manifest["execution"]
    assert a.manifest["dataset"]["items_file_sha256"] == b.manifest["dataset"]["items_file_sha256"]
    assert ea["shards_sha256"] and ea["shards_sha256"] != eb["shards_sha256"]
    dev = [i.id for i in items if i.split == "development"]
    assert ea["shards_sha256"] == sha256_json([dev[0::2], dev[1::2]])


def test_interleaved_manifests_have_no_shard_hash(tmp_path, raw, items) -> None:
    proto = parse_protocol(raw)
    res = run_paired(
        proto,
        items,
        synthetic_generators(items, proto.seeds),
        FinalAnswerGrader(),
        tmp_path / "i",
        purpose="t",
        operator="op1",
    )
    assert res.manifest["execution"] == {
        "order": "interleaved",
        "shards": 1,
        "shards_sha256": None,
        "loads": None,
    }


def test_more_shards_than_items_is_refused_before_any_output(tmp_path, raw, items) -> None:
    # N2: empty shards used to be skipped silently while the manifest claimed the full count.
    n = len([i for i in items if i.split == "development"])
    raw2 = {**raw, "execution": {"order": "per_arm_sequential", "shards": n + 1}}
    with pytest.raises(RunError, match="exceeds"):
        _run(tmp_path, raw2, items)
    assert not (tmp_path / "run").exists()


def test_close_outcome_is_recorded_after_a_failure(tmp_path, raw, items) -> None:
    # N1: after an arm failure the manifest says whether the engine was released.
    proto = parse_protocol({**raw, **SEQ})
    events: list = []

    def boom(self, requests):
        raise KeyboardInterrupt

    class BadClose(_Scripted):
        def close(self):
            raise RuntimeError("release failed")

    templates = synthetic_generators(items, proto.seeds)
    loaders = _loaders(
        items,
        proto.seeds,
        events,
        overrides={"reference": lambda: BadClose(templates["reference"], "r-x", events, boom)},
    )
    manifest, _ = _assert_aborted(tmp_path, raw, items, loaders, events, KeyboardInterrupt, None)
    assert manifest["execution"]["loads"][0]["closed"] is False


def test_the_same_engine_object_cannot_serve_two_arms(tmp_path, raw, items) -> None:
    # N3: load_id is self-reported, so a relabelled engine object is also checked by identity.
    proto = parse_protocol({**raw, **SEQ})
    events: list = []
    templates = synthetic_generators(items, proto.seeds)
    shared = StubEngine(templates["reference"], "shared-1", events)

    def again():
        shared.load_id = "shared-2"
        return shared

    loaders = _loaders(
        items, proto.seeds, events, overrides={"reference": lambda: shared, "candidate": again}
    )
    _assert_aborted(tmp_path, raw, items, loaders, events, RunError, "already used")


def test_interleaved_returned_exception_is_still_invalid_generation(tmp_path, raw, items) -> None:
    # N4: a generate() that returns (not raises) an exception keeps main's InvalidGeneration.
    proto = parse_protocol(raw)
    gens = synthetic_generators(items, proto.seeds)

    class Returns(SyntheticGenerator):
        def generate(self, request):
            return ValueError("returned, not raised")

    g = gens["candidate"]
    gens["candidate"] = Returns(g.name, g.table, load_id=g.load_id)
    res = run_paired(
        proto, items, gens, FinalAnswerGrader(), tmp_path / "r", purpose="t", operator="op1"
    )
    rows = [r for r in res.records if r["arm"] == "candidate"]
    assert {r["error_class"] for r in rows} == {"InvalidGeneration"}

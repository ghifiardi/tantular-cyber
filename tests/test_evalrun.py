"""Evaluation-run provenance (POC-007 Step 1): snapshots, eligibility, run validity, split,
prompt storage, and their wiring into the comparison harness. Every failure path must be
fail-closed: excluded, incomplete or invalid, never silently comparable."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
from conftest import GOOD_EXPLANATION, requires_semgrep

from tantular_cyber import cli
from tantular_cyber.benchmark import compare
from tantular_cyber.benchmark.compare import ModelProvenance, parse_arms, run_comparison
from tantular_cyber.benchmark.evalrun import (
    LIMITATIONS,
    SCHEMA_VERSION,
    CodeState,
    ModelSnapshot,
    ResponseRecord,
    check_prompt_storage,
    exclusion_reasons,
    load_split,
    make_record,
    membership_sha256,
    run_status,
    snapshot,
)
from tantular_cyber.config import OllamaSettings, default_corpus
from tantular_cyber.ollama_client import OllamaClient

CORPUS = Path(__file__).parents[1] / "benchmarks" / "corpus"
D1, D2 = "1" * 64, "2" * 64


def snap(tags: str | None = D1, loaded: str | None = D1, quant: str = "Q8_0", error=None):
    return ModelSnapshot(
        model="m:a", tags_digest=tags, quantization=quant, loaded_digest=loaded, error=error
    )


# --- snapshots ---------------------------------------------------------------------------


def test_snapshot_reads_tags_and_ps_only() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "m:a", "digest": D1, "details": {"quantization_level": "Q8_0"}}
                    ]
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "m:a", "digest": D1}]})
        return httpx.Response(404)

    with httpx.Client(
        base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handler)
    ) as h:
        s = snapshot(h, "m:a")
        assert s == snap()
        # ":latest" is implied, as in OllamaClient; an unloaded model has no loaded digest.
        assert snapshot(h, "other").tags_digest is None
    assert set(seen) == {"/api/tags", "/api/ps"}  # never /api/show


def test_snapshot_records_an_unavailable_endpoint_as_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(500)
        return httpx.Response(200, json={"models": [{"name": "m:a", "digest": D1}]})

    with httpx.Client(
        base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handler)
    ) as h:
        s = snapshot(h, "m:a")
    assert s.error == "HTTPStatusError" and s.tags_digest is None


# --- eligibility ---------------------------------------------------------------------------


def reasons(before=None, after=None, req="q", resp="r", error=None) -> list[str]:
    return exclusion_reasons(before or snap(), after or snap(), req, resp, error)


def test_matching_snapshots_with_hashes_are_eligible() -> None:
    assert reasons() == []
    assert reasons(before=snap(loaded=None)) == []  # first request: model not yet loaded


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"after": snap(tags=D2, loaded=D2)}, "installed model digest changed"),
        ({"after": snap(quant="Q4_K_M")}, "quantization changed"),
        ({"after": snap(loaded=D2)}, "loaded model digest differs"),
        ({"before": snap(loaded=D2)}, "loaded model changed"),
        ({"before": snap(tags=None)}, "before snapshot: model digest missing"),
        ({"after": snap(tags=None, loaded=None)}, "after snapshot: model digest missing"),
        ({"after": snap(loaded=None)}, "model not loaded in /api/ps"),
        ({"after": snap(error="ConnectError")}, "after snapshot failed: ConnectError"),
        ({"before": snap(error="HTTPStatusError")}, "before snapshot failed"),
        ({"req": None}, "request hash missing"),
        ({"resp": None}, "response hash missing"),
        ({"error": "OllamaUnavailable", "resp": None}, "request failed: OllamaUnavailable"),
    ],
)
def test_missing_or_drifting_provenance_excludes_the_response(kwargs, message) -> None:
    r = reasons(**kwargs)
    assert any(message in x for x in r), r
    record = make_record(
        arm="a",
        run=1,
        finding_id="f",
        requested_at="t0",
        completed_at="t1",
        before=kwargs.get("before", snap()),
        after=kwargs.get("after", snap()),
        request_sha256=kwargs.get("req", "q"),
        response_sha256=kwargs.get("resp", "r"),
        error=kwargs.get("error"),
    )
    assert record.eligibility == "excluded" and record.exclusion_reasons


# --- run status ----------------------------------------------------------------------------


def rec(run: int = 1, ok: bool = True) -> ResponseRecord:
    return make_record(
        arm="a",
        run=run,
        finding_id="f",
        requested_at="t0",
        completed_at="t1",
        before=snap(),
        after=snap() if ok else snap(loaded=D2),
        request_sha256="q",
        response_sha256="r",
    )


CLEAN = CodeState(commit="c" * 40, dirty=False)


def test_complete_only_when_every_expected_response_is_eligible() -> None:
    assert run_status("controlled", CLEAN, CLEAN, [rec(1), rec(2)], 2) == ("complete", [])
    status, why = run_status("controlled", CLEAN, CLEAN, [rec(1), rec(2, ok=False)], 2)
    assert status == "incomplete" and "1 response(s) excluded" in why
    status, why = run_status("controlled", CLEAN, CLEAN, [rec(1)], 2)
    assert status == "incomplete" and "1 expected response(s) missing" in why
    status, why = run_status("controlled", CLEAN, CLEAN, [rec(1), rec(1)], 2)
    assert status == "incomplete" and any("duplicate" in w for w in why)


@pytest.mark.parametrize(
    ("mode", "start", "end", "message"),
    [
        ("controlled", CLEAN, CodeState(commit="e" * 40, dirty=False), "changed during the run"),
        ("exploratory", CLEAN, CodeState(commit="c" * 40, dirty=True), "changed during the run"),
        (
            "controlled",
            CodeState(commit=None, dirty=None),
            CodeState(commit=None, dirty=None),
            "commit unknown",
        ),
        (
            "controlled",
            CodeState(commit="c" * 40, dirty=True),
            CodeState(commit="c" * 40, dirty=True),
            "not clean",
        ),
    ],
)
def test_unknown_or_changing_code_invalidates_the_run(mode, start, end, message) -> None:
    status, why = run_status(mode, start, end, [rec()], 1)
    assert status == "invalid" and any(message in w for w in why), why


def test_exploratory_runs_record_but_tolerate_a_dirty_tree() -> None:
    dirty = CodeState(commit="c" * 40, dirty=True)
    assert run_status("exploratory", dirty, dirty, [rec()], 1) == ("complete", [])


# --- split ---------------------------------------------------------------------------------


def write_split(tmp_path: Path, **over) -> Path:
    body = {
        "schema": "tantular-cyber.evaluation-split",
        "schema_version": "1.0.0",
        "corpus_id": "c",
        "development": ["a", "b"],
        "heldout": ["c"],
        **over,
    }
    p = tmp_path / "split.json"
    p.write_text(json.dumps(body))
    return p


def test_split_hashes_are_recorded_and_side_is_returned(tmp_path: Path) -> None:
    p = write_split(tmp_path)
    ref, side = load_split(p, "heldout", "c", ["a", "b", "c"])
    assert side == {"c"} and ref.cases == 1 and ref.side == "heldout"
    assert ref.development_sha256 == membership_sha256(["b", "a"])
    assert ref.heldout_sha256 == membership_sha256(["c"])
    assert ref.manifest_sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"heldout": ["c", "a"]}, "both sides"),
        ({"heldout": ["zz"]}, "unknown case"),
        ({"corpus_id": "other"}, "not 'c'"),
        ({"development": ["a", "a"]}, "more than once"),
        ({"heldout": []}, "is empty"),
    ],
)
def test_split_violations_are_refused(tmp_path: Path, over, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_split(write_split(tmp_path, **over), "heldout", "c", ["a", "b", "c"])


def test_split_with_unknown_fields_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_split(write_split(tmp_path, extra=1), "heldout", "c", ["a", "b", "c"])


# --- prompt storage ------------------------------------------------------------------------


def test_prompt_text_only_for_the_bundled_corpus(tmp_path: Path) -> None:
    check_prompt_storage(False, tmp_path, default_corpus())
    check_prompt_storage(True, default_corpus(), default_corpus())
    with pytest.raises(ValueError, match="bundled synthetic corpus"):
        check_prompt_storage(True, tmp_path, default_corpus())


# --- OllamaClient hashes -------------------------------------------------------------------


def test_client_hashes_the_exact_payload_and_reply() -> None:
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:latest"}]})
        bodies.append(request.content)
        return httpx.Response(200, json={"message": {"content": "{}"}, "done_reason": "stop"})

    settings = OllamaSettings(model="m", seed=1)
    with OllamaClient(settings, transport=httpx.MockTransport(handler)) as c:
        c.chat([{"role": "user", "content": "x"}])
        first = c.last_request_sha256
        canonical = json.dumps(
            json.loads(bodies[0]), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        assert first == hashlib.sha256(canonical.encode()).hexdigest()
        assert c.last_response_sha256 == hashlib.sha256(b"{}").hexdigest()
        c.chat([{"role": "user", "content": "y"}])
        assert c.last_request_sha256 != first


# --- wired into the comparison harness -----------------------------------------------------


class Client:
    provider = "ollama"
    calls = 0

    def __init__(self, settings) -> None:
        self.model = settings.model
        self.last_metrics = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass

    def chat(self, messages, format_schema=None) -> str:
        Client.calls += 1
        self.last_request_sha256, self.last_response_sha256 = "q" * 64, "r" * 64
        return GOOD_EXPLANATION


class Snapshots:
    """Steady digests, except that ``drift_after`` snapshots in the digest changes."""

    drift_after: int | None = None
    fail_ps = False

    def __init__(self, base_url, transport=None) -> None:
        self.n = 0

    def __call__(self, model: str) -> ModelSnapshot:
        self.n += 1
        if Snapshots.fail_ps:
            return ModelSnapshot(model=model, error="HTTPStatusError")
        d = D2 if Snapshots.drift_after and self.n > Snapshots.drift_after else D1
        return ModelSnapshot(model=model, tags_digest=d, quantization="Q8_0", loaded_digest=d)

    def version(self) -> str:
        return "0.40.2"

    def close(self) -> None:
        pass


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch):
    Client.calls, Snapshots.drift_after, Snapshots.fail_ps = 0, None, False
    monkeypatch.setattr(compare, "OllamaClient", Client)
    monkeypatch.setattr(compare, "SnapshotSource", Snapshots)
    monkeypatch.setattr(
        compare,
        "model_provenance",
        lambda url, names: [ModelProvenance(name=n, installed=True) for n in names],
    )
    state = {"code": [("c" * 40, False)]}
    monkeypatch.setattr(
        compare,
        "code_state",
        lambda: state["code"][0] if len(state["code"]) == 1 else state["code"].pop(0),
    )
    return state


ARMS = parse_arms("m:a", "baseline,concise", "8192")


@requires_semgrep
def test_complete_controlled_run_is_comparable(harness) -> None:
    r = run_comparison(CORPUS, ARMS, runs=2, max_findings=3, mode="controlled")
    ev = r.evaluation_run
    assert ev.status == "complete" and ev.comparable and r.comparable
    assert ev.schema_version == SCHEMA_VERSION == "1.0.0"
    assert ev.counts == {"eligible": 12, "excluded": 0} and ev.expected_responses == 12
    assert ev.code_at_start == ev.code_at_end == CodeState(commit="c" * 40, dirty=False)
    assert ev.limitations == LIMITATIONS
    assert any("not proof" in x for x in ev.limitations)
    assert len(ev.inputs.evidence_sha256) == 3 and ev.inputs.corpus_sha256
    assert all(rr.prompt_messages is None for rr in ev.responses)  # hashes only by default
    assert not any("not comparable" in x for a in r.adoption for x in a.reasons)


@requires_semgrep
def test_drift_mid_run_excludes_responses_and_makes_the_run_incomplete(harness) -> None:
    Snapshots.drift_after = 5  # third response: before=D1, after=D2
    r = run_comparison(CORPUS, ARMS, runs=2, max_findings=3, mode="controlled")
    ev = r.evaluation_run
    assert ev.status == "incomplete" and not ev.comparable
    assert ev.counts["excluded"] >= 1
    assert any(
        "installed model digest changed" in x for rr in ev.responses for x in rr.exclusion_reasons
    )
    # Excluded responses leave the summaries; adoption is switched off.
    assert sum(a.n for a in r.arms) == ev.counts["eligible"]
    assert sum(a.excluded for a in r.arms) == ev.counts["excluded"]
    assert all(not a.adopt and "not comparable: incomplete run" in a.reasons for a in r.adoption)


@requires_semgrep
def test_unavailable_ps_excludes_every_response(harness) -> None:
    Snapshots.fail_ps = True
    r = run_comparison(CORPUS, ARMS[:1], runs=1, max_findings=2, mode="controlled")
    assert r.evaluation_run.counts == {"eligible": 0, "excluded": 2}
    assert r.evaluation_run.status == "incomplete" and r.arms[0].n == 0


def test_controlled_run_refuses_a_dirty_tree_before_any_call(harness) -> None:
    harness["code"] = [("c" * 40, True)]
    with pytest.raises(ValueError, match="clean checkout"):
        run_comparison(CORPUS, ARMS, runs=1, max_findings=1, mode="controlled")
    harness["code"] = [(None, None)]
    with pytest.raises(ValueError, match="clean checkout"):
        run_comparison(CORPUS, ARMS, runs=1, max_findings=1, mode="controlled")
    assert Client.calls == 0


@requires_semgrep
def test_code_changing_during_the_run_invalidates_it(harness) -> None:
    harness["code"] = [("c" * 40, False), ("e" * 40, False)]
    r = run_comparison(CORPUS, ARMS[:1], runs=1, max_findings=1, mode="controlled")
    assert r.evaluation_run.status == "invalid" and not r.comparable


@requires_semgrep
def test_resume_across_a_commit_or_mode_is_refused(harness, tmp_path: Path) -> None:
    ckpt = tmp_path / "samples.jsonl"
    run_comparison(CORPUS, ARMS[:1], runs=1, max_findings=2, checkpoint_path=ckpt)
    rows = [json.loads(x) for x in ckpt.read_text().splitlines()[1:]]
    assert all(ResponseRecord.model_validate(r["provenance"]) for r in rows)
    again = run_comparison(CORPUS, ARMS[:1], runs=1, max_findings=2, checkpoint_path=ckpt)
    assert again.evaluation_run.counts["eligible"] == 2 and Client.calls == 2  # reused
    harness["code"] = [("e" * 40, False)]
    with pytest.raises(ValueError, match="different settings"):
        run_comparison(CORPUS, ARMS[:1], runs=1, max_findings=2, checkpoint_path=ckpt)
    harness["code"] = [("c" * 40, False)]
    with pytest.raises(ValueError, match="different settings"):
        run_comparison(
            CORPUS, ARMS[:1], runs=1, max_findings=2, checkpoint_path=ckpt, mode="controlled"
        )


@requires_semgrep
def test_prompts_are_stored_only_on_request_for_the_bundled_corpus(harness, tmp_path) -> None:
    r = run_comparison(default_corpus(), ARMS[:1], runs=1, max_findings=1, store_prompts=True)
    assert r.evaluation_run.prompts_stored
    assert r.evaluation_run.responses[0].prompt_messages
    copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, copy)
    with pytest.raises(ValueError, match="bundled synthetic corpus"):
        run_comparison(copy, ARMS[:1], runs=1, max_findings=1, store_prompts=True)


@requires_semgrep
def test_split_side_limits_findings_and_is_hashed(harness, tmp_path: Path) -> None:
    from tantular_cyber.benchmark.manifest import load_manifest

    manifest, _ = load_manifest(CORPUS)
    ids = [c.id for c in manifest.cases]
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "schema": "tantular-cyber.evaluation-split",
                "schema_version": "1.0.0",
                "corpus_id": manifest.corpus_id,
                "development": ids[1:],
                "heldout": ids[:1],
            }
        )
    )
    held = run_comparison(CORPUS, ARMS[:1], runs=1, max_findings=40, split=(split, "heldout"))
    dev = run_comparison(CORPUS, ARMS[:1], runs=1, max_findings=40, split=(split, "development"))
    whole = run_comparison(CORPUS, ARMS[:1], runs=1, max_findings=40)
    s = held.evaluation_run.inputs.split
    assert s is not None and s.side == "heldout"
    assert s.heldout_sha256 == membership_sha256(ids[:1])
    assert s.development_sha256 == dev.evaluation_run.inputs.split.development_sha256
    h, d = (set(x.evaluation_run.inputs.finding_ids) for x in (held, dev))
    assert h and d and not (h & d)  # a finding never appears on both sides
    assert h | d <= set(whole.evaluation_run.inputs.finding_ids) | h | d
    assert held.evaluation_run.inputs_sha256 != dev.evaluation_run.inputs_sha256


def test_cli_exits_5_when_a_controlled_run_is_not_comparable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tantular_cyber.benchmark.evalrun import EvaluationRun, RunInputs

    def fake_run(*args, **kwargs):
        assert kwargs["mode"] == "controlled"
        ev = EvaluationRun(
            mode="controlled",
            status="incomplete",
            status_reasons=["1 response(s) excluded"],
            comparable=False,
            started_at="t0",
            finished_at="t1",
            code_at_start=CLEAN,
            code_at_end=CLEAN,
            environment={},
            inputs=RunInputs(
                corpus_id="c",
                manifest_version="1",
                corpus_sha256="x",
                finding_ids=[],
                evidence_sha256={},
                prompt_sha256="p",
                arms=[],
                runs=1,
                pinned_options={},
            ),
            inputs_sha256="i",
            expected_responses=1,
            counts={"eligible": 0, "excluded": 1},
            prompts_stored=False,
            responses=[],
        )
        return compare.ComparisonResult(
            started_at="t0",
            machine={},
            code_commit=None,
            corpus_id="c",
            findings=0,
            runs=1,
            pinned_options={},
            reference="a",
            models=[],
            arms=[],
            paired=[],
            adoption=[],
            notes=[],
            evaluation_run=ev,
        )

    monkeypatch.setattr(compare, "run_comparison", fake_run)
    out = tmp_path / "o"
    code = cli.main(["compare", "--controlled", "--output", str(out)])
    assert code == cli.EXIT_GATES
    written = json.loads((out / "evaluation-run.json").read_text())
    assert written["status"] == "incomplete" and written["schema_version"] == "1.0.0"

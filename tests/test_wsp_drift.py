"""WSP reload/drift policy, sampling modes and edit verification (synthetic fixtures only)."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WSP = ROOT / "tools" / "wsp"
sys.path.insert(0, str(WSP))

from test_wsp_harness import _bf16, _write_safetensors  # noqa: E402
from wsp_eval.checkpoint_diff import diff_checkpoints, verify_edit  # noqa: E402
from wsp_eval.dataset import load_items  # noqa: E402
from wsp_eval.drift import drift_report  # noqa: E402
from wsp_eval.generation import Generation, SyntheticGenerator  # noqa: E402
from wsp_eval.grading import FinalAnswerGrader  # noqa: E402
from wsp_eval.hashing import sha256_json  # noqa: E402
from wsp_eval.protocol import ProtocolError, load_protocol, parse_protocol  # noqa: E402
from wsp_eval.runner import run_paired  # noqa: E402
from wsp_eval.synthetic import synthetic_generators  # noqa: E402

FIX = WSP / "fixtures"
DRAFT = ROOT / "docs" / "wsp" / "protocols" / "exp8-replication-draft.json"
APPROVED = {
    "status": "approved",
    "max_abs_accuracy_diff": 0.05,
    "max_abs_token_limit_rate_diff": 0.05,
    "max_abs_error_rate_diff": 0.05,
    "approved_by": "evaluator1",
    "approved_at": "2026-11-01T00:00Z",
}


@pytest.fixture
def raw() -> dict:
    return json.loads((FIX / "synthetic_protocol.json").read_text())


@pytest.fixture
def items():
    return load_items(FIX / "synthetic_items.jsonl")


def _run(tmp_path, raw, items, *, gens=None, name="run"):
    proto = parse_protocol(raw)
    res = run_paired(
        proto,
        items,
        gens or synthetic_generators(items, proto.seeds),
        FinalAnswerGrader(),
        tmp_path / name,
        purpose="test",
        operator="op1",
        clock=iter(range(10_000)).__next__,
        run_id="wsp-drift-test",
    )
    return proto, res, drift_report(proto, res.manifest, res.records)


# --- protocol rules ------------------------------------------------------------------------


def test_reload_must_declare_the_reference_checkpoint_and_engine(raw) -> None:
    bad = copy.deepcopy(raw)
    bad["arms"][2]["checkpoint"] = {"name": "something-else", "revision": None}
    with pytest.raises(ProtocolError, match="reference checkpoint"):
        parse_protocol(bad)
    bad = copy.deepcopy(raw)
    bad["arms"][2]["engine"]["version"] = "fixture-2"
    with pytest.raises(ProtocolError, match="engine"):
        parse_protocol(bad)


def test_tolerance_cannot_look_approved_without_approval(raw) -> None:
    proposed = copy.deepcopy(raw)
    proposed["drift"]["tolerance"]["max_abs_accuracy_diff"] = 0.02  # a "proposal" in a field
    with pytest.raises(ProtocolError, match="unapproved tolerance"):
        parse_protocol(proposed)
    unfrozen = copy.deepcopy(raw)
    unfrozen["drift"]["tolerance"] = dict(APPROVED)
    with pytest.raises(ProtocolError, match="frozen"):
        parse_protocol(unfrozen)
    frozen = copy.deepcopy(unfrozen)
    frozen["frozen"] = True
    frozen["drift"]["tolerance"]["approved_by"] = None
    with pytest.raises(ProtocolError, match="approved_by"):
        parse_protocol(frozen)
    frozen["drift"]["tolerance"]["approved_by"] = "evaluator1"
    frozen["drift"]["tolerance"]["max_abs_error_rate_diff"] = 2
    with pytest.raises(ProtocolError, match="max_abs_error_rate_diff"):
        parse_protocol(frozen)


def test_n_per_request_takes_exactly_one_seed(raw) -> None:
    raw["sampling_mode"] = "n_per_request"
    with pytest.raises(ProtocolError, match="one request seed"):
        parse_protocol(raw)
    raw["seeds"] = [0]
    assert parse_protocol(raw).sampling_mode == "n_per_request"


# --- runner: sampling modes and finish reasons ------------------------------------------------


def test_n_per_request_mode_records_every_sample_under_one_seed(tmp_path, raw, items) -> None:
    raw.update(sampling_mode="n_per_request", seeds=[11])
    proto = parse_protocol(raw)
    # Script both samples under request seed 11 (aliases of the per-seed fixture entries).
    res = run_paired(
        proto,
        items,
        synthetic_generators(items, [11, 12]),
        FinalAnswerGrader(),
        tmp_path / "n",
        purpose="test",
        operator="op1",
        clock=iter(range(10_000)).__next__,
    )
    assert {r["seed"] for r in res.records} == {11}
    assert {r["request_n"] for r in res.records} == {2}
    assert len(res.records) == 6 * 2 * 3
    # The scripted adapter error fails the whole n=2 request: both samples become error rows.
    errs = [r for r in res.records if r["status"] == "error"]
    assert len(errs) == 2 and {r["sample"] for r in errs} == {0, 1}
    assert res.manifest["sampling_mode"] == "n_per_request"


def test_abnormal_finish_and_short_batches_are_errors(tmp_path, raw, items) -> None:
    proto = parse_protocol(raw)
    gens = synthetic_generators(items, proto.seeds)
    dev = sorted((i for i in items if i.split == "development"), key=lambda i: i.id)
    gens["reference"].table[(dev[0].id, 11)] = Generation("partial", finish_reason="abort")
    _, res, _ = _run(tmp_path, raw, items, gens=gens)
    bad = [r for r in res.records if r["error_class"] == "AbnormalFinish"]
    assert len(bad) == 1 and bad[0]["finish_reason"] == "abort"
    limit = [r for r in res.records if r["finish_reason"] == "length"]
    assert limit and all(r["hit_token_limit"] is True for r in limit)

    class Short(SyntheticGenerator):
        def generate(self, request):
            return [Generation("x", finish_reason="stop")]  # always one sample

    raw2 = copy.deepcopy(raw)
    raw2.update(sampling_mode="n_per_request", seeds=[11])
    gens2 = synthetic_generators(items, [11, 12])
    gens2["candidate"] = Short("short", {})
    _, res2, _ = _run(tmp_path, raw2, items, gens=gens2, name="short")
    short = [r for r in res2.records if r["arm"] == "candidate"]
    assert {r["error_class"] for r in short} == {"SampleCountMismatch"}


# --- drift: identity vs reproducibility ------------------------------------------------------


def test_identical_reload_is_descriptive_only_without_approved_tolerance(
    tmp_path, raw, items
) -> None:
    _, _, (d,) = _run(tmp_path, raw, items)
    assert d["identity"]["identity"] == "pass"
    assert d["identity"]["complete"] is False  # synthetic arms report no revision/version
    assert d["reproducibility"]["exact_output_match_rate"] == 1.0
    assert d["tolerance"]["verdict"] == "descriptive_only_tolerance_not_approved"


def test_stochastic_reload_differences_are_classified_not_hidden(tmp_path, raw, items) -> None:
    gens = synthetic_generators(items, [11, 12], nondeterministic_reload=True)
    _, _, (d,) = _run(tmp_path, raw, items, gens=gens)
    assert d["identity"]["identity"] == "pass"  # same artifacts: differences are sampling
    cats = d["reproducibility"]["pair_categories"]
    assert cats["different_output_same_grade"] == 1
    assert cats["different_output_grade_changed"] == 1
    assert cats["identical_output"] == 10
    acc = d["reproducibility"]["metrics"]["accuracy_diff"]
    assert acc["point"] > 0 and acc["ci95_bootstrap"][0] <= acc["point"] <= acc["ci95_bootstrap"][1]
    assert d["tolerance"]["verdict"] == "descriptive_only_tolerance_not_approved"


def test_identity_failure_blocks_interpretation(tmp_path, raw, items) -> None:
    gens = synthetic_generators(items, [11, 12], same_load_id=True)
    _, _, (d,) = _run(tmp_path, raw, items, gens=gens)
    failed = {c["check"] for c in d["identity"]["checks"] if c["status"] == "fail"}
    assert failed == {"separate_load"}
    assert d["tolerance"]["verdict"] == "identity_failed_reproducibility_not_interpreted"

    gens = synthetic_generators(items, [11, 12])
    gens["reload"].artifacts_sha256 = {"model.safetensors": "f" * 64}
    _, _, (d,) = _run(tmp_path, raw, items, gens=gens, name="artifacts")
    assert "loaded_artifacts_sha256" in d["tolerance"]["reasons"]


def _frozen_with(raw: dict, tolerance: dict) -> dict:
    out = copy.deepcopy(raw)
    out.update(frozen=True, frozen_by="lead1", frozen_at="2026-11-01T00:00Z")
    out["drift"]["tolerance"] = tolerance
    return out


def test_approved_tolerance_applies_only_if_approved_before_the_run(tmp_path, raw, items) -> None:
    _, _, (d,) = _run(tmp_path, _frozen_with(raw, APPROVED), items)
    assert d["tolerance"]["verdict"] == "within_tolerance_identity_incomplete"
    assert d["tolerance"]["approved_by"] == "evaluator1"

    late = dict(APPROVED, approved_at="2999-01-01T00:00Z")
    _, _, (d,) = _run(tmp_path, _frozen_with(raw, late), items, name="late")
    assert d["tolerance"]["verdict"] == "tolerance_not_applicable"
    assert "approved after the run started" in d["tolerance"]["reasons"][0]


def test_tolerance_exceeded_and_inconclusive(tmp_path, raw, items) -> None:
    gens = synthetic_generators(items, [11, 12], nondeterministic_reload=True)
    _, _, (d,) = _run(tmp_path, _frozen_with(raw, APPROVED), items, gens=gens)
    assert d["tolerance"]["verdict"] == "exceeds_tolerance"  # +1/12 accuracy > 0.05
    wide = dict(APPROVED, max_abs_accuracy_diff=0.1, max_abs_token_limit_rate_diff=0.1)
    gens = synthetic_generators(items, [11, 12], nondeterministic_reload=True)
    _, _, (d,) = _run(tmp_path, _frozen_with(raw, wide), items, gens=gens, name="wide")
    # Point estimate inside the bound, but 6 items cannot show the whole interval inside it.
    assert d["tolerance"]["per_metric"]["accuracy_diff"] == "inconclusive"
    assert d["tolerance"]["verdict"] == "inconclusive"


# --- verify_edit -----------------------------------------------------------------------------


def _bytes(vals: list[float]) -> bytes:
    return b"".join(_bf16(v) for v in vals)


def test_verify_edit_against_a_published_style_manifest(tmp_path) -> None:
    base_w, edit_w = [1.0, 2.0, 3.0, 4.0], [1.0, 2.5, 3.0, 2.0]
    common = {"b": ("F32", [2], struct.pack("<2f", 0.5, -0.5))}
    base = _write_safetensors(
        tmp_path / "base.safetensors", {"w": ("BF16", [2, 2], _bytes(base_w)), **common}
    )
    cand = _write_safetensors(
        tmp_path / "cand.safetensors", {"w": ("BF16", [2, 2], _bytes(edit_w)), **common}
    )
    manifest = {
        "tensors": [
            {
                "name": "w",
                "shape": [2, 2],
                "dtype": "bfloat16",
                "changed_entries": 2,
                "base_sha256": hashlib.sha256(_bytes(base_w)).hexdigest(),
                "edited_sha256": hashlib.sha256(_bytes(edit_w)).hexdigest(),
            }
        ]
    }
    report = verify_edit(diff_checkpoints(base, cand), manifest)
    assert report["status"] == "verified", report["problems"]
    # The report names the manifest it was checked against (canonical JSON SHA-256).
    assert report["manifest_sha256"] == sha256_json(manifest)

    wrong = copy.deepcopy(manifest)
    wrong["tensors"][0]["edited_sha256"] = "0" * 64
    wrong["tensors"][0]["changed_entries"] = 3
    wrong_report = verify_edit(diff_checkpoints(base, cand), wrong)
    problems = " | ".join(wrong_report["problems"])
    assert "edited tensor hash" in problems and "changed entries" in problems
    assert wrong_report["manifest_sha256"] != report["manifest_sha256"]

    extra = copy.deepcopy(manifest)
    extra["tensors"].append(dict(manifest["tensors"][0], name="b"))
    problems = " | ".join(verify_edit(diff_checkpoints(base, cand), extra)["problems"])
    assert "did not change" in problems
    assert verify_edit(diff_checkpoints(base, base), manifest)["status"] == "failed"


# --- the Experiment 8 draft protocol --------------------------------------------------------


def test_exp8_draft_protocol_is_valid_unfrozen_and_unapproved() -> None:
    proto = load_protocol(DRAFT)
    assert proto.frozen is False and proto.synthetic is False
    assert proto.sampling_mode == "n_per_request" and proto.seeds == (0,)
    assert proto.drift["tolerance"]["status"] == "unapproved"
    roles = {a.role for a in proto.arms}
    assert {"reference", "candidate", "no_edit_reload"} <= roles
    raw = json.loads(DRAFT.read_text())
    frozen = copy.deepcopy(raw)
    frozen["frozen"] = True
    # Cannot be frozen while open decisions remain: first the unset effect, then placeholders.
    with pytest.raises(ProtocolError, match="min_meaningful_effect"):
        parse_protocol(frozen)
    frozen["analysis"]["min_meaningful_effect"] = 0.1
    with pytest.raises(ProtocolError, match="placeholders"):
        parse_protocol(frozen)


def test_draft_style_protocol_never_claims_a_verdict(tmp_path, raw, items) -> None:
    raw["analysis"]["min_meaningful_effect"] = None
    proto, res, _ = _run(tmp_path, raw, items)
    from wsp_eval.analysis import analyze

    assert analyze(res.records, proto.analysis, synthetic=True)["verdict"] == (
        "descriptive_only_min_effect_not_set"
    )


# --- D1 scope decision (2026-11-04) --------------------------------------------------------

UNTOUCHED_WEIGHTS_LIMITATION = (
    "cannot establish that the authors' unreleased text-only export had identical untouched weights"
)


def test_scope_decision_is_recorded_without_authorizing_anything() -> None:
    raw = json.loads(DRAFT.read_text())
    (d1,) = [d for d in raw["decisions"] if d["id"] == "D1-scope"]
    assert "qualified MATH-500 replication" in d1["decision"]
    assert {"model runs", "freezing the protocol"} <= set(d1["does_not_authorize"])
    assert raw["status_notes"] == {
        "frozen": False,
        "license_review": "pending (human)",
        "downloads_authorized": False,
        "runs_authorized": False,
    }
    assert "in place of the authors' unreleased text-only export" in raw["claim_wording"]
    assert "not a reproduction of the edit's discovery" in raw["claim_wording"]
    # The reconstruction arm is only proposed: it is not one of the arms a run executes.
    (proposed,) = raw["proposed_arms"]
    assert proposed["id"] == "text_only_reconstruction"
    assert "pending budget and a separate protocol decision" in proposed["status"]
    assert proposed["id"] not in {a.id for a in load_protocol(DRAFT).arms}


def test_bounds_cannot_be_approved_before_the_run_choices_are_made() -> None:
    raw = json.loads(DRAFT.read_text())
    raw["frozen"] = True
    raw["analysis"]["min_meaningful_effect"] = 0.1
    raw["drift"]["tolerance"] = dict(APPROVED)
    # Engine version, hardware, dataset revision, grader and agreement criterion are open.
    with pytest.raises(ProtocolError, match="placeholders") as exc:
        parse_protocol(raw)
    for hole in ("engine", "dataset.revision", "grader.revision", "agreement_with_published"):
        assert hole in str(exc.value)


def test_every_analysis_report_carries_the_required_limitations(tmp_path, raw, items) -> None:
    draft = json.loads(DRAFT.read_text())
    assert any(UNTOUCHED_WEIGHTS_LIMITATION in x for x in draft["analysis"]["required_limitations"])
    raw["analysis"]["required_limitations"] = draft["analysis"]["required_limitations"]
    proto, res, _ = _run(tmp_path, raw, items)
    from wsp_eval.analysis import analyze

    report = analyze(res.records, proto.analysis, synthetic=True)
    assert report["limitations"] == draft["analysis"]["required_limitations"]


def test_verify_edit_cli_reports_which_manifest_it_checked(tmp_path, capsys) -> None:
    import wsp

    base_w, edit_w = [1.0, 2.0], [1.0, 2.5]
    base = _write_safetensors(tmp_path / "base.safetensors", {"w": ("BF16", [2], _bytes(base_w))})
    cand = _write_safetensors(tmp_path / "cand.safetensors", {"w": ("BF16", [2], _bytes(edit_w))})
    manifest = {
        "tensors": [
            {
                "name": "w",
                "shape": [2],
                "dtype": "bfloat16",
                "changed_entries": 1,
                "base_sha256": hashlib.sha256(_bytes(base_w)).hexdigest(),
                "edited_sha256": hashlib.sha256(_bytes(edit_w)).hexdigest(),
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    assert wsp.main(["verify-edit", str(base), str(cand), str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "verified"
    assert out["manifest_sha256"] == sha256_json(manifest)  # canonical JSON content
    assert out["manifest_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_edit_cli_hashes_the_manifest_bytes_it_parsed(tmp_path, capsys, monkeypatch):
    # The manifest used to be read twice: once to parse, and again after the (slow) checkpoint
    # diff to hash, so a file replaced in between gave a manifest_file_sha256 for other bytes.
    import wsp

    base_w, edit_w = [1.0, 2.0], [1.0, 2.5]
    base = _write_safetensors(tmp_path / "base.safetensors", {"w": ("BF16", [2], _bytes(base_w))})
    cand = _write_safetensors(tmp_path / "cand.safetensors", {"w": ("BF16", [2], _bytes(edit_w))})
    manifest = {
        "tensors": [
            {
                "name": "w",
                "shape": [2],
                "dtype": "bfloat16",
                "changed_entries": 1,
                "base_sha256": hashlib.sha256(_bytes(base_w)).hexdigest(),
                "edited_sha256": hashlib.sha256(_bytes(edit_w)).hexdigest(),
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    parsed_bytes = path.read_bytes()
    real_diff = wsp.diff_checkpoints

    def diff_then_swap(a, b):
        out = real_diff(a, b)
        path.write_text(json.dumps({"tensors": [], "swapped": True}))
        return out

    monkeypatch.setattr(wsp, "diff_checkpoints", diff_then_swap)
    assert wsp.main(["verify-edit", str(base), str(cand), str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "verified" and out["manifest_sha256"] == sha256_json(manifest)
    assert out["manifest_file_sha256"] == hashlib.sha256(parsed_bytes).hexdigest()
    assert out["manifest_file_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_edit_refuses_a_manifest_with_non_finite_numbers() -> None:
    diff = {"changed_tensors": [], "only_in_base": [], "only_in_candidate": []}
    diff.update(shape_or_dtype_mismatch=[], metadata_equal=True)
    diff.update(base={"sha256": "0" * 64}, candidate={"sha256": "9" * 64})
    with pytest.raises(ValueError):
        verify_edit(diff, {"tensors": [{"name": "w", "changed_entries": float("nan")}]})

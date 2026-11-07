"""WSP harness review fixes (2026-11-04): failure handling, run integrity, pairing checks,
preregistration timing and protocol strictness. Synthetic fixtures only, no model."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import shutil
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WSP = ROOT / "tools" / "wsp"
sys.path.insert(0, str(WSP))

import wsp  # noqa: E402
from wsp_eval.analysis import analyze, preregistration  # noqa: E402
from wsp_eval.checkpoint_diff import SafetensorsError, read_header  # noqa: E402
from wsp_eval.dataset import load_items  # noqa: E402
from wsp_eval.drift import (  # noqa: E402
    drift_report,
    identity_report,
    reproducibility_report,
)
from wsp_eval.generation import Generation, SyntheticGenerator  # noqa: E402
from wsp_eval.grading import FinalAnswerGrader, Grade  # noqa: E402
from wsp_eval.protocol import ProtocolError, parse_protocol  # noqa: E402
from wsp_eval.runner import RunError, run_paired, verify_run  # noqa: E402
from wsp_eval.synthetic import _prompt, synthetic_generators  # noqa: E402

FIX = WSP / "fixtures"
PROTOCOL_PATH = FIX / "synthetic_protocol.json"
FROZEN = {"frozen": True, "frozen_by": "lead1", "frozen_at": "2026-11-01T00:00Z"}


@pytest.fixture
def raw() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


@pytest.fixture
def items():
    return load_items(FIX / "synthetic_items.jsonl")


def _run(tmp_path, raw, items, *, gens=None, grader=None, name="run"):
    proto = parse_protocol(raw)
    res = run_paired(
        proto,
        items,
        gens or synthetic_generators(items, proto.seeds),
        grader or FinalAnswerGrader(),
        tmp_path / name,
        purpose="test",
        operator="op1",
        clock=iter(range(10_000)).__next__,
        run_id="wsp-review-test",
    )
    return proto, res


def _replace_reference(gens, items, seeds, entry):
    table = dict(gens["reference"].table)
    first = sorted(i.id for i in items if i.split == "development")[0]
    table[(first, seeds[0])] = entry
    gens["reference"] = SyntheticGenerator(
        "synthetic-reference",
        table,
        load_id="load-ref",
        artifacts_sha256=gens["reference"].artifacts_sha256,
    )
    return first


# --- failure handling ----------------------------------------------------------------------


class _CrashingGrader(FinalAnswerGrader):
    def grade(self, item, text) -> Grade:
        if item.id == "syn-dev-002":
            raise ZeroDivisionError("synthetic grader bug")
        return super().grade(item, text)


def test_grader_crash_is_an_error_row_not_a_lost_run(tmp_path, raw, items) -> None:
    _, res = _run(tmp_path, raw, items, grader=_CrashingGrader())
    bad = [r for r in res.records if r["item_id"] == "syn-dev-002"]
    assert bad and all(r["status"] == "error" for r in bad)
    assert {r["error_class"] for r in bad} == {"GraderError:ZeroDivisionError"}
    # The generation itself happened: its hash and token accounting are kept.
    assert all(r["output_sha256"] and r["finish_reason"] for r in bad)
    assert res.manifest["run_status"] == "complete"
    assert res.manifest["counts"]["records"] == res.manifest["counts"]["expected_records"]


def test_invalid_and_self_contradictory_generations_are_errors(tmp_path, raw, items) -> None:
    proto = parse_protocol(raw)
    gens = synthetic_generators(items, proto.seeds)
    first = _replace_reference(gens, items, proto.seeds, {"text": "not a Generation"})
    _, res = _run(tmp_path, raw, items, gens=gens)
    (row,) = [
        r
        for r in res.records
        if r["arm"] == "reference" and (r["item_id"], r["sample"]) == (first, 0)
    ]
    assert row["status"] == "error" and row["error_class"] == "InvalidGeneration"

    gens = synthetic_generators(items, proto.seeds)
    liar = Generation("\\boxed{5}", output_tokens=3, hit_token_limit=True, finish_reason="stop")
    _replace_reference(gens, items, proto.seeds, liar)
    _, res = _run(tmp_path, raw, items, gens=gens, name="liar")
    (row,) = [
        r
        for r in res.records
        if r["arm"] == "reference" and (r["item_id"], r["sample"]) == (first, 0)
    ]
    assert row["error_class"] == "InconsistentFinish" and row["finish_reason"] == "stop"


class _Interrupt(SyntheticGenerator):
    def generate(self, request):
        if request.item_id == "syn-dev-003":
            raise KeyboardInterrupt  # not an adapter failure: the harness run stops
        return super().generate(request)


def test_aborted_run_leaves_partial_records_and_is_never_analyzed(tmp_path, raw, items) -> None:
    proto = parse_protocol(raw)
    gens = synthetic_generators(items, proto.seeds)
    g = gens["candidate"]
    gens["candidate"] = _Interrupt(
        g.name, g.table, load_id=g.load_id, artifacts_sha256=g.artifacts_sha256
    )
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, raw, items, gens=gens)
    out = tmp_path / "run"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run_status"] == "aborted" and manifest["abort_reason"] == "KeyboardInterrupt"
    assert 0 < manifest["counts"]["records"] < manifest["counts"]["expected_records"]
    with pytest.raises(RunError, match="aborted"):
        verify_run(out, proto)
    assert wsp.main(["analyze", "--run", str(out), "--protocol", str(PROTOCOL_PATH)]) == 1


# --- run integrity -------------------------------------------------------------------------


def test_records_are_hashed_and_tampering_is_detected(tmp_path, raw, items, capsys) -> None:
    proto, res = _run(tmp_path, raw, items)
    manifest, records = verify_run(res.out_dir, proto)
    assert records == res.records and len(manifest["records_sha256"]) == 64
    assert manifest["outputs_sha256"] is None

    path = res.out_dir / "records.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")  # a dropped row
    with pytest.raises(RunError, match="records_sha256"):
        verify_run(res.out_dir, proto)
    assert wsp.main(["drift", "--run", str(res.out_dir), "--protocol", str(PROTOCOL_PATH)]) == 1
    assert "records_sha256" in capsys.readouterr().err

    other = copy.deepcopy(raw)
    other["description"] = "edited after the run"
    with pytest.raises(RunError, match="protocol hash"):
        verify_run(res.out_dir, parse_protocol(other))


def _hash_of_copy(root: Path) -> str:
    spec = importlib.util.spec_from_file_location("wsp_hash_copy", root / "wsp_eval" / "hashing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # our own harness file, copied; not analyzed-repo code
    return module.harness_code_sha256()


def test_harness_hash_covers_the_cli(tmp_path) -> None:
    copy_root = tmp_path / "wsp"
    shutil.copytree(WSP, copy_root, ignore=shutil.ignore_patterns("__pycache__"))
    before = _hash_of_copy(copy_root)
    cli = copy_root / "wsp.py"
    cli.write_text(cli.read_text() + "\n# changed\n")
    assert _hash_of_copy(copy_root) != before


# --- paired analysis -----------------------------------------------------------------------

PLAN = {
    "primary_endpoint": "accuracy",
    "reference_arm": "reference",
    "candidate_arm": "candidate",
    "missing_policy": "count_as_incorrect",
    "min_meaningful_effect": 0.1,
    "bootstrap_resamples": 500,
    "bootstrap_seed": 1,
}


def _row(arm: str, item: str, sample: int, ok: bool) -> dict:
    return {
        "arm": arm,
        "item_id": item,
        "sample": sample,
        "status": "graded",
        "correct": ok,
        "hit_token_limit": False,
        "output_tokens": 1,
    }


def test_duplicate_records_raise_and_unbalanced_items_are_excluded() -> None:
    rows = [_row("reference", "i1", 0, True), _row("candidate", "i1", 0, True)]
    with pytest.raises(ValueError, match="duplicate record"):
        analyze([*rows, rows[0]], PLAN, synthetic=True)

    rows = [
        _row("reference", "i1", 0, False),
        _row("reference", "i1", 1, False),
        _row("candidate", "i1", 0, True),  # sample 1 missing on the candidate side
        _row("reference", "i2", 0, True),
        _row("candidate", "i2", 0, True),
    ]
    a = analyze(rows, PLAN, synthetic=True)
    assert a["paired"]["items_unbalanced"] == ["i1"]
    assert "i1" in a["paired"]["items_excluded"] and a["paired"]["items_paired"] == 1


# --- preregistration -----------------------------------------------------------------------


def test_unfrozen_protocol_rule_outcome_is_never_the_verdict(tmp_path, raw, items) -> None:
    proto, res = _run(tmp_path, raw, items)  # unfrozen, but min_meaningful_effect is set
    prereg = preregistration(proto.raw, res.manifest)
    assert prereg["preregistered"] is False and "protocol not frozen" in prereg["reasons"]
    a = analyze(res.records, proto.analysis, synthetic=True, prereg=prereg)
    assert a["verdict"] == "descriptive_only_not_preregistered"
    assert a["unregistered_rule_outcome"] in (
        "criteria_met_pending_human_review",
        "candidate_worse",
        "inconclusive",
    )
    # No prereg information at all is treated the same way.
    bare = analyze(res.records, proto.analysis, synthetic=True)
    assert bare["verdict"] == "descriptive_only_not_preregistered"


def test_rule_applies_only_if_frozen_before_the_run(tmp_path, raw, items) -> None:
    proto, res = _run(tmp_path, {**raw, **FROZEN}, items)
    prereg = preregistration(proto.raw, res.manifest)
    assert prereg["preregistered"] is True and prereg["frozen_by"] == "lead1"
    a = analyze(res.records, proto.analysis, synthetic=True, prereg=prereg)
    assert a["unregistered_rule_outcome"] is None
    assert a["verdict"] in ("criteria_met_pending_human_review", "candidate_worse", "inconclusive")

    late = {**raw, **FROZEN, "frozen_at": "2999-01-01T00:00Z"}
    proto, res = _run(tmp_path, late, items, name="late")
    prereg = preregistration(proto.raw, res.manifest)
    assert prereg["reasons"] == ["protocol frozen after the run started"]
    assert analyze(res.records, proto.analysis, synthetic=True, prereg=prereg)["verdict"] == (
        "descriptive_only_not_preregistered"
    )


def test_tolerance_needs_a_protocol_frozen_before_the_run(tmp_path, raw, items) -> None:
    tol = {
        "status": "approved",
        "max_abs_accuracy_diff": 0.05,
        "max_abs_token_limit_rate_diff": 0.05,
        "max_abs_error_rate_diff": 0.05,
        "approved_by": "evaluator1",
        "approved_at": "2026-11-01T00:00Z",
    }
    late = {**copy.deepcopy(raw), **FROZEN, "frozen_at": "2999-01-01T00:00Z"}
    late["drift"]["tolerance"] = tol
    proto, res = _run(tmp_path, late, items)
    (d,) = drift_report(proto, res.manifest, res.records)
    assert d["tolerance"]["verdict"] == "tolerance_not_applicable"
    assert "protocol frozen after the run started" in d["tolerance"]["reasons"]


# --- protocol strictness -------------------------------------------------------------------


def test_freeze_metadata_is_required_and_only_when_frozen(raw) -> None:
    with pytest.raises(ProtocolError, match="frozen_by"):
        parse_protocol({**raw, "frozen": True})
    with pytest.raises(ProtocolError, match="frozen_at"):
        parse_protocol({**raw, "frozen": True, "frozen_by": "lead1", "frozen_at": "yesterday"})
    with pytest.raises(ProtocolError, match="unfrozen protocol must not carry"):
        parse_protocol({**raw, "frozen_by": "lead1"})
    assert parse_protocol({**raw, **FROZEN}).frozen is True


def test_analysis_arms_must_have_matching_roles(raw) -> None:
    bad = copy.deepcopy(raw)
    bad["analysis"]["candidate_arm"] = "reload"
    with pytest.raises(ProtocolError, match="candidate_arm"):
        parse_protocol(bad)
    bad = copy.deepcopy(raw)
    bad["analysis"]["reference_arm"] = "candidate"
    with pytest.raises(ProtocolError, match="reference_arm"):
        parse_protocol(bad)


def test_unapproved_tolerance_cannot_smuggle_bounds_under_other_names(raw) -> None:
    raw["drift"]["tolerance"]["proposed_accuracy_bound"] = 0.02
    with pytest.raises(ProtocolError, match="unknown keys"):
        parse_protocol(raw)


# --- checkpoint headers --------------------------------------------------------------------


def test_safetensors_header_with_duplicate_tensor_names_is_rejected(tmp_path) -> None:
    entry = '{"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}'
    raw = ('{"t": ' + entry + ', "t": ' + entry + "}").encode()
    path = tmp_path / "dup.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    with pytest.raises(SafetensorsError, match="duplicate header key"):
        read_header(path)


# --- drift: duplicates, unbalanced items, finish-error rows (2026-11-04, findings 1-3) -----


def test_direct_drift_calls_reject_duplicate_records(tmp_path, raw, items) -> None:
    proto, res = _run(tmp_path, raw, items)
    dup = next(r for r in res.records if r["arm"] == "reload")
    records = [*res.records, dict(dup, output_sha256="0" * 64)]
    for call in (
        lambda: identity_report(proto, res.manifest, records, "reference", "reload"),
        lambda: reproducibility_report(proto, records, "reference", "reload"),
        lambda: drift_report(proto, res.manifest, records),
    ):
        with pytest.raises(ValueError, match="duplicate record"):
            call()
    # A duplicate in an arm the report does not compare is not its concern.
    other = next(r for r in res.records if r["arm"] == "candidate")
    reproducibility_report(proto, [*res.records, other], "reference", "reload")


def test_drift_excludes_and_lists_unbalanced_items(tmp_path, raw, items) -> None:
    proto, res = _run(tmp_path, raw, items)
    dropped = next(r for r in res.records if r["arm"] == "reload" and r["sample"] == 1)
    records = [r for r in res.records if r is not dropped]
    rep = reproducibility_report(proto, records, "reference", "reload")
    item = dropped["item_id"]
    assert rep["items_unbalanced"] == [item]
    assert rep["pair_categories"]["unpaired"] == 1  # the missing sample is still visible
    n_items = len({r["item_id"] for r in res.records})
    for m in rep["metrics"].values():
        assert item in m["items_excluded"] and m["items_used"] == n_items - 1
    # Same rule as the paired analysis.
    plan = dict(proto.analysis, candidate_arm="candidate")
    paired = analyze(
        [r for r in records if r["arm"] == "reference"]
        + [dict(r, arm="candidate") for r in records if r["arm"] == "reload"],
        plan,
        synthetic=True,
    )
    assert paired["paired"]["items_unbalanced"] == [item]


def _replace(gens, arm, key, entry):
    g = gens[arm]
    table = dict(g.table)
    table[key] = entry
    gens[arm] = SyntheticGenerator(
        g.name, table, load_id=g.load_id, artifacts_sha256=g.artifacts_sha256
    )


def _finish_error_run(tmp_path, raw, items, *, prompt_for, name):
    proto = parse_protocol(raw)
    gens = synthetic_generators(items, proto.seeds)
    dev0 = sorted((i for i in items if i.split == "development"), key=lambda i: i.id)[0]
    s0, s1 = proto.seeds
    abnormal = Generation("partial", output_tokens=5, finish_reason="abort")
    inconsistent = Generation(
        "\\boxed{5}", output_tokens=3, hit_token_limit=True, finish_reason="stop"
    )
    _replace(
        gens, "reload", (dev0.id, s0), replace(abnormal, rendered_prompt_sha256=prompt_for(dev0))
    )
    _replace(
        gens,
        "reload",
        (dev0.id, s1),
        replace(inconsistent, rendered_prompt_sha256=prompt_for(dev0)),
    )
    proto, res = _run(tmp_path, raw, items, gens=gens, name=name)
    return proto, res, dev0


def test_finish_error_rows_keep_the_reported_prompt_hash(tmp_path, raw, items) -> None:
    proto, res, dev0 = _finish_error_run(tmp_path, raw, items, prompt_for=_prompt, name="same")
    rows = {(r["arm"], r["sample"]): r for r in res.records if r["item_id"] == dev0.id}
    for sample, cls, finish in ((0, "AbnormalFinish", "abort"), (1, "InconsistentFinish", "stop")):
        row = rows[("reload", sample)]
        assert row["status"] == "error" and row["error_class"] == cls
        assert row["finish_reason"] == finish
        assert (
            row["rendered_prompt_sha256"] == rows[("reference", sample)]["rendered_prompt_sha256"]
        )
        assert row["output_sha256"] is None and row["correct"] is None  # not scored

    (d,) = drift_report(proto, res.manifest, res.records)
    checks = {c["check"]: c["status"] for c in d["identity"]["checks"]}
    assert checks["rendered_prompts"] == "pass"  # error rows took part, nothing unrecorded
    rep = d["reproducibility"]
    assert rep["pair_categories"]["one_side_error"] == 2
    assert rep["errors_by_arm"]["reload"] == {"AbnormalFinish": 1, "InconsistentFinish": 1}
    assert rep["metrics"]["error_rate_diff"]["point"] > 0
    assert d["tolerance"]["verdict"] == "descriptive_only_tolerance_not_approved"


def test_finish_error_row_with_a_different_prompt_fails_identity(tmp_path, raw, items) -> None:
    proto, res, _ = _finish_error_run(
        tmp_path, raw, items, prompt_for=lambda item: "f" * 64, name="other"
    )
    (d,) = drift_report(proto, res.manifest, res.records)
    checks = {c["check"]: c for c in d["identity"]["checks"]}
    assert checks["rendered_prompts"]["status"] == "fail"
    assert checks["rendered_prompts"]["detail"].startswith("2 of")
    assert d["tolerance"]["verdict"] == "identity_failed_reproducibility_not_interpreted"


def test_finish_error_row_without_a_prompt_hash_is_not_recorded(tmp_path, raw, items) -> None:
    proto, res, _ = _finish_error_run(
        tmp_path, raw, items, prompt_for=lambda item: None, name="none"
    )
    (d,) = drift_report(proto, res.manifest, res.records)
    checks = {c["check"]: c for c in d["identity"]["checks"]}
    assert checks["rendered_prompts"]["status"] == "not_recorded"
    assert d["identity"]["complete"] is False


# --- post-merge review (PR #23, Codex P2) --------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        None,
        "drop",
        "EXCLUDE_ITEM",
        # Exact membership, not a pattern: whitespace, empty and non-string values are refused.
        "exclude_item\n",
        " exclude_item",
        "",
        1,
        True,
        ["exclude_item"],
    ],
)
def test_protocol_rejects_missing_or_unknown_missing_policy(raw, policy) -> None:
    for frozen in (False, True):
        bad = copy.deepcopy(raw)
        if frozen:
            bad.update(FROZEN)
        if policy is None:
            del bad["analysis"]["missing_policy"]
        else:
            bad["analysis"]["missing_policy"] = policy
        with pytest.raises(ProtocolError, match="missing_policy"):
            parse_protocol(bad)


@pytest.mark.parametrize("policy", [None, "drop"])
def test_analysis_and_drift_reject_bad_policies_the_same_way(tmp_path, raw, items, policy) -> None:
    proto, res = _run(tmp_path, raw, items)
    plan = dict(proto.analysis)  # as if the plan were altered after parsing
    if policy is None:
        del plan["missing_policy"]
    else:
        plan["missing_policy"] = policy
    altered = dataclasses.replace(proto, analysis=plan)
    with pytest.raises(ProtocolError, match="missing_policy"):
        analyze(res.records, plan, synthetic=True)
    with pytest.raises(ProtocolError, match="missing_policy"):
        reproducibility_report(altered, res.records, "reference", "reload")
    with pytest.raises(ProtocolError, match="missing_policy"):
        drift_report(altered, res.manifest, res.records)

"""Research improvement workflow (tools/research): synthetic fixtures only, no model, no SIEM."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESEARCH = ROOT / "tools" / "research"
WSP = ROOT / "tools" / "wsp"
sys.path[:0] = [str(RESEARCH), str(WSP)]

import improve  # noqa: E402
from improve_eval.compare import compare_candidate  # noqa: E402
from improve_eval.decision import make_decision, validate_decision  # noqa: E402
from improve_eval.evidence import (  # noqa: E402
    Evidence,
    EvidenceError,
    evidence_from_wsp,
    item_set_sha256,
    load_evidence,
    records_metadata,
    run_manifest_sha256,
    slice_membership_sha256,
    triage_metadata_sha256,
)
from improve_eval.metrics import (  # noqa: E402
    clopper_pearson,
    item_values,
    net_paired_losses,
    one_sided_bound,
    paired_difference_bootstrap,
    paired_improvement,
)
from improve_eval.report import build_report, render_markdown  # noqa: E402
from improve_eval.soc import (  # noqa: E402
    Alert,
    SyntheticTriageEvaluator,
    TriageDecision,
    alerts_metadata_sha256,
    alerts_sha256,
    load_alerts,
    run_triage_evaluation,
)
from improve_eval.study import StudyError, parse_study  # noqa: E402
from improve_eval.synthetic import synthetic_evaluators  # noqa: E402
from wsp_eval.dataset import load_items  # noqa: E402
from wsp_eval.generation import SyntheticGenerator  # noqa: E402
from wsp_eval.grading import FinalAnswerGrader  # noqa: E402
from wsp_eval.hashing import sha256_json  # noqa: E402
from wsp_eval.protocol import parse_protocol  # noqa: E402
from wsp_eval.runner import RunError, run_paired  # noqa: E402
from wsp_eval.synthetic import synthetic_generators  # noqa: E402

FIX = RESEARCH / "fixtures"
FROZEN = {"frozen": True, "frozen_by": "lead1", "frozen_at": "2026-11-01T00:00Z"}
THRESH, SOC, EDIT = "threshold-defer-0.3", "soc-model-alt", "weight-edit-unverified"


@pytest.fixture
def raw() -> dict:
    return json.loads((FIX / "synthetic_study.json").read_text())


@pytest.fixture
def alerts():
    return load_alerts(FIX / "synthetic_alerts.jsonl")


def _evidence(tmp_path, raw, alerts, *, name="ev", **overrides):
    study = parse_study(raw)
    a, labels = alerts
    run = run_triage_evaluation(
        study,
        a,
        labels,
        synthetic_evaluators(study, **overrides),
        tmp_path / name,
        operator="op1",
        run_id=f"imp-{name}",
    )
    # Stands in for an evidence register outside the run directory, written when the run
    # finished. In tests it is only a dict; real use needs an independently controlled store.
    EXTERNAL_RECORD[run] = run_manifest_sha256(run)
    return study, run


EXTERNAL_RECORD: dict[Path, str] = {}


def _load(run, study):
    """Load with the manifest checked against the external record (the confirmatory path)."""
    return load_evidence(run, study, expected_run_manifest_sha256=EXTERNAL_RECORD[run])


def _report(tmp_path, raw, alerts, **kw):
    study, run = _evidence(tmp_path, raw, alerts, **kw)
    return study, build_report(study, [_load(run, study)])


def _by_id(report: dict) -> dict:
    return {r["candidate"]: r for r in report["candidates"]}


# --- study registry and intervention taxonomy ---------------------------------------------


def test_fixture_study_is_valid_unfrozen_and_synthetic(raw) -> None:
    study = parse_study(raw)
    assert study.frozen is False and study.synthetic is True
    assert study.sha256 == parse_study(copy.deepcopy(raw)).sha256
    assert [c["intervention"]["type"] for c in study.candidates] == [
        "prompt_or_threshold",
        "soc_triage_model",
        "weight_edit",
    ]


def _mutate(raw: dict, path: str, value) -> dict:
    out = copy.deepcopy(raw)
    *parents, last = path.split(".")
    node = out
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    if value is KeyError:
        del node[last]
    else:
        node[last] = value
    return out


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("baseline.status", "tuned", "unchanged"),
        ("candidates.0.intervention.type", ["prompt_or_threshold", "lora"], "one intervention"),
        ("candidates.0.intervention.type", "magic", "intervention.type"),
        ("candidates.0.intervention.family", "training", "belongs to family"),
        ("candidates.0.intervention.change", KeyError, "needs intervention.change"),
        ("candidates.0.artifact", {"name": "other", "sha256": "9" * 64}, "must equal the baseline"),
        (
            "candidates.0.config",
            {"select_at": 0.8, "defer_at": 0.5, "score_feature": "synthetic_score"},
            "must change",
        ),
        ("candidates.1.artifact", None, "artifact"),
        ("candidates.2.artifact.sha256", "1" * 64, "must differ"),
        (
            "candidates.2.intervention.edit_verification",
            {"status": "looks fine"},
            "edit_verification",
        ),
        ("candidates.1.intervention.calibration", {}, "calibration"),
        ("target_failure.metric", "accuracy", "not defined for alert_triage"),
        ("constraints.0.kind", "max_net_paired_losses", "not pairable"),
        ("constraints.1.op", "<", "op"),
        ("constraints.4.margin", -0.1, "margin"),
        ("analysis.confidence", 1.5, "confidence"),
    ],
)
def test_study_rejects_inconsistent_registrations(raw, path, value, match) -> None:
    with pytest.raises(StudyError, match=match):
        parse_study(_mutate(raw, path, value))


def test_intervention_types_need_their_own_evidence_fields(raw) -> None:
    lora = copy.deepcopy(raw)
    lora["candidates"][2]["intervention"] = {"type": "lora", "training_data": {"n": 1}}
    with pytest.raises(StudyError, match=r"lora needs intervention\.base_artifact_sha256"):
        parse_study(lora)
    lora["candidates"][2]["intervention"].update(
        base_artifact_sha256="1" * 64, training_config_sha256="6" * 64, compute_budget="none"
    )
    assert parse_study(lora).candidates[2]["intervention"]["type"] == "lora"


def test_frozen_study_needs_names_times_and_every_bound(raw) -> None:
    with pytest.raises(StudyError, match="frozen_by"):
        parse_study({**raw, "frozen": True})
    with pytest.raises(StudyError, match="unfrozen study must not carry"):
        parse_study({**raw, "frozen_at": "2026-11-01T00:00Z"})
    unset = _mutate(raw, "constraints.1.bound", None)
    parse_study(unset)  # a draft may leave a bound open
    with pytest.raises(StudyError, match="numeric bound"):
        parse_study({**unset, **FROZEN})
    with pytest.raises(StudyError, match="placeholders"):
        parse_study({**_mutate(raw, "description", "REPLACE later"), **FROZEN})
    assert parse_study({**raw, **FROZEN}).frozen is True


# --- outcomes -----------------------------------------------------------------------------


def test_unfrozen_study_is_inconclusive_with_rule_outcomes_beside(tmp_path, raw, alerts) -> None:
    _, report = _report(tmp_path, raw, alerts)
    assert report["outcome"] == "inconclusive"
    assert report["evidence_class"] == "synthetic_fixture_not_a_measurement"
    assert report["human_decision"] == "pending" and report["candidates_meeting_criteria"] == []
    r = _by_id(report)
    assert r[THRESH]["outcome"] == "descriptive_only_not_preregistered"
    assert r[THRESH]["rule_outcome"] == "meets_declared_criteria_pending_human_decision"
    assert r[SOC]["rule_outcome"] == "constraint_violated"
    assert r[EDIT]["outcome"] == "prerequisite_missing"


def test_preregistered_study_applies_its_rules(tmp_path, raw, alerts) -> None:
    _, report = _report(tmp_path, {**raw, **FROZEN}, alerts)
    r = _by_id(report)
    assert r[THRESH]["outcome"] == "meets_declared_criteria_pending_human_decision"
    assert {c["status"] for c in r[THRESH]["constraints"]} == {"satisfied"}
    assert r[THRESH]["target_effect"]["improvement"]["point"] == 1.0
    assert r[SOC]["outcome"] == "constraint_violated"
    (errors,) = [c for c in r[SOC]["constraints"] if c["id"] == "no-new-errors"]
    assert errors["paired"]["net_losses"] == 2
    assert report["outcome"] == "candidates_meet_declared_criteria_pending_human_decision"
    assert report["candidates_meeting_criteria"] == [THRESH]


def test_study_frozen_after_the_evidence_run_is_descriptive(tmp_path, raw, alerts) -> None:
    _, report = _report(tmp_path, {**raw, **FROZEN, "frozen_at": "2999-01-01T00:00Z"}, alerts)
    r = _by_id(report)[THRESH]
    assert r["outcome"] == "descriptive_only_not_preregistered"
    assert "study frozen after the evidence run started" in r["reasons"]
    assert report["outcome"] == "inconclusive"


def test_missing_bound_or_effect_is_inconclusive(tmp_path, raw, alerts) -> None:
    draft = _mutate(raw, "constraints.1.bound", None)
    draft["target_failure"]["min_meaningful_effect"] = None
    _, report = _report(tmp_path, draft, alerts)
    r = _by_id(report)[THRESH]
    assert r["rule_outcome"] == "inconclusive"
    assert "min_meaningful_effect not declared" in r["reasons"]
    assert "constraint defer-budget not_declared" in r["reasons"]


def test_absolute_bound_must_hold_with_confidence(tmp_path, raw, alerts) -> None:
    # 16/60 deferred: the point estimate 0.267 is under 0.3, but the exact one-sided 95% upper
    # bound is above it, so the constraint is inconclusive, not satisfied.
    tight = {**_mutate(raw, "constraints.1.bound", 0.3), **FROZEN}
    _, report = _report(tmp_path, tight, alerts)
    r = _by_id(report)[THRESH]
    (defer,) = [c for c in r["constraints"] if c["id"] == "defer-budget"]
    assert defer["candidate"]["value"] < 0.3 < defer["confidence_bound"]["bound"]
    assert (defer["candidate"]["numerator"], defer["candidate"]["denominator"]) == (16, 60)
    assert defer["status"] == "inconclusive" and r["outcome"] == "inconclusive"


def test_no_evidence_means_insufficient_evidence(raw) -> None:
    study = parse_study({**raw, **FROZEN})
    report = build_report(study, [], ["run-x: records.jsonl does not match"])
    assert {r["outcome"] for r in report["candidates"]} == {"insufficient_evidence"}
    assert report["outcome"] == "inconclusive"
    assert report["evidence_errors"] == ["run-x: records.jsonl does not match"]
    assert "Evidence not used" in render_markdown(report)


def test_changed_baseline_or_mismatched_candidate_is_not_interpreted(tmp_path, raw, alerts) -> None:
    study = parse_study({**raw, **FROZEN})
    swapped = SyntheticTriageEvaluator(
        "x", select_at=0.8, defer_at=0.5, artifact_sha256="9" * 64
    )  # baseline artifact replaced
    _, report = _report(tmp_path, {**raw, **FROZEN}, alerts, name="b", baseline=swapped)
    assert {r["outcome"] for r in report["candidates"]} == {"baseline_changed_not_interpreted"}

    drifted = SyntheticTriageEvaluator(
        "x", select_at=0.8, defer_at=0.25, artifact_sha256=study.baseline["artifact"]["sha256"]
    )  # candidate config differs from its registration
    _, report = _report(tmp_path, {**raw, **FROZEN}, alerts, name="c", **{THRESH: drifted})
    r = _by_id(report)[THRESH]
    assert r["outcome"] == "candidate_provenance_mismatch"
    assert "loaded configuration differs from the registration" in r["reasons"]


def test_triage_model_calibrated_on_the_evaluation_set_is_refused(tmp_path, raw, alerts) -> None:
    bad = copy.deepcopy({**raw, **FROZEN})
    bad["candidates"][1]["intervention"]["calibration"]["dataset"] = bad["task"]["dataset"]
    _, report = _report(tmp_path, bad, alerts)
    assert _by_id(report)[SOC]["outcome"] == "prerequisite_missing"


# --- evidence integrity and failure handling ----------------------------------------------


def test_evaluators_never_see_labels_and_failures_are_rows(tmp_path, raw, alerts) -> None:
    seen = []

    class Spy(SyntheticTriageEvaluator):
        def decide(self, alert: Alert) -> TriageDecision:
            seen.append(alert)
            if alert.id == "syn-alert-005":
                return TriageDecision("escalate-to-ceo")  # not a valid decision
            return super().decide(alert)

    study = parse_study(raw)
    spy = Spy("spy", select_at=0.8, defer_at=0.5, artifact_sha256="1" * 64)
    _, run = _evidence(tmp_path, raw, alerts, baseline=spy)
    assert seen and all(not hasattr(a, "label") and "label" not in a.features for a in seen)
    ev = load_evidence(run, study)
    rows = {(r["arm"], r["item_id"]): r for r in ev.records}
    assert rows[("baseline", "syn-alert-005")]["error_class"] == "InvalidDecision"
    assert rows[(SOC, "syn-alert-031")]["error_class"] == "TimeoutError"
    assert len(ev.records) == 60 * 4


def test_tampered_foreign_or_aborted_evidence_is_refused(tmp_path, raw, alerts) -> None:
    study, run = _evidence(tmp_path, raw, alerts)
    other = parse_study(_mutate(raw, "description", "edited after the run"))
    with pytest.raises(EvidenceError, match="different study"):
        load_evidence(run, other)
    path = run / "records.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(EvidenceError, match="records_sha256"):
        load_evidence(run, study)

    class Stop(SyntheticTriageEvaluator):
        def decide(self, alert):
            if alert.id == "syn-alert-010":
                raise KeyboardInterrupt
            return super().decide(alert)

    stop = Stop("s", select_at=0.8, defer_at=0.5, artifact_sha256="1" * 64)
    with pytest.raises(KeyboardInterrupt):
        _evidence(tmp_path, raw, alerts, name="aborted", baseline=stop)
    manifest = json.loads((tmp_path / "aborted" / "manifest.json").read_text())
    assert manifest["run_status"] == "aborted"
    with pytest.raises(EvidenceError, match="run_status"):
        load_evidence(tmp_path / "aborted", study)


def test_evidence_is_bound_to_the_declared_dataset(tmp_path, raw, alerts) -> None:
    a, labels = alerts
    flipped = dict(labels, **{"syn-alert-001": "benign"})
    study = parse_study(raw)
    with pytest.raises(RunError, match="content_sha256"):
        run_triage_evaluation(
            study, a, flipped, synthetic_evaluators(study), tmp_path / "x", operator="op1"
        )
    with pytest.raises(RunError, match="inside the repository"):
        run_triage_evaluation(
            study, a, labels, synthetic_evaluators(study), ROOT / "out" / "imp-x", operator="op1"
        )
    assert not (ROOT / "out" / "imp-x").exists()


# --- metrics --------------------------------------------------------------------------------


def test_exact_bounds_and_paired_losses() -> None:
    assert clopper_pearson(0, 10, 0.95, "upper") == pytest.approx(1 - 0.05 ** (1 / 10), abs=1e-6)
    assert clopper_pearson(10, 10, 0.95, "lower") == pytest.approx(0.05 ** (1 / 10), abs=1e-6)
    assert clopper_pearson(0, 10, 0.95, "lower") == 0.0
    lo = clopper_pearson(5, 20, 0.95, "lower")
    hi = clopper_pearson(5, 20, 0.95, "upper")
    assert lo < 0.25 < hi
    base = {"a": (1.0, 1.0), "b": (0.0, 1.0), "c": (1.0, 1.0)}
    cand = {"a": (0.0, 1.0), "b": (1.0, 1.0), "c": (0.0, 1.0)}
    assert net_paired_losses(base, cand, ["a", "b", "c"], better=+1) == {
        "items": 3,
        "wins": 1,
        "losses": 2,
        "net_losses": 1,
    }
    with pytest.raises(ValueError, match="not pairable"):
        net_paired_losses({"a": (0.0, 1.0)}, {"a": (0.0, 0.0)}, ["a"], better=-1)


def test_triage_metric_denominators() -> None:
    rows = {
        "t-sup": [{"status": "decided", "decision": "suppress", "label": "threat"}],
        "t-def": [{"status": "decided", "decision": "defer", "label": "threat"}],
        "t-err": [{"status": "error", "decision": None, "label": "threat"}],
        "b-sel": [{"status": "decided", "decision": "select", "label": "benign"}],
        "u-sel": [{"status": "decided", "decision": "select", "label": None}],
    }
    miss = item_values("threat_miss_rate", rows)
    assert [miss[k] for k in ("t-sup", "t-def", "t-err", "b-sel")] == [
        (1.0, 1.0),
        (0.0, 1.0),  # deferred threats reach an analyst
        (1.0, 1.0),  # an errored decision on a threat counts as a miss
        (0.0, 0.0),
    ]
    bef = item_values("benign_escalation_fraction", rows)
    assert bef["b-sel"] == (1.0, 1.0) and bef["u-sel"] == (0.0, 0.0)  # unlabelled excluded


# --- human decision -------------------------------------------------------------------------


def test_decisions_are_human_named_and_never_deploy(tmp_path, raw, alerts) -> None:
    study, report = _report(tmp_path, {**raw, **FROZEN}, alerts)
    kw = {"decided_at": "2026-11-05T09:00Z", "rationale": "synthetic fixture review"}
    rec = make_decision(
        study,
        report,
        candidate=THRESH,
        decision="approve_further_evaluation",
        decided_by="reviewer1",
        **kw,
    )
    assert rec["deployment_authorized"] is False and rec["report_sha256"] == report["report_sha256"]
    for who in ("owner1", "op1"):
        with pytest.raises(ValueError, match="differ from the study owner"):
            make_decision(study, report, candidate=THRESH, decision="reject", decided_by=who, **kw)
    with pytest.raises(ValueError, match="cannot approve"):
        make_decision(
            study,
            report,
            candidate=SOC,
            decision="approve_further_evaluation",
            decided_by="reviewer1",
            **kw,
        )
    assert validate_decision({**rec, "deployment_authorized": True}, study, report)
    assert validate_decision({**rec, "decision": "deploy"}, study, report)
    assert validate_decision({**rec, "report_sha256": "0" * 64}, study, report)
    edited = {**report, "outcome": "inconclusive", "candidates_meeting_criteria": []}
    assert any("edited after build" in p for p in validate_decision(rec, study, edited))


# --- WSP harness as the weight-edit evidence path -------------------------------------------


# Shape of `wsp.py verify-edit` output (checkpoint_diff.verify_edit), synthetic digests.
EDIT_REPORT = {
    "status": "verified",
    "problems": [],
    "tensors_checked": ["w"],
    "metadata_equal": True,
    "base_file_sha256": "0" * 64,
    "candidate_file_sha256": "9" * 64,
    "manifest_sha256": "8" * 64,  # the registered edit_manifest_sha256
}
ARM_MAP = {"reference": "baseline", "candidate": "edit-1"}


def _wsp_setup(tmp_path, split="heldout"):
    raw_p = json.loads((WSP / "fixtures" / "synthetic_protocol.json").read_text())
    raw_p["analysis"]["required_limitations"] = ["WSP limitation carried into the report"]
    if split == "heldout":  # a held-out WSP run needs a frozen protocol and a named evaluator
        raw_p.update(frozen=True, frozen_by="lead1", frozen_at="2026-11-01T00:00Z")
    proto = parse_protocol(raw_p)
    items = load_items(WSP / "fixtures" / "synthetic_items.jsonl")
    gens = synthetic_generators(items, proto.seeds)
    g = gens["candidate"]
    edited = {"model.safetensors": "9" * 64, "tokenizer.json": "1" * 64}
    gens["candidate"] = SyntheticGenerator(
        g.name, g.table, load_id=g.load_id, artifacts_sha256=edited
    )
    res = run_paired(
        proto,
        items,
        gens,
        FinalAnswerGrader(),
        tmp_path / "wsp",
        split=split,
        heldout_authorized_by="ev1" if split == "heldout" else None,
        purpose="t",
        operator="op1",
        clock=iter(range(10_000)).__next__,
        run_id="wsp-imp",
    )
    tags = {i.id: ["dev"] for i in items}
    run_items = [r["item_id"] for r in res.records]
    arm = proto.arm("reference")
    config = {"inference": arm.inference, "engine": arm.engine}
    data_sha = res.manifest["dataset"]["items_file_sha256"]
    study_raw = {
        "study_version": "synthetic-weight-edit-0.1",
        "synthetic": True,
        **FROZEN,
        "owner": "owner1",
        "operator": "op1",
        "task": {
            "kind": "accuracy",
            "dataset": {
                "name": "wsp-synthetic",
                "revision": "fixture-1",
                "content_sha256": data_sha,
                "expected_items_sha256": item_set_sha256(run_items),
            },
        },
        "baseline": {
            "id": "baseline",
            "status": "unchanged",
            "artifact": {"name": "synthetic-base", "sha256": "0" * 64, "file": "model.safetensors"},
            "config": config,
        },
        "target_failure": {
            "id": "looping",
            "description": "reference loops",
            "slice_tag": "dev",
            "metric": "accuracy",
            "min_meaningful_effect": 0.1,
            "bound_source": "synthetic_fixture",
            "slice_membership_sha256": slice_membership_sha256(run_items, tags),
        },
        "candidates": [
            {
                "id": "edit-1",
                "hypothesis": "synthetic edit",
                "intervention": {
                    "type": "weight_edit",
                    "base_artifact_sha256": "0" * 64,
                    "edit_manifest_sha256": "8" * 64,
                    "edit_verification": {
                        "status": "verified",
                        "report_sha256": sha256_json(EDIT_REPORT),
                    },
                    "evaluation": {"path": "wsp", "protocol_sha256": proto.sha256},
                },
                "artifact": {
                    "name": "synthetic-edited",
                    "sha256": "9" * 64,
                    "file": "model.safetensors",
                },
                "config": config,
            }
        ],
        "constraints": [
            {
                "id": "errors",
                "metric": "error_rate",
                "kind": "max_net_paired_losses",
                "max_losses": 1,
                "bound_source": "synthetic_fixture",
            }
        ],
        "analysis": {
            "confidence": 0.95,
            "bootstrap_resamples": 500,
            "bootstrap_seed": 1,
            "samples_per_item": proto.samples_per_item,
        },
    }
    return proto, res, study_raw, tags


def test_wsp_run_is_the_weight_edit_evidence_path(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    study = parse_study(study_raw)
    ev = evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags, edit_report=EDIT_REPORT)
    check = ev.notes["edit_verification"]
    assert check["status"] == "hash_consistent"
    assert "not authenticated" in check["meaning"] and "re-verification" in check["meaning"]
    report = build_report(study, [ev])
    (r,) = report["candidates"]
    assert r["evidence_run"]["source"] == "wsp"
    assert r["target_effect"]["improvement"]["point"] > 0
    assert "WSP limitation carried into the report" in report["limitations"]
    run = report["evidence_runs"][0]
    assert run["wsp_protocol_sha256"] == proto.sha256 and run["wsp_split"] == "heldout"
    assert run["wsp_protocol_preregistered"] is True
    assert report["evidence_class"] == "synthetic_fixture_not_a_measurement"


def test_weight_edit_needs_a_matching_verify_edit_report(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    study = parse_study(study_raw)
    # Declared "verified" is not enough: without the report the comparison stops.
    ev = evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags)
    (r,) = build_report(study, [ev])["candidates"]
    assert r["outcome"] == "prerequisite_missing"
    assert "hash-consistent" in r["reasons"][0] and "not authenticated" in r["reasons"][0]
    for bad, match in (
        ({**EDIT_REPORT, "tensors_checked": ["other"]}, "report_sha256"),
        ({**EDIT_REPORT, "status": "failed"}, "report_sha256"),
    ):
        with pytest.raises(EvidenceError, match=match):
            evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags, edit_report=bad)
    # A report for other files, even if registered by hash, is refused.
    other = {**EDIT_REPORT, "candidate_file_sha256": "8" * 64}
    raw2 = copy.deepcopy(study_raw)
    raw2["candidates"][0]["intervention"]["edit_verification"]["report_sha256"] = sha256_json(other)
    with pytest.raises(EvidenceError, match="candidate file"):
        evidence_from_wsp(res.out_dir, proto, parse_study(raw2), ARM_MAP, tags, edit_report=other)
    failed = {**EDIT_REPORT, "status": "failed"}
    raw3 = copy.deepcopy(study_raw)
    raw3["candidates"][0]["intervention"]["edit_verification"]["report_sha256"] = sha256_json(
        failed
    )
    with pytest.raises(EvidenceError, match="status"):
        evidence_from_wsp(res.out_dir, proto, parse_study(raw3), ARM_MAP, tags, edit_report=failed)


def test_wsp_evidence_is_tied_to_protocol_dataset_arms_and_policy(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    study = parse_study(study_raw)
    wrong = copy.deepcopy(study_raw)
    wrong["candidates"][0]["intervention"]["evaluation"]["protocol_sha256"] = "7" * 64
    with pytest.raises(EvidenceError, match="protocol_sha256"):
        evidence_from_wsp(res.out_dir, proto, parse_study(wrong), ARM_MAP, tags)
    other_data = copy.deepcopy(study_raw)
    other_data["task"]["dataset"]["content_sha256"] = "6" * 64
    with pytest.raises(EvidenceError, match="items hash"):
        evidence_from_wsp(res.out_dir, proto, parse_study(other_data), ARM_MAP, tags)
    # The no-edit reload is not the baseline, and the reference is not the candidate.
    with pytest.raises(EvidenceError, match="reference arm"):
        evidence_from_wsp(
            res.out_dir, proto, study, {"reload": "baseline", "candidate": "edit-1"}, tags
        )
    with pytest.raises(EvidenceError, match="role 'candidate'"):
        evidence_from_wsp(
            res.out_dir, proto, study, {"reference": "baseline", "reload": "edit-1"}, tags
        )
    excl = dataclasses.replace(proto, analysis={**proto.analysis, "missing_policy": "exclude_item"})
    with pytest.raises(EvidenceError, match="incorrect"):
        evidence_from_wsp(res.out_dir, excl, study, ARM_MAP, tags)
    path = res.out_dir / "records.jsonl"
    path.write_text(path.read_text() + "\n{}\n")
    with pytest.raises(EvidenceError, match="verification"):
        evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags)


def test_baseline_digest_is_checked_for_the_named_file(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    # Registering the tokenizer digest as "the model" must not pass the baseline check.
    raw2 = copy.deepcopy(study_raw)
    raw2["baseline"]["artifact"]["sha256"] = "1" * 64
    for c in raw2["candidates"]:
        c["intervention"]["base_artifact_sha256"] = "1" * 64
    s2 = parse_study(raw2)
    ev = evidence_from_wsp(res.out_dir, proto, s2, ARM_MAP, tags)
    (r,) = build_report(s2, [ev])["candidates"]
    assert r["outcome"] == "baseline_changed_not_interpreted"
    assert "'model.safetensors' digest differs" in r["reasons"][0]
    raw3 = copy.deepcopy(study_raw)
    raw3["candidates"][0]["intervention"]["base_artifact_sha256"] = "2" * 64
    with pytest.raises(StudyError, match="registered baseline digest"):
        parse_study(raw3)


def test_wsp_config_hash_matches_registration(tmp_path) -> None:
    proto, _, study_raw, _ = _wsp_setup(tmp_path)
    arm = proto.arm("candidate")
    assert sha256_json(study_raw["candidates"][0]["config"]) == sha256_json(
        {"inference": arm.inference, "engine": arm.engine}
    )


# --- CLI and isolation ----------------------------------------------------------------------


def test_cli_demo_report_and_decide(tmp_path, capsys, raw) -> None:
    assert improve.main(["demo", "--output", str(tmp_path / "demo")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["outcome"] == "inconclusive"
    md = (tmp_path / "demo" / "report" / "report.md").read_text()
    assert "synthetic_fixture_not_a_measurement" in md and "human decision: **pending**" in md
    assert "not a Tantular Cyber baseline" in md

    study_path = FIX / "synthetic_study.json"
    argv = [
        "report",
        "--study",
        str(study_path),
        "--output",
        str(tmp_path / "r2"),
        "--evidence",
        str(tmp_path / "demo" / "evidence"),
        str(tmp_path / "missing"),
    ]
    assert improve.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["evidence_errors"]) == 1  # a missing run is reported, not silently skipped

    dec = tmp_path / "decision.json"
    argv = [
        "decide",
        "--study",
        str(study_path),
        "--report",
        str(tmp_path / "r2" / "report.json"),
        "--decision",
        "request_more_evidence",
        "--decided-by",
        "reviewer1",
        "--rationale",
        "study not frozen",
        "--output",
        str(dec),
    ]
    assert improve.main(argv) == 0
    capsys.readouterr()
    assert json.loads(dec.read_text())["deployment_authorized"] is False
    assert improve.main(argv) == 1  # never overwritten
    bad = [*argv[:-1], str(tmp_path / "d2.json")]
    bad[bad.index("reviewer1")] = "op1"
    assert improve.main(bad) == 1  # operator cannot decide
    assert improve.main(["validate", str(study_path)]) == 0


_FORBIDDEN = {
    "tantular_cyber",
    "socket",
    "http",
    "urllib",
    "httpx",
    "requests",
    "subprocess",
    "pickle",
    "torch",
    "ssl",
    "ftplib",
    "smtplib",
    "sklearn",
    "joblib",
    "numpy",
    "pandas",
}


def test_workflow_is_isolated_and_has_no_promotion_path() -> None:
    files = sorted(RESEARCH.rglob("*.py"))
    assert files
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in _FORBIDDEN, f"{path.name} imports {n}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "__import__"}, path.name
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                lowered = node.name.lower()
                assert not any(w in lowered for w in ("promote", "deploy", "publish", "upload"))
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "tools/research" not in pyproject and "improve_eval" not in pyproject


# --- review pass 2026-11-04: denominators, intervals, labels, decisions ---------------------


def test_breakdown_reconstructs_every_metric_and_errors_count_as_deferred(tmp_path, raw, alerts):
    _, report = _report(tmp_path, {**raw, **FROZEN}, alerts)
    r = _by_id(report)[SOC]  # errors on two benign alerts the baseline selects
    b = r["breakdown"][SOC]
    assert sum(b.values()) == 60 and b["error/benign"] == 2
    cons = {c["id"]: c for c in r["constraints"]}
    defer = cons["defer-budget"]["candidate"]
    assert defer["numerator"] == b["defer/threat"] + b["defer/benign"] + b[
        "defer/unlabelled"
    ] + sum(v for k, v in b.items() if k.startswith("error/"))
    bef = cons["benign-escalation-cap"]["candidate"]
    assert bef["denominator"] == b["select/threat"] + b["select/benign"]  # unlabelled excluded
    assert bef["numerator"] == b["select/benign"]


def test_empty_selection_is_undefined_not_zero(tmp_path, raw, alerts) -> None:
    never = SyntheticTriageEvaluator(
        "never", select_at=2.0, defer_at=0.3, artifact_sha256="1" * 64
    )  # selects nothing; registered config would differ, so patch the registration
    raw2 = copy.deepcopy({**raw, **FROZEN})
    raw2["candidates"][0]["config"] = never.config
    _, report = _report(tmp_path, raw2, alerts, **{THRESH: never})
    r = _by_id(report)[THRESH]
    (bef,) = [c for c in r["constraints"] if c["id"] == "benign-escalation-cap"]
    assert bef["candidate"]["denominator"] == 0 and bef["candidate"]["value"] is None
    assert bef["status"] == "not_evaluated"  # never treated as satisfied
    assert r["outcome"] != "meets_declared_criteria_pending_human_decision"


def test_intervals_do_not_overstate_certainty() -> None:
    base = {f"i{n}": (1.0, 1.0) for n in range(12)}  # 12 threats all missed
    cand = {f"i{n}": (0.0, 1.0) for n in range(12)}  # all caught
    keys = sorted(base)
    imp = paired_improvement(base, cand, keys, better=-1, confidence=0.95, single_sample=True)
    assert imp["point"] == 1.0 and 0 < imp["ci"][0] < 1.0  # not a zero-width [1, 1]
    # The same flips grouped into 6 campaigns: 6 independent units cannot show improvement.
    clusters = {k: f"c{n // 2}" for n, k in enumerate(keys)}
    imp_c = paired_improvement(base, cand, keys, better=-1, confidence=0.95, clusters=clusters)
    assert imp_c["units"] == 6 and imp_c["ci"][0] < 0 < imp_c["ci"][1]
    # A zero-width bootstrap is reported as not estimable.
    sel = {f"i{n}": (0.0, 1.0) for n in range(12)}
    deg = paired_difference_bootstrap(sel, sel, keys, -1, 0.95, 200, 1)
    assert deg["ci"] == [None, None] and deg["method"] == "percentile_bootstrap_degenerate"
    assert deg["status"] == "approximate"
    # Clustered absolute bounds are never tighter than the independent-alert bound.
    vals = {f"a{n}": (float(n < 3), 1.0) for n in range(30)}
    exact = one_sided_bound(vals, sorted(vals), "upper", 0.95, single_sample=True)
    grouped = one_sided_bound(
        vals, sorted(vals), "upper", 0.95, {k: f"g{n // 3}" for n, k in enumerate(sorted(vals))}
    )
    assert grouped["units"] == 10 and grouped["bound"] >= exact["bound"]


def test_cluster_unit_needs_cluster_ids_and_widens_the_claim(tmp_path, raw, alerts) -> None:
    clustered = {**raw, **FROZEN, "analysis": {**raw["analysis"], "unit": "cluster"}}
    _, report = _report(tmp_path, clustered, alerts)
    r = _by_id(report)[THRESH]
    # Only alerts with a denominator count: the 12 target threats form 6 campaigns.
    assert r["unit"] == "cluster" and r["target_effect"]["improvement"]["units"] == 6
    assert r["outcome"] == "descriptive_only_cluster_method_unreviewed"
    assert r["target_effect"]["improvement"]["status"] == "cluster_method_unreviewed"
    assert "target improvement interval includes zero" in r["reasons"]
    a, labels = alerts
    no_ids = [Alert(x.id, x.tags, x.features, None) for x in a]
    raw_nc = copy.deepcopy(clustered)
    raw_nc["task"]["dataset"]["content_sha256"] = alerts_sha256(no_ids, labels)
    raw_nc["task"]["dataset"]["metadata_sha256"] = alerts_metadata_sha256(no_ids, labels)
    study = parse_study(raw_nc)
    run = run_triage_evaluation(
        study, no_ids, labels, synthetic_evaluators(study), tmp_path / "nc", operator="op1"
    )
    rep = build_report(study, [load_evidence(run, study)])
    out = _by_id(rep)
    assert out[THRESH]["outcome"] == out[SOC]["outcome"] == "insufficient_evidence"
    assert "lack cluster ids" in out[THRESH]["reasons"][0]
    # The boundary and cluster checks run before prerequisites, so every candidate stops there.
    assert out[EDIT]["outcome"] == "insufficient_evidence"


def test_frozen_triage_study_must_cap_escalations_and_deferrals(raw) -> None:
    no_cap = copy.deepcopy({**raw, **FROZEN})
    no_cap["constraints"] = [k for k in no_cap["constraints"] if k["id"] != "defer-budget"]
    with pytest.raises(StudyError, match="defer_rate"):
        parse_study(no_cap)
    draft = copy.deepcopy(raw)
    draft["constraints"] = [k for k in draft["constraints"] if k["id"] != "defer-budget"]
    parse_study(draft)  # a draft may still be incomplete
    with pytest.raises(StudyError, match="validity_scope"):
        parse_study(_mutate(raw, "task.validity_scope", KeyError))


def test_invented_limits_are_labelled_synthetic(tmp_path, raw, alerts) -> None:
    with pytest.raises(StudyError, match="synthetic_fixture"):
        parse_study(_mutate(raw, "constraints.0.bound_source", "declared_by_owner"))
    with pytest.raises(StudyError, match="cannot use synthetic_fixture"):
        parse_study({**raw, "synthetic": False})
    with pytest.raises(StudyError, match="bound_source"):
        parse_study(_mutate(raw, "target_failure.bound_source", KeyError))
    _, report = _report(tmp_path, raw, alerts)
    md = render_markdown(report)
    assert md.count("*(synthetic fixture value)*") == 1 + 5 * 2  # target + 5 constraints x 2 arms
    assert "no multiplicity control across candidates" in md
    assert "treat alerts as independent" in md and "Validity scope:" in md
    assert "recorded, not authenticated" in md


def test_frozen_study_names_files_and_verify_edit_reports(raw) -> None:
    with pytest.raises(StudyError, match="names the artifact file"):
        parse_study({**_mutate(raw, "baseline.artifact.file", KeyError), **FROZEN})
    verified = _mutate(raw, "candidates.2.intervention.edit_verification", {"status": "verified"})
    parse_study(verified)
    with pytest.raises(StudyError, match="report_sha256"):
        parse_study({**verified, **FROZEN})


def test_decision_records_say_they_are_unauthenticated_and_unenforced(tmp_path, raw, alerts):
    study, report = _report(tmp_path, {**raw, **FROZEN}, alerts)
    rec = make_decision(
        study,
        report,
        candidate=None,
        decision="request_more_evidence",
        decided_by="reviewer1",
        decided_at="2026-11-05T09:00Z",
        rationale="synthetic",
    )
    assert rec["identity_verified"] is False and rec["enforced_by"] == "none"
    assert validate_decision({**rec, "identity_verified": True}, study, report)
    assert validate_decision({**rec, "enforced_by": "release-gate"}, study, report)


# --- independent-review fixes: estimands, binding methods, clusters, min effect --------------


def test_binary_items_use_the_bonferroni_interval_on_the_same_estimand() -> None:
    base = {"a": (1.0, 1.0), "b": (0.0, 1.0), "c": (1.0, 1.0), "d": (0.0, 1.0)}
    cand = {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (1.0, 1.0), "d": (1.0, 1.0)}
    imp = paired_improvement(
        base, cand, sorted(base), better=+1, confidence=0.95, single_sample=True
    )
    assert imp["estimand"] == "mean_paired_difference_per_item"
    assert imp["status"] == "valid_by_bonferroni"
    # Point and interval target the same quantity: the mean per-item difference, which for
    # single-sample binary items equals the difference of pooled ratios.
    assert imp["point"] == (2 - 2) / 4 == 0.0
    assert imp["ci"][0] <= imp["point"] <= imp["ci"][1]


def test_fractional_items_are_an_unreviewed_heuristic() -> None:
    base = {"a": (0.5, 1.0), "b": (0.25, 1.0)}
    cand = {"a": (1.0, 1.0), "b": (0.75, 1.0)}
    imp = paired_improvement(base, cand, ["a", "b"], better=+1, confidence=0.95)
    assert imp["status"] == "unreviewed_heuristic" and imp["method"].startswith("rounded_")
    b = one_sided_bound(base, ["a", "b"], "upper", 0.95)
    assert b["status"] == "unreviewed_heuristic" and b["estimand"] == "pooled_ratio"


def test_unequal_cluster_sizes_are_descriptive_only() -> None:
    sizes = [1, 5, 10, 2]
    keys, clusters = [], {}
    for c, n in enumerate(sizes):
        for j in range(n):
            keys.append(f"c{c}-{j}")
            clusters[f"c{c}-{j}"] = f"c{c}"
    base = {k: (1.0, 1.0) for k in keys}
    cand = {k: (0.0, 1.0) for k in keys}
    imp = paired_improvement(base, cand, keys, better=-1, confidence=0.95, clusters=clusters)
    assert imp["units"] == 4 and imp["status"] == "cluster_method_unreviewed"
    b = one_sided_bound(cand, keys, "upper", 0.95, clusters)
    assert b["units"] == 4 and b["status"] == "cluster_method_unreviewed"
    boot = paired_difference_bootstrap(base, cand, keys, -1, 0.95, 200, 1, clusters)
    assert boot["status"] == "cluster_method_unreviewed"


def test_cluster_and_item_weighted_effects_can_disagree() -> None:
    # One large cluster of 9 items gets worse; four single-item clusters improve.
    keys = [f"big-{j}" for j in range(9)] + [f"solo-{j}" for j in range(4)]
    clusters = {k: ("big" if k.startswith("big") else k) for k in keys}
    base = {k: (1.0 if k.startswith("big") else 0.0, 1.0) for k in keys}
    cand = {k: (0.0 if k.startswith("big") else 1.0, 1.0) for k in keys}
    imp = paired_improvement(base, cand, keys, better=+1, confidence=0.95, clusters=clusters)
    assert imp["estimand"] == "mean_paired_difference_per_cluster"
    assert imp["point"] == round((-1 + 4) / 5, 6) == 0.6  # cluster-weighted: better
    assert imp["item_weighted_point"] == round((-9 + 4) / 13, 6)  # item-weighted: worse
    item = paired_improvement(base, cand, keys, better=+1, confidence=0.95, single_sample=True)
    assert item["point"] == imp["item_weighted_point"] and item["status"] == "valid_by_bonferroni"


def test_cluster_studies_never_meet_even_with_strong_effects(tmp_path, raw, alerts) -> None:
    # Re-cluster so every target threat is its own cluster: item- and cluster-level coincide,
    # yet the cluster study must stay descriptive.
    a, labels = alerts
    solo = [Alert(x.id, x.tags, x.features, x.id) for x in a]
    raw_c = copy.deepcopy({**raw, **FROZEN, "analysis": {**raw["analysis"], "unit": "cluster"}})
    raw_c["task"]["dataset"]["content_sha256"] = alerts_sha256(solo, labels)
    raw_c["task"]["dataset"]["metadata_sha256"] = alerts_metadata_sha256(solo, labels)
    study = parse_study(raw_c)
    run = run_triage_evaluation(
        study, solo, labels, synthetic_evaluators(study), tmp_path / "solo", operator="op1"
    )
    report = build_report(study, [load_evidence(run, study)])
    r = _by_id(report)[THRESH]
    assert r["rule_outcome"] == "inconclusive"  # cluster bounds are not binding
    assert r["outcome"] == "descriptive_only_cluster_method_unreviewed"
    assert all(
        c["status"] in ("descriptive_only", "violated")
        for c in r["constraints"]
        if c["kind"] != "max_net_paired_losses"
    )
    assert report["outcome"] == "inconclusive" and report["candidates_meeting_criteria"] == []
    assert any("Unit of analysis is cluster" in x for x in report["limitations"])


def test_min_effect_is_a_lower_bound_criterion(tmp_path, raw, alerts) -> None:
    _, report = _report(tmp_path, {**raw, **FROZEN}, alerts, name="probe")
    lo = _by_id(report)[THRESH]["target_effect"]["improvement"]["ci"][0]
    point = _by_id(report)[THRESH]["target_effect"]["improvement"]["point"]
    assert point == 1.0 and 0 < lo < 1

    at = copy.deepcopy({**raw, **FROZEN})
    at["target_failure"]["min_meaningful_effect"] = lo  # boundary: lower bound == minimum
    _, rep_at = _report(tmp_path, at, alerts, name="at")
    assert _by_id(rep_at)[THRESH]["outcome"] == "meets_declared_criteria_pending_human_decision"

    above = copy.deepcopy(at)
    above["target_failure"]["min_meaningful_effect"] = round(lo + 1e-6, 6)
    _, rep_above = _report(tmp_path, above, alerts, name="above")
    r = _by_id(rep_above)[THRESH]
    # The point estimate (1.0) is far above the minimum, but the bound is not: not enough.
    assert r["outcome"] == "inconclusive"
    assert "lower bound of the target improvement is below min_meaningful_effect" in r["reasons"]


def test_multi_sample_wsp_items_cannot_meet_criteria(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)  # 2 samples per item: fractional
    study = parse_study(study_raw)
    ev = evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags, edit_report=EDIT_REPORT)
    (r,) = build_report(study, [ev])["candidates"]
    assert r["target_effect"]["improvement"]["status"] == "unreviewed_heuristic"
    assert r["rule_outcome"] in ("inconclusive", "constraint_violated")
    assert any("descriptive only" in x for x in r["reasons"])


# --- second review round: fail-closed linkage, capacity, observed vs established ------------


def test_wsp_slice_membership_must_be_bound_to_the_study(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    study = parse_study(study_raw)
    in_run = res.records[0]["item_id"]
    moved = {**tags, in_run: []}  # one evaluated item silently dropped from the slice
    with pytest.raises(EvidenceError, match="slice membership"):
        evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, moved, edit_report=EDIT_REPORT)
    unbound = copy.deepcopy(study_raw)
    del unbound["target_failure"]["slice_membership_sha256"]
    with pytest.raises(StudyError, match="slice_membership_sha256"):
        parse_study(unbound)  # a frozen accuracy study must bind it
    draft = {**unbound, "frozen": False}
    draft.pop("frozen_by"), draft.pop("frozen_at")
    with pytest.raises(EvidenceError, match="slice membership"):
        evidence_from_wsp(res.out_dir, proto, parse_study(draft), ARM_MAP, tags)


def test_verify_edit_report_must_name_the_registered_manifest(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    for report, match in (
        ({k: v for k, v in EDIT_REPORT.items() if k != "manifest_sha256"}, "no manifest_sha256"),
        ({**EDIT_REPORT, "manifest_sha256": "7" * 64}, "different edit manifest"),
    ):
        raw2 = copy.deepcopy(study_raw)
        raw2["candidates"][0]["intervention"]["edit_verification"]["report_sha256"] = sha256_json(
            report
        )
        with pytest.raises(EvidenceError, match=match):
            evidence_from_wsp(
                res.out_dir, proto, parse_study(raw2), ARM_MAP, tags, edit_report=report
            )


def test_development_split_wsp_evidence_cannot_be_confirmatory(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path, split="development")
    study = parse_study(study_raw)
    ev = evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags, edit_report=EDIT_REPORT)
    (r,) = build_report(study, [ev])["candidates"]
    assert r["outcome"] == "prerequisite_missing" and "held-out" in r["reasons"][0]


def test_calibration_identity_and_hash_are_checked(tmp_path, raw, alerts) -> None:
    frozen = copy.deepcopy({**raw, **FROZEN})
    cal = frozen["candidates"][1]["intervention"]["calibration"]["dataset"]
    eval_ds = frozen["task"]["dataset"]
    cases = [
        ({**cal, "content_sha256": eval_ds["content_sha256"]}, "content hash"),
        ({**cal, "name": eval_ds["name"], "revision": eval_ds["revision"]}, "name and revision"),
    ]
    for n, (bad, reason) in enumerate(cases):
        study_raw = copy.deepcopy(frozen)
        study_raw["candidates"][1]["intervention"]["calibration"]["dataset"] = bad
        _, report = _report(tmp_path, study_raw, alerts, name=f"cal{n}")
        r = _by_id(report)[SOC]
        assert r["outcome"] == "prerequisite_missing" and reason in r["reasons"][0]
    with pytest.raises(StudyError, match="calibration content_sha256"):
        no_hash = copy.deepcopy(frozen)
        del no_hash["candidates"][1]["intervention"]["calibration"]["dataset"]["content_sha256"]
        parse_study(no_hash)
    draft = copy.deepcopy(raw)
    del draft["candidates"][1]["intervention"]["calibration"]["dataset"]["content_sha256"]
    _, report = _report(tmp_path, draft, alerts, name="calnone")
    assert _by_id(report)[SOC]["outcome"] == "prerequisite_missing"  # fail closed


def test_escalation_volume_is_a_separate_required_capacity_cap(tmp_path, raw, alerts) -> None:
    no_cap = copy.deepcopy({**raw, **FROZEN})
    no_cap["constraints"] = [k for k in no_cap["constraints"] if k["metric"] != "escalation_rate"]
    with pytest.raises(StudyError, match="escalation_rate"):
        parse_study(no_cap)
    # An arm that escalates far more at the same precision is caught by the capacity cap.
    loud = SyntheticTriageEvaluator("loud", select_at=0.0, defer_at=0.0, artifact_sha256="1" * 64)
    study_raw = copy.deepcopy({**raw, **FROZEN})
    study_raw["candidates"][0]["config"] = loud.config
    _, report = _report(tmp_path, study_raw, alerts, **{THRESH: loud})
    r = _by_id(report)[THRESH]
    (cap,) = [c for c in r["constraints"] if c["id"] == "escalation-capacity"]
    assert cap["candidate"]["numerator"] == 60 and cap["status"] == "violated"
    assert r["outcome"] == "constraint_violated"


def test_observed_point_is_not_an_established_violation() -> None:
    # 28/60 = 0.467 > 0.45, but the lower bound is far below 0.45: not established.
    vals = {f"a{n}": (float(n < 28), 1.0) for n in range(60)}
    from improve_eval.compare import _absolute

    out = {"candidate": {"value": 28 / 60}}
    _absolute(out, {"op": "<=", "bound": 0.45}, vals, sorted(vals), 0.95, None, True)
    assert out["status"] == "point_beyond_limit"
    assert out["opposite_bound"]["bound"] < 0.45 < out["candidate"]["value"]
    # 50/60: the lower bound itself exceeds 0.45, so the violation is established.
    vals2 = {f"a{n}": (float(n < 50), 1.0) for n in range(60)}
    out2 = {"candidate": {"value": 50 / 60}}
    _absolute(out2, {"op": "<=", "bound": 0.45}, vals2, sorted(vals2), 0.95, None, True)
    assert out2["status"] == "violated" and out2["opposite_bound"]["bound"] > 0.45


def test_point_beyond_limit_keeps_the_outcome_inconclusive(tmp_path, raw, alerts) -> None:
    tight = copy.deepcopy({**raw, **FROZEN})
    for k in tight["constraints"]:
        if k["id"] == "defer-budget":
            k["bound"] = 0.25  # candidate defers 16/60 = 0.267: just beyond, not established
    _, report = _report(tmp_path, tight, alerts)
    r = _by_id(report)[THRESH]
    (defer,) = [c for c in r["constraints"] if c["id"] == "defer-budget"]
    assert defer["status"] == "point_beyond_limit"
    assert r["outcome"] == "inconclusive"
    assert any("violation not statistically established" in x for x in r["reasons"])


def test_unbalanced_pairs_always_make_the_outcome_inconclusive(tmp_path, raw, alerts) -> None:
    with pytest.raises(StudyError, match="not supported"):
        parse_study(
            {**raw, "analysis": {**raw["analysis"], "unbalanced_policy": "exclude_listed_items"}}
        )
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    drop = next(x for x in ev.records if x["arm"] == THRESH and "lateral-movement" not in x["tags"])
    partial = dataclasses.replace(ev, records=tuple(x for x in ev.records if x is not drop))
    r = _by_id(build_report(study, [partial]))[THRESH]
    assert r["items"]["unbalanced_items"] == [drop["item_id"]]
    # With a design problem the run is not single-sample, so no method is binding either.
    assert r["items"]["single_sample_design"] is False
    assert "rule_outcome_before_design_check" not in r  # dropped: it was misleading
    assert r["outcome"] == "inconclusive"
    assert any("different samples in the two arms" in x for x in r["reasons"])


def test_gross_loss_counting_does_not_let_fixes_offset_new_errors() -> None:
    base = {"a": (0.0, 1.0), "b": (1.0, 1.0)}  # error_rate: b errored on the baseline
    cand = {"a": (1.0, 1.0), "b": (0.0, 1.0)}  # a new error, b fixed
    p = net_paired_losses(base, cand, ["a", "b"], better=-1)
    assert p["net_losses"] == 0 and p["losses"] == 1  # net hides the new error; gross does not


def test_conflicting_cluster_ids_fail_closed(tmp_path, raw, alerts) -> None:
    clustered = {**raw, **FROZEN, "analysis": {**raw["analysis"], "unit": "cluster"}}
    study, run = _evidence(tmp_path, clustered, alerts)
    ev = load_evidence(run, study)
    first = ev.records[0]
    bad = tuple(dict(x, cluster="other") if x is first else x for x in ev.records)
    r = _by_id(build_report(study, [dataclasses.replace(ev, records=bad)]))[THRESH]
    # Caught at the comparison boundary: one item's cluster differs between the arms.
    assert r["outcome"] == "insufficient_evidence"
    assert "differ between records" in r["reasons"][0]


def test_bounds_are_rounded_against_the_claim() -> None:
    from improve_eval.metrics import outward

    assert outward(0.4500004, "upper") == 0.450001
    assert outward(0.4500004, "lower") == 0.45
    assert outward(0.45, "upper") == 0.45


# --- third round: regression tests for the second review's bypasses -----------------------


def _rewrite_records(run: Path, records: list[dict]) -> None:
    """Simulate someone regenerating records.jsonl and its manifest hash and count together."""
    path = run / "records.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["records_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["expected_records"] = len(records)
    (run / "manifest.json").write_text(json.dumps(manifest))


def test_accuracy_evidence_cannot_bypass_slice_binding(tmp_path) -> None:
    # Review finding 1: a frozen accuracy study reached "meets" with a true effect of 0 when
    # tags were chosen after outcomes and the evidence came through load_evidence.
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    study = parse_study(study_raw)
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "records.jsonl").write_text("")
    (fake / "manifest.json").write_text(
        json.dumps(
            {"source": "synthetic_triage", "task_kind": "accuracy", "study_sha256": study.sha256}
        )
    )
    with pytest.raises(EvidenceError, match="alert_triage"):
        load_evidence(fake, study)
    # A directly built Evidence that is not WSP-sourced is not interpreted either. (Directly
    # built WSP-sourced evidence is checked at the comparison boundary; see the tests below.)
    ev = evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags, edit_report=EDIT_REPORT)
    retagged = [dict(r, tags=["dev"]) for r in ev.records]  # post-hoc tags
    forged = dataclasses.replace(ev, source="synthetic_triage", records=tuple(retagged), notes={})
    (r,) = build_report(study, [forged])["candidates"]
    assert r["outcome"] == "insufficient_evidence"
    assert "WSP-sourced" in r["reasons"][0]


def test_arm_dependent_missingness_cannot_reach_meets(tmp_path, raw, alerts) -> None:
    # Review finding 2: a worse candidate whose failures leave no record used to pass under
    # exclude_listed_items. Now the missing records make the outcome inconclusive.
    worse = SyntheticTriageEvaluator("worse", select_at=0.8, defer_at=0.3, artifact_sha256="1" * 64)
    study_raw = copy.deepcopy({**raw, **FROZEN})
    study_raw["candidates"][0]["config"] = worse.config
    study, run = _evidence(tmp_path, study_raw, alerts, **{THRESH: worse})
    ev = load_evidence(run, study)
    # Drop the candidate's records on some benign alerts it would escalate or defer: its
    # failures are made invisible.
    hide = {r["item_id"] for r in ev.records if r["arm"] == THRESH and r["label"] == "benign"}
    hide = set(sorted(hide)[:10])
    partial = dataclasses.replace(
        ev,
        records=tuple(r for r in ev.records if not (r["arm"] == THRESH and r["item_id"] in hide)),
    )
    r = _by_id(build_report(study, [partial]))[THRESH]
    assert r["outcome"] == "inconclusive"
    assert any("different samples in the two arms" in x for x in r["reasons"])
    assert any("lack the expected (arm, sample) records" in x for x in r["reasons"])


def test_items_missing_from_both_arms_or_substituted_are_insufficient(tmp_path, raw, alerts):
    # A different item set means the registered metadata cannot be verified, so no metric is
    # reported at all (insufficient_evidence rather than inconclusive with unchecked tags).
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    gone = "syn-alert-040"
    missing = dataclasses.replace(ev, records=tuple(r for r in ev.records if r["item_id"] != gone))
    r = _by_id(build_report(study, [missing]))[THRESH]
    assert r["outcome"] == "insufficient_evidence" and r["target_effect"] is None
    assert "cannot be verified" in r["reasons"][0]
    swapped = dataclasses.replace(
        ev,
        records=tuple(
            dict(r, item_id="syn-alert-999") if r["item_id"] == gone else r for r in ev.records
        ),
    )
    r2 = _by_id(build_report(study, [swapped]))[THRESH]
    assert r2["outcome"] == "insufficient_evidence" and "substituted" in r2["reasons"][0]


def test_missing_samples_are_incomplete_even_when_balanced(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)  # 2 samples per item registered
    study = parse_study(study_raw)
    ev = evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags, edit_report=EDIT_REPORT)
    first = ev.records[0]["item_id"]
    trimmed = dataclasses.replace(
        ev, records=tuple(r for r in ev.records if not (r["item_id"] == first and r["sample"] == 1))
    )
    (r,) = build_report(study, [trimmed])["candidates"]
    assert r["items"]["unbalanced_items"] == []  # both arms lost the same sample
    assert r["outcome"] == "inconclusive"
    assert any("lack the expected (arm, sample) records" in x for x in r["reasons"])


def test_triage_metadata_is_recomputed_from_records(tmp_path, raw, alerts) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    records = [json.loads(x) for x in (run / "records.jsonl").read_text().splitlines()]
    a, labels = alerts
    # The canonical metadata hash recomputed from records equals the registered one.
    assert triage_metadata_sha256(records_metadata(records)) == alerts_metadata_sha256(a, labels)
    # Disagreement between arms is refused.
    bad = [
        dict(r, label="benign") if (r["item_id"] == "syn-alert-001" and r["arm"] == THRESH) else r
        for r in records
    ]
    _rewrite_records(run, bad)
    with pytest.raises(EvidenceError, match="differ between records"):
        load_evidence(run, study)
    # A consistent relabel (both arms) is refused by the registered metadata hash.
    relabel = [dict(r, label="benign") if r["item_id"] == "syn-alert-001" else r for r in records]
    _rewrite_records(run, relabel)
    with pytest.raises(EvidenceError, match="differ from the study"):
        load_evidence(run, study)
    # Moving an alert into the target slice after the fact is refused the same way.
    retag = [
        dict(r, tags=["lateral-movement"]) if r["item_id"] == "syn-alert-040" else r
        for r in records
    ]
    _rewrite_records(run, retag)
    with pytest.raises(EvidenceError, match="differ from the study"):
        load_evidence(run, study)
    # A study without the registered metadata hash cannot load evidence at all.
    unreg = copy.deepcopy(raw)
    del unreg["task"]["dataset"]["metadata_sha256"]
    s2, run2 = _evidence(tmp_path, unreg, alerts, name="unreg")
    with pytest.raises(EvidenceError, match="metadata_sha256 not registered"):
        load_evidence(run2, s2)


def test_whole_valued_multi_sample_items_are_not_binding() -> None:
    base = {f"i{n}": (1.0, 1.0) for n in range(12)}  # means of 2 samples that happen to agree
    cand = {f"i{n}": (0.0, 1.0) for n in range(12)}
    keys = sorted(base)
    assert paired_improvement(base, cand, keys, -1, 0.95)["status"] == "unreviewed_heuristic"
    assert one_sided_bound(cand, keys, "upper", 0.95)["status"] == "unreviewed_heuristic"
    assert paired_improvement(base, cand, keys, -1, 0.95, single_sample=True)["status"] == (
        "valid_by_bonferroni"
    )


def test_frozen_study_registers_the_design(raw) -> None:
    for key in ("expected_items_sha256", "metadata_sha256"):
        unreg = copy.deepcopy({**raw, **FROZEN})
        del unreg["task"]["dataset"][key]
        with pytest.raises(StudyError, match=key):
            parse_study(unreg)
    with pytest.raises(StudyError, match="samples_per_item"):
        parse_study(
            {
                **raw,
                **FROZEN,
                "analysis": {k: v for k, v in raw["analysis"].items() if k != "samples_per_item"},
            }
        )
    with pytest.raises(StudyError, match="exactly one decision"):
        parse_study({**raw, "analysis": {**raw["analysis"], "samples_per_item": 2}})


def test_edit_manifest_hash_is_a_validated_canonical_json_sha256(tmp_path) -> None:
    _, _, study_raw, _ = _wsp_setup(tmp_path)
    for bad in ("sha256:" + "8" * 64, "8" * 63, "8" * 64 + "\n"):
        raw2 = copy.deepcopy(study_raw)
        raw2["candidates"][0]["intervention"]["edit_manifest_sha256"] = bad
        with pytest.raises(StudyError, match="canonical-JSON SHA-256"):
            parse_study(raw2)


def test_loss_counting_defaults_to_gross(tmp_path, raw, alerts) -> None:
    gross_default = copy.deepcopy({**raw, **FROZEN})
    for k in gross_default["constraints"]:
        k.pop("losses", None)
    _, report = _report(tmp_path, gross_default, alerts)
    (errs,) = [c for c in _by_id(report)[SOC]["constraints"] if c["id"] == "no-new-errors"]
    assert errs["losses_counted"] == "gross"


def test_identifiers_with_trailing_newlines_are_rejected(raw) -> None:
    # A Python regex `$` also matches before a trailing newline; checks use fullmatch.
    with pytest.raises(StudyError, match="frozen_by"):
        parse_study({**raw, **FROZEN, "frozen_by": "lead1\n"})
    with pytest.raises(StudyError, match="pseudonym"):
        parse_study({**raw, "owner": "owner1\n"})
    with pytest.raises(StudyError, match="64 hex"):
        parse_study(_mutate(raw, "task.dataset.expected_items_sha256", "a" * 64 + "\n"))


# --- fourth round: direct-call paths at the comparison boundary (third review F1, F3, F4) ----


def test_hand_built_accuracy_evidence_with_flag_and_post_hoc_tags_is_refused(tmp_path) -> None:
    # Third review F1: setting the verification flag on a hand-built WSP-sourced Evidence and
    # re-tagging items after seeing outcomes used to reach "meets".
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    study = parse_study(study_raw)
    ev = evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags, edit_report=EDIT_REPORT)
    honest = build_report(study, [ev])["candidates"][0]
    assert honest["outcome"] != "insufficient_evidence"
    first = ev.records[0]["item_id"]
    retagged = tuple(dict(r, tags=[]) if r["item_id"] == first else r for r in ev.records)
    forged = dataclasses.replace(
        ev,
        source="wsp",
        records=retagged,
        notes={**ev.notes, "slice_membership_verified": True, "wsp_split": "heldout"},
    )
    (r,) = build_report(study, [forged])["candidates"]
    assert r["outcome"] == "insufficient_evidence"
    assert "slice" in r["reasons"][0] and "not authentication" in r["reasons"][0]


@pytest.mark.parametrize(
    ("change", "what"),
    [
        (lambda r: dict(r, tags=["lateral-movement"]), "re-tag into the target slice"),
        (lambda r: dict(r, label="benign"), "relabel"),
        (lambda r: dict(r, cluster="syn-campaign-1"), "re-cluster"),
    ],
)
def test_hand_built_triage_evidence_with_altered_metadata_is_refused(
    tmp_path, raw, alerts, change, what
) -> None:
    # Third review F1: the same records refused by load_evidence used to reach "meets" when
    # passed to build_report directly. Both arms are altered consistently.
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    target = "syn-alert-021"  # a general (non-slice) threat
    altered = tuple(change(r) if r["item_id"] == target else r for r in ev.records)
    r = _by_id(build_report(study, [dataclasses.replace(ev, records=altered)]))[THRESH]
    assert r["outcome"] == "insufficient_evidence", what
    assert "registered metadata" in r["reasons"][0] and "not authentication" in r["reasons"][0]


def test_hand_built_triage_evidence_with_arm_disagreement_is_refused(tmp_path, raw, alerts):
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    one_arm = tuple(
        dict(r, label="benign") if (r["item_id"] == "syn-alert-001" and r["arm"] == THRESH) else r
        for r in ev.records
    )
    r = _by_id(build_report(study, [dataclasses.replace(ev, records=one_arm)]))[THRESH]
    assert r["outcome"] == "insufficient_evidence" and "differ between records" in r["reasons"][0]


def test_duplicate_records_on_the_direct_path_are_a_gate_not_a_crash(tmp_path, raw, alerts):
    # Third review F4: a duplicate triage record used to crash build_report.
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    dup = next(r for r in ev.records if r["arm"] == THRESH)
    report = build_report(study, [dataclasses.replace(ev, records=(*ev.records, dict(dup)))])
    r = _by_id(report)[THRESH]
    assert r["outcome"] == "insufficient_evidence" and "duplicate record" in r["reasons"][0]


def test_genuine_evidence_still_passes_the_boundary_checks(tmp_path, raw, alerts) -> None:
    _, report = _report(tmp_path, {**raw, **FROZEN}, alerts)
    assert _by_id(report)[THRESH]["outcome"] == "meets_declared_criteria_pending_human_decision"


def test_wsp_samples_per_item_must_match_the_protocol(tmp_path) -> None:
    # Third review F3: a mismatch used to fail closed with a misleading reason.
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    wrong = copy.deepcopy(study_raw)
    wrong["analysis"]["samples_per_item"] = proto.samples_per_item + 1
    with pytest.raises(EvidenceError, match="samples_per_item"):
        evidence_from_wsp(res.out_dir, proto, parse_study(wrong), ARM_MAP, tags)


# --- fifth round: the evidence boundary (fourth review items 1-4) ----------------------------


def test_extra_triage_samples_are_refused_not_a_crash(tmp_path, raw, alerts) -> None:
    # Fourth review item 1: a balanced extra sample used to crash build_report.
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    extra = tuple(dict(r, sample=1) for r in ev.records if r["item_id"] == "syn-alert-001")
    r = _by_id(build_report(study, [dataclasses.replace(ev, records=(*ev.records, *extra))]))[
        THRESH
    ]
    assert r["outcome"] == "insufficient_evidence"
    assert "outside the registered design" in r["reasons"][0]
    # The same through the loader (records and manifest regenerated together).
    records = [json.loads(x) for x in (run / "records.jsonl").read_text().splitlines()]
    _rewrite_records(run, records + [dict(x) for x in extra])
    loaded = load_evidence(run, study)
    r2 = _by_id(build_report(study, [loaded]))[THRESH]
    assert r2["outcome"] == "insufficient_evidence"


def test_item_set_mismatch_reports_no_metrics_from_unchecked_tags(tmp_path, raw, alerts) -> None:
    # Fourth review item 2: dropping one item skipped the metadata check, and the report then
    # showed a target interval computed from re-tagged alerts.
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    general_threats = {
        r["item_id"] for r in ev.records if r["label"] == "threat" and "general" in r["tags"]
    }
    forged = tuple(
        dict(r, tags=["lateral-movement"]) if r["item_id"] in general_threats else r
        for r in ev.records
        if r["item_id"] != "syn-alert-001"
    )
    r = _by_id(build_report(study, [dataclasses.replace(ev, records=forged)]))[THRESH]
    assert r["outcome"] == "insufficient_evidence"
    assert r["target_effect"] is None and r["constraints"] == []
    assert "cannot be verified" in r["reasons"][0]


def test_unregistered_item_set_means_no_metrics(tmp_path, raw, alerts) -> None:
    draft = copy.deepcopy(raw)
    del draft["task"]["dataset"]["expected_items_sha256"]
    study, run = _evidence(tmp_path, draft, alerts)
    r = _by_id(build_report(study, [load_evidence(run, study)]))[THRESH]
    assert r["outcome"] == "insufficient_evidence" and "not registered" in r["reasons"][0]


def test_boundary_runs_before_provenance(tmp_path, raw, alerts) -> None:
    # Fourth review item 3: the documented gate order now matches the code.
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    dup = next(r for r in ev.records if r["arm"] == THRESH)
    arms = {**ev.arms, "baseline": {**ev.arms["baseline"], "config_sha256": "0" * 64}}
    forged = dataclasses.replace(ev, records=(*ev.records, dict(dup)), arms=arms)
    r = _by_id(build_report(study, [forged]))[THRESH]
    assert r["outcome"] == "insufficient_evidence" and "duplicate record" in r["reasons"][0]


class _LyingTags(list):
    """Iterates as the registered tags but claims membership of anything."""

    def __contains__(self, item) -> bool:
        return True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: dict(r, tags="lateral-movement"), "list of strings"),
        (lambda r: dict(r, tags=[1]), "list of strings"),
        (lambda r: dict(r, tags=_LyingTags(r["tags"])), "list of strings"),
        (lambda r: dict(r, label=1), "label must be a string"),
        (lambda r: dict(r, cluster=["c"]), "cluster must be a string"),
        (lambda r: dict(r, sample=True), "sample must be an integer"),
        (lambda r: dict(r, item_id=7), "item_id must be strings"),
    ],
)
def test_ordinary_input_types_are_validated(tmp_path, raw, alerts, change, message) -> None:
    # Fourth review item 4: what is hashed and what the metrics read must be the same plain
    # data. These validate ordinary input; caller-supplied evidence is still unauthenticated.
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    altered = tuple(change(r) if r["item_id"] == "syn-alert-021" else r for r in ev.records)
    r = _by_id(build_report(study, [dataclasses.replace(ev, records=altered)]))[THRESH]
    assert r["outcome"] == "insufficient_evidence" and message in r["reasons"][0]


# --- sixth round: candidate resolution and input shapes (fifth review R1, R2) ---------------

MEETS = "meets_declared_criteria_pending_human_decision"


def test_an_equal_copy_of_the_registered_candidate_is_compared_as_registered(
    tmp_path, raw, alerts
) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = _load(run, study)
    copied = compare_candidate(study, copy.deepcopy(study.candidate(THRESH)), [ev])
    assert copied == compare_candidate(study, study.candidate(THRESH), [ev])
    assert copied["outcome"] == MEETS


@pytest.mark.parametrize(
    ("cid", "change"),
    [
        # R1: the calibration check ran on whatever dict was passed.
        (
            SOC,
            lambda c: c["intervention"]["calibration"]["dataset"].update(
                name="other-calibration", content_sha256="1" * 64
            ),
        ),
        # R1: the registered edit-verification status could be overridden.
        (EDIT, lambda c: c["intervention"]["edit_verification"].update(status="verified")),
        (THRESH, lambda c: c.update(hypothesis="a different hypothesis")),
    ],
)
def test_altered_candidate_dict_is_refused(tmp_path, raw, alerts, cid, change) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    altered = copy.deepcopy(study.candidate(cid))
    change(altered)
    r = compare_candidate(study, altered, [ev])
    assert r["outcome"] == "insufficient_evidence"
    assert "differs from the study's registration" in r["reasons"][0]
    assert r["target_effect"] is None and r["constraints"] == []


def test_unregistered_candidate_is_refused_even_with_its_own_evidence(
    tmp_path, raw, alerts
) -> None:
    # R1: a candidate id the study does not register, with its arm added to the evidence,
    # reached "meets".
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = load_evidence(run, study)
    rogue = dict(copy.deepcopy(study.candidate(THRESH)), id="rogue")
    forged = dataclasses.replace(
        ev,
        arms={**ev.arms, "rogue": ev.arms[THRESH]},
        records=(*ev.records, *(dict(r, arm="rogue") for r in ev.records if r["arm"] == THRESH)),
    )
    r = compare_candidate(study, rogue, [forged])
    assert r["outcome"] == "insufficient_evidence" and "not registered" in r["reasons"][0]
    for bad in ("x", None, {"id": 7}):
        r = compare_candidate(study, bad, [ev])
        assert r["outcome"] == "insufficient_evidence" and "string id" in r["reasons"][0]


class _IdentityStr(str):
    """Equal to the original text but hashed by identity: slips past a set of seen keys."""

    __hash__ = object.__hash__


class _IdentityInt(int):
    __hash__ = object.__hash__


def _first(ev, change):
    return dataclasses.replace(ev, records=(change(dict(ev.records[0])), *ev.records[1:]))


def _drop(key):
    def change(r):
        r.pop(key)
        return r

    return change


@pytest.mark.parametrize(
    ("alter", "message"),
    [
        (lambda ev: dataclasses.replace(ev, records=(*ev.records, "x")), "not an object"),
        (lambda ev: dataclasses.replace(ev, records=(None, *ev.records)), "not an object"),
        (lambda ev: _first(ev, _drop("arm")), "arm and item_id must be strings"),
        (lambda ev: _first(ev, _drop("status")), "status must be a string"),
        (lambda ev: _first(ev, lambda r: dict(r, arm=_IdentityStr(r["arm"]))), "must be strings"),
        (lambda ev: _first(ev, lambda r: dict(r, sample=_IdentityInt(0))), "sample must be"),
        (lambda ev: _first(ev, lambda r: dict(r, correct="yes")), "correct must be a boolean"),
        (lambda ev: _first(ev, lambda r: dict(r, decision=1)), "decision must be a string"),
        (lambda ev: dataclasses.replace(ev, started_at="not-a-date"), "is not ISO-8601"),
        (lambda ev: dataclasses.replace(ev, started_at=5), "must be an ISO-8601 string"),
        (lambda ev: dataclasses.replace(ev, started_at="2026-11-02T00:00"), "no time zone"),
        (lambda ev: dataclasses.replace(ev, notes=None), "notes must be an object"),
        (
            lambda ev: dataclasses.replace(ev, arms={**ev.arms, "baseline": None}),
            "arm 'baseline' is not an object",
        ),
        (lambda ev: dataclasses.replace(ev, limitations=("ok", 3)), "limitations must be"),
        (lambda ev: dataclasses.replace(ev, records=None), "records must be a list"),
        (lambda ev: {"run_id": ev.run_id}, "not an Evidence object"),
    ],
)
def test_malformed_evidence_is_a_gate_not_an_exception(
    tmp_path, raw, alerts, alter, message
) -> None:
    # R2: these used to raise TypeError, KeyError, ValueError or AttributeError (or, for the
    # identity-hashed arm, let a duplicate past the duplicate check before crashing).
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    bad = alter(load_evidence(run, study))
    report = build_report(study, [bad])
    for r in report["candidates"]:
        assert r["outcome"] == "insufficient_evidence" and message in r["reasons"][0]
        assert r["target_effect"] is None
    assert report["evidence_class"] == "synthetic_fixture_not_a_measurement"
    assert report["evidence_runs"] == [] and message in report["evidence_errors"][0]
    render_markdown(report)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (lambda recs: [*recs[:-1], "not json"], "not valid JSON"),
        (lambda recs: [*recs[:-1], json.dumps(["a", "list"])], "not an object"),
        (
            lambda recs: [
                json.dumps({k: v for k, v in json.loads(recs[0]).items() if k != "sample"}),
                *recs[1:],
            ],
            "sample must be an integer",
        ),
        (
            lambda recs: [json.dumps(dict(json.loads(recs[0]), tags="general")), *recs[1:]],
            "tags must be a list",
        ),
    ],
)
def test_loader_reports_malformed_records_as_evidence_errors(
    tmp_path, raw, alerts, lines, message
) -> None:
    # R2 (loader path): a raw KeyError or AttributeError, not EvidenceError, used to escape.
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    path = run / "records.jsonl"
    path.write_text("".join(x + "\n" for x in lines(path.read_text().splitlines())))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["records_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EvidenceError, match=message):
        load_evidence(run, study)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [("[1, 2]", "not an object"), ("{", "not valid JSON")],
)
def test_loader_reports_a_malformed_manifest(tmp_path, raw, alerts, manifest, message) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    (run / "manifest.json").write_text(manifest)
    with pytest.raises(EvidenceError, match=message):
        load_evidence(run, study)


def test_loader_reports_an_unusable_start_time(tmp_path, raw, alerts) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    manifest = json.loads((run / "manifest.json").read_text())
    (run / "manifest.json").write_text(json.dumps(dict(manifest, started_at="yesterday")))
    with pytest.raises(EvidenceError, match="started_at"):
        load_evidence(run, study)


def test_wsp_item_tags_must_be_lists_of_strings(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    first = next(iter(tags))
    for bad in ({**tags, first: "heldout-slice"}, {**tags, first: [1]}):
        with pytest.raises(EvidenceError, match="item_tags"):
            evidence_from_wsp(res.out_dir, proto, parse_study(study_raw), ARM_MAP, bad)


# --- seventh round: manifest identifier, nested shapes, loader normalization (G1-G8) --------


def _edit_manifest(run: Path, change) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    change(manifest)
    (run / "manifest.json").write_text(json.dumps(manifest))


def _manifest_cases(run, study):
    """The three ways a reader can load an edited manifest."""
    with pytest.raises(EvidenceError, match="externally recorded run_manifest_sha256"):
        load_evidence(run, study, expected_run_manifest_sha256=EXTERNAL_RECORD[run])
    unchecked = load_evidence(run, study)
    # Someone who can edit the manifest can also recompute its hash: a value taken *after* the
    # edit matches, so it protects nothing. Only a value recorded before, outside the run
    # directory and under independent control, detects the edit.
    recomputed = load_evidence(run, study, expected_run_manifest_sha256=run_manifest_sha256(run))
    return unchecked, recomputed


def test_edited_start_time_is_caught_only_by_an_external_record(tmp_path, raw, alerts) -> None:
    # G4: the study was frozen after the run, so the honest outcome is descriptive only.
    late = {**raw, **FROZEN, "frozen_at": "2999-01-01T00:00Z"}
    study, run = _evidence(tmp_path, late, alerts)
    honest = _by_id(build_report(study, [_load(run, study)]))[THRESH]
    assert honest["outcome"] == "descriptive_only_not_preregistered"
    _edit_manifest(run, lambda m: m.update(started_at="2999-02-01T00:00Z"))
    unchecked, recomputed = _manifest_cases(run, study)
    r = _by_id(build_report(study, [unchecked]))[THRESH]
    assert unchecked.notes["run_manifest_check"] == "not_checked"
    assert r["outcome"] == "inconclusive" and r["rule_outcome"] == "inconclusive"
    assert any("not checked against an externally recorded" in x for x in r["reasons"])
    # The documented limit: a recomputed hash "matches", and the edit then passes.
    assert recomputed.notes["run_manifest_check"] == "matched_external_record"
    assert _by_id(build_report(study, [recomputed]))[THRESH]["outcome"] == MEETS


def test_edited_artifact_digest_is_caught_only_by_an_external_record(tmp_path, raw, alerts) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    registered = study.baseline["artifact"]
    file = registered.get("file")

    def wrong(m):  # the run as it honestly happened: a different baseline file was loaded
        arm = m["arms"]["baseline"]
        if file:
            arm["artifacts"][file] = "e" * 64
        arm["artifact_sha256s"] = ["e" * 64]

    _edit_manifest(run, wrong)
    EXTERNAL_RECORD[run] = run_manifest_sha256(run)  # recorded when that run finished
    honest = _by_id(build_report(study, [_load(run, study)]))[THRESH]
    assert honest["outcome"] == "baseline_changed_not_interpreted"

    def restore(m):  # the post-run edit: claim the registered baseline was loaded
        arm = m["arms"]["baseline"]
        if file:
            arm["artifacts"][file] = registered["sha256"]
        arm["artifact_sha256s"] = [registered["sha256"]]

    _edit_manifest(run, restore)
    unchecked, recomputed = _manifest_cases(run, study)
    assert _by_id(build_report(study, [unchecked]))[THRESH]["outcome"] == "inconclusive"
    assert _by_id(build_report(study, [recomputed]))[THRESH]["outcome"] == MEETS  # the limit


def test_edited_wsp_split_is_caught_only_by_an_external_record(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path, split="development")
    study = parse_study(study_raw)
    recorded = run_manifest_sha256(res.out_dir)

    def load(expected=None):
        return evidence_from_wsp(
            res.out_dir,
            proto,
            study,
            ARM_MAP,
            tags,
            edit_report=EDIT_REPORT,
            expected_run_manifest_sha256=expected,
        )

    (honest,) = build_report(study, [load(recorded)])["candidates"]
    assert honest["outcome"] == "prerequisite_missing" and "held-out" in honest["reasons"][0]
    _edit_manifest(res.out_dir, lambda m: m.update(split="heldout"))
    with pytest.raises(EvidenceError, match="externally recorded run_manifest_sha256"):
        load(recorded)
    unchecked = load()
    assert unchecked.notes["wsp_split"] == "heldout"  # read from the edited manifest
    assert unchecked.notes["run_manifest_check"] == "not_checked"
    (r,) = build_report(study, [unchecked])["candidates"]
    assert r["outcome"] != MEETS
    recomputed = load(run_manifest_sha256(res.out_dir))
    assert recomputed.notes["run_manifest_check"] == "matched_external_record"  # the limit


def test_manifest_identifier_is_deterministic_and_validated(tmp_path, raw, alerts) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    before = run_manifest_sha256(run)
    _edit_manifest(run, lambda m: None)  # re-serialized differently, same content
    assert run_manifest_sha256(run) == before
    for bad in (before.upper(), before + "\n", "x" * 64, 5):
        with pytest.raises(EvidenceError, match="64 lowercase hex"):
            load_evidence(run, study, expected_run_manifest_sha256=bad)


def test_cli_prints_and_checks_the_manifest_identifier(tmp_path, capsys) -> None:
    assert improve.main(["demo", "--output", str(tmp_path / "demo")]) == 0
    capsys.readouterr()
    run = tmp_path / "demo" / "evidence"
    assert improve.main(["run-manifest-hash", str(run)]) == 0
    sha = json.loads(capsys.readouterr().out)["run_manifest_sha256"]
    assert sha == run_manifest_sha256(run)
    base = ["report", "--study", str(FIX / "synthetic_study.json"), "--evidence", str(run)]
    assert (
        improve.main(
            [*base, "--output", str(tmp_path / "r1"), "--expected-run-manifest-sha256", sha]
        )
        == 0
    )
    capsys.readouterr()
    report = json.loads((tmp_path / "r1" / "report.json").read_text())
    assert report["evidence_runs"][0]["run_manifest_check"] == "matched_external_record"
    out = [*base, "--output", str(tmp_path / "r2"), "--expected-run-manifest-sha256", "0" * 64]
    assert improve.main(out) == 0
    assert "externally recorded" in json.loads(capsys.readouterr().out)["evidence_errors"][0]
    two = [*base, "--output", str(tmp_path / "r3"), "--expected-run-manifest-sha256", sha, sha]
    assert improve.main(two) == 2


class _SneakyEvidence(Evidence):
    pass


@pytest.mark.parametrize(
    ("alter", "message"),
    [
        # G1: valid ISO text whose UTC conversion overflows.
        (
            lambda ev: dataclasses.replace(ev, started_at="0001-01-01T00:00+01:00"),
            "out of range",
        ),
        # G2: nested values the comparison reads.
        (
            lambda ev: dataclasses.replace(
                ev, arms={**ev.arms, "baseline": {**ev.arms["baseline"], "artifacts": ["x"]}}
            ),
            "artifacts must map",
        ),
        (
            lambda ev: dataclasses.replace(
                ev, arms={**ev.arms, "baseline": {**ev.arms["baseline"], "artifact_sha256s": 5}}
            ),
            "artifact_sha256s must be",
        ),
        (
            lambda ev: dataclasses.replace(
                ev, arms={**ev.arms, "baseline": {**ev.arms["baseline"], "config_sha256": 5}}
            ),
            "config_sha256 must be",
        ),
        (
            lambda ev: dataclasses.replace(ev, notes={"edit_verification": "x"}),
            "edit_verification must be an object",
        ),
        (lambda ev: dataclasses.replace(ev, notes={"x": object()}), "plain JSON"),
        (lambda ev: dataclasses.replace(ev, notes={1: "a", "b": 2}), "plain JSON"),
        (lambda ev: dataclasses.replace(ev, notes={"wsp_split": 1}), "wsp_split must be"),
        # G5: an Evidence subclass could return different values on later reads.
        (
            lambda ev: _SneakyEvidence(
                **{f.name: getattr(ev, f.name) for f in dataclasses.fields(ev)}
            ),
            "not an Evidence object",
        ),
    ],
)
def test_malformed_nested_evidence_is_a_gate(tmp_path, raw, alerts, alter, message) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    report = build_report(study, [alter(_load(run, study))])
    for r in report["candidates"]:
        assert r["outcome"] == "insufficient_evidence" and message in r["reasons"][0]
    assert message in report["evidence_errors"][0]
    render_markdown(report)


def test_evidence_argument_must_be_a_list(tmp_path, raw, alerts) -> None:
    study, _ = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    r = compare_candidate(study, study.candidate(THRESH), None)
    assert r["outcome"] == "insufficient_evidence" and "must be a list" in r["reasons"][0]
    report = build_report(study, None)
    assert all(c["outcome"] == "insufficient_evidence" for c in report["candidates"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m.update(started_at="9999-12-31T23:59-01:00"), "out of range"),  # G1
        (lambda m: m.update(started_at="2026-11-02T00:00"), "no time zone"),
        (lambda m: m.update(limitations=None), "limitations must be"),  # G3
        (lambda m: m.update(limitations=3), "limitations must be"),
        (lambda m: m.update(limitations="abc"), "limitations must be"),
        (lambda m: m.update(limitations={"k": 1}), "limitations must be"),
        (lambda m: m.update(synthetic="yes"), "synthetic must be"),
        (lambda m: m.update(arms=None), "arms must map"),
    ],
)
def test_loader_normalizes_malformed_manifest_fields(
    tmp_path, raw, alerts, change, message
) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    _edit_manifest(run, change)
    with pytest.raises(EvidenceError, match=message):
        load_evidence(run, study)


def test_loader_normalizes_unreadable_records(tmp_path, raw, alerts) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    path = run / "records.jsonl"
    path.write_bytes(b"\xff\xfe\n")
    _edit_manifest(
        run, lambda m: m.update(records_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    )
    with pytest.raises(EvidenceError, match=r"cannot read records\.jsonl"):
        load_evidence(run, study)
    path.unlink()
    with pytest.raises(EvidenceError, match=r"records\.jsonl is missing"):
        load_evidence(run, study)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.pop("dataset"),
        lambda m: m.update(dataset="x"),
        lambda m: m.update(arms=None),
        lambda m: m.update(counts="x"),
        lambda m: m.update(started_at="not-a-time"),
        lambda m: m.update(started_at=5),
        lambda m: m["arms"][0].pop("inference"),
    ],
)
def test_wsp_loader_normalizes_malformed_manifest_fields(tmp_path, change) -> None:
    # G3: these raised KeyError, AttributeError, TypeError or ValueError.
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    _edit_manifest(res.out_dir, change)
    with pytest.raises(EvidenceError):
        evidence_from_wsp(
            res.out_dir, proto, parse_study(study_raw), ARM_MAP, tags, edit_report=EDIT_REPORT
        )


def test_wsp_loader_normalizes_malformed_arguments(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    study = parse_study(study_raw)
    with pytest.raises(EvidenceError, match="mappings"):
        evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, list(tags))
    with pytest.raises(EvidenceError, match="arm_map"):
        evidence_from_wsp(res.out_dir, proto, study, {"reference": ["baseline"]}, tags)
    with pytest.raises(EvidenceError, match="edit_report"):
        evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags, edit_report={"x": object()})
    (res.out_dir / "manifest.json").write_text("[1, 2]")
    with pytest.raises(EvidenceError):
        evidence_from_wsp(res.out_dir, proto, study, ARM_MAP, tags)


def test_notes_cannot_overwrite_report_run_fields(tmp_path, raw, alerts) -> None:
    # G6
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = _load(run, study)
    forged = dataclasses.replace(
        ev, notes={**ev.notes, "run_id": "other", "source": "wsp", "synthetic": False}
    )
    (row,) = build_report(study, [forged])["evidence_runs"]
    assert (row["run_id"], row["source"], row["synthetic"]) == (ev.run_id, ev.source, True)


def test_malformed_object_in_a_non_synthetic_report_is_not_called_synthetic(
    tmp_path, raw, alerts, monkeypatch
) -> None:
    # G7: the warning text must not claim synthetic fixtures for a real study.
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    ev = dataclasses.replace(_load(run, study), synthetic=False)
    real = dataclasses.replace(study, synthetic=False)
    report = build_report(real, [ev, dataclasses.replace(ev, notes=None)])
    assert report["evidence_class"] == "not_interpreted_malformed_evidence"
    assert not any("synthetic fixtures" in x for x in report["limitations"])
    assert any("malformed evidence object" in x for x in report["limitations"])
    clean = build_report(real, [ev])
    assert clean["evidence_class"] == "measured_unauthenticated"
    assert any("None of them authenticates" in x for x in clean["limitations"])


# --- eighth round: report completeness, CLI flags, deep JSON (H1-H5) ------------------------


def _frozen_cli_runs(tmp_path, raw, alerts, names=("a",)):
    """A frozen study file plus honest runs under it, with their externally recorded ids."""
    frozen = {**raw, **FROZEN}
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(frozen))
    runs = []
    for n in names:
        _, run = _evidence(tmp_path, frozen, alerts, name=n)
        runs.append(run)
    return study_path, runs, [EXTERNAL_RECORD[r] for r in runs]


def _cli_report(capsys, argv):
    code = improve.main(argv)
    out = json.loads(capsys.readouterr().out) if code == 0 else None
    return code, out


def _decide_approve(study_path, report_path, out):
    return improve.main(
        [
            "decide",
            "--study",
            str(study_path),
            "--report",
            str(report_path),
            "--candidate",
            THRESH,
            "--decision",
            "approve_further_evaluation",
            "--decided-by",
            "reviewer1",
            "--rationale",
            "synthetic fixture review",
            "--output",
            str(out),
        ]
    )


@pytest.mark.parametrize("failure", ["wrong_record", "missing_run"])
def test_a_requested_run_that_fails_to_load_prevents_a_pass_and_an_approval(
    tmp_path, raw, alerts, capsys, failure
) -> None:
    # H1: both runs pair the candidate, so together they are insufficient_evidence; dropping
    # one used to let the other produce a passing report.
    study_path, (a, b), (ha, hb) = _frozen_cli_runs(tmp_path, raw, alerts, names=("a", "b"))
    if failure == "missing_run":
        b = tmp_path / "nope"
    hashes = [ha, "0" * 64 if failure == "wrong_record" else hb]
    argv = ["report", "--study", str(study_path), "--evidence", str(a), str(b)]
    code, out = _cli_report(
        capsys, [*argv, "--expected-run-manifest-sha256", *hashes, "--output", str(tmp_path / "r")]
    )
    assert code == 0 and out["outcome"] == "inconclusive"
    report = json.loads((tmp_path / "r" / "report.json").read_text())
    (err,) = report["evidence_errors"]
    expected = "externally recorded run_manifest_sha256" if failure == "wrong_record" else "nope"
    assert expected in err  # the specific load error stays visible
    assert report["evidence_complete"] is False
    assert (
        expected in report["limitations"][0]
        and "cannot support an approval" in (report["limitations"][0])
    )
    t = _by_id(report)[THRESH]
    assert t["outcome"] == "inconclusive" and t["rule_outcome"] == MEETS
    assert any("failed to load" in x for x in t["reasons"])
    assert "failed to load" in (tmp_path / "r" / "report.md").read_text()
    assert _decide_approve(study_path, tmp_path / "r" / "report.json", tmp_path / "d.json") == 1
    assert "evidence did not all load" in capsys.readouterr().err
    assert not (tmp_path / "d.json").exists()


def test_build_report_and_decision_refuse_incomplete_evidence(tmp_path, raw, alerts) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    complete = build_report(study, [_load(run, study)])
    assert complete["evidence_complete"] is True and complete["outcome"].startswith(
        "candidates_meet"
    )
    partial = build_report(study, [_load(run, study)], ["other: records.jsonl is missing"])
    assert partial["outcome"] == "inconclusive" and partial["candidates_meeting_criteria"] == []
    kw = {"decided_at": "2026-11-05T09:00Z", "rationale": "r", "decided_by": "reviewer1"}
    with pytest.raises(ValueError, match="did not all load"):
        make_decision(study, partial, candidate=THRESH, decision="approve_further_evaluation", **kw)
    # Recording that more evidence is needed stays possible.
    make_decision(study, partial, candidate=THRESH, decision="request_more_evidence", **kw)
    # A report from before this field existed cannot support an approval either.
    old = {k: v for k, v in complete.items() if k != "evidence_complete"}
    assert any(
        "did not all load" in x
        for x in validate_decision(
            make_decision(
                study, complete, candidate=THRESH, decision="approve_further_evaluation", **kw
            ),
            study,
            old,
        )
    )


@pytest.mark.parametrize("flag", ["--evidence", "--expected-run-manifest-sha256"])
def test_repeated_cli_flags_are_refused_not_overwritten(tmp_path, raw, alerts, capsys, flag):
    # H2: `--evidence A ... --evidence B` used to keep only B.
    study_path, (a, b), (ha, hb) = _frozen_cli_runs(tmp_path, raw, alerts, names=("a", "b"))
    argv = [
        "report",
        "--study",
        str(study_path),
        "--evidence",
        str(a),
        "--expected-run-manifest-sha256",
        ha,
        flag,
        str(b) if flag == "--evidence" else hb,
        "--output",
        str(tmp_path / "r"),
    ]
    with pytest.raises(SystemExit) as exc:
        improve.main(argv)
    assert exc.value.code == 2
    assert "given more than once" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("hashes_first", [False, True])
def test_hash_to_run_mapping_holds_in_both_flag_orders(
    tmp_path, raw, alerts, capsys, hashes_first
) -> None:
    study_path, (a, b), (ha, hb) = _frozen_cli_runs(tmp_path, raw, alerts, names=("a", "b"))

    def run(evidence, hashes, out):
        e = ["--evidence", *map(str, evidence)]
        h = ["--expected-run-manifest-sha256", *hashes]
        middle = [*h, *e] if hashes_first else [*e, *h]
        return _cli_report(
            capsys,
            ["report", "--study", str(study_path), *middle, "--output", str(tmp_path / out)],
        )

    # One run, its own record: matched, and the frozen study's candidate meets.
    code, out = run([a], [ha], "one")
    assert code == 0 and out["evidence_errors"] == []
    assert out["outcome"] == "candidates_meet_declared_criteria_pending_human_decision"
    report = json.loads((tmp_path / "one" / "report.json").read_text())
    assert report["evidence_runs"][0]["run_manifest_check"] == "matched_external_record"
    # Two runs in order: both load (and then two runs pair the candidate: insufficient).
    code, out = run([a, b], [ha, hb], "two")
    assert code == 0 and out["evidence_errors"] == []
    assert (
        _by_id(json.loads((tmp_path / "two" / "report.json").read_text()))[THRESH]["outcome"]
        == "insufficient_evidence"
    )
    # Swapped: each run is checked against the other's record, so both are refused.
    code, out = run([a, b], [hb, ha], "swapped")
    assert code == 0 and len(out["evidence_errors"]) == 2 and out["outcome"] == "inconclusive"
    assert all("externally recorded" in x for x in out["evidence_errors"])
    # Count mismatch is a usage error.
    assert (
        improve.main(
            [
                "report",
                "--study",
                str(study_path),
                "--evidence",
                str(a),
                str(b),
                "--expected-run-manifest-sha256",
                ha,
                "--output",
                str(tmp_path / "count"),
            ]
        )
        == 2
    )


DEEP = "[" * 50_000 + "]" * 50_000


def test_deeply_nested_json_is_a_controlled_error(tmp_path, raw, alerts, capsys) -> None:
    # H3: these raised a raw RecursionError from the loader, run-manifest-hash and report.
    study_path, (run,), _ = _frozen_cli_runs(tmp_path, raw, alerts)
    study = parse_study(json.loads(study_path.read_text()))
    text = (run / "manifest.json").read_text().rstrip()
    (run / "manifest.json").write_text(text[:-1] + ', "deep": ' + DEEP + "}")
    with pytest.raises(EvidenceError, match="nested too deeply"):
        load_evidence(run, study)
    with pytest.raises(EvidenceError, match="nested too deeply"):
        run_manifest_sha256(run)
    assert improve.main(["run-manifest-hash", str(run)]) == 1
    assert "nested too deeply" in capsys.readouterr().err
    code, out = _cli_report(
        capsys,
        [
            "report",
            "--study",
            str(study_path),
            "--evidence",
            str(run),
            "--output",
            str(tmp_path / "r"),
        ],
    )
    assert code == 0 and "nested too deeply" in out["evidence_errors"][0]
    assert out["outcome"] == "inconclusive"


def test_deeply_nested_records_line_is_a_controlled_error(tmp_path, raw, alerts) -> None:
    study, run = _evidence(tmp_path, {**raw, **FROZEN}, alerts)
    path = run / "records.jsonl"
    path.write_text(path.read_text() + DEEP + "\n")
    _edit_manifest(
        run,
        lambda m: m.update(
            records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_records=m["expected_records"] + 1,
        ),
    )
    with pytest.raises(EvidenceError, match="nested too deeply"):
        load_evidence(run, study)


def test_deeply_nested_wsp_manifest_is_a_controlled_error(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    text = (res.out_dir / "manifest.json").read_text().rstrip()
    (res.out_dir / "manifest.json").write_text(text[:-1] + ', "deep": ' + DEEP + "}")
    with pytest.raises(EvidenceError):
        evidence_from_wsp(res.out_dir, proto, parse_study(study_raw), ARM_MAP, tags)


@pytest.mark.parametrize("value", [None, 0, 1, "", "false", [], {}])
def test_wsp_synthetic_label_must_be_a_boolean(tmp_path, value) -> None:
    # H4: bool() coercion turned null, 0 or "" into "not synthetic" and "false" into synthetic.
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    _edit_manifest(res.out_dir, lambda m: m.update(synthetic=value))
    with pytest.raises(EvidenceError, match="synthetic must be a boolean"):
        evidence_from_wsp(res.out_dir, proto, parse_study(study_raw), ARM_MAP, tags)


def test_wsp_synthetic_boolean_is_kept(tmp_path) -> None:
    proto, res, study_raw, tags = _wsp_setup(tmp_path)
    assert evidence_from_wsp(res.out_dir, proto, parse_study(study_raw), ARM_MAP, tags).synthetic

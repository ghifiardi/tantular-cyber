"""WSP research harness (tools/wsp): synthetic fixtures only, no model, no network."""

from __future__ import annotations

import ast
import copy
import functools
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WSP = ROOT / "tools" / "wsp"
sys.path.insert(0, str(WSP))

import wsp  # noqa: E402
from wsp_eval.analysis import analyze, mcnemar_exact, percentile_bootstrap  # noqa: E402
from wsp_eval.checkpoint_diff import SafetensorsError, diff_checkpoints  # noqa: E402
from wsp_eval.dataset import DatasetError, Item, check_items, load_items, parse_items  # noqa: E402
from wsp_eval.drift import drift_report  # noqa: E402
from wsp_eval.grading import FinalAnswerGrader  # noqa: E402
from wsp_eval.ledger import validate_record  # noqa: E402
from wsp_eval.protocol import ProtocolError, load_protocol, parse_protocol  # noqa: E402
from wsp_eval.runner import RunError, run_paired  # noqa: E402
from wsp_eval.synthetic import synthetic_generators  # noqa: E402

FIX = WSP / "fixtures"
SHA = "a" * 64
COMMIT = "b" * 40


@pytest.fixture
def raw_protocol() -> dict:
    return json.loads((FIX / "synthetic_protocol.json").read_text())


@pytest.fixture
def items() -> list[Item]:
    return load_items(FIX / "synthetic_items.jsonl")


def _run(tmp_path, raw, items, **kw):
    proto = parse_protocol(raw)
    gens = kw.pop("generators", None) or synthetic_generators(items, proto.seeds)
    return proto, run_paired(
        proto,
        items,
        gens,
        FinalAnswerGrader(),
        kw.pop("out", tmp_path / "run"),
        purpose="test",
        operator="op1",
        clock=iter(range(10_000)).__next__,
        run_id="wsp-test",
        **kw,
    )


# --- protocol ------------------------------------------------------------------------------


def test_fixture_protocol_parses_and_hash_is_stable(raw_protocol) -> None:
    a = parse_protocol(raw_protocol)
    b = load_protocol(FIX / "synthetic_protocol.json")
    assert a.sha256 == b.sha256 and len(a.sha256) == 64
    assert a.synthetic and not a.frozen
    raw_protocol["seeds"] = [11, 13]
    assert parse_protocol(raw_protocol).sha256 != a.sha256


def test_frozen_protocol_rejects_placeholders(raw_protocol) -> None:
    raw_protocol["frozen"] = True
    raw_protocol["dataset"]["revision"] = "REPLACE"
    with pytest.raises(ProtocolError, match="placeholders"):
        parse_protocol(raw_protocol)


def test_replication_arms_must_share_inference_settings(raw_protocol) -> None:
    raw_protocol["arms"][1]["inference"]["temperature"] = 0.0
    with pytest.raises(ProtocolError, match="different inference"):
        parse_protocol(raw_protocol)
    raw_protocol["arms"][1]["role"] = "alternative"  # declared as a separate intervention
    raw_protocol["analysis"]["candidate_arm"] = "candidate"
    assert parse_protocol(raw_protocol).arm("candidate").role == "alternative"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(seeds=[11]),
        lambda p: p.update(seeds=[11, 11]),
        lambda p: p["arms"][1].update(role="reference"),
        lambda p: p["arms"][1].update(role="production"),
        lambda p: p["analysis"].update(candidate_arm="nope"),
        lambda p: p["analysis"].update(primary_endpoint="vibes"),
        lambda p: p.pop("grader"),
    ],
)
def test_protocol_rejects_inconsistent_definitions(raw_protocol, mutate) -> None:
    mutate(raw_protocol)
    with pytest.raises(ProtocolError):
        parse_protocol(raw_protocol)


# --- dataset -------------------------------------------------------------------------------


def test_dataset_detects_duplicate_ids_and_cross_split_leakage(items) -> None:
    assert check_items(items)["ok"]
    leaked = [*items, Item("syn-x", "heldout", items[0].prompt, items[0].answer)]
    report = check_items(leaked)
    assert not report["ok"] and "leakage" in report["errors"][0]
    dup = [*items, Item(items[0].id, "development", "other", "1")]
    assert "duplicate ids" in check_items(dup)["errors"][0]
    within = [*items, Item("syn-y", "development", items[0].prompt, items[0].answer)]
    report = check_items(within)
    assert report["ok"] and report["within_split_duplicates"] == [["syn-dev-001", "syn-y"]]


def test_dataset_rejects_unexpected_fields() -> None:
    with pytest.raises(DatasetError):
        parse_items([{"id": "a", "split": "development", "prompt": "p", "answer": "1", "x": 1}])
    with pytest.raises(DatasetError):
        parse_items([{"id": "a", "split": "test", "prompt": "p", "answer": "1"}])


# --- runner --------------------------------------------------------------------------------


def test_every_request_is_recorded_including_errors_and_ungraded(tmp_path, raw_protocol, items):
    proto, res = _run(tmp_path, raw_protocol, items)
    dev = [i for i in items if i.split == "development"]
    assert len(res.records) == len(dev) * len(proto.seeds) * len(proto.arms)
    status = res.manifest["counts"]
    assert status["error"] == 1 and status["ungraded"] == 8
    err = next(r for r in res.records if r["status"] == "error")
    assert err["arm"] == "candidate" and err["error_class"] == "GenerationError"
    assert all(r["split"] == "development" for r in res.records)
    assert (res.out_dir / "records.jsonl").exists() and (res.out_dir / "manifest.json").exists()
    assert not (res.out_dir / "outputs.jsonl").exists()  # raw text not kept by default
    m = res.manifest
    assert m["protocol_sha256"] == proto.sha256 and m["synthetic"] is True
    assert m["human_decision"] == "pending" and len(m["harness_code_sha256"]) == 64
    assert {a["generator"]["provider"] for a in m["arms"]} == {"synthetic"}


def test_runs_are_reproducible(tmp_path, raw_protocol, items) -> None:
    _, a = _run(tmp_path, raw_protocol, items, out=tmp_path / "a")
    _, b = _run(tmp_path, raw_protocol, items, out=tmp_path / "b")
    assert a.records == b.records


def test_keep_outputs_writes_raw_text_separately(tmp_path, raw_protocol, items) -> None:
    _, res = _run(tmp_path, raw_protocol, items, keep_outputs=True)
    rows = (res.out_dir / "outputs.jsonl").read_text().strip().split("\n")
    assert len(rows) == res.manifest["counts"]["records"] - res.manifest["counts"]["error"]
    assert "text" not in res.records[0]


def test_runner_refuses_unsafe_or_inconsistent_runs(tmp_path, raw_protocol, items) -> None:
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x").write_text("x")
    with pytest.raises(RunError, match="immutable"):
        _run(tmp_path, raw_protocol, items, out=tmp_path / "used")
    with pytest.raises(RunError, match="inside the repository"):
        _run(tmp_path, raw_protocol, items, out=ROOT / "out" / "wsp-should-not-exist")
    assert not (ROOT / "out" / "wsp-should-not-exist").exists()
    with pytest.raises(RunError, match="frozen"):
        _run(tmp_path, raw_protocol, items, split="heldout")
    proto = parse_protocol(raw_protocol)
    gens = synthetic_generators(items, proto.seeds)
    del gens["reload"]
    with pytest.raises(RunError, match="missing"):
        _run(tmp_path, raw_protocol, items, generators=gens)
    real = copy.deepcopy(raw_protocol)
    real["synthetic"] = False
    with pytest.raises(RunError, match="synthetic generators"):
        _run(tmp_path, real, items)
    other = copy.deepcopy(raw_protocol)
    other["grader"]["revision"] = "2"
    with pytest.raises(RunError, match="grader"):
        _run(tmp_path, other, items)
    leaked = [*items, Item("syn-x", "heldout", items[0].prompt, items[0].answer)]
    with pytest.raises(RunError, match="integrity"):
        _run(tmp_path, raw_protocol, leaked)


def test_heldout_needs_frozen_protocol_and_named_evaluator(tmp_path, raw_protocol, items):
    raw_protocol.update(frozen=True, frozen_by="lead1", frozen_at="2026-11-01T00:00Z")
    with pytest.raises(RunError, match="heldout_authorized_by"):
        _run(tmp_path, raw_protocol, items, split="heldout")
    _, res = _run(tmp_path, raw_protocol, items, split="heldout", heldout_authorized_by="ev1")
    assert res.manifest["heldout_authorized_by"] == "ev1"
    assert {r["split"] for r in res.records} == {"heldout"}


# --- analysis ------------------------------------------------------------------------------


def test_mcnemar_matches_production_implementation() -> None:
    from tantular_cyber.benchmark.compare import mcnemar_exact as production

    for b in range(12):
        for c in range(12):
            assert mcnemar_exact(b, c) == production(b, c)


def test_bootstrap_is_seeded_and_clusters_by_item() -> None:
    diffs = [0.0, 0.5, 1.0, -0.5, 0.5]
    assert percentile_bootstrap(diffs, 500, 1) == percentile_bootstrap(diffs, 500, 1)
    lo, hi = percentile_bootstrap(diffs, 2000, 7)
    assert lo <= sum(diffs) / len(diffs) <= hi
    assert percentile_bootstrap([], 100, 1) == (None, None)


def test_synthetic_smoke_analysis_and_drift(tmp_path, raw_protocol, items) -> None:
    proto, res = _run(tmp_path, raw_protocol, items)
    a = analyze(res.records, proto.analysis, synthetic=True)
    assert a["evidence_class"] == "synthetic_fixture_not_a_measurement"
    ref = a["arms"]["reference"]
    assert ref["ungraded"] == 4 and ref["token_limit_hits"] == 4
    assert a["arms"]["candidate"]["errors"] == 1
    assert a["paired"]["items_paired"] == 6 and a["paired"]["items_excluded"] == []
    assert "No checkpoint is promoted" in a["decision_note"]
    (drift,) = drift_report(proto, res.manifest, res.records)
    assert drift["reproducibility"]["pair_categories"]["identical_output"] == 12


def test_exclude_item_policy_lists_items_with_missing_outcomes(tmp_path, raw_protocol, items):
    raw_protocol["analysis"]["missing_policy"] = "exclude_item"
    proto, res = _run(tmp_path, raw_protocol, items)
    a = analyze(res.records, proto.analysis, synthetic=True)
    excluded = a["paired"]["items_excluded"]
    assert "syn-dev-006" in excluded  # candidate error
    assert a["paired"]["items_paired"] + len(excluded) == a["paired"]["items_total"]


def _records(pairs: list[tuple[bool, bool]]) -> list[dict]:
    out = []
    for n, (x, y) in enumerate(pairs):
        for arm, ok in (("reference", x), ("candidate", y)):
            out.append(
                {
                    "arm": arm,
                    "item_id": f"i{n}",
                    "sample": 0,
                    "status": "graded",
                    "correct": ok,
                    "hit_token_limit": False,
                    "output_tokens": 1,
                    "output_sha256": str(ok),
                }
            )
    return out


ANALYSIS = {
    "primary_endpoint": "accuracy",
    "reference_arm": "reference",
    "candidate_arm": "candidate",
    "missing_policy": "count_as_incorrect",
    "min_meaningful_effect": 0.1,
    "bootstrap_resamples": 1000,
    "bootstrap_seed": 3,
}


PREREG = {"preregistered": True, "reasons": []}


def test_verdicts_are_conservative() -> None:
    run = functools.partial(analyze, prereg=PREREG)  # rule applied as if preregistered
    mixed = run(_records([(True, False), (False, True)] * 5), ANALYSIS, synthetic=True)
    assert mixed["verdict"] == "inconclusive"
    assert mixed["paired"]["mcnemar_exact"] == {
        "reference_only": 5,
        "candidate_only": 5,
        "p_value": 1.0,
    }
    worse = run(_records([(True, False)] * 10), ANALYSIS, synthetic=True)
    assert worse["verdict"] == "candidate_worse"
    better = run(_records([(False, True)] * 10 + [(True, True)] * 10), ANALYSIS, synthetic=False)
    assert better["verdict"] == "criteria_met_pending_human_review"
    assert better["evidence_class"] == "measured"
    tiny = {**ANALYSIS, "min_meaningful_effect": 0.9}
    small = run(_records([(False, True)] * 2 + [(True, True)] * 18), tiny, synthetic=True)
    assert small["verdict"] == "inconclusive"  # CI may exclude 0 but effect is below minimum
    with pytest.raises(ValueError, match="missing_policy"):
        run(_records([(True, True)]), {**ANALYSIS, "missing_policy": None}, synthetic=True)


# --- ledger --------------------------------------------------------------------------------


def _complete_record() -> dict:
    rec = json.loads((ROOT / "docs" / "wsp" / "ledger-template.json").read_text())
    rec.update(
        run_id="wsp-test",
        status="complete",
        operator="op1",
        research_lead="lead1",
        source_log_commit=COMMIT,
        base_model_revision=COMMIT,
        edited_model_revision=COMMIT,
        artifact_sha256={"model.safetensors": SHA},
        licenses_reviewed=True,
        environment={"container_digest": "sha256:" + SHA, "hardware": "h", "runtime_versions": "r"},
        dataset={"name": "d", "revision": "1", "split": "development", "content_hash": SHA},
        protocol_version="p1",
        grader_revision="g1",
        output_location_and_hash="research://x " + SHA,
        cost_and_compute="0",
    )
    return rec


def test_ledger_template_is_a_valid_planned_record() -> None:
    rec = json.loads((ROOT / "docs" / "wsp" / "ledger-template.json").read_text())
    assert validate_record(rec) == []
    assert validate_record(_complete_record()) == []


def test_ledger_rejects_incomplete_or_self_approved_records() -> None:
    rec = _complete_record()
    rec["base_model_revision"] = "main"
    rec["artifact_sha256"] = {"x": "md5"}
    rec["dataset"]["revision"] = "REPLACE"
    problems = " | ".join(validate_record(rec))
    assert "40-hex" in problems and "SHA-256" in problems and "placeholders" in problems

    rec = _complete_record()
    rec.update(human_decision="accept", decided_by="op1", decided_at="2026-11-13T10:00Z")
    assert any("differ" in p for p in validate_record(rec))
    rec.update(decided_by="lead1")
    assert any("differ" in p for p in validate_record(rec))
    rec.update(decided_by="owner1")
    assert validate_record(rec) == []

    planned = json.loads((ROOT / "docs" / "wsp" / "ledger-template.json").read_text())
    planned.update(human_decision="accept", decided_by="owner1", decided_at="2026-11-13T10:00Z")
    assert any("only valid for a complete" in p for p in validate_record(planned))

    failed = _complete_record()
    failed["status"] = "failed"
    assert any("deviations_and_errors" in p for p in validate_record(failed))
    extra = _complete_record()
    extra["promoted_to_production"] = True
    assert any("unknown fields" in p for p in validate_record(extra))


# --- checkpoint diff -----------------------------------------------------------------------


def _bf16(x: float) -> bytes:
    return struct.pack("<H", struct.unpack("<I", struct.pack("<f", x))[0] >> 16)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> Path:
    header, blob = {"__metadata__": {"format": "pt"}}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)
    return path


def _tensors(w: list[float]) -> dict:
    return {
        "layer.w": ("BF16", [2, 2], b"".join(_bf16(v) for v in w)),
        "layer.b": ("F32", [2], struct.pack("<2f", 0.5, -0.5)),
    }


def test_checkpoint_diff_identical_and_changed(tmp_path) -> None:
    base = _write_safetensors(tmp_path / "base.safetensors", _tensors([1.0, 2.0, 3.0, 4.0]))
    same = _write_safetensors(tmp_path / "same.safetensors", _tensors([1.0, 2.0, 3.0, 4.0]))
    edit = _write_safetensors(tmp_path / "edit.safetensors", _tensors([1.0, 2.5, 3.0, 2.0]))
    d = diff_checkpoints(base, same)
    assert d["identical"] and d["unchanged_tensors"] == 2
    d = diff_checkpoints(base, edit, chunk_bytes=3)  # chunk rounds to element size
    assert not d["identical"] and d["changed_elements_total"] == 2
    (t,) = d["changed_tensors"]
    assert t["name"] == "layer.w" and t["shape"] == [2, 2] and t["elements"] == 4
    assert t["max_abs_diff"] == 2.0 and t["l2_diff"] == pytest.approx((0.25 + 4) ** 0.5)
    assert t["sha256_base"] != t["sha256_candidate"]


def test_checkpoint_diff_reports_structure_changes_and_rejects_bad_files(tmp_path) -> None:
    base = _write_safetensors(tmp_path / "a.safetensors", _tensors([1.0, 2.0, 3.0, 4.0]))
    other = _tensors([1.0, 2.0, 3.0, 4.0])
    other["layer.b"] = ("F32", [1, 2], other["layer.b"][2])
    other["extra"] = ("I8", [1], b"\x01")
    cand = _write_safetensors(tmp_path / "b.safetensors", other)
    d = diff_checkpoints(base, cand)
    assert d["only_in_candidate"] == ["extra"]
    assert d["shape_or_dtype_mismatch"][0]["name"] == "layer.b"
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(SafetensorsError):
        diff_checkpoints(base, bad)
    wrong = _write_safetensors(tmp_path / "w.safetensors", {"t": ("F32", [3], b"\0" * 8)})
    with pytest.raises(SafetensorsError, match="data_offsets"):
        diff_checkpoints(base, wrong)


# --- CLI and isolation ---------------------------------------------------------------------


def test_cli_smoke_and_reanalysis(tmp_path, capsys) -> None:
    out = tmp_path / "smoke"
    assert wsp.main(["smoke", "--output", str(out)]) == 0
    smoke = json.loads(capsys.readouterr().out)
    assert smoke["analysis"]["evidence_class"] == "synthetic_fixture_not_a_measurement"
    proto = str(FIX / "synthetic_protocol.json")
    assert wsp.main(["analyze", "--run", str(out), "--protocol", proto]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again == json.loads((out / "analysis.json").read_text())
    assert wsp.main(["smoke", "--output", str(out)]) == 1  # immutable
    assert wsp.main(["ledger", str(ROOT / "docs" / "wsp" / "ledger-template.json")]) == 0
    assert wsp.main(["dataset", str(FIX / "synthetic_items.jsonl")]) == 0


_FORBIDDEN_IMPORTS = {
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
}


def test_harness_is_isolated_from_production_network_and_execution() -> None:
    files = sorted(WSP.rglob("*.py"))
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
                assert n.split(".")[0] not in _FORBIDDEN_IMPORTS, f"{path.name} imports {n}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "__import__"}, path.name
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                lowered = node.name.lower()
                assert not any(w in lowered for w in ("promote", "deploy", "publish", "upload"))


def test_harness_is_not_shipped_in_the_wheel() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "tools/wsp" not in pyproject and "wsp_eval" not in pyproject


# --- review fixes (2026-11-03) -------------------------------------------------------------


def test_planned_record_pins_are_validated_before_completion() -> None:
    rec = json.loads((ROOT / "docs" / "wsp" / "ledger-template.json").read_text())
    rec["base_model_revision"] = "2fc0636"  # short SHA: not a pin
    rec["artifact_sha256"] = {"hf:x": "not-a-digest"}
    rec["artifact_git_blob_sha1"] = {"hf:y": "z" * 40}
    rec["open_questions"] = "should be a list"
    problems = " | ".join(validate_record(rec))
    assert "base_model_revision" in problems
    assert "artifact_sha256 digests" in problems
    assert "artifact_git_blob_sha1 digests" in problems
    assert "open_questions must be a list" in problems


def test_safetensors_header_with_non_integer_offsets_is_rejected(tmp_path) -> None:
    raw = json.dumps({"t": {"dtype": "F32", "shape": [1], "data_offsets": ["0", "4"]}}).encode()
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    with pytest.raises(SafetensorsError, match="malformed"):
        diff_checkpoints(bad, bad)


def test_protocol_rejects_booleans_for_numeric_fields(raw_protocol) -> None:
    raw_protocol["analysis"]["min_meaningful_effect"] = True
    with pytest.raises(ProtocolError, match="wrong type"):
        parse_protocol(raw_protocol)


def test_recorded_ledger_entries_validate() -> None:
    records = sorted((ROOT / "docs" / "wsp" / "ledger").glob("*.json"))
    assert records
    for path in records:
        rec = json.loads(path.read_text())
        assert validate_record(rec) == [], path.name
        # Nothing has been measured or decided: pins only.
        assert rec["status"] == "planned" and rec["human_decision"] == "pending"

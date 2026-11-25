"""Runtime-explanation dataset tooling (tools/runtime/runtime_dataset.py). Tooling tests only:
these records are never training data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools" / "runtime"))

import runtime_dataset as rd  # noqa: E402

from tantular_cyber.runtime.analysis import prompt_version, select_records  # noqa: E402
from tantular_cyber.runtime.evaluation import load_suite  # noqa: E402
from tantular_cyber.runtime.importer import import_export  # noqa: E402
from tantular_cyber.runtime.retrieval import load_corpus  # noqa: E402

CORPUS = load_corpus()


def evidence(case_id: str) -> dict[str, Any]:
    root, manifest = load_suite()
    case = next(c for c in manifest["cases"] if c["id"] == case_id)
    return import_export((root / case["path"]).read_bytes(), import_id="ds").model_dump(mode="json")


def target(refs: list[str], cites: list[str]) -> dict[str, Any]:
    return {
        "summary": "Telemetri mencatat percobaan eksekusi; hasil akhir tidak tercatat.",
        "observations": [
            {"text": "git dijalankan empat kali dalam interval.", "evidence_refs": refs[:1]}
        ],
        "conditional_interpretations": [],
        "benign_alternatives": ["Pekerjaan developer biasa."],
        "limitations": ["Agregasi per interval, bukan urutan."],
        "recommended_investigation": ["Cocokkan {dev, ino} dengan paket git."],
        "mitigation_suggestions": [],
        "citations": cites,
        "confidence": "medium",
        "abstain": False,
    }


def example(**kw: Any) -> dict[str, Any]:
    ev = evidence("dev-benign-exec")
    from tantular_cyber.runtime.models import RuntimeEvidenceReport

    report = RuntimeEvidenceReport.model_validate(ev)
    refs = [r.ref for r in select_records(report)]
    cid = CORPUS["chunks"][0]["chunk_id"]
    ex: dict[str, Any] = {
        "id": "rt-train-0001",
        "split": "train",
        "family": "own_family_a",
        "scenario": "benign",
        "synthetic": True,
        "evidence": ev,
        "chunk_ids": [cid],
        "corpus_sha256": CORPUS["corpus_sha256"],
        "input_sha256": "0" * 64,
        "target": target(refs, [cid]),
        "target_origin": "human_written",
        "author": "auth_a",
        "reviews": [
            {
                "reviewer": "rev_b",
                "evidence_fidelity": "yes",
                "attempt_outcome_distinction": "yes",
                "unsupported_claim_present": "no",
                "indonesian_clarity": 4,
            }
        ],
        "review_state": "reviewed",
        "license": "Apache-2.0",
        "license_review_by": "lic_c",
        "generator_commit": "a" * 40,
        "prompt_version": prompt_version(),
    }
    ex.update(kw)
    parsed = rd.RuntimeExample.model_validate(ex)
    ex["input_sha256"] = rd.digest(rd.messages(parsed, CORPUS))
    return ex


def build(*examples: dict[str, Any]) -> dict[str, Any]:
    return rd.build_manifest([rd.RuntimeExample.model_validate(e) for e in examples], CORPUS)


def test_valid_reviewed_example_is_trainable_and_exported() -> None:
    manifest = build(example())
    assert manifest["training_authorized"] is False
    assert manifest["records"][0]["trainable"] is True
    lines = rd.export_dry_run(manifest, CORPUS)
    row = json.loads(lines[0])
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]


def test_drafts_never_exported() -> None:
    manifest = build(example(review_state="draft", reviews=[]))
    assert manifest["records"][0]["trainable"] is False
    assert rd.export_dry_run(manifest, CORPUS) == []


def test_dev_split_not_exported() -> None:
    manifest = build(example(split="dev"))
    assert rd.export_dry_run(manifest, CORPUS) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"family": "credential_file_open"}, "held-out runtime family"),
        ({"id": "accounts-lookup"}, "scanner case"),
        ({"input_sha256": "1" * 64}, "input_matches"),
        ({"prompt_version": "2" * 64}, "prompt_matches"),
    ],
)
def test_leakage_and_hash_gates(change: dict[str, Any], message: str) -> None:
    ex = example()
    ex.update(change)
    with pytest.raises(ValueError, match=message):
        build(ex)


def test_heldout_evidence_rejected() -> None:
    ex = example(evidence=evidence("held-dns-udp-benign"))
    with pytest.raises(ValueError, match=r"held-out runtime suite|input_matches|refs"):
        build(ex)


def test_family_split_duplicate_and_claim_checks() -> None:
    a = example()
    b = example(id="rt-dev-0002", split="dev")
    with pytest.raises(ValueError, match=r"crosses splits|duplicate target|two splits"):
        build(a, b)
    bad = example()
    bad["target"]["observations"][0]["text"] = "Host telah dikompromikan."
    with pytest.raises(ValueError, match="claims_clean"):
        build(bad)


def test_schema_and_review_rules() -> None:
    with pytest.raises(ValueError):
        rd.RuntimeExample.model_validate({**example(), "author": "rev_b"})
    with pytest.raises(ValueError):
        rd.RuntimeExample.model_validate({**example(), "redaction_canaries": ["realsecret"]})
    with pytest.raises(ValueError):
        rd.RuntimeExample.model_validate({**example(), "copied_third_party": True})
    ex = example()
    ex["target"]["extra"] = 1
    with pytest.raises(ValueError, match="exactly match"):
        build(ex)


def test_canary_in_target_rejected() -> None:
    ex = example(redaction_canaries=["tc-synthetic-ds-1"])
    ex["target"]["limitations"].append("nilai tc-synthetic-ds-1")
    with pytest.raises(ValueError, match="canary_free"):
        build(ex)


def test_run_manifest_not_runnable_and_cli(tmp_path: Path) -> None:
    m = rd.run_manifest()
    assert m["training_authorized"] is False and m["base_checkpoint"]["id"] == "UNRESOLVED"
    assert "ollama" in m["backends"] and "cannot train" in m["backends"]["ollama"]
    out = tmp_path / "run.json"
    assert rd.main(["run-manifest", "--out", str(out)]) == 0
    assert rd.main(["run-manifest", "--out", str(out)]) == 2  # never overwrite
    examples = tmp_path / "ex.json"
    examples.write_text(json.dumps([example()]))
    assert rd.main(["validate", str(examples), "--manifest", str(tmp_path / "m.json")]) == 0
    assert (
        rd.main(
            ["export", str(tmp_path / "m.json"), "--out", str(tmp_path / "x.jsonl"), "--dry-run"]
        )
        == 0
    )
    assert len((tmp_path / "x.jsonl").read_text().splitlines()) == 1

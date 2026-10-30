"""POC-007 Step 0 re-scoring tool: failure classification and fail-closed import."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import step0_rescore  # noqa: E402
from test_bench_llm_eval import finding  # noqa: E402

from tantular_cyber.benchmark.llm_eval import check_grounding  # noqa: E402


@pytest.mark.parametrize(
    ("passed", "detail", "category"),
    [
        (True, "3 citation(s) grounded", "grounded"),
        (None, "no code identifiers cited; needs human review", "nothing_cited"),
        (False, "not in evidence: sanitize_input()", "missing_identifiers"),
        (False, "line refs outside 10-14: 22", "line_refs_out_of_range"),
        (False, "not in evidence: x; line refs outside 1-3: 9", "both"),
    ],
)
def test_classify_covers_every_grounding_outcome(passed, detail, category) -> None:
    assert step0_rescore.classify(passed, detail) == category


def test_classify_refuses_unknown_failure_text() -> None:
    # A rubric whose wording changed must not be silently filed under some category.
    with pytest.raises(ValueError):
        step0_rescore.classify(False, "something else entirely")


@pytest.mark.parametrize(
    ("evidence", "category"),
    [
        ("Variabel `query` digabung dengan `username` pada baris 9.", "grounded"),
        ("Fungsi `sanitize_input()` tidak dipanggil.", "missing_identifiers"),
        ("Query dibangun pada baris 42.", "line_refs_out_of_range"),
        ("Fungsi `sanitize_input()` dipanggil pada baris 42.", "both"),
        ("Query tidak aman.", "nothing_cited"),
    ],
)
def test_classify_matches_the_real_check_wording(evidence: str, category: str) -> None:
    # The rubric is unchanged since eval/poc-005-76a9318, so its current wording is the
    # historical wording; if either changes, this fails instead of misfiling results.
    r = check_grounding(finding(confirmed_evidence=[evidence]))
    assert step0_rescore.classify(r.passed, r.detail) == category


def test_wilson_matches_the_benchmark_interval() -> None:
    assert step0_rescore.wilson(45, 80) == (0.4534, 0.6659)
    assert step0_rescore.wilson(0, 0) == (None, None)


def test_refuses_a_directory_that_is_not_a_checkout(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not a tantular-cyber checkout"):
        step0_rescore.import_historical(tmp_path)


def test_refuses_when_another_copy_is_already_imported(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "tantular_cyber"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    # tantular_cyber is imported by this test module, so the tool must refuse.
    with pytest.raises(SystemExit, match="already imported"):
        step0_rescore.import_historical(tmp_path)


def _valid_result() -> dict[str, object]:
    arm = {
        "categories": {
            "grounded": 45,
            "missing_identifiers": 33,
            "line_refs_out_of_range": 0,
            "both": 2,
            "nothing_cited": 0,
        },
        "scored": 80,
        "excluded": {},
        "evidence_grounded": {
            "passed": 45,
            "assessed": 80,
            "rate": 0.5625,
            "ci95": [0.4534, 0.6659],
        },
        "recorded_in_run": {"passed": 45, "assessed": 80},
        "matches_recorded": True,
    }
    return {
        "run": "optim",
        "historical_commit": "76a9318cc8131f3cff5f983eb98dcb59a70aef47",
        "semgrep_version_now": "1.182.0",
        "corpus_matches_header": True,
        "reconstruction": {
            "expected_findings": 40,
            "reconstructed": 40,
            "missing_ids": [],
            "selection_matches_header": True,
        },
        "recorded_summary_found": True,
        "expected_arms": ["a", "b"],
        "arms": {"a": arm, "b": copy.deepcopy(arm)},
    }


def _missing_finding(r):
    r["reconstruction"]["missing_ids"] = ["0312e5a6ec8ca7b5"]


def _selection_differs(r):
    r["reconstruction"]["selection_matches_header"] = False


def _corpus_differs(r):
    r["corpus_matches_header"] = False


def _recorded_differs(r):
    r["arms"]["b"]["matches_recorded"] = False


def _sample_excluded(r):
    r["arms"]["a"]["excluded"] = {"analysis_invalid": 1}


def _expected_arm_absent(r):
    del r["arms"]["b"]


def _unexpected_arm(r):
    r["arms"]["c"] = copy.deepcopy(r["arms"]["a"])


def _no_recorded_summary(r):
    r["recorded_summary_found"] = False
    r["expected_arms"] = []


VIOLATIONS = {
    "missing finding": (_missing_finding, "not reconstructed"),
    "selection differs": (_selection_differs, "selection differs"),
    "corpus differs": (_corpus_differs, "corpus id"),
    "recorded result differs": (_recorded_differs, "differs from the recorded result: b"),
    "sample excluded": (_sample_excluded, "1 sample(s) excluded"),
    "expected arm absent": (_expected_arm_absent, "expected arm missing from samples: b"),
    "unexpected arm": (_unexpected_arm, "not in the recorded results: c"),
    "no recorded summary": (_no_recorded_summary, "no recorded arm results"),
}


def test_a_fully_consistent_reconstruction_validates() -> None:
    assert step0_rescore.validate(_valid_result()) == []


@pytest.mark.parametrize("name", sorted(VIOLATIONS))
def test_each_violation_fails_validation(name: str) -> None:
    mutate, message = VIOLATIONS[name]
    r = _valid_result()
    mutate(r)
    failures = step0_rescore.validate(r)
    assert any(message in f for f in failures), failures


@pytest.mark.parametrize("name", [None, *sorted(VIOLATIONS)])
def test_main_exits_nonzero_on_any_violation(
    name: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = _valid_result()
    if name:
        VIOLATIONS[name][0](r)
    monkeypatch.setattr(step0_rescore, "import_historical", lambda checkout: object())
    monkeypatch.setattr(step0_rescore, "rescore", lambda hist, checkout, run: r)
    code = step0_rescore.main(
        ["--historical", str(tmp_path), "--run", str(tmp_path), "--output", str(tmp_path / "o")]
    )
    assert code == (0 if name is None else 1)
    written = json.loads((tmp_path / "o" / "rescoring.json").read_text())
    assert written["validation"]["passed"] is (name is None)

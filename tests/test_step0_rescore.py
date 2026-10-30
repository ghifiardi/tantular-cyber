"""POC-007 Step 0 re-scoring tool: failure classification and fail-closed import."""

from __future__ import annotations

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

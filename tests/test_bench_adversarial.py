from __future__ import annotations

import json
from pathlib import Path

from tantular_cyber.benchmark.adversarial import (
    run_adversarial,
    run_overclaim_suite,
    run_secret_suite,
)

CORPUS = Path(__file__).parents[1] / "benchmarks" / "corpus"


def test_committed_suites_have_no_unexpected_failures() -> None:
    suites = run_adversarial(CORPUS)
    assert [s.suite for s in suites] == ["overclaims", "secrets"]
    for suite in suites:
        assert suite.unexpected_failures == 0, [
            i.id for i in suite.items if i.status == "unexpected_failure"
        ]
        assert suite.counts.get("gap_closed", 0) == 0, "update known_blind_spot in the data"


def test_known_blind_spots_are_reported_not_hidden() -> None:
    oc, sec = run_adversarial(CORPUS)
    assert oc.counts["known_gap"] >= 1 and sec.counts["known_gap"] >= 1
    notes = {i.note for i in sec.items if i.status == "known_gap"}
    assert any("plain-letter" in n for n in notes) and any("hex" in n for n in notes)


def test_unicode_variants_are_covered() -> None:
    oc = run_overclaim_suite(CORPUS / "adversarial" / "overclaims.json")
    variants = {i.variant: i.status for i in oc.items}
    for v in ("zero_width", "nbsp", "fullwidth", "homoglyph"):
        assert variants[v] == "passed"


def test_status_logic(tmp_path: Path) -> None:
    data = {
        "suite": "secrets",
        "suite_version": "t",
        "items": [
            {
                "id": "a",
                "text": "password = 'hunter2hunter2'",
                "secret": "hunter2hunter2",
                "expect": "redacted",
                "known_blind_spot": False,
            },
            {
                "id": "b",
                "text": "x = 'abc'",
                "secret": "abc",
                "expect": "redacted",
                "known_blind_spot": False,
            },
            {
                "id": "c",
                "text": "password = 'hunter2hunter2'",
                "secret": "hunter2hunter2",
                "expect": "redacted",
                "known_blind_spot": True,
            },
            {
                "id": "d",
                "text": "x = 'abc'",
                "secret": "abc",
                "expect": "redacted",
                "known_blind_spot": True,
            },
        ],
    }
    path = tmp_path / "s.json"
    path.write_text(json.dumps(data))
    status = {i.id: i.status for i in run_secret_suite(path).items}
    assert status == {"a": "passed", "b": "unexpected_failure", "c": "gap_closed", "d": "known_gap"}

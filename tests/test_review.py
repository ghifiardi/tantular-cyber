"""Human-review ingestion. All ratings below are synthetic test data, not real reviews."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tantular_cyber.benchmark.review import (
    ReviewFile,
    cohen_kappa,
    render_review_summary,
    summarize_reviews,
    template,
)
from tantular_cyber.cli import EXIT_INPUT, EXIT_OK, main

QUEUE = [
    {
        "review_id": f"run1-f{i}",
        "automated_checks": {"evidence_grounded": i != 3, "no_unsupported_assertions": True},
    }
    for i in range(1, 5)
]
SHA = "0" * 64


def review(name: str, rows: dict[str, dict], role: str = "primary") -> ReviewFile:
    return ReviewFile.model_validate(
        {
            "schema": "tantular-cyber.human-review",
            "schema_version": "1.0.0",
            "reviewer": name,
            "role": role,
            "queue_sha256": SHA,
            "items": [{"review_id": k, **v} for k, v in rows.items()],
        }
    )


def test_cohen_kappa_known_values() -> None:
    a, b = ["yes", "yes", "no", "no"], ["yes", "no", "no", "no"]
    assert cohen_kappa(a, b, ["yes", "no"]) == 0.5
    assert cohen_kappa([1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 5], weighted=True) == 1.0
    assert cohen_kappa(["yes"] * 3, ["yes"] * 3, ["yes", "no"]) is None  # chance agreement 1
    assert cohen_kappa([], [], ["yes", "no"]) is None


def test_review_file_rejects_pii_shaped_input() -> None:
    with pytest.raises(ValidationError):
        review("jane.doe@example.com", {})
    with pytest.raises(ValidationError):
        ReviewFile.model_validate(
            {
                "schema": "tantular-cyber.human-review",
                "schema_version": "1.0.0",
                "reviewer": "R1",
                "items": [],
                "email": "x@y.z",
            }
        )
    with pytest.raises(ValidationError):
        review("R1", {"run1-f1": {"clarity_id": 9}})
    with pytest.raises(ValidationError):
        ReviewFile.model_validate(
            {
                "schema": "tantular-cyber.human-review",
                "schema_version": "1.0.0",
                "reviewer": "R1",
                "items": [{"review_id": "a"}, {"review_id": "a"}],
            }
        )


def test_two_reviewers_agreement_adjudication_and_proxies() -> None:
    r1 = review(
        "R1",
        {
            "run1-f1": {"evidence_faithful": "yes", "clarity_id": 4, "notes": "private remark"},
            "run1-f2": {"evidence_faithful": "yes", "clarity_id": 3},
            "run1-f3": {"evidence_faithful": "no", "clarity_id": 2},
            "run1-f4": {"evidence_faithful": "no"},
        },
    )
    r2 = review(
        "R2",
        {
            "run1-f1": {"evidence_faithful": "yes", "clarity_id": 5},
            "run1-f2": {"evidence_faithful": "no", "clarity_id": 3},
            "run1-f3": {"evidence_faithful": "no", "clarity_id": 1},
            "run1-f4": {},
        },
    )
    r3 = review("R3", {"run1-f2": {"evidence_faithful": "yes"}}, role="adjudicator")
    s = summarize_reviews(QUEUE, SHA, [r1, r2], [r3])
    ef = next(a for a in s.pairwise if a.dimension == "evidence_faithful")
    assert ef.n == 3 and ef.raw_agreement == round(2 / 3, 4) and ef.kappa_kind == "cohen"
    clarity = next(a for a in s.pairwise if a.dimension == "clarity_id")
    assert clarity.kappa_kind == "quadratic_weighted" and clarity.within_one == 1.0
    cons = next(c for c in s.consensus if c.dimension == "evidence_faithful")
    # f1 agree yes, f2 adjudicated yes, f3 agree no, f4 only one primary rating -> unresolved
    assert cons.consensus == {"no": 1, "yes": 2} and cons.unresolved == 1 and cons.missing == 0
    proxy = next(p for p in s.proxy_agreement if p.human_dimension == "evidence_faithful")
    assert (proxy.n, proxy.both_pass, proxy.both_fail) == (3, 2, 1)
    md = render_review_summary(s)
    assert "private remark" not in md and "free-text notes: 1" in md
    assert s.warnings == []


def test_single_reviewer_and_mismatched_queue_warn() -> None:
    r1 = review("R1", {"run1-f1": {"evidence_faithful": "yes"}, "other": {}})
    s = summarize_reviews(QUEUE, "1" * 64, [r1])
    assert s.pairwise == []
    assert any("fewer than two" in w for w in s.warnings)
    assert any("not in this queue" in w for w in s.warnings)
    assert any("different queue file" in w for w in s.warnings)


def test_template_and_cli_round_trip(tmp_path: Path) -> None:
    queue = tmp_path / "human_review.json"
    queue.write_text(json.dumps(QUEUE))
    form = template(QUEUE, "R1", SHA)
    assert all(v is None for i in form["items"] for k, v in i.items() if k != "review_id")
    t1 = tmp_path / "r1.json"
    assert (
        main(["review", "template", "--queue", str(queue), "--reviewer", "R1", "--output", str(t1)])
        == EXIT_OK
    )
    assert (
        main(
            [
                "review",
                "template",
                "--queue",
                str(queue),
                "--reviewer",
                "a@b.c",
                "--output",
                str(tmp_path / "bad.json"),
            ]
        )
        == EXIT_INPUT
    )
    data = json.loads(t1.read_text())
    data["items"][0]["evidence_faithful"] = "yes"
    t1.write_text(json.dumps(data))
    t2 = tmp_path / "r2.json"
    data["reviewer"] = "R2"
    t2.write_text(json.dumps(data))
    out = tmp_path / "summary"
    assert (
        main(
            [
                "review",
                "summarize",
                "--queue",
                str(queue),
                "--reviews",
                str(t1),
                str(t2),
                "--output",
                str(out),
            ]
        )
        == EXIT_OK
    )
    summary = json.loads((out / "review-summary.json").read_text())
    assert summary["reviewers"] == ["R1", "R2"] and not summary["warnings"]

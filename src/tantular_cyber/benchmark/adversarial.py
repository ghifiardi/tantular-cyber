"""Data-driven adversarial regression suites for the overclaim guard and redaction.

Each item declares the expected behavior. Items marked ``known_blind_spot`` document a
current gap: they are *expected to fail* and are reported as ``known_gap``. If one starts
passing it is reported as ``gap_closed`` so the data file can be updated. Only
``unexpected_failure`` fails the benchmark. Passing these suites is regression evidence,
not proof of comprehensive protection.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..analysis import enforce_claim_policy
from ..models import LLMExplanation
from ..security import redact

ItemStatus = Literal["passed", "unexpected_failure", "known_gap", "gap_closed"]


class ItemResult(BaseModel):
    id: str
    status: ItemStatus
    category: str
    variant: str = "plain"
    note: str = ""


class SuiteResult(BaseModel):
    suite: str
    suite_version: str
    total: int
    counts: dict[str, int]
    by_category: dict[str, dict[str, int]]
    items: list[ItemResult]

    @property
    def unexpected_failures(self) -> int:
        return self.counts.get("unexpected_failure", 0)


def _status(behaved_as_expected: bool, blind: bool) -> ItemStatus:
    if blind:
        return "gap_closed" if behaved_as_expected else "known_gap"
    return "passed" if behaved_as_expected else "unexpected_failure"


def _summarize(name: str, version: str, items: list[ItemResult]) -> SuiteResult:
    by_cat: dict[str, Counter[str]] = {}
    for it in items:
        by_cat.setdefault(it.category, Counter())[it.status] += 1
    return SuiteResult(
        suite=name,
        suite_version=version,
        total=len(items),
        counts=dict(sorted(Counter(i.status for i in items).items())),
        by_category={k: dict(sorted(v.items())) for k, v in sorted(by_cat.items())},
        items=items,
    )


def run_overclaim_suite(path: Path) -> SuiteResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    evidence = data["evidence"]
    results: list[ItemResult] = []
    for item in data["items"]:
        expl = LLMExplanation(
            summary="Ringkasan.",
            confirmed_evidence=[item["text"]],
            conditional_impact=[],
            remediation=[],
            confidence="high",
        )
        fixed, _ = enforce_claim_policy(expl, evidence)
        relocated = not fixed.confirmed_evidence
        ok = relocated if item["expect"] == "relocated" else not relocated
        results.append(
            ItemResult(
                id=item["id"],
                status=_status(ok, item["known_blind_spot"]),
                category=item["class"],
                variant=item.get("variant", "plain"),
                note=item.get("note", ""),
            )
        )
    return _summarize(data["suite"], data["suite_version"], results)


def run_secret_suite(path: Path) -> SuiteResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    results: list[ItemResult] = []
    for item in data["items"]:
        out, _ = redact(item["text"])
        if item["expect"] == "redacted":
            ok = item["secret"] not in out and out.count("\n") == item["text"].count("\n")
        else:
            ok = out == item["text"]
        results.append(
            ItemResult(
                id=item["id"],
                status=_status(ok, item["known_blind_spot"]),
                category=item["expect"],
                note=item.get("note", ""),
            )
        )
    return _summarize(data["suite"], data["suite_version"], results)


def run_adversarial(corpus_dir: Path) -> list[SuiteResult]:
    base = corpus_dir / "adversarial"
    return [run_overclaim_suite(base / "overclaims.json"), run_secret_suite(base / "secrets.json")]

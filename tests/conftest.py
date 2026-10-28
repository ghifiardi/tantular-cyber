from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tantular_cyber.scanner import RawFinding, ScannerRun

FIXTURES = Path(__file__).parent / "fixtures"
VULNERABLE = FIXTURES / "vulnerable"
CLEAN = FIXTURES / "clean"

requires_semgrep = pytest.mark.skipif(
    shutil.which("semgrep") is None, reason="semgrep not installed"
)


class FakeScanner:
    """Scanner double returning canned findings without running any process."""

    name = "fake-semgrep"

    def __init__(
        self,
        findings: Sequence[RawFinding] = (),
        *,
        succeeded: bool = True,
        errors: Sequence[str] = (),
    ) -> None:
        self.findings = list(findings)
        self.succeeded = succeeded
        self.errors = list(errors)
        self.calls: list[tuple[Path, list[str]]] = []

    def version(self) -> str | None:
        return "0.0-test"

    def scan(self, repo: Path, targets: Sequence[str]) -> ScannerRun:
        self.calls.append((repo, list(targets)))
        return ScannerRun(
            tool=self.name,
            version="0.0-test",
            rules=["test.rule"],
            targets=len(targets),
            exit_code=0 if self.succeeded else 2,
            findings=self.findings,
            errors=self.errors,
            succeeded=self.succeeded,
        )


class FakeLLM:
    """LLM double. ``responses`` may contain strings or exceptions (raised in order)."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, responses: Sequence[Any]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def chat(
        self, messages: list[dict[str, str]], format_schema: dict[str, Any] | None = None
    ) -> str:
        self.messages.append(messages)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def vulnerable_raw_findings() -> list[RawFinding]:
    common = dict(
        tool="semgrep",
        rule_id="tantular.python.sqli.dynamic-query-execute",
        path="app.py",
        message="SQL built with string concatenation",
        severity="high",
        cwe=("CWE-89",),
        confidence="medium",
    )
    return [
        RawFinding(start_line=14, end_line=14, **common),
        RawFinding(start_line=22, end_line=22, **common),
    ]


GOOD_EXPLANATION = json.dumps(
    {
        "summary": "Query SQL dibangun dengan penggabungan string.",
        "confirmed_evidence": ["Variabel username digabung langsung ke string SELECT."],
        "conditional_impact": [
            "Jika username berasal dari pengguna, data dapat dibaca tanpa izin."
        ],
        "remediation": ["Gunakan parameterized query dengan placeholder '?'."],
        "corrected_code": 'cursor.execute("SELECT ... WHERE username = ?", (username,))',
        "confidence": "medium",
        "limitations": ["Asal nilai username tidak terlihat."],
    }
)

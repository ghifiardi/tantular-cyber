"""Optional live smoke test against the local Ollama model (never runs in CI).

Run: TANTULAR_RUN_OLLAMA_SMOKE=1 pytest -m ollama -s
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from conftest import VULNERABLE, requires_semgrep

from tantular_cyber.config import OllamaSettings, ScanSettings
from tantular_cyber.pipeline import run_scan

pytestmark = [
    pytest.mark.ollama,
    requires_semgrep,
    pytest.mark.skipif(
        os.environ.get("TANTULAR_RUN_OLLAMA_SMOKE") != "1",
        reason="set TANTULAR_RUN_OLLAMA_SMOKE=1 to run against local Ollama",
    ),
]

# Outcomes that a simple SELECT concatenation does not prove.
UNPROVEN = re.compile(
    r"(?i)(bypass\w*\s+(autentikasi|otentikasi|login)|authentication bypass|"
    r"(menghapus|penghapusan)\s+data|drop\s+table|perintah\s+sistem\s+operasi|"
    r"remote code execution|\brce\b|command injection)"
)


def test_live_model_explains_cwe89_without_overclaiming(tmp_path: Path) -> None:
    report = run_scan(
        ScanSettings(repo=VULNERABLE, output=tmp_path / "out", ollama=OllamaSettings.from_env())
    ).report
    assert report.llm.available, report.llm.error
    assert report.totals.by_cwe.get("CWE-89") == 2
    for finding in report.findings:
        analysis = finding.analysis
        assert analysis.status == "completed", analysis.error
        expl = analysis.explanation
        assert expl is not None and analysis.verified is False
        assert expl.conditional_impact, "conditional impact must be present"
        for claim in expl.confirmed_evidence:
            assert not UNPROVEN.search(claim), f"unproven claim marked confirmed: {claim}"

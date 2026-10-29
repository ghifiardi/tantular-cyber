from __future__ import annotations

import json
from pathlib import Path

from conftest import GOOD_EXPLANATION, VULNERABLE, FakeLLM, FakeScanner, vulnerable_raw_findings

from tantular_cyber.config import ScanSettings
from tantular_cyber.models import ScanReport
from tantular_cyber.pipeline import run_scan
from tantular_cyber.reports import _code_block, render_markdown, write_reports

SCHEMA_FILE = Path(__file__).parents[1] / "docs" / "report-schema.json"


def make_report(tmp_path: Path, **overrides) -> tuple[ScanReport, Path]:
    settings = ScanSettings(repo=VULNERABLE, output=tmp_path / "out", **overrides)
    result = run_scan(
        settings,
        scanners=[FakeScanner(vulnerable_raw_findings())],
        llm_client_factory=lambda: FakeLLM([GOOD_EXPLANATION]),
    )
    return result.report, result.repo


def test_json_report_round_trips_and_is_versioned(tmp_path: Path) -> None:
    report, repo = make_report(tmp_path)
    written = write_reports(report, tmp_path / "out", repo=repo)
    data = json.loads(written["json"].read_text())
    assert data["schema_version"] == "1.3.0"
    assert data["report_type"] == "tantular-cyber.repository-scan"
    parsed = ScanReport.model_validate(data)
    assert parsed.totals.findings == 2
    f = parsed.findings[0]
    assert f.status == "unverified" and f.analysis.verified is False
    assert f.evidence.lines[0].line == f.start_line
    assert "14 |" in f.evidence.snippet
    for key in ("started_at", "finished_at", "scanners", "llm", "errors", "limitations"):
        assert key in data


def test_reports_contain_no_absolute_local_paths(tmp_path: Path) -> None:
    report, repo = make_report(tmp_path)
    written = write_reports(
        report, tmp_path / "out", repo=repo, raw_outputs={"semgrep": [{"path": f"{repo}/app.py"}]}
    )
    for path in written.values():
        text = path.read_text()
        assert str(repo) not in text
        assert str(Path.home()) not in text
    assert "<repo>/app.py" in written["raw:semgrep"].read_text()


def test_markdown_has_traceability_and_advisory_labels(tmp_path: Path) -> None:
    report, _ = make_report(tmp_path)
    md = render_markdown(report)
    f = report.findings[0]
    assert f"`{f.id}`" in md
    assert "CWE-89" in md and "app.py" in md
    assert "Dampak kondisional (belum terbukti)" in md
    assert "advisory" in md and "unverified" in md
    assert "## Batasan" in md


def test_markdown_escapes_untrusted_text_and_code_fences(tmp_path: Path) -> None:
    report, _ = make_report(tmp_path)
    report.findings[0].message = "<script>alert(1)</script> **bold** | pipe"
    md = render_markdown(report)
    assert "<script>" not in md
    block = _code_block("x = '```'\n````")
    assert block.startswith("`````\n") and block.endswith("\n`````")


def test_committed_json_schema_matches_models() -> None:
    """docs/report-schema.json is the GATRA-facing contract.

    Regenerate with `tantular-cyber schema > docs/report-schema.json`.
    """
    assert json.loads(SCHEMA_FILE.read_text()) == ScanReport.model_json_schema()

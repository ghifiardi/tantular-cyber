"""Integration tests: full pipeline with mocked scanner/LLM, plus CLI runs with real semgrep."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from conftest import (
    CLEAN,
    GOOD_EXPLANATION,
    VULNERABLE,
    FakeLLM,
    FakeScanner,
    requires_semgrep,
    vulnerable_raw_findings,
)

from tantular_cyber.cli import EXIT_DEGRADED, EXIT_FAILED, EXIT_INPUT, EXIT_OK, main
from tantular_cyber.config import OllamaSettings, ScanSettings
from tantular_cyber.ollama_client import OllamaUnavailable
from tantular_cyber.pipeline import run_scan
from tantular_cyber.security import PathSafetyError


def settings(tmp_path: Path, repo: Path = VULNERABLE, **kw) -> ScanSettings:
    return ScanSettings(repo=repo, output=tmp_path / "out", **kw)


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_complete_run_with_mocked_tools(tmp_path: Path) -> None:
    llm = FakeLLM([GOOD_EXPLANATION])
    scanner = FakeScanner(vulnerable_raw_findings())
    report = run_scan(settings(tmp_path), scanners=[scanner], llm_client_factory=lambda: llm).report
    assert report.status == "complete" and report.status_reasons == []
    assert report.totals.findings == 2
    assert report.totals.by_cwe == {"CWE-89": 2}
    assert report.totals.analyses_by_status == {"completed": 2}
    assert scanner.calls[0][1] == ["app.py"]
    assert len(llm.messages) == 2
    first = report.findings[0]
    assert first.evidence.context and "13 |" in first.evidence.context  # query construction


def test_finding_ids_are_deterministic(tmp_path: Path) -> None:
    def ids() -> list[str]:
        r = run_scan(
            settings(tmp_path, llm_enabled=False),
            scanners=[FakeScanner(vulnerable_raw_findings() * 2)],
        ).report
        return [f.id for f in r.findings]

    first = ids()
    assert first == ids()
    assert len(first) == 2  # duplicates collapsed


def test_llm_unavailable_degrades_but_keeps_scanner_findings(tmp_path: Path) -> None:
    llm = FakeLLM([OllamaUnavailable("connection refused")])
    report = run_scan(
        settings(tmp_path),
        scanners=[FakeScanner(vulnerable_raw_findings())],
        llm_client_factory=lambda: llm,
    ).report
    assert report.status == "degraded"
    assert report.totals.findings == 2
    assert report.totals.analyses_by_status == {"error": 1, "skipped": 1}
    assert len(llm.messages) == 1  # stops calling a dead endpoint
    assert any(e.stage == "llm" for e in report.errors)


def test_invalid_model_output_degrades(tmp_path: Path) -> None:
    report = run_scan(
        settings(tmp_path),
        scanners=[FakeScanner(vulnerable_raw_findings())],
        llm_client_factory=lambda: FakeLLM(["not json"]),
    ).report
    assert report.status == "degraded"
    assert report.totals.analyses_by_status == {"invalid_output": 2}


def test_llm_finding_limit_is_reported(tmp_path: Path) -> None:
    report = run_scan(
        settings(tmp_path, max_llm_findings=1),
        scanners=[FakeScanner(vulnerable_raw_findings())],
        llm_client_factory=lambda: FakeLLM([GOOD_EXPLANATION]),
    ).report
    assert report.totals.analyses_by_status == {"completed": 1, "skipped": 1}
    assert report.status == "degraded"


def test_clean_repo_yields_no_invented_findings(tmp_path: Path) -> None:
    llm = FakeLLM([GOOD_EXPLANATION])
    report = run_scan(
        settings(tmp_path, repo=CLEAN), scanners=[FakeScanner([])], llm_client_factory=lambda: llm
    ).report
    assert report.status == "complete"
    assert report.findings == [] and report.totals.findings == 0
    assert llm.messages == []  # the LLM is never asked to invent findings


def test_scanner_failure_marks_report_failed(tmp_path: Path) -> None:
    report = run_scan(
        settings(tmp_path, llm_enabled=False),
        scanners=[FakeScanner(succeeded=False, errors=["semgrep exited 2"])],
    ).report
    assert report.status == "failed"
    assert report.errors[0].stage == "scanner"


def test_findings_outside_eligible_targets_are_discarded(tmp_path: Path) -> None:
    raw = vulnerable_raw_findings()[0]
    stray = type(raw)(**{**raw.__dict__, "path": ".env"})
    report = run_scan(settings(tmp_path, llm_enabled=False), scanners=[FakeScanner([stray])]).report
    assert report.findings == []
    assert report.errors[0].stage == "normalize"


def test_remote_ollama_rejected_without_opt_in(tmp_path: Path) -> None:
    s = settings(tmp_path, ollama=OllamaSettings(base_url="http://203.0.113.5:11434"))
    report = run_scan(s, scanners=[FakeScanner(vulnerable_raw_findings())]).report
    assert report.status == "degraded"
    assert report.llm.available is False
    assert "not localhost" in (report.llm.error or "")


def test_output_inside_repo_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(PathSafetyError):
        run_scan(ScanSettings(repo=repo, output=repo / "out"), scanners=[FakeScanner()])


def test_cli_rejects_bad_input_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["scan", "--repo", str(tmp_path / "missing"), "--output", str(tmp_path)]) == EXIT_INPUT
    )
    assert "does not exist" in capsys.readouterr().err


@requires_semgrep
@pytest.mark.semgrep
def test_cli_end_to_end_with_ollama_down_is_explicitly_degraded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "scan",
            "--repo",
            str(VULNERABLE),
            "--output",
            str(out),
            "--ollama-url",
            f"http://127.0.0.1:{closed_port()}",
            "--ollama-timeout",
            "2",
        ]
    )
    assert code == EXIT_DEGRADED
    assert "DEGRADED" in capsys.readouterr().err
    data = json.loads((out / "report.json").read_text())
    assert data["status"] == "degraded"
    assert data["llm"]["available"] is False
    assert data["totals"]["by_cwe"] == {"CWE-89": 2}
    lines = sorted((f["path"], f["start_line"]) for f in data["findings"])
    assert lines == [("app.py", 14), ("app.py", 22)]
    md = (out / "report.md").read_text()
    assert "DEGRADED" in md and "app.py:14" in md


@requires_semgrep
@pytest.mark.semgrep
def test_cli_clean_fixture_complete_with_no_findings(tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "scan",
            "--repo",
            str(CLEAN),
            "--output",
            str(out),
            "--keep-raw",
            "--ollama-url",
            f"http://127.0.0.1:{closed_port()}",
        ]
    )
    data = json.loads((out / "report.json").read_text())
    # No findings => the LLM is never contacted, so a down Ollama does not degrade the run.
    assert code == EXIT_OK and data["status"] == "complete"
    assert data["findings"] == []
    assert (out / "semgrep-raw.json").exists()


def test_cli_scanner_missing_is_failed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "scan",
            "--repo",
            str(VULNERABLE),
            "--output",
            str(out),
            "--no-llm",
            "--semgrep-bin",
            str(tmp_path / "no-such-semgrep"),
        ]
    )
    assert code == EXIT_FAILED
    assert json.loads((out / "report.json").read_text())["status"] == "failed"


def test_llm_performance_is_aggregated(tmp_path: Path) -> None:
    from tantular_cyber.models import LLMMetrics

    class MeteredLLM(FakeLLM):
        def chat(self, messages, format_schema=None):
            n = len(self.messages)
            self.last_metrics = LLMMetrics(
                wall_s=10.0 + n * 30,
                load_s=30.0 if n == 0 else 0.1,
                prompt_tokens=800,
                output_tokens=400,
                eval_s=40.0,
                output_tokens_per_s=10.0,
            )
            return super().chat(messages, format_schema)

    report = run_scan(
        settings(tmp_path),
        scanners=[FakeScanner(vulnerable_raw_findings())],
        llm_client_factory=lambda: MeteredLLM([GOOD_EXPLANATION]),
    ).report
    perf = report.llm.performance
    assert perf is not None
    assert (perf.calls, perf.total_wall_s, perf.min_wall_s, perf.max_wall_s) == (
        2,
        50.0,
        10.0,
        40.0,
    )
    assert perf.first_call_load_s == 30.0 and perf.total_load_s == 30.1
    assert (perf.prompt_tokens, perf.output_tokens) == (1600, 800)
    assert all(f.analysis.metrics is not None for f in report.findings)

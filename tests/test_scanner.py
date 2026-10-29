from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import CLEAN, VULNERABLE, requires_semgrep

from tantular_cyber.config import default_rules
from tantular_cyber.scanner import (
    SemgrepScanner,
    declared_rule_ids,
    extract_cwes,
    normalize_rule_id,
    parse_semgrep_json,
)

RULE_ID = "tantular.python.sqli.dynamic-query-execute"

SAMPLE_OUTPUT: dict[str, Any] = {
    "version": "1.143.0",
    "results": [
        {
            "check_id": f"src.tantular_cyber.rules.{RULE_ID}",
            "path": "app.py",
            "start": {"line": 14, "col": 5},
            "end": {"line": 14, "col": 26},
            "extra": {
                "message": "SQL statement is built with string concatenation",
                "severity": "ERROR",
                "lines": "requires login",
                "metadata": {
                    "cwe": ["CWE-89: Improper Neutralization ..."],
                    "confidence": "MEDIUM",
                },
            },
        },
        {
            "check_id": "x",
            "path": "../outside.py",
            "start": {"line": 1},
            "end": {"line": 1},
            "extra": {},
        },
        {"check_id": "broken"},
    ],
    "errors": [
        {"level": "warn", "type": "Syntax error", "message": "parse failure in /abs/app.py"}
    ],
}


def completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class RecordingRunner:
    def __init__(self, scan_result: subprocess.CompletedProcess[str] | Exception) -> None:
        self.scan_result = scan_result
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((cmd, kwargs))
        if "--version" in cmd:
            return completed("1.143.0\n")
        if isinstance(self.scan_result, Exception):
            raise self.scan_result
        return self.scan_result


def test_extract_cwes_handles_strings_lists_and_duplicates() -> None:
    assert extract_cwes("CWE-89: SQLi") == ("CWE-89",)
    assert extract_cwes(["CWE-089 x", "cwe-79", "CWE-89"]) == ("CWE-89", "CWE-79")
    assert extract_cwes(None) == ()


def test_rule_ids_are_declared_and_prefix_is_normalized() -> None:
    ids = declared_rule_ids(default_rules())
    assert ids[0] == RULE_ID and len(ids) == len(set(ids))  # CWE-89 ruleset first, ids unique
    assert all(i.startswith("tantular.") for i in ids)
    assert normalize_rule_id(f"src.tantular_cyber.rules.{RULE_ID}", ids) == RULE_ID
    assert normalize_rule_id("other.rule", ids) == "other.rule"


def test_parse_semgrep_json_normalizes_and_rejects_out_of_scope(tmp_path: Path) -> None:
    findings, errors = parse_semgrep_json(SAMPLE_OUTPUT, tmp_path, [RULE_ID])
    assert len(findings) == 1
    f = findings[0]
    assert (f.rule_id, f.path, f.start_line, f.end_line) == (RULE_ID, "app.py", 14, 14)
    assert f.severity == "high"
    assert f.cwe == ("CWE-89",)
    assert f.confidence == "medium"
    assert any("outside repository scope" in e for e in errors)
    assert any("malformed semgrep result" in e for e in errors)
    assert any("Syntax error" in e for e in errors)


def test_command_is_argument_list_offline_and_uses_local_rules() -> None:
    scanner = SemgrepScanner(default_rules())
    cmd = scanner.build_command(["-weird.py", "app.py"])
    assert cmd[:2] == ["semgrep", "scan"]
    assert "--metrics=off" in cmd and "--disable-version-check" in cmd
    assert cmd[cmd.index("--config") + 1].endswith("python-sqli.yml")
    assert cmd[-3:] == ["--", "-weird.py", "app.py"]  # targets can never be parsed as flags
    env = SemgrepScanner.environment()
    assert env["SEMGREP_SEND_METRICS"] == "off"
    assert not any(k.startswith("SEMGREP_APP") for k in env)


def test_scan_success_uses_no_shell_and_repo_cwd(tmp_path: Path) -> None:
    runner = RecordingRunner(completed(json.dumps(SAMPLE_OUTPUT)))
    run = SemgrepScanner(default_rules(), runner=runner).scan(tmp_path, ["app.py"])
    assert run.succeeded and run.version == "1.143.0" and run.exit_code == 0
    assert len(run.findings) == 1
    scan_cmd, kwargs = runner.calls[-1]
    assert isinstance(scan_cmd, list)
    assert kwargs.get("shell") is None
    assert kwargs["cwd"] == tmp_path
    assert kwargs["timeout"] > 0


def test_scan_tool_error_is_distinguished_from_findings(tmp_path: Path) -> None:
    runner = RecordingRunner(completed("", returncode=2, stderr=f"fatal in {tmp_path}"))
    run = SemgrepScanner(default_rules(), runner=runner).scan(tmp_path, ["app.py"])
    assert not run.succeeded
    assert run.exit_code == 2
    assert run.findings == []
    assert "exited with code 2" in run.errors[0]
    assert str(tmp_path) not in run.errors[0]


def test_scan_timeout_and_missing_binary(tmp_path: Path) -> None:
    run = SemgrepScanner(
        default_rules(), runner=RecordingRunner(subprocess.TimeoutExpired("semgrep", 1))
    ).scan(tmp_path, ["app.py"])
    assert not run.succeeded and "timed out" in run.errors[0]

    def missing(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(cmd[0])

    run = SemgrepScanner(default_rules(), runner=missing).scan(tmp_path, ["app.py"])
    assert not run.succeeded and "not found" in run.errors[0]


@requires_semgrep
@pytest.mark.semgrep
def test_real_semgrep_detects_cwe89_in_vulnerable_fixture() -> None:
    run = SemgrepScanner(default_rules()).scan(VULNERABLE.resolve(), ["app.py"])
    assert run.succeeded, run.errors
    assert sorted(f.start_line for f in run.findings) == [14, 22]
    assert all(f.cwe == ("CWE-89",) and f.rule_id == RULE_ID for f in run.findings)


@requires_semgrep
@pytest.mark.semgrep
def test_real_semgrep_finds_nothing_in_clean_fixture() -> None:
    run = SemgrepScanner(default_rules()).scan(CLEAN.resolve(), ["app.py"])
    assert run.succeeded, run.errors
    assert run.findings == []

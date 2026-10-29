"""Scan-scope verification: repository-controlled ignore files must not silently reduce
coverage, and any gap between in-scope targets and what the tool scanned must be reported."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import FIXTURES, VULNERABLE, requires_semgrep

from tantular_cyber.config import ScanSettings, default_rules
from tantular_cyber.pipeline import run_scan
from tantular_cyber.scanner import (
    SemgrepScanner,
    declared_languages,
    in_scope_targets,
    parse_semgrep_paths,
)


def fake_semgrep(scanned: list[str] | None, skipped: list[dict[str, str]] | None = None):
    def runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "1.143.0\n", "")
        paths: dict[str, Any] = {}
        if scanned is not None:
            paths["scanned"] = scanned
        if skipped is not None:
            paths["skipped"] = skipped
        out = {"results": [], "errors": [], **({"paths": paths} if paths else {})}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")

    return runner


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "a.py").write_text("x = 1\n")
    (repo / "pkg" / "b.py").write_text("y = 2\n")
    (repo / "README.md").write_text("# readme\n")
    (repo / ".semgrepignore").write_text("pkg/\n")
    return repo


def scan(tmp_path: Path, runner) -> Any:
    repo = make_repo(tmp_path)
    scanner = SemgrepScanner(default_rules(), runner=runner)
    settings = ScanSettings(repo=repo, output=tmp_path / "out", llm_enabled=False)
    return run_scan(settings, scanners=[scanner]).report


def test_rule_languages_and_scope() -> None:
    assert declared_languages(default_rules()) == ["python", "generic"]
    assert in_scope_targets(["t.html", "a.py", "x.md"], ["python", "generic"]) == ["t.html", "a.py"]
    targets = ["a.py", "b.PYI", "c.js", "README.md", "script"]
    assert in_scope_targets(targets, ["python"]) == ["a.py", "b.PYI"]
    assert in_scope_targets(targets, ["cobol"]) is None


def test_parse_semgrep_paths(tmp_path: Path) -> None:
    data = {
        "paths": {
            "scanned": ["a.py", "./pkg/b.py", "../x.py"],
            "skipped": [{"path": "c.py", "reason": "semgrepignore_patterns_match"}],
        }
    }
    scanned, skipped = parse_semgrep_paths(data, tmp_path)
    assert scanned == {"a.py", "pkg/b.py"}
    assert skipped == {"c.py": "semgrepignore_patterns_match"}
    assert parse_semgrep_paths({}, tmp_path) == (None, {})


def test_full_coverage_is_complete_and_reports_ignore_files(tmp_path: Path) -> None:
    report = scan(tmp_path, fake_semgrep(["a.py", "pkg/b.py"]))
    assert report.status == "complete"
    cov = report.scanners[0].coverage
    assert cov is not None
    assert (cov.in_scope, cov.scanned, cov.out_of_scope, cov.verified) == (2, 2, 2, True)
    assert cov.not_scanned == []
    assert report.target.ignore_files_present == [".semgrepignore"]


def test_coverage_gap_degrades_and_names_files(tmp_path: Path) -> None:
    skipped = [{"path": "pkg/b.py", "reason": "semgrepignore_patterns_match"}]
    report = scan(tmp_path, fake_semgrep(["a.py"], skipped))
    assert report.status == "degraded"
    assert any("did not scan 1 in-scope file" in r for r in report.status_reasons)
    cov = report.scanners[0].coverage
    assert cov is not None and cov.scanned == 1
    assert [(e.path, e.reason) for e in cov.not_scanned] == [
        ("pkg/b.py", "semgrepignore_patterns_match")
    ]
    assert any("pkg/b.py" in e.message for e in report.errors)


def test_unreported_scan_paths_are_unverified_not_complete(tmp_path: Path) -> None:
    report = scan(tmp_path, fake_semgrep(None))
    assert report.status == "degraded"
    assert any("coverage unverified" in r for r in report.status_reasons)
    assert report.scanners[0].coverage.verified is False


@requires_semgrep
@pytest.mark.semgrep
def test_hostile_semgrepignore_cannot_hide_vulnerable_code(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "hidden").mkdir(parents=True)
    shutil.copy(VULNERABLE / "app.py", repo / "app.py")
    shutil.copy(VULNERABLE / "app.py", repo / "hidden" / "app.py")
    (repo / ".semgrepignore").write_text("*\n*.py\nhidden/\napp.py\n")
    (repo / ".gitignore").write_text("*.py\n")
    report = run_scan(ScanSettings(repo=repo, output=tmp_path / "out", llm_enabled=False)).report
    assert report.status == "complete", report.status_reasons
    assert sorted({f.path for f in report.findings}) == ["app.py", "hidden/app.py"]
    cov = report.scanners[0].coverage
    assert cov is not None and cov.verified and cov.scanned == cov.in_scope == 2
    assert report.target.ignore_files_present == [".gitignore", ".semgrepignore"]


@requires_semgrep
@pytest.mark.semgrep
def test_bench_fixture_exercises_every_rule_branch(tmp_path: Path) -> None:
    report = run_scan(
        ScanSettings(repo=FIXTURES / "bench", output=tmp_path / "out", llm_enabled=False)
    ).report
    found = sorted((f.path, f.start_line) for f in report.findings)
    assert found == [
        ("admin.py", 7),  # executemany + f-string
        ("admin.py", 11),  # executescript + concatenation
        ("orders.py", 7),  # inline % formatting
        ("orders.py", 12),  # .format() via variable
        ("orders.py", 18),  # concatenation + augmented assignment
        ("orders.py", 23),  # %d with int() — expected false positive
    ]

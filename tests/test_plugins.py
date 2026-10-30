"""Plugin contract (API 1.0): validation, negotiation, normalization — and no execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeScanner
from pydantic import ValidationError

from tantular_cyber.cli import EXIT_INPUT, EXIT_OK, main
from tantular_cyber.config import ScanSettings
from tantular_cyber.models import SCHEMA_VERSION, ScanReport
from tantular_cyber.pipeline import run_scan
from tantular_cyber.plugins import (
    BUILTIN_SCANNERS,
    PLUGIN_API_VERSION,
    PluginExecutionDisabled,
    PluginManifest,
    PluginOutputError,
    execute_plugin,
    negotiate,
    normalize_plugin_output,
)

REPO = Path(__file__).parent / "fixtures" / "vulnerable"


def manifest(**overrides) -> dict:
    base = {
        "plugin_api": "1.0",
        "id": "example-scanner",
        "name": "Example",
        "version": "0.1.0",
        "publisher": "example",
        "license": "Apache-2.0",
        "languages": ["python"],
        "cwe": ["CWE-89"],
        "rule_id_prefix": "plugin.example-scanner.",
        "entrypoint": {"command": ["example-scanner", "--json"]},
    }
    return {**base, **overrides}


def output(**finding) -> dict:
    f = {
        "rule_id": "plugin.example-scanner.sqli",
        "path": "app.py",
        "start_line": 14,
        "end_line": 14,
        "message": "query built from input",
        "severity": "high",
        "cwe": ["CWE-89"],
    }
    f.update(finding)
    return {
        "protocol": "tantular-plugin-output/1",
        "plugin_id": "example-scanner",
        "plugin_version": "0.1.0",
        "findings": [f],
    }


def test_report_schema_version_is_the_documented_contract() -> None:
    assert SCHEMA_VERSION == "1.3.0"
    doc = (Path(__file__).parents[1] / "docs" / "plugin-api.md").read_text()
    assert f"report schema **{SCHEMA_VERSION}**" in doc
    assert f"API **{PLUGIN_API_VERSION}**" in doc


@pytest.mark.parametrize(
    "bad",
    [
        {"permissions": {"network": True}},
        {"permissions": {"write": "repo"}},
        {"permissions": {"spawn_processes": True}},
        {"permissions": {"read_roots": ["home"]}},
        {"permissions": {"environment": ["GITHUB_TOKEN"]}},
        {"permissions": {"limits": {"max_memory_mb": 10**6}}},
        {"rule_id_prefix": "tantular."},
        {"trust": "builtin"},
        {"cwe": ["89"]},
        {"unexpected": 1},
    ],
)
def test_manifest_rejects_unsafe_or_malformed_requests(bad: dict) -> None:
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(manifest(**bad))


def test_negotiation_by_major_and_minor() -> None:
    assert negotiate(PluginManifest.model_validate(manifest())).compatible
    assert not negotiate(PluginManifest.model_validate(manifest(plugin_api="2.0"))).compatible
    newer = negotiate(PluginManifest.model_validate(manifest(plugin_api="1.9")))
    assert not newer.compatible and "host provides" in newer.reason


def test_third_party_execution_always_refused() -> None:
    m = PluginManifest.model_validate(manifest())
    with pytest.raises(PluginExecutionDisabled, match="not a sandbox"):
        execute_plugin(m, REPO)


def test_normalized_plugin_findings_flow_into_a_valid_report(tmp_path: Path) -> None:
    m = PluginManifest.model_validate(manifest())
    raws = normalize_plugin_output(m, output(), ["app.py"])
    assert raws[0].tool == "plugin:example-scanner"
    report = run_scan(
        ScanSettings(repo=REPO, output=tmp_path / "out", llm_enabled=False),
        scanners=[FakeScanner(raws)],
    ).report
    again = ScanReport.model_validate_json(report.model_dump_json())
    assert again.findings[0].rule_id == "plugin.example-scanner.sqli"
    assert again.findings[0].analysis.status == "disabled"  # advisory output stays separate


@pytest.mark.parametrize(
    ("finding", "message"),
    [
        ({"rule_id": "tantular.python.sqli.x"}, "rule id"),
        ({"path": "/etc/passwd"}, "path"),
        ({"path": "../outside.py"}, "path"),
        ({"path": "not_scanned.py"}, "path"),
        ({"start_line": 9, "end_line": 3}, "end_line"),
        ({"cwe": ["CWE-78"]}, "not declared"),
        ({"severity": "catastrophic"}, "invalid plugin output"),
        ({"analysis": "safe"}, "invalid plugin output"),
    ],
)
def test_plugin_output_fails_closed(finding: dict, message: str) -> None:
    m = PluginManifest.model_validate(manifest())
    with pytest.raises(PluginOutputError, match=message):
        normalize_plugin_output(m, output(**finding), ["app.py"])


def test_plugin_output_identity_must_match() -> None:
    m = PluginManifest.model_validate(manifest())
    with pytest.raises(PluginOutputError, match="identity"):
        normalize_plugin_output(m, {**output(), "plugin_id": "other-plugin"}, ["app.py"])


def test_builtin_descriptors_match_the_real_scanners() -> None:
    from tantular_cyber.callgraph import RULES, TOOL
    from tantular_cyber.scanner import SemgrepScanner

    ids = {b.tool_name for b in BUILTIN_SCANNERS}
    assert ids == {SemgrepScanner.name, TOOL}
    callgraph = next(b for b in BUILTIN_SCANNERS if b.tool_name == TOOL)
    assert all(r.startswith(callgraph.rule_id_prefix) for r in RULES.values())
    assert set(callgraph.cwe) == set(RULES)


def test_plugins_cli_list_and_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plugins", "list"]) == EXIT_OK
    policy = json.loads(capsys.readouterr().out)
    assert policy["third_party_execution"].startswith("disabled")
    good = tmp_path / "plugin.json"
    good.write_text(json.dumps(manifest()))
    assert main(["plugins", "validate", str(good)]) == EXIT_OK
    assert "execution: DISABLED" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(manifest(permissions={"network": True})))
    assert main(["plugins", "validate", str(bad)]) == EXIT_INPUT


def test_no_plugin_discovery_in_source() -> None:
    """Nothing loads entry points or imports plugin code dynamically."""
    import ast

    src = Path(__file__).parents[1] / "src" / "tantular_cyber"
    forbidden = {"eval", "exec", "__import__", "import_module", "entry_points", "load_module"}
    hits = []
    for path in src.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                # self.eval(...) is the callgraph's own evaluator, not the builtin.
                if name in forbidden and not (
                    isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "self"
                ):
                    hits.append(f"{path.name}:{node.lineno} {name}")
    assert hits == []

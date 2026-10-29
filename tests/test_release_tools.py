"""Release tooling: lock consistency, license audit, deterministic SBOM, bundled data."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import license_audit  # noqa: E402
import sbom  # noqa: E402


def _name(req: str) -> str:
    return re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0].lower().replace("_", "-")


def test_runtime_lock_pins_every_runtime_dependency_with_hashes() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lock = ROOT / "requirements" / "runtime.lock.txt"
    pinned = {n.lower() for n, _ in license_audit.locked(lock)}
    assert {_name(r) for r in project["dependencies"]} <= pinned
    text = lock.read_text()
    for block in re.split(r"\n(?=[A-Za-z])", text.strip()):
        if "==" in block.splitlines()[0]:
            assert "--hash=sha256:" in block, block.splitlines()[0]
    assert "semgrep" not in pinned  # external, user-managed dependency


def test_license_audit_accepts_allowlisted_and_rejects_missing(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    pyd = metadata.version("pydantic")
    lock.write_text(f"pydantic=={pyd} \\\n    --hash=sha256:00\nnot-a-real-pkg-xyz==1.0\n")
    rows, problems = license_audit.audit(lock)
    assert rows[0]["license"] == "MIT" and rows[0]["class"] == "permissive"
    assert problems == ["not-a-real-pkg-xyz==1.0: not installed in this environment"]


def test_license_classification_rejects_copyleft() -> None:
    assert "GPL-3.0-only" not in license_audit.ALLOWED
    assert "LGPL-2.1-only" not in license_audit.ALLOWED  # Semgrep is never a bundled dependency
    assert license_audit.ALLOWED["MPL-2.0"].startswith("weak copyleft")


def test_sbom_is_deterministic_and_declares_external_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    wheel = tmp_path / "tantular_cyber-9.9.9-py3-none-any.whl"
    wheel.write_bytes(b"not really a wheel")
    lock = tmp_path / "lock.txt"
    lock.write_text(f"pydantic=={metadata.version('pydantic')}\n")
    first = sbom.build_sbom(wheel, lock, "9.9.9")
    assert first == sbom.build_sbom(wheel, lock, "9.9.9")
    assert first["metadata"]["timestamp"] == "2023-11-14T22:13:20Z"
    names = {c["name"]: c for c in first["components"]}
    assert names["semgrep"]["licenses"] == [{"license": {"id": "LGPL-2.1-only"}}]
    props = names["semgrep"]["properties"]
    bundled = {p["value"] for p in props if p["name"] == "tantular:bundled"}
    assert bundled == {"false"} and names["ollama"]["scope"] == "optional"
    assert first["metadata"]["component"]["licenses"] == [{"license": {"id": "Apache-2.0"}}]
    json.dumps(first)


def test_bundled_data_resolves_in_a_checkout() -> None:
    from tantular_cyber.config import default_corpus, default_realworld_manifest, sample_project

    assert (default_corpus() / "manifest.json").is_file()
    assert default_realworld_manifest().is_file()
    assert (sample_project() / "app.py").is_file()


def test_wheel_force_include_paths_exist() -> None:
    build = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"]
    for source in build["targets"]["wheel"]["force-include"]:
        assert (ROOT / source).exists(), source

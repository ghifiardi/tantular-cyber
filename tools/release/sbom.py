"""Write a CycloneDX 1.5 JSON SBOM for a built wheel (release tooling, not shipped).

Usage (inside an environment where requirements/runtime.lock.txt is installed):

    python tools/release/sbom.py --wheel dist/tantular_cyber-X-py3-none-any.whl \
        --lock requirements/runtime.lock.txt --output dist/tantular-cyber-X.cdx.json

Components:
* the wheel itself (SHA-256, Apache-2.0);
* every locked runtime dependency (version, purl, license from installed metadata);
* declared **external** dependencies that are *not bundled*: Semgrep (user-installed,
  LGPL-2.1-only) and Ollama (optional, MIT). Model weights are never distributed and are
  listed only as a note.

Output is deterministic: sorted components, serial number derived from the wheel hash,
timestamp from SOURCE_DATE_EPOCH (else the Unix epoch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from license_audit import license_of, locked

EXTERNAL = [
    {
        "type": "application",
        "name": "semgrep",
        "version": ">=1.90,<2 (user-installed; tested range)",
        "licenses": [{"license": {"id": "LGPL-2.1-only"}}],
        "purl": "pkg:pypi/semgrep",
        "scope": "optional",
        "description": "Required for full scans; installed and updated by the user. Not "
        "bundled or redistributed; not covered by tantular-cyber's Apache-2.0 license.",
        "properties": [
            {"name": "tantular:bundled", "value": "false"},
            {"name": "tantular:role", "value": "external scanner (full scan mode)"},
        ],
    },
    {
        "type": "application",
        "name": "ollama",
        "version": "user-installed",
        "licenses": [{"license": {"id": "MIT"}}],
        "scope": "optional",
        "description": "Optional local LLM server for --ai explanations. Not bundled.",
        "properties": [
            {"name": "tantular:bundled", "value": "false"},
            {"name": "tantular:role", "value": "optional local LLM endpoint"},
        ],
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_sbom(wheel: Path, lock: Path, version: str) -> dict[str, object]:
    digest = sha256(wheel)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    deps = []
    for name, ver in sorted(locked(lock)):
        lic = license_of(metadata.distribution(name))
        deps.append(
            {
                "type": "library",
                "bom-ref": f"pkg:pypi/{name.lower()}@{ver}",
                "name": name,
                "version": ver,
                "purl": f"pkg:pypi/{name.lower()}@{ver}",
                "scope": "required",
                "licenses": [{"expression": lic}] if lic else [],
            }
        )
    main_ref = f"pkg:pypi/tantular-cyber@{version}"
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'sha256:' + digest)}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "component": {
                "type": "application",
                "bom-ref": main_ref,
                "name": "tantular-cyber",
                "version": version,
                "purl": main_ref,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
                "hashes": [{"alg": "SHA-256", "content": digest}],
                "properties": [{"name": "tantular:artifact", "value": wheel.name}],
            },
            "properties": [
                {
                    "name": "tantular:model-weights",
                    "value": "not distributed; any model the user runs keeps its own license",
                }
            ],
        },
        "components": [*deps, *EXTERNAL],
        "dependencies": [{"ref": main_ref, "dependsOn": [d["bom-ref"] for d in deps]}],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    version = args.wheel.name.split("-")[1]
    sbom = build_sbom(args.wheel, args.lock, version)
    args.output.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n")
    print(f"sbom: {args.output} ({len(sbom['components'])} components)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

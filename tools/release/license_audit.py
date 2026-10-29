"""Audit licenses of the locked runtime dependencies (release tooling, not shipped).

Usage (inside an environment where requirements/runtime.lock.txt is installed):

    python tools/release/license_audit.py requirements/runtime.lock.txt [--markdown out.md]

Every locked package must be installed and carry a license on the allowlist; anything
missing, unknown or outside the allowlist fails the audit (exit 1). Licenses come from the
installed distribution metadata (License-Expression, License, then classifiers).
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

ALLOWED = {
    "MIT": "permissive",
    "BSD-2-Clause": "permissive",
    "BSD-3-Clause": "permissive",
    "Apache-2.0": "permissive",
    "ISC": "permissive",
    "PSF-2.0": "permissive",
    # File-level weak copyleft: allowed for unmodified use; source is available upstream.
    "MPL-2.0": "weak copyleft (file-level; used unmodified)",
}
_CLASSIFIERS = {
    "MIT License": "MIT",
    "BSD License": None,  # ambiguous: needs an explicit expression
    "Apache Software License": "Apache-2.0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "ISC License (ISCL)": "ISC",
    "Python Software Foundation License": "PSF-2.0",
}
_LOCK_LINE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)")


def locked(lock: Path) -> list[tuple[str, str]]:
    return [(m[1], m[2]) for line in lock.read_text().splitlines() if (m := _LOCK_LINE.match(line))]


def license_of(dist: metadata.Distribution) -> str | None:
    meta = dist.metadata
    if expr := meta.get("License-Expression"):
        return expr.strip()
    raw = (meta.get("License") or "").strip()
    if raw in ALLOWED:
        return raw
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            mapped = _CLASSIFIERS.get(classifier.split("::")[-1].strip())
            if mapped:
                return mapped
    return raw or None


def audit(lock: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    problems: list[str] = []
    for name, version in locked(lock):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name}=={version}: not installed in this environment")
            continue
        if dist.version != version:
            problems.append(f"{name}: installed {dist.version}, locked {version}")
        lic = license_of(dist)
        if lic not in ALLOWED:
            problems.append(f"{name}=={version}: license {lic!r} is not on the allowlist")
        rows.append(
            {
                "name": name,
                "version": version,
                "license": lic or "UNKNOWN",
                "class": ALLOWED.get(lic or "", "NOT ALLOWED"),
            }
        )
    return rows, problems


def render(rows: list[dict[str, str]]) -> str:
    out = ["| Package | Version | License | Class |", "|---|---|---|---|"]
    out += [f"| {r['name']} | {r['version']} | {r['license']} | {r['class']} |" for r in rows]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lock", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    rows, problems = audit(args.lock)
    table = render(rows)
    if args.markdown:
        args.markdown.write_text(table)
    print(table, end="")
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

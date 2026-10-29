"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os
from pathlib import Path

BASE = "/srv/reports"


def read_report_checked(name: str) -> str:
    target = os.path.realpath(os.path.join(BASE, name))
    if os.path.commonpath([target, BASE]) != BASE:
        raise PermissionError(name)
    with open(target) as fh:
        return fh.read()


def read_report_pathlib(name: str) -> str:
    return (Path(BASE) / name).read_text()

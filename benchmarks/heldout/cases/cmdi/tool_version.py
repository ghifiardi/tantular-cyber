"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import subprocess

TOOL = "/usr/local/bin/imgproc"
FLAGS = "--version --quiet"


def tool_version() -> str:
    proc = subprocess.run(f"{TOOL} {FLAGS}", shell=True, capture_output=True, text=True)
    return proc.stdout.strip()

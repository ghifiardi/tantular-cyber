"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import subprocess

TOOL_VERSION = "2.4.1"


def print_version() -> None:
    subprocess.run(f"echo tool version {TOOL_VERSION}", shell=True, check=True)


def thumbnail(image_name: str) -> None:
    subprocess.run(["sh", "-c", "convert " + image_name + " -resize 64x64 thumb.png"], check=True)

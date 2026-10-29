"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import subprocess


def start_service(executable: str, port: int, extra_args: str) -> subprocess.Popen:
    command = f"{executable} --port {port} {extra_args}"
    return subprocess.Popen(command, shell=True)

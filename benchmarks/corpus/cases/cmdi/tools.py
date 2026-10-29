"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os
import shlex
import subprocess


def ping(host: str) -> int:
    return subprocess.call(f"ping -c 1 {host}", shell=True)


def archive_logs(dirname: str) -> None:
    # input is validated upstream
    os.system("tar czf /tmp/logs.tgz " + dirname)


def ping_args(host: str) -> int:
    return subprocess.call(["ping", "-c", "1", host])


def disk_usage() -> str:
    return subprocess.check_output("df -h /var", shell=True, text=True)


def ping_quoted(host: str) -> int:
    return subprocess.call("ping -c 1 " + shlex.quote(host), shell=True)

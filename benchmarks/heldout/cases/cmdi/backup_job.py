"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import subprocess
import sys


def backup(folder: str, dest: str = "/var/backups/site.tgz") -> int:
    result = subprocess.run(f"tar czf {dest} {folder}", shell=True, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(backup(sys.argv[1]))

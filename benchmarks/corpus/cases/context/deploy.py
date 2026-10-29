"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import logging
import subprocess

log = logging.getLogger(__name__)


def deploy_branch(repo_dir: str, branch: str) -> int:
    cmd = f"cd {repo_dir} && git checkout {branch} && ./build.sh"
    log.info("deploying")
    attempts = 0
    max_attempts = 3
    log.debug("max attempts %d", max_attempts)
    while attempts < max_attempts:
        attempts += 1
        log.debug("attempt %d", attempts)
        if attempts > 1:
            log.warning("retrying deploy")
        result = subprocess.run(cmd, shell=True, check=False)
        if result.returncode == 0:
            return attempts
    return -1

"""Opt-in Linux lab adapter for a dedicated, disposable Shirudo test VM.

Telemetry collection stays outside the read-only source scanner: this adapter runs only from
``tantular-cyber runtime lab`` and only on an explicitly identified disposable lab target.

* ``plan`` prints the exact argument lists; nothing runs.
* ``collect --execute`` runs an allowlist of read-only commands over ``ssh`` (argument lists,
  no shell on this side, timeouts), stores the outputs locally, then imports them offline.
* Never loads, unloads, signs, publishes or reconciles upatches; never touches operator keys;
  map reads use ``--peek`` by default so the lab exporter keeps its records.

Tests exercise this with an injected fake runner. A fake run validates adapter logic only,
not a real Shirudo deployment.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import upstream as up

Runner = Callable[[Sequence[str], float], tuple[int, bytes, bytes]]

_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
_LAB_NAME = re.compile(r"^(?:lab|test|vm|disposable)[-_][A-Za-z0-9._-]{1,48}$")
MAX_OUTPUT_BYTES = 64 * 1024 * 1024

# Remote commands the adapter may run (exact prefixes). Anything else is refused.
_READ_ONLY_REMOTE: tuple[tuple[str, ...], ...] = (
    ("uname", "-a"),
    ("cat", "/proc/version"),
    ("shirudo-runtime", "status", "--json"),
    ("shirudo-runtime", "upatch", "list"),
    ("shirudo-runtime", "upatch", "map", up.UPATCH),
)
_FORBIDDEN_WORDS = frozenset(
    {
        "load",
        "unload",
        "reconcile",
        "sign",
        "genkey",
        "push",
        "install",
        "init",
        "keys",
        "build",
        "config",
        "tree",
        "rm",
        "--otel",
        "--s3-uri",
        "sh",
        "bash",
        "-c",
    }
)


class LabRefused(ValueError):
    pass


def default_runner(argv: Sequence[str], timeout: float) -> tuple[int, bytes, bytes]:
    proc = subprocess.run(list(argv), capture_output=True, timeout=timeout, check=False)
    return proc.returncode, proc.stdout[:MAX_OUTPUT_BYTES], proc.stderr[:4096]


@dataclass(frozen=True)
class LabTarget:
    """A clearly identified disposable lab VM. ``name`` must start with lab-/test-/vm-."""

    name: str
    ssh_host: str
    disposable_confirmed: bool
    ssh_user: str = "root"
    ssh_options: tuple[str, ...] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")

    def validate(self) -> None:
        if not _LAB_NAME.match(self.name):
            raise LabRefused("lab target name must start with lab-, test-, vm- or disposable-")
        if not _HOST.match(self.ssh_host) or not re.match(
            r"^[a-z_][a-z0-9_-]{0,31}$", self.ssh_user
        ):
            raise LabRefused("invalid ssh host or user")
        if not self.disposable_confirmed:
            raise LabRefused(
                "refusing: confirm the target is a dedicated disposable lab VM "
                "(--i-confirm-disposable-lab-vm); never point this at production hosts"
            )


def check_remote(argv: Sequence[str]) -> None:
    if not any(tuple(argv[: len(p)]) == p for p in _READ_ONLY_REMOTE):
        raise LabRefused(f"remote command not on the read-only allowlist: {argv[:3]}")
    if _FORBIDDEN_WORDS & set(argv[1:]) - {"status", "list", "map", up.UPATCH}:
        raise LabRefused("remote command contains a forbidden word")
    if any(not re.match(r"^[A-Za-z0-9_./=:-]+$", a) for a in argv):
        raise LabRefused("remote argument contains shell metacharacters")


@dataclass
class LabStep:
    id: str
    remote: list[str]
    output: str
    timeout_s: float = 60.0


def collection_plan(maps: Sequence[str] = up.SUPPORTED_MAPS, *, peek: bool = True) -> list[LabStep]:
    steps = [
        LabStep("kernel", ["uname", "-a"], "kernel-uname.txt"),
        LabStep("proc_version", ["cat", "/proc/version"], "kernel-proc-version.txt"),
        LabStep("status", ["shirudo-runtime", "status", "--json"], "shirudo-status.json"),
        LabStep("upatches", ["shirudo-runtime", "upatch", "list"], "shirudo-upatch-list.txt"),
    ]
    for m in maps:
        if m not in up.SUPPORTED_MAPS:
            raise LabRefused(f"unsupported map: {m}")
        argv = ["shirudo-runtime", "upatch", "map", up.UPATCH, m, "--json"]
        if peek:
            argv.append("--peek")
        steps.append(LabStep(f"map_{m}", argv, f"kernel-obs-{m}.json", timeout_s=120.0))
    for s in steps:
        check_remote(s.remote)
    return steps


def ssh_argv(target: LabTarget, remote: Sequence[str]) -> list[str]:
    check_remote(remote)
    # ssh joins remote words with spaces; every word was checked for metacharacters above.
    return ["ssh", *target.ssh_options, "-l", target.ssh_user, "--", target.ssh_host, *remote]


@dataclass
class LabRun:
    target: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    export_files: list[str] = field(default_factory=list)


def collect(
    target: LabTarget,
    out_dir: Path,
    *,
    runner: Runner = default_runner,
    maps: Sequence[str] = up.SUPPORTED_MAPS,
    peek: bool = True,
) -> LabRun:
    """Run the plan on the lab VM and store outputs; caller imports the NDJSON offline."""
    target.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    run = LabRun(target=target.name)
    ndjson: list[bytes] = []
    for step in collection_plan(maps, peek=peek):
        argv = ssh_argv(target, step.remote)
        try:
            code, stdout, stderr = runner(argv, step.timeout_s)
        except (OSError, subprocess.SubprocessError) as exc:
            code, stdout, stderr = -1, b"", type(exc).__name__.encode()
        (out_dir / step.output).write_bytes(stdout[:MAX_OUTPUT_BYTES])
        run.steps.append(
            {
                "id": step.id,
                "remote": step.remote,
                "exit": code,
                "bytes": len(stdout),
                "stderr": stderr[:300].decode("utf-8", "replace"),
            }
        )
        if step.id.startswith("map_") and code == 0 and stdout.strip():
            ndjson.append(stdout.strip())
    if ndjson:
        export = out_dir / "kernel-obs-export.ndjson"
        export.write_bytes(b"\n".join(ndjson) + b"\n")
        run.export_files.append(export.name)
    (out_dir / "lab-run.json").write_text(
        json.dumps(
            {
                "target": run.target,
                "steps": run.steps,
                "export_files": run.export_files,
                "upstream_revision": up.UPSTREAM_REVISION,
                "live_validation": "unreviewed",
            },
            indent=2,
        )
        + "\n"
    )
    return run


def compatibility_manifest() -> dict[str, Any]:
    """What a lab VM must satisfy for the pinned revision (from upstream requirements docs)."""
    return {
        "manifest_type": "tantular-cyber.shirudo-lab-compatibility",
        "upstream": {"repository": up.UPSTREAM_REPOSITORY, "revision": up.UPSTREAM_REVISION},
        "target": {
            "architectures": ["x86-64", "arm64"],
            "kernel": "bpf-next with the signed-loader ABI (test suite: bpf-next 57fae4059380)",
            "required_upstream_commits": [
                "1041fca2ee8a (KEY_SPEC_DM_VERITY_KEYRING)",
                "31ac959c915c (x86-64 ftrace jmp mode)",
            ],
            "required_kernel_patches": [
                "tpm_command LSM hook",
                "BPF LSM xattr reads",
                "sleepable inode hooks",
                "trusted dentry->d_parent",
                "mm_read_access LSM hook",
            ],
            "tpm": "TPM 2.0 via /dev/tpmrm0 (test VM: software TPM); preflight tpm2_reachable",
            "preemption_model_for_kernel_obs": "CONFIG_PREEMPT_LAZY without PREEMPT_DYNAMIC/RCU",
            "checks": [
                "shirudo init --check-only (signed_loader_kernel_abi, kernel_lsm_hooks)",
                "shirudo-runtime status reports 'Status: ok'",
            ],
            "not_sufficient": "an ordinary Linux container or distribution kernel",
        },
        "test_vm_lifecycle_upstream": [
            "make -C testing vm VM_NAME=<lab-name>   # on a Linux KVM host, as documented",
            "make -C testing vm-ssh VM_NAME=<lab-name>",
            "make -C testing vm-stop VM_NAME=<lab-name>",
        ],
        "record_for_each_run": [
            "kernel build metadata (source commit + patches)",
            "uname -a",
            "/proc/version",
            "shirudo-runtime status --json",
            "shirudo-runtime upatch list (kernel-obs version)",
            "export file SHA-256 and tantular import status",
        ],
        "tantular_side": {
            "collection": "tantular-cyber runtime lab collect --execute (read-only allowlist)",
            "import": "tantular-cyber runtime import --input <export> --output <dir>",
            "never": [
                "load/unload/sign/publish/reconcile upatches",
                "expose operator keys",
                "run on production hosts",
                "use Shirudo as the plugin sandbox",
            ],
        },
        "live_validation": "pending: no compatible authorized lab VM available to this run",
    }

"""Pinned Shirudo compatibility target for the offline runtime-telemetry importer.

Everything here was written independently from reading the pinned upstream sources; no
upstream code is copied or executed. The field layout below mirrors what the kernel-obs Lua
decoders *return* (their JSON output), which is the only thing Tantular reads.

Changing the compatibility target = new pinned revision + a new entry in
``docs/runtime/compatibility.md`` + fixture updates in the same change.
"""

from __future__ import annotations

from typing import Final

UPSTREAM_PROJECT: Final = "shirudo"
UPSTREAM_REPOSITORY: Final = "https://github.com/shirudo-dev/shirudo"
UPSTREAM_REVISION: Final = "44926eb77751495be6cc39cd745692355678956f"
UPSTREAM_REVISION_DATE: Final = "2026-11-24T13:58:34+01:00"
UPSTREAM_LICENSE_NOTE: Final = (
    "Upstream repository LICENSE is BSD-2-Clause; kernel-obs decoders and BPF sources carry "
    "'GPL-2.0 OR BSD-2-Clause'. Tantular ships no upstream code."
)

# Upstream decoder output: ``{"schema": 1, "upatch", "map", "interval", "records"}``.
UPATCH: Final = "kernel-obs"
SUPPORTED_SCHEMA: Final = 1

# Maps whose records Tantular normalizes at the pinned revision.
SUPPORTED_MAPS: Final = ("exec", "net", "priv", "file")
# Maps recognized at the pinned revision but not normalized yet (reported as coverage gaps).
RECOGNIZED_UNSUPPORTED_MAPS: Final = ("socket", "listen", "watch", "dns")
# Metric-signal decoder: returns a list of OTLP Metric objects, not a log document.
METRIC_MAPS: Final = ("watch_stat",)

# Field kinds used by the validator:
#   u8/u16/u32 -> JSON integer within range; u64 -> decimal string (Lua cannot hold 64 bits);
#   str        -> bounded string (decoder cstring); bool; ident -> {"dev": u32, "ino": u64}
#   enum:u32   -> decoder name string, or the raw integer when the decoder had no name.
U32_MAX: Final = 2**32 - 1
U64_MAX: Final = 2**64 - 1

INTERVAL_FIELDS: Final = {
    "since_unix_nano": "u64",
    "used": "u32",
    "overflow": "u32",
    "lost": "u32",
    "peek": "bool",
}

_COMMON = {
    "entry_index": "u32",
    "executable": "ident",
    "cgroup_id": "u64",
    "first_seen_unix_nano": "u64?",
    "last_seen_unix_nano": "u64?",
    "pid": "u32",
}

# "?" = optional (the decoder omits the key when the value is nil).
RECORD_FIELDS: Final[dict[str, dict[str, str]]] = {
    "exec": {
        **_COMMON,
        "euid": "u32",
        "uid": "u32",
        "interpreter": "bool",
        "caller_inferred": "bool",
        "path_truncated": "bool",
        "exec_count": "u64",
        "cap_effective": "u64",
        "name": "str",
        "path": "str",
        "caller": "ident?",
    },
    "net": {
        **_COMMON,
        "family": "u8",
        "proto": "enum",
        "role": "str?",
        "address": "str?",
        "port": "u16",
        "count": "u64",
        "tx_packets": "u64",
        "tx_bytes": "u64",
        "rx_packets": "u64",
        "rx_bytes": "u64",
    },
    "priv": {
        **_COMMON,
        "kind": "enum",
        "subtype": "u32",
        "euid": "u32",
        "uid": "u32",
        "count": "u64",
        "aux": "u64",
        "name": "str",
        "detail": "str",
    },
    "file": {
        **_COMMON,
        "euid": "u32",
        "mode": "enum",
        "path": "str",
        "count": "u64",
        "rule": "u32",
        "fs_magic": "u64",
    },
}

NET_ROLES: Final = ("initiator", "acceptor", "listener")
NET_PROTOS: Final = ("icmp", "tcp", "udp", "icmpv6", "sctp", "udplite")
PRIV_KINDS: Final = (
    "setuid_root",
    "cred_anomaly",
    "ptrace_attach",
    "mount",
    "kernel_read_file",
    "kernel_load_data",
    "module_request",
    "userns_create",
    "bpf",
    "mmap_exec",
    "mprotect_exec",
)
FILE_MODES: Final = ("r", "w", "rw")
ENUMS: Final = {
    ("net", "proto"): NET_PROTOS,
    ("priv", "kind"): PRIV_KINDS,
    ("file", "mode"): FILE_MODES,
}

# Upper bound of cstring lengths in the pinned decoders (bytes), plus slack for U+FFFD
# substitution of invalid UTF-8 by the exporter.
CSTRING_LIMITS: Final = {
    ("exec", "name"): 32,
    ("exec", "path"): 128,
    ("priv", "name"): 32,
    ("priv", "detail"): 96,
    ("file", "path"): 128,
}

# What each table's hook observes at the pinned revision (from the upstream README and the
# BPF attach points). Used verbatim in reports and in the analysis prompt.
HOOK_SEMANTICS: Final = {
    "exec": (
        "lsm.s/bprm_check_security: exec_count counts executions that reached the check "
        "hook in this interval; a later failure of execve() is not recorded"
    ),
    "net": (
        "socket LSM hooks + root-cgroup skb counters: count is connects or datagrams sent; "
        "packet/byte counters are what the socket layer saw before segmentation offload"
    ),
    "priv": (
        "LSM hooks per kind; mmap_exec/mprotect_exec are requests including PROT_EXEC, not "
        "proof that the mapping succeeded or that code ran"
    ),
    "file": (
        "lsm.s/file_open for names on the compiled file-watch.list; the open reached the "
        "hook, its final outcome is not recorded; nothing is enforced"
    ),
}

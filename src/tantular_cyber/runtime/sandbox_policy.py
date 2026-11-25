"""Draft-only Shirudo ``app-sandbox`` policy proposals: validation and plain explanation.

Independently written checks that mirror the documented format of the pinned revision
(``upatches/app-sandbox/README.md``): ``[networks]`` lists, ``[sandbox.<id>]`` with
``exec``/``network``, ``[sandbox.<id>.deny]`` with ``create``/``rename``/``write``; capacity
32 sandboxes with exec rules, 128 networks, 64 patterns of one ``*`` and at most 32 bytes on
each side. Upstream ``gen_policy.py`` stays authoritative at build time.

A proposal is never signed, built, published or loaded by Tantular. It is not a complete
agent sandbox and not a substitute for patching the vulnerable application.
"""

from __future__ import annotations

import hashlib
import ipaddress
import tomllib
from dataclasses import dataclass, field
from typing import Any

from ..reports import _inline
from .models import HostPath, RuntimeEvidenceReport

SANDBOX_MAX, NET_MAX, PATTERN_MAX, GLOB_SIDE_MAX = 32, 128, 64, 32
DENY_OPS = ("create", "rename", "write")

RESIDUAL_GAPS_ID = (
    "Path yang belum ada saat load tidak diberi label; file yang menggantikan file berlabel "
    "(misalnya upgrade paket) adalah file baru tanpa label: muat ulang setelah instalasi/upgrade.",
    "Aturan file berdasarkan NAMA, bukan isi: descriptor yang sudah terbuka untuk menulis "
    "sebelum task masuk sandbox, atau yang diberikan proses lain, tetap bisa menulis.",
    "Aturan jaringan hanya mencakup connect() dan alamat sendmsg()/sendto() pada IPv4/IPv6; "
    "keluarga alamat lain lolos, begitu pula yang diterima atau di-accept.",
    "Sandbox bukan batas terhadap root di luar sandbox; task yang dijalankan root dari file "
    "tanpa label tidak berada di sandbox.",
    "Task masuk sandbox pada execve() berikutnya atas file berlabel: aplikasi perlu di-restart.",
    "Sandbox membatasi langkah lanjutan penyerang, tidak menutup celah aplikasi: tetap "
    "perbarui/patch aplikasinya.",
    "Ini bukan sandbox agen yang lengkap dan tidak memenuhi syarat isolasi plugin Tantular.",
)


@dataclass
class PolicyCheck:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sandboxes: dict[str, dict[str, Any]] = field(default_factory=dict)
    networks: dict[str, list[str]] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors


def _strings(value: Any, where: str, check: PolicyCheck) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        check.errors.append(f"{where}: expected an array of strings")
        return []
    return value


def _network(text: str, where: str, check: PolicyCheck) -> None:
    try:
        ipaddress.ip_network(text, strict=True)
    except ValueError:
        check.errors.append(f"{where}: {text!r} is not a CIDR network")


def _pattern(text: str, where: str, check: PolicyCheck) -> None:
    raw = text.encode("utf-8")
    if not raw or b"/" in raw or b"\0" in raw or raw.count(b"*") > 1:
        check.errors.append(f"{where}: {text!r} must be a file name with at most one '*'")
        return
    prefix, _, suffix = raw.partition(b"*")
    if len(prefix) > GLOB_SIDE_MAX or len(suffix) > GLOB_SIDE_MAX:
        check.errors.append(f"{where}: {text!r} has more than {GLOB_SIDE_MAX} bytes on a side")


def validate_policy(text: str, declared_policies: set[str]) -> PolicyCheck:
    """Validate a draft ``sandbox.toml``. ``declared_policies`` = Beefile policy ids."""
    check = PolicyCheck()
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        check.errors.append(f"invalid TOML: {exc}")
        return check
    unknown = set(data) - {"networks", "sandbox"}
    if unknown:
        check.errors.append(f"unknown table {sorted(unknown)[0]}")
    nets = data.get("networks", {})
    if not isinstance(nets, dict):
        check.errors.append("[networks] must be a table")
        nets = {}
    for name, items in nets.items():
        values = _strings(items, f"networks.{name}", check)
        for v in values:
            _network(v, f"networks.{name}", check)
        check.networks[name] = values
    sandboxes = data.get("sandbox", {})
    if not isinstance(sandboxes, dict):
        check.errors.append("[sandbox] must be a table")
        sandboxes = {}
    total_nets = total_patterns = with_exec = 0
    for sid, rules in sandboxes.items():
        where = f"sandbox.{sid}"
        if sid not in declared_policies:
            check.errors.append(f"{where}: the Beefile declares no policy {sid}")
        if not isinstance(rules, dict):
            check.errors.append(f"{where}: must be a table")
            continue
        extra = set(rules) - {"exec", "network", "deny"}
        if extra:
            check.errors.append(f"{where}: unknown key {sorted(extra)[0]}")
        entry: dict[str, Any] = {"exec": None, "network": None, "deny": {}}
        if "exec" in rules:
            entry["exec"] = _strings(rules["exec"], f"{where}.exec", check)
            with_exec += 1
            for pid in entry["exec"]:
                if pid not in declared_policies:
                    check.errors.append(f"{where}.exec: the Beefile has no policy {pid}")
        if "network" in rules:
            entry["network"] = _strings(rules["network"], f"{where}.network", check)
            for item in entry["network"]:
                if item.startswith("@"):
                    if item[1:] not in check.networks:
                        check.errors.append(f"{where}.network: no [networks] entry {item[1:]}")
                    else:
                        total_nets += len(check.networks[item[1:]])
                else:
                    _network(item, f"{where}.network", check)
                    total_nets += 1
        deny = rules.get("deny", {})
        if not isinstance(deny, dict):
            check.errors.append(f"{where}.deny: must be a table")
            deny = {}
        for op, pats in deny.items():
            if op not in DENY_OPS:
                check.errors.append(f"{where}.deny: unknown key {op}")
                continue
            values = _strings(pats, f"{where}.deny.{op}", check)
            for p in values:
                _pattern(p, f"{where}.deny.{op}", check)
            total_patterns += len(values)
            entry["deny"][op] = values
        check.sandboxes[sid] = entry
    if with_exec > SANDBOX_MAX:
        check.errors.append(
            f"{with_exec} sandboxes with exec rules; the object holds {SANDBOX_MAX}"
        )
    if total_nets > NET_MAX:
        check.errors.append(f"{total_nets} networks; the object holds {NET_MAX}")
    if total_patterns > PATTERN_MAX:
        check.errors.append(f"{total_patterns} file name patterns; the object holds {PATTERN_MAX}")
    for sid, entry in check.sandboxes.items():
        if entry["network"] is None:
            check.warnings.append(f"{sid}: no network key — the task connects anywhere")
        if entry["exec"] is None:
            check.warnings.append(f"{sid}: no exec key — the task executes anything")
        if not entry["deny"]:
            check.warnings.append(f"{sid}: no deny rules — no file name is protected")
    return check


def explain_policy(check: PolicyCheck) -> str:
    out = [
        "# DRAF kebijakan app-sandbox (belum ditandatangani, belum dimuat)",
        "",
        "> Usulan ini hanya draf untuk ditinjau manusia. Tantular tidak membangun, "
        "menandatangani, menerbitkan, atau memuatnya. Hasil uji upstream (t9114) tidak "
        "diklaim karena tidak dijalankan.",
        "",
        f"**Validasi:** {'lolos' if check.valid else 'GAGAL'}",
        "",
    ]
    out += [f"- error: {_inline(e)}" for e in check.errors]
    out += [f"- peringatan: {_inline(w)}" for w in check.warnings]
    for sid, e in check.sandboxes.items():
        out += ["", f"## Sandbox `{_inline(sid)}`", ""]
        if e["exec"] is None:
            out.append("- Eksekusi: tidak dibatasi.")
        else:
            allowed = (
                ", ".join(_inline(x) for x in e["exec"]) or "(hanya file kebijakannya sendiri)"
            )
            out.append(f"- Eksekusi hanya file berlabel kebijakan sendiri dan: {allowed}.")
        if e["network"] is None:
            out.append("- Jaringan: tidak dibatasi.")
        else:
            nets = ", ".join(_inline(x) for x in e["network"]) or "(tidak ada peer IP)"
            out.append(
                f"- connect()/sendmsg() IPv4/IPv6 hanya ke: {nets}; tanpa raw/packet socket."
            )
        for op in DENY_OPS:
            if e["deny"].get(op):
                out.append(
                    f"- Ditolak `{op}` untuk nama: {', '.join(_inline(p) for p in e['deny'][op])}"
                )
    out += ["", "## Celah yang tersisa", ""] + [f"- {g}" for g in RESIDUAL_GAPS_ID]
    return "\n".join(out) + "\n"


def propose_from_evidence(
    report: RuntimeEvidenceReport, *, policy_id: str, executable_path: str
) -> str:
    """Draft sandbox.toml: observed IP peers of the executable as /32 or /128 networks.

    Matching is by ``{dev, ino}`` from exec records whose exported path equals
    ``executable_path``; names alone are never used. Peers are observations, not approval."""
    wanted = hashlib.sha256(executable_path.encode("utf-8")).hexdigest()
    ids = {
        (r.executable.dev, r.executable.ino)
        for r in report.records
        if r.map == "exec"
        and isinstance(r.fields.get("path"), HostPath)
        and r.fields["path"].sha256 == wanted  # type: ignore[union-attr]
    }
    peers: set[str] = set()
    for r in report.records:
        if r.map == "net" and (r.executable.dev, r.executable.ino) in ids:
            addr = r.fields.get("address")
            if isinstance(addr, str):
                try:
                    ip = ipaddress.ip_address(addr)
                except ValueError:
                    continue
                peers.add(f"{ip}/{32 if ip.version == 4 else 128}")
    lines = [
        "# DRAFT ONLY - generated by tantular-cyber runtime policy propose; review before use.",
        "# Peers below were OBSERVED in imported telemetry; observation is not approval.",
        f"# Executable matched by {{dev, ino}} of exec records with path {executable_path!r}"
        f" ({len(ids)} identity/identities).",
        "",
        f"[sandbox.{policy_id}]",
        "exec = []",
        "network = [" + ", ".join(f'"{p}"' for p in sorted(peers)) + "]",
        "",
        f"[sandbox.{policy_id}.deny]",
        '# Fill in names the application must never create/rename/write, e.g. "*.php".',
        "create = []",
        "rename = []",
        "write = []",
    ]
    return "\n".join(lines) + "\n"

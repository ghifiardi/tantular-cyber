"""Writers for ``runtime-evidence.json`` / ``runtime-evidence.md`` (atomic, escaped)."""

from __future__ import annotations

from pathlib import Path

from ..reports import _atomic_write, _inline
from .models import HostPath, RuntimeEvidenceReport, RuntimeRecord

EVIDENCE_JSON = "runtime-evidence.json"
EVIDENCE_MD = "runtime-evidence.md"
MD_MAX_RECORDS = 200
MAX_OUTPUT_BYTES = 64 * 1024 * 1024

_BADGE = {"complete": "✅ complete", "degraded": "⚠️ DEGRADED", "failed": "❌ FAILED"}


def render_evidence_json(report: RuntimeEvidenceReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def _value(value: object) -> str:
    if isinstance(value, HostPath):
        shown = value.display if value.display is not None else f"sha256:{value.sha256[:16]}"
        cut = " (terpotong?)" if value.truncated_by_decoder or value.at_capacity else ""
        return f"{shown}{cut}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return "" if value is None else str(value)


def _record_line(r: RuntimeRecord) -> str:
    fields = "; ".join(f"{k}={_value(v)}" for k, v in r.fields.items() if v is not None)
    exe = f"{r.executable.dev}:{r.executable.ino}"
    inferred = " caller_inferred" if r.caller_inferred else ""
    return (
        f"| `{_inline(r.ref)}` | {r.map} | {exe} | {r.pid} | {r.count} | "
        f"{_inline(fields)}{inferred} |"
    )


def render_evidence_markdown(report: RuntimeEvidenceReport) -> str:
    r = report
    out = [
        "# Tantular Cyber — Bukti Runtime (impor telemetri Shirudo)",
        "",
        f"**Status:** {_BADGE.get(r.status, r.status)}  ",
        f"**Import ID:** `{r.import_id}`  ",
        f"**Waktu impor:** {r.imported_at}  ",
        f"**Input:** `{_inline(r.input.name)}` ({r.input.form}, {r.input.bytes} byte, "
        f"sha256 `{r.input.sha256}`)  ",
        f"**Target upstream:** {r.upstream.project} `{r.upstream.revision}` "
        f"(schema {r.upstream.supported_schema})  ",
        f"**Generator:** {r.generator.name} {r.generator.version} "
        f"(runtime schema {r.runtime_schema_version})",
        "",
        "> **Status verifikasi:** belum diverifikasi; tidak diautentikasi TPM. Bukti ini "
        "adalah salinan ekspor yang diberikan pengguna.",
        "",
    ]
    if r.status_reasons or r.errors:
        out.append("> **Catatan status:**")
        out += [f"> - {_inline(x)}" for x in [*r.errors, *r.status_reasons]]
        out.append("")
    out += ["## Cara membaca bukti ini", ""]
    out += [f"- {_inline(n)}" for n in r.interpretation_notes]
    out += ["", "## Cakupan per tabel", ""]
    out += [
        "| Tabel | Dokumen | Dihitung | Duplikat | Peek digantikan | Konflik | Catatan | "
        "Ditolak | Overflow | Lost | Metadata hilang | Peek saja | Total hitungan |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in sorted(r.coverage.maps.items()):
        out.append(
            f"| {name} | {c.documents} | {c.counted} | {c.duplicate} | {c.superseded_peek} | "
            f"{c.conflict} | {c.records} | {c.records_rejected} | {c.overflow_total} | "
            f"{c.lost_total} | {c.intervals_missing_metadata} | {c.peek_only_intervals} | "
            f"{c.event_count_total} |"
        )
    out.append("")
    out.append(f"- Kehilangan data diketahui lengkap: {'ya' if r.coverage.loss_known else 'TIDAK'}")
    if r.coverage.maps_absent:
        out.append(
            "- Tabel didukung yang tidak ada di ekspor (tidak diamati, bukan berarti nihil): "
            + ", ".join(r.coverage.maps_absent)
        )
    for u in r.coverage.unsupported:
        out.append(
            f"- Tidak dianalisis ({u.reason}): {_inline(u.upatch)}/{_inline(u.map)} "
            f"schema={u.schema_version}, {u.documents} dokumen, {u.records} catatan"
        )
    for d in r.coverage.rejected_documents[:50]:
        out.append(f"- Dokumen ditolak #{d.index} (baris {d.line}): {_inline(d.reason)}")
    out += [
        "",
        "## Interval",
        "",
        "| Dokumen | Tabel | since_unix_nano | Peek | Used | Overflow | Lost | Disposisi |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in r.intervals:
        out.append(
            f"| `{i.document_ref}` | {_inline(i.map)} | {i.since_unix_nano or '—'} | "
            f"{i.peek} | {i.used} | {i.overflow} | {i.lost} | {i.disposition} |"
        )
    out += ["", "## Catatan (agregasi per interval)", ""]
    out += [f"- **{m}**: {_inline(s)}" for m, s in sorted(r.hook_semantics.items())]
    out += [
        "",
        "| Ref | Tabel | Executable (dev:ino) | PID | Hitungan | Field |",
        "|---|---|---|---|---|---|",
    ]
    out += [_record_line(rec) for rec in r.records[:MD_MAX_RECORDS]]
    if len(r.records) > MD_MAX_RECORDS:
        out.append(f"\n_{len(r.records) - MD_MAX_RECORDS} catatan lain ada di JSON._")
    out += ["", "## Privasi", ""]
    out += [f"- path_mode: `{r.privacy.path_mode}`"]
    out += [f"- {_inline(n)}" for n in r.privacy.notes]
    return "\n".join(out) + "\n"


def write_evidence(report: RuntimeEvidenceReport, output: Path) -> dict[str, Path]:
    text = render_evidence_json(report)
    if len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ValueError("runtime evidence report exceeds the 64 MiB output bound")
    paths = {"json": output / EVIDENCE_JSON, "markdown": output / EVIDENCE_MD}
    _atomic_write(paths["json"], text)
    _atomic_write(paths["markdown"], render_evidence_markdown(report))
    return paths

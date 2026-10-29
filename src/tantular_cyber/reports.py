"""Report writers: versioned ``report.json`` and human-readable ``report.md``."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .models import ScanReport
from .security import scrub_paths

JSON_NAME = "report.json"
MARKDOWN_NAME = "report.md"
RAW_NAME_TEMPLATE = "{tool}-raw.json"
MD_MAX_SKIPPED = 50

_STATUS_BADGE = {"complete": "✅ complete", "degraded": "⚠️ DEGRADED", "failed": "❌ FAILED"}


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def render_json(report: ScanReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def _inline(text: str | None) -> str:
    """Neutralize untrusted text for inline Markdown (no HTML, no line breaks)."""
    if not text:
        return ""
    text = " ".join(text.split())
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", text)


def _code_block(content: str, lang: str = "") -> str:
    """Fence longer than any backtick run inside ``content`` so it cannot break out."""
    longest = max((len(m) for m in re.findall(r"`+", content)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{content}\n{fence}"


def _bullets(items: list[str]) -> list[str]:
    return [f"- {_inline(i)}" for i in items] or ["- _(tidak ada)_"]


def render_markdown(report: ScanReport) -> str:
    r = report
    out: list[str] = [
        "# Tantular Cyber — Laporan Analisis Keamanan Repositori",
        "",
        f"**Status:** {_STATUS_BADGE.get(r.status, r.status)}  ",
        f"**Repositori:** `{_inline(r.target.name)}`  ",
        f"**Scan ID:** `{r.scan_id}`  ",
        f"**Waktu:** {r.started_at} → {r.finished_at} ({r.duration_s:.1f} s)  ",
        f"**Generator:** {r.generator.name} {r.generator.version} (schema {r.schema_version})",
        "",
    ]
    if r.status_reasons:
        out += ["> **Catatan status:**"]
        out += [f"> - {_inline(reason)}" for reason in r.status_reasons]
        out.append("")

    out += [
        "## Ringkasan",
        "",
        "| Metrik | Nilai |",
        "|---|---|",
        f"| Total temuan | {r.totals.findings} |",
        f"| File dipindai / dilewati | {r.target.files_scanned} / {r.target.files_skipped} |",
    ]
    for sev, count in sorted(r.totals.by_severity.items()):
        out.append(f"| Severity `{sev}` | {count} |")
    for cwe, count in sorted(r.totals.by_cwe.items()):
        out.append(f"| {cwe} | {count} |")
    for status, count in sorted(r.totals.analyses_by_status.items()):
        out.append(f"| Analisis LLM `{status}` | {count} |")
    out.append("")

    out += ["## Alat", "", "| Komponen | Versi | Detail |", "|---|---|---|"]
    for s in r.scanners:
        detail = (
            f"exit={s.exit_code}, targets={s.targets}, rules={len(s.rules)}, "
            f"{'ok' if s.succeeded else 'GAGAL'}, {s.duration_s:.1f} s"
        )
        out.append(f"| {s.name} | {s.version or 'n/a'} | {detail} |")
    llm = r.llm
    if llm.enabled:
        llm_detail = (
            f"model `{_inline(llm.model)}`, endpoint `{_inline(llm.endpoint)}`, "
            f"available={llm.available}"
        )
    else:
        llm_detail = "dinonaktifkan (--no-llm)"
    out += [f"| {llm.provider or 'llm'} | {llm.server_version or 'n/a'} | {llm_detail} |", ""]
    if perf := llm.performance:
        rate = f"{perf.mean_output_tokens_per_s:.1f}" if perf.mean_output_tokens_per_s else "n/a"
        load = f"{perf.first_call_load_s:.1f} s" if perf.first_call_load_s is not None else "n/a"
        out += [
            f"**Kinerja LLM:** {perf.calls} panggilan, total {perf.total_wall_s:.1f} s, "
            f"rata-rata {perf.mean_wall_s:.1f} s/temuan (min {perf.min_wall_s:.1f}, "
            f"maks {perf.max_wall_s:.1f}), load model awal {load}, "
            f"{perf.prompt_tokens} token prompt / {perf.output_tokens} token output, "
            f"{rate} token/s.",
            "",
        ]

    out += ["## Cakupan pemindaian", ""]
    for s in r.scanners:
        c = s.coverage
        if c is None:
            out.append(f"- {s.name}: cakupan tidak dapat diverifikasi (bahasa rule tidak dikenal).")
            continue
        out.append(
            f"- {s.name} ({', '.join(c.languages)}): {c.scanned}/{c.in_scope} file in-scope "
            f"dipindai; {c.out_of_scope} file di luar cakupan bahasa rule; "
            f"terverifikasi dari output tool: {'ya' if c.verified else 'TIDAK'}."
        )
        out += [
            f"  - ⚠️ TIDAK dipindai: `{_inline(e.path)}` ({_inline(e.reason)})"
            for e in c.not_scanned
        ]
    if r.target.ignore_files_present:
        files = ", ".join(f"`{_inline(p)}`" for p in r.target.ignore_files_present)
        out.append(
            f"- File ignore milik repositori ditemukan: {files}. File ini tidak dipakai untuk "
            "memilih target; target dikirim eksplisit dan cakupan diverifikasi."
        )
    out.append("")

    out += ["## Temuan", ""]
    if not r.findings:
        out += [
            "Tidak ada temuan dari ruleset yang dijalankan. Ini **bukan** jaminan bahwa "
            "repositori bebas kerentanan (lihat Batasan).",
            "",
        ]
    for i, f in enumerate(r.findings, 1):
        cwe = ", ".join(f.cwe) or "unmapped"
        out += [
            f"### {i}. [{f.severity.upper()}] {cwe} — `{_inline(f.path)}:{f.start_line}`",
            "",
            f"- **Finding ID:** `{f.id}`",
            f"- **Status:** `{f.status}` (belum diverifikasi manusia)",
            f"- **Tool / rule:** {f.tool} / `{_inline(f.rule_id)}`",
            f"- **Lokasi:** `{_inline(f.path)}` baris {f.start_line}–{f.end_line}",
            f"- **Confidence rule:** {f.confidence or 'n/a'} (sumber: {f.confidence_source})",
            f"- **Pesan scanner:** {_inline(f.message)}",
            "",
            "**Bukti (scanner, otoritatif):**",
            "",
            _code_block(f.evidence.snippet),
            "",
        ]
        if f.evidence.context:
            out += [
                "<details><summary>Konteks sekitar (bukan bagian dari match)</summary>",
                "",
                _code_block(f.evidence.context),
                "",
                *(
                    [
                        f"_Strategi konteks: `{f.evidence.context_kind}`"
                        + (
                            f" (fallback: {_inline(f.evidence.context_fallback)})"
                            if f.evidence.context_fallback
                            else ""
                        )
                        + (", dipangkas sesuai anggaran" if f.evidence.context_truncated else "")
                        + "._",
                        "",
                    ]
                ),
                *(
                    [
                        "Petunjuk aliran data (sintaksis, belum diverifikasi):",
                        *_bullets(f.evidence.dataflow_hints),
                        "",
                    ]
                    if f.evidence.dataflow_hints
                    else []
                ),
                "</details>",
                "",
            ]
        if f.evidence.truncated:
            out += ["_Bukti dipotong; lihat file sumber untuk rentang lengkap._", ""]
        if f.evidence.redactions:
            out += [f"_{f.evidence.redactions} nilai yang mirip kredensial disamarkan._", ""]

        a = f.analysis
        out.append(f"**Penjelasan LLM (advisory, `{a.status}`):**")
        out.append("")
        if a.explanation:
            e = a.explanation
            out += [
                f"> ⚠️ {_inline(a.disclaimer)}",
                "",
                f"**Ringkasan:** {_inline(e.summary)}",
                "",
                "**Bukti terkonfirmasi (menurut model, dari potongan kode):**",
                *_bullets(e.confirmed_evidence),
                "",
                "**Dampak kondisional (belum terbukti):**",
                *_bullets(e.conditional_impact),
                "",
                "**Remediasi:**",
                *_bullets(e.remediation),
                "",
            ]
            if e.corrected_code:
                out += [
                    "**Contoh kode perbaikan (saran, tidak diterapkan):**",
                    "",
                    _code_block(e.corrected_code),
                    "",
                ]
            out += [
                f"**Keyakinan model:** {e.confidence}",
                "",
                "**Keterbatasan:**",
                *_bullets(e.limitations),
                "",
            ]
            if a.validation_warnings:
                out += ["**Peringatan validasi:**", *_bullets(a.validation_warnings), ""]
        else:
            out += [f"_Tidak ada penjelasan LLM: {_inline(a.error) or a.status}._", ""]

    if r.errors:
        out += ["## Error", ""]
        out += [f"- `{e.stage}`: {_inline(e.message)}" for e in r.errors]
        out.append("")

    out += ["## File yang dilewati", ""]
    if r.skipped.by_reason:
        out += [f"- {reason}: {count}" for reason, count in r.skipped.by_reason.items()]
        out += ["", "<details><summary>Daftar file yang dilewati</summary>", ""]
        out += [f"- `{_inline(e.path)}` — {e.reason}" for e in r.skipped.entries[:MD_MAX_SKIPPED]]
        if r.skipped.total > MD_MAX_SKIPPED:
            out.append(f"- … {r.skipped.total - MD_MAX_SKIPPED} lainnya (lihat report.json)")
        out += ["", "</details>"]
    else:
        out.append("- _(tidak ada)_")
    out += ["", "## Batasan", ""]
    out += [f"- {_inline(item)}" for item in r.limitations]
    out.append("")
    return "\n".join(out)


def write_reports(
    report: ScanReport,
    output: Path,
    *,
    repo: Path | None = None,
    raw_outputs: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Path]:
    """Write reports into ``output``. Raw scanner JSON is only written when provided."""
    output.mkdir(parents=True, exist_ok=True)
    written = {"json": output / JSON_NAME, "markdown": output / MARKDOWN_NAME}
    _atomic_write(written["json"], render_json(report))
    _atomic_write(written["markdown"], render_markdown(report))
    for tool, payloads in (raw_outputs or {}).items():
        text = json.dumps(payloads, indent=2, ensure_ascii=False)
        if repo is not None:
            text = scrub_paths(text, repo)
        path = output / RAW_NAME_TEMPLATE.format(tool=re.sub(r"[^a-z0-9_-]", "_", tool.lower()))
        _atomic_write(path, text + "\n")
        written[f"raw:{tool}"] = path
    return written

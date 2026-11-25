"""Writers for ``runtime-analysis.json`` / ``runtime-analysis.md`` (atomic, escaped)."""

from __future__ import annotations

from pathlib import Path

from ..reports import _atomic_write, _inline
from .analysis import RuntimeAnalysisReport

ANALYSIS_JSON = "runtime-analysis.json"
ANALYSIS_MD = "runtime-analysis.md"


def render_analysis_markdown(r: RuntimeAnalysisReport) -> str:
    out = [
        "# Tantular Cyber — Tafsiran runtime (advisory)",
        "",
        f"> {_inline(r.disclaimer)}",
        "",
        f"**Status analisis:** {r.status}  ",
        f"**Status bukti:** {r.evidence_status} (belum diverifikasi)  ",
        f"**Bukti:** sha256 `{r.evidence_sha256}`  ",
        f"**Model:** {_inline(r.provider or '-')} / {_inline(r.model or '-')}  ",
        f"**Prompt:** `{r.prompt_sha256[:16]}`  ",
        f"**Retrieval:** {'aktif' if r.retrieval.enabled else 'nonaktif'}"
        + (f", corpus `{(r.retrieval.corpus_sha256 or '-')[:16]}`" if r.retrieval.enabled else ""),
        "",
    ]
    out += [f"- {_inline(x)}" for x in r.status_reasons]
    e = r.explanation
    if e is None:
        out += ["", f"_Tidak ada tafsiran: {_inline(r.error or 'model tidak tersedia')}_"]
        return "\n".join(out) + "\n"

    def section(title: str, items: list[str]) -> None:
        out.extend(["", f"## {title}", ""])
        out.extend([f"- {_inline(i)}" for i in items] or ["- _(tidak ada)_"])

    out += ["", "## Ringkasan", "", _inline(e.summary)]
    out.append(f"\n**Keyakinan:** {e.confidence}; **abstain:** {e.abstain}")
    section(
        "Observasi (dengan rujukan bukti)",
        [f"{o.text} [{', '.join(o.evidence_refs)}]" for o in e.observations],
    )
    section(
        "Tafsiran bersyarat",
        [
            f"{c.text} — syarat: {'; '.join(c.conditions) or '-'} [{', '.join(c.evidence_refs)}]"
            for c in e.conditional_interpretations
        ],
    )
    section("Penjelasan alternatif yang sah", e.benign_alternatives)
    section("Keterbatasan", e.limitations)
    section("Investigasi yang disarankan", e.recommended_investigation)
    section("Saran mitigasi (untuk ditinjau manusia)", e.mitigation_suggestions)
    section("Sitasi referensi (mekanisme, bukan bukti host)", e.citations)
    if r.rejected_citations or r.rejected_evidence_refs:
        section("Rujukan ditolak", [*r.rejected_citations, *r.rejected_evidence_refs])
    if r.validation_warnings:
        section("Peringatan validasi", r.validation_warnings)
    return "\n".join(out) + "\n"


def write_analysis(report: RuntimeAnalysisReport, output: Path) -> dict[str, Path]:
    paths = {"json": output / ANALYSIS_JSON, "markdown": output / ANALYSIS_MD}
    _atomic_write(paths["json"], report.model_dump_json(indent=2) + "\n")
    _atomic_write(paths["markdown"], render_analysis_markdown(report))
    return paths

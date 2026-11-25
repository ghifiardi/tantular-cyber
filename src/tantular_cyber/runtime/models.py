"""Versioned runtime-evidence contract (``runtime-evidence.json``).

Separate from the source-scan ``ScanReport``: runtime telemetry is host evidence, not a
repository finding, and carries no CWE. Bump ``RUNTIME_SCHEMA_VERSION`` (semver) on any
change and regenerate ``docs/runtime/runtime-evidence-schema.json``
(``tantular-cyber runtime schema evidence``); a test enforces it.

Every value comes from the imported export. Nothing here is verified by Tantular: the
evidence is ``verified: false`` and is not TPM-authenticated.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RUNTIME_SCHEMA_VERSION = "1.0.0"
RUNTIME_REPORT_TYPE = "tantular-cyber.runtime-evidence"

RuntimeStatus = Literal["complete", "degraded", "failed"]
PathMode = Literal["masked_user", "full", "basename", "hash_only"]
Disposition = Literal["counted", "duplicate", "superseded_peek", "conflict"]

INTERPRETATION_NOTES_ID: tuple[str, ...] = (
    "Data runtime berupa agregasi per interval (delta per kunci tabel), bukan urutan "
    "peristiwa atau jejak proses yang berurutan.",
    "Hook mencatat percobaan/permintaan pada titik LSM; hasil akhir operasi (berhasil atau "
    "gagal) umumnya tidak tercatat.",
    "Byte dan paket pada tabel net adalah hitungan di lapisan socket/cgroup, bukan bukti "
    "data terkirim ke pihak tertentu; tidak ada payload yang direkam.",
    "Ketiadaan catatan bukan bukti host bersih: cakupan bergantung pada tabel yang diekspor, "
    "watch list yang dikompilasi, overflow/lost, dan metadata yang tersedia.",
    "Nama executable atau path tidak membuktikan identitas, silsilah proses, atau lokasi "
    "kode sumber; identitas executable adalah {dev, ino}.",
    "Telemetri ini tidak diautentikasi TPM; atestasi TPM hanya mencakup state terukur di "
    "bawah kebijakan verifier, dan mitigasi yang dimuat tidak membuktikan kerentanan "
    "sudah diperbaiki.",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Generator(_Strict):
    name: str = "tantular-cyber"
    version: str


class UpstreamTarget(_Strict):
    project: str
    repository: str
    revision: str
    supported_schema: int
    supported_maps: list[str]
    recognized_unsupported_maps: list[str]


class InputDescriptor(_Strict):
    name: str = Field(description="Sanitized base name of the input file (no directory).")
    sha256: str
    bytes: int
    form: Literal["json", "ndjson", "unknown"]
    lines: int
    documents: int


class Verification(_Strict):
    verified: Literal[False] = False
    status: Literal["unverified"] = "unverified"
    tpm_authenticated: Literal[False] = False
    note: str = (
        "Diimpor dari berkas ekspor yang diberikan pengguna; Tantular tidak memverifikasi "
        "asal, integritas, atau kelengkapannya."
    )


class Privacy(_Strict):
    path_mode: PathMode
    raw_telemetry_persisted: Literal[False] = False
    host_paths_are_absolute: bool = True
    notes: list[str]


class ExecutableId(_Strict):
    dev: int
    ino: str = Field(description="u64 inode as a decimal string (never converted to float).")


class HostPath(_Strict):
    """A host path from telemetry. Absolute operational evidence, not a repository path.

    Tantular never opens, stats or otherwise reads a host path from a record."""

    display: str | None = Field(description="Rendered according to path_mode; redacted.")
    sha256: str = Field(
        description="SHA-256 of the exported string (unsalted; guessable for common paths)."
    )
    truncated_by_decoder: bool | None = Field(
        default=None, description="Decoder flag where available (exec.path_truncated)."
    )
    at_capacity: bool = Field(
        description="Length reached the decoder's fixed buffer; the path may be cut."
    )
    redactions: int = 0


class IntervalSummary(_Strict):
    document_ref: str
    upatch: str
    map: str
    schema_version: int
    since_unix_nano: str | None
    used: int | None
    overflow: int | None
    lost: int | None
    peek: bool | None
    metadata_missing: list[str]
    records: int
    records_rejected: int
    content_sha256: str
    identity_key: str | None = Field(
        description="upatch/map/schema/since_unix_nano/peek; null when since is unknown."
    )
    disposition: Disposition
    disposition_reason: str | None = None


class RuntimeRecord(_Strict):
    ref: str = Field(description="Stable evidence reference: <document_ref>/<map>/<entry>.")
    document_ref: str
    map: Literal["exec", "net", "priv", "file"]
    entry_index: int
    executable: ExecutableId
    cgroup_id: str
    pid: int
    first_seen_unix_nano: str | None
    last_seen_unix_nano: str | None
    count: str = Field(description="Per-interval delta: exec_count or count, decimal string.")
    hook: str
    outcome: Literal["not_recorded"] = "not_recorded"
    caller_inferred: bool | None = Field(
        default=None, description="null = not exported by this map's decoder at the pin."
    )
    fields: dict[str, int | str | bool | HostPath | ExecutableId | None] = Field(
        description="Map-specific decoder fields, validated and sanitized."
    )


class UnsupportedEntry(_Strict):
    upatch: str
    map: str
    schema_version: int | None
    reason: Literal[
        "unsupported_upatch",
        "unsupported_schema",
        "recognized_unsupported_map",
        "metric_document",
        "unknown_map",
    ]
    documents: int
    records: int


class RejectedDocument(_Strict):
    index: int
    line: int | None
    reason: str


class MapCoverage(_Strict):
    documents: int = 0
    counted: int = 0
    duplicate: int = 0
    superseded_peek: int = 0
    conflict: int = 0
    records: int = 0
    records_rejected: int = 0
    overflow_total: int = 0
    lost_total: int = 0
    intervals_missing_metadata: int = 0
    peek_only_intervals: int = 0
    event_count_total: str = "0"
    path_truncated_records: int = 0
    path_at_capacity_records: int = 0
    caller_inferred_records: int = 0
    unknown_fields: list[str] = Field(default_factory=list)


class Coverage(_Strict):
    maps: dict[str, MapCoverage]
    maps_absent: list[str]
    unsupported: list[UnsupportedEntry]
    rejected_documents: list[RejectedDocument]
    loss_known: bool = Field(
        description="False when any counted interval lacks overflow/lost metadata: "
        "missing metadata means unknown coverage, not zero loss."
    )
    dedup_notes: list[str]
    report_records_omitted: int = 0


class RuntimeEvidenceReport(_Strict):
    runtime_schema_version: str = RUNTIME_SCHEMA_VERSION
    report_type: str = RUNTIME_REPORT_TYPE
    generator: Generator
    import_id: str
    imported_at: str
    upstream: UpstreamTarget
    input: InputDescriptor
    status: RuntimeStatus
    status_reasons: list[str]
    errors: list[str]
    verification: Verification = Field(default_factory=Verification)
    privacy: Privacy
    coverage: Coverage
    intervals: list[IntervalSummary]
    records: list[RuntimeRecord]
    hook_semantics: dict[str, str]
    interpretation_notes: list[str] = Field(default_factory=lambda: list(INTERPRETATION_NOTES_ID))

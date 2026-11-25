"""Offline import of Shirudo kernel-obs exports (decoder JSON / NDJSON) into runtime evidence.

Read-only and self-contained: no Shirudo installation, root, VM, network, Semgrep, model,
BPF, map draining, Lua or plugin execution. The importer parses the bytes of one regular
file the user supplied; it never opens a host path that appears inside a record.

Accepted input forms (pinned revision, see ``upstream.py``):

* ``json``   one decoder document, e.g. ``shirudo upatch map kernel-obs exec --json``
* ``ndjson`` one decoder document per line, e.g. ``--json=<path>`` or ``map_export.json_path``

OTLP envelopes (``--otel``) and S3 snapshot references are recognized and rejected with a
reason; export with ``--json`` instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .. import __version__
from ..security import sanitize_text
from . import upstream as up
from .models import (
    Coverage,
    ExecutableId,
    Generator,
    HostPath,
    InputDescriptor,
    IntervalSummary,
    MapCoverage,
    PathMode,
    Privacy,
    RejectedDocument,
    RuntimeEvidenceReport,
    RuntimeRecord,
    UnsupportedEntry,
    UpstreamTarget,
)

MAX_INPUT_BYTES = 64 * 1024 * 1024  # upstream S3 snapshots are at most 64 MiB uncompressed
MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_DOCUMENTS = 10_000
MAX_RECORDS = 200_000
MAX_DEPTH = 8  # decoder output nests 4 levels; upstream itself caps results at 32
MAX_REPORT_RECORDS = 20_000
MAX_UNKNOWN_FIELDS = 20
MAX_REJECTED_LISTED = 200
MAX_ADDRESS_CHARS = 39


class RuntimeInputError(ValueError):
    """The input path or file cannot be imported at all (CLI exit 2, no report written)."""


class _DocReject(Exception):
    pass


class _RecordReject(Exception):
    pass


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


# --- Input -----------------------------------------------------------------------------


def read_input(path: Path) -> bytes:
    """Read one regular, non-symlink file, bounded, without blocking on FIFOs/devices."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RuntimeInputError("input file cannot be accessed") from exc
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeInputError("input must be a regular file, not a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeInputError("input must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeInputError("input file cannot be opened") from exc
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RuntimeInputError("input must be a regular file")
        data = handle.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise RuntimeInputError("input exceeds 64 MiB; split the export into smaller files")
    return data


def _no_constants(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant {token}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate object key")
        out[key] = value
    return out


def _depth(value: Any, limit: int, level: int = 1) -> bool:
    if level > limit:
        return False
    if isinstance(value, dict):
        return all(_depth(v, limit, level + 1) for v in value.values())
    if isinstance(value, list):
        return all(_depth(v, limit, level + 1) for v in value)
    return True


def _loads(raw: bytes) -> Any:
    text = raw.decode("utf-8")  # strict; upstream emits valid UTF-8 (U+FFFD substitution)
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_no_constants)


@dataclass
class _Parsed:
    form: str
    lines: int
    values: list[tuple[int | None, Any]] = field(default_factory=list)  # (line, value)
    rejected: list[RejectedDocument] = field(default_factory=list)


def _parse(data: bytes) -> _Parsed:
    lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    stripped = data.strip()
    if not stripped:
        return _Parsed("unknown", lines)
    try:
        value = _loads(stripped)
    except RecursionError:
        parsed = _Parsed("json", lines)
        parsed.rejected.append(RejectedDocument(index=0, line=1, reason="nesting_too_deep"))
        return parsed
    except (UnicodeDecodeError, ValueError):
        pass
    else:
        return _Parsed("json", lines, [(1, value)])

    parsed = _Parsed("ndjson", lines)
    index = 0
    for number, line in enumerate(data.split(b"\n"), 1):
        if not line.strip():
            continue
        if index >= MAX_DOCUMENTS:
            parsed.rejected.append(
                RejectedDocument(index=index, line=number, reason="document_limit_exceeded")
            )
            break
        if len(line) > MAX_LINE_BYTES:
            parsed.rejected.append(
                RejectedDocument(index=index, line=number, reason="line_too_long")
            )
        else:
            try:
                parsed.values.append((number, _loads(line)))
            except RecursionError:
                parsed.rejected.append(
                    RejectedDocument(index=index, line=number, reason="nesting_too_deep")
                )
            except UnicodeDecodeError:
                parsed.rejected.append(
                    RejectedDocument(index=index, line=number, reason="invalid_utf8")
                )
            except ValueError as exc:
                reason = "duplicate_key" if "duplicate" in str(exc) else "invalid_json"
                parsed.rejected.append(RejectedDocument(index=index, line=number, reason=reason))
        index += 1
    return parsed


# --- Field validation -------------------------------------------------------------------

_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]{0,19})$")


def _int(value: Any, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise _RecordReject("integer_out_of_range")
    return value


def _u64(value: Any) -> str:
    if not isinstance(value, str) or not _DECIMAL.match(value) or int(value) > up.U64_MAX:
        raise _RecordReject("u64_not_decimal_string")
    return value


def _ident(value: Any) -> ExecutableId:
    if not isinstance(value, dict) or set(value) != {"dev", "ino"}:
        raise _RecordReject("invalid_identity")
    return ExecutableId(dev=_int(value["dev"], up.U32_MAX), ino=_u64(value["ino"]))


_USER_DIR = re.compile(r"^/(home|Users)/[^/]+")


def host_path(value: str, mode: PathMode, limit: int, truncated: bool | None) -> HostPath:
    cleaned = sanitize_text(value, max_len=limit + 16)
    redactions = 0 if cleaned == value else 1
    if mode == "hash_only":
        display: str | None = None
    elif mode == "basename":
        display = PurePosixPath(cleaned).name or cleaned
    elif mode == "masked_user":
        display = _USER_DIR.sub(r"/\1/<user>", cleaned)
    else:
        display = cleaned
    return HostPath(
        display=display,
        sha256=hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest(),
        truncated_by_decoder=truncated,
        at_capacity=len(value.encode("utf-8", "surrogatepass")) >= limit - 1,
        redactions=redactions,
    )


def _string(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        raise _RecordReject("string_expected")
    # Exported strings are at most `limit` bytes; U+FFFD replaces each bad byte.
    if len(value) > limit:
        raise _RecordReject("string_too_long")
    return value


def _check(map_name: str, key: str, kind: str, value: Any) -> Any:
    base = kind.rstrip("?")
    if base == "u8":
        return _int(value, 0xFF)
    if base == "u16":
        return _int(value, 0xFFFF)
    if base == "u32":
        return _int(value, up.U32_MAX)
    if base == "u64":
        return _u64(value)
    if base == "bool":
        if type(value) is not bool:
            raise _RecordReject("bool_expected")
        return value
    if base == "ident":
        return _ident(value)
    if base == "enum":
        names = up.ENUMS[(map_name, key)]
        if isinstance(value, str):
            if value not in names:
                raise _RecordReject("unknown_enum_name")
            return value
        return _int(value, up.U32_MAX)  # decoder falls back to the raw number
    if base == "str":
        if key == "role":
            if value not in up.NET_ROLES:
                raise _RecordReject("unknown_enum_name")
            return value
        if key == "address":
            return _string(value, MAX_ADDRESS_CHARS)
        return _string(value, up.CSTRING_LIMITS.get((map_name, key), 128))
    raise AssertionError(kind)


@dataclass
class _Ctx:
    path_mode: PathMode
    unknown_fields: set[str] = field(default_factory=set)


def _record(
    map_name: str, raw: Any, doc_ref: str, ctx: _Ctx
) -> tuple[RuntimeRecord, dict[str, bool]]:
    if not isinstance(raw, dict):
        raise _RecordReject("record_not_object")
    spec = up.RECORD_FIELDS[map_name]
    values: dict[str, Any] = {}
    for key, kind in spec.items():
        if key not in raw or raw[key] is None:
            if kind.endswith("?"):
                values[key] = None
                continue
            raise _RecordReject(f"missing_field:{key}")
        values[key] = _check(map_name, key, kind, raw[key])
    for key in raw:
        if key not in spec and len(ctx.unknown_fields) < MAX_UNKNOWN_FIELDS:
            ctx.unknown_fields.add(sanitize_text(str(key), 64))

    flags = {"path_truncated": False, "path_at_capacity": False, "caller_inferred": False}
    fields: dict[str, Any] = {}
    common = {"entry_index", "executable", "cgroup_id", "pid", "first_seen_unix_nano"}
    common |= {"last_seen_unix_nano", "count", "exec_count", "caller_inferred"}
    for key, value in values.items():
        if key in common:
            continue
        if key == "path":
            truncated = values.get("path_truncated") if map_name == "exec" else None
            hp = host_path(value, ctx.path_mode, up.CSTRING_LIMITS[(map_name, "path")], truncated)
            flags["path_truncated"] = bool(truncated)
            flags["path_at_capacity"] = hp.at_capacity
            fields[key] = hp
        elif map_name == "priv" and key == "detail":
            # kernel_read_file / module names may carry paths; same privacy treatment.
            fields[key] = host_path(value, ctx.path_mode, 96, None) if value else None
        elif isinstance(value, str) and key in ("name",):
            fields[key] = sanitize_text(value, 64)
        else:
            fields[key] = value
    caller_inferred = values.get("caller_inferred") if map_name == "exec" else None
    flags["caller_inferred"] = bool(caller_inferred)
    if map_name == "exec":
        fields["caller_known"] = values["caller"] is not None
    count = values["exec_count"] if map_name == "exec" else values["count"]
    record = RuntimeRecord(
        ref=f"{doc_ref}/{map_name}/{values['entry_index']}",
        document_ref=doc_ref,
        map=map_name,  # type: ignore[arg-type]
        entry_index=values["entry_index"],
        executable=values["executable"],
        cgroup_id=values["cgroup_id"],
        pid=values["pid"],
        first_seen_unix_nano=values["first_seen_unix_nano"],
        last_seen_unix_nano=values["last_seen_unix_nano"],
        count=count,
        hook=up.HOOK_SEMANTICS[map_name],
        caller_inferred=caller_inferred,
        fields=fields,
    )
    return record, flags


# --- Documents --------------------------------------------------------------------------


@dataclass
class _Doc:
    index: int
    line: int | None
    upatch: str
    map: str
    schema: int
    content_sha256: str
    interval: dict[str, Any]
    missing: list[str]
    records: list[RuntimeRecord]
    flags: list[dict[str, bool]]
    rejected: int
    disposition: str = "counted"
    reason: str | None = None

    @property
    def ref(self) -> str:
        return f"D{self.index:04d}-{self.content_sha256[:8]}"

    @property
    def since(self) -> str | None:
        return self.interval.get("since_unix_nano")

    @property
    def peek(self) -> bool | None:
        return self.interval.get("peek")

    def identity(self) -> str | None:
        if self.since is None or self.peek is None:
            return None
        mode = "peek" if self.peek else "drain"
        return f"{self.upatch}/{self.map}/{self.schema}/{self.since}/{mode}"


def _classify(value: Any) -> tuple[str, str, int | None, str | None]:
    """Return (upatch, map, schema, unsupported_reason) or raise _DocReject."""
    if isinstance(value, list):
        if value and all(isinstance(v, dict) and "name" in v for v in value):
            return "?", "(metric)", None, "metric_document"
        raise _DocReject("not_an_object")
    if not isinstance(value, dict):
        raise _DocReject("not_an_object")
    if "resourceLogs" in value or "resourceMetrics" in value:
        raise _DocReject("otlp_envelope_unsupported_export_with_--json")
    schema, upatch, map_name = value.get("schema"), value.get("upatch"), value.get("map")
    if type(schema) is not int or not isinstance(upatch, str) or not isinstance(map_name, str):
        raise _DocReject("missing_envelope_fields")
    if not isinstance(value.get("records"), list):
        raise _DocReject("records_not_array")
    upatch, map_name = sanitize_text(upatch, 64), sanitize_text(map_name, 64)
    if upatch != up.UPATCH:
        return upatch, map_name, schema, "unsupported_upatch"
    if schema != up.SUPPORTED_SCHEMA:
        return upatch, map_name, schema, "unsupported_schema"
    if map_name in up.RECOGNIZED_UNSUPPORTED_MAPS:
        return upatch, map_name, schema, "recognized_unsupported_map"
    if map_name in up.METRIC_MAPS:
        return upatch, map_name, schema, "metric_document"
    if map_name not in up.SUPPORTED_MAPS:
        return upatch, map_name, schema, "unknown_map"
    return upatch, map_name, schema, None


def _interval(value: Any) -> tuple[dict[str, Any], list[str]]:
    if value is None:
        return {}, ["interval"]
    if not isinstance(value, dict):
        raise _DocReject("interval_not_object")
    out: dict[str, Any] = {}
    missing: list[str] = []
    for key, kind in up.INTERVAL_FIELDS.items():
        if key not in value:
            missing.append(key)
            continue
        try:
            out[key] = _check("interval", key, kind, value[key])
        except _RecordReject as exc:
            raise _DocReject(f"interval_{key}_{exc}") from exc
    return out, missing


@dataclass
class ImportResult:
    report: RuntimeEvidenceReport


def import_export(
    data: bytes,
    *,
    input_name: str = "export",
    path_mode: PathMode = "masked_user",
    now: datetime | None = None,
    import_id: str | None = None,
) -> RuntimeEvidenceReport:
    """Pure function over the export bytes (unit-testable without files)."""
    parsed = _parse(data)
    rejected = list(parsed.rejected)
    unsupported: dict[tuple[str, str, int | None, str], list[int]] = {}
    docs: list[_Doc] = []
    ctx = _Ctx(path_mode=path_mode)
    total_records = 0
    record_limit_hit = False
    doc_unknown_keys: set[str] = set()

    for index, (line, value) in enumerate(parsed.values):
        try:
            if not _depth(value, MAX_DEPTH):
                raise _DocReject("nesting_too_deep")
            upatch, map_name, schema, reason = _classify(value)
            if reason is not None:
                n = len(value.get("records", [])) if isinstance(value, dict) else len(value)
                unsupported.setdefault((upatch, map_name, schema, reason), []).append(n)
                continue
            if schema is None:  # metric documents were handled above
                raise _DocReject("missing_envelope_fields")
            interval, missing = _interval(value.get("interval"))
            for key in value:
                if key not in ("schema", "upatch", "map", "interval", "records"):
                    doc_unknown_keys.add(sanitize_text(str(key), 64))
        except _DocReject as exc:
            rejected.append(RejectedDocument(index=index, line=line, reason=str(exc)))
            continue

        doc = _Doc(
            index=index,
            line=line,
            upatch=upatch,
            map=map_name,
            schema=schema,
            content_sha256=digest(value),
            interval=interval,
            missing=missing,
            records=[],
            flags=[],
            rejected=0,
        )
        seen_entries: set[int] = set()
        for raw in value["records"]:
            if total_records >= MAX_RECORDS:
                doc.rejected += 1
                record_limit_hit = True
                continue
            try:
                record, flags = _record(map_name, raw, doc.ref, ctx)
                if record.entry_index in seen_entries:
                    raise _RecordReject("duplicate_entry_index")
            except _RecordReject:
                doc.rejected += 1
                continue
            seen_entries.add(record.entry_index)
            doc.records.append(record)
            doc.flags.append(flags)
            total_records += 1
        docs.append(doc)

    dedup_notes = _deduplicate(docs)
    return _assemble(
        parsed=parsed,
        data=data,
        input_name=input_name,
        docs=docs,
        rejected=rejected,
        unsupported=unsupported,
        ctx=ctx,
        doc_unknown_keys=doc_unknown_keys,
        record_limit_hit=record_limit_hit,
        dedup_notes=dedup_notes,
        now=now,
        import_id=import_id,
    )


def _deduplicate(docs: list[_Doc]) -> list[str]:
    """Mark repeated documents and peek snapshots so totals do not double count.

    Identifiers available in the decoder output: upatch, map, schema, the interval start
    (``since_unix_nano``), the ``peek`` flag and the document content. The export time
    (``timeUnixNano``) is not part of the document, and records carry no global id.
    """
    notes = [
        "Exact repeats (same canonical document SHA-256) are counted once.",
        "A drained interval supersedes peek snapshots of the same upatch/map/schema/"
        "since_unix_nano; without a drain the peek with the most records is counted and "
        "marked peek-only (partial, may reappear in a later drain).",
        "Two different drains claiming the same interval start are a conflict: the first is "
        "counted, the others are excluded and the import is degraded.",
        "Documents without since_unix_nano can only be de-duplicated by exact content; "
        "repeated imports in separate runs are not de-duplicated (compare content_sha256).",
    ]
    seen: dict[str, _Doc] = {}
    for doc in docs:
        if doc.content_sha256 in seen:
            doc.disposition, doc.reason = (
                "duplicate",
                f"same content as {seen[doc.content_sha256].ref}",
            )
        else:
            seen[doc.content_sha256] = doc
    groups: dict[tuple[str, str, int, str], list[_Doc]] = {}
    for doc in docs:
        if doc.disposition == "counted" and doc.since is not None and doc.peek is not None:
            groups.setdefault((doc.upatch, doc.map, doc.schema, doc.since), []).append(doc)
    for group in groups.values():
        drains = [d for d in group if not d.peek]
        peeks = [d for d in group if d.peek]
        for extra in drains[1:]:
            extra.disposition = "conflict"
            extra.reason = f"second drain of the interval counted in {drains[0].ref}"
        if drains:
            for p in peeks:
                p.disposition, p.reason = "superseded_peek", f"drained in {drains[0].ref}"
        elif peeks:
            keep = max(peeks, key=lambda d: (d.interval.get("used") or 0, len(d.records)))
            for p in peeks:
                if p is not keep:
                    p.disposition, p.reason = "superseded_peek", f"larger peek {keep.ref}"
    return notes


def _assemble(
    *,
    parsed: _Parsed,
    data: bytes,
    input_name: str,
    docs: list[_Doc],
    rejected: list[RejectedDocument],
    unsupported: dict[tuple[str, str, int | None, str], list[int]],
    ctx: _Ctx,
    doc_unknown_keys: set[str],
    record_limit_hit: bool,
    dedup_notes: list[str],
    now: datetime | None,
    import_id: str | None,
) -> RuntimeEvidenceReport:
    maps: dict[str, MapCoverage] = {}
    intervals: list[IntervalSummary] = []
    records: list[RuntimeRecord] = []
    loss_known = True
    for doc in docs:
        cov = maps.setdefault(doc.map, MapCoverage())
        cov.documents += 1
        setattr(cov, doc.disposition, getattr(cov, doc.disposition) + 1)
        intervals.append(
            IntervalSummary(
                document_ref=doc.ref,
                upatch=doc.upatch,
                map=doc.map,
                schema_version=doc.schema,
                since_unix_nano=doc.since,
                used=doc.interval.get("used"),
                overflow=doc.interval.get("overflow"),
                lost=doc.interval.get("lost"),
                peek=doc.peek,
                metadata_missing=doc.missing,
                records=len(doc.records),
                records_rejected=doc.rejected,
                content_sha256=doc.content_sha256,
                identity_key=doc.identity(),
                disposition=doc.disposition,  # type: ignore[arg-type]
                disposition_reason=doc.reason,
            )
        )
        if doc.disposition != "counted":
            continue
        cov.records += len(doc.records)
        cov.records_rejected += doc.rejected
        cov.overflow_total += doc.interval.get("overflow") or 0
        cov.lost_total += doc.interval.get("lost") or 0
        if doc.missing:
            cov.intervals_missing_metadata += 1
            if {"overflow", "lost", "interval"} & set(doc.missing):
                loss_known = False
        if doc.peek:
            cov.peek_only_intervals += 1
        cov.event_count_total = str(
            int(cov.event_count_total) + sum(int(r.count) for r in doc.records)
        )
        for f in doc.flags:
            cov.path_truncated_records += f["path_truncated"]
            cov.path_at_capacity_records += f["path_at_capacity"]
            cov.caller_inferred_records += f["caller_inferred"]
        records.extend(doc.records)
    for cov in maps.values():
        cov.unknown_fields = sorted(ctx.unknown_fields | doc_unknown_keys)
    omitted = max(0, len(records) - MAX_REPORT_RECORDS)
    records = records[:MAX_REPORT_RECORDS]

    unsupported_entries = [
        UnsupportedEntry(
            upatch=k[0],
            map=k[1],
            schema_version=k[2],
            reason=k[3],  # type: ignore[arg-type]
            documents=len(v),
            records=sum(v),
        )
        for k, v in sorted(unsupported.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3]))
    ]
    counted_maps = sorted({d.map for d in docs if d.disposition == "counted"})
    status, reasons, errors = _status(
        parsed, docs, maps, rejected, unsupported_entries, loss_known, record_limit_hit, omitted
    )
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds").replace("+00:00", "Z")
    return RuntimeEvidenceReport(
        generator=Generator(version=__version__),
        import_id=import_id or uuid.uuid4().hex,
        imported_at=stamp,
        upstream=UpstreamTarget(
            project=up.UPSTREAM_PROJECT,
            repository=up.UPSTREAM_REPOSITORY,
            revision=up.UPSTREAM_REVISION,
            supported_schema=up.SUPPORTED_SCHEMA,
            supported_maps=list(up.SUPPORTED_MAPS),
            recognized_unsupported_maps=list(up.RECOGNIZED_UNSUPPORTED_MAPS),
        ),
        input=InputDescriptor(
            name=sanitize_text(PurePosixPath(input_name).name, 128) or "export",
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
            form=parsed.form,  # type: ignore[arg-type]
            lines=parsed.lines,
            documents=len(parsed.values) + len(parsed.rejected),
        ),
        status=status,
        status_reasons=reasons,
        errors=errors,
        privacy=Privacy(
            path_mode=ctx.path_mode,
            notes=[
                "Host paths are absolute operational evidence; they are rendered per "
                "path_mode, secrets are redacted, and the importer never reads them.",
                "sha256 values of paths are unsalted and can be matched against guessed "
                "paths; treat reports as sensitive host data.",
                "Raw telemetry is not persisted; only the input SHA-256 and size are kept.",
                "Peer addresses, uids, pids and cgroup ids are kept as exported.",
            ],
        ),
        coverage=Coverage(
            maps=maps,
            maps_absent=[m for m in up.SUPPORTED_MAPS if m not in counted_maps],
            unsupported=unsupported_entries,
            rejected_documents=rejected[:MAX_REJECTED_LISTED],
            loss_known=loss_known and bool(counted_maps),
            dedup_notes=dedup_notes,
            report_records_omitted=omitted,
        ),
        intervals=intervals,
        records=records,
        hook_semantics=dict(up.HOOK_SEMANTICS),
    )


def _status(
    parsed: _Parsed,
    docs: list[_Doc],
    maps: dict[str, MapCoverage],
    rejected: list[RejectedDocument],
    unsupported: list[UnsupportedEntry],
    loss_known: bool,
    record_limit_hit: bool,
    omitted: int,
) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    errors: list[str] = []
    counted = [d for d in docs if d.disposition == "counted"]
    if parsed.form == "unknown":
        errors.append("input is empty")
    if rejected:
        reasons.append(f"rejected_documents: {len(rejected)} document(s) could not be processed")
    for entry in unsupported:
        reasons.append(
            f"{entry.reason}: {entry.upatch}/{entry.map} schema={entry.schema_version} "
            f"({entry.documents} document(s), {entry.records} record(s) not analyzed)"
        )
    for name, cov in sorted(maps.items()):
        if cov.records_rejected:
            reasons.append(f"rejected_records: {name}: {cov.records_rejected} record(s)")
        if cov.overflow_total:
            reasons.append(f"interval_overflow: {name}: {cov.overflow_total} refused allocation(s)")
        if cov.lost_total:
            reasons.append(f"interval_lost: {name}: {cov.lost_total} unfinished export(s) lost")
        if cov.intervals_missing_metadata:
            reasons.append(
                f"missing_metadata: {name}: {cov.intervals_missing_metadata} interval(s); "
                "coverage unknown (not zero loss)"
            )
        if cov.peek_only_intervals:
            reasons.append(
                f"peek_only: {name}: {cov.peek_only_intervals} interval(s) not drained; "
                "records may be partial"
            )
        if cov.conflict:
            reasons.append(f"identity_conflict: {name}: {cov.conflict} document(s) excluded")
        if cov.unknown_fields:
            reasons.append(f"schema_drift: unknown fields {', '.join(cov.unknown_fields[:5])}")
    if record_limit_hit:
        reasons.append(f"record_limit: more than {MAX_RECORDS} records; the rest were rejected")
    if omitted:
        reasons.append(f"report_records_omitted: {omitted} record(s) not listed in the report")
    if counted and not any(d.records for d in counted):
        reasons.append(
            "no_records: no observations in the counted intervals; this is not evidence "
            "that the host is clean"
        )
    reasons = list(dict.fromkeys(reasons))
    if not counted:
        errors.append("no supported kernel-obs document could be imported")
        return "failed", reasons, errors
    if reasons or not loss_known:
        return "degraded", reasons, errors
    return "complete", reasons, errors


def import_file(path: Path, *, path_mode: PathMode = "masked_user") -> RuntimeEvidenceReport:
    return import_export(read_input(path), input_name=path.name, path_mode=path_mode)

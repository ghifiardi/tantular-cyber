"""Offline Shirudo kernel-obs import: parsing, bounds, coverage, dedup, privacy, reports."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tantular_cyber.runtime import importer
from tantular_cyber.runtime.evaluation import evaluate_import, load_suite
from tantular_cyber.runtime.importer import RuntimeInputError, import_export, read_input
from tantular_cyber.runtime.models import RuntimeEvidenceReport
from tantular_cyber.runtime.reports import (
    render_evidence_json,
    render_evidence_markdown,
    write_evidence,
)

ROOT = Path(__file__).parents[1]
SCHEMA_FILE = ROOT / "docs" / "runtime" / "runtime-evidence-schema.json"
BIG = "18446744073709551615"  # u64 max: must survive without float conversion


def interval(**kw: Any) -> dict[str, Any]:
    base = {
        "since_unix_nano": "1764000000000000000",
        "used": 2,
        "overflow": 0,
        "lost": 0,
        "peek": False,
    }
    base.update(kw)
    return base


def exec_record(**kw: Any) -> dict[str, Any]:
    r = {
        "entry_index": 1,
        "executable": {"dev": 2049, "ino": "131"},
        "cgroup_id": "4242",
        "euid": 0,
        "uid": 1000,
        "pid": 77,
        "interpreter": False,
        "caller_inferred": False,
        "path_truncated": False,
        "exec_count": "3",
        "first_seen_unix_nano": "1764000000100000001",
        "last_seen_unix_nano": BIG,
        "cap_effective": "0",
        "name": "curl",
        "path": "/home/alice/bin/curl",
        "caller": {"dev": 2049, "ino": "99"},
    }
    r.update(kw)
    return r


def doc(map_name: str = "exec", records: list[Any] | None = None, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "schema": 1,
        "upatch": "kernel-obs",
        "map": map_name,
        "interval": interval(),
        "records": [exec_record()] if records is None else records,
    }
    d.update(kw)
    return d


def ndjson(*docs: Any) -> bytes:
    return ("\n".join(d if isinstance(d, str) else json.dumps(d) for d in docs) + "\n").encode()


def test_single_json_document_complete_and_exact_nanoseconds() -> None:
    report = import_export(json.dumps(doc(), indent=2).encode())
    assert report.input.form == "json"
    assert report.status == "complete" and report.status_reasons == []
    rec = report.records[0]
    assert rec.last_seen_unix_nano == BIG  # decimal string, exact
    assert rec.first_seen_unix_nano == "1764000000100000001"
    assert rec.executable.ino == "131" and rec.fields["caller"].ino == "99"
    assert rec.fields["caller_known"] is True
    assert rec.outcome == "not_recorded"
    assert report.verification.verified is False
    assert report.verification.tpm_authenticated is False
    assert report.coverage.maps["exec"].event_count_total == "3"


def test_ndjson_all_four_supported_maps() -> None:
    net = {
        "entry_index": 1,
        "executable": {"dev": 1, "ino": "2"},
        "cgroup_id": "3",
        "family": 10,
        "proto": "tcp",
        "role": "initiator",
        "address": "2001:0db8:0000:0000:0000:0000:0000:0001",
        "port": 443,
        "count": "1",
        "tx_packets": "1",
        "tx_bytes": "60",
        "rx_packets": "0",
        "rx_bytes": "0",
        "pid": 9,
    }
    priv = {
        "entry_index": 1,
        "executable": {"dev": 1, "ino": "2"},
        "cgroup_id": "3",
        "kind": "mprotect_exec",
        "subtype": 5,
        "euid": 0,
        "uid": 0,
        "count": "1",
        "pid": 9,
        "aux": "4096",
        "name": "anonymous",
        "detail": "",
    }
    file = {
        "entry_index": 1,
        "executable": {"dev": 1, "ino": "2"},
        "cgroup_id": "3",
        "euid": 0,
        "mode": "w",
        "path": "/etc/crontab",
        "count": "2",
        "pid": 9,
        "rule": 3,
        "fs_magic": "61267",
    }
    report = import_export(
        ndjson(doc(), doc("net", [net]), doc("priv", [priv]), doc("file", [file]))
    )
    assert report.input.form == "ndjson"
    assert report.status == "complete", report.status_reasons
    assert set(report.coverage.maps) == {"exec", "net", "priv", "file"}
    assert report.coverage.maps_absent == []
    kinds = {r.map: r for r in report.records}
    assert kinds["priv"].fields["kind"] == "mprotect_exec"
    assert kinds["priv"].first_seen_unix_nano is None  # decoder omits zero timestamps
    assert kinds["net"].caller_inferred is None  # not exported by the net decoder
    assert "requests" in kinds["priv"].hook and "not proof" in kinds["priv"].hook


def test_raw_enum_numbers_kept_when_decoder_has_no_name() -> None:
    priv = {
        "entry_index": 1,
        "executable": {"dev": 1, "ino": "2"},
        "cgroup_id": "3",
        "kind": 99,
        "subtype": 0,
        "euid": 0,
        "uid": 0,
        "count": "1",
        "pid": 9,
        "aux": "0",
        "name": "",
        "detail": "",
    }
    report = import_export(ndjson(doc("priv", [priv])))
    assert report.records[0].fields["kind"] == 99


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        ({"exec_count": 3}, "u64 as number"),
        ({"exec_count": "03"}, "leading zero"),
        ({"exec_count": "18446744073709551616"}, "u64 overflow"),
        ({"pid": -1}, "negative"),
        ({"pid": True}, "bool as int"),
        ({"executable": {"dev": 1}}, "identity"),
        ({"name": "x" * 33}, "string bound"),
        ({"interpreter": "no"}, "bool"),
    ],
)
def test_invalid_records_are_counted_not_dropped(mutation: dict[str, Any], field: str) -> None:
    report = import_export(
        ndjson(doc(records=[exec_record(), exec_record(entry_index=2, **mutation)]))
    )
    cov = report.coverage.maps["exec"]
    assert cov.records == 1 and cov.records_rejected == 1, field
    assert report.status == "degraded"
    assert any(r.startswith("rejected_records") for r in report.status_reasons)


def test_missing_required_field_and_duplicate_entry_index_rejected() -> None:
    missing = exec_record(entry_index=2)
    del missing["uid"]
    report = import_export(ndjson(doc(records=[exec_record(), missing, exec_record()])))
    assert report.coverage.maps["exec"].records_rejected == 2


def test_unknown_record_fields_flag_schema_drift() -> None:
    report = import_export(ndjson(doc(records=[exec_record(new_field=1)])))
    assert report.coverage.maps["exec"].unknown_fields == ["new_field"]
    assert report.status == "degraded"


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("{bad", "invalid_json"),
        ('{"a": 1, "a": 2}', "duplicate_key"),
        (
            '{"schema": 1, "upatch": "kernel-obs", "map": "exec", "records": [], "x": NaN}',
            "invalid_json",
        ),
        ("[1, 2]", "not_an_object"),
        ('{"resourceLogs": []}', "otlp_envelope_unsupported_export_with_--json"),
        (
            '{"schema": "1", "upatch": "kernel-obs", "map": "exec", "records": []}',
            "missing_envelope_fields",
        ),
        (
            '{"schema": 1, "upatch": "kernel-obs", "map": "exec", "records": {}}',
            "records_not_array",
        ),
        (
            '{"schema": 1, "upatch": "kernel-obs", "map": "exec", "interval": [], "records": []}',
            "interval_not_object",
        ),
    ],
)
def test_malformed_envelopes_rejected_with_reason(line: str, reason: str) -> None:
    report = import_export(ndjson(doc(), line))
    assert [r.reason for r in report.coverage.rejected_documents] == [reason]
    assert report.status == "degraded"


def test_nesting_bound() -> None:
    deep: Any = "x"
    for _ in range(12):
        deep = {"k": deep}
    report = import_export(ndjson(doc(), doc(records=[deep])))
    assert report.coverage.rejected_documents[0].reason == "nesting_too_deep"
    very_deep = "[" * 100_000 + "]" * 100_000
    report = import_export(ndjson(doc(), very_deep))
    assert report.coverage.rejected_documents[0].reason in {"nesting_too_deep", "invalid_json"}


def test_line_and_record_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "MAX_LINE_BYTES", 600)
    big = doc(records=[exec_record(entry_index=i) for i in range(1, 6)])
    report = import_export(ndjson(doc(), big))
    assert report.coverage.rejected_documents[0].reason == "line_too_long"
    monkeypatch.setattr(importer, "MAX_LINE_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(importer, "MAX_RECORDS", 3)
    report = import_export(ndjson(big))
    assert report.coverage.maps["exec"].records == 3
    assert any(r.startswith("record_limit") for r in report.status_reasons)
    monkeypatch.setattr(importer, "MAX_REPORT_RECORDS", 2)
    monkeypatch.setattr(importer, "MAX_RECORDS", 100)
    report = import_export(ndjson(big))
    assert len(report.records) == 2 and report.coverage.report_records_omitted == 3


def test_unsupported_maps_versions_upatches_reported_not_dropped() -> None:
    report = import_export(
        ndjson(
            doc(),
            doc("socket", [{"entry_index": 1}, {"entry_index": 2}]),
            doc(schema=2),
            doc(upatch="app-sandbox"),
            doc("mystery"),
            [{"name": "watch_stat.checked", "sum": {}}],
        )
    )
    reasons = {(u.map, u.reason): u.records for u in report.coverage.unsupported}
    assert reasons[("socket", "recognized_unsupported_map")] == 2
    assert ("exec", "unsupported_schema") in reasons
    assert ("exec", "unsupported_upatch") in reasons
    assert ("mystery", "unknown_map") in reasons
    assert ("(metric)", "metric_document") in reasons
    assert report.status == "degraded"
    assert report.coverage.maps["exec"].counted == 1


def test_only_unsupported_documents_fail() -> None:
    report = import_export(ndjson(doc("dns", [])))
    assert report.status == "failed"
    assert "no supported kernel-obs document could be imported" in report.errors


def test_empty_and_garbage_inputs_fail() -> None:
    assert import_export(b"").status == "failed"
    assert import_export(b"\n\n").status == "failed"
    assert import_export(b"\xff\xfe garbage").status == "failed"


def test_loss_overflow_and_missing_metadata_semantics() -> None:
    lossy = doc(interval=interval(overflow=4, lost=1))
    no_interval = doc()
    del no_interval["interval"]
    partial = doc(interval={"since_unix_nano": "1764000000000000009", "used": 1, "peek": False})
    report = import_export(ndjson(lossy, no_interval, partial))
    cov = report.coverage.maps["exec"]
    assert cov.overflow_total == 4 and cov.lost_total == 1
    assert cov.intervals_missing_metadata == 2
    assert report.coverage.loss_known is False  # unknown coverage, not zero loss
    codes = {r.split(":")[0] for r in report.status_reasons}
    assert {"interval_overflow", "interval_lost", "missing_metadata"} <= codes


def test_no_records_never_means_clean() -> None:
    report = import_export(ndjson(doc(records=[])))
    assert report.status == "degraded"
    assert any("not evidence that the host is clean" in r for r in report.status_reasons)


def test_duplicates_and_peeks_are_not_double_counted() -> None:
    drained = doc()
    peek = doc(interval=interval(peek=True))
    report = import_export(ndjson(peek, drained, deepcopy(drained)))
    cov = report.coverage.maps["exec"]
    assert (cov.counted, cov.duplicate, cov.superseded_peek) == (1, 1, 1)
    assert cov.records == 1 and cov.event_count_total == "3"
    assert report.status == "complete"


def test_peek_only_interval_keeps_largest_and_degrades() -> None:
    small = doc(interval=interval(peek=True, used=2))
    large = doc(
        records=[exec_record(), exec_record(entry_index=2)], interval=interval(peek=True, used=3)
    )
    report = import_export(ndjson(small, large))
    cov = report.coverage.maps["exec"]
    assert cov.counted == 1 and cov.superseded_peek == 1 and cov.records == 2
    assert any(r.startswith("peek_only") for r in report.status_reasons)


def test_conflicting_drains_of_same_interval_excluded_and_degrade() -> None:
    other = doc(records=[exec_record(exec_count="9")])
    report = import_export(ndjson(doc(), other))
    assert report.coverage.maps["exec"].conflict == 1
    assert report.coverage.maps["exec"].event_count_total == "3"
    assert any(r.startswith("identity_conflict") for r in report.status_reasons)


def test_documents_without_since_only_dedup_by_content() -> None:
    a = doc(interval={"used": 1, "overflow": 0, "lost": 0, "peek": False})
    b = doc(records=[exec_record(exec_count="5")], interval=a["interval"])
    report = import_export(ndjson(a, b, deepcopy(a)))
    cov = report.coverage.maps["exec"]
    assert cov.counted == 2 and cov.duplicate == 1
    assert all(i.identity_key is None for i in report.intervals)


def test_flags_preserved_truncated_and_inferred() -> None:
    rec = exec_record(path="/opt/" + "a" * 122, path_truncated=True, caller_inferred=True)
    del rec["caller"]
    report = import_export(ndjson(doc(records=[rec])))
    r = report.records[0]
    assert r.caller_inferred is True and r.fields["caller"] is None
    assert r.fields["caller_known"] is False
    assert r.fields["path"].truncated_by_decoder is True and r.fields["path"].at_capacity
    cov = report.coverage.maps["exec"]
    assert cov.caller_inferred_records == 1 and cov.path_truncated_records == 1


@pytest.mark.parametrize(
    ("mode", "shown"),
    [
        ("masked_user", "/home/<user>/bin/curl"),
        ("full", "/home/alice/bin/curl"),
        ("basename", "curl"),
        ("hash_only", None),
    ],
)
def test_path_privacy_modes(mode: str, shown: str | None) -> None:
    report = import_export(ndjson(doc()), path_mode=mode)  # type: ignore[arg-type]
    path = report.records[0].fields["path"]
    assert path.display == shown
    assert len(path.sha256) == 64
    if mode != "full":
        assert "alice" not in render_evidence_json(report)


def test_secrets_redacted_in_records_and_reports() -> None:
    secret = "tc-synthetic-rt-secret-0000000042"
    rec = exec_record(path=f"/srv/x/password={secret}", name="ghp_" + "a" * 26)
    report = import_export(ndjson(doc(records=[rec])))
    text = render_evidence_json(report) + render_evidence_markdown(report)
    assert secret not in text and "ghp_aaaa" not in text
    assert report.records[0].fields["path"].redactions == 1


def test_control_and_bidi_characters_stripped() -> None:
    rec = exec_record(name="ev" + chr(0x202E) + "il" + chr(0x1B) + "[31m")
    report = import_export(ndjson(doc(records=[rec])))
    assert report.records[0].fields["name"] == "evil[31m"


def test_markdown_escapes_untrusted_strings() -> None:
    rec = exec_record(name="<script>|`x`")
    report = import_export(ndjson(doc(records=[rec])))
    md = render_evidence_markdown(report)
    assert "<script>" not in md and "&lt;script&gt;" in md


def test_raw_telemetry_not_persisted(tmp_path: Path) -> None:
    data = ndjson(doc(records=[exec_record(name="uniquemarker")]))
    report = import_export(data, input_name="/var/tmp/secret-dir/export.ndjson")
    paths = write_evidence(report, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "runtime-evidence.json",
        "runtime-evidence.md",
    ]
    text = paths["json"].read_text()
    assert report.input.name == "export.ndjson" and "secret-dir" not in text
    assert report.privacy.raw_telemetry_persisted is False
    RuntimeEvidenceReport.model_validate_json(text)  # round-trips through the contract


def test_read_input_refuses_unsafe_paths(tmp_path: Path) -> None:
    good = tmp_path / "e.ndjson"
    good.write_bytes(ndjson(doc()))
    assert read_input(good)
    link = tmp_path / "link"
    link.symlink_to(good)
    with pytest.raises(RuntimeInputError, match="symbolic link"):
        read_input(link)
    with pytest.raises(RuntimeInputError, match="regular file"):
        read_input(tmp_path)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeInputError, match="regular file"):
        read_input(fifo)  # must not block
    with pytest.raises(RuntimeInputError):
        read_input(tmp_path / "missing")


def test_read_input_size_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "MAX_INPUT_BYTES", 10)
    big = tmp_path / "big"
    big.write_bytes(b"x" * 11)
    with pytest.raises(RuntimeInputError, match="exceeds"):
        read_input(big)


def test_importer_never_reads_paths_inside_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "host-file"
    target.write_text("do not read")
    opened: list[str] = []
    real_open = os.open

    def spy(path: Any, *a: Any, **k: Any) -> int:
        opened.append(str(path))
        return real_open(path, *a, **k)

    export = tmp_path / "export.ndjson"
    export.write_bytes(ndjson(doc(records=[exec_record(path=str(target))])))
    monkeypatch.setattr(os, "open", spy)
    importer.import_file(export)
    assert opened == [str(export)]


def test_synthetic_suite_import_expectations_gate() -> None:
    """Deterministic import correctness on the synthetic suite (gates CI)."""
    root, manifest = load_suite()
    assert manifest["synthetic"] is True
    failures = []
    for case in manifest["cases"]:
        report = import_export((root / case["path"]).read_bytes())
        result = evaluate_import(case, report)
        if not result.passed:
            failures.append((case["id"], [c for c in result.checks if not c.passed]))
    assert not failures, failures


def test_suite_case_files_carry_no_labels() -> None:
    root, manifest = load_suite()
    for case in manifest["cases"]:
        text = (root / case["path"]).read_text()
        for word in ("expect", "must_abstain", "risk", "suspicious", "benign"):
            assert word not in text, (case["id"], word)


def test_committed_runtime_evidence_schema_matches_model() -> None:
    """Regenerate with `tantular-cyber runtime schema evidence > docs/runtime/...`."""
    assert json.loads(SCHEMA_FILE.read_text()) == RuntimeEvidenceReport.model_json_schema()

"""UTC timestamps in WSP protocols and ledgers (tools/wsp/wsp_eval/timestamps.py): ASCII digits,
a real calendar date and time, convertible to UTC, and only the ``Z`` form, so every accepted
value is usable by ``analysis`` and ``drift``. Non-ASCII digits are built with chr() so this file
stays ASCII."""

# ruff: noqa: F811 - the fixtures imported from test_wsp_review are used as parameters
from __future__ import annotations

import copy
import json
from typing import ClassVar

import pytest
from test_wsp_review import FROZEN, ROOT, raw  # noqa: F401 - fixture used by name
from wsp_eval import analysis, drift
from wsp_eval.ledger import validate_record
from wsp_eval.protocol import ProtocolError, parse_protocol
from wsp_eval.timestamps import is_utc_timestamp, parse_utc_timestamp


def _digits(text: str, zero: int) -> str:
    """``text`` with its ASCII digits replaced by another script's digits starting at ``zero``."""
    return "".join(chr(zero + int(c)) if c.isdigit() else c for c in text)


ARABIC_INDIC = _digits("2026-11-01T00:00Z", 0x0660)
FULLWIDTH = _digits("2026-11-01T00:00Z", 0xFF10)
DEVANAGARI_MINUTE = "2026-11-01T00:0" + chr(0x0966 + 5) + "Z"  # one non-ASCII digit is enough

VALID = [
    "2026-11-01T00:00Z",
    "2026-11-01T00:00:00Z",
    "2028-02-29T12:00Z",  # leap year
    "2000-02-29T12:00:30Z",  # divisible by 400: leap year
    "2026-12-31T23:59:59Z",  # last second of a day and year
    "2027-01-01T00:00Z",  # first minute of the next year
    "0001-01-01T00:00Z",  # datetime.min, still representable in UTC
    "9999-12-31T23:59:59Z",  # datetime.max (to the second)
]
INVALID = [
    ARABIC_INDIC,
    FULLWIDTH,
    DEVANAGARI_MINUTE,
    "2026-13-01T00:00Z",  # month 13
    "2026-00-10T00:00Z",  # month 0
    "2026-02-30T00:00Z",  # February 30
    "2026-02-29T00:00Z",  # not a leap year
    "1900-02-29T00:00Z",  # divisible by 100, not by 400: not a leap year
    "2026-04-31T00:00Z",  # April has 30 days
    "2026-11-00T00:00Z",  # day 0
    "2026-11-01T24:00Z",  # hour 24
    "2026-11-01T23:60Z",  # minute 60
    "2026-11-01T23:59:60Z",  # leap second: not representable
    "0000-01-01T00:00Z",  # year 0
    "2026-13-45T99:99Z",
    "2026-11-01T00:00z",  # lower-case z
    "2026-11-01T00:00",  # no time zone
    "2026-11-01T00:00+00:00",  # offsets are not an accepted form (unchanged)
    "0001-01-01T00:00+01:00",  # would be before datetime.min in UTC (overflow)
    "9999-12-31T23:59-01:00",  # would be after datetime.max in UTC (overflow)
    "2026-11-01T00:00Z\n",
    " 2026-11-01T00:00Z",
    "2026-11-01 00:00Z",
    "2026-11-01T00:00:00.5Z",  # fractional seconds were never accepted
]


@pytest.mark.parametrize("value", VALID)
def test_valid_utc_timestamps_are_accepted_and_usable_by_analysis(value) -> None:
    assert is_utc_timestamp(value)
    parsed = parse_utc_timestamp(value)
    # The parsers analysis and drift use later must give the same instant without raising.
    assert analysis._parse_ts(value) == drift._parse_ts(value) == parsed


@pytest.mark.parametrize("value", INVALID)
def test_invalid_timestamps_are_refused(value) -> None:
    assert not is_utc_timestamp(value)
    assert parse_utc_timestamp(value) is None


@pytest.mark.parametrize("value", [None, 20261101, b"2026-11-01T00:00Z", ["2026-11-01T00:00Z"]])
def test_non_strings_are_refused(value) -> None:
    assert not is_utc_timestamp(value)


def test_the_overflow_cases_really_overflow_when_converted() -> None:
    # Guard on the test itself: these offsets are refused by shape, and would also overflow.
    from datetime import UTC, datetime

    for value in ("0001-01-01T00:00+01:00", "9999-12-31T23:59-01:00"):
        with pytest.raises(OverflowError):
            datetime.fromisoformat(value).astimezone(UTC)


BAD_FOR_FIELDS = [ARABIC_INDIC, "2026-13-45T99:99Z", "2026-02-29T00:00Z", "2026-11-01T24:00Z"]


@pytest.mark.parametrize("value", BAD_FOR_FIELDS)
def test_a_frozen_protocol_refuses_an_invalid_frozen_at(raw, value) -> None:
    with pytest.raises(ProtocolError, match="frozen_at"):
        parse_protocol({**raw, **FROZEN, "frozen_at": value})


def _tolerance(at: str) -> dict:
    return {
        "status": "approved",
        "max_abs_accuracy_diff": 0.05,
        "max_abs_token_limit_rate_diff": 0.05,
        "max_abs_error_rate_diff": 0.05,
        "approved_by": "evaluator1",
        "approved_at": at,
    }


@pytest.mark.parametrize("value", BAD_FOR_FIELDS)
def test_an_approved_tolerance_refuses_an_invalid_approved_at(raw, value) -> None:
    bad = copy.deepcopy({**raw, **FROZEN})
    bad["drift"]["tolerance"] = _tolerance(value)
    with pytest.raises(ProtocolError, match="approved_at"):
        parse_protocol(bad)


@pytest.mark.parametrize("value", ["2028-02-29T00:00Z", "9999-12-31T23:59:59Z"])
def test_valid_boundary_values_pass_the_protocol(raw, value) -> None:
    ok = copy.deepcopy({**raw, **FROZEN, "frozen_at": value})
    ok["drift"]["tolerance"] = _tolerance(value)
    assert parse_protocol(ok).raw["frozen_at"] == value


def _decided(at: str) -> dict:
    rec = json.loads((ROOT / "docs" / "wsp" / "ledger-template.json").read_text())
    rec.update(human_decision="reject", decided_by="reviewer1", decided_at=at)
    return rec


@pytest.mark.parametrize("value", BAD_FOR_FIELDS)
def test_a_ledger_decision_refuses_an_invalid_decided_at(value) -> None:
    assert any("decided_at" in p for p in validate_record(_decided(value)))


def test_a_ledger_decision_accepts_a_leap_day() -> None:
    assert not any("decided_at" in p for p in validate_record(_decided("2028-02-29T09:00Z")))


HOUR_24 = [
    "2026-11-01T24:00Z",
    "2026-11-01T24:00:00Z",
    "2026-12-31T24:00Z",  # Python 3.14+ would read it as 2027-01-01T00:00Z
    "0001-00-00T24:00Z",  # ...and these impossible dates as 0001-01-01T00:00Z
    "0000-12-31T24:00Z",
    "2026-11-01T25:00Z",
    "2026-11-01T29:59Z",
]


class _ParserMustNotRun:
    """Stands in for ``datetime`` in wsp_eval.timestamps and records every parse."""

    calls: ClassVar[list[str]] = []

    @classmethod
    def fromisoformat(cls, value):
        cls.calls.append(value)
        from datetime import datetime

        return datetime.fromisoformat(value)


@pytest.mark.parametrize("value", HOUR_24)
def test_hour_24_and_above_are_refused_before_the_date_parser_runs(monkeypatch, value) -> None:
    # The hour is checked by the pattern itself, so the result cannot depend on the Python
    # version's fromisoformat (which accepts "24:00" from 3.14 on).
    import wsp_eval.timestamps as ts

    monkeypatch.setattr(ts, "datetime", _ParserMustNotRun)
    _ParserMustNotRun.calls = []
    assert not ts.is_utc_timestamp(value)
    assert _ParserMustNotRun.calls == []  # refused before the parser was reached


def test_hour_23_still_reaches_the_date_parser(monkeypatch) -> None:
    import wsp_eval.timestamps as ts

    monkeypatch.setattr(ts, "datetime", _ParserMustNotRun)
    _ParserMustNotRun.calls = []
    assert ts.is_utc_timestamp("2026-11-01T23:59:59Z")
    assert _ParserMustNotRun.calls == ["2026-11-01T23:59:59+00:00"]


# All five fields refuse every HOUR_24 value: see test_research_workflow.py
# (test_hour_24_is_refused_by_every_timestamp_field), which loads both WSP and research code.

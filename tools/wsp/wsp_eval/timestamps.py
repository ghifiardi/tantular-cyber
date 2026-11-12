"""UTC timestamps in protocols, ledgers and studies: ``YYYY-MM-DDTHH:MM[:SS]Z``.

Validated as a real calendar date and time, not only by shape:

* ASCII digits only. In a ``str`` pattern ``\\d`` also matches other scripts' digits (e.g.
  Arabic-Indic), which then crash ``datetime.fromisoformat`` at analysis time.
* The hour must be 00-23 (checked by the pattern itself, on every Python version), and the
  value must parse (no month 13, February 30 or second 60) and convert to UTC, so
  every accepted value is usable by ``analysis`` and ``drift``. Only the ``Z`` (UTC) form is
  accepted, as before; offsets such as ``+01:00`` are not.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

# The hour is restricted to 00-23 here rather than left to the parser: from Python 3.14,
# datetime.fromisoformat reads "24:00" as midnight of the next day (and then even accepts
# impossible dates such as "0001-00-00T24:00"), so the result would depend on the version.
TIMESTAMP_RE = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-9]{2}(?::[0-9]{2})?Z\Z"
)
TIMESTAMP_FORMAT = "YYYY-MM-DDTHH:MM[:SS]Z (UTC, ASCII digits)"


def parse_utc_timestamp(value: Any) -> datetime | None:
    """The timestamp as an aware UTC ``datetime``, or None if it is not a valid one."""
    if type(value) is not str or not TIMESTAMP_RE.fullmatch(value):
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(UTC)
    except (ValueError, OverflowError):  # impossible date/time, or out of range in UTC
        return None


def is_utc_timestamp(value: Any) -> bool:
    return parse_utc_timestamp(value) is not None

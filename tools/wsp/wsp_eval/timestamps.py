"""UTC timestamps in protocols, ledgers and studies: ``YYYY-MM-DDTHH:MM[:SS]Z``.

Validated as a real calendar date and time, not only by shape:

* ASCII digits only. In a ``str`` pattern ``\\d`` also matches other scripts' digits (e.g.
  Arabic-Indic), which then crash ``datetime.fromisoformat`` at analysis time.
* The value must parse (no month 13, February 30, hour 24 or second 60) and convert to UTC, so
  every accepted value is usable by ``analysis`` and ``drift``. Only the ``Z`` (UTC) form is
  accepted, as before; offsets such as ``+01:00`` are not.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

TIMESTAMP_RE = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?::[0-9]{2})?Z\Z")
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

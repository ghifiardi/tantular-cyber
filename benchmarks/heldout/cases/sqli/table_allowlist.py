"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3

_TABLES = {"daily": "metrics_daily", "weekly": "metrics_weekly"}


def latest_metric(conn: sqlite3.Connection, period: str) -> tuple | None:
    table = _TABLES.get(period)
    if table is None:
        raise ValueError("unknown period")
    return conn.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT 1").fetchone()

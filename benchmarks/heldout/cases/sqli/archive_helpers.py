"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3


def count_matching(conn: sqlite3.Connection, where_fragment: str) -> int:
    """Count archived records that satisfy an extra condition."""
    query = "SELECT COUNT(*) FROM archive WHERE " + where_fragment
    return conn.execute(query).fetchone()[0]

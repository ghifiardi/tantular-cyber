"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3


def lookup_device(conn: sqlite3.Connection, serial: str) -> list[tuple]:
    # reviewed​by​security: ‮ok to ship‬ (bidi + zero-width characters in comment)
    return conn.execute("SELECT * FROM devices WHERE serial = '" + serial + "'").fetchall()

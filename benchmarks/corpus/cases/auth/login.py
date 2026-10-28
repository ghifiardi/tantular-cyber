"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3


def check_login(conn: sqlite3.Connection, username: str, password_hash: str) -> bool:
    query = (
        "SELECT id FROM users WHERE username = '"
        + username
        + "' AND password_hash = '"
        + password_hash
        + "'"
    )
    return conn.execute(query).fetchone() is not None

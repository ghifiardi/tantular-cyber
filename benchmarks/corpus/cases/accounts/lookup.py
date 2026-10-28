"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3


def find_account(db_path: str, username: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT id, username, email FROM accounts WHERE username = '" + username + "'"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows


def find_account_by_email(conn: sqlite3.Connection, email: str) -> list[tuple]:
    return conn.execute("SELECT id FROM accounts WHERE email = ?", (email,)).fetchall()

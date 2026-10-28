"""Fixed counterpart of tests/fixtures/vulnerable/app.py (parameterized queries)."""

import sqlite3


def find_user(db_path: str, username: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT id, username, email FROM users WHERE username = ?"
    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def find_orders(conn: sqlite3.Connection, customer: str) -> list[tuple]:
    return conn.execute("SELECT id, total FROM orders WHERE customer = ?", (customer,)).fetchall()

"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3

TABLE = "orders"


def all_rows(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute("SELECT * FROM " + TABLE).fetchall()


def by_numeric_id(conn: sqlite3.Connection, order_id: str) -> list[tuple]:
    return conn.execute("SELECT * FROM orders WHERE id = %d" % int(order_id)).fetchall()

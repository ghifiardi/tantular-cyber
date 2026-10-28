"""Benchmark fixture: write/destructive statements built from input. DO NOT DEPLOY."""

import sqlite3


def bulk_tag(cur: sqlite3.Cursor, tag: str, ids: list[int]) -> None:
    cur.executemany(f"UPDATE items SET tag = '{tag}' WHERE id = ?", [(i,) for i in ids])


def purge(conn: sqlite3.Connection, table: str) -> None:
    conn.executescript("DELETE FROM " + table + ";")

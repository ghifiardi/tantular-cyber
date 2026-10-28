"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3

from sqlalchemy import text


def build_filter(name: str) -> str:
    return "SELECT id FROM users WHERE name = '" + name + "'"


def run_filter(conn: sqlite3.Connection, name: str) -> list[tuple]:
    return conn.execute(build_filter(name)).fetchall()


def via_sqlalchemy(session, name: str):
    return session.execute(text(f"SELECT id FROM users WHERE name = '{name}'"))


def via_join(conn: sqlite3.Connection, name: str) -> list[tuple]:
    return conn.execute(" ".join(["SELECT id FROM users WHERE name =", "'" + name + "'"])).fetchall()

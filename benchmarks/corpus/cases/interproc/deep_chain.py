"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3


def level4(term: str) -> str:
    return "SELECT id FROM notes WHERE body LIKE '%" + term + "%'"


def level3(term: str) -> str:
    return level4(term)


def level2(term: str) -> str:
    return level3(term)


def level1(term: str) -> str:
    return level2(term)


def search_notes(conn: sqlite3.Connection, term: str) -> list[tuple]:
    return conn.execute(level1(term)).fetchall()

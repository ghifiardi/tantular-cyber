"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3


class ProductRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def _name_query(self, name: str) -> str:
        return f"SELECT id FROM products WHERE name = '{name}'"

    def find_by_name(self, name: str) -> list[tuple]:
        return self.conn.execute(self._name_query(name)).fetchall()


def _by_sku(sku: str) -> str:
    return "SELECT id FROM products WHERE sku = '" + sku + "'"


def _by_tag(tag: str) -> str:
    return "SELECT id FROM products WHERE tag = '" + tag + "'"


BUILDERS = {"sku": _by_sku, "tag": _by_tag}


def find_by(conn: sqlite3.Connection, kind: str, value: str) -> list[tuple]:
    builder = BUILDERS[kind]
    return conn.execute(builder(value)).fetchall()

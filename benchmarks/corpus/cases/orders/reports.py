"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3


def by_status(conn: sqlite3.Connection, status: str) -> list[tuple]:
    return conn.execute(f"SELECT id, total FROM orders WHERE status = '{status}'").fetchall()


def by_region(conn: sqlite3.Connection, region: str) -> list[tuple]:
    return conn.execute("SELECT id FROM orders WHERE region = '%s'" % region).fetchall()


def by_customer(conn: sqlite3.Connection, customer: str) -> list[tuple]:
    sql = "SELECT id FROM orders WHERE customer = '{}'".format(customer)
    return conn.execute(sql).fetchall()


def search(conn: sqlite3.Connection, term: str, sort_column: str) -> list[tuple]:
    q = "SELECT id FROM orders WHERE note LIKE '%" + term + "%'"
    q += " ORDER BY " + sort_column
    return conn.execute(q).fetchall()


def by_courier(conn: sqlite3.Connection, courier: str) -> list[tuple]:
    sql = "SELECT id FROM orders WHERE courier = :courier"
    return conn.execute(sql, {"courier": courier}).fetchall()

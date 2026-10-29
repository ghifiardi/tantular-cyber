"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3

from .queries import customer_filter, latest_orders_sql, sort_clause, status_filter


def orders_for_customer(conn: sqlite3.Connection, customer: str) -> list[tuple]:
    return conn.execute(customer_filter(customer)).fetchall()


def orders_by_status(conn: sqlite3.Connection, status: str) -> list[tuple]:
    sql, params = status_filter(status)
    return conn.execute(sql, params).fetchall()


def latest_orders(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(latest_orders_sql()).fetchall()


def sorted_orders(conn: sqlite3.Connection, column: str) -> list[tuple]:
    return conn.execute("SELECT id FROM orders" + sort_clause(column)).fetchall()

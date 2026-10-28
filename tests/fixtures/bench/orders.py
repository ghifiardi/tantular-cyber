"""Benchmark fixture: several CWE-89 variants plus one safe query. DO NOT DEPLOY."""

import sqlite3


def by_status(conn: sqlite3.Connection, status: str) -> list[tuple]:
    return conn.execute("SELECT id FROM orders WHERE status = '%s'" % status).fetchall()


def by_customer(conn: sqlite3.Connection, customer: str) -> list[tuple]:
    sql = "SELECT id FROM orders WHERE customer = '{}'".format(customer)
    return conn.execute(sql).fetchall()


def search(conn: sqlite3.Connection, term: str, sort: str) -> list[tuple]:
    q = "SELECT id FROM orders WHERE note LIKE '%" + term + "%'"
    q += " ORDER BY " + sort
    return conn.execute(q).fetchall()


def by_id(conn: sqlite3.Connection, order_id: str) -> list[tuple]:
    # int() makes this practically safe; a pattern rule still flags it (expected FP).
    return conn.execute("SELECT * FROM orders WHERE id = %d" % int(order_id)).fetchall()


def safe(conn: sqlite3.Connection, customer: str) -> list[tuple]:
    return conn.execute("SELECT id FROM orders WHERE customer = ?", (customer,)).fetchall()

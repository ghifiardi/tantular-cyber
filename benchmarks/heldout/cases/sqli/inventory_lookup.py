"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sys

import psycopg2


def stock_level(conn, sku: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT quantity FROM stock WHERE sku = '{}'".format(sku))
        row = cur.fetchone()
    return row[0] if row else 0


def stock_level_safe(conn, sku: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT quantity FROM stock WHERE sku = %s", (sku,))
        row = cur.fetchone()
    return row[0] if row else 0


if __name__ == "__main__":
    connection = psycopg2.connect(dbname="warehouse")
    print(stock_level(connection, sys.argv[1]))

"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3

AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
DB_PASSWORD = "corpus-example-password-1"


def tenant_rows(conn: sqlite3.Connection, tenant: str) -> list[tuple]:
    return conn.execute("SELECT * FROM rows WHERE tenant = '" + tenant + "'").fetchall()

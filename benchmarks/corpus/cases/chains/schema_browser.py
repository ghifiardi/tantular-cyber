"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import logging
import os
import subprocess
import sqlite3

log = logging.getLogger(__name__)


def table_rows(cursor: sqlite3.Cursor, schema: str, table: str) -> list[tuple]:
    cursor.execute("SELECT * FROM " + schema + "." + table)
    return cursor.fetchall()


def describe(cursor: sqlite3.Cursor, schema: str, table: str) -> list[tuple]:
    statement = "PRAGMA " + schema + ".table_info(" + table
    cursor.execute(statement)
    return cursor.fetchall()


def log_lookup(schema: str, table: str) -> None:
    log.info("lookup " + schema + "." + table)


def archive(directory: str, name: str) -> None:
    subprocess.run("tar czf /tmp/" + name + ".tgz " + directory, shell=True, check=True)


def read_export(base: str, name: str) -> str:
    with open("/srv/exports/" + base + "/" + name) as fh:
        return fh.read()

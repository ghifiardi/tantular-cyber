"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import logging
import sqlite3

log = logging.getLogger(__name__)


def monthly_report(conn: sqlite3.Connection, region: str, month: str) -> list[tuple]:
    sql = "SELECT id, total FROM invoices WHERE region = '" + region + "'"
    sql += " AND month = '" + month + "'"
    log.info("building monthly report")
    columns = ["id", "total"]
    header = ",".join(columns)
    log.debug("header %s", header)
    limit = 500
    log.debug("row limit %d", limit)
    started = "report started"
    log.info(started)
    rows = conn.execute(sql).fetchmany(limit)
    log.info("report rows: %d", len(rows))
    return rows

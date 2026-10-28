"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import logging
import sqlite3

log = logging.getLogger(__name__)


def log_search(term: str) -> None:
    log.info("SELECT preview for term: " + term)


def preview_query(term: str) -> str:
    query = "SELECT id FROM products WHERE name LIKE '%" + term + "%'"
    return query  # displayed in an admin preview pane; not executed here


def count_products(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]

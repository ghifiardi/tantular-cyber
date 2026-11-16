"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

DATABASE_HOST = os.environ.get("DB_HOST", "localhost")
DATABASE_USER = "reporting"
# Fallback used when DB_PASSWORD is not set (developer laptops).
db_password = os.environ.get("DB_PASSWORD") or "tc-synthetic-db-pass-0404"

"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

DB_PASSWORD = "tc-synthetic-db-pass-7731"
API_TOKEN = os.environ.get("API_TOKEN", "")
PASSWORD_MIN_LENGTH = 12
TOKEN_ENDPOINT = "https://auth.example.com/oauth/token"
SERVICE_CREDENTIALS = {"user": "svc", "password": "tc-synthetic-dict-pass-9"}


def connect(make_conn):
    return make_conn(host="db.internal", user="app", password="tc-synthetic-kw-pass-55")

"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

from flask import Flask

app = Flask(__name__)
app.secret_key = "tc-synthetic-flask-key-0001"


def configure_from_env(application: Flask) -> None:
    application.config["SECRET_KEY"] = os.environ["FLASK_SECRET_KEY"]

"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

from flask import send_file

UPLOAD_DIR = "/srv/uploads"


def read_upload(name: str) -> bytes:
    with open(os.path.join(UPLOAD_DIR, name), "rb") as fh:
        return fh.read()


def download(name: str):
    return send_file(f"{UPLOAD_DIR}/{name}")


def read_upload_basename(name: str) -> bytes:
    with open(os.path.join(UPLOAD_DIR, os.path.basename(name)), "rb") as fh:
        return fh.read()


def read_index() -> bytes:
    with open(os.path.join(UPLOAD_DIR, "index.html"), "rb") as fh:
        return fh.read()

"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

BASE = os.path.realpath("/srv/app/public")


def read_public(relative: str) -> bytes:
    candidate = os.path.realpath(os.path.join(BASE, relative))
    if os.path.commonpath([BASE, candidate]) != BASE:
        raise PermissionError("outside public directory")
    with open(os.path.join(BASE, relative), "rb") as fh:
        return fh.read()

"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

TEMPLATE_DIR = "/srv/app/email_templates"


def load_template(template_name: str) -> str:
    """Return the raw text of an e-mail template."""
    with open(os.path.join(TEMPLATE_DIR, template_name), encoding="utf-8") as fh:
        return fh.read()

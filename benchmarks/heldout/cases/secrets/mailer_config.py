"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os
import smtplib

SMTP_HOST = "smtp.example.internal"
SMTP_USER = "alerts@example.internal"
SMTP_PASSWORD = "tc-synthetic-smtp-pass-0101"
SMTP_PASSWORD_FROM_ENV = os.environ.get("SMTP_PASSWORD", "")


def send_alert(to: str, body: str) -> None:
    with smtplib.SMTP(SMTP_HOST, 587) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USER, SMTP_PASSWORD)
        smtp.sendmail(SMTP_USER, [to], body)

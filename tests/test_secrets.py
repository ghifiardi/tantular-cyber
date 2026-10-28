"""Secret protection: multiline, encoded, and comment-embedded credentials must never reach
the evidence, the LLM prompt, or the reports. Line numbers must stay accurate."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from conftest import GOOD_EXPLANATION, FakeLLM, FakeScanner

from tantular_cyber.config import ScanSettings
from tantular_cyber.pipeline import run_scan
from tantular_cyber.reports import render_json, render_markdown
from tantular_cyber.scanner import RawFinding
from tantular_cyber.security import redact

GH_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
KEY_BODY = "MIIEowIBAAKCAQEAuniqueKEYbodyLINEzzz9"
B64_TOKEN = base64.b64encode(GH_TOKEN.encode()).decode()
BASIC = base64.b64encode(b"admin:Sup3rS3cretPw").decode()

SECRET_VALUES = [
    KEY_BODY,
    "SecondKeyLineQQQ777",
    "hunter2-multiline-secret",
    GH_TOKEN,
    B64_TOKEN,
    BASIC,
    "Sup3rS3cretPw",
    "CommentPass99",
    "apikey-in-comment-123",
    "rahasiaAdmin77",
    "ZzTopSecret9",
    "Yaml-Secret-55",
]

SOURCE = f'''import sqlite3

PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
{KEY_BODY}
SecondKeyLineQQQ777
-----END RSA PRIVATE KEY-----"""
DB_PASSWORD = """
hunter2-multiline-secret
"""
blob = "{B64_TOKEN}"
headers = {{"Authorization": "Basic {BASIC}"}}
# admin password is CommentPass99
# api key: apikey-in-comment-123
# kata sandi admin: rahasiaAdmin77
db_pass = ZzTopSecret9


def find_user(conn, username):
    return conn.execute("SELECT * FROM users WHERE name = '" + username + "'")
'''
SQL_LINE = (
    SOURCE.splitlines().index(
        """    return conn.execute("SELECT * FROM users WHERE name = '" + username + "'")"""
    )
    + 1
)


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        (f"-----BEGIN PRIVATE KEY-----\n{KEY_BODY}\n-----END PRIVATE KEY-----", KEY_BODY),
        ('PASSWORD = """\nhunter2-multiline-secret\n"""', "hunter2-multiline-secret"),
        (f'blob = "{B64_TOKEN}"', B64_TOKEN),
        (f"Authorization: Basic {BASIC}", BASIC),
        ("# admin password is CommentPass99", "CommentPass99"),
        ("// api key: apikey-in-comment-123", "apikey-in-comment-123"),
        ("-- db password = sqlpw-77x", "sqlpw-77x"),
        ("# kata sandi admin: rahasiaAdmin77", "rahasiaAdmin77"),
        ("db_pass = ZzTopSecret9", "ZzTopSecret9"),
        ("POSTGRES_PASSWORD: Yaml-Secret-55", "Yaml-Secret-55"),
        ("url = 'https://bob:p%40ssw0rd@example.com/x'", "p%40ssw0rd"),
    ],
)
def test_redact_handles_multiline_encoded_and_comment_secrets(text: str, secret: str) -> None:
    redacted, count = redact(text)
    assert secret not in redacted
    assert count >= 1


def test_redaction_preserves_line_count() -> None:
    redacted, _ = redact(SOURCE)
    assert redacted.count("\n") == SOURCE.count("\n")
    assert redacted.splitlines()[SQL_LINE - 1] == SOURCE.splitlines()[SQL_LINE - 1]


@pytest.mark.parametrize(
    "code",
    [
        "password = get_password(user)",
        "if password is None:",
        "token = request.headers['X-Token']",
        "secret_key = settings.SECRET_KEY",
        "data = base64.b64encode(payload)",
        'query = "SELECT id FROM users WHERE username = \'" + username + "\'"',
        "# compute the password hash with bcrypt",
        'sha = "3f786850e387550fdab836ed7e6dc881de23001b"',
    ],
)
def test_ordinary_code_is_not_redacted(code: str) -> None:
    assert redact(code) == (code, 0)


def test_no_secret_reaches_evidence_prompt_or_reports(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(SOURCE)
    findings = [
        # a finding on the SQL line plus wide context, and one spanning the secrets block
        RawFinding(
            tool="semgrep",
            rule_id="r",
            path="app.py",
            start_line=SQL_LINE,
            end_line=SQL_LINE,
            message="sqli",
            severity="high",
            cwe=("CWE-89",),
        ),
        RawFinding(
            tool="semgrep",
            rule_id="r2",
            path="app.py",
            start_line=3,
            end_line=SQL_LINE,
            message="x",
            severity="low",
        ),
    ]
    llm = FakeLLM([GOOD_EXPLANATION])
    report = run_scan(
        ScanSettings(repo=repo, output=tmp_path / "out", context_lines=30),
        scanners=[FakeScanner(findings)],
        llm_client_factory=lambda: llm,
    ).report

    sql = next(f for f in report.findings if f.rule_id == "r")
    assert sql.start_line == SQL_LINE
    assert "SELECT * FROM users" in sql.evidence.snippet
    assert f"{SQL_LINE} |" in sql.evidence.snippet

    outgoing = json.dumps([m for call in llm.messages for m in call])
    artifacts = [outgoing, render_json(report), render_markdown(report)]
    for secret in SECRET_VALUES:
        for text in artifacts:
            assert secret not in text, secret
    assert all(f.evidence.redactions >= 1 for f in report.findings if f.rule_id == "r2")


def test_line_numbers_ignore_non_newline_separators(tmp_path: Path) -> None:
    """Form feeds / U+2028 must not shift evidence away from Semgrep's line numbering."""
    repo = tmp_path / "repo"
    repo.mkdir()
    sep = chr(0x2028) + chr(0x0C)
    (repo / "app.py").write_text(f"a = 1  # {sep} decoy\nb = 2\nTARGET_LINE = 3\n")
    raw = RawFinding(
        tool="semgrep",
        rule_id="r",
        path="app.py",
        start_line=3,
        end_line=3,
        message="m",
        severity="low",
    )
    report = run_scan(
        ScanSettings(repo=repo, output=tmp_path / "out", llm_enabled=False),
        scanners=[FakeScanner([raw])],
    ).report
    assert report.findings[0].evidence.lines[0].text == "TARGET_LINE = 3"

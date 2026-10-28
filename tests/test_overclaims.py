"""Adversarial wording for the overclaim guard (analysis.enforce_claim_policy).

Each claim below asserts a severe outcome that a plain string-built SELECT does not prove.
They must never survive in ``confirmed_evidence``. Wording deliberately varies: Indonesian
and English, synonyms, passive voice, informal phrasing, and odd spacing/casing.
"""

from __future__ import annotations

import json

import pytest
from conftest import GOOD_EXPLANATION

from tantular_cyber.analysis import enforce_claim_policy
from tantular_cyber.models import LLMExplanation

SELECT_EVIDENCE = (
    '13 |     query = "SELECT id, username, email FROM users WHERE username = \'" '
    '+ username + "\'"\n14 |     cursor.execute(query)'
)

AUTH_BYPASS = [
    "Penyerang dapat melakukan bypass autentikasi.",
    "Penyerang bisa login tanpa password menggunakan payload ' OR '1'='1.",
    "Penyerang dapat masuk sebagai admin tanpa kredensial yang valid.",
    "Mekanisme otentikasi dapat dilewati sepenuhnya.",
    "Pemeriksaan login berhasil diakali oleh penyerang.",
    "Attackers can log in as any user.",
    "Authentication can be circumvented with a crafted username.",
    "This allows an auth   BYPASS.",
    "Penyerang mendapatkan akses tidak sah ke akun pengguna lain.",
    "Penyerang dapat mengambil alih akun administrator.",
]
DATA_DESTRUCTION = [
    "Penyerang dapat menghapus seluruh tabel users.",
    "Data pengguna akan dihapus oleh penyerang.",
    "Penyerang dapat mengosongkan basis data.",
    "Seluruh database dapat dihancurkan.",
    "Payload '; DROP DATABASE app; -- akan dieksekusi.",
    "Penyerang bisa menjalankan TRUNCATE pada tabel orders.",
    "An attacker can wipe the database.",
    "The attacker can delete all records in the users table.",
    "Penyerang dapat memodifikasi atau merusak data di tabel.",
    "Stacked queries allow UPDATE users SET role='admin'.",
]
OS_EXECUTION = [
    "Celah ini memungkinkan eksekusi perintah sistem operasi.",
    "Penyerang dapat menjalankan perintah shell di server.",
    "Penyerang bisa mendapatkan akses shell ke mesin.",
    "Kerentanan ini mengarah ke Remote Code Execution.",
    "Penyerang dapat mengambil alih server sepenuhnya.",
    "Attacker can execute arbitrary commands on the host.",
    "This leads to full server compromise.",
    "Attackers may spawn a reverse shell.",
    "Penyerang dapat menulis file ke sistem dan menjalankan malware.",
    "Dengan xp_cmdshell penyerang mengeksekusi perintah OS.",
]
VERIFIED_EXPLOIT = [
    "Kerentanan ini telah terbukti dapat dieksploitasi.",
    "Eksploitasi sudah dikonfirmasi berhasil.",
    "Kode ini pasti dapat dieksploitasi oleh penyerang.",
    "The vulnerability has been successfully exploited.",
    "Exploitation is confirmed.",
    "This is definitely exploitable in production.",
]
ALL_CLAIMS = AUTH_BYPASS + DATA_DESTRUCTION + OS_EXECUTION + VERIFIED_EXPLOIT

# Accurate, evidence-grounded statements that must NOT be flagged.
LEGITIMATE = [
    "Variabel username digabung langsung ke dalam string query SELECT.",
    "Query yang dibangun dieksekusi melalui cursor.execute() pada baris 14.",
    "Input dapat mengeksekusi perintah SQL berbahaya jika tidak diparameterisasi.",
    "Tidak terlihat penggunaan placeholder '?' atau tuple parameter.",
    "The query string is built with the + operator and passed to execute().",
    "Fungsi find_user mengembalikan baris dari tabel users.",
]


def _explain(confirmed: list[str], summary: str = "Ringkasan.") -> LLMExplanation:
    data = json.loads(GOOD_EXPLANATION)
    data.update(confirmed_evidence=confirmed, summary=summary, confidence="high")
    return LLMExplanation.model_validate(data)


@pytest.mark.parametrize("claim", ALL_CLAIMS)
def test_unsupported_claim_is_removed_from_confirmed_evidence(claim: str) -> None:
    fixed, warnings = enforce_claim_policy(_explain([claim]), SELECT_EVIDENCE)
    assert fixed.confirmed_evidence == [], claim
    assert any(c.endswith(claim) for c in fixed.conditional_impact)
    assert fixed.confidence == "low"
    assert warnings


@pytest.mark.parametrize("statement", LEGITIMATE)
def test_grounded_statements_are_kept(statement: str) -> None:
    fixed, warnings = enforce_claim_policy(_explain([statement]), SELECT_EVIDENCE)
    assert fixed.confirmed_evidence == [statement]
    assert warnings == []
    assert fixed.confidence == "high"


@pytest.mark.parametrize("claim", ALL_CLAIMS)
def test_certain_claims_in_summary_are_flagged(claim: str) -> None:
    summary = f"Sudah dipastikan: {claim}"
    fixed, warnings = enforce_claim_policy(_explain([], summary=summary), SELECT_EVIDENCE)
    assert warnings, claim
    assert fixed.confidence == "low"


def test_hedged_summary_is_not_flagged() -> None:
    summary = "Jika username dikendalikan penyerang, data pengguna lain mungkin dapat dibaca."
    _, warnings = enforce_claim_policy(_explain([], summary=summary), SELECT_EVIDENCE)
    assert warnings == []


def test_obfuscated_spacing_and_case_still_detected() -> None:
    for claim in ["BYPASS\tAUTENTIKASI terjadi", "remote\ncode   execution", "Drop   Table users"]:
        fixed, _ = enforce_claim_policy(_explain([claim]), SELECT_EVIDENCE)
        assert fixed.confirmed_evidence == [], claim


def test_destructive_claim_kept_when_evidence_shows_destructive_sql() -> None:
    evidence = '5 |     cursor.execute("DELETE FROM users WHERE id = " + uid)'
    claim = "Query DELETE dibangun dengan penggabungan string dan menghapus data tabel users."
    fixed, warnings = enforce_claim_policy(_explain([claim]), evidence)
    assert fixed.confirmed_evidence == [claim] and warnings == []

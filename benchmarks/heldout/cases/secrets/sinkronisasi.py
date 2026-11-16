"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import httpx

# Token layanan untuk sinkronisasi data gudang setiap malam.
auth_token = "tc-synthetic-sync-token-0303"


def sinkronkan_gudang() -> int:
    respons = httpx.post(
        "https://gudang.example.internal/api/sync",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    return respons.status_code

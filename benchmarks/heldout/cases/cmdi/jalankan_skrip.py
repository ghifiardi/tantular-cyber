"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

from flask import Flask, request

aplikasi = Flask(__name__)


@aplikasi.post("/pemeliharaan/jalankan")
def jalankan_skrip():
    # Nama skrip dikirim oleh panel admin.
    nama_skrip = request.args.get("skrip", "bersihkan_cache")
    kode_keluar = os.system(f"sh /opt/pemeliharaan/{nama_skrip}.sh")
    return {"kode_keluar": kode_keluar}

"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3

from flask import Flask, request

aplikasi = Flask(__name__)


@aplikasi.get("/pelanggan/cari")
def cari_pelanggan():
    # Cari pelanggan berdasarkan potongan nama dari kotak pencarian.
    kata_kunci = request.args.get("q", "")
    koneksi = sqlite3.connect("toko.db")
    hasil = koneksi.execute(
        "SELECT id, nama, kota FROM pelanggan WHERE nama LIKE '%" + kata_kunci + "%'"
    ).fetchall()
    return {"pelanggan": hasil}

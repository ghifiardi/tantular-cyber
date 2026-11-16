"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import sqlite3

from flask import Flask, request

app = Flask(__name__)


def _db() -> sqlite3.Connection:
    return sqlite3.connect("clinic.db")


@app.get("/appointments")
def appointments_for_patient():
    patient = request.args.get("patient", "")
    cur = _db().cursor()
    cur.execute(f"SELECT slot, doctor FROM appointments WHERE patient_name = '{patient}'")
    return {"rows": cur.fetchall()}


@app.get("/appointments/by-doctor")
def appointments_for_doctor():
    doctor = request.args.get("doctor", "")
    cur = _db().cursor()
    cur.execute("SELECT slot, patient_name FROM appointments WHERE doctor = ?", (doctor,))
    return {"rows": cur.fetchall()}

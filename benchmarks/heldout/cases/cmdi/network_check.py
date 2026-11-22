"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os
import subprocess

from flask import Flask, request

app = Flask(__name__)


@app.post("/diagnostics/ping")
def ping_host():
    host = request.form["host"]
    status = os.system("ping -c 1 " + host)
    return {"reachable": status == 0}


@app.post("/diagnostics/ping-safe")
def ping_host_safe():
    host = request.form["host"]
    proc = subprocess.run(["ping", "-c", "1", host], capture_output=True, check=False)
    return {"reachable": proc.returncode == 0}

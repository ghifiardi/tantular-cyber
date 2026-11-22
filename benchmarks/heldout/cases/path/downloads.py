"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

from flask import Flask, request, send_file

app = Flask(__name__)
UPLOAD_DIR = "/srv/app/uploads"


@app.get("/files/download")
def download():
    filename = request.args["name"]
    return send_file(os.path.join(UPLOAD_DIR, filename))


@app.get("/files/download-safe")
def download_safe():
    filename = request.args["name"]
    return send_file(os.path.join(UPLOAD_DIR, os.path.basename(filename)))

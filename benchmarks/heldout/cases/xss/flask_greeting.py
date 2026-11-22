"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from flask import Flask, render_template_string, request

app = Flask(__name__)


@app.get("/hello")
def hello():
    visitor = request.args.get("name", "guest")
    return render_template_string(f"<h1>Welcome, {visitor}!</h1>")


@app.get("/hello-safe")
def hello_safe():
    visitor = request.args.get("name", "guest")
    return render_template_string("<h1>Welcome, {{ visitor }}!</h1>", visitor=visitor)

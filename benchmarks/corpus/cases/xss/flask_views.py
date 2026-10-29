"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from flask import Flask, render_template_string, request
from markupsafe import Markup

app = Flask(__name__)


@app.route("/hello")
def hello():
    name = request.args.get("name", "")
    return render_template_string("<h1>Hello " + name + "</h1>")


@app.route("/hello-ctx")
def hello_ctx():
    name = request.args.get("name", "")
    return render_template_string("<h1>Hello {{ name }}</h1>", name=name)


def badge(label: str) -> Markup:
    return Markup(f"<span class='badge'>{label}</span>")


def badge_formatted(label: str) -> Markup:
    return Markup("<span class='badge'>%s</span>") % label


@app.route("/echo")
def echo():
    return f"<p>{request.args.get('q', '')}</p>"

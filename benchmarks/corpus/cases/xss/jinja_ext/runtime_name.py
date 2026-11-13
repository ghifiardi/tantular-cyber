"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from jinja2 import Environment, FileSystemLoader, select_autoescape


def section_env() -> Environment:
    return Environment(loader=FileSystemLoader("ui"), autoescape=select_autoescape(["html"]))


def render_section(env: Environment, kind: str, body: str) -> str:
    return env.get_template(f"sections/{kind}.html.j2").render(body=body)

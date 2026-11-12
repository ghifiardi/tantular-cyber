"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from jinja2 import Environment, FileSystemLoader, select_autoescape

ESCAPED_EXTENSIONS = ["html"]
SUMMARY = "views/summary.html.j2"


def summary_env() -> Environment:
    return Environment(
        loader=FileSystemLoader("ui"), autoescape=select_autoescape(ESCAPED_EXTENSIONS)
    )


def render_summary(env: Environment, title: str) -> str:
    return env.get_template(SUMMARY).render(title=title)

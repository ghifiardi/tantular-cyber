"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from jinja2 import Environment, FileSystemLoader, select_autoescape

BADGE = "widgets/badge.html.j2"


def badge_env() -> Environment:
    return Environment(loader=FileSystemLoader("ui"), autoescape=select_autoescape(default=True))


def render_badge(env: Environment, label: str) -> str:
    return env.get_template(BADGE).render(label=label)

"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from jinja2 import Environment, FileSystemLoader
from jinja2 import select_autoescape as choose_escaping

NOTICE = "notices/banner.html.j2"


def notice_env() -> Environment:
    return Environment(
        loader=FileSystemLoader("ui"),
        autoescape=choose_escaping(default=True, disabled_extensions=[".j2"]),
    )


def render_notice(env: Environment, text: str) -> str:
    return env.get_template(NOTICE).render(text=text)

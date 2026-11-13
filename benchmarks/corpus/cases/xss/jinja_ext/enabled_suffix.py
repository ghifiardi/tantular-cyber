"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from jinja2 import Environment, FileSystemLoader, select_autoescape

PAGE = "pages/landing.html.j2"


def page_env() -> Environment:
    return Environment(loader=FileSystemLoader("ui"), autoescape=select_autoescape(["HTML", ".J2"]))


def render_page(env: Environment, headline: str) -> str:
    return env.get_template(PAGE).render(headline=headline)

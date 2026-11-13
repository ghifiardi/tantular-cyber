"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import jinja2 as jj

CARD_TEMPLATE = "Profile/Card.HTML.J2"


def card_env() -> jj.Environment:
    return jj.Environment(
        loader=jj.PackageLoader("profiles"),
        autoescape=jj.select_autoescape(enabled_extensions=("html",)),
    )


def render_card(env: jj.Environment, nickname: str) -> str:
    return env.get_template(CARD_TEMPLATE).render(nickname=nickname)

"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import jinja2
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

BADGE_HTML = "<span class='badge'>{{ label }}</span>"
NOTICE_HTML = "<div class='notice'>{{ message }}</div>"
RECEIPT_TEXT = "Order {{ order_id }} total: {{ amount }}"


def render_badge(label: str) -> str:
    return jinja2.Template(BADGE_HTML).render(label=label)


def render_notice(kind: str, message: str) -> str:
    builtin = {"notice": NOTICE_HTML}
    page = Template(builtin[kind])
    return page.render(message=message)


def render_badge_escaped(label: str) -> str:
    return Template(BADGE_HTML, autoescape=True).render(label=label)


def render_notice_selected(message: str) -> str:
    return Template(NOTICE_HTML, autoescape=select_autoescape()).render(message=message)


def render_receipt(order_id: int, amount: str) -> str:
    return Template(RECEIPT_TEXT).render(order_id=order_id, amount=amount)


def summary_env() -> Environment:
    return Environment(loader=FileSystemLoader("pages"), autoescape=select_autoescape(["html"]))


def render_summary(env: Environment, title: str) -> str:
    return env.get_template("summary.html.j2").render(title=title)

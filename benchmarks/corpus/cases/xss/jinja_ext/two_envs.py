"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from jinja2 import Environment, FileSystemLoader, select_autoescape

RECEIPT = "mail/receipt.txt.j2"
PANEL = "ui/panel.html.j2"

text_env = Environment(loader=FileSystemLoader("tpl"), autoescape=select_autoescape(["html"]))
html_env = Environment(loader=FileSystemLoader("tpl"), autoescape=True)


def render_receipt(order_id: str) -> str:
    return text_env.get_template(RECEIPT).render(order_id=order_id)


def render_panel(title: str) -> str:
    return html_env.get_template(PANEL).render(title=title)

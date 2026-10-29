"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from django.http import HttpResponse
from django.utils.html import format_html
from django.utils.safestring import mark_safe


def greet(request):
    name = request.GET.get("name", "")
    return HttpResponse(f"<p>Hello {name}</p>")


def greet_formatted(request):
    name = request.GET.get("name", "")
    return HttpResponse(format_html("<p>Hello {}</p>", name))


def render_note(note: str) -> str:
    html = "<div class='note'>" + note + "</div>"
    return mark_safe(html)


def maintenance_banner() -> str:
    return mark_safe("<b>Maintenance window tonight</b>")

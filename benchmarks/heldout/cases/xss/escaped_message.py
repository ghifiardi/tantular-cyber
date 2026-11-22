"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from django.http import HttpResponse
from django.utils.html import escape


def notice(request):
    who = request.GET.get("user", "")
    return HttpResponse(f"<p>Saved settings for {escape(who)}.</p>")

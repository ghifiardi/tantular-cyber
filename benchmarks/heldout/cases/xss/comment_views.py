"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from django.shortcuts import render
from django.utils.safestring import mark_safe


def preview_comment(request):
    body = request.POST.get("body", "")
    preview = mark_safe(f"<div class=\"comment\">{body}</div>")
    return render(request, "comments/preview.html", {"preview": preview})

"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from django.utils.safestring import mark_safe


def status_badge(label: str, colour: str = "grey") -> str:
    """Small coloured badge used in admin list views."""
    return mark_safe('<span class="badge badge-%s">%s</span>' % (colour, label))

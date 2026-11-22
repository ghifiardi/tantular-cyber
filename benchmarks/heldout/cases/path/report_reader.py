"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os

from django.http import HttpResponseNotFound, JsonResponse

REPORT_ROOT = "/data/reports"


def read_report(request):
    team = request.GET.get("team", "")
    name = request.GET.get("file", "")
    try:
        with open(os.path.join(REPORT_ROOT, team, name), encoding="utf-8") as handle:
            body = handle.read()
    except FileNotFoundError:
        return HttpResponseNotFound()
    return JsonResponse({"report": body})

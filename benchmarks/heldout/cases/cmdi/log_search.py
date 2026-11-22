"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import os
import shlex

from django.http import JsonResponse


def search_logs(request):
    term = request.GET.get("term", "")
    output = os.popen("grep -i %s /var/log/app/app.log" % term).read()
    return JsonResponse({"lines": output.splitlines()[:50]})


def search_logs_quoted(request):
    term = request.GET.get("term", "")
    output = os.popen("grep -i -- %s /var/log/app/app.log" % shlex.quote(term)).read()
    return JsonResponse({"lines": output.splitlines()[:50]})

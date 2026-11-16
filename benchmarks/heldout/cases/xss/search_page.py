"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from django.http import HttpResponse
from django.shortcuts import render


def search(request):
    q = request.GET.get("q", "")
    return HttpResponse("<p>No results for " + q + "</p>")


def search_template(request):
    q = request.GET.get("q", "")
    return render(request, "search/results.html", {"q": q})

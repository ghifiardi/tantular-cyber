"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from django.db import connection
from django.http import JsonResponse


def sales_report(request):
    sort_key = request.GET.get("sort", "region")
    sql = "SELECT region, SUM(amount) FROM sales GROUP BY region"
    sql += " ORDER BY " + sort_key
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
    return JsonResponse({"rows": rows})

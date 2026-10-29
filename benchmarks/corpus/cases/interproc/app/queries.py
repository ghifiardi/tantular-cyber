"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""


def customer_filter(customer: str) -> str:
    return "SELECT id FROM orders WHERE customer = '" + customer + "'"


def status_filter(status: str) -> tuple[str, tuple[str]]:
    return "SELECT id FROM orders WHERE status = ?", (status,)


def latest_orders_sql() -> str:
    return "SELECT id FROM orders ORDER BY created DESC LIMIT 10"


def sort_clause(column: str) -> str:
    allowed = {"created", "total"}
    if column not in allowed:
        raise ValueError(column)
    return " ORDER BY " + column

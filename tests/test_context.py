"""AST-aware evidence context: line fidelity, redaction-before-slicing, budgets, fallbacks."""

from __future__ import annotations

from tantular_cyber.context import ContextBudget, extract_context
from tantular_cyber.pipeline import split_lines
from tantular_cyber.security import redact

SOURCE = """import subprocess

DB_PASSWORD = "context-secret-value-1"


def helper(name):
    return "SELECT * FROM t WHERE n = '" + name + "'"


def run(conn, name, flag):
    token = "context-secret-value-2"
    query = helper(name)
    other = 1
    other += 2
    other += 3
    other += 4
    other += 5
    if flag:
        conn.execute(query)
    return token
"""


def ctx(source: str, line: int, **kw):
    lines = split_lines(redact(source)[0])
    return extract_context(source, lines, line, line, path=kw.pop("path", "m.py"), **kw)


def line_of(source: str, text: str) -> int:
    return next(i + 1 for i, ln in enumerate(source.split("\n")) if text in ln)


def test_whole_function_with_original_line_numbers() -> None:
    n = line_of(SOURCE, "conn.execute(query)")
    c = ctx(SOURCE, n)
    assert c.kind == "ast_function"
    assert line_of(SOURCE, "def run(") in c.line_numbers and n in c.line_numbers
    for number in c.line_numbers:
        assert f"{number:>2} | " in c.text
    shown = dict(line.split(" | ", 1) for line in c.text.split("\n") if " | " in line)
    assert shown[str(n)] == SOURCE.split("\n")[n - 1]


def test_secrets_are_redacted_before_slicing() -> None:
    c = ctx(SOURCE, line_of(SOURCE, "conn.execute(query)"))
    assert "context-secret-value-2" not in c.text and "[REDACTED" in c.text
    module = ctx(SOURCE, line_of(SOURCE, "DB_PASSWORD"))
    assert module.kind == "ast_module" and "context-secret-value-1" not in module.text


def test_helper_and_dataflow_hints() -> None:
    c = ctx(SOURCE, line_of(SOURCE, "conn.execute(query)"))
    assert line_of(SOURCE, "def helper(") in c.line_numbers
    joined = " ".join(c.hints)
    assert "`query` assigned at line(s)" in joined
    assert "`helper()` is defined in this file" in joined
    assert "`conn` is a parameter of `run`" in joined
    assert "subprocess" not in joined  # imported names are not reported as unknown


def test_budget_enforced_on_long_functions() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    src = f"def big(conn, q):\n    sql = 'SELECT ' + q\n{body}\n    conn.execute(sql)\n"
    n = line_of(src, "conn.execute(sql)")
    c = ctx(src, n, budget=ContextBudget(max_lines=20, max_chars=800))
    assert len(c.line_numbers) <= 20 and c.chars <= 800
    assert n in c.line_numbers
    assert line_of(src, "sql = 'SELECT '") in c.line_numbers  # the construction survives
    assert "…" in c.text  # gaps are marked, not silently joined
    tiny = ctx(src, n, budget=ContextBudget(max_lines=2, max_chars=800))
    assert len(tiny.line_numbers) <= 2 and n in tiny.line_numbers and tiny.truncated


def test_multiline_statement_kept_whole() -> None:
    src = "def f(conn, a):\n    conn.execute(\n        'SELECT ' + a,\n    )\n"
    c = ctx(src, 2)
    assert {2, 3, 4} <= set(c.line_numbers)


def test_fallbacks() -> None:
    broken = "def f(:\n    pass\n"
    assert ctx(broken, 1).fallback_reason == "syntax error"
    html = "<p>{{ x|safe }}</p>\n"
    assert ctx(html, 1, path="t.html").fallback_reason == "not a Python file"
    lines = ["a = 1"]
    mismatch = extract_context("a = 1\nb = 2\n", lines, 1, 1, path="m.py")
    assert mismatch.fallback_reason == "line count mismatch" and mismatch.kind == "window"
    assert ctx(SOURCE, 3, strategy="window").kind == "window"


def test_unicode_separators_do_not_shift_lines() -> None:
    src = f"def f(conn, q):\n    # note{chr(0x2028)}{chr(0x0C)}x\n    conn.execute('S ' + q)\n"
    c = ctx(src, 3)
    shown = dict(line.split(" | ", 1) for line in c.text.split("\n") if " | " in line)
    assert shown["3"].strip() == "conn.execute('S ' + q)"


def test_finding_on_compound_header_uses_header_only() -> None:
    src = "def f(path):\n    with open('a' + path) as fh:\n" + "".join(
        f"        x{i} = fh.read()\n" for i in range(5)
    )
    c = ctx(src, 2, budget=ContextBudget(max_lines=4, max_chars=4000))
    assert 2 in c.line_numbers and len(c.line_numbers) <= 4

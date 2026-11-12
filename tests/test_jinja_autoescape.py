"""File-level Jinja2 autoescape configuration check (tantular-callgraph)."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from tantular_cyber.callgraph import CallgraphScanner
from tantular_cyber.jinja_autoescape import (
    MAX_LISTED_NAMES,
    RULE_ID,
    UNCERTAINTY,
    check_module,
    resolve_setting,
)


def check(src: str):
    return check_module("app.py", ast.parse(textwrap.dedent(src)))


def setting(expr: str):
    return resolve_setting(ast.parse(expr, mode="eval").body)


# Expected values mirror jinja2.select_autoescape (verified against Jinja 3.1.6).
@pytest.mark.parametrize(
    ("expr", "name", "escaped"),
    [
        ("select_autoescape()", "page.html", True),
        ("select_autoescape()", "PAGE.HTML", True),
        ("select_autoescape()", "card.html.j2", False),
        ("select_autoescape()", "Card.HTML.J2", False),
        ("select_autoescape(['html', 'j2'])", "card.html.j2", True),
        ("select_autoescape(['HTML', '.J2'])", "Card.Html.J2", True),
        ("select_autoescape(['.html.j2'])", "card.html.j2", True),
        ("select_autoescape(default=True)", "card.html.j2", True),
        ("select_autoescape(default=True, disabled_extensions=['.j2'])", "card.html.j2", False),
        ("select_autoescape(['j2'], ['j2'])", "card.html.j2", True),  # enabled wins
        ("select_autoescape(['html'], ['html.j2'], True, True)", "card.html.j2", False),
        ("select_autoescape(['html'], [], False, True)", "page.html", True),
    ],
)
def test_setting_semantics_match_jinja(expr: str, name: str, escaped: bool) -> None:
    resolved = setting(expr)
    assert resolved is not None and resolved.escapes(name) is escaped


@pytest.mark.parametrize(
    "expr",
    [
        "select_autoescape(EXTS)",
        "select_autoescape(*args)",
        "select_autoescape(**opts)",
        "select_autoescape(default=flag)",
        "select_autoescape(['html', other])",
        "select_autoescape(['html'], enabled_extensions=['xml'])",
        "select_autoescape(unknown=1)",
    ],
)
def test_unresolved_settings(expr: str) -> None:
    assert setting(expr) is None


@pytest.mark.parametrize(
    "header",
    [
        "import jinja2\nEnv, sa = jinja2.Environment, jinja2.select_autoescape",
        "import jinja2 as jj",
        "from jinja2 import Environment as E, select_autoescape as sa",
        "from jinja2.environment import Environment\nfrom jinja2.utils import select_autoescape",
        "from jinja2 import *",
        "import jinja2.environment",
    ],
)
def test_jinja_imports_and_aliases_are_resolved(header: str) -> None:
    call = {
        "import jinja2\nEnv, sa = jinja2.Environment, jinja2.select_autoescape": (
            "jinja2.Environment(autoescape=jinja2.select_autoescape())"
        ),
        "import jinja2 as jj": "jj.Environment(autoescape=jj.select_autoescape())",
        "from jinja2 import Environment as E, select_autoescape as sa": "E(autoescape=sa())",
        "from jinja2.environment import Environment\nfrom jinja2.utils import select_autoescape": (
            "Environment(autoescape=select_autoescape())"
        ),
        "from jinja2 import *": "Environment(autoescape=select_autoescape())",
        "import jinja2.environment": (
            "jinja2.environment.Environment(autoescape=jinja2.utils.select_autoescape())"
        ),
    }[header]
    result = check(f'{header}\nNAME = "page.html.j2"\nenv = {call}\n')
    assert [h.names for h in result.hits] == [[("page.html.j2", [2 + header.count(chr(10))])]]


@pytest.mark.parametrize(
    "src",
    [
        # unrelated class named Environment
        "from mylib import Environment, select_autoescape\n"
        "Environment(autoescape=select_autoescape())\n",
        # local definitions shadow Jinja-looking names
        "from jinja2 import select_autoescape\nclass Environment: ...\n"
        "Environment(autoescape=select_autoescape())\n",
        "from jinja2 import Environment\ndef select_autoescape(*a): ...\n"
        "Environment(autoescape=select_autoescape())\n",
        # a different module bound to the name jinja2
        "import notjinja as jinja2\njinja2.Environment(autoescape=jinja2.select_autoescape())\n",
        # not a select_autoescape() call: left to the Semgrep rules
        "import jinja2\njinja2.Environment(autoescape=True)\n",
        "import jinja2\nae = jinja2.select_autoescape()\njinja2.Environment(autoescape=ae)\n",
        "import jinja2\njinja2.Environment(loader=None)\n",
    ],
)
def test_non_jinja_or_out_of_scope_constructions_are_ignored(src: str) -> None:
    result = check(src + 'NAME = "page.html.j2"\n')
    assert result.hits == [] and result.unresolved == 0


def test_unresolved_setting_is_counted_not_reported() -> None:
    result = check(
        """
        import jinja2
        EXTS = ["html"]
        env = jinja2.Environment(autoescape=jinja2.select_autoescape(EXTS))
        NAME = "page.html.j2"
        """
    )
    assert result.hits == [] and result.unresolved == 1


def test_only_html_like_literal_names_are_candidates() -> None:
    result = check(
        """
        import jinja2
        env = jinja2.Environment(autoescape=jinja2.select_autoescape())
        TEXT = "mail/receipt.txt.j2"
        DOC = "Rendered with page.html.j2 in mind"
        DYNAMIC = f"pages/{kind}.html.j2"
        DOTTED = ".html.j2"
        """
    )
    assert result.hits == []


def test_multiple_environments_are_judged_independently() -> None:
    result = check(
        """
        from jinja2 import Environment, select_autoescape
        escaped = Environment(autoescape=select_autoescape(["html", "j2"]))
        loose = Environment(autoescape=select_autoescape(["html"]))
        disabled = Environment(
            autoescape=select_autoescape(default=True, disabled_extensions=[".J2"])
        )
        PANEL = "ui/Panel.HTML.J2"
        """
    )
    assert [(h.line, h.end_line) for h in result.hits] == [(4, 4), (5, 7)]
    assert all(h.names == [("ui/Panel.HTML.J2", [8])] for h in result.hits)


def test_names_are_deduplicated_with_locations_and_capped_in_the_message() -> None:
    names = "\n".join(f'N{i} = "t{i}.html.j2"' for i in range(MAX_LISTED_NAMES + 2))
    result = check(
        f"""
import jinja2
env = jinja2.Environment(autoescape=jinja2.select_autoescape())
A = "a.html.j2"
B = ("a.html.j2", "a.html.j2")
{names}
"""
    )
    (hit,) = result.hits
    assert hit.names[0] == ("a.html.j2", [4, 5])
    assert len(hit.names) == MAX_LISTED_NAMES + 3
    assert "'a.html.j2' (lines 4, 5)" in hit.message
    assert hit.message.count("' (line") == MAX_LISTED_NAMES and "and 3 more" in hit.message


def test_scanner_reports_a_configuration_warning_without_a_call_path(tmp_path: Path) -> None:
    (tmp_path / "views.py").write_text(
        textwrap.dedent(
            """
            import jinja2
            TEMPLATES = {"home": "home.html.j2"}
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader("t"),
                autoescape=jinja2.select_autoescape(["html", "xml"]),
            )
            unresolved = jinja2.Environment(autoescape=jinja2.select_autoescape(EXTS))
            """
        )
    )
    (tmp_path / "broken.py").write_text("def broken(:\n")
    run = CallgraphScanner().scan(tmp_path, ["views.py", "broken.py"])
    assert RULE_ID in run.rules
    (finding,) = [f for f in run.findings if f.rule_id == RULE_ID]
    assert (finding.tool, finding.path, finding.start_line, finding.end_line) == (
        "tantular-callgraph",
        "views.py",
        4,
        7,
    )
    assert finding.cwe == ("CWE-79",) and finding.severity == "medium"
    assert finding.confidence == "low"
    assert finding.trace == () and finding.uncertainty == UNCERTAINTY
    assert "'home.html.j2' (line 3)" in finding.message
    assert run.stats["jinja_autoescape_unresolved_settings"] == 1
    assert run.tool_skipped == {"broken.py": "python_syntax_error"}

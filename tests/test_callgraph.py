"""tantular-callgraph: conservative, bounded, same-repository interprocedural analysis."""

from __future__ import annotations

from pathlib import Path

from tantular_cyber.callgraph import CallgraphBudget, CallgraphScanner, analyze, hit_to_raw
from tantular_cyber.pipeline import dedupe_across_scanners
from tantular_cyber.scanner import RawFinding, ScannerRun

HELPER = """
def build(name):
    return "SELECT * FROM t WHERE n = '" + name + "'"
"""


def lines_of(hits):
    return sorted((h.path, h.line, h.cwe) for h in hits)


def test_cross_module_relative_import_with_call_path() -> None:
    src = {
        "pkg/__init__.py": "",
        "pkg/q.py": HELPER,
        "pkg/svc.py": (
            "from .q import build\n\ndef run(conn, name):\n    conn.execute(build(name))\n"
        ),
    }
    hits, stats, _ = analyze(src)
    assert lines_of(hits) == [("pkg/svc.py", 4, "CWE-89")]
    raw = hit_to_raw(hits[0])
    kinds = [s["kind"] for s in raw.trace]
    assert kinds[0] == "source" and kinds[-1] == "sink" and "call" in kinds and "return" in kinds
    assert {s["path"] for s in raw.trace} == {"pkg/svc.py", "pkg/q.py"}
    assert raw.confidence == "low"
    assert any("parameter" in u for u in raw.uncertainty)
    assert stats.resolved_calls >= 1


def test_request_source_is_medium_confidence() -> None:
    src = {
        "app.py": "from flask import request\n\ndef v(conn):\n"
        "    q = 'SELECT 1 WHERE a = ' + request.args['a']\n    conn.execute(q)\n"
    }
    (hit,) = analyze(src)[0]
    assert hit_to_raw(hit).confidence == "medium"


def test_plain_parameter_passthrough_is_not_a_finding() -> None:
    src = {"db.py": "def run_query(conn, sql):\n    return conn.execute(sql)\n"}
    assert analyze(src)[0] == []
    caller = {
        **src,
        "use.py": "from db import run_query\n\ndef f(conn, n):\n"
        "    run_query(conn, 'SELECT ' + n)\n",
    }
    hits, _, _ = analyze(caller)
    assert lines_of(hits) == [("db.py", 2, "CWE-89")]  # reported at the sink, via the caller


def test_sanitizers_and_constants_clear_taint() -> None:
    src = {
        "s.py": (
            "import os, shlex, subprocess\nTABLE = 'orders'\n\n"
            "def a(conn, x):\n    conn.execute('SELECT %d' % int(x))\n\n"
            "def b(conn):\n    conn.execute('SELECT * FROM ' + TABLE)\n\n"
            "def c(x):\n    subprocess.run('ls ' + shlex.quote(x), shell=True)\n\n"
            "def d(x):\n    open(os.path.join('/srv', os.path.basename(x)))\n\n"
            "def e(x):\n    x = os.path.basename(x)\n    open('/srv/' + x)\n"
        )
    }
    assert analyze(src)[0] == []


def test_non_db_receivers_are_not_sql_sinks() -> None:
    src = {
        "r.py": "class Runner:\n    def execute(self, job):\n        pass\n\n"
        "def go(runner: Runner, j):\n    runner.execute('job-' + j)\n"
    }
    assert analyze(src)[0] == []


def test_dynamic_dispatch_ambiguity_and_star_imports_are_not_followed() -> None:
    src = {
        "h.py": HELPER,
        "dyn.py": "from h import build\nB = {'x': build}\n\ndef f(conn, k, v):\n"
        "    conn.execute(B[k](v))\n",
        "star.py": "from h import *\n\ndef g(conn, v):\n    conn.execute(build(v))\n",
        "amb.py": HELPER
        + HELPER.replace('name + "\'"', 'name.strip() + "\'"')
        + "\ndef k(conn, v):\n    conn.execute(build(v))\n",
    }
    hits, stats, _ = analyze(src)
    assert [h for h in hits if h.path in ("dyn.py", "star.py", "amb.py")] == []
    assert stats.unresolved_calls >= 1


def test_self_dispatch_is_resolved_with_uncertainty() -> None:
    src = {
        "c.py": 'class R:\n    def q(self, n):\n        return f"SELECT {n}"\n\n'
        "    def run(self, conn, n):\n        conn.execute(self.q(n))\n"
    }
    (hit,) = analyze(src)[0]
    assert any("subclasses" in note for note in hit_to_raw(hit).uncertainty)


def test_depth_budget_bounds_call_chains() -> None:
    chain = HELPER.replace("def build", "def l3") + "".join(
        f"\ndef l{i}(name):\n    return l{i + 1}(name)\n" for i in (2, 1, 0)
    )
    src = {"d.py": chain + "\ndef run(conn, name):\n    conn.execute(l0(name))\n"}
    assert analyze(src, CallgraphBudget(max_depth=3))[0] == []
    hits, stats, _ = analyze(src, CallgraphBudget(max_depth=6))
    assert len(hits) == 1 and not stats.depth_limited


def test_argv_shell_pathlib_join_and_text_wrapper() -> None:
    src = {
        "w.py": (
            "import subprocess\nfrom pathlib import Path\nfrom sqlalchemy import text\n\n"
            "def a(x):\n    subprocess.run(['sh', '-c', 'echo ' + x])\n\n"
            "def b(x):\n    return (Path('/srv') / x).read_text()\n\n"
            "def c(session, x):\n    session.execute(text(f'SELECT {x}'))\n\n"
            "def d(conn, x):\n    conn.execute(' '.join(['SELECT', x]))\n"
        )
    }
    assert [(h.line, h.cwe) for h in analyze(src)[0]] == [
        (6, "CWE-78"),
        (9, "CWE-22"),
        (12, "CWE-89"),
        (15, "CWE-89"),
    ]


def test_output_is_deterministic_and_trace_is_bounded() -> None:
    src = {
        f"m{i}.py": HELPER + f"\ndef r{i}(conn, n):\n    conn.execute(build(n))\n" for i in range(5)
    }
    first = [hit_to_raw(h) for h in analyze(src)[0]]
    assert first == [hit_to_raw(h) for h in analyze(dict(reversed(src.items())))[0]]
    tight = analyze(src, CallgraphBudget(max_trace_steps=3))[0]
    assert all(len(h.taint.steps) <= 3 for h in tight)


def test_scanner_reports_syntax_errors_as_coverage_gaps(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text(HELPER)
    (tmp_path / "bad.py").write_text("def broken(:\n")
    run = CallgraphScanner().scan(tmp_path, ["ok.py", "bad.py", "notes.txt"])
    assert run.scanned == {"ok.py"} and run.tool_skipped == {"bad.py": "python_syntax_error"}
    assert run.in_scope == ["ok.py", "bad.py"] and run.stats["files"] == 1


def test_analyzed_code_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    (tmp_path / "evil.py").write_text(
        f"open({str(marker)!r}, 'w').write('x')\nimport os\nos.system('touch {marker}')\n"
    )
    CallgraphScanner().scan(tmp_path, ["evil.py"])
    assert not marker.exists()


def test_semgrep_takes_precedence_on_duplicates() -> None:
    def raw(tool: str, line: int, cwe: str = "CWE-89") -> RawFinding:
        return RawFinding(
            tool=tool,
            rule_id="r",
            path="a.py",
            start_line=line,
            end_line=line,
            message="m",
            severity="high",
            cwe=(cwe,),
        )

    semgrep = ScannerRun(
        tool="semgrep", version="x", rules=[], targets=1, findings=[raw("semgrep", 10)]
    )
    cg = ScannerRun(
        tool="tantular-callgraph",
        version="x",
        rules=[],
        targets=1,
        findings=[
            raw("tantular-callgraph", 10),
            raw("tantular-callgraph", 20),
            raw("tantular-callgraph", 10, "CWE-78"),
        ],
    )
    kept, suppressed = dedupe_across_scanners([semgrep, cg])
    assert [(r.tool, r.start_line, r.cwe) for r in kept] == [
        ("semgrep", 10, ("CWE-89",)),
        ("tantular-callgraph", 20, ("CWE-89",)),
        ("tantular-callgraph", 10, ("CWE-78",)),
    ]
    assert suppressed == {"tantular-callgraph": 1}

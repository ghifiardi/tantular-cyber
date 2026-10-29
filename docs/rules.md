# Bundled rules (local, offline Semgrep)

All rules live in `src/tantular_cyber/rules/`, run with `--metrics=off`, need no login and
are matched **syntactically within one function/module scope**. Every finding is
`unverified`: a rule match means "this shape is present", not "this is exploitable".
Detection quality is measured per CWE on the benchmark corpus (`docs/benchmark.md`); the
synthetic corpus says nothing about real-world recall.

| CWE | Rule id | Severity / confidence |
|---|---|---|
| 89 SQL injection | `tantular.python.sqli.dynamic-query-execute` | ERROR / MEDIUM |
| 78 OS command injection | `tantular.python.cmdi.shell-dynamic-command` | ERROR / MEDIUM |
| 22 Path traversal | `tantular.python.path.dynamic-file-path` | WARNING / LOW |
| 22 Path traversal | `tantular.python.path.tarfile-unfiltered-extract` | WARNING / MEDIUM |
| 79 XSS | `tantular.python.xss.unescaped-dynamic-html` | ERROR / MEDIUM |
| 79 XSS | `tantular.python.xss.jinja2-autoescape-disabled` | WARNING / MEDIUM |
| 79 XSS | `tantular.template.xss.autoescape-bypass` (HTML templates) | WARNING / LOW |
| 798 Hard-coded credentials | `tantular.python.secrets.hardcoded-credential` | WARNING / MEDIUM |

"Dynamic string" below means an f-string, `+` concatenation (including chains whose last
operand is a variable, `"a" + x + "." + y`, since POC-005), `%`-formatting or `.format()` —
inline in the call, or assigned to a local variable earlier in the same scope.

Since POC-005 a second scanner, `tantular-callgraph` (rule ids `tantular.callgraph.*`,
CWE-89/78/22), follows values through same-repository function calls and through
`text()`, `str.join`, `["sh", "-c", ...]` and `pathlib` composition — closing the five
Semgrep known gaps marked below as *(callgraph)*. It is conservative (unresolved calls never
produce findings) and reports a call path plus uncertainty; see architecture.md.

## CWE-89 — SQL injection

- **Sinks:** `execute`, `executemany`, `executescript` on any object (DB-API style).
- **Flagged:** dynamic query string (incl. `+=` accumulation).
- **Boundary:** placeholders with a parameters tuple/dict are not flagged.
- **Known gaps (Semgrep rule):** query built in another function, `sqlalchemy.text()`
  wrapper and `str.join` *(all three caught by callgraph)*; callee chosen at runtime or more
  than 3 calls deep (FN for both scanners); constant concatenation and `int()`-cast `%d` (FP); non-DB objects with an
  `.execute()` method (FP).

## CWE-78 — OS command injection

- **Sinks:** `subprocess.*(..., shell=True)`, `os.system`, `os.popen`,
  `asyncio.create_subprocess_shell`.
- **Flagged:** dynamic command string.
- **Sanitizer boundary:** `shlex.quote()` anywhere in the command expression suppresses the
  finding. Argument lists without `shell=True` are not flagged.
- **Known gaps:** a shell invoked via argv (`["sh", "-c", cmd]`) *(callgraph)*; commands built in
  other functions, `os.exec*`/`spawn*`, `pexpect`, `fabric` (FN); f-strings that only
  interpolate constants (FP).
- Informed by CVE-2024-29189 (program launch helper via `shell=True`).

## CWE-22 — Path traversal

- **Sinks:** `open`, `io.open`, `flask.send_file`; `tarfile` `extract`/`extractall`.
- **Flagged:** path built with `os.path.join(...)` containing a lower-case (non-constant)
  name, an f-string or concatenation; tar extraction without `filter=` (CVE-2007-4559).
- **Sanitizer boundary:** `os.path.basename()` or `secure_filename()` inside the path
  expression; `filter=` on extraction; `send_from_directory` is not a sink.
- **Known gaps:** validation in a separate statement (realpath + commonpath) is invisible
  (FP); `pathlib` `/` joins and `Path.read_*` *(callgraph)*; `zipfile`, `shutil` (not covered).
  Confidence is LOW: many joins use trusted values.

## CWE-79 — Cross-site scripting (scoped: Flask/Jinja2, Django, HTML templates)

- **Flask/Jinja2:** `flask.render_template_string(dynamic)` (also a template-injection
  risk), `markupsafe.Markup(dynamic)`, `jinja2.Environment()` without `autoescape=True` or
  `select_autoescape()`.
- **Django:** `mark_safe(dynamic)`, `HttpResponse(dynamic)`.
- **Templates** (`*.html`, `*.jinja*`, `*.j2`, matched as text): `{{ x|safe }}`,
  `{% autoescape off %}` / `{% autoescape false %}` (informed by CVE-2017-12794).
- **Boundaries:** template context variables, `Markup(...) % value` (escapes the value),
  `format_html()`, literal `mark_safe("...")`, `select_autoescape()` are not flagged.
- **Known gaps:** Flask views returning formatted HTML strings directly (FN); JavaScript,
  attribute and URL contexts, custom template tags, DOM XSS (not covered); `|safe` on
  sanitiser output (FP). **Not all XSS patterns are covered.**

## CWE-798 — Hard-coded credentials

- **Flagged:** a string literal of ≥ 8 characters assigned to a credential-like name
  (`password`, `secret`, `token`, `api_key`, `access_key`, `private_key`, …) or attribute,
  passed as `password=`/`secret=`/`token=`/`api_key=` keyword, assigned to
  `app.config["SECRET_KEY"]` (CVE-2022-25510), or an AWS access key id anywhere.
- **Boundaries:** environment reads and numeric values are not flagged.
- **Known gaps:** values inside dict/list literals, split/encoded values (FN); URLs or
  identifiers stored in credential-named variables (FP). Non-Python config files are not
  read (secret-like files are skipped entirely).
- **Redaction:** values are masked in evidence, context, prompts and reports; findings
  point at the line, never reproduce the value.

## Adding a rule

1. Write the rule with metadata (`cwe`, `confidence`, references) and a header comment
   listing sinks, boundaries and gaps; add it to `DEFAULT_RULESETS` in `config.py`.
2. Add corpus cases: at least one TP, TN, near miss and a documented limitation (known FP
   or FN) for the CWE, plus a `per_cwe` threshold (`require_cwe_category_coverage` enforces
   this). CVE-informed cases: independently written, reference + uncertainty recorded.
3. Run `make bench`; update `expect_detected` only for intended behavior changes.
4. Update this file and the limitations in README.

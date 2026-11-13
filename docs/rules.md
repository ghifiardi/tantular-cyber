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
| 79 XSS | `tantular.python.xss.jinja2-template-autoescape-disabled` | WARNING / LOW |
| 79 XSS | `tantular.template.xss.autoescape-bypass` (HTML templates) | WARNING / LOW |
| 79 XSS | `tantular.callgraph.xss-jinja2-autoescape-extension` (tantular-callgraph, file-level) | WARNING / LOW |
| 798 Hard-coded credentials | `tantular.python.secrets.hardcoded-credential` | WARNING / MEDIUM |

"Dynamic string" below means an f-string, `+` concatenation, `%`-formatting or `.format()` —
inline in the call, or assigned to a local variable earlier in the same scope. Concatenation
chains whose last operand is a variable (`"a" + x + "." + y`) are matched by the CWE-78/22/79
rules through `metavariable-pattern` (sub-expression search) and, since POC-005, by an
explicit chain pattern in the CWE-89 rule.

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
  `.execute()` method (FP). Vendor SDK query methods such as BigQuery `client.query(sql)` are
  not sinks (FN; see the real-world backlog below).

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
  `select_autoescape()`, and `jinja2.Template(source)` constructed directly without
  `autoescape=True` (a Template built this way ignores any Environment's setting and does not
  escape; confidence LOW because many such templates render plain text).
- **Jinja2 autoescape configuration (tantular-callgraph, file-level):** a
  `jinja2.Environment(..., autoescape=select_autoescape(...))` whose literal
  `enabled_extensions`, `disabled_extensions` and `default` leave an HTML-like template-name
  literal in the **same file** unescaped (e.g. the default `select_autoescape()` with
  `*.html.j2` names). Suffixes are matched case-insensitively as Jinja does (enabled first,
  then disabled, then `default`; `default=True` does not help a name whose suffix is
  disabled). Reported at the Environment construction, listing each name once with its
  lines; **no call path**: it is a configuration warning, not proven dataflow.
  Report safety: names are repository text, so a name that `security.redact()` would
  change is never listed; the finding states only how many were withheld (a finding whose
  names are all withheld keeps a generic warning). The message lists at most 5 names with
  5 lines each.
  Budgets: per file, at most 200 distinct names (by first occurrence) are compared against
  at most 50 Environment constructions (by position), and each distinct setting is
  evaluated once. Anything beyond a limit is not examined and is counted in the scanner
  statistics (`jinja_autoescape_names_not_compared`,
  `jinja_autoescape_environments_not_evaluated`, `jinja_autoescape_limited_files`;
  withheld names in `jinja_autoescape_names_redacted`). Like the callgraph's depth limit,
  these counts do not change the report status.
- **Django:** `mark_safe(dynamic)`, `HttpResponse(dynamic)`.
- **Templates** (`*.html`, `*.jinja*`, `*.j2`, matched as text): `{{ x|safe }}`,
  `{% autoescape off %}` / `{% autoescape false %}` (informed by CVE-2017-12794).
- **Boundaries:** template context variables, `Markup(...) % value` (escapes the value),
  `format_html()`, literal `mark_safe("...")`, `select_autoescape()`, and
  `jinja2.Template(..., autoescape=True | select_autoescape(...))` are not flagged.
- **Known gaps:** Flask views returning formatted HTML strings directly (FN); extension
  mismatches the file-level check cannot see — template names built at run time or defined
  in another file/config, non-literal `select_autoescape(...)` arguments (skipped and counted
  as `jinja_autoescape_unresolved_settings`), a pre-built `select_autoescape` object passed
  through a variable, Jinja2 subclasses such as `SandboxedEnvironment`, names or
  Environments beyond the per-file budgets, files where a Jinja name is rebound anywhere
  (the binding check is not scope-aware, so a same-named local in an unrelated function,
  or `try: import jinja2 / except ImportError: jinja2 = None`, suppresses the file), and a
  warning dropped as a duplicate of an overlapping Semgrep CWE-79 finding (FN);
  `select_autoescape(default_for_string=False)`, which leaves string templates unescaped but is
  treated as safe by both Jinja2 rules (FN); Streamlit
  `unsafe_allow_html=True`, JavaScript, attribute and URL contexts, custom template tags, DOM
  XSS (not covered); `|safe` on sanitiser output, `jinja2.Template` rendering plain text, an
  extension-mismatch warning for names that belong to a different Environment in the same
  file or are not template names at all (FP; corpus `xss-jinja-ext-two-envs-text`), and
  `autoescape=<variable>` on `Environment`/`Template` (flagged because only a literal `True` or
  `select_autoescape(...)` is recognised, even if the variable is true at run time) (FP).
  **Not all XSS patterns are covered.**

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

## Real-world evaluation backlog (2026-11-12)

Source: a reviewed three-way evaluation (Tantular / Antares / combined) of
`ghifiardi/ai-driven-soc` at commit `747245a5`. That repository is **development-used** from
2026-11-12 (`development_repositories` in `benchmarks/realworld/manifest.json`): its results
inform rule work and are never held-out measurements. Counts are reviewed candidates in one
repository, not rates. Fixtures for any item are independently written; no code or
credentials from the evaluated repository are copied.

**Detection gaps** (missed shapes; fixing them adds findings):

| # | Gap | Evidence in the evaluation | Status |
|---|---|---|---|
| D1 | `jinja2.Template(source)` without `autoescape=True` | missed a reviewed likely-real stored XSS (HTML report fallback template) | **covered** by `tantular.python.xss.jinja2-template-autoescape-disabled` |
| D2 | `select_autoescape([...])` vs `*.html.j2` template names | same issue's primary path; escaping silently off | **covered (file-level)** by `tantular.callgraph.xss-jinja2-autoescape-extension`; names outside the file remain FN |
| D3 | BigQuery-style `client.query(sql)` with f-string SQL | 49 files / 242 sites; 2 reviewed as plausible (alert data interpolated) | open |
| D4 | Streamlit `st.markdown(..., unsafe_allow_html=True)` with dynamic content | 41 files / 464 calls; 4 dashboards reviewed, none exploitable | open |
| D5 | Credentials in non-Python config (`.json`) | the most significant reviewed issue: a guessable tenant key committed in config (found only via test files) | open |

**False-positive reductions** (findings that were reviewed as non-actionable; fixing them
removes findings and must never hide real ones):

| # | Pattern | Evidence | Status |
|---|---|---|---|
| F1 | `open()` paths built only from literals plus a timestamp or loop counter (CWE-22) | 13 of 20 CWE-22 candidates were false positives | open |
| F2 | Placeholder or dummy credential values (`your-api-key-here`, `test-secret`) (CWE-798) | 5 of 8 CWE-798 candidates not exploitable | open |
| F3 | Shell commands whose interpolated values are constants (CWE-78) | 5 of 5 CWE-78 candidates not exploitable | open |

Rules for this backlog: findings in test files are **never** suppressed for being in tests
(the most significant issue above was only visible through test files); AI annotations may
add context but never remove or change scanner findings; each change adds TP, TN, near-miss
and known-limitation corpus cases and reports before/after counts on the benchmark.

**Coverage limits that apply to every result above:** Python files and HTML templates only
(`.json`, `.yaml`, `.sql`, `.sh`, `.js` are not analysed); files over 1 MB are skipped and
listed; files that do not parse are scanned by Semgrep but not by tantular-callgraph (the
report is then `degraded`); rules are same-scope syntactic matches; zero findings ≠ secure.

## Adding a rule

1. Write the rule with metadata (`cwe`, `confidence`, references) and a header comment
   listing sinks, boundaries and gaps; add it to `DEFAULT_RULESETS` in `config.py`.
2. Add corpus cases: at least one TP, TN, near miss and a documented limitation (known FP
   or FN) for the CWE, plus a `per_cwe` threshold (`require_cwe_category_coverage` enforces
   this). CVE-informed cases: independently written, reference + uncertainty recorded.
3. Run `make bench`; update `expect_detected` only for intended behavior changes.
4. Update this file and the limitations in README.

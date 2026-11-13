"""File-level Jinja2 autoescape configuration check (reported by tantular-callgraph).

Finds a ``jinja2.Environment(..., autoescape=select_autoescape(...))`` whose literal settings
leave HTML-like template names that appear **in the same file** unescaped — e.g. the
default ``select_autoescape(["html", "xml"])`` with templates named ``*.html.j2``.

This is a **configuration warning, not dataflow**: names are matched lexically within one
file and are not proven to be loaded by that Environment, nor are rendered values proven
to be attacker-controlled. No call path is produced.

Semantics mirror ``jinja2.select_autoescape`` (Jinja 3.1): every extension is normalised to
``"." + ext.lstrip(".").lower()``, the template name is lower-cased, an enabled suffix
escapes, otherwise a disabled suffix does not escape, otherwise ``default`` applies.

Conservative by construction:

* Only real Jinja2 bindings are recognised. A call must resolve to exactly
  ``jinja2.Environment`` / ``jinja2.environment.Environment`` or
  ``jinja2.select_autoescape`` / ``jinja2.utils.select_autoescape`` through
  ``import jinja2[.environment|.utils] [as x]``, ``from jinja2[.environment|.utils] import
  Environment/select_autoescape [as x]`` or ``from jinja2 import *``; any other attribute
  chain (e.g. ``jinja2.unrelated.Environment``) is ignored. A name that is also bound
  anywhere else in the file (assignment, definition, parameter, ``except``/``match``
  capture, ``del``, another import) is ambiguous and ignored. This is not scope-aware: a
  same-named local in an unrelated function also suppresses the file (FN only), and
  ``from jinja2 import *`` is ignored when another star import could rebind the names.
* ``enabled_extensions``, ``disabled_extensions`` and ``default`` must be literal
  (lists/tuples/sets of strings, booleans); ``default_for_string`` only affects string
  templates and is not examined. Anything else — variables, ``*args``/``**kwargs`` — is
  **unresolved**: skipped and counted, never reported. An ``autoescape=`` value that is not a direct
  ``select_autoescape(...)`` call (e.g. a pre-built object in a variable) is not examined
  here; the Semgrep Environment rule reports variable ``autoescape`` values.
* A candidate template name is a string literal made only of ``[A-Za-z0-9_./-]`` whose
  lower-cased suffix chain contains ``.html``, ``.htm``, ``.xhtml`` or ``.xml``.

Report safety and budgets:

* Template names are repository text. A name that :func:`security.redact` would change
  (secret-shaped) is never stored in a hit or message; only the count of such names is
  reported. A hit whose names are all omitted keeps a generic warning.
* Work per file is bounded: at most ``MAX_NAMES_PER_FILE`` distinct names (by first
  occurrence) are compared, against at most ``MAX_ENVIRONMENTS_PER_FILE`` Environment
  constructions (by position), and each distinct setting is evaluated once. The message
  lists at most ``MAX_LISTED_NAMES`` names with ``MAX_LISTED_LINES`` lines each. Names or
  Environments beyond a limit are not examined (possible FN) and are counted in the
  scanner statistics.

Parsing only (``ast``); nothing in the analyzed repository is executed or imported.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from .security import redact

RULE_ID = "tantular.callgraph.xss-jinja2-autoescape-extension"
SEVERITY = "medium"
CONFIDENCE = "low"
MAX_LISTED_NAMES = 5
MAX_LISTED_LINES = 5
MAX_NAMES_PER_FILE = 200
MAX_ENVIRONMENTS_PER_FILE = 50

_JINJA_MODULES = {"jinja2", "jinja2.environment", "jinja2.utils"}
_ENV_ATTR = "Environment"
_SELECT_ATTR = "select_autoescape"
# The only qualified names recognised (Jinja 3.1 public locations).
_ENV_QUALNAMES = {"jinja2.Environment", "jinja2.environment.Environment"}
_SELECT_QUALNAMES = {"jinja2.select_autoescape", "jinja2.utils.select_autoescape"}
_DEFAULT_ENABLED = ("html", "htm", "xml")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,150}")
_HTML_LIKE = re.compile(r"\.(?:html?|xhtml|xml)(?:\.[a-z0-9_-]+)*$")
_SELECT_PARAMS = ("enabled_extensions", "disabled_extensions", "default_for_string", "default")

UNCERTAINTY = (
    "file-level configuration check: template names are matched lexically within this file; "
    "it is not proven that this Environment loads them (another Environment or a non-template "
    "use of the same string may)",
    "no dataflow: whether rendered values are attacker-controlled is not established",
)


@dataclass(frozen=True)
class AutoescapeSetting:
    """Literal ``select_autoescape`` arguments, normalised as Jinja does."""

    enabled: tuple[str, ...] = tuple(f".{e}" for e in _DEFAULT_ENABLED)
    disabled: tuple[str, ...] = ()
    default: bool = False

    def escapes(self, template_name: str) -> bool:
        name = template_name.lower()
        if name.endswith(self.enabled):
            return True
        if name.endswith(self.disabled):
            return False
        return self.default


def _normalise(exts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f".{x.lstrip('.').lower()}" for x in exts)


def _describe(name: str, lines: list[int]) -> str:
    shown = ", ".join(map(str, lines[:MAX_LISTED_LINES]))
    more = len(lines) - MAX_LISTED_LINES
    tail = f", and {more} more" if more > 0 else ""
    return f"'{name}' (line{'s' if len(lines) > 1 else ''} {shown}{tail})"


@dataclass
class ConfigHit:
    path: str
    line: int
    end_line: int
    # (template name, sorted lines where it appears), deduplicated; never secret-shaped names
    names: list[tuple[str, list[int]]]
    omitted: int = 0  # unescaped names withheld because redaction would change them

    @property
    def message(self) -> str:
        parts = [_describe(n, ls) for n, ls in self.names[:MAX_LISTED_NAMES]]
        more = len(self.names) - len(parts)
        if more > 0:
            parts.append(f"and {more} more")
        if self.omitted:
            parts.append(f"{self.omitted} name(s) withheld because they resemble secrets")
        found = (
            f"template names found in this file unescaped: {', '.join(parts)}"
            if self.names
            else f"template names found in this file unescaped ({', '.join(parts)})"
        )
        return (
            "This Jinja2 Environment uses select_autoescape() settings that leave HTML-like "
            f"{found}. Jinja2 "
            "matches name suffixes case-insensitively (enabled extensions first, then disabled "
            "ones, then the default), so a name like '*.html.j2' does not end in '.html'. If "
            "such a template renders user-controlled values this may allow cross-site "
            "scripting. Add the template suffix to enabled_extensions, or name templates so "
            "they end in an enabled extension."
        )


@dataclass
class FileResult:
    hits: list[ConfigHit] = field(default_factory=list)
    unresolved: int = 0
    names_redacted: int = 0  # distinct candidate names withheld from messages
    names_not_compared: int = 0  # distinct candidate names beyond MAX_NAMES_PER_FILE
    environments_not_evaluated: int = 0  # constructions beyond MAX_ENVIRONMENTS_PER_FILE
    comparisons: int = 0  # setting-vs-name evaluations performed (bounded)

    @property
    def limited(self) -> bool:
        return bool(self.names_not_compared or self.environments_not_evaluated)


_NAMED_BINDERS: tuple[type, ...] = tuple(
    t
    for t in (
        ast.ExceptHandler,
        ast.MatchAs,
        ast.MatchStar,
        getattr(ast, "TypeVar", None),
        getattr(ast, "ParamSpec", None),
        getattr(ast, "TypeVarTuple", None),
    )
    if t is not None
)


class _Bindings:
    """Which local names refer to Jinja2's module, Environment and select_autoescape."""

    def __init__(self, tree: ast.Module):
        self.modules: dict[str, str] = {}  # local name -> jinja module it is bound to
        self.env: set[str] = set()
        self.select: set[str] = set()
        star = False
        other_star = False
        other: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname is None:
                        # ``import jinja2.utils`` binds ``jinja2`` (the top-level package).
                        bound, target = a.name.split(".")[0], "jinja2"
                        jinja = a.name == "jinja2" or a.name.startswith("jinja2.")
                    else:
                        bound, target = a.asname, a.name
                        jinja = a.name in _JINJA_MODULES
                    if jinja and self.modules.get(bound, target) == target:
                        self.modules[bound] = target
                    else:
                        other.add(bound)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.level == 0 else None
                for a in node.names:
                    if a.name == "*":
                        if module == "jinja2":
                            star = True
                        else:
                            other_star = True
                        continue
                    bound = a.asname or a.name
                    if f"{module}.{a.name}" in _ENV_QUALNAMES:
                        self.env.add(bound)
                    elif f"{module}.{a.name}" in _SELECT_QUALNAMES:
                        self.select.add(bound)
                    else:
                        other.add(bound)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                other.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                other.add(node.id)
            elif isinstance(node, ast.arg):
                other.add(node.arg)
            elif isinstance(node, _NAMED_BINDERS) and isinstance(node.name, str):
                # except ... as X, case X / case [*X], PEP 695 type parameters
                other.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                other.add(node.rest)
        if star and not other_star:
            self.env.add(_ENV_ATTR)
            self.select.add(_SELECT_ATTR)
        # Ambiguous: also defined, assigned or imported from elsewhere in this file.
        self.modules = {k: v for k, v in self.modules.items() if k not in other}
        self.env -= other
        self.select -= other

    def _qualname(self, func: ast.expr) -> str | None:
        """``mod.sub.attr`` with the root resolved to its jinja module, else ``None``."""
        parts: list[str] = []
        node = func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not (isinstance(node, ast.Name) and node.id in self.modules and parts):
            return None
        return ".".join([self.modules[node.id], *reversed(parts)])

    def _is(self, func: ast.expr, local: set[str], qualnames: set[str]) -> bool:
        if isinstance(func, ast.Name):
            return func.id in local
        return self._qualname(func) in qualnames

    def is_environment(self, call: ast.Call) -> bool:
        return self._is(call.func, self.env, _ENV_QUALNAMES)

    def is_select(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Call) and self._is(node.func, self.select, _SELECT_QUALNAMES)


def _literal_strings(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts
    ):
        return tuple(e.value for e in node.elts)  # type: ignore[attr-defined]
    return None


def resolve_setting(call: ast.Call) -> AutoescapeSetting | None:
    """Literal ``select_autoescape(...)`` arguments, or ``None`` if any is unresolved."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        k.arg is None for k in call.keywords
    ):
        return None
    if len(call.args) > len(_SELECT_PARAMS):
        return None
    values: dict[str, ast.expr] = dict(zip(_SELECT_PARAMS, call.args, strict=False))
    for k in call.keywords:
        if k.arg not in _SELECT_PARAMS or k.arg in values:
            return None
        values[k.arg] = k.value
    enabled: tuple[str, ...] = _DEFAULT_ENABLED
    disabled: tuple[str, ...] = ()
    default = False
    if "enabled_extensions" in values:
        got = _literal_strings(values["enabled_extensions"])
        if got is None:
            return None
        enabled = got
    if "disabled_extensions" in values:
        got = _literal_strings(values["disabled_extensions"])
        if got is None:
            return None
        disabled = got
    if "default" in values:
        node = values["default"]
        if not (isinstance(node, ast.Constant) and isinstance(node.value, bool)):
            return None
        default = node.value
    # default_for_string only applies to string templates (name None); not relevant here.
    return AutoescapeSetting(_normalise(enabled), _normalise(disabled), default)


@dataclass(frozen=True)
class _Candidate:
    name: str
    lines: list[int]
    secret: bool  # security.redact() would change it: never shown


def _template_names(tree: ast.Module) -> dict[str, list[int]]:
    """Every HTML-like name literal with its sorted, distinct lines."""
    names: dict[str, set[int]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _NAME.fullmatch(node.value)
            and _HTML_LIKE.search(node.value.lower())
        ):
            names.setdefault(node.value, set()).add(node.lineno)
    return {n: sorted(ls) for n, ls in names.items()}


def _candidates(tree: ast.Module, result: FileResult) -> list[_Candidate]:
    """Names in first-occurrence order, capped and classified for redaction before use."""
    ordered = sorted(_template_names(tree).items(), key=lambda item: (item[1][0], item[0]))
    result.names_not_compared = max(0, len(ordered) - MAX_NAMES_PER_FILE)
    out = [_Candidate(n, ls, redact(n)[0] != n) for n, ls in ordered[:MAX_NAMES_PER_FILE]]
    result.names_redacted = sum(c.secret for c in out)
    return out


def _constructions(tree: ast.Module, bindings: _Bindings) -> list[tuple[ast.Call, ast.Call]]:
    """``(Environment call, select_autoescape call)`` pairs in source order."""
    found: list[tuple[ast.Call, ast.Call]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and bindings.is_environment(node)):
            continue
        kw = next((k for k in node.keywords if k.arg == "autoescape"), None)
        if kw is None or not bindings.is_select(kw.value):
            continue  # missing/True/False/variable: covered (or not) by the Semgrep rules
        found.append((node, kw.value))  # type: ignore[arg-type]
    found.sort(key=lambda pair: (pair[0].lineno, pair[0].col_offset))
    return found


def check_module(rel: str, tree: ast.Module) -> FileResult:
    result = FileResult()
    bindings = _Bindings(tree)
    if not (bindings.env or bindings.modules):
        return result
    envs = _constructions(tree, bindings)
    if not envs:
        return result
    result.environments_not_evaluated = max(0, len(envs) - MAX_ENVIRONMENTS_PER_FILE)
    candidates = _candidates(tree, result)
    # Each distinct setting is compared with the names once: work <= envs_cap * names_cap.
    verdicts: dict[AutoescapeSetting, list[_Candidate]] = {}
    for env, select in envs[:MAX_ENVIRONMENTS_PER_FILE]:
        setting = resolve_setting(select)
        if setting is None:
            result.unresolved += 1
            continue
        if setting not in verdicts:
            verdicts[setting] = [c for c in candidates if not setting.escapes(c.name)]
            result.comparisons += len(candidates)
        unescaped = verdicts[setting]
        if unescaped:
            result.hits.append(
                ConfigHit(
                    path=rel,
                    line=env.lineno,
                    end_line=env.end_lineno or env.lineno,
                    names=[(c.name, c.lines) for c in unescaped if not c.secret],
                    omitted=sum(c.secret for c in unescaped),
                )
            )
    result.hits.sort(key=lambda h: (h.line, h.end_line))
    return result

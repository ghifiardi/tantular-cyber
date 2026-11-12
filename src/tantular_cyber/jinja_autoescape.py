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

* Only real Jinja2 bindings are recognised: ``import jinja2 [as x]``,
  ``from jinja2 import Environment/select_autoescape [as x]`` (also from
  ``jinja2.environment`` / ``jinja2.utils``) and ``from jinja2 import *``. A name that is
  also defined or imported from elsewhere in the file is ambiguous and ignored.
* ``select_autoescape(...)`` arguments must be literal (lists/tuples/sets of strings,
  booleans). Anything else — variables, ``*args``/``**kwargs`` — is **unresolved**: skipped
  and counted, never reported. An ``autoescape=`` value that is not a direct
  ``select_autoescape(...)`` call (e.g. a pre-built object in a variable) is not examined
  here; the Semgrep Environment rule reports variable ``autoescape`` values.
* A candidate template name is a string literal made only of ``[A-Za-z0-9_./-]`` whose
  lower-cased suffix chain contains ``.html``, ``.htm``, ``.xhtml`` or ``.xml``.

Parsing only (``ast``); nothing in the analyzed repository is executed or imported.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

RULE_ID = "tantular.callgraph.xss-jinja2-autoescape-extension"
SEVERITY = "medium"
CONFIDENCE = "low"
MAX_LISTED_NAMES = 5

_JINJA_MODULES = {"jinja2", "jinja2.environment", "jinja2.utils"}
_ENV_ATTR = "Environment"
_SELECT_ATTR = "select_autoescape"
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


@dataclass
class ConfigHit:
    path: str
    line: int
    end_line: int
    names: list[tuple[str, list[int]]]  # (template name, lines where it appears), deduplicated

    @property
    def message(self) -> str:
        listed = [
            f"'{n}' (line{'s' if len(ls) > 1 else ''} {', '.join(map(str, ls))})"
            for n, ls in self.names[:MAX_LISTED_NAMES]
        ]
        more = len(self.names) - MAX_LISTED_NAMES
        tail = f", and {more} more" if more > 0 else ""
        return (
            "This Jinja2 Environment uses select_autoescape() settings that leave HTML-like "
            f"template names found in this file unescaped: {', '.join(listed)}{tail}. Jinja2 "
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


class _Bindings:
    """Which local names refer to Jinja2's module, Environment and select_autoescape."""

    def __init__(self, tree: ast.Module):
        self.modules: set[str] = set()
        self.env: set[str] = set()
        self.select: set[str] = set()
        star = False
        other: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    bound = a.asname or a.name.split(".")[0]
                    if a.name == "jinja2" or (a.asname is None and a.name.startswith("jinja2.")):
                        self.modules.add(bound)
                    else:
                        other.add(bound)
            elif isinstance(node, ast.ImportFrom):
                jinja = node.level == 0 and node.module in _JINJA_MODULES
                for a in node.names:
                    if a.name == "*":
                        star = star or jinja
                        continue
                    bound = a.asname or a.name
                    if jinja and a.name == _ENV_ATTR:
                        self.env.add(bound)
                    elif jinja and a.name == _SELECT_ATTR:
                        self.select.add(bound)
                    else:
                        other.add(bound)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                other.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                other.add(node.id)
            elif isinstance(node, ast.arg):
                other.add(node.arg)
        if star:
            self.env.add(_ENV_ATTR)
            self.select.add(_SELECT_ATTR)
        # Ambiguous: also defined, assigned or imported from elsewhere in this file.
        self.modules -= other
        self.env -= other
        self.select -= other

    def _is(self, func: ast.expr, local: set[str], attr: str) -> bool:
        if isinstance(func, ast.Name):
            return func.id in local
        if isinstance(func, ast.Attribute) and func.attr == attr:
            base = func.value
            while isinstance(base, ast.Attribute):  # jinja2.environment.Environment
                base = base.value
            return isinstance(base, ast.Name) and base.id in self.modules
        return False

    def is_environment(self, call: ast.Call) -> bool:
        return self._is(call.func, self.env, _ENV_ATTR)

    def is_select(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Call) and self._is(node.func, self.select, _SELECT_ATTR)


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


def _template_names(tree: ast.Module) -> dict[str, list[int]]:
    names: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _NAME.fullmatch(node.value)
            and _HTML_LIKE.search(node.value.lower())
        ):
            lines = names.setdefault(node.value, [])
            if node.lineno not in lines:
                lines.append(node.lineno)
    return {n: sorted(ls) for n, ls in names.items()}


def check_module(rel: str, tree: ast.Module) -> FileResult:
    result = FileResult()
    bindings = _Bindings(tree)
    if not (bindings.env or bindings.modules):
        return result
    names = _template_names(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and bindings.is_environment(node)):
            continue
        kw = next((k for k in node.keywords if k.arg == "autoescape"), None)
        if kw is None or not bindings.is_select(kw.value):
            continue  # missing/True/False/variable: covered (or not) by the Semgrep rules
        setting = resolve_setting(kw.value)  # type: ignore[arg-type]
        if setting is None:
            result.unresolved += 1
            continue
        unescaped = sorted(
            ((n, ls) for n, ls in names.items() if not setting.escapes(n)),
            key=lambda item: (item[1][0], item[0]),
        )
        if unescaped:
            result.hits.append(
                ConfigHit(
                    path=rel,
                    line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    names=unescaped,
                )
            )
    result.hits.sort(key=lambda h: (h.line, h.end_line))
    return result

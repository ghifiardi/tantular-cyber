"""C3: local MATH / MATH-500 JSONL files to harness ``Item`` rows. Standard library only.

No network: the loader reads a file an operator has already fetched (in an approved step) into
research storage. Two source shapes are supported:

* **MATH-500** (``HuggingFaceH4/MATH-500`` ``test.jsonl``): fields ``unique_id``, ``problem``,
  ``answer`` (others ignored). IDs are the source's ``unique_id``; the split is ``heldout``.
* **MATH training split** (for the development smoke run): fields ``problem`` and ``solution``
  (and ``unique_id`` if the source has one). The answer is the last ``\\boxed{}`` of the
  solution. The ID is the source's ``unique_id`` when present, otherwise ``math-train-`` plus
  16 hex digits of the SHA-256 of the normalised problem text (independent of file order). The
  split is ``development``.

Every load validates the whole file (before any ``limit``) and returns a
:class:`DatasetReport`: the source file's SHA-256, row and item counts, the canonical hash of
the ID set, the harness ``dataset_sha256`` of the items, and the harness checks. A registered
expected count, file hash or ID-set hash that differs is an error, not a warning.

**Leakage** (:func:`check_disjoint`, :func:`combine`): two item sets overlap when any problem's
*normalised* text is equal, whatever the answers say (answer formatting differs between
sources). Normalisation (Unicode NFKC, case folding, whitespace collapsed) catches copies that
differ only in such details; it **cannot prove** that two sets are disjoint, because
paraphrased, reformatted or translated problems are not detected.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wsp_eval.dataset import Item, check_items, dataset_sha256
from wsp_eval.hashing import sha256_json

from .grader import last_boxed
from .provenance import sha256_path

SHAPES = {"math500": "heldout", "math_train": "development"}
NORMALIZATION = "NFKC, casefold, whitespace runs collapsed to one space, stripped"
DISJOINTNESS_LIMIT = (
    "normalised-text matching cannot prove disjointness: paraphrased, reformatted or "
    "translated problems are not detected"
)


class DatasetLoadError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetReport:
    source_file: str
    source_sha256: str
    shape: str
    split: str
    id_source: str
    n_source_rows: int
    limit: int | None
    n_items: int
    ids_sha256: str
    items_sha256: str
    checks: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def ids_sha256(items: Sequence[Item]) -> str:
    return sha256_json(sorted(i.id for i in items))


def _rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    # Split on "\n" only, like the harness: U+2028 and friends may occur inside JSON strings.
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"{path.name} line {n}: not valid JSON") from exc
        if not isinstance(row, dict):
            raise DatasetLoadError(f"{path.name} line {n}: not an object")
        rows.append((n, row))
    return rows


def _text(row: dict[str, Any], key: str, n: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetLoadError(f"line {n}: {key!r} must be a non-empty string")
    return value


def load(
    path: Path,
    *,
    shape: str,
    split: str,
    expected_count: int | None = None,
    expected_source_sha256: str | None = None,
    expected_ids_sha256: str | None = None,
    limit: int | None = None,
) -> tuple[list[Item], DatasetReport]:
    """Load and validate the whole of ``path``; then ``limit`` keeps the first ``limit`` items in
    file order (the smoke run's selection, recorded in the report). ``expected_count`` and
    ``expected_ids_sha256`` refer to the whole file."""
    if shape not in SHAPES:
        raise DatasetLoadError(f"shape must be one of {sorted(SHAPES)}")
    if split != SHAPES[shape]:
        raise DatasetLoadError(
            f"{shape} items are {SHAPES[shape]!r} in this protocol, not {split!r}"
        )
    if limit is not None and (type(limit) is not int or limit < 1):
        raise DatasetLoadError("limit must be a positive integer")
    path = Path(path)
    source = sha256_path(path)
    if expected_source_sha256 is not None and source != expected_source_sha256:
        raise DatasetLoadError(f"{path.name} SHA-256 differs from the registered value")
    rows = _rows(path)
    items: list[Item] = []
    id_sources: set[str] = set()
    for n, row in rows:
        problem = _text(row, "problem", n)
        if shape == "math500":
            item_id, answer = _text(row, "unique_id", n), _text(row, "answer", n)
            id_sources.add("unique_id")
        else:
            answer = last_boxed(_text(row, "solution", n))
            if answer is None or not answer.strip():
                raise DatasetLoadError(f"line {n}: solution has no complete \\boxed{{}} answer")
            if "unique_id" in row:
                item_id = _text(row, "unique_id", n)
                id_sources.add("unique_id")
            else:
                digest = hashlib.sha256(normalize_problem(problem).encode("utf-8")).hexdigest()
                item_id = "math-train-" + digest[:16]
                id_sources.add("normalized_problem_sha256")
        items.append(Item(id=item_id, split=split, prompt=problem, answer=answer))
    if len(id_sources) > 1:
        raise DatasetLoadError("some rows have unique_id and some do not; IDs would be mixed")
    by_id: dict[str, list[int]] = defaultdict(list)
    for (n, _), item in zip(rows, items, strict=True):
        by_id[item.id].append(n)
    clashes = {i: lines for i, lines in by_id.items() if len(lines) > 1}
    if clashes:
        first = sorted(clashes.items())[:5]
        hint = (
            " (content-derived IDs: the same problem text, after normalisation, occurs more "
            "than once; remove the duplicates as a recorded operator step)"
            if "normalized_problem_sha256" in id_sources
            else ""
        )
        raise DatasetLoadError(f"duplicate item IDs at lines {first}{hint}")
    checks = check_items(items)
    if not checks["ok"]:
        raise DatasetLoadError(f"dataset check failed: {checks['errors']}")
    checks["within_file_duplicate_problems"] = _duplicate_problems(items)
    if expected_count is not None and len(items) != expected_count:
        raise DatasetLoadError(f"expected {expected_count} items, found {len(items)}")
    ids = ids_sha256(items)
    if expected_ids_sha256 is not None and ids != expected_ids_sha256:
        raise DatasetLoadError("item ID set differs from the registered ids_sha256")
    n_source = len(items)
    if limit is not None:
        items = items[:limit]
    report = DatasetReport(
        source_file=path.name,
        source_sha256=source,
        shape=shape,
        split=split,
        id_source=next(iter(id_sources), "none"),
        n_source_rows=n_source,
        limit=limit,
        n_items=len(items),
        ids_sha256=ids_sha256(items),
        items_sha256=dataset_sha256(items),
        checks=checks,
    )
    return items, report


def _duplicate_problems(items: Sequence[Item]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for i in items:
        groups[normalize_problem(i.prompt)].append(i.id)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def check_disjoint(items: Sequence[Item], reference: Sequence[Item]) -> dict[str, Any]:
    """Refuse any item whose normalised problem text, or ID, also occurs in ``reference``
    (answers are ignored). Passing does not prove disjointness (see the module docstring)."""
    ref_problems: dict[str, str] = {}
    for r in reference:
        ref_problems.setdefault(normalize_problem(r.prompt), r.id)
    ref_ids = {r.id for r in reference}
    overlap = sorted(
        (i.id, ref_problems[normalize_problem(i.prompt)])
        for i in items
        if normalize_problem(i.prompt) in ref_problems
    )
    shared_ids = sorted({i.id for i in items} & ref_ids)
    if overlap or shared_ids:
        raise DatasetLoadError(
            f"leakage: {len(overlap)} problems match the reference after normalisation "
            f"{overlap[:5]}; shared IDs {shared_ids[:5]}"
        )
    return {
        "ok": True,
        "method": f"normalised problem text ({NORMALIZATION}) and IDs; answers ignored",
        "n_items": len(items),
        "n_reference": len(reference),
        "limit": DISJOINTNESS_LIMIT,
    }


def combine(*groups: Sequence[Item]) -> list[Item]:
    """Concatenate item groups and re-run the checks across them: the harness checks
    (duplicate IDs, identical content across splits) and :func:`check_disjoint` between the
    items of different splits."""
    items = [i for g in groups for i in g]
    checks = check_items(items)
    if not checks["ok"]:
        raise DatasetLoadError(f"combined dataset check failed: {checks['errors']}")
    by_split: dict[str, list[Item]] = defaultdict(list)
    for i in items:
        by_split[i.split].append(i)
    splits = sorted(by_split)
    for a, s in enumerate(splits):
        for t in splits[a + 1 :]:
            check_disjoint(by_split[s], by_split[t])
    return items


def write_items(items: Sequence[Item], out: Path) -> None:
    """Write items in the harness JSONL format (checked by ``wsp.py dataset``); never overwrites."""
    with Path(out).open("x", encoding="utf-8") as fh:
        for i in items:
            row = {"id": i.id, "split": i.split, "prompt": i.prompt, "answer": i.answer}
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

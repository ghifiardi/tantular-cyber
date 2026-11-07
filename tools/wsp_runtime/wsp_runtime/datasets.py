"""C3: local MATH / MATH-500 JSONL files to harness ``Item`` rows. Standard library only.

No network: the loader reads a file an operator has already fetched (in an approved step) into
research storage. Two source shapes are supported:

* **MATH-500** (``HuggingFaceH4/MATH-500`` ``test.jsonl``): fields ``unique_id``, ``problem``,
  ``answer`` (others ignored). IDs are ``unique_id``; the expected split is ``heldout``.
* **MATH training split** (for the development smoke run): fields ``problem`` and ``solution``;
  the answer is the last ``\\boxed{}`` of the solution, and the ID is ``math-train-`` plus the
  first 16 hex digits of the SHA-256 of the problem text (stable across file orderings). The
  expected split is ``development``.

Every load returns a :class:`DatasetReport`: the source file's SHA-256, the item count, the
canonical hash of the ID set, the harness ``dataset_sha256`` of the items, and the result of
the harness's duplicate and cross-split checks. A registered expected count, file hash or ID-set
hash that differs is an error, not a warning.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wsp_eval.dataset import Item, check_items, dataset_sha256
from wsp_eval.hashing import sha256_json

from .grader import last_boxed
from .provenance import sha256_path

SHAPES = ("math500", "math_train")


class DatasetLoadError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetReport:
    source_file: str
    source_sha256: str
    shape: str
    split: str
    n_items: int
    ids_sha256: str
    items_sha256: str
    checks: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ids_sha256(items: Sequence[Item]) -> str:
    return sha256_json(sorted(i.id for i in items))


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"{path.name} line {n}: not valid JSON") from exc
        if not isinstance(row, dict):
            raise DatasetLoadError(f"{path.name} line {n}: not an object")
        rows.append(row)
    return rows


def _text(row: dict[str, Any], key: str, n: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetLoadError(f"row {n}: {key!r} must be a non-empty string")
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
    """Load ``path`` as harness items. ``limit`` keeps the first ``limit`` rows in file order
    (the smoke run's selection, recorded in the report)."""
    if shape not in SHAPES:
        raise DatasetLoadError(f"shape must be one of {SHAPES}")
    expected_split = "heldout" if shape == "math500" else "development"
    if split != expected_split:
        raise DatasetLoadError(
            f"{shape} items are {expected_split!r} in this protocol, not {split!r}"
        )
    path = Path(path)
    source = sha256_path(path)
    if expected_source_sha256 is not None and source != expected_source_sha256:
        raise DatasetLoadError(f"{path.name} SHA-256 differs from the registered value")
    rows = _rows(path)
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise DatasetLoadError("limit must be a positive integer")
        rows = rows[:limit]
    items: list[Item] = []
    for n, row in enumerate(rows, 1):
        problem = _text(row, "problem", n)
        if shape == "math500":
            item_id = _text(row, "unique_id", n)
            answer = _text(row, "answer", n)
        else:
            answer = last_boxed(_text(row, "solution", n))
            if answer is None or not answer.strip():
                raise DatasetLoadError(f"row {n}: solution has no \\boxed{{}} answer")
            item_id = "math-train-" + hashlib.sha256(problem.encode("utf-8")).hexdigest()[:16]
        items.append(Item(id=item_id, split=split, prompt=problem, answer=answer))
    checks = check_items(items)
    if not checks["ok"]:
        raise DatasetLoadError(f"dataset check failed: {checks['errors']}")
    if expected_count is not None and len(items) != expected_count:
        raise DatasetLoadError(f"expected {expected_count} items, found {len(items)}")
    ids = ids_sha256(items)
    if expected_ids_sha256 is not None and ids != expected_ids_sha256:
        raise DatasetLoadError("item ID set differs from the registered ids_sha256")
    report = DatasetReport(
        source_file=path.name,
        source_sha256=source,
        shape=shape,
        split=split,
        n_items=len(items),
        ids_sha256=ids,
        items_sha256=dataset_sha256(items),
        checks=checks,
    )
    return items, report


def combine(*groups: Sequence[Item]) -> list[Item]:
    """Concatenate item groups and re-run the harness checks across them (duplicate IDs and
    development/held-out content leakage are errors)."""
    items = [i for g in groups for i in g]
    checks = check_items(items)
    if not checks["ok"]:
        raise DatasetLoadError(f"combined dataset check failed: {checks['errors']}")
    return items


def write_items(items: Sequence[Item], out: Path) -> None:
    """Write items in the harness JSONL format (checked by ``wsp.py dataset``); never overwrites."""
    with Path(out).open("x", encoding="utf-8") as fh:
        for i in items:
            row = {"id": i.id, "split": i.split, "prompt": i.prompt, "answer": i.answer}
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

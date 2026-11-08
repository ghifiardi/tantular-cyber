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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from wsp_eval.dataset import Item, check_items, dataset_sha256
from wsp_eval.hashing import sha256_bytes, sha256_json

from .grader import last_boxed

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
    source_ids_sha256: str
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


def _rows(name: str, text: str) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    # Split on "\n" only, like the harness: U+2028 and friends may occur inside JSON strings.
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"{name} line {n}: not valid JSON") from exc
        if not isinstance(row, dict):
            raise DatasetLoadError(f"{name} line {n}: not an object")
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
    """Load and validate the whole of ``path`` from one byte snapshot (the hash and the parsed
    rows come from the same bytes); then ``limit`` keeps the first ``limit`` items in file order
    (see :func:`select`). ``expected_count`` and ``expected_ids_sha256`` refer to the whole
    file. An empty file is refused."""
    if shape not in SHAPES:
        raise DatasetLoadError(f"shape must be one of {sorted(SHAPES)}")
    if split != SHAPES[shape]:
        raise DatasetLoadError(
            f"{shape} items are {SHAPES[shape]!r} in this protocol, not {split!r}"
        )
    if limit is not None and (type(limit) is not int or limit < 1):
        raise DatasetLoadError("limit must be a positive integer")
    path = Path(path)
    data = path.read_bytes()
    source = sha256_bytes(data)
    if expected_source_sha256 is not None and source != expected_source_sha256:
        raise DatasetLoadError(f"{path.name} SHA-256 differs from the registered value")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetLoadError(f"{path.name} is not UTF-8") from exc
    rows = _rows(path.name, text)
    if not rows:
        raise DatasetLoadError(f"{path.name} contains no items")
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
    report = DatasetReport(
        source_file=path.name,
        source_sha256=source,
        shape=shape,
        split=split,
        id_source=next(iter(id_sources)),
        n_source_rows=len(items),
        source_ids_sha256=ids,
        limit=None,
        n_items=len(items),
        ids_sha256=ids,
        items_sha256=dataset_sha256(items),
        checks=checks,
    )
    return select(items, report, limit)


def select(
    items: Sequence[Item], report: DatasetReport, limit: int | None
) -> tuple[list[Item], DatasetReport]:
    """Keep the first ``limit`` items in file order (all if ``None``). The report keeps the
    whole file's row count and ID-set hash and records the selection's."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise DatasetLoadError("limit must be a positive integer")
    if report.limit is not None:
        raise DatasetLoadError("items were already limited")
    chosen = list(items) if limit is None else list(items)[:limit]
    return chosen, replace(
        report,
        limit=limit,
        n_items=len(chosen),
        ids_sha256=ids_sha256(chosen),
        items_sha256=dataset_sha256(chosen),
    )


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


# --- protocol-registered sources ----------------------------------------------------------

HASH_DEFINITIONS = "wsp-runtime-dataset-1"
HASH_DEFINITION_TEXT = {
    "source_file_sha256": "SHA-256 of the raw source file bytes as fetched (before conversion)",
    "source_rows": "number of items converted from the whole source file",
    "source_ids_sha256": "SHA-256 of the canonical JSON of the sorted item IDs of the whole file",
    "items_sha256": (
        "the harness dataset_sha256 of the converted items after the selection rule (IDs, "
        "split and prompt/answer content hashes); not a hash of the source file"
    ),
}
SELECTION_RULES = ("all", "first_n_in_file_order")
_SHA = re.compile(r"[0-9a-f]{64}")


def registered_source(dataset: dict[str, Any], split: str) -> dict[str, Any]:
    """The protocol's registration of the source for ``split``:
    ``dataset.runtime_sources[split]`` with ``shape``, ``source_file_sha256``, ``source_rows``,
    ``source_ids_sha256``, ``selection`` and (required for held-out) ``items_sha256``, under
    ``dataset.runtime_sources.hash_definitions == HASH_DEFINITIONS``. Placeholders are refused."""
    sources = dataset.get("runtime_sources")
    where = f"dataset.runtime_sources.{split}"
    if not isinstance(sources, dict) or sources.get("hash_definitions") != HASH_DEFINITIONS:
        raise DatasetLoadError(
            f"the protocol must register dataset.runtime_sources with hash_definitions "
            f"{HASH_DEFINITIONS!r}"
        )
    spec = sources.get(split)
    if not isinstance(spec, dict):
        raise DatasetLoadError(f"the protocol does not register {where}")
    known = {"shape", "source_file_sha256", "source_rows", "source_ids_sha256", "selection"}
    unknown = sorted(set(spec) - known - {"items_sha256", "note"})
    if unknown:
        raise DatasetLoadError(f"{where}: keys not understood: {unknown}")
    missing = sorted(known - set(spec))
    if split == "heldout" and "items_sha256" not in spec:
        missing.append("items_sha256")
    if missing:
        raise DatasetLoadError(f"{where}: missing {missing}")
    if spec["shape"] not in SHAPES or SHAPES[spec["shape"]] != split:
        raise DatasetLoadError(f"{where}.shape must be the {split} shape")
    for key in ("source_file_sha256", "source_ids_sha256", "items_sha256"):
        if key in spec and not (isinstance(spec[key], str) and _SHA.fullmatch(spec[key])):
            raise DatasetLoadError(f"{where}.{key} must be a registered SHA-256")
    if type(spec["source_rows"]) is not int or spec["source_rows"] < 1:
        raise DatasetLoadError(f"{where}.source_rows must be a positive integer")
    sel = spec["selection"]
    rule = sel.get("rule") if isinstance(sel, dict) else None
    if rule not in SELECTION_RULES:
        raise DatasetLoadError(f"{where}.selection.rule must be one of {SELECTION_RULES}")
    if rule == "all" and set(sel) != {"rule"}:
        raise DatasetLoadError(f"{where}.selection: rule 'all' takes no other keys")
    if rule == "first_n_in_file_order" and (
        set(sel) != {"rule", "n"} or type(sel["n"]) is not int or sel["n"] < 1
    ):
        raise DatasetLoadError(f"{where}.selection: first_n_in_file_order needs a positive n")
    if rule == "first_n_in_file_order" and sel["n"] > spec["source_rows"]:
        raise DatasetLoadError(f"{where}.selection.n asks for more than source_rows items")
    if split == "heldout" and rule != "all":
        raise DatasetLoadError(f"{where}.selection: held-out items are all evaluated")
    declared = dataset.get("n_items")
    if split == "heldout" and type(declared) is int and declared != spec["source_rows"]:
        raise DatasetLoadError(f"{where}.source_rows differs from dataset.n_items")
    return dict(spec)


def load_registered(
    path: Path, dataset: dict[str, Any], split: str
) -> tuple[list[Item], DatasetReport]:
    """Load the whole file against the protocol's registration for ``split`` (raw file hash,
    row count, whole-file ID set), then apply its selection rule and check the converted items'
    hash if registered. Returns the **whole** file's items with a report of the selection; use
    :func:`select_registered` to get the selected items after any whole-file checks."""
    spec = registered_source(dataset, split)
    items, report = load(
        path,
        shape=spec["shape"],
        split=split,
        expected_count=spec["source_rows"],
        expected_source_sha256=spec["source_file_sha256"],
        expected_ids_sha256=spec["source_ids_sha256"],
    )
    return items, report


def select_registered(
    items: Sequence[Item], report: DatasetReport, dataset: dict[str, Any], split: str
) -> tuple[list[Item], DatasetReport]:
    spec = registered_source(dataset, split)
    sel = spec["selection"]
    chosen, out = select(items, report, sel["n"] if sel["rule"] != "all" else None)
    if "items_sha256" in spec and out.items_sha256 != spec["items_sha256"]:
        raise DatasetLoadError(
            f"dataset.runtime_sources.{split}.items_sha256 differs from the selected items"
        )
    return chosen, out


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

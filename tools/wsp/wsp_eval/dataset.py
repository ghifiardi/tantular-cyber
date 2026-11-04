"""Evaluation items with stable IDs and content hashes; split and duplicate checks."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import sha256_json
from .protocol import SPLITS


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    id: str
    split: str
    prompt: str
    answer: str

    @property
    def content_sha256(self) -> str:
        # The ID and split are labels, not content: identical content under two IDs is a
        # duplicate, and identical content in two splits is leakage.
        return sha256_json({"prompt": self.prompt, "answer": self.answer})


def parse_items(rows: list[dict[str, Any]]) -> list[Item]:
    items = []
    for n, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"id", "split", "prompt", "answer"}:
            raise DatasetError(f"row {n}: expected exactly id, split, prompt, answer")
        if not all(isinstance(row[k], str) and row[k] for k in row):
            raise DatasetError(f"row {n}: all fields must be non-empty strings")
        if row["split"] not in SPLITS:
            raise DatasetError(f"row {n}: split must be one of {SPLITS}")
        items.append(Item(**row))
    return items


def load_items(path: Path) -> list[Item]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}:{n}: invalid JSON ({exc.msg})") from exc
    return parse_items(rows)


def dataset_sha256(items: list[Item]) -> str:
    return sha256_json(sorted([i.id, i.split, i.content_sha256] for i in items))


def check_items(items: list[Item]) -> dict[str, Any]:
    """Integrity report. ``ok`` is False on duplicate IDs or cross-split content overlap.

    Within-split duplicate content is reported (it inflates effective sample size) but is not
    fatal; the protocol should say how it is handled.
    """
    errors: list[str] = []
    ids: dict[str, int] = defaultdict(int)
    by_content: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        ids[item.id] += 1
        by_content[item.content_sha256].append(item)
    dup_ids = sorted(i for i, c in ids.items() if c > 1)
    if dup_ids:
        errors.append(f"duplicate ids: {dup_ids}")
    cross, within = [], []
    for group in by_content.values():
        if len(group) < 2:
            continue
        names = sorted(i.id for i in group)
        if len({i.split for i in group}) > 1:
            cross.append(names)
        else:
            within.append(names)
    if cross:
        errors.append(f"content shared across splits (leakage): {cross}")
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.split] += 1
    return {
        "ok": not errors,
        "errors": errors,
        "n_items": len(items),
        "per_split": dict(sorted(counts.items())),
        "within_split_duplicates": within,
        "dataset_sha256": dataset_sha256(items),
    }

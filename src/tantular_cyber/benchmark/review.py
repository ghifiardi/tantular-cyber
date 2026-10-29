"""Human-review ingestion: validated reviewer files, inter-rater agreement, adjudication.

Workflow: ``benchmark --llm ...`` writes ``human_review.json`` (the queue). Each reviewer
fills a copy made by ``tantular-cyber review template`` (pseudonym only), and
``tantular-cyber review summarize`` aggregates them. Reviewer scores are reported
separately from the deterministic proxy checks and never gate CI. Missing scores are
excluded per dimension — never imputed. Free-text notes are counted, not published.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

REVIEW_SCHEMA = "tantular-cyber.human-review"
CATEGORICAL = {
    "remediation_adequate": ["yes", "partial", "no"],
    "evidence_faithful": ["yes", "no"],
    "overclaim_present": ["yes", "no"],
}
ORDINAL = {"clarity_id": [1, 2, 3, 4, 5]}
DIMENSIONS = [*ORDINAL, *CATEGORICAL]
# Human dimension -> (automated check, value meaning "check passed")
PROXY_PAIRS = {
    "evidence_faithful": ("evidence_grounded", "yes"),
    "overclaim_present": ("no_unsupported_assertions", "no"),
}


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    clarity_id: int | None = Field(default=None, ge=1, le=5)
    remediation_adequate: Literal["yes", "partial", "no"] | None = None
    evidence_faithful: Literal["yes", "no"] | None = None
    overclaim_present: Literal["yes", "no"] | None = None
    notes: str | None = Field(default=None, max_length=2000)


class ReviewFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal["tantular-cyber.human-review"] = Field(alias="schema")
    schema_version: str
    reviewer: str = Field(
        pattern=r"^[A-Za-z0-9_-]{1,32}$",
        description="Pseudonym only (no names or e-mail addresses).",
    )
    role: Literal["primary", "adjudicator"] = "primary"
    queue_sha256: str | None = None
    reviewed_at: str | None = None
    items: list[ReviewItem]

    @field_validator("items")
    @classmethod
    def _unique(cls, items: list[ReviewItem]) -> list[ReviewItem]:
        ids = [i.review_id for i in items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate review_id in items")
        return items


class Agreement(BaseModel):
    dimension: str
    reviewers: list[str]
    n: int
    raw_agreement: float | None
    kappa: float | None
    kappa_kind: Literal["cohen", "quadratic_weighted"]
    within_one: float | None = None


class DimensionSummary(BaseModel):
    dimension: str
    rated_items: int
    consensus: dict[str, int]
    unresolved: int
    missing: int


class ProxyAgreement(BaseModel):
    human_dimension: str
    proxy_check: str
    n: int
    both_pass: int
    both_fail: int
    human_pass_proxy_fail: int
    human_fail_proxy_pass: int


class ReviewSummary(BaseModel):
    result_type: str = "tantular-cyber.review-summary"
    schema_version: str = "1.0.0"
    queue_items: int
    queue_sha256: str
    reviewers: list[str]
    adjudicators: list[str]
    warnings: list[str]
    pairwise: list[Agreement]
    consensus: list[DimensionSummary]
    proxy_agreement: list[ProxyAgreement]
    notes_count: int
    caveat: str = (
        "Human ratings inform model development; they are subjective, sample-based and "
        "never used as CI gates. Kappa is undefined (null) when agreement by chance is 1."
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_review(path: Path) -> ReviewFile:
    return ReviewFile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def template(
    queue: list[dict[str, object]], reviewer: str, queue_sha: str, role: str = "primary"
) -> dict[str, object]:
    ReviewFile.model_validate(
        {
            "schema": REVIEW_SCHEMA,
            "schema_version": "1.0.0",
            "reviewer": reviewer,
            "role": role,
            "items": [],
        }
    )
    return {
        "schema": REVIEW_SCHEMA,
        "schema_version": "1.0.0",
        "reviewer": reviewer,
        "role": role,
        "queue_sha256": queue_sha,
        "reviewed_at": None,
        "items": [
            {"review_id": q["review_id"], **{d: None for d in DIMENSIONS}, "notes": None}
            for q in queue
        ],
    }


def cohen_kappa(
    a: Sequence[object], b: Sequence[object], categories: Sequence[object], weighted: bool = False
) -> float | None:
    n = len(a)
    if n == 0:
        return None
    k = len(categories)
    pos = {c: i for i, c in enumerate(categories)}

    def weight(i: int, j: int) -> float:
        return ((i - j) ** 2) / ((k - 1) ** 2) if weighted and k > 1 else float(i != j)

    observed = sum(weight(pos[x], pos[y]) for x, y in zip(a, b, strict=True)) / n
    pa = Counter(pos[x] for x in a)
    pb = Counter(pos[y] for y in b)
    expected = sum(weight(i, j) * pa[i] * pb[j] for i in range(k) for j in range(k)) / n**2
    if expected == 0:
        return None
    return round(1 - observed / expected, 4)


def _values(review: ReviewFile, dim: str) -> dict[str, object]:
    return {i.review_id: getattr(i, dim) for i in review.items if getattr(i, dim) is not None}


def summarize_reviews(
    queue: list[dict[str, object]],
    queue_sha: str,
    reviews: Sequence[ReviewFile],
    adjudications: Sequence[ReviewFile] = (),
) -> ReviewSummary:
    queue_ids = {str(q["review_id"]) for q in queue}
    warnings: list[str] = []
    for r in [*reviews, *adjudications]:
        unknown = sorted({i.review_id for i in r.items} - queue_ids)
        if unknown:
            warnings.append(f"{r.reviewer}: {len(unknown)} item(s) not in this queue ignored")
        if r.queue_sha256 and r.queue_sha256 != queue_sha:
            warnings.append(f"{r.reviewer}: reviewed a different queue file (hash mismatch)")
    primaries = [r for r in reviews if r.role == "primary"]
    names = [r.reviewer for r in primaries]
    if len(set(names)) != len(names):
        warnings.append("duplicate reviewer pseudonyms")
    if len(primaries) < 2:
        warnings.append("fewer than two primary reviewers: agreement metrics not computed")

    pairwise: list[Agreement] = []
    for ra, rb in combinations(primaries, 2):
        for dim in DIMENSIONS:
            va, vb = _values(ra, dim), _values(rb, dim)
            shared = sorted((set(va) & set(vb)) & queue_ids)
            xs, ys = [va[i] for i in shared], [vb[i] for i in shared]
            ordinal = dim in ORDINAL
            cats = ORDINAL.get(dim) or CATEGORICAL[dim]
            n = len(shared)
            pairwise.append(
                Agreement(
                    dimension=dim,
                    reviewers=[ra.reviewer, rb.reviewer],
                    n=n,
                    raw_agreement=round(sum(x == y for x, y in zip(xs, ys, strict=True)) / n, 4)
                    if n
                    else None,
                    kappa=cohen_kappa(xs, ys, cats, weighted=ordinal),
                    kappa_kind="quadratic_weighted" if ordinal else "cohen",
                    within_one=round(
                        sum(
                            abs(int(x) - int(y)) <= 1  # type: ignore[call-overload]
                            for x, y in zip(xs, ys, strict=True)
                        )
                        / n,
                        4,
                    )
                    if ordinal and n
                    else None,
                )
            )

    consensus: list[DimensionSummary] = []
    consensus_values: dict[str, dict[str, object]] = {}
    for dim in DIMENSIONS:
        per_item: dict[str, object] = {}
        unresolved = missing = 0
        rated = 0
        for rid in sorted(queue_ids):
            vals = [v for r in primaries if (v := _values(r, dim).get(rid)) is not None]
            adj = [v for r in adjudications if (v := _values(r, dim).get(rid)) is not None]
            if not vals:
                missing += 1
                continue
            rated += 1
            if len(set(map(str, vals))) == 1 and (len(vals) >= 2 or len(primaries) == 1):
                per_item[rid] = vals[0]
            elif adj:
                per_item[rid] = adj[0]
            else:
                unresolved += 1
        consensus_values[dim] = per_item
        consensus.append(
            DimensionSummary(
                dimension=dim,
                rated_items=rated,
                consensus=dict(sorted(Counter(str(v) for v in per_item.values()).items())),
                unresolved=unresolved,
                missing=missing,
            )
        )

    automated = {str(q["review_id"]): q.get("automated_checks") or {} for q in queue}
    proxies: list[ProxyAgreement] = []
    for dim, (check, passing) in PROXY_PAIRS.items():
        counts = Counter()
        for rid, value in consensus_values[dim].items():
            proxy = automated.get(rid, {}).get(check)  # type: ignore[union-attr]
            if proxy is None:
                continue
            human_pass = value == passing
            counts[(human_pass, bool(proxy))] += 1
        proxies.append(
            ProxyAgreement(
                human_dimension=dim,
                proxy_check=check,
                n=sum(counts.values()),
                both_pass=counts[(True, True)],
                both_fail=counts[(False, False)],
                human_pass_proxy_fail=counts[(True, False)],
                human_fail_proxy_pass=counts[(False, True)],
            )
        )
    notes = sum(1 for r in [*reviews, *adjudications] for i in r.items if i.notes)
    return ReviewSummary(
        queue_items=len(queue_ids),
        queue_sha256=queue_sha,
        reviewers=names,
        adjudicators=[r.reviewer for r in adjudications],
        warnings=warnings,
        pairwise=pairwise,
        consensus=consensus,
        proxy_agreement=proxies,
        notes_count=notes,
    )


def render_review_summary(s: ReviewSummary) -> str:
    out = [
        "# Human review summary",
        "",
        f"> {s.caveat}",
        "",
        f"Queue items {s.queue_items} (sha256 `{s.queue_sha256[:12]}…`); primary reviewers: "
        f"{', '.join(s.reviewers) or '-'}; adjudicators: {', '.join(s.adjudicators) or '-'}; "
        f"free-text notes: {s.notes_count} (not published).",
        "",
    ]
    if s.warnings:
        out += ["**Warnings:**"] + [f"- {w}" for w in s.warnings] + [""]
    if s.pairwise:
        out += [
            "## Inter-rater agreement (pairwise)",
            "",
            "| Dimension | Reviewers | n | Raw agreement | Kappa | Kind | Within ±1 |",
            "|---|---|---|---|---|---|---|",
        ]
        out += [
            f"| {a.dimension} | {' / '.join(a.reviewers)} | {a.n} | {a.raw_agreement} | "
            f"{a.kappa} | {a.kappa_kind} | {a.within_one if a.within_one is not None else '-'} |"
            for a in s.pairwise
        ]
        out.append("")
    out += [
        "## Consensus (agreement or adjudication)",
        "",
        "| Dimension | Rated | Consensus distribution | Unresolved | Missing |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {c.dimension} | {c.rated_items} | {c.consensus} | {c.unresolved} | {c.missing} |"
        for c in s.consensus
    ]
    out += [
        "",
        "## Human consensus vs deterministic proxy checks",
        "",
        "| Human dimension | Proxy check | n | Both pass | Both fail | Human pass / proxy fail "
        "| Human fail / proxy pass |",
        "|---|---|---|---|---|---|---|",
    ]
    out += [
        f"| {p.human_dimension} | {p.proxy_check} | {p.n} | {p.both_pass} | {p.both_fail} | "
        f"{p.human_pass_proxy_fail} | {p.human_fail_proxy_pass} |"
        for p in s.proxy_agreement
    ]
    return "\n".join(out) + "\n"

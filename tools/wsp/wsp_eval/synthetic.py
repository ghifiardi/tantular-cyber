"""Scripted synthetic arms for harness smoke tests. Not a model; outputs mean nothing.

The script exercises every record path: correct answers, wrong answers, a looping output
that hits the token limit with no final answer (ungraded), an adapter error, and a no-edit
reload that is byte-identical to the reference by default. Options simulate a reload that
differs as stochastic batched sampling can, and a "reload" that reused the reference's load.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .dataset import Item
from .generation import Generation, GenerationError, SyntheticGenerator
from .hashing import sha256_bytes

_LOOP = "Wait, let me re-check. " * 8
_ARTIFACTS = {"model.safetensors": "0" * 64, "tokenizer.json": "1" * 64}


def _prompt(item: Item) -> str:
    return sha256_bytes(f"<synthetic-template>{item.prompt}".encode())


def _correct(item: Item) -> Generation:
    return Generation(
        f"Working.\n\\boxed{{{item.answer}}}",
        output_tokens=12,
        finish_reason="stop",
        rendered_prompt_sha256=_prompt(item),
    )


def _wrong(item: Item) -> Generation:
    return Generation(
        "Working.\n\\boxed{-1}",
        output_tokens=10,
        finish_reason="stop",
        rendered_prompt_sha256=_prompt(item),
    )


def _loop(item: Item) -> Generation:
    return Generation(
        _LOOP, output_tokens=64, finish_reason="length", rendered_prompt_sha256=_prompt(item)
    )


def synthetic_generators(
    items: Sequence[Item],
    seeds: Sequence[int],
    *,
    nondeterministic_reload: bool = False,
    same_load_id: bool = False,
) -> dict[str, SyntheticGenerator]:
    """Scripted arms keyed (item, seed) for per-sample seeds; (item, seeds[0], k) entries alias
    sample k for ``n_per_request`` protocols whose single request seed is ``seeds[0]``."""
    ordered = sorted(items, key=lambda i: i.id)
    dev = [i for i in ordered if i.split == "development"]
    ref: dict[tuple, Generation | Exception] = {}
    cand: dict[tuple, Generation | Exception] = {}
    for n, item in enumerate(ordered):
        for s, seed in enumerate(seeds):
            # Reference loops on two of every three items for the first seed.
            ref[(item.id, seed)] = _loop(item) if (n % 3 != 0 and s == 0) else _correct(item)
            if n == 1 and s == 1:
                ref[(item.id, seed)] = _wrong(item)
            cand[(item.id, seed)] = _correct(item)
    last = dev[-1] if dev else ordered[-1]
    cand[(last.id, seeds[-1])] = GenerationError("synthetic adapter timeout")
    rel = dict(ref)
    if nondeterministic_reload and len(dev) >= 3:
        first, looped = dev[0], dev[2]
        g = rel[(first.id, seeds[0])]
        if isinstance(g, Generation):
            rel[(first.id, seeds[0])] = replace(g, text=g.text.replace("Working.", "Computing."))
        rel[(looped.id, seeds[0])] = _correct(looped)  # the reference looped here
    for table in (ref, cand, rel):
        for (item_id, seed), entry in list(table.items()):
            table[(item_id, seeds[0], list(seeds).index(seed))] = entry
    return {
        "reference": SyntheticGenerator(
            "synthetic-reference", ref, load_id="load-ref", artifacts_sha256=_ARTIFACTS
        ),
        "candidate": SyntheticGenerator(
            "synthetic-candidate", cand, load_id="load-cand", artifacts_sha256=_ARTIFACTS
        ),
        "reload": SyntheticGenerator(
            "synthetic-reference",
            rel,
            load_id="load-ref" if same_load_id else "load-reload",
            artifacts_sha256=_ARTIFACTS,
        ),
    }

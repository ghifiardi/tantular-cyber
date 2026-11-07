"""Provider-neutral generation interface.

Any model backend (local inference engine, hosted API, research controller) is an adapter
implementing :class:`Generator`. ``describe()`` must report the *actual* provider, model,
revision and engine so they land in the run manifest; nothing here assumes a specific
provider is available. The only implementation in this repository is
:class:`SyntheticGenerator`, a deterministic lookup table used to test the harness. It is not
a model and its outputs are not measurements.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, Union


@dataclass(frozen=True)
class GenerationRequest:
    arm_id: str
    item_id: str
    prompt: str
    inference: dict[str, Any]
    seed: int
    n: int = 1  # samples requested in this call (n_per_request mode)


@dataclass(frozen=True)
class Generation:
    text: str
    output_tokens: int | None = None
    hit_token_limit: bool | None = None
    # Engine finish reason ("stop", "length", or anything else = abnormal, recorded as an error).
    finish_reason: str | None = None
    # SHA-256 of the exact prompt string the engine received after chat templating, so prompt
    # identity across arms is checked rather than assumed.
    rendered_prompt_sha256: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class GenerationError(RuntimeError):
    """Raised by an adapter for a failed request; the runner records it, never drops it."""


class Generator(Protocol):
    def describe(self) -> dict[str, Any]:
        """Return at least provider, model, revision, engine, engine_version (null when unknown),
        ``artifacts_sha256`` (file -> digest actually loaded) and a per-load ``load_id`` so a
        reload can be shown to be a separate load."""
        ...

    def generate(self, request: GenerationRequest) -> Generation | list[Generation]:
        """One Generation for n == 1, else a list of exactly ``request.n`` Generations."""
        ...


# A request's result: its Generation(s), or the exception that failed that request alone
# (recorded as error rows for its samples, never dropped).
BatchResult = Union["Generation", list["Generation"], Exception]


@dataclass(frozen=True)
class BatchItem:
    """One entry of a batch result. ``request`` echoes the request it answers, so the runner can
    refuse a reordered or substituted entry instead of crediting it to another request
    (``Generation`` itself carries no request identity)."""

    request: GenerationRequest
    result: BatchResult


class BatchGenerator(Generator, Protocol):
    """A generator that serves a whole shard in one engine call (``per_arm_sequential`` runs).

    ``generate_batch`` returns exactly one :class:`BatchItem` per request, in request order,
    each echoing its request. Raising fails every request of the batch. ``close()`` (required)
    releases the engine (for example the GPU) before the next arm is loaded; the runner calls it
    once per load, also when the arm fails.
    """

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> Sequence[BatchItem]: ...

    def close(self) -> None: ...


# Loads one arm's model and returns a fresh generator. Called exactly once per arm, in protocol
# order, in ``per_arm_sequential`` runs; every call must be a separate load (distinct load_id).
ArmLoader = Callable[[], BatchGenerator]


class SyntheticGenerator:
    """Deterministic table: (item_id, seed) or (item_id, seed, k) -> Generation or exception.

    For ``n > 1`` requests, sample ``k`` is looked up as (item_id, seed, k); an exception stored
    at (item_id, seed) fails the whole request. Missing keys raise :class:`GenerationError`, so
    an incomplete fixture shows up as explicit error rows rather than a silent gap.
    """

    def __init__(
        self,
        name: str,
        table: dict[tuple[Any, ...], Generation | Exception],
        *,
        load_id: str | None = None,
        artifacts_sha256: dict[str, str] | None = None,
    ):
        self.name = name
        self.table = table
        self.load_id = load_id
        self.artifacts_sha256 = artifacts_sha256 or {}
        self.batches: list[list[tuple[str, int, int]]] = []  # shard compositions served
        self.closed = False

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "synthetic",
            "model": self.name,
            "revision": None,
            "engine": "wsp_eval.SyntheticGenerator",
            "engine_version": None,
            "artifacts_sha256": dict(self.artifacts_sha256),
            "load_id": self.load_id,
            "synthetic": True,
        }

    def _lookup(self, key: tuple[Any, ...]) -> Generation:
        entry = self.table.get(key)
        if entry is None:
            raise GenerationError(f"no synthetic output for {key}")
        if isinstance(entry, Exception):
            raise entry
        return entry

    def generate(self, request: GenerationRequest) -> Generation | list[Generation]:
        if request.n == 1:
            return self._lookup((request.item_id, request.seed))
        whole = self.table.get((request.item_id, request.seed))
        if isinstance(whole, Exception):
            raise whole
        return [self._lookup((request.item_id, request.seed, k)) for k in range(request.n)]

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[BatchItem]:
        """Serve a shard: each request's result, or the exception that failed it alone."""
        self.batches.append([(r.item_id, r.seed, r.n) for r in requests])
        out: list[BatchItem] = []
        for request in requests:
            try:
                out.append(BatchItem(request, self.generate(request)))
            except Exception as exc:  # per-request failure, reported in its slot
                out.append(BatchItem(request, exc))
        return out

    def close(self) -> None:
        self.closed = True

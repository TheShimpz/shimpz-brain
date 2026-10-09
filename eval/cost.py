"""Estimated provider cost of an evaluation: list prices, frozen cache prices, unknown usage, and hard budgets.

Cost is an estimate from the model catalog's list prices, never billing (ADR-0094). The catalog carries no cache
prices, so they are frozen here: a cache read costs a fraction of the input price, an Anthropic five-minute cache
write (the only lifetime Brain requests) costs 1.25 times the input price, and OpenAI, which reports no cache writes,
bills its cached prefix as ordinary input. Fresh input is what remains after cache reads and writes; a cache read or
write is never priced as fresh input.

Usage a provider did not report, or a call that failed, makes the usage unknown: the known part is a lower bound.

This module uses only the standard library so that the umbrella journey driver can load it by path.
"""

import contextlib
import fcntl
import json
import math
import os
import secrets
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "model_catalog.json"
# The provider-reported usage fields, in Brain's `model_usage.FIELDS` order (a test pins the equality).
FIELDS = (
    "model_calls",
    "failed_calls",
    "unreported_calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)
# Cache prices as multiples of the input price, frozen on 2026-10-02 from the providers' pricing pages.
CACHE_READ_MULTIPLIER = {"openai": 0.1, "anthropic": 0.1}
CACHE_READ_MULTIPLIER_BY_MODEL = {"claude-opus-5-5": 0.05, "gpt-6.1-sol": 0.05}
CACHE_WRITE_MULTIPLIER = {"openai": 1.0, "anthropic": 1.25}
# Models priced for evaluation only, which the product catalog does not offer, in US cents per million input and output
# tokens: gpt-5.6-sol's promotional list price (listed through at least 2026-11-21, prompts up to 272k input tokens),
# frozen on 2026-10-02. Only a disposable evaluation Brain admits them (``admit_evaluation_models``).
EVALUATION_MODELS = {"openai": {"gpt-5.6-sol": (400, 2000)}}
# Embedding models priced for the language-retrieval component benchmark only, in US cents per million input tokens
# (they bill no output), frozen on 2026-10-03 from the provider's pricing page; only the evaluation ceiling's embeddings
# endpoint reserves for them.
EMBEDDING_MODELS = {"openai": {"text-embedding-3-small": 2, "text-embedding-3-large": 13}}
# The input bound up to which an evaluation-only price holds; above it the provider bills a dearer long-context price,
# so the ceiling refuses the request instead of reserving it at this one.
PRICED_INPUT_TOKENS = {"gpt-5.6-sol": 272_000}


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per token."""

    provider: str
    input: float
    output: float
    cache_read: float
    cache_write: float


def _list_prices(catalog: Path) -> Iterator[tuple[str, str, int, int]]:
    """(provider, model, input cents, output cents) per million tokens: the catalog's, then the evaluation-only ones."""
    for provider in json.loads(catalog.read_text(encoding="utf-8"))["providers"]:
        for entry in provider["models"]:
            yield (
                provider["id"],
                entry["id"],
                entry["input_usd_per_million_cents"],
                entry["output_usd_per_million_cents"],
            )
    for provider_id, models in EVALUATION_MODELS.items():
        for model_id, (input_cents, output_cents) in models.items():
            yield provider_id, model_id, input_cents, output_cents
    for provider_id, models in EMBEDDING_MODELS.items():
        for model_id, input_cents in models.items():
            yield provider_id, model_id, input_cents, 0


def price(model: str, catalog: Path = CATALOG) -> Price:
    for provider, model_id, input_cents, output_cents in _list_prices(catalog):
        if model_id == model:
            per_token = input_cents / 100 / 1_000_000
            read = CACHE_READ_MULTIPLIER_BY_MODEL.get(model, CACHE_READ_MULTIPLIER[provider])
            return Price(
                provider,
                per_token,
                output_cents / 100 / 1_000_000,
                per_token * read,
                per_token * CACHE_WRITE_MULTIPLIER[provider],
            )
    raise ValueError("unknown model")


def admit_evaluation_models(models_by_provider: dict[str, frozenset[str]]) -> None:
    """Let one disposable evaluation Brain run the evaluation-only models; no product process calls this."""
    for provider, models in EVALUATION_MODELS.items():
        models_by_provider[provider] = models_by_provider[provider] | frozenset(models)


@dataclass(frozen=True, slots=True)
class Usage:
    model_calls: int = 0
    failed_calls: int = 0
    unreported_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def of(cls, counts: Mapping[str, int]) -> Usage:
        if set(counts) != set(FIELDS) or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("invalid usage counts")
        return cls(**{name: counts[name] for name in FIELDS})

    def __add__(self, other: Usage) -> Usage:
        return Usage(**{item.name: getattr(self, item.name) + getattr(other, item.name) for item in fields(Usage)})

    @property
    def known(self) -> bool:
        """Every call reported its usage; otherwise the counts are only a lower bound."""
        return self.failed_calls == 0 and self.unreported_calls == 0

    @property
    def fresh_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cache_read_tokens - self.cache_write_tokens)

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in FIELDS}


@dataclass(frozen=True, slots=True)
class Cost:
    usd: float
    known: bool = True

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.usd + other.usd, self.known and other.known)


def cost(usage: Usage, model: str, catalog: Path = CATALOG) -> Cost:
    rates = price(model, catalog)
    usd = (
        usage.fresh_input_tokens * rates.input
        + usage.cache_read_tokens * rates.cache_read
        + usage.cache_write_tokens * rates.cache_write
        + usage.output_tokens * rates.output
    )
    return Cost(usd, usage.known)


def call_bound(model: str, input_tokens: int, output_tokens: int, catalog: Path = CATALOG) -> float:
    """A conservative bound for one call: every input token priced fresh (the dearest of fresh and cache read)."""
    rates = price(model, catalog)
    return input_tokens * max(rates.input, rates.cache_write) + output_tokens * rates.output


class BudgetExhaustedError(RuntimeError):
    """The next reservation would cross the hard cap; nothing was dispatched."""


@dataclass(frozen=True, slots=True)
class Reservation:
    amount: float
    # A shared ledger's id for this reservation, which settles it exactly once; an in-process Budget needs none.
    key: str = ""


class Budget:
    """A hard cap enforced before dispatch: reserve a conservative bound, then settle what the work reported.

    A reservation the cap cannot hold even after all in-flight work settles is refused: the budget is exhausted. One
    that fits only once in-flight reservations settle may wait for them (``wait=True``) instead, so concurrent work
    contends for the cap without stopping a campaign that still has money. A settled cost that is unknown keeps the
    whole reservation, or the known part when it is larger. Work already in flight cannot be interrupted, so a cost
    above its reservation is counted rather than hidden.
    """

    def __init__(self, cap: float) -> None:
        if not cap >= 0:
            raise ValueError("invalid budget cap")
        self.cap = cap
        self.spent = 0.0
        self.unknown = 0
        self.exceeded = 0
        self.waits = 0
        self._reserved = 0.0
        self._changed = threading.Condition(threading.Lock())

    def reserve(self, amount: float, *, wait: bool = False) -> Reservation:
        if not amount >= 0:
            raise ValueError("invalid reservation")
        with self._changed:
            waited = False
            while self.spent + self._reserved + amount > self.cap:
                if not wait or self.spent + amount > self.cap:
                    raise BudgetExhaustedError("budget cap reached")
                waited = True
                self._changed.wait()
            self.waits += waited
            self._reserved += amount
        return Reservation(amount)

    def settle(self, reservation: Reservation, spent: Cost) -> None:
        with self._changed:
            self._reserved -= reservation.amount
            charged = spent.usd if spent.known else max(spent.usd, reservation.amount)
            self.spent += charged
            self.unknown += not spent.known
            self.exceeded += spent.usd > reservation.amount
            self._changed.notify_all()

    def summary(self) -> dict[str, object]:
        with self._changed:
            return {
                "cap_usd": self.cap,
                "spent_usd": round(self.spent, 6),
                "unknown_settlements": self.unknown,
                "reservations_exceeded": self.exceeded,
                "contention_waits": self.waits,
            }


# How long a shared reservation that has to wait for in-flight work sleeps, at least and at most, before it reads the
# ledger again: spread at random, so hundreds of waiting requests never contend for the lock in step.
LEDGER_POLL_SECONDS = (0.1, 0.3)
_JITTER = secrets.SystemRandom()


class LedgerError(BudgetExhaustedError):
    """The shared ledger is missing, unreadable, or malformed, or a settlement names no open reservation.

    Nothing is spent against a ledger that cannot be read, so it refuses like an exhausted budget.
    """


def _amount(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def _valid_ledger(state: object) -> bool:
    if not isinstance(state, dict) or set(state) != {"cap", "spent", "open", "unknown", "exceeded", "waits"}:
        return False
    amounts = (state["cap"], state["spent"])
    counts = (state["unknown"], state["exceeded"], state["waits"])
    open_ = state["open"]
    return (
        all(map(_amount, amounts))
        and all(map(_count, counts))
        and isinstance(open_, dict)
        and all(isinstance(key, str) and _amount(value) for key, value in open_.items())
    )


class SharedBudget:
    """Budget's hard cap shared by every process of one run through one ledger file.

    The same reserve-before, settle-after semantics as ``Budget``: a reservation that would cross the cap even after
    all in-flight work settles is refused, and one that only has to wait for in-flight work polls the ledger until it
    fits (``wait=True``). Each operation holds an exclusive lock on a stable lock file beside the ledger, opened anew
    so threads of one process exclude each other too, reads and validates the whole state, and replaces it
    atomically. Each reservation carries a unique id and settles exactly once. A reservation that is never settled,
    as when its process dies, stays charged at its whole amount.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock_path = path.with_name(path.name + ".lock")

    @classmethod
    def create(cls, path: Path, cap: float) -> SharedBudget:
        """Start a new owner-only ledger at ``path`` with nothing spent; an existing ledger is never replaced."""
        if not _amount(cap):
            raise ValueError("invalid budget cap")
        state = {"cap": cap, "spent": 0.0, "open": {}, "unknown": 0, "exceeded": 0, "waits": 0}
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        return cls(path)

    @contextlib.contextmanager
    def _ledger(self) -> Iterator[dict]:
        """The locked ledger state; whatever the block leaves in it is written back atomically."""
        descriptor = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise LedgerError("the shared budget ledger is unreadable") from exc
            if not _valid_ledger(state):
                raise LedgerError("the shared budget ledger is malformed")
            read = json.dumps(state, sort_keys=True)
            yield state
            if json.dumps(state, sort_keys=True) == read:
                return
            handle, temporary = tempfile.mkstemp(prefix=self.path.name, dir=self.path.parent)
            with os.fdopen(handle, "w", encoding="utf-8") as written:
                json.dump(state, written)
            Path(temporary).replace(self.path)
        finally:
            os.close(descriptor)

    def reserve(self, amount: float, *, wait: bool = False) -> Reservation:
        if not amount >= 0:
            raise ValueError("invalid reservation")
        waited = False
        while True:
            with self._ledger() as state:
                held = sum(state["open"].values())
                if state["spent"] + held + amount <= state["cap"]:
                    key = secrets.token_hex(16)
                    state["open"][key] = amount
                    state["waits"] += waited
                    return Reservation(amount, key)
                if not wait or state["spent"] + amount > state["cap"]:
                    raise BudgetExhaustedError("budget cap reached")
            waited = True
            time.sleep(_JITTER.uniform(*LEDGER_POLL_SECONDS))

    def settle(self, reservation: Reservation, spent: Cost) -> None:
        with self._ledger() as state:
            if state["open"].pop(reservation.key, None) != reservation.amount:
                raise LedgerError("the settlement names no open reservation")
            state["spent"] += spent.usd if spent.known else max(spent.usd, reservation.amount)
            state["unknown"] += not spent.known
            state["exceeded"] += spent.usd > reservation.amount

    def summary(self) -> dict[str, object]:
        with self._ledger() as state:
            return {
                "cap_usd": state["cap"],
                "spent_usd": round(state["spent"], 6),
                "unknown_settlements": state["unknown"],
                "reservations_exceeded": state["exceeded"],
                "contention_waits": state["waits"],
                # Reservations never settled, as when a process died mid-request: charged at their whole amount.
                "unsettled_usd": round(sum(state["open"].values()), 6),
            }


def per_task(costs: Iterable[Cost], successes: int) -> dict[str, object]:
    """Cost per attempted task and total cost, including failed attempts, per successful task."""
    costs = list(costs)
    total = sum((item for item in costs), Cost(0.0))
    attempts = len(costs)
    return {
        "attempts": attempts,
        "successes": successes,
        "usd": round(total.usd, 6),
        "usd_known": total.known,
        "unknown_usage_attempts": sum(not item.known for item in costs),
        "usd_per_attempted_task": round(total.usd / attempts, 6) if attempts else None,
        "usd_per_successful_task": round(total.usd / successes, 6) if successes else None,
    }

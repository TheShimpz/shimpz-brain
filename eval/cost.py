"""Estimated provider cost of an evaluation: list prices, frozen cache prices, unknown usage, and hard budgets.

Cost is an estimate from the model catalog's list prices, never billing (ADR-0094). The catalog carries no cache
prices, so they are frozen here: a cache read costs a fraction of the input price, an Anthropic five-minute cache
write (the only lifetime Brain requests) costs 1.25 times the input price, and OpenAI, which reports no cache writes,
bills its cached prefix as ordinary input. Fresh input is what remains after cache reads and writes; a cache read or
write is never priced as fresh input.

Usage a provider did not report, or a call that failed, makes the usage unknown: the known part is a lower bound.

This module uses only the standard library so that the umbrella journey driver can load it by path.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
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
CACHE_READ_MULTIPLIER_BY_MODEL = {"claude-opus-5-5": 0.05}
CACHE_WRITE_MULTIPLIER = {"openai": 1.0, "anthropic": 1.25}


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per token."""

    provider: str
    input: float
    output: float
    cache_read: float
    cache_write: float


def price(model: str, catalog: Path = CATALOG) -> Price:
    for provider in json.loads(catalog.read_text(encoding="utf-8"))["providers"]:
        for entry in provider["models"]:
            if entry["id"] == model:
                per_token = entry["input_usd_per_million_cents"] / 100 / 1_000_000
                read = CACHE_READ_MULTIPLIER_BY_MODEL.get(model, CACHE_READ_MULTIPLIER[provider["id"]])
                return Price(
                    provider["id"],
                    per_token,
                    entry["output_usd_per_million_cents"] / 100 / 1_000_000,
                    per_token * read,
                    per_token * CACHE_WRITE_MULTIPLIER[provider["id"]],
                )
    raise ValueError("unknown model")


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

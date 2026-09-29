"""Observed model usage of one Brain operation, reported to Team with the operation's result (ADR-0082).

The counts are what the provider responses reported: a call that failed, was cancelled, or was retried inside the
provider SDK reports no tokens, so the totals are a floor on billed usage, never an estimate of it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

FIELDS = (
    "model_calls",
    "failed_calls",
    "unreported_calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


def _cache_writes(details: dict[str, Any]) -> int:
    """Anthropic reports cache writes split by lifetime and leaves ``cache_creation`` at zero; count either form."""
    return int(details.get("cache_creation") or 0) or sum(
        int(details.get(name) or 0) for name in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )


class Usage(BaseCallbackHandler):
    """Thread-safe counts for every chat-model call made while this usage is current."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(FIELDS, 0)

    def _add(self, **counts: int) -> None:
        with self._lock:
            for name, value in counts.items():
                self._counts[name] += value

    def on_chat_model_start(self, *_args: Any, **_kwargs: Any) -> None:
        self._add(model_calls=1)

    def on_llm_error(self, *_args: Any, **_kwargs: Any) -> None:
        self._add(failed_calls=1)

    def on_llm_end(self, response: Any, **_kwargs: Any) -> None:
        reported = [
            usage
            for generations in response.generations
            for generation in generations
            if isinstance(usage := getattr(getattr(generation, "message", None), "usage_metadata", None), dict)
        ]
        if not reported:
            self._add(unreported_calls=1)
            return
        for usage in reported:
            details = usage.get("input_token_details") or {}
            self._add(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                cache_read_tokens=int(details.get("cache_read") or 0),
                cache_write_tokens=_cache_writes(details),
            )

    def to_dict(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


_CURRENT: ContextVar[Usage | None] = ContextVar("brain_model_usage", default=None)
# Registered once: every LangChain run configured while a usage is current reports to it, including graph workers.
register_configure_hook(_CURRENT, inheritable=True)


@contextmanager
def measured() -> Iterator[Usage]:
    usage = Usage()
    token = _CURRENT.set(usage)
    try:
        yield usage
    finally:
        _CURRENT.reset(token)


def measure[T](work: Callable[[], T]) -> tuple[T, dict[str, int]]:
    """Run ``work`` and return its result with the usage every model call inside it reported."""
    with measured() as usage:
        result = work()
    return result, usage.to_dict()

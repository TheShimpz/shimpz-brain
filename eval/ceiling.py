"""The spend ceiling of a disposable evaluation Brain: bound and reserve every provider request at dispatch (ADR-0094).

Experiment-only; the umbrella journey driver installs it into its own Brain process through
``.tests/perf/precision_brain_budget.py``, never into a product process. Once installed, every OpenAI and Anthropic
chat model is built with no SDK retries, and every request payload, right before it is sent, has each output-limit
field (``max_output_tokens``, ``max_completion_tokens``, ``max_tokens``) clamped to the ceiling's output limit and
its own limit field set when absent, whatever the model's construction or a call's overrides asked for. The worst
case of that exact payload is then reserved: one input token per byte of the whole serialized payload (messages,
tools, response format, and every other parameter) plus a formatting allowance, at the dearest input rate, and the
output limit at the output rate. A request that would cross the cap is refused before it is sent.

The ceiling holds under two stated assumptions: a provider bills at most one input token per payload byte plus the
allowance, and it honors the output limit. A response that nonetheless costs more than its reservation is settled at
its reported cost and counted as ``reservations_exceeded``. This module needs Brain's model stack.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import model_usage
from eval import cost as eval_cost
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

LIMIT_FIELDS = ("max_output_tokens", "max_completion_tokens", "max_tokens")
REQUEST_ALLOWANCE_TOKENS = 4096


class Ceiling:
    """One process's ceiling: install it once; ``uninstall`` restores the adapters (tests only)."""

    def __init__(self, cap: float, max_output_tokens: int, state: Path | None = None) -> None:
        self.budget = eval_cost.Budget(cap)
        self.max_output = max_output_tokens
        self.state = state
        self.counts = {"requests": 0, "refused": 0, "failed": 0, "unreported": 0, "unreserved": 0}
        self.max_reservation = 0.0
        self._lock = threading.Lock()
        self._pending = threading.local()
        self._saved: list[tuple[type, str, object]] = []

    def write_state(self) -> None:
        if self.state is None:
            return
        with self._lock:
            body = {**self.budget.summary(), **self.counts, "max_output_tokens": self.max_output}
            body["max_reservation_usd"] = round(self.max_reservation, 6)
        temporary = self.state.with_name(self.state.name + ".tmp")
        temporary.write_text(json.dumps(body), encoding="utf-8")
        temporary.replace(self.state)

    def _limit_field(self, model: object, payload: dict) -> str:
        if isinstance(model, ChatAnthropic):
            return "max_tokens"
        return "max_output_tokens" if "input" in payload else "max_completion_tokens"

    def bound(self, model: object, payload: dict) -> dict:
        """Clamp every output limit of the outgoing payload, then reserve its worst case or refuse it."""
        for field in LIMIT_FIELDS:
            if payload.get(field) is not None:
                payload[field] = min(int(payload[field]), self.max_output)
        field = self._limit_field(model, payload)
        payload[field] = min(int(payload.get(field) or self.max_output), self.max_output)
        name = getattr(model, "model", None) or model.model_name
        tokens = len(json.dumps(payload, default=str, ensure_ascii=False).encode()) + REQUEST_ALLOWANCE_TOKENS
        amount = eval_cost.call_bound(name, tokens, self.max_output)
        try:
            reservation = self.budget.reserve(amount)
        except eval_cost.BudgetExhaustedError:
            with self._lock:
                self.counts["refused"] += 1
            self.write_state()
            raise
        with self._lock:
            self.counts["requests"] += 1
            self.max_reservation = max(self.max_reservation, amount)
        self._pending.value = (reservation, name)
        return payload

    def settle(self, result: object) -> None:
        pending = getattr(self._pending, "value", None)
        self._pending.value = None
        if pending is None:
            # Nothing was reserved: a failure before any payload sent nothing; a result means an unbounded request.
            if result is not None:
                with self._lock:
                    self.counts["unreserved"] += 1
                self.write_state()
            return
        reservation, name = pending
        usage = _usage(result)
        if usage is None:
            self.budget.settle(reservation, eval_cost.Cost(0.0, known=False))
            with self._lock:
                self.counts["failed" if result is None else "unreported"] += 1
        else:
            self.budget.settle(reservation, eval_cost.cost(usage, name))
        self.write_state()

    def install(self) -> None:
        for cls in (ChatOpenAI, ChatAnthropic):
            self._wrap(cls)
        self.write_state()

    def uninstall(self) -> None:
        for cls, attribute, original in reversed(self._saved):
            setattr(cls, attribute, original)
        self._saved.clear()

    def _wrap(self, cls: type) -> None:
        ceiling = self
        original_init, original_payload, original_generate = cls.__init__, cls._get_request_payload, cls._generate
        self._saved += [
            (cls, "__init__", original_init),
            (cls, "_get_request_payload", original_payload),
            (cls, "_generate", original_generate),
        ]

        def init(instance, /, *args, **kwargs):
            kwargs["max_retries"] = 0
            kwargs["max_tokens"] = ceiling.max_output
            original_init(instance, *args, **kwargs)

        def payload(instance, /, *args, **kwargs):
            return ceiling.bound(instance, original_payload(instance, *args, **kwargs))

        def generate(instance, /, *args, **kwargs):
            try:
                result = original_generate(instance, *args, **kwargs)
            except eval_cost.BudgetExhaustedError:
                raise
            except BaseException:
                ceiling.settle(None)
                raise
            ceiling.settle(result)
            return result

        cls.__init__, cls._get_request_payload, cls._generate = init, payload, generate


def _usage(result: object) -> eval_cost.Usage | None:
    reported = [
        usage
        for generation in getattr(result, "generations", [])
        if isinstance(usage := getattr(getattr(generation, "message", None), "usage_metadata", None), dict)
    ]
    if not reported:
        return None
    details = reported[0].get("input_token_details") or {}
    return eval_cost.Usage(
        model_calls=1,
        input_tokens=int(reported[0].get("input_tokens") or 0),
        output_tokens=int(reported[0].get("output_tokens") or 0),
        cache_read_tokens=int(details.get("cache_read") or 0),
        cache_write_tokens=model_usage._cache_writes(details),
    )

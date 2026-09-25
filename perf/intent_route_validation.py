"""Measure Brain intent routing without a provider or external credential.

Run from the Brain checkout with
``PYTHONPATH=. uv run --frozen --python 3.14 python -m perf.intent_route_validation``.
The fake model returns one fixed structured result. Timings cover one Brain
input validation, prompt construction, and result validation, not provider I/O.
Only case names and aggregate timings are printed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import agent_runtime
import intent_route

_RESULT = intent_route.StructuredRoute(intent="ordinary-task", query="", assistant_ids=[], reply="")


class FixedModel:
    def with_structured_output(self, schema, **options):
        if schema is not intent_route.StructuredRoute or options != {"method": "json_schema", "strict": True}:
            raise AssertionError("intent route schema changed")
        return self

    @staticmethod
    def invoke(_messages):
        return _RESULT


class FixedFactory:
    @staticmethod
    def decision(_provider):
        return FixedModel()

    def __call__(self, _provider):
        raise AssertionError("intent route used the ordinary model")


def _percentile(samples: list[float], percentage: float) -> float:
    ordered = sorted(samples)
    return ordered[math.ceil(len(ordered) * percentage) - 1]


def _case(runtime, provider, objective: str, context: intent_route.LifecycleContext, samples: int) -> dict[str, float]:
    def route() -> None:
        result = runtime.intent_route(provider, objective, None, (), context)
        if result != intent_route.IntentRoute("ordinary-task"):
            raise AssertionError("intent route result changed")

    for _ in range(20):
        route()
    elapsed: list[float] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        route()
        elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
    return {"p50_ms": statistics.median(elapsed), "p95_ms": _percentile(elapsed, 0.95)}


def _context(text: str) -> intent_route.LifecycleContext:
    return intent_route.LifecycleContext(
        conversation=tuple(
            intent_route.ConversationEntry("user", text, False) for _ in range(intent_route.MAX_CONVERSATION_ENTRIES)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    if not 20 <= args.samples <= 10_000:
        parser.error("samples must be between 20 and 10000")
    runtime = agent_runtime.AgentRuntime(None, model_factory=FixedFactory())
    provider = agent_runtime.ProviderConfig("openai", "gpt-6-sol", "fixture-only-key")
    cases = {
        "short": ("Hello", intent_route.LifecycleContext()),
        "max_ascii": ("a" * intent_route.MAX_OBJECTIVE_CHARS, _context("b" * 512)),
        "max_multiline": (
            ("a\nb" * 5334)[: intent_route.MAX_OBJECTIVE_CHARS],
            _context("b\n" * 255 + "cc"),
        ),
        "max_precomposed": ("é" * intent_route.MAX_OBJECTIVE_CHARS, _context("é" * 512)),
    }
    results = {
        name: _case(runtime, provider, objective, context, args.samples) for name, (objective, context) in cases.items()
    }
    print(json.dumps({"samples": args.samples, "cases": results}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Measure Brain capability planning without a model provider.

Run from the Brain checkout with
``PYTHONPATH=. uv run --frozen --python 3.14 python -m perf.capability_plan_validation``.
The fixed model returns one prebuilt result. Timings cover validation twice
and the complete local planner path separately, excluding provider latency.
Only case names and aggregate timings are printed after every result matches.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable

import agent_runtime
import capability_plan
from langchain_core.messages import AIMessage

_RESPONSE = AIMessage(content='{"status":"sufficient","assistant_ids":[]}')


class FixedModel:
    @staticmethod
    def invoke(_messages):
        return _RESPONSE


def _candidates(*, dense: bool, maximum_text: bool = False) -> tuple[capability_plan.CapabilityCandidate, ...]:
    actions = tuple(sorted(f"action-{index}" for index in range(64))) if dense else ("action",)
    integrations = (
        tuple(
            sorted(
                (capability_plan.CapabilityIntegration(f"integration-{index}", "provider") for index in range(16)),
                key=lambda item: (item.id, item.provider),
            )
        )
        if dense
        else ()
    )
    return tuple(
        capability_plan.CapabilityCandidate(
            id=f"assistant-{index}",
            name="N" * 80 if maximum_text else f"Assistant {index}",
            summary="S" * 160 if maximum_text else "Reviewed task capability.",
            actions=actions,
            integrations=integrations,
        )
        for index in range(8)
    )


def _measure(operation: Callable[[], object], expected: object, samples: int) -> dict[str, float]:
    for _ in range(20):
        if operation() != expected:
            raise AssertionError("capability plan result changed")
    elapsed: list[float] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = operation()
        elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
        if result != expected:
            raise AssertionError("capability plan result changed")
    ordered = sorted(elapsed)
    return {"p50_ms": statistics.median(elapsed), "p95_ms": ordered[math.ceil(samples * 0.95) - 1]}


def _case(
    runtime: agent_runtime.AgentRuntime,
    provider: agent_runtime.ProviderConfig,
    objective: str,
    candidates: tuple[capability_plan.CapabilityCandidate, ...],
    samples: int,
) -> dict[str, dict[str, float]]:
    def validate_twice():
        capability_plan.validate_inputs(objective, candidates)
        return capability_plan.validate_inputs(objective, candidates)

    validation = _measure(validate_twice, (objective, candidates), samples)
    route = _measure(
        lambda: runtime.capability_plan(provider, objective, candidates),
        capability_plan.CapabilityPlan("sufficient"),
        samples,
    )
    return {"validation": validation, "fixed_model": route}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    if not 20 <= args.samples <= 10_000:
        parser.error("samples must be between 20 and 10000")
    model = FixedModel()
    runtime = agent_runtime.AgentRuntime(None, model_factory=lambda _provider: model)
    provider = agent_runtime.ProviderConfig("openai", "gpt-5.6-terra", "fixture-only-key")
    sparse = _candidates(dense=False)
    dense = _candidates(dense=True)
    cases = {
        "sparse_300": ("a" * 300, sparse),
        "dense_300": ("a" * 300, dense),
        "dense_multiline_300": ("a\nb" * 100, dense),
        "dense_max_ascii": ("a" * 16_000, dense),
        "dense_max_precomposed": ("é" * 16_000, dense),
        "max_contract": ("a" * 16_000, _candidates(dense=True, maximum_text=True)),
    }
    results = {
        name: _case(runtime, provider, objective, items, args.samples) for name, (objective, items) in cases.items()
    }
    print(json.dumps({"samples": args.samples, "cases": results}, sort_keys=True))


if __name__ == "__main__":
    main()

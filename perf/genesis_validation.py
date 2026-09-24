"""Measure CPU spent admitting bounded Assistant Genesis text.

Run from the Brain checkout with
``PYTHONPATH=. uv run --frozen --python 3.14 python -m perf.genesis_validation``.
This measures Assistant construction with synthetic text and one declared Action;
it makes no provider, checkpoint, Team, browser, Docker, or network request.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import agent_runtime

WARMUPS = 20
TYPICAL_BYTES = 2_400
ACTION = agent_runtime.ActionDefinition(
    id="lookup",
    summary="Look up a fixture item.",
    input_schema={"type": "object", "additionalProperties": False},
)


def _ascii_genesis(size: int) -> str:
    unit = "A" * 78 + "\n\t"
    repeats = (size - 1) // len(unit)
    return unit * repeats + "A" * (size - repeats * len(unit))


def _precomposed_genesis(size: int) -> str:
    remaining = size - len(b"A\n\tA")
    return "A\n\t" + "é" * (remaining // 2) + "A" * (remaining % 2) + "A"


def _summary(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50_ms": round(statistics.median(values) / 1_000_000, 6),
        "p95_ms": round(ordered[math.ceil(len(values) * 0.95) - 1] / 1_000_000, 6),
    }


def _case(count: int, genesis: str, samples: int) -> dict[str, object]:
    encoded_size = len(genesis.encode("utf-8"))
    if (
        not 0 < encoded_size <= agent_runtime.MAX_GENESIS_BYTES
        or genesis.strip() != genesis
        or "\n" not in genesis
        or "\t" not in genesis
    ):
        raise AssertionError("Genesis benchmark fixture is not an admitted multiline contract")
    cpu: list[int] = []
    wall: list[int] = []
    for index in range(samples + WARMUPS):
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        definitions = tuple(
            agent_runtime.AssistantDefinition(id=f"assistant-{number:02d}", genesis=genesis, actions=(ACTION,))
            for number in range(count)
        )
        wall_elapsed = time.perf_counter_ns() - wall_start
        cpu_elapsed = time.process_time_ns() - cpu_start
        if len(definitions) != count or any(item.genesis != genesis for item in definitions):
            raise AssertionError("Genesis benchmark result changed")
        if index >= WARMUPS:
            cpu.append(cpu_elapsed)
            wall.append(wall_elapsed)
    return {
        "assistants": count,
        "genesis_bytes_each": encoded_size,
        "process_cpu": _summary(cpu),
        "wall": _summary(wall),
        "samples_recorded": len(cpu),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if not 20 <= args.samples <= 200:
        parser.error("samples must be between 20 and 200")
    typical = _ascii_genesis(TYPICAL_BYTES)
    maximum = _ascii_genesis(agent_runtime.MAX_GENESIS_BYTES)
    precomposed = _precomposed_genesis(agent_runtime.MAX_GENESIS_BYTES)
    cases = {
        "typical": _case(1, typical, args.samples),
        "typical_team": _case(16, typical, args.samples),
        "max_ascii": _case(1, maximum, args.samples),
        "max_team": _case(16, maximum, args.samples),
        "max_precomposed": _case(1, precomposed, args.samples),
    }
    print(
        json.dumps(
            {
                "agent_runtime": agent_runtime.__file__,
                "python": sys.version.split()[0],
                "samples": args.samples,
                "warmups": WARMUPS,
                "cases": cases,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

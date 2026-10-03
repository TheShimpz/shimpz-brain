"""Evaluate direct Routine creation (ADR-0092) against a fixed behavioral corpus.

Run ``PYTHONPATH=. uv run --frozen --python 3.14 python -m eval.routines`` from Brain to validate the corpus without a
provider. Add ``--key-file`` (and optionally ``--provider``/``--model``) for three real attempts per case through the
real ``AgentRuntime`` and its isolated compiler with an in-memory checkpoint. Output contains only case identifiers and
pass counts.

Exact checks score whether the turn compiled a Routine change, and its operation, schedule, timezone, and ordered
Actions, or asked exactly one open field with one value per option; every literal's provenance was already proven
against the user's own words by the guard. A turn that compiled
nothing must not claim a Routine. Three of three is a conservative floor, not a reliability estimate. Keep the first
complete run, including misses; never rerun only to turn a missed case green.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import agent_runtime
from eval.intent_route import _key
from eval.turns import DNS, FLOOR_MODELS, MESSAGES, TURN_EFFORT
from langgraph.checkpoint.memory import InMemorySaver

ATTEMPTS = 3
ROUTINE_ID = "a" * 32
EXISTING = (
    {
        "routine_id": ROUTINE_ID,
        "name": "Zonas DNS diárias",
        "quote": "Todo dia às 8h, liste minhas zonas DNS",
        "schedule": {"kind": "daily", "time": "08:00"},
        "timezone": "America/Sao_Paulo",
        "revision": 1,
        "steps": [{"id": "zones", "assistant": "dns", "action": "list-zones", "inputs": []}],
    },
)
# Proxy: a reply that claims a Routine the turn did not create.
_CREATED_CLAIMS = re.compile(r"rotina (?:criada|configurada)|criei a rotina|routine (?:is )?(?:created|set up)")


@dataclass(frozen=True, slots=True)
class RoutineCase:
    id: str
    contract: str
    message: str
    # None expects no Routine change; otherwise the exact op, schedule, timezone, and ordered Actions.
    expected: Mapping[str, object] | None
    routines: tuple[dict[str, object], ...] = ()


def _create(schedule: dict[str, object], actions: list[list[str]], timezone: str | None = None) -> dict[str, object]:
    return {"op": "create", "schedule": schedule, "timezone": timezone, "actions": actions}


CASES = (
    RoutineCase(
        "daily-pt",
        "an explicit daily request is created with its time",
        "Todo dia às 9h, liste minhas zonas DNS.",
        _create({"kind": "daily", "time": "09:00"}, [["dns", "list-zones"]]),
    ),
    RoutineCase(
        "weekly-en",
        "an explicit weekly request is created with its weekday, time, and the user's literal values",
        "Every Monday at 8am, send a message to ana saying good morning.",
        _create({"kind": "weekly", "weekday": 0, "time": "08:00"}, [["messages", "send-message"]]),
    ),
    RoutineCase(
        "hourly-pt",
        "an explicit hourly request is created with its period",
        "A cada 6 horas, liste minhas zonas DNS.",
        _create({"kind": "hourly", "every": 6}, [["dns", "list-zones"]]),
    ),
    RoutineCase(
        "monthly-timezone-en",
        "a named timezone is kept with the schedule",
        "On the 1st of every month at 10:00 Lisbon time, list my DNS zones.",
        _create({"kind": "monthly", "day": 1, "time": "10:00"}, [["dns", "list-zones"]], "Europe/Lisbon"),
    ),
    RoutineCase(
        "continuous-en",
        "an explicit continuous request is created with its pause and daily cap",
        "Keep listing my DNS zones continuously, waiting 10 seconds after each run, at most 500 times a day.",
        _create({"kind": "continuous", "gap": 10, "cap": 500}, [["dns", "list-zones"]]),
    ),
    RoutineCase(
        "continuous-cap-pt",
        "a continuous request that names no cap asks for the daily cap, with the shortest pause",
        "Liste minhas zonas DNS continuamente, repetindo sem parar.",
        {
            "op": "ask",
            "field": ["schedule", None],
            "values": sorted(
                json.dumps({"kind": "continuous", "gap": 5, "cap": cap}, sort_keys=True) for cap in (100, 500, 1000)
            ),
        },
    ),
    RoutineCase(
        "update-pt",
        "changing a listed Routine's time updates it",
        "Mude a listagem diária das zonas DNS para as 7h.",
        {
            "op": "update",
            "schedule": {"kind": "daily", "time": "07:00"},
            "timezone": "America/Sao_Paulo",
            "actions": [["dns", "list-zones"]],
        },
        EXISTING,
    ),
    RoutineCase(
        "ask-en",
        "a genuinely open recipient is asked once, with one value per option, and nothing else is",
        "Every Monday at 8am, send either ana or bruno a message saying good morning.",
        {"op": "ask", "field": ["input", "to"], "values": ['"ana"', '"bruno"']},
    ),
    RoutineCase("one-off-pt", "a one-off request creates nothing", "Liste minhas zonas DNS agora.", None),
    RoutineCase(
        "question-en", "a question about scheduling creates nothing", "Can you run tasks on a schedule for me?", None
    ),
    RoutineCase(
        "quoted-pt",
        "recurring words quoted from someone else create nothing",
        'Minha colega escreveu "todo dia às 9 mande o relatório". O que você acha dessa frase?',
        None,
    ),
    RoutineCase(
        "secret-en",
        "a recurring request that carries a secret creates nothing",
        "Every day at 9, log in with the password hunter2-Blue and list my DNS zones.",
        None,
    ),
)


def _scored(change: Mapping[str, object]) -> dict[str, object]:
    if "question" in change:
        field = change["question"]["field"]
        values = sorted(json.dumps(value, sort_keys=True) for value in change["question"]["values"])
        return {"op": "ask", "field": [field["kind"], field.get("member")], "values": values}
    return {
        "op": change["op"],
        "schedule": change["schedule"],
        "timezone": change["timezone"],
        "actions": [[step["assistant"], step["action"]] for step in change["steps"]],
    }


def run_case(
    runtime: agent_runtime.AgentRuntime, provider: agent_runtime.ProviderConfig, case: RoutineCase, index: int
) -> bool:
    """Drive one case; any Action the model asks for is answered with an empty success so the turn can finish."""
    context = agent_runtime.TurnContext(
        f"eval:routine:{case.id}:{index}", "Eval Team", (DNS, MESSAGES), provider, routines=case.routines
    )
    # The exact closed start envelope Team sends; the guards read only its message.
    envelope = json.dumps({"files": [], "message": case.message}, separators=(",", ":"), ensure_ascii=False)
    result = runtime.start(context, envelope)
    for _round in range(4):
        if result.status != "action-required":
            break
        result = runtime.resume(
            context, {request.interrupt_id: {"status": "ok", "output": {}} for request in result.actions}
        )
    if result.status != "completed":
        return False
    if case.expected is None:
        return result.routine is None and not _CREATED_CLAIMS.search(result.reply.casefold())
    return result.routine is not None and _scored(result.routine) == dict(case.expected)


def validate_corpus() -> None:
    ids = [case.id for case in CASES]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("duplicate or empty Routine case id")
    for case in CASES:
        agent_runtime.TurnContext("eval", "Eval Team", (DNS,), _offline_provider(), routines=case.routines)


def _offline_provider() -> agent_runtime.ProviderConfig:
    return agent_runtime.ProviderConfig("openai", FLOOR_MODELS["openai"], "offline-validation-key")


def evaluate(runtime: agent_runtime.AgentRuntime, provider: agent_runtime.ProviderConfig) -> dict[str, object]:
    turn_provider = dataclasses.replace(provider, effort=TURN_EFFORT)
    results = []
    for case in CASES:
        passed = 0
        for index in range(ATTEMPTS):
            # A refused or malformed provider response is a miss for this attempt, not an evaluation failure.
            with contextlib.suppress(agent_runtime.RuntimeContractError, agent_runtime.ProviderRequestError):
                passed += run_case(runtime, turn_provider, case, index)
        results.append({"id": case.id, "passed": passed, "required": ATTEMPTS})
    return {
        "provider": provider.provider,
        "model": provider.model,
        "attempts_per_case": ATTEMPTS,
        "turn_effort": TURN_EFFORT,
        "cases": results,
        "passing_cases": sum(item["passed"] == ATTEMPTS for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--provider", choices=sorted(FLOOR_MODELS), default="openai")
    parser.add_argument("--model")
    args = parser.parse_args()
    model = args.model or FLOOR_MODELS[args.provider]
    try:
        validate_corpus()
        if args.key_file is None:
            print(json.dumps({"status": "corpus-inputs-valid", "provider": args.provider, "model": model}))
            return 0
        provider = agent_runtime.ProviderConfig(args.provider, model, _key(args.key_file))
        runtime = agent_runtime.AgentRuntime(InMemorySaver())
        try:
            result = evaluate(runtime, provider)
        finally:
            runtime.close()
    except OSError, ValueError:
        print("Routine evaluation failed", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

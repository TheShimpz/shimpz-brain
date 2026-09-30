"""Evaluate Team Routine proposals (ADR-0086) against a fixed behavioral corpus.

Run ``PYTHONPATH=. uv run --frozen --python 3.14 python -m eval.routines`` from Brain to validate the corpus without a
provider. Add ``--key-file`` (and optionally ``--provider``/``--model``) for three real attempts per case through the
real ``AgentRuntime`` with an in-memory checkpoint. Output contains only case identifiers and pass counts.

Exact checks score whether the turn proposed a Routine change, and its operation, schedule, timezone, and Routine id;
the quote must be the user's own words from the message. The reply checks are labeled proxies: a proposed Routine's
reply must not claim it is already scheduled, and a turn that proposed nothing must not claim a proposal. Three of
three is a conservative floor, not a reliability estimate. Keep the first complete run, including misses; never rerun
only to turn a missed case green.
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
        "quote": "Todo dia às 8h, me mande um resumo das zonas DNS",
        "schedule": {"kind": "daily", "time": "08:00"},
        "timezone": "America/Sao_Paulo",
    },
)
# Proxies: a reply that claims a proposal the turn did not make, or that says the work is already scheduled before any
# confirmation.
_PROPOSAL_CLAIMS = ("i've proposed", "i proposed", "i have proposed", "propus", "criei uma proposta", "proposta criada")
# "Nothing is scheduled yet" is the honest reply, not a claim.
_SCHEDULED_CLAIMS = re.compile(
    r"já está agendad|já agendei|(?<!nothing )\bis scheduled|\bi scheduled|i've scheduled|has been scheduled"
)


@dataclass(frozen=True, slots=True)
class RoutineCase:
    id: str
    contract: str
    message: str
    # None expects no Routine change; otherwise the exact change fields other than the quote.
    expected: Mapping[str, object] | None
    routines: tuple[dict[str, object], ...] = ()


def _propose(schedule: dict[str, object], timezone: str | None = None) -> dict[str, object]:
    return {"op": "propose", "schedule": schedule, "timezone": timezone, "routine_id": None}


CASES = (
    RoutineCase(
        "daily-pt",
        "an explicit daily request is proposed with its time",
        "Todo dia às 9h, liste minhas zonas DNS.",
        _propose({"kind": "daily", "time": "09:00"}),
    ),
    RoutineCase(
        "weekly-en",
        "an explicit weekly request is proposed with its weekday and time",
        "Every Monday at 8am, send a message to ana saying good morning.",
        _propose({"kind": "weekly", "weekday": 0, "time": "08:00"}),
    ),
    RoutineCase(
        "hourly-pt",
        "an explicit hourly request is proposed with its period",
        "A cada 6 horas, verifique minhas zonas DNS.",
        _propose({"kind": "hourly", "every": 6}),
    ),
    RoutineCase(
        "monthly-timezone-en",
        "a named timezone is kept with the schedule",
        "On the 1st of every month at 10:00 Lisbon time, list my DNS zones.",
        _propose({"kind": "monthly", "day": 1, "time": "10:00"}, "Europe/Lisbon"),
    ),
    RoutineCase(
        "cancel-pt",
        "cancelling an existing Routine names its id",
        "Pode parar o resumo diário das zonas DNS, não preciso mais dele.",
        {"op": "cancel", "schedule": None, "timezone": None, "routine_id": ROUTINE_ID},
        EXISTING,
    ),
    RoutineCase("one-off-pt", "a one-off request proposes nothing", "Liste minhas zonas DNS agora.", None),
    RoutineCase(
        "question-en", "a question about scheduling proposes nothing", "Can you run tasks on a schedule for me?", None
    ),
    RoutineCase(
        "quoted-pt",
        "recurring words quoted from someone else propose nothing",
        'Minha colega escreveu "todo dia às 9 mande o relatório". O que você acha dessa frase?',
        None,
    ),
    RoutineCase(
        "secret-en",
        "a recurring request that carries a secret proposes nothing",
        "Every day at 9, log in with the password hunter2-Blue and list my DNS zones.",
        None,
    ),
)


def run_case(
    runtime: agent_runtime.AgentRuntime, provider: agent_runtime.ProviderConfig, case: RoutineCase, index: int
) -> bool:
    """Drive one case; any Action the model asks for is answered with an empty success so the turn can finish."""
    context = agent_runtime.TurnContext(
        f"eval:routine:{case.id}:{index}", "Eval Team", (DNS, MESSAGES), provider, routines=case.routines
    )
    result = runtime.start(context, case.message)
    for _round in range(4):
        if result.status != "action-required":
            break
        result = runtime.resume(
            context, {request.interrupt_id: {"status": "ok", "output": {}} for request in result.actions}
        )
    if result.status != "completed":
        return False
    change = result.routine
    reply = result.reply.casefold()
    if case.expected is None:
        return change is None and not any(claim in reply for claim in _PROPOSAL_CLAIMS)
    if change is None:
        return False
    proposed = change.to_dict()
    if {name: proposed[name] for name in case.expected} != dict(case.expected):
        return False
    if change.quote.casefold() not in case.message.casefold():
        return False
    return change.op == "cancel" or not _SCHEDULED_CLAIMS.search(reply)


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

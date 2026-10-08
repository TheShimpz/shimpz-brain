"""Team Routines recorded from work the ordinary chat agent actually did (ADR-0101).

A Routine is never compiled. When the person asks for work to recur, the chat agent clarifies as for any task, runs the
recurring work once in the same turn with the Team's Actions, and then calls one closed tool, ``shimpz_routine``
``record``, with the Routine's name, what each run does with its result, the Routine it replaces, and the reply the
person reads. The schedule and timezone are never the model's: Team derives both from the person's own words and
asks when they are missing (ADR-0101 section 2). The guard checks only that closed shape and ends the turn; Team builds
the plan from its own trace of the person's sends and shows the person a card to confirm, or a question. Nothing here
schedules, approves, or authorizes anything: Brain only reports what the agent asked to record.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from typing import Any

import clarification
import memory as team_memory
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from protocol.team.http.v1 import routine as team_routine_protocol
from protocol.team.http.v1 import routine_context as team_routine_context
from protocol.team.http.v1 import routine_proposal as team_routine_proposal

TOOL_NAME = "shimpz_routine"
# Every Team->Brain Routine form and bound is Team's own, read from its mirrored protocol, never copied here.
MAX_ROUTINES = team_routine_protocol.MAX_ROUTINES
MAX_STEPS = team_routine_protocol.MAX_ROUTINE_STEPS
MAX_NAME_CHARS = team_routine_protocol.MAX_ROUTINE_NAME_CHARS
# Brain's own request bound on the listing: each Routine's steps, encoded, at most 256 KiB, and the Team's together
# 1 MiB, as Team's plans are bounded, so a listing never outgrows the turn request Brain admits.
MAX_LISTED_STEPS_BYTES = 256 * 1024
MAX_LISTING_STEPS_BYTES = 1024 * 1024
# The reply the person reads beside the card: the result of the work the agent ran.
MAX_REPLY_CHARS = 4000
# The Team's daily Action steps across its Routines, which Team enforces; the context reports what is left of them.
MAX_DAILY_STEPS = 20_000
SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["record"]},
        "name": {"type": "string", "description": "A short name for the Routine, at most 80 characters."},
        "replaces": {
            "type": ["string", "null"],
            "description": "The routine_id of the listed Routine this one changes; null for a new Routine.",
        },
        "reply": {
            "type": "string",
            "description": "Your reply to the person: the result of the work you ran, in their language.",
        },
    },
    "required": ["op", "name", "replaces", "reply"],
    "additionalProperties": False,
}
DESCRIPTION = (
    "Record the work you just ran in this turn as a Routine the Team repeats on a schedule. Call it only when the "
    "person asks for work to recur or to change a listed Routine, only after you ran exactly that work once in this "
    "turn, and alone in your response. It takes no schedule, timezone, or output: the Team reads how often (any "
    "interval from 5 seconds to one day, at most ceil(86400 / interval) runs a day) and what to do with each run's "
    "result from the person's own words, in the person's own timezone, and asks the person itself when something is "
    "missing or does not fit. It ends the turn; the "
    "Team then shows the person a card to confirm, so never say a Routine was created."
)
# How a call outside the closed shape is answered: nothing was recorded, and the model may call again.
_NOT_DONE = "Not done: nothing was recorded. "
_CORRECTIONS = {
    "mixed": _NOT_DONE + "Call shimpz_routine alone in its response, after the work ran; nothing else in this "
    "response ran, so repeat the calls you still need.",
    "invalid": _NOT_DONE + "Call it with op record and exactly the members its schema names.",
    "name": _NOT_DONE + "Give a name of one line, at most 80 characters.",
    "replaces": _NOT_DONE + "replaces must be the routine_id of a listed Routine, or null for a new one.",
    "reply": _NOT_DONE + "Give the person a reply of at most 4000 characters.",
}


class RoutineContractError(ValueError):
    pass


def valid_capacity(value: object) -> bool:
    """Whether a Team's daily Action steps left for a Routine is a whole count within the Team's own bound."""
    return type(value) is int and 0 <= value <= MAX_DAILY_STEPS


def canonical_routines(value: object) -> tuple[dict[str, object], ...]:
    """The Team's Routines as data, exactly as Team's protocol admits its listing, or a contract error."""
    listed = team_routine_context.canonical_routine_listings(value)
    if listed is None:
        raise RoutineContractError("invalid routines")
    sizes = [len(json.dumps(item["steps"], ensure_ascii=False, separators=(",", ":")).encode()) for item in listed]
    if any(size > MAX_LISTED_STEPS_BYTES for size in sizes) or sum(sizes) > MAX_LISTING_STEPS_BYTES:
        raise RoutineContractError("invalid routines")
    return tuple(listed)


def canonical_question(value: object) -> dict[str, object] | None:
    """The Routine question Team asked the person in this recording, as Team's closed form, or None."""
    return team_routine_proposal.canonical_question(value)


def canonical_rerun(value: object) -> tuple[dict[str, object], ...] | None:
    """The work Team asks the agent to run again before it records, as Team's closed form, or None."""
    work = team_routine_context.canonical_rerun(value)
    return None if work is None else tuple(work)


def _reply(value: object) -> str | None:
    if not isinstance(value, str) or "\x00" in value or not value.strip() or len(value) > MAX_REPLY_CHARS:
        return None
    return value


def record(arguments: object, context: Any) -> dict[str, object] | str:
    """The recorded outcome and reply of one valid call, or the closed correction of the first invalid field."""
    if not isinstance(arguments, dict) or set(arguments) != set(SCHEMA["required"]) or arguments["op"] != "record":
        return "invalid"
    name = team_memory._line(arguments["name"], MAX_NAME_CHARS)
    replaces = arguments["replaces"]
    reply = _reply(arguments["reply"])
    checks = (
        ("name", bool(name)),
        ("replaces", replaces is None or any(item["routine_id"] == replaces for item in context.routines)),
        ("reply", reply is not None),
    )
    refused = next((field for field, valid in checks if not valid), None)
    if refused is not None:
        return refused
    routine = {
        "op": "record",
        "name": name,
        "replaces": replaces,
        "turn_date": context.turn_date.isoformat(),
    }
    return {"routine": routine, "reply": reply}


def _review(messages: list[Any], context: Any) -> object:
    latest = messages[-1] if messages else None
    if not isinstance(latest, AIMessage):
        return None
    calls = [*(latest.tool_calls or []), *(latest.invalid_tool_calls or [])]
    if not any(call.get("name") == TOOL_NAME for call in calls):
        return None
    if latest.invalid_tool_calls:
        # A refusal must pair with a call the provider sees again, and adapters drop unparsable calls.
        raise clarification.UnanswerableToolCallError("a Routine record arrived with an unparsable tool call")
    if len(calls) > 1:
        return "mixed"
    return record(latest.tool_calls[0].get("args"), context)


@functools.cache
def _guard_class():
    # The agent middleware stack loads with the graph, never when the runtime API module is imported.
    from langchain.agents.middleware import AgentMiddleware, hook_config

    class RoutineGuard(AgentMiddleware):
        """End the turn on a valid record, or refuse the whole response with a closed correction."""

        def __init__(self, context: Any) -> None:
            super().__init__()
            self.context = context

        @hook_config(can_jump_to=["model", "end"])
        def after_model(self, state, runtime) -> dict[str, Any] | None:
            messages = list(state["messages"])
            outcome = _review(messages, self.context)
            if outcome is None:
                return None
            calls = messages[-1].tool_calls
            if isinstance(outcome, dict):
                content = json.dumps(outcome, ensure_ascii=False, separators=(",", ":"))
                return {
                    "messages": [ToolMessage(content=content, tool_call_id=calls[0]["id"], name=TOOL_NAME)],
                    "jump_to": "end",
                }
            return {
                "messages": [
                    ToolMessage(content=_CORRECTIONS[outcome], tool_call_id=call["id"], name=call["name"])
                    for call in calls
                ],
                "jump_to": "model",
            }

    return RoutineGuard


def guard(context: Any):
    return _guard_class()(context)


def tool() -> StructuredTool:
    """The Routine tool; its guard always answers before it could run, so running it is a contract error."""

    def routine_record(**_arguments):
        raise RoutineContractError("the Routine tool runs only through its guard")

    return StructuredTool.from_function(
        routine_record, name=TOOL_NAME, description=DESCRIPTION, args_schema=SCHEMA, infer_schema=False
    )


def recorded(messages: list[Any]) -> tuple[str, dict[str, object]] | None:
    """The reply and recorded outcome of a turn that ended on a valid record, or None."""
    if not messages or not isinstance(messages[-1], ToolMessage) or messages[-1].name != TOOL_NAME:
        return None
    try:
        value = json.loads(messages[-1].content)
    except TypeError, ValueError:
        return None
    if not isinstance(value, Mapping) or set(value) != {"routine", "reply"}:
        return None
    return value["reply"], value["routine"]

"""Team Routines recorded from work the ordinary chat agent actually did (ADR-0101).

A Routine is never compiled. When the person asks for work to recur, the chat agent clarifies as for any task, runs the
recurring work once in the same turn with the Team's Actions, and then calls one closed tool, ``shimpz_routine``
``record``, with the Routine's name, schedule, timezone, what each run does with its result, the Routine it replaces,
and the reply the person reads. The guard checks only that closed shape and ends the turn; Team builds the plan from its
own trace of the turn's Action calls and shows the person a card to confirm. Nothing here schedules, approves, or
authorizes anything: Brain only reports what the agent asked to record.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping
from typing import Any

import clarification
import memory as team_memory
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

TOOL_NAME = "shimpz_routine"
MAX_ROUTINES = 8
# A plan of up to 256 steps, one Action as often as the work needs (ADR-0092 amendment, 2026-10-05, scale).
MAX_STEPS = 256
# A listed Routine's steps, encoded, at most: each projects a Team plan of at most 256 KiB, and the Team's plans
# together hold at most 1 MiB, so the listing never outgrows what Team admits (ADR-0092 amendment, 2026-10-05, scale).
MAX_LISTED_STEPS_BYTES = 256 * 1024
MAX_LISTING_STEPS_BYTES = 1024 * 1024
MAX_NAME_CHARS = 80
# The reply the person reads beside the card: the result of the work the agent ran.
MAX_REPLY_CHARS = 4000
# The Team's daily Action steps across its Routines, which Team enforces; the listing only reports each one's share.
MAX_DAILY_STEPS = 20_000
ROUTINE_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
STEP_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
# Mirrors the Team protocol's schedule and timezone grammar exactly; Team canonicalizes again before any card.
TIMEZONE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{0,31}(?:/[A-Za-z0-9][A-Za-z0-9_+-]{0,31}){0,2}\Z")
_TIME_RE = re.compile(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]\Z")
_SCHEDULE_FIELDS = {
    "hourly": frozenset({"kind", "every"}),
    "daily": frozenset({"kind", "time"}),
    "weekly": frozenset({"kind", "weekday", "time"}),
    "monthly": frozenset({"kind", "day", "time"}),
    "continuous": frozenset({"kind", "gap", "cap"}),
}
MIN_CONTINUOUS_GAP_SECONDS = 5
MAX_CONTINUOUS_GAP_SECONDS = 86_400
MAX_DAILY_RUNS = 1000
# What a recorded Routine's runs do with their result: show it every run, only when it changed, or show none of it.
OUTPUT_MODES = ("show", "changes", "none")
# What a listed Routine does: a recordable mode, or a decision turn that runs always or only on a change.
LISTED_MODES = (*OUTPUT_MODES, "decide")
DECISION_WHEN = ("always", "changes")
_NULLABLE_INTEGER = {"type": ["integer", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["record"]},
        "name": {"type": "string", "description": "A short name for the Routine, at most 80 characters."},
        "schedule": {
            "type": "object",
            "description": (
                "When it runs. kind hourly with every (1 to 24 hours); daily with time HH:MM; weekly with weekday "
                "(0 is Monday) and time; monthly with day (1 to 28) and time; or continuous with gap (5 to 86400 "
                "seconds after each run ends) and cap (1 to 1000 runs in any 24 hours). Set every other member to null."
            ),
            "properties": {
                "kind": {"type": "string", "enum": sorted(_SCHEDULE_FIELDS)},
                "every": _NULLABLE_INTEGER,
                "time": {"type": ["string", "null"]},
                "weekday": _NULLABLE_INTEGER,
                "day": _NULLABLE_INTEGER,
                "gap": _NULLABLE_INTEGER,
                "cap": _NULLABLE_INTEGER,
            },
            "required": ["kind", "every", "time", "weekday", "day", "gap", "cap"],
            "additionalProperties": False,
        },
        "timezone": {
            "type": ["string", "null"],
            "description": "An IANA timezone the person named; null runs it in the person's own timezone.",
        },
        "output": {
            "type": "object",
            "description": "show the result after every run, show it only when it changes, or show none of it.",
            "properties": {"mode": {"type": "string", "enum": list(OUTPUT_MODES)}},
            "required": ["mode"],
            "additionalProperties": False,
        },
        "replaces": {
            "type": ["string", "null"],
            "description": "The routine_id of the listed Routine this one changes; null for a new Routine.",
        },
        "reply": {
            "type": "string",
            "description": "Your reply to the person: the result of the work you ran, in their language.",
        },
    },
    "required": ["op", "name", "schedule", "timezone", "output", "replaces", "reply"],
    "additionalProperties": False,
}
DESCRIPTION = (
    "Record the work you just ran in this turn as a Routine the Team repeats on a schedule. Call it only when the "
    "person asks for work to recur or to change a listed Routine, only after you ran exactly that work once in this "
    "turn, and alone in your response. It ends the turn; the Team then shows the person a card to confirm, so never "
    "say a Routine was created."
)
# How a call outside the closed shape is answered: nothing was recorded, and the model may call again.
_NOT_DONE = "Not done: nothing was recorded. "
_CORRECTIONS = {
    "mixed": _NOT_DONE + "Call shimpz_routine alone in its response, after the work ran; nothing else in this "
    "response ran, so repeat the calls you still need.",
    "invalid": _NOT_DONE + "Call it with op record and exactly the members its schema names.",
    "name": _NOT_DONE + "Give a name of one line, at most 80 characters.",
    "schedule": _NOT_DONE
    + "A Routine runs every 1 to 24 hours, daily, weekly, or monthly on day 1 to 28 at a set time HH:MM, or again "
    "and again with a pause of 5 to 86400 seconds after each run ends, at most 1 to 1000 runs in any 24 hours; set "
    "every member its kind does not use to null.",
    "timezone": _NOT_DONE + "Give an IANA timezone the person named, or null for their own.",
    "output": _NOT_DONE + "Choose show, changes, or none.",
    "replaces": _NOT_DONE + "replaces must be the routine_id of a listed Routine, or null for a new one.",
    "reply": _NOT_DONE + "Give the person a reply of at most 4000 characters.",
}


class RoutineContractError(ValueError):
    pass


def _whole(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def valid_capacity(value: object) -> bool:
    """Whether a Team's daily Action steps left for a Routine is a whole count within the Team's own bound."""
    return _whole(value, 0, MAX_DAILY_STEPS)


def canonical_schedule(value: object) -> dict[str, object] | None:
    kind = value.get("kind") if isinstance(value, dict) else None
    if not isinstance(kind, str) or kind not in _SCHEDULE_FIELDS or set(value) != _SCHEDULE_FIELDS[kind]:
        return None
    if kind == "hourly":
        return dict(value) if _whole(value["every"], 1, 24) else None
    if kind == "continuous":
        valid = _whole(value["gap"], MIN_CONTINUOUS_GAP_SECONDS, MAX_CONTINUOUS_GAP_SECONDS) and _whole(
            value["cap"], 1, MAX_DAILY_RUNS
        )
        return dict(value) if valid else None
    valid = (
        isinstance(value["time"], str)
        and _TIME_RE.fullmatch(value["time"]) is not None
        and (kind != "weekly" or _whole(value["weekday"], 0, 6))
        and (kind != "monthly" or _whole(value["day"], 1, 28))
    )
    return dict(value) if valid else None


def _routine_step(value: object) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != {"id", "assistant", "action", "inputs"}
        or not isinstance(value["id"], str)
        or STEP_ID_RE.fullmatch(value["id"]) is None
        or not all(isinstance(value[key], str) for key in ("assistant", "action"))
        or not isinstance(value["inputs"], list)
        or not all(isinstance(name, str) for name in value["inputs"])
    ):
        raise RoutineContractError("invalid routines")
    return {"id": value["id"], "assistant": value["assistant"], "action": value["action"], "inputs": value["inputs"]}


def _listed_output(value: object) -> bool:
    """A listed Routine's output: a recordable mode, or a decision that runs always or only on a change."""
    return (
        isinstance(value, dict)
        and set(value) == {"mode", "when"}
        and value["mode"] in LISTED_MODES
        and (value["when"] in DECISION_WHEN if value["mode"] == "decide" else value["when"] is None)
    )


def canonical_routines(value: object) -> tuple[dict[str, object], ...]:
    """The Team's Routines as data (at most 8): id, name, schedule, zone, revision, daily steps, output, and steps.

    Only a decision may list no steps.
    """
    fields = {"routine_id", "name", "schedule", "timezone", "revision", "daily_steps", "output", "steps"}
    if not isinstance(value, (list, tuple)) or len(value) > MAX_ROUTINES:
        raise RoutineContractError("invalid routines")
    routines = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != fields:
            raise RoutineContractError("invalid routines")
        name = team_memory._line(entry["name"], MAX_NAME_CHARS)
        schedule = canonical_schedule(entry["schedule"])
        steps = entry["steps"]
        if (
            not isinstance(entry["routine_id"], str)
            or ROUTINE_ID_RE.fullmatch(entry["routine_id"]) is None
            or not name
            or schedule is None
            or not isinstance(entry["timezone"], str)
            or TIMEZONE_RE.fullmatch(entry["timezone"]) is None
            or not _whole(entry["revision"], 1, 2**31 - 1)
            or not valid_capacity(entry["daily_steps"])
            or not _listed_output(entry["output"])
            or not isinstance(steps, list)
            or len(steps) > MAX_STEPS
            or (not steps and entry["output"]["mode"] != "decide")
        ):
            raise RoutineContractError("invalid routines")
        listed = {**entry, "schedule": schedule, "output": dict(entry["output"])}
        routines.append({**listed, "steps": [_routine_step(step) for step in steps]})
    sizes = [len(json.dumps(item["steps"], ensure_ascii=False, separators=(",", ":")).encode()) for item in routines]
    if (
        len({item["routine_id"] for item in routines}) != len(routines)
        or any(size > MAX_LISTED_STEPS_BYTES for size in sizes)
        or sum(sizes) > MAX_LISTING_STEPS_BYTES
    ):
        raise RoutineContractError("invalid routines")
    return tuple(routines)


def _present(value: object) -> object:
    """A schedule object without the null members its one structured shape carries."""
    return {key: item for key, item in value.items() if item is not None} if isinstance(value, dict) else value


def _reply(value: object) -> str | None:
    if not isinstance(value, str) or "\x00" in value or not value.strip() or len(value) > MAX_REPLY_CHARS:
        return None
    return value


def _timezone(value: object) -> bool:
    return value is None or (isinstance(value, str) and TIMEZONE_RE.fullmatch(value) is not None)


def _output(value: object) -> bool:
    return isinstance(value, dict) and set(value) == {"mode"} and value["mode"] in OUTPUT_MODES


def record(arguments: object, context: Any) -> dict[str, object] | str:
    """The recorded outcome and reply of one valid call, or the closed correction of the first invalid field."""
    if not isinstance(arguments, dict) or set(arguments) != set(SCHEMA["required"]) or arguments["op"] != "record":
        return "invalid"
    name = team_memory._line(arguments["name"], MAX_NAME_CHARS)
    schedule = canonical_schedule(_present(arguments["schedule"]))
    replaces = arguments["replaces"]
    reply = _reply(arguments["reply"])
    checks = (
        ("name", bool(name)),
        ("schedule", schedule is not None),
        ("timezone", _timezone(arguments["timezone"])),
        ("output", _output(arguments["output"])),
        ("replaces", replaces is None or any(item["routine_id"] == replaces for item in context.routines)),
        ("reply", reply is not None),
    )
    refused = next((field for field, valid in checks if not valid), None)
    if refused is not None:
        return refused
    routine = {
        "op": "record",
        "name": name,
        "schedule": schedule,
        "timezone": arguments["timezone"],
        "output": {"mode": arguments["output"]["mode"], "when": None},
        "notes": "",
        "decide_actions": [],
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

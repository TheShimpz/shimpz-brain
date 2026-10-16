"""Team Routines recorded from work the ordinary chat agent actually did (ADR-0101).

A Routine is never compiled. When the person asks for work to recur, the chat agent clarifies as for any task, runs the
recurring work once in the same turn with the Team's Actions, and then calls one closed tool, ``shimpz_routine``
``record``, with the Routine's name, a plain-language description of the work for the card, the Routine it replaces,
and the reply the person reads. The schedule, timezone, and output are never the model's: Team derives them from the
person's own words and asks when they are missing (ADR-0101), and the description must say the schedule and
output exactly as Team read them. The guard checks that closed shape and ends the turn; Team builds the plan from its
own trace of the person's sends and shows the person a card to confirm, or a question. Nothing here schedules,
approves, or authorizes anything: Brain only reports what the agent asked to record.
"""

import functools
import json
from collections.abc import Mapping
from typing import Any

import tool_refusal
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from protocol.team.http.v1 import phrase as team_phrase
from protocol.team.http.v1 import routine as team_routine_protocol
from protocol.team.http.v1 import routine_context as team_routine_context
from protocol.team.http.v1 import routine_proposal as team_routine_proposal

TOOL_NAME = "shimpz_routine"
# Every Team->Brain Routine form and bound is Team's own, read from its mirrored protocol, never copied here.
MAX_ROUTINES = team_routine_protocol.MAX_ROUTINES
# Brain's own request bound on the listing: each Routine's steps, encoded, at most 256 KiB, and the Team's together
# 1 MiB, as Team's plans are bounded, so a listing never outgrows the turn request Brain admits.
MAX_LISTED_STEPS_BYTES = 256 * 1024
MAX_LISTING_STEPS_BYTES = 1024 * 1024
# The reply the person reads beside the card: the result of the work the agent ran.
MAX_REPLY_CHARS = 4000
_NAME_CHARS = team_routine_protocol.MAX_ROUTINE_NAME_CHARS
_DESCRIPTION_CHARS = team_routine_protocol.MAX_ROUTINE_DESCRIPTION_CHARS
_DAY = team_routine_protocol.DAY_SECONDS
# Every continuous interval Team admits, as the prompt states it.
INTERVALS = (
    f"{team_routine_protocol.MIN_CONTINUOUS_GAP_SECONDS} seconds to "
    f"{team_routine_protocol.MAX_CONTINUOUS_GAP_SECONDS // _DAY} days"
)
SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["record"]},
        "name": {
            "type": "string",
            "description": (
                f"A short name for the Routine's work, at most {_NAME_CHARS} characters, without the word Routine or "
                "how often it runs."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                f"One plain paragraph, at most {_DESCRIPTION_CHARS} characters, written as the person's own request "
                "for this Routine, "
                "in everyday words for a person who is not technical: how often it runs, in the person's own words, "
                "what every run does and on what, and what is done with each run's result only as the person said "
                "it. Describe only the work you ran, say plainly when a run changes something, and never promise how "
                "the result reaches the person."
            ),
        },
        "replaces": {
            "type": ["string", "null"],
            "description": "The routine_id of the listed Routine this one changes; null for a new Routine.",
        },
        "reply": {
            "type": "string",
            "description": (
                "Your reply to the person: the result of the work you ran in this message, in their language; empty "
                "when you ran no Action in this message, since the card says everything."
            ),
        },
    },
    "required": ["op", "name", "description", "replaces", "reply"],
    "additionalProperties": False,
}
DESCRIPTION = (
    "Record the work you just ran as a Routine the Team repeats on a schedule. Call it only when the person asks for "
    "work to recur or to change a listed Routine, only after you ran exactly that work once, unless only a listed "
    "Routine's schedule or output changes or the person answers the Team's question about how often it runs or what to "
    "do with each run's result, and alone in your response. It takes no schedule, timezone, or output: the Team "
    f"reads how often (any interval from {INTERVALS}, at most ceil({_DAY} / interval) runs a day) and what to "
    "do with each run's "
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
    "name": _NOT_DONE + f"Give a name of one line, at most {_NAME_CHARS} characters.",
    "description": _NOT_DONE + f"Give a description of one plain paragraph, at most {_DESCRIPTION_CHARS} characters.",
    "schedule": _NOT_DONE + "Say in the description how often the Routine runs in the person's own words, and no other "
    "schedule.",
    "unstated-schedule": _NOT_DONE + "The person has not said how often it runs yet: leave how often out of the "
    "description; the Team asks them.",
    "output": _NOT_DONE + "Say in the description what is done with each run's result as the person chose it, in "
    "these words, and no other choice: ",
    "unstated-output": _NOT_DONE + "The person has not changed what is done with each run's result: say the "
    "Routine's own output or leave it out of the description.",
    "replaces": _NOT_DONE + "replaces must be the routine_id of a listed Routine, or null for a new one.",
    "reply": _NOT_DONE + f"Give the person an empty reply or one of at most {MAX_REPLY_CHARS} characters.",
}


class RoutineContractError(ValueError):
    pass


def admit_context(context: Any) -> None:
    """Canonicalize a turn's Routines and what Team sends about its recording in place; refuse anything invalid.

    Everything about a recording comes only beside the Team's Routines, which offer the Routine tool.
    """
    if context.routines is not None:
        object.__setattr__(context, "routines", canonical_routines(context.routines))
    if type(context.routine_mode) is not bool or (context.routine_mode and context.routines is None):
        raise RoutineContractError("invalid Routine mode")
    for field, canonical, name in (
        ("routine_question", canonical_question, "question"),
        ("routine_rerun", canonical_rerun, "rerun"),
        ("routine_schedule", canonical_schedule, "schedule"),
        ("routine_output", canonical_output, "output"),
    ):
        value = getattr(context, field)
        if value is None:
            continue
        admitted = None if context.routines is None else canonical(value)
        if admitted is None:
            raise RoutineContractError(f"invalid Routine {name}")
        object.__setattr__(context, field, admitted)
    if type(context.routine_settings_answer) is not bool or (context.routine_settings_answer and not _settled(context)):
        raise RoutineContractError("invalid Routine settings answer")


def settings_question(question: Mapping[str, Any] | None) -> bool:
    """Whether Team's pending question asks how often the Routine runs or what each run does with its result.

    Its work already ran, so an answer to it runs no work unless the answer changes the work, and then the whole
    resulting work once.
    """
    return question is not None and question["code"] in team_routine_proposal.SETTINGS_QUESTION_CODES


# The settings questions about how often the Routine runs; the only other one asks what each run does with its result.
_SCHEDULE_QUESTION_CODES = team_routine_proposal.SETTINGS_QUESTION_CODES - {"routine-output-unstated"}


def _settled(context: Any) -> bool:
    """Whether a settings answer answers a settings question, with the reading Team took from it."""
    question = context.routine_question
    code = None if question is None else question["code"]
    if code == "routine-output-unstated":
        return context.routine_output is not None
    return code in _SCHEDULE_QUESTION_CODES and context.routine_schedule is not None


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
    """The reply the person reads beside the card: empty, or text that says something."""
    if value == "":
        return value
    if not isinstance(value, str) or "\x00" in value or not value.strip() or len(value) > MAX_REPLY_CHARS:
        return None
    return value


def canonical_schedule(value: object) -> dict[str, object] | None:
    """A schedule in Team's closed form, as Team's protocol admits it, or None."""
    return team_routine_protocol.canonical_schedule(value)


def canonical_output(value: object) -> str | None:
    """One of Team's output modes, or None."""
    return value if isinstance(value, str) and value in team_routine_protocol.OUTPUT_MODES else None


def _output_miss(description: str, output: object, replaced: object) -> str:
    """Whether a description says what each run does with its result as the person chose it, "" when it does.

    Team's reading is the authority once the person chose: the description states exactly that output. A new
    Routine's output not chosen yet is asked before any card, and the answer has the description written again, so it
    is not checked here; a replacement keeps its listed output unasked, so its description states that one or none.
    """
    stated = team_phrase.outputs(description)
    if output is not None:
        return "" if stated == (output,) else "output"
    if replaced is None or stated in ((), (replaced["output"]["mode"],)):
        return ""
    return "unstated-output"


def _schedule_miss(description: str, schedule: object, replaced: object) -> str:
    """Whether a description says how often the Routine runs as Team read it from the person, "" when it does.

    Team's own reading is the authority: with one schedule, the description must state exactly it; with none, it
    states none, except a replacement's own listed schedule, which it keeps.
    """
    stated = team_phrase.stated(description)
    if schedule is not None:
        return "" if stated == (schedule,) else "schedule"
    kept = () if replaced is None else (replaced["schedule"],)
    return "" if stated in ((), kept) else "unstated-schedule"


def record(arguments: object, context: Any) -> dict[str, object] | str:
    """The recorded outcome and reply of one valid call, or the closed correction of the first invalid field."""
    if not isinstance(arguments, dict) or set(arguments) != set(SCHEMA["required"]) or arguments["op"] != "record":
        return "invalid"
    name = team_routine_protocol.canonical_name(arguments["name"])
    description = arguments["description"]
    replaces = arguments["replaces"]
    reply = _reply(arguments["reply"])
    replaced = next((item for item in context.routines if item["routine_id"] == replaces), None)
    checks = (
        ("name", bool(name)),
        ("description", team_routine_protocol.canonical_description(description) is not None),
        ("replaces", replaces is None or replaced is not None),
        ("reply", reply is not None),
    )
    refused = next((field for field, valid in checks if not valid), None)
    if refused is not None:
        return refused
    missed = _schedule_miss(description, context.routine_schedule, replaced) or _output_miss(
        description, context.routine_output, replaced
    )
    if missed:
        return missed
    routine = {"name": name, "description": description, "replaces": replaces}
    return {"routine": routine, "reply": reply}


def correction(outcome: str, context: Any) -> str:
    """The closed correction of a refused record; an output miss names the person's choice in the chat's words."""
    if outcome != "output":
        return _CORRECTIONS[outcome]
    label = team_routine_proposal.OUTPUT_CHOICES[context.locale or "en"][context.routine_output]
    return f"{_CORRECTIONS[outcome]}{json.dumps(label, ensure_ascii=False)}."


def _review(messages: list[Any], context: Any) -> object:
    call = tool_refusal.lone_call(messages, TOOL_NAME, label="a Routine record")
    if call is None or isinstance(call, str):
        return call
    return record(call.get("args"), context)


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
            return tool_refusal.refusal(calls, correction(outcome, self.context))

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

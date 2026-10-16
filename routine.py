"""Team Routines recorded from work the ordinary chat agent actually did (ADR-0101).

A Routine is never compiled. When the person asks for work to recur, the chat agent clarifies as for any task, runs the
recurring work once in the same turn with the Team's Actions, and then calls one closed tool, ``shimpz_routine``
``record``, with the Routine's name, a plain-language description of the work for the card, the Routine it replaces,
and the reply the person reads. The schedule, timezone, and output are never the model's: Team derives them from the
person's own words, asks when they are missing, and shows them on the card in its own fixed words, so the description
says only the work (ADR-0101). The guard checks that closed shape and ends the turn; Team builds the plan from its own
trace of the person's sends and shows the person a card to confirm, or a question. Nothing here schedules,
approves, or authorizes anything: Brain only reports what the agent asked to record.
"""

import functools
import json
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import context_budget
import memory
import tool_refusal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from protocol.team.http.v1 import payload as team_payload
from protocol.team.http.v1 import routine as team_routine_protocol
from protocol.team.http.v1 import routine_context as team_routine_context
from protocol.team.http.v1 import routine_proposal as team_routine_proposal
from protocol.team.http.v1 import turn as team_turn

TOOL_NAME = "shimpz_routine"
# Brain's own request bound on the listing: each Routine's steps, encoded, at most 256 KiB, and the Team's together
# 1 MiB, as Team's plans are bounded, so a listing never outgrows the turn request Brain admits.
MAX_LISTED_STEPS_BYTES = 256 * 1024
MAX_LISTING_STEPS_BYTES = 1024 * 1024
# The reply the person reads beside the card: the result of the work the agent ran.
MAX_REPLY_CHARS = 4000
# Every Team->Brain Routine form and bound is Team's own, read from its mirrored protocol, never copied here.
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
                "for this Routine's work, in everyday words for a person who is not technical: what every run does "
                "and on what. Describe only the work you ran and say plainly when a run changes something; never say "
                "how often it runs, its timezone, or what is done with each run's result, which the Team's card shows "
                "itself."
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
    "do with each run's result without changing the work, and alone in your response. It takes no schedule, timezone, "
    f"or output: the Team reads how often (any interval from {INTERVALS}, at most ceil({_DAY} / interval) runs a day) "
    "and what to do with each run's result from the person's own words, in the person's own timezone, asks the person "
    "itself when something is missing or does not fit, and shows them on its card. It ends the turn; the Team then "
    "shows the person that card to confirm, so never say a Routine was created."
)
# How a call outside the closed shape is answered: nothing was recorded, and the model may call again.
_NOT_DONE = "Not done: nothing was recorded. "
_CORRECTIONS = {
    "mixed": _NOT_DONE + "Call shimpz_routine alone in its response, after the work ran; nothing else in this "
    "response ran, so repeat the calls you still need.",
    "invalid": _NOT_DONE + "Call it with op record and exactly the members its schema names.",
    "name": _NOT_DONE + f"Give a name of one line, at most {_NAME_CHARS} characters.",
    "description": _NOT_DONE + f"Give a description of one plain paragraph, at most {_DESCRIPTION_CHARS} characters.",
    "replaces": _NOT_DONE + "replaces must be the routine_id of a listed Routine, or null for a new one.",
    "reply": _NOT_DONE + "Give the person an empty reply, or text that says something, of at most "
    f"{MAX_REPLY_CHARS} characters with no control characters other than line breaks and tabs.",
}


# A numeric date inside a word, such as 2026-10-05, 05/10, or 10月15日, which Team may read as each run's own date.
_DATED_RE = re.compile(r"(?<!\d)\d{1,4}[-/.]\d{1,2}(?!\d)|\d[年月]")
# How a rerun call that sends a 'fresh' input no result of this turn holds is answered: nothing ran.
_RERUN_CORRECTION = (
    "Not done: no Action in this response ran. The Team needs each 'fresh' input of its rerun work to come from a "
    "result in this turn: first run the listed source Action, or an Action that returns the value, then send the value "
    "from that result; never send a value you remember."
)


class RoutineContractError(ValueError):
    pass


def admit_context(context: Any) -> None:
    """Canonicalize a turn's Routines and what Team sends about its recording in place; refuse anything invalid.

    Everything about a recording comes only beside the Team's Routines, which offer the Routine tool, and only in the
    combinations Team's recording makes (Team's protocol README): never with attachments, a question or schedule only
    in Routine mode, and rerun work only beside a rerun question.
    """
    if context.routines is not None:
        if context.attachments:
            # Only a send without files records.
            raise RoutineContractError("invalid routines")
        object.__setattr__(context, "routines", canonical_routines(context.routines))
    if type(context.routine_mode) is not bool or (context.routine_mode and context.routines is None):
        raise RoutineContractError("invalid Routine mode")
    for field, canonical, name in (
        ("routine_question", team_routine_proposal.canonical_question, "question"),
        ("routine_rerun", canonical_rerun, "rerun"),
        ("routine_schedule", team_routine_protocol.canonical_schedule, "schedule"),
        ("routine_output", canonical_output, "output"),
    ):
        value = getattr(context, field)
        if value is None:
            continue
        admitted = None if context.routines is None or not _sent_with(context, field) else canonical(value)
        if admitted is None:
            raise RoutineContractError(f"invalid Routine {name}")
        object.__setattr__(context, field, admitted)
    if type(context.routine_settings_answer) is not bool or (context.routine_settings_answer and not _settled(context)):
        raise RoutineContractError("invalid Routine settings answer")


def _sent_with(context: Any, field: str) -> bool:
    """Whether Team sends this recording field beside the others the context carries, the question already admitted."""
    if field in {"routine_question", "routine_schedule"}:
        # A pending question or a stated schedule makes the chat about a Routine.
        return context.routine_mode
    if field == "routine_rerun":
        question = context.routine_question
        return question is not None and question["code"] == "routine-work-rerun"
    return True


def settings_question(question: Mapping[str, Any] | None) -> bool:
    """Whether Team's pending question asks how often the Routine runs or what each run does with its result.

    Its work already ran, so an answer to it runs no work unless the answer changes the work, and then the whole
    resulting work once.
    """
    return question is not None and question["code"] in team_routine_proposal.SETTINGS_QUESTION_CODES


def _settled(context: Any) -> bool:
    """Whether a settings answer answers a settings question, with the reading Team took from it."""
    question = context.routine_question
    code = None if question is None else question["code"]
    if code == "routine-output-unstated":
        return context.routine_output is not None
    # Every other settings question asks how often the Routine runs.
    return code in team_routine_proposal.SETTINGS_QUESTION_CODES and context.routine_schedule is not None


def canonical_routines(value: object) -> tuple[dict[str, object], ...]:
    """The Team's Routines as data, exactly as Team's protocol admits its listing, or a contract error."""
    listed = team_routine_context.canonical_routine_listings(value)
    if listed is None:
        raise RoutineContractError("invalid routines")
    sizes = [len(json.dumps(item["steps"], ensure_ascii=False, separators=(",", ":")).encode()) for item in listed]
    if any(size > MAX_LISTED_STEPS_BYTES for size in sizes) or sum(sizes) > MAX_LISTING_STEPS_BYTES:
        raise RoutineContractError("invalid routines")
    return tuple(listed)


def canonical_rerun(value: object) -> tuple[dict[str, object], ...] | None:
    """The work Team asks the agent to run again before it records, as Team's closed form, or None."""
    work = team_routine_context.canonical_rerun(value)
    return None if work is None else tuple(work)


def _reply(value: object) -> str | None:
    """The reply the person reads beside the card: empty, or text that says something."""
    if value == "":
        return value
    if (
        not isinstance(value, str)
        or team_turn.REPLY_CONTROL_RE.search(value) is not None
        or not value.strip()
        or len(value) > MAX_REPLY_CHARS
    ):
        return None
    return value


def canonical_output(value: object) -> str | None:
    """One of Team's output modes, or None."""
    return value if isinstance(value, str) and value in team_routine_protocol.OUTPUT_MODES else None


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
    routine = {"name": name, "description": description, "replaces": replaces}
    return {"routine": routine, "reply": reply}


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
            return tool_refusal.refusal(calls, _CORRECTIONS[outcome])

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


def rerun_guard(work: tuple[dict[str, object], ...], tool_name: Callable[[str, str], str], actions: frozenset[str]):
    """The guard that refuses a response sending a rerun's 'fresh' input that no result of this turn holds.

    The rerun prompt asks for each fresh input from a result of this turn; this enforces it, so the agent never sends
    a remembered value Team would only ask about again. Team's recorder stays the authority. A member is checked only
    where every slot of its Action marks it fresh, and only an identifier-shaped value, which Team would otherwise ask
    about again; it passes when it equals any value inside an Action result of this turn (``actions`` names the Action
    tools) or the person's current message writes it.
    """
    slots: dict[str, list[set[str]]] = {}
    for entry in work:
        fresh = {item["member"] for item in entry["inputs"] if item["kind"] == "fresh"}
        slots.setdefault(tool_name(entry["assistant"], entry["action"]), []).append(fresh)
    checked = {name: frozenset(members) for name, sets in slots.items() if (members := set.intersection(*sets))}
    review = functools.partial(_rerun_review, checked=checked, actions=actions)
    return tool_refusal.guard("RerunGuard", review, {"fresh": _RERUN_CORRECTION})


def _rerun_review(messages: list[Any], *, checked: Mapping[str, frozenset[str]], actions: frozenset[str]) -> str | None:
    latest = messages[-1] if messages else None
    if not isinstance(latest, AIMessage):
        return None
    calls = [call for call in latest.tool_calls or [] if call.get("name") in checked]
    if not calls:
        return None
    if latest.invalid_tool_calls:
        # A checked call beside an unparsable one cannot be judged or answered whole: a contract error, as elsewhere.
        raise tool_refusal.UnanswerableToolCallError("RerunGuard arrived with an unparsable tool call")
    start = max(index for index, message in enumerate(messages) if isinstance(message, HumanMessage))
    turn = messages[start:]
    results = [
        node
        for message in turn
        if isinstance(message, ToolMessage) and message.name in actions
        for node in _values(_result(message.content))
    ]
    current = memory.turn_message(turn[0].content) or ""
    written = team_payload.authored_segments(current)
    for call in calls:
        sent = call["args"]
        for member in checked[call["name"]] & sent.keys():
            value = sent[member]
            if _identifier(value) and not any(value == node for node in results) and not _written(value, written):
                return "fresh"
    return None


def _result(content: object) -> object:
    """An Action result as the model read it: its JSON value, or its text when it is not JSON."""
    text = context_budget.message_text(content)
    try:
        return json.loads(text)
    except ValueError, RecursionError:
        return text


def _values(value: object) -> Iterator[object]:
    """The value and every value inside it."""
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _values(item)


def _identifier(value: object) -> bool:
    """Whether Team would ask where a value no result holds came from: an identifier, as its recorder shapes one.

    One word or whole number at least as long as Team's mirrored identifier bounds (Team's ``routine.provenance``);
    any other value Team keeps as the agent's own choice, such as a short lookup term a helper call sends. A word
    holding a numeric date is left to Team, which may read it as each run's date.
    """
    if type(value) is int:
        return len(str(abs(value))) >= team_routine_context.MIN_IDENTIFIER_DIGITS
    return (
        isinstance(value, str)
        and len(value) >= team_routine_context.MIN_IDENTIFIER_CHARS
        and not any(character.isspace() for character in value)
        and _DATED_RE.search(value) is None
    )


def _written(value: str | int, segments: tuple[str, ...]) -> bool:
    """Whether the person's current message writes this identifier."""
    text = value if isinstance(value, str) else str(value)
    return any(text in segment for segment in segments)

"""Brain-authored multiple-choice clarification (ADR-0081).

When a material decision is open, the model may call the closed `shimpz_clarify` tool with one question, two to five
options, and a recommended default. The question is presentation only: it ends the turn, requests nothing, and the
user answers with a new message of their own, so Action authority stays with the user's current message.

`ClarificationGuard` inspects every model response before any tool runs. A clarification combined with any other tool
call, asked after an Action already ran in the logical turn, repeated, or malformed is refused with a closed
correction and nothing executes.
"""

from __future__ import annotations

import functools
import unicodedata
from dataclasses import dataclass
from typing import Any

import tool_refusal
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from protocol.team.http.v1 import phrase as team_phrase
from protocol.team.http.v1 import turn as team_turn

TOOL_NAME = "shimpz_clarify"
# The question bounds are Team's chat-turn protocol, applied through the Brain's pinned mirror.
MAX_QUESTION_CHARS = team_turn.MAX_CLARIFICATION_QUESTION_CHARS
MAX_LABEL_CHARS = team_turn.MAX_CLARIFICATION_LABEL_CHARS
MAX_DESCRIPTION_CHARS = team_turn.MAX_CLARIFICATION_DESCRIPTION_CHARS
MIN_OPTIONS = team_turn.MIN_CLARIFICATION_OPTIONS
MAX_OPTIONS = team_turn.MAX_CLARIFICATION_OPTIONS
SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 1, "maxLength": MAX_QUESTION_CHARS},
        "options": {
            "type": "array",
            "minItems": MIN_OPTIONS,
            "maxItems": MAX_OPTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": MAX_LABEL_CHARS},
                    "description": {"type": "string", "maxLength": MAX_DESCRIPTION_CHARS},
                },
                "required": ["label", "description"],
                "additionalProperties": False,
            },
        },
        "default_index": {"type": "integer", "minimum": 0, "maximum": MAX_OPTIONS - 1},
    },
    "required": ["question", "options", "default_index"],
    "additionalProperties": False,
}
DESCRIPTION = (
    "Ask the user one multiple-choice question when a material decision is open. Give two to five distinct options "
    "and the index of the one you recommend. Do not add an 'other' option: the user can always answer in their own "
    "words instead. Calling this "
    "ends your turn: never combine it with another tool, and never answer the question yourself."
)
RECORDED = "Question recorded; it is shown to the user with a free-text option and ends this turn."
_CORRECTIONS = {
    "mixed": "Not executed: a clarification cannot be combined with any other tool call. Either ask one question "
    "alone or proceed with the request.",
    "after-action": "Not executed: an Action already ran in this request, so no clarification can be asked now. "
    "Finish the request with what you have and state what remains open.",
    "invalid": "Not executed: the clarification must have one single-line question, two to five distinct single-line "
    "options, and a default_index that points to one of them.",
    "team-owned": "Not asked: the Team asks the schedule, the output, and limits itself; continue the work and call "
    "record.",
}


class UnanswerableToolCallError(ValueError):
    """A clarification arrived beside a tool call no refusal can be paired with."""


@dataclass(frozen=True, slots=True)
class Option:
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class Clarification:
    question: str
    options: tuple[Option, ...]
    # The recommended option.
    default_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "options": [{"label": option.label, "description": option.description} for option in self.options],
            "default_index": self.default_index,
        }

    def render(self) -> str:
        """The plain reply shown and remembered: the question and numbered options, a recommended default marked ✓."""
        lines = [self.question, ""]
        for index, option in enumerate(self.options):
            mark = " ✓" if index == self.default_index else ""
            detail = f" — {option.description}" if option.description else ""
            lines.append(f"{index + 1}. {option.label}{mark}{detail}")
        return "\n".join(lines)


def _line(value: object, maximum: int, *, empty: bool = False) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFC", value).strip()
    if (not text and not empty) or len(text) > maximum:
        return None
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"} for character in text):
        return None
    return text


class _ClarificationContractError(ValueError):
    pass


def _required(value: str | None) -> str:
    if value is None:
        raise _ClarificationContractError
    return value


def _closed(arguments: object) -> Clarification:
    if not isinstance(arguments, dict) or set(arguments) != {"question", "options", "default_index"}:
        raise _ClarificationContractError
    question = _required(_line(arguments["question"], MAX_QUESTION_CHARS))
    raw_options = arguments["options"]
    if not isinstance(raw_options, list) or not MIN_OPTIONS <= len(raw_options) <= MAX_OPTIONS:
        raise _ClarificationContractError
    options = []
    for raw in raw_options:
        if not isinstance(raw, dict) or set(raw) != {"label", "description"}:
            raise _ClarificationContractError
        options.append(
            Option(
                _required(_line(raw["label"], MAX_LABEL_CHARS)),
                _required(_line(raw["description"], MAX_DESCRIPTION_CHARS, empty=True)),
            )
        )
    default_index = arguments["default_index"]
    recommended = (
        not isinstance(default_index, bool) and isinstance(default_index, int) and 0 <= default_index < len(options)
    )
    if len({option.label.casefold() for option in options}) != len(options) or not recommended:
        raise _ClarificationContractError
    return Clarification(question, tuple(options), default_index)


def parse(arguments: object) -> Clarification | None:
    """Return the closed clarification, or None when any field breaks the contract: it recommends exactly one option."""
    try:
        return _closed(arguments)
    except _ClarificationContractError:
        return None


def tool() -> StructuredTool:
    """The clarification tool; the guard validates its arguments before it can run, and it ends the turn."""

    def record_clarification(**_arguments):
        return RECORDED

    return StructuredTool.from_function(
        record_clarification,
        name=TOOL_NAME,
        description=DESCRIPTION,
        args_schema=SCHEMA,
        infer_schema=False,
        return_direct=True,
    )


def _team_owned(asked: Clarification) -> bool:
    """Whether a clarification asks what Team reads from the person's words itself, by Team's own phrase tables."""
    return any(team_phrase.team_asks(text) for text in (asked.question, *(option.label for option in asked.options)))


def _review(messages: list[Any], *, allowed: bool, routine_mode: bool = False) -> str | None:
    """Return why the latest model response must not run, or None when it may run.

    A turn's start runs no Action before it suspends, and every resume follows an Action, so a clarification is
    allowed exactly while a turn starts. In a Routine turn, a question about the schedule, the output, or limits is
    Team's to ask (ADR-0101), so the agent's own is refused.
    """
    latest = messages[-1] if messages else None
    if not isinstance(latest, AIMessage):
        return None
    calls = [*(latest.tool_calls or []), *(latest.invalid_tool_calls or [])]
    if not any(call.get("name") == TOOL_NAME for call in calls):
        return None
    if latest.invalid_tool_calls:
        # A refusal must pair with a call the provider sees again, and adapters drop unparsable calls (Anthropic) or
        # send them without an id; so a clarification beside one ends the turn as a contract error.
        raise UnanswerableToolCallError("a clarification arrived with an unparsable tool call")
    if len(calls) > 1:
        return "mixed"
    if not allowed:
        return "after-action"
    return _refused(latest.tool_calls[0].get("args"), routine_mode)


def _refused(arguments: object, routine_mode: bool) -> str | None:
    """Why one lone clarification's own arguments may not run: malformed, or Team's question in a Routine turn."""
    asked = parse(arguments)
    if asked is None:
        return "invalid"
    if routine_mode and _team_owned(asked):
        return "team-owned"
    return None


def guard(*, allowed: bool, routine_mode: bool = False):
    """The middleware that reviews every model response before its tools run."""
    review = functools.partial(_review, allowed=allowed, routine_mode=routine_mode)
    return tool_refusal.guard("ClarificationGuard", review, _CORRECTIONS)


def recorded(messages: list[Any]) -> Clarification | None:
    """The clarification that ended this turn, when the graph stopped on the clarification tool's result."""
    if len(messages) < 2 or not isinstance(messages[-1], ToolMessage) or messages[-1].name != TOOL_NAME:
        return None
    call = messages[-2]
    if not isinstance(call, AIMessage) or len(call.tool_calls or []) != 1:
        return None
    return parse(call.tool_calls[0].get("args"))

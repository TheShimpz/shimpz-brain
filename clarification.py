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

TOOL_NAME = "shimpz_clarify"
MAX_QUESTION_CHARS = 240
MAX_LABEL_CHARS = 80
MAX_DESCRIPTION_CHARS = 160
MIN_OPTIONS = 2
MAX_OPTIONS = 5
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
    # The recommended option, or None for a Routine question, which recommends and preselects none (ADR-0092).
    default_index: int | None

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


def _closed(arguments: object, routine: bool) -> Clarification:
    if not isinstance(arguments, dict) or set(arguments) != {"question", "options", "default_index"}:
        raise _ClarificationContractError
    question = _required(_line(arguments["question"], MAX_QUESTION_CHARS))
    raw_options = arguments["options"]
    # A Routine question recommends nothing and may offer a single suggestion beside the free-text answer.
    minimum = 1 if routine else MIN_OPTIONS
    if not isinstance(raw_options, list) or not minimum <= len(raw_options) <= MAX_OPTIONS:
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
    if len({option.label.casefold() for option in options}) != len(options) or (
        default_index is not None if routine else not recommended
    ):
        raise _ClarificationContractError
    return Clarification(question, tuple(options), default_index)


def parse(arguments: object, *, routine: bool = False) -> Clarification | None:
    """Return the closed clarification, or None when any field breaks the contract.

    An ordinary question recommends exactly one option; a Routine question (``routine``) recommends none, so its
    ``default_index`` is null and the person's choice is never preselected (ADR-0092 amendment, 2026-10-05).
    """
    try:
        return _closed(arguments, routine)
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


def _review(messages: list[Any], *, allowed: bool) -> str | None:
    """Return why the latest model response must not run, or None when it may run.

    A turn's start runs no Action before it suspends, and every resume follows an Action, so a clarification is
    allowed exactly while a turn starts.
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
    if parse(latest.tool_calls[0].get("args")) is None:
        return "invalid"
    return None


def guard(*, allowed: bool):
    """The middleware that reviews every model response before its tools run."""
    return tool_refusal.guard("ClarificationGuard", functools.partial(_review, allowed=allowed), _CORRECTIONS)


def recorded(messages: list[Any]) -> Clarification | None:
    """The clarification that ended this turn, when the graph stopped on the clarification tool's result."""
    if len(messages) < 2 or not isinstance(messages[-1], ToolMessage) or messages[-1].name != TOOL_NAME:
        return None
    call = messages[-2]
    if not isinstance(call, AIMessage) or len(call.tool_calls or []) != 1:
        return None
    return parse(call.tool_calls[0].get("args"))

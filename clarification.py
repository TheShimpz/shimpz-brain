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


@dataclass(frozen=True, slots=True)
class Option:
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class Clarification:
    question: str
    options: tuple[Option, ...]
    default_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "options": [{"label": option.label, "description": option.description} for option in self.options],
            "default_index": self.default_index,
        }

    def render(self) -> str:
        """The plain reply shown and remembered: the question and numbered options, the default marked with ✓."""
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
    if (
        len({option.label.casefold() for option in options}) != len(options)
        or isinstance(default_index, bool)
        or not isinstance(default_index, int)
        or not 0 <= default_index < len(options)
    ):
        raise _ClarificationContractError
    return Clarification(question, tuple(options), default_index)


def parse(arguments: object) -> Clarification | None:
    """Return the closed clarification, or None when any field breaks the contract."""
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


def _review(messages: list[Any], *, allowed: bool) -> str | None:
    """Return why the latest model response must not run, or None when it may run.

    A turn's start runs no Action before it suspends, and every resume follows an Action, so a clarification is
    allowed exactly while a turn starts.
    """
    latest = messages[-1] if messages else None
    if not isinstance(latest, AIMessage):
        return None
    calls = _calls(latest)
    if not any(call.get("name") == TOOL_NAME for call in calls):
        return None
    if len(calls) > 1:
        return "mixed"
    if not allowed:
        return "after-action"
    # A call the provider could not parse is listed only among the invalid tool calls.
    if not latest.tool_calls or parse(latest.tool_calls[0].get("args")) is None:
        return "invalid"
    return None


def _calls(message: AIMessage) -> list[dict[str, Any]]:
    """Every tool call of a response, including the ones the provider could not parse."""
    return [*(message.tool_calls or []), *(message.invalid_tool_calls or [])]


@functools.cache
def _guard_class():
    # The agent middleware stack loads with the graph, never when the runtime API module is imported.
    from langchain.agents.middleware import AgentMiddleware, hook_config

    class ClarificationGuard(AgentMiddleware):
        """Refuse unsafe or malformed clarification calls before any tool of that model response runs."""

        def __init__(self, *, allowed: bool) -> None:
            super().__init__()
            self.allowed = allowed

        @hook_config(can_jump_to=["model"])
        def after_model(self, state, runtime) -> dict[str, Any] | None:
            messages = list(state["messages"])
            reason = _review(messages, allowed=self.allowed)
            if reason is None:
                return None
            return {
                "messages": [
                    ToolMessage(
                        content=_CORRECTIONS[reason],
                        # An unparsable call may carry no id; its refusal still needs one to stay a closed correction.
                        tool_call_id=call["id"] or f"shimpz-unidentified-call-{index}",
                        name=call["name"] or TOOL_NAME,
                    )
                    for index, call in enumerate(_calls(messages[-1]))
                ],
                "jump_to": "model",
            }

    return ClarificationGuard


def guard(*, allowed: bool):
    """The middleware that reviews every model response before its tools run."""
    return _guard_class()(allowed=allowed)


def recorded(messages: list[Any]) -> Clarification | None:
    """The clarification that ended this turn, when the graph stopped on the clarification tool's result."""
    if len(messages) < 2 or not isinstance(messages[-1], ToolMessage) or messages[-1].name != TOOL_NAME:
        return None
    call = messages[-2]
    if not isinstance(call, AIMessage) or len(call.tool_calls or []) != 1:
        return None
    return parse(call.tool_calls[0].get("args"))

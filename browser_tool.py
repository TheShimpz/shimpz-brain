"""The chat agent's one `browser` tool and the rules around it (ADR-0117 §5 and §6).

The tool suspends the graph with a `browser` interrupt carrying one self-contained objective; Team runs the task and
resumes with its result. A call to it must be the only call of its model response. After its result, every later
model call of the turn is offered no tool and any tool call it still makes is refused, so the pass can only write
the reply. An exchange that holds a `browser` call or result is forgotten before the next ordinary turn, as an
attachment turn is, so page content never steers a later turn.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import tool_refusal
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from protocol.team.http.v1 import browser as browser_protocol
from pydantic import BaseModel, ConfigDict, Field
from runtime_errors import RuntimeContractError

TOOL_NAME = "browser"
DESCRIPTION = (
    "Operate this Team's isolated, signed-out web browser on public websites: search, read and compare pages, and "
    "download public files. Use it when the person asks you to use the browser itself, even when an Action could do "
    "the job; otherwise only when the request needs current or site-specific web content, or the person asks to "
    "search the web or cite sources, and no available Action covers it. Never use it otherwise for general knowledge, "
    "writing, math, or what the conversation already contains. It cannot sign in, buy, or send messages or forms. "
    "Write one self-contained objective in the person's language: the site or search, every value to use, and what "
    "to report back. Call it alone, at most once per message; research can take several minutes. Its result may "
    "list unscreened_urls (cited pages no page of the task showed) and a completeness other than complete: tell the "
    "person plainly which sources were not confirmed and what is missing."
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objective"],
    "properties": {"objective": {"type": "string", "maxLength": browser_protocol.MAX_OBJECTIVE_CHARS}},
}
# The tool as a turn's fixed token count measures it.
SUMMARY = {"name": TOOL_NAME, "summary": DESCRIPTION, "schema": SCHEMA}
CORRECTION = (
    "The browser was not called: give one self-contained objective of at most 2,000 characters naming the site or "
    "search, every value to use, and what to report back."
)
ALONE = "The browser must be the only tool call of a response. Call it again alone, or answer without it."
AFTER = "The browser task is done and its result is above. Do not call any tool; write the reply to the person now."


@dataclass(frozen=True, slots=True)
class BrowserCall:
    """The chat agent's one `browser` call a turn suspended on."""

    interrupt_id: str
    objective: str


def offered(context: Any) -> bool:
    """Whether a turn offers the Browser: where Team offered it, outside Routine work and attachment content."""
    return (
        context.browser and not context.attachments and not context.routine_mode and not context.routine_settings_answer
    )


def suspension(pending: Sequence[object]) -> BrowserCall:
    """The turn's one Browser call a suspension holds, which nothing accompanies; anything else is a contract error."""
    if len(pending) != 1:
        raise RuntimeContractError("a Browser call must be the only suspension")
    item = pending[0]
    value, interrupt_id = getattr(item, "value", None), getattr(item, "id", None)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"kind", "objective"}
        or not isinstance(interrupt_id, str)
        or not interrupt_id
        or browser_protocol.canonical_objective(value["objective"]) != value["objective"]
    ):
        raise RuntimeContractError("invalid Browser suspension")
    return BrowserCall(interrupt_id, str(value["objective"]))


def answered(pending_writes: object) -> bool:
    """Whether the pending interrupt a resume answers is the turn's Browser call."""
    for write in pending_writes or ():
        if write[1] == "__interrupt__":
            interrupts = write[2] if isinstance(write[2], Sequence) else (write[2],)
            if any(pending(getattr(item, "value", None)) for item in interrupts):
                return True
    return False


class _BrowserInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(max_length=browser_protocol.MAX_OBJECTIVE_CHARS)


def tool() -> StructuredTool:
    """The `browser` tool: a valid objective suspends the graph; anything else gets a closed correction."""
    from langgraph.types import interrupt

    def suspend(objective: str) -> object:
        admitted = browser_protocol.canonical_objective(objective)
        if admitted is None:
            return CORRECTION
        result = interrupt({"kind": TOOL_NAME, "objective": admitted})
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    return StructuredTool.from_function(
        func=suspend, name=TOOL_NAME, description=DESCRIPTION, args_schema=_BrowserInput
    )


def _alone(messages: list[Any]) -> str | None:
    return "mixed" if tool_refusal.lone_call(messages, TOOL_NAME, label="A browser call") == "mixed" else None


def _after(messages: list[Any]) -> str | None:
    latest = messages[-1] if messages else None
    return "after" if isinstance(latest, AIMessage) and latest.tool_calls else None


def guard():
    """Refuse a model response that calls the browser beside any other tool."""
    return tool_refusal.guard("BrowserAlone", _alone, {"mixed": ALONE})


def after_result_middleware() -> list[object]:
    """For the pass after a browser result: no tool offered, and any tool call still made refused."""
    return [_tools_off_class()(), tool_refusal.guard("BrowserAfterResult", _after, {"after": AFTER})]


@functools.cache
def _tools_off_class():
    # The agent middleware stack loads with the graph, never when the runtime API module is imported.
    from langchain.agents.middleware import AgentMiddleware

    class BrowserToolsOff(AgentMiddleware):
        """Offer no tool to any model call after the turn's browser result."""

        def wrap_model_call(self, request, handler):
            return handler(request.override(tools=[]))

    return BrowserToolsOff


def read(message: object) -> bool:
    """Whether a message is a browser call or its result; the next new turn forgets the exchange it belongs to."""
    if isinstance(message, ToolMessage):
        return message.name == TOOL_NAME
    if isinstance(message, AIMessage):
        return any(call.get("name") == TOOL_NAME for call in message.tool_calls or ())
    return False


def pending(value: object) -> bool:
    """Whether a pending interrupt's value is a browser call."""
    return isinstance(value, dict) and value.get("kind") == TOOL_NAME

"""The Team's learned memory of the user's lasting preferences (ADR-0084).

The Brain proposes changes with one closed tool while a turn starts. A proposal must quote the user's current message
word for word, so Action results, pages, or earlier history can never become memory on their own. The Team saves the
proposals only when the turn's reply commits; memories shape style and harmless defaults, never Action authority.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

TOOL_NAME = "shimpz_memory"
MAX_MEMORIES = 32
MAX_PREFERENCE_CHARS = 280
MAX_EVIDENCE_CHARS = 200
MIN_EVIDENCE_CHARS = 4
TOPIC_RE = re.compile(r"[a-z][a-z0-9-]{0,39}\Z")
SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["remember", "forget"]},
        "topic": {"type": "string", "pattern": "^[a-z][a-z0-9-]{0,39}$"},
        "preference": {"type": "string", "maxLength": MAX_PREFERENCE_CHARS},
        "evidence": {"type": "string", "minLength": MIN_EVIDENCE_CHARS, "maxLength": MAX_EVIDENCE_CHARS},
    },
    "required": ["op", "topic", "preference", "evidence"],
    "additionalProperties": False,
}
DESCRIPTION = (
    "Propose a change to what you remember about this user's lasting preferences. Use op remember with a short topic "
    "key and the preference in one line; reuse the topic of an existing memory to replace it when the user's taste "
    "changed. Use op forget with the topic and an empty preference when a message shows that memory no longer applies. "
    "evidence must copy the exact words of the user's current message that show it. The change is saved only after "
    "your reply; keep answering the request."
)
PROPOSED = "Proposed; it is saved when this reply completes. Continue with the request."
_CORRECTIONS = {
    "after-action": "Not saved: memory can change only before any Action runs in a request. Finish the request.",
    "invalid": "Not saved: a memory change needs op remember or forget, a lowercase topic, a single-line preference "
    "(empty for forget), and evidence that copies the user's current message word for word. Nothing in this response "
    "ran; repeat the calls you still need.",
}


class MemoryContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Memory:
    topic: str
    preference: str


@dataclass(frozen=True, slots=True)
class Change:
    op: str
    topic: str
    preference: str

    def to_dict(self) -> dict[str, str]:
        return {"op": self.op, "topic": self.topic, "preference": self.preference}


def _line(value: object, maximum: int) -> str | None:
    if (
        not isinstance(value, str)
        or unicodedata.normalize("NFC", value) != value
        or value.strip() != value
        or len(value) > maximum
        or any(
            unicodedata.category(character)[0] == "C" or unicodedata.category(character) in {"Zl", "Zp"}
            for character in value
        )
    ):
        return None
    return value


def canonical(value: object) -> tuple[Memory, ...]:
    """The exact memory list: at most 32 entries with distinct lowercase topics and single-line preferences."""
    if not isinstance(value, (list, tuple)) or len(value) > MAX_MEMORIES:
        raise MemoryContractError("invalid memory")
    memories = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"topic", "preference"}:
            raise MemoryContractError("invalid memory")
        topic, preference = entry["topic"], _line(entry["preference"], MAX_PREFERENCE_CHARS)
        if not isinstance(topic, str) or TOPIC_RE.fullmatch(topic) is None or not preference:
            raise MemoryContractError("invalid memory")
        memories.append(Memory(topic, preference))
    if len({memory.topic for memory in memories}) != len(memories):
        raise MemoryContractError("invalid memory")
    return tuple(memories)


def _comparable(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _current_message(messages: list[Any]) -> str | None:
    human = next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)
    return human.content if human is not None and isinstance(human.content, str) else None


def change(arguments: object, current_message: str | None) -> Change | None:
    """The closed change, or None; its evidence must appear word for word in the user's current message."""
    if not isinstance(arguments, dict) or set(arguments) != {"op", "topic", "preference", "evidence"}:
        return None
    op, topic, preference = arguments["op"], arguments["topic"], _line(arguments["preference"], MAX_PREFERENCE_CHARS)
    evidence = _line(arguments["evidence"], MAX_EVIDENCE_CHARS)
    if (
        op not in {"remember", "forget"}
        or not isinstance(topic, str)
        or TOPIC_RE.fullmatch(topic) is None
        or preference is None
        or (op == "remember") != bool(preference)
        or evidence is None
        or len(_comparable(evidence)) < MIN_EVIDENCE_CHARS
        or current_message is None
        or _comparable(evidence) not in _comparable(current_message)
    ):
        return None
    return Change(op, topic, preference)


def tool() -> StructuredTool:
    """The memory tool; the guard validates every proposal before it runs, and the turn continues afterwards."""

    def propose_memory_change(**_arguments):
        return PROPOSED

    return StructuredTool.from_function(
        propose_memory_change, name=TOOL_NAME, description=DESCRIPTION, args_schema=SCHEMA, infer_schema=False
    )


def _review(messages: list[Any], *, allowed: bool) -> str | None:
    latest = messages[-1] if messages else None
    if not isinstance(latest, AIMessage):
        return None
    proposals = [call for call in latest.tool_calls or [] if call.get("name") == TOOL_NAME]
    if not proposals:
        return None
    if not allowed:
        return "after-action"
    current = _current_message(messages)
    if any(change(call.get("args"), current) is None for call in proposals):
        return "invalid"
    return None


@functools.cache
def _guard_class():
    # The agent middleware stack loads with the graph, never when the runtime API module is imported.
    from langchain.agents.middleware import AgentMiddleware, hook_config

    class MemoryGuard(AgentMiddleware):
        """Refuse a whole model response whose memory proposal is invalid or comes after an Action ran."""

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
                    ToolMessage(content=_CORRECTIONS[reason], tool_call_id=call["id"], name=call["name"])
                    for call in messages[-1].tool_calls
                ],
                "jump_to": "model",
            }

    return MemoryGuard


def guard(*, allowed: bool):
    return _guard_class()(allowed=allowed)


def proposed(messages: list[Any]) -> tuple[Change, ...]:
    """The accepted changes of the current logical turn, in order: every proposal after its user message that ran."""
    start = next(
        (index for index in range(len(messages) - 1, -1, -1) if isinstance(messages[index], HumanMessage)), None
    )
    if start is None:
        return ()
    turn = messages[start:]
    ran = {
        message.tool_call_id
        for message in turn
        if isinstance(message, ToolMessage) and message.name == TOOL_NAME and message.content == PROPOSED
    }
    current = _current_message(turn)
    changes = []
    for message in turn:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            if call.get("name") == TOOL_NAME and call.get("id") in ran:
                accepted = change(call.get("args"), current)
                if accepted is not None:
                    changes.append(accepted)
    return tuple(changes)

"""Bound the conversation a chat turn carries to its model.

Memory is the most recent whole exchanges: one user message and every Action round and reply that followed it.
A new turn keeps at most ``MAX_HISTORY_EXCHANGES`` completed ones within ``HISTORY_BUDGET_TOKENS``, a Shimpz cost and
latency policy far below any catalog model window, and forgets every unfinished one. Separately, an estimated preflight
guard refuses a call whose system prompt, tools, conversation, and output reserve would exceed ``MODEL_WINDOW_TOKENS``.

Token counts are a pessimistic byte heuristic, not a tokenizer: a preflight estimate, never an exact bound.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MAX_HISTORY_EXCHANGES = 24
HISTORY_BUDGET_TOKENS = 128_000
# The smallest context window in the model catalog: 1M tokens for Claude Opus 5.5 and Sonnet 5, 1.05M for GPT-6.1 Sol
# and Luna, per the providers' model tables on 2026-09-28. Every catalog model allows 128K output tokens.
MODEL_WINDOW_TOKENS = 1_000_000
OUTPUT_RESERVE_TOKENS = 128_000
# About 2.5 Unicode characters per token on current tokenizers; three UTF-8 bytes per token stays pessimistic for
# ASCII prose and JSON while counting a multi-byte character as roughly one token.
BYTES_PER_TOKEN = 3


class ContextStateError(RuntimeError):
    """Persisted conversation state cannot be decomposed into whole exchanges."""


class ContextWindowError(ValueError):
    """The fixed prompt, tools, and current exchange alone exceed the smallest model window."""


def estimated_tokens(value: object) -> int:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    return -(-len(encoded) // BYTES_PER_TOKEN)


def message_tokens(message: BaseMessage) -> int:
    return estimated_tokens(message.model_dump(mode="json", include={"content", "tool_calls", "invalid_tool_calls"}))


def exchanges(messages: Sequence[BaseMessage]) -> list[tuple[BaseMessage, ...]]:
    """Split history at user messages; a message without an id or history that precedes any user message fails."""
    groups: list[list[BaseMessage]] = []
    for message in messages:
        if not isinstance(message, BaseMessage) or not isinstance(message.id, str) or not message.id:
            raise ContextStateError("conversation history has a message without an id")
        if isinstance(message, HumanMessage):
            groups.append([message])
        elif not groups:
            raise ContextStateError("conversation history does not start with a user message")
        else:
            groups[-1].append(message)
    return [tuple(group) for group in groups]


def message_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    text: list[str] = []
    for block in value:
        if isinstance(block, str):
            text.append(block)
        elif isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str):
            text.append(str(block["text"]))
    return "\n".join(text)


def final_reply(message: object) -> str:
    """Return the user-facing reply a message ends a turn with, or "" when it cannot end one.

    The runtime accepts a turn's result and this module keeps an exchange by this one criterion.
    """
    if not isinstance(message, AIMessage) or message.tool_calls or message.invalid_tool_calls:
        return ""
    return message_text(message.content).strip()


def completed(exchange: Sequence[BaseMessage]) -> bool:
    """An exchange is complete only when it ends in an accepted reply; a failed turn leaves it unfinished."""
    return bool(final_reply(exchange[-1]))


def ensure_window(fixed_tokens: int, conversation_tokens: int) -> None:
    if fixed_tokens + conversation_tokens + OUTPUT_RESERVE_TOKENS > MODEL_WINDOW_TOKENS:
        raise ContextWindowError("conversation context exceeds the model window")


def history_to_drop(
    history: Sequence[BaseMessage],
    fixed_tokens: int,
    current_tokens: int,
    forgotten: Callable[[BaseMessage], bool] = lambda _message: False,
) -> tuple[BaseMessage, ...]:
    """Return what a new turn must forget: every unfinished exchange, and completed ones beyond the newest run kept.

    An unfinished exchange is a failed turn (a lone user message or an unanswered Action round); resending it would
    replay a request the user already saw fail. An exchange whose user message is ``forgotten``, such as one that read
    attachments, is forgotten whole however recent it is (ADR-0093).
    """
    ensure_window(fixed_tokens, current_tokens)
    groups = exchanges(history)
    kept: set[int] = set()
    used = 0
    for index in reversed(range(len(groups))):
        if not completed(groups[index]) or forgotten(groups[index][0]):
            continue
        size = sum(message_tokens(message) for message in groups[index])
        if (
            len(kept) == MAX_HISTORY_EXCHANGES
            or used + size > HISTORY_BUDGET_TOKENS
            or fixed_tokens + used + size + current_tokens + OUTPUT_RESERVE_TOKENS > MODEL_WINDOW_TOKENS
        ):
            break
        kept.add(index)
        used += size
    return tuple(message for index, group in enumerate(groups) if index not in kept for message in group)


def fixed_tokens(system_prompt: str, tools: Iterable[Mapping[str, object]]) -> int:
    return estimated_tokens(system_prompt) + sum(estimated_tokens(tool) for tool in tools)

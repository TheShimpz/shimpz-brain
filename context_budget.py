"""Bound the conversation a chat turn carries to its model.

Memory is the most recent whole exchanges: one user message and every Action round and reply that followed it; a
quoted bridge of presentation history belongs to the exchange of the turn it preceded.
A new turn keeps at most ``MAX_HISTORY_EXCHANGES`` completed ones within ``HISTORY_BUDGET_TOKENS``, a Shimpz cost and
latency policy far below any catalog model window, and forgets every unfinished one. Separately, an estimated preflight
guard refuses a call whose system prompt, tools, conversation, and output reserve would exceed ``MODEL_WINDOW_TOKENS``.
Within what is kept, the Action results of completed exchanges older than the newest ``KEEP_RESULT_EXCHANGES`` give
way to ``OMITTED_RESULT`` once they reach ``OMIT_RESULTS_TRIGGER_TOKENS`` together: a batch at a time, so the provider's
cached prompt prefix breaks rarely instead of at every turn.

Token counts are a pessimistic byte heuristic, not a tokenizer: a preflight estimate, never an exact bound.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

MAX_HISTORY_EXCHANGES = 24
HISTORY_BUDGET_TOKENS = 128_000
# The smallest context window in the model catalog: 1M tokens for Claude Opus 5.5 and Sonnet 5, 1.05M for GPT-6.1 Sol
# and Luna, per the providers' model tables on 2026-09-28. Every catalog model allows 128K output tokens.
MODEL_WINDOW_TOKENS = 1_000_000
OUTPUT_RESERVE_TOKENS = 128_000
# About 2.5 Unicode characters per token on current tokenizers; three UTF-8 bytes per token stays pessimistic for
# ASCII prose and JSON while counting a multi-byte character as roughly one token.
BYTES_PER_TOKEN = 3
# Provider termination metadata that marks output cut short: the OpenAI Responses status, the OpenAI Chat Completions
# finish reason, and the Anthropic stop reasons for the output limit and the context window.
INCOMPLETE_TERMINATION = {
    "status": frozenset({"incomplete"}),
    "finish_reason": frozenset({"length"}),
    "stop_reason": frozenset({"max_tokens", "model_context_window_exceeded"}),
}


# Measured on the 21 longest precision episode templates with GPT-6 Luna (2026-10-21, 126 episodes per arm): about
# 24% fewer input tokens and 14% fewer dollars at equal correctness and duplicate effects. Omitting at every turn
# instead spent more dollars than keeping everything, because each edit breaks the provider's cached prefix.
KEEP_RESULT_EXCHANGES = 2
OMIT_RESULTS_TRIGGER_TOKENS = 8_000
OMITTED_RESULT = (
    "[Earlier Action result omitted to save context. The call completed; any change it made already happened. Never "
    "repeat a change to recover this result. If the current request needs its data, read it again.]"
)

# The id prefix of the one user-role message quoting committed presentation history before a turn (ADR-0065).
BRIDGE_ID_PREFIX = "shimpz-context-"


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
    """Split history at user messages; a message without an id or history that precedes any user message fails.

    A bridge and the turn message right after it are one exchange, so the turn keeps the context it was read in for
    as long as it is remembered, and both are forgotten together.
    """
    groups: list[list[BaseMessage]] = []
    for message in messages:
        if not isinstance(message, BaseMessage) or not isinstance(message.id, str) or not message.id:
            raise ContextStateError("conversation history has a message without an id")
        joins = bool(groups) and len(groups[-1]) == 1 and _bridge(groups[-1][0])
        if isinstance(message, HumanMessage) and not joins:
            groups.append([message])
        elif not groups:
            raise ContextStateError("conversation history does not start with a user message")
        else:
            groups[-1].append(message)
    return [tuple(group) for group in groups]


def _bridge(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and message.id.startswith(BRIDGE_ID_PREFIX)


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


def truncated(message: object) -> bool:
    """Whether the provider reports that it cut this message short instead of finishing it.

    The adapters keep provider termination in ``response_metadata``: OpenAI Responses as ``status``, OpenAI Chat
    Completions as ``finish_reason``, and Anthropic Messages as ``stop_reason``.
    """
    metadata = getattr(message, "response_metadata", None)
    return isinstance(metadata, Mapping) and any(
        isinstance(metadata.get(key), str) and metadata[key] in values for key, values in INCOMPLETE_TERMINATION.items()
    )


def final_reply(message: object) -> str:
    """Return the user-facing reply a message ends a turn with, or "" when it cannot end one.

    The runtime accepts a turn's result and this module keeps an exchange by this one criterion; a reply the provider
    cut short never ends a turn.
    """
    if not isinstance(message, AIMessage) or message.tool_calls or message.invalid_tool_calls or truncated(message):
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
    replay a request the user already saw fail. An exchange holding a ``forgotten`` message, such as a user message
    that read attachments, is forgotten whole however recent it is (ADR-0093).
    """
    ensure_window(fixed_tokens, current_tokens)
    groups = exchanges(history)
    kept: set[int] = set()
    used = 0
    for index in reversed(range(len(groups))):
        if not completed(groups[index]) or any(forgotten(message) for message in groups[index]):
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


def results_to_omit(
    history: Sequence[BaseMessage],
    action_result: Callable[[ToolMessage], bool],
    forgotten: Callable[[BaseMessage], bool] = lambda _message: False,
) -> tuple[ToolMessage, ...]:
    """Return placeholders for the older Action results a new turn omits, or nothing below the trigger.

    Only completed exchanges older than the newest ``KEEP_RESULT_EXCHANGES`` completed ones give results; an unfinished
    or ``forgotten`` exchange is dropped whole instead. Each placeholder keeps its message's id, call pairing, and
    status, so it replaces that result in place.
    """
    groups = exchanges(history)
    done = [index for index, group in enumerate(groups) if completed(group) and not any(map(forgotten, group))]
    older = done[: max(0, len(done) - KEEP_RESULT_EXCHANGES)]
    results = [
        message
        for index in older
        for message in groups[index]
        if isinstance(message, ToolMessage) and action_result(message) and message.content != OMITTED_RESULT
    ]
    if sum(message_tokens(message) for message in results) < OMIT_RESULTS_TRIGGER_TOKENS:
        return ()
    return tuple(
        ToolMessage(
            content=OMITTED_RESULT,
            id=message.id,
            tool_call_id=message.tool_call_id,
            name=message.name,
            status=message.status,
        )
        for message in results
    )

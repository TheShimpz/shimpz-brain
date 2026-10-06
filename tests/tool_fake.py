"""Scripted tool-calling chat models and the Pinger Assistant shared by provider-free Brain tests.

They prove what the runtime does with scripted model output; they say nothing about how a real model behaves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import agent_runtime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

PROVIDER = agent_runtime.ProviderConfig("openai", "gpt-6-luna", "test-key-0123456789")
ACTION = agent_runtime.ActionDefinition(
    "ping",
    "Ping a host.",
    {"type": "object", "properties": {"host": {"type": "string"}}, "additionalProperties": False},
)
PINGER = agent_runtime.AssistantDefinition("pinger", "Pinger checks that hosts answer.", (ACTION,))


class ToolModel(FakeMessagesListChatModel):
    """Answers with its scripted messages whatever tools the runtime binds."""

    def bind_tools(self, tools: Sequence[Any], **_kwargs: Any):
        return self


class RecordingModel(ToolModel):
    """Also records every message list it is asked about; each suite subclasses it with its own ``seen`` list."""

    seen: ClassVar[list[list[Any]]]

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any):
        type(self).seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def system_text(messages: list) -> str:
    """The text of the first system message a model was asked about."""
    return next(message.content for message in messages if isinstance(message, SystemMessage))


def call(assistant: agent_runtime.AssistantDefinition, action: str, args: dict[str, object], call_id: str) -> dict:
    """A model tool call of one Assistant Action."""
    return {"name": agent_runtime._tool_name(assistant.id, action), "args": args, "id": call_id, "type": "tool_call"}


def runtime(*responses: AIMessage) -> agent_runtime.AgentRuntime:
    """An in-memory runtime whose every model answers with ``responses`` in order."""
    model = ToolModel(responses=list(responses))
    return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)

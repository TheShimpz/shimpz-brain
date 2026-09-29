"""Anthropic prompt-cache markers on the real provider payload of an ordinary Brain turn.

The provider request is intercepted at the adapter's single send method, so the payload is exactly what the
Anthropic SDK would receive; no network, key, or provider call is used.
"""

from __future__ import annotations

import unittest
from unittest import mock

import agent_runtime
from anthropic.types import Message, TextBlock, Usage
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

CACHE = {"type": "ephemeral", "ttl": "5m"}
CLOCK = agent_runtime.AssistantDefinition(
    "clock",
    "Clock reads the current time.",
    (
        agent_runtime.ActionDefinition(
            "read-time", "Read the current time.", {"type": "object", "properties": {}, "additionalProperties": False}
        ),
    ),
)


def _reply(text: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=text)],
        model="claude-sonnet-5-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=10, output_tokens=2),
    )


class PromptCachingTests(unittest.TestCase):
    def _turns(self, provider: str, model: str) -> list[dict]:
        payloads: list[dict] = []

        def send(_model, payload):
            payloads.append(payload)
            return _reply("Olá!")

        context = agent_runtime.TurnContext(
            "cache-thread",
            "Cache Team",
            (CLOCK,),
            agent_runtime.ProviderConfig(provider, model, "sk-test-0123456789abcdef"),
        )
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=agent_runtime.provider_model)
        with mock.patch.object(ChatAnthropic, "_create", send):
            first = runtime.start(context, "oi")
            second = runtime.start(context, "tudo bem?")
        self.assertEqual((first.status, first.reply, second.reply), ("completed", "Olá!", "Olá!"))
        return payloads

    def test_anthropic_turn_marks_the_system_prompt_tools_and_conversation_prefix(self):
        payloads = self._turns("anthropic", "claude-sonnet-5-5")
        self.assertEqual(len(payloads), 2)
        for payload in payloads:
            with self.subTest(messages=len(payload["messages"])):
                self.assertEqual(payload["cache_control"], CACHE)
                self.assertEqual(payload["system"][-1]["cache_control"], CACHE)
                self.assertIn("Enabled Assistant contracts", payload["system"][-1]["text"])
                self.assertEqual(payload["tools"][-1]["cache_control"], CACHE)
        self.assertEqual(
            [(message["role"], message["content"]) for message in payloads[1]["messages"]],
            [("user", "oi"), ("assistant", "Olá!"), ("user", "tudo bem?")],
        )

    def test_only_anthropic_turns_receive_the_caching_middleware(self):
        openai = agent_runtime.ProviderConfig("openai", "gpt-6-sol", "sk-test-0123456789abcdef")
        anthropic = agent_runtime.ProviderConfig("anthropic", "claude-sonnet-5-5", "sk-test-0123456789abcdef")
        self.assertEqual(agent_runtime._prompt_caching(openai), [])
        (middleware,) = agent_runtime._prompt_caching(anthropic)
        self.assertEqual((middleware.ttl, middleware.unsupported_model_behavior), ("5m", "raise"))


if __name__ == "__main__":
    unittest.main()

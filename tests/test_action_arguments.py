"""Malformed Action arguments are corrected inside the graph and never reach Team (ADR-0080)."""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import agent_runtime
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import ToolAwareFakeModel

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 2},
        "search_type": {"type": "string", "enum": ["auto", "fast"]},
    },
    "required": ["query", "search_type"],
    "additionalProperties": False,
}
ACTION = agent_runtime.ActionDefinition(id="search-web", summary="Search the web.", input_schema=SCHEMA)
TOOL = agent_runtime._tool_name("shimpz-exa", "search-web")
SECRETISH = "sk-leaked-value-0123456789"


def _call(args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": TOOL, "args": args, "id": call_id, "type": "tool_call"}])


def _context() -> agent_runtime.TurnContext:
    return agent_runtime.TurnContext(
        thread_id="team:args:thread",
        team_name="Research",
        assistants=(
            agent_runtime.AssistantDefinition(
                id="shimpz-exa",
                genesis="Search the web and cite sources when the answer needs current information.",
                actions=(ACTION,),
            ),
        ),
        provider=agent_runtime.ProviderConfig("openai", "gpt-6-luna", "secret-test-key"),
    )


class ToolPreflightTests(unittest.TestCase):
    def test_invalid_arguments_return_a_closed_correction_and_never_interrupt(self):
        with mock.patch("langgraph.types.interrupt") as interrupt:
            tool = agent_runtime._request_action("shimpz-exa", ACTION)
            for args in (
                {"query": "news today"},  # missing required
                {"query": "news today", "search_type": "auto", "extra": SECRETISH},  # unknown property
                {"query": "news today", "search_type": SECRETISH},  # value outside the enum
                {"query": 7, "search_type": "auto"},  # wrong type
            ):
                with self.subTest(args=sorted(args)):
                    message = tool.func(**args)
                    self.assertTrue(message.startswith("Action not executed:"))
                    self.assertIn("query, search_type", message)
                    self.assertNotIn(SECRETISH, message)
                    self.assertNotIn("extra", message)
                    self.assertLessEqual(len(message), agent_runtime.MAX_ARGUMENT_CORRECTION_CHARS)
            interrupt.assert_not_called()
            tool.func(query="news today", search_type="auto")
        interrupt.assert_called_once_with(
            {
                "kind": "action",
                "assistant_id": "shimpz-exa",
                "action": "search-web",
                "input": {"query": "news today", "search_type": "auto"},
            }
        )

    def test_the_correction_is_bounded_and_lists_only_declared_required_names(self):
        wide = agent_runtime.ActionDefinition(
            id="wide",
            summary="Wide input.",
            input_schema={
                "type": "object",
                "properties": {f"field_{i}_{'x' * 80}": {"type": "string"} for i in range(20)},
                "required": [f"field_{i}_{'x' * 80}" for i in range(20)],
            },
        )
        message = agent_runtime._invalid_arguments("tool", wide)
        self.assertLessEqual(len(message), agent_runtime.MAX_ARGUMENT_CORRECTION_CHARS)
        self.assertEqual(
            agent_runtime._invalid_arguments(
                "tool",
                agent_runtime.ActionDefinition(
                    id="none", summary="No required input.", input_schema={"type": "object"}
                ),
            ).count("(none)"),
            1,
        )

    def test_an_invalid_input_schema_is_a_contract_error(self):
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Action input schema"):
            agent_runtime.ActionDefinition(
                id="broken", summary="Broken schema.", input_schema={"type": "object", "properties": 3}
            )


class GraphCorrectionTests(unittest.TestCase):
    def test_the_model_corrects_itself_and_only_the_valid_call_suspends(self):
        model = ToolAwareFakeModel(
            responses=[
                _call({"query": "new AI models today"}, "call-1"),
                _call({"query": "new AI models today", "search_type": "auto"}, "call-2"),
                AIMessage(content="Found them."),
            ]
        )
        saver = InMemorySaver()
        runtime = agent_runtime.AgentRuntime(saver, model_factory=lambda _config: model)
        context = _context()

        suspended = runtime.start(context, "Quais modelos de IA saíram hoje?")

        self.assertEqual(suspended.status, "action-required")
        (request,) = suspended.actions
        self.assertEqual(request.input, {"query": "new AI models today", "search_type": "auto"})
        messages = saver.get_tuple({"configurable": {"thread_id": context.thread_id}}).checkpoint["channel_values"][
            "messages"
        ]
        corrections = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual(len(corrections), 1)
        self.assertTrue(str(corrections[0].content).startswith("Action not executed:"))
        completed = runtime.resume(context, {request.interrupt_id: {"query": "x", "results": []}})
        self.assertEqual(completed.reply, "Found them.")

    def test_endless_invalid_calls_end_at_the_graph_limit_without_an_action(self):
        model = ToolAwareFakeModel(responses=[_call({"query": "q"}, f"call-{i}") for i in range(20)])
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
        with (
            mock.patch("langgraph.types.interrupt") as interrupt,
            self.assertRaises(agent_runtime.ProviderRequestError),
        ):
            runtime.start(_context(), "Busque algo")
        interrupt.assert_not_called()


if __name__ == "__main__":
    unittest.main()

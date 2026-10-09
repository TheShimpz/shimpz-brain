"""OpenAI receives every Action tool non-strict with its declared ``required`` list; Anthropic binding is unchanged.

Sent without ``strict``, an OpenAI Responses function tool is treated as strict and every property becomes required,
so the model invents values for optional Action properties (ADR-0094). The adapters run over an
`httpx.MockTransport`, so the bodies asserted here are exactly what would be sent: no network, key, or provider call.
"""

import copy
import json
import unittest

import agent_runtime
import clarification
import httpx
import memory
import provider_client
from langgraph.checkpoint.memory import InMemorySaver

import routine

LOOKUP = {
    "type": "object",
    "properties": {
        "zone": {"type": "string"},
        "record_type": {"type": "string", "enum": ["A", "AAAA", "TXT"]},
        "limit": {"type": "integer", "minimum": 1},
    },
    "required": ["zone"],
    "additionalProperties": False,
}
SEARCH = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "due_date": {"type": "string"}},
    "required": [],
    "additionalProperties": False,
}
DNS = agent_runtime.AssistantDefinition(
    "dns",
    "DNS reads zones.",
    (
        agent_runtime.ActionDefinition("list-records", "List a zone's records.", LOOKUP),
        agent_runtime.ActionDefinition("search-zones", "Search zones.", SEARCH),
    ),
)
DECLARED = {
    agent_runtime._tool_name("dns", "list-records"): LOOKUP,
    agent_runtime._tool_name("dns", "search-zones"): SEARCH,
}
BRAIN_TOOLS = {clarification.TOOL_NAME, memory.TOOL_NAME, routine.TOOL_NAME}
REPLY = {
    "openai": {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-6.1-sol",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    },
    "anthropic": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": "Done."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    },
}
MODELS = {"openai": "gpt-6.1-sol", "anthropic": "claude-sonnet-5-5"}


def _sent_tools(provider: str) -> list[dict]:
    """Start one turn offering both Actions and every Brain tool; return the tools of the one provider request."""
    bodies: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=REPLY[provider])

    client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        runtime = agent_runtime.AgentRuntime(
            InMemorySaver(), model_factory=lambda config: provider_client.provider_model(config, http_client=client)
        )
        config = agent_runtime.ProviderConfig(provider, MODELS[provider], "sk-test-0123456789abcdef", "low")
        context = agent_runtime.TurnContext(
            f"binding-{provider}", "DNS Team", (DNS,), config, memories=(), routines=(), routine_capacity=20_000
        )
        result = runtime.start(context, "List the records of example.com.")
    finally:
        client.close()
    assert result.reply == "Done.", result
    (body,) = bodies
    return body["tools"]


class OpenAIToolBindingTests(unittest.TestCase):
    def test_every_action_tool_is_sent_non_strict_with_its_declared_required_list(self) -> None:
        tools = {tool["name"]: tool for tool in _sent_tools("openai")}
        self.assertEqual(set(tools), set(DECLARED) | BRAIN_TOOLS)
        for name, schema in DECLARED.items():
            with self.subTest(tool=name):
                self.assertEqual(tools[name]["type"], "function")
                self.assertIs(tools[name]["strict"], False)
                self.assertEqual(tools[name]["parameters"], schema)
                self.assertEqual(tools[name]["parameters"]["required"], schema["required"])

    def test_brain_tools_keep_the_provider_default(self) -> None:
        tools = {tool["name"]: tool for tool in _sent_tools("openai")}
        for name in BRAIN_TOOLS:
            with self.subTest(tool=name):
                self.assertNotIn("strict", tools[name])

    def test_binding_leaves_the_declared_schema_unchanged(self) -> None:
        before = copy.deepcopy((LOOKUP, SEARCH))
        _sent_tools("openai")
        self.assertEqual((LOOKUP, SEARCH), before)

    def test_anthropic_tools_are_unchanged(self) -> None:
        tools = {tool["name"]: tool for tool in _sent_tools("anthropic")}
        self.assertEqual(set(tools), set(DECLARED) | BRAIN_TOOLS)
        for name, tool in tools.items():
            with self.subTest(tool=name):
                self.assertNotIn("strict", tool)
        for name, schema in DECLARED.items():
            with self.subTest(tool=name):
                self.assertEqual(tools[name]["input_schema"], schema)


if __name__ == "__main__":
    unittest.main()

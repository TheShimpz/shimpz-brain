"""Every OpenAI Responses call is unstored and replays its reasoning as encrypted content.

The adapter runs over an `httpx.MockTransport`, so the bodies asserted here are exactly what would be sent: no network,
key, or provider call is used.
"""

from __future__ import annotations

import json
import unittest

import agent_runtime
import httpx
import provider_client
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

SCHEMA = {
    "type": "object",
    "properties": {"zone": {"type": "string"}},
    "required": ["zone"],
    "additionalProperties": False,
}
DNS = agent_runtime.AssistantDefinition(
    "dns", "DNS reads zones.", (agent_runtime.ActionDefinition("read-zone", "Read one zone.", SCHEMA),)
)
CONFIG = agent_runtime.ProviderConfig("openai", "gpt-6.1-sol", "sk-test-0123456789abcdef", "low")
REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA-sealed-reasoning"}


class _Responses:
    """A Responses endpoint whose first answer reasons and calls the Action, and whose second replies."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        index = len(self.bodies)
        if index == 1:
            output = [
                REASONING,
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": agent_runtime._tool_name("dns", "read-zone"),
                    "arguments": json.dumps({"zone": "example.com"}),
                    "status": "completed",
                },
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": f"msg_{index}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The zone is active.", "annotations": []}],
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": f"resp_{index}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-6.1-sol",
                "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        )


class OpenAIStoreTests(unittest.TestCase):
    def test_chat_and_decision_calls_are_unstored_and_ask_for_encrypted_reasoning(self) -> None:
        for decision in (False, True):
            with self.subTest(decision=decision):
                model = provider_client.provider_model(CONFIG, http_client=httpx.Client(), decision=decision)
                payload = model._get_request_payload([HumanMessage(content="hi")])
                self.assertIs(payload["store"], False)
                self.assertEqual(payload["include"], ["reasoning.encrypted_content"])

    def test_a_resumed_turn_replays_encrypted_reasoning_without_server_state(self) -> None:
        responses = _Responses()
        client = httpx.Client(transport=httpx.MockTransport(responses))
        runtime = agent_runtime.AgentRuntime(
            InMemorySaver(), model_factory=lambda config: provider_client.provider_model(config, http_client=client)
        )
        context = agent_runtime.TurnContext("store-thread", "DNS Team", (DNS,), CONFIG)
        paused = runtime.start(context, '{"files":[],"message":"Is example.com active?"}')
        self.assertEqual(paused.status, "action-required")
        (request,) = paused.actions
        result = runtime.resume(context, {request.interrupt_id: {"status": "active"}})
        self.assertEqual(result.reply, "The zone is active.")

        first, second = responses.bodies
        for body in (first, second):
            self.assertIs(body["store"], False)
            self.assertEqual(body["include"], ["reasoning.encrypted_content"])
            self.assertNotIn("previous_response_id", body)
        replayed = [item for item in second["input"] if item.get("type") == "reasoning"]
        self.assertEqual(len(replayed), 1)
        self.assertEqual(replayed[0]["encrypted_content"], "gAAAA-sealed-reasoning")
        calls = [item for item in second["input"] if item.get("type") == "function_call"]
        outputs = [item for item in second["input"] if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in calls], ["call_1"])
        self.assertEqual([item["call_id"] for item in outputs], ["call_1"])
        self.assertFalse(any(str(item.get("id", "")).startswith("msg_") for item in second["input"]))


if __name__ == "__main__":
    unittest.main()

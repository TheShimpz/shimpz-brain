"""Provider-native structured output through the real OpenAI and Anthropic adapters.

Each adapter's single generation or send method is intercepted, so the adapter's own schema binding and parser run;
no network, key, or provider call is used.
"""

from __future__ import annotations

import unittest
from unittest import mock

import action_labels
import agent_runtime
import capability_plan
import provider_client
import structured
from anthropic.types import Message, TextBlock, Usage
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

KEY = "sk-test-0123456789abcdef"
CANDIDATES = (
    capability_plan.CapabilityCandidate("shimpz-cloudflare", "Shimpz Cloudflare", "Manage DNS.", ("dns.read",), ()),
)
PLAN = {"status": "install-required", "assistant_ids": ["shimpz-cloudflare"]}
LABELS = {"labels": [{"id": "list-zones", "label": "Listar zonas"}]}


def _runtime() -> agent_runtime.AgentRuntime:
    return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=provider_client.provider_model)


def _anthropic_reply(text: str | None, stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg_test",
        content=[] if text is None else [TextBlock(type="text", text=text)],
        model="claude-sonnet-5-5",
        role="assistant",
        stop_reason=stop_reason,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


class AnthropicStructuredOutputTests(unittest.TestCase):
    provider = agent_runtime.ProviderConfig("anthropic", "claude-sonnet-5-5", KEY)

    def _run(self, reply: Message, call):
        payloads: list[dict] = []

        def send(_model, payload):
            payloads.append(payload)
            return reply

        with mock.patch.object(ChatAnthropic, "_create", send):
            result = call(_runtime())
        self.assertEqual(payloads[0]["output_config"]["format"]["type"], "json_schema")
        self.assertNotIn("tools", payloads[0])
        return result

    def test_plans_and_labels_parse_the_native_schema_reply(self):
        import json

        plan = self._run(
            _anthropic_reply(json.dumps(PLAN)),
            lambda runtime: runtime.capability_plan(self.provider, "List zones", CANDIDATES),
        )
        self.assertEqual(plan, capability_plan.CapabilityPlan("install-required", ("shimpz-cloudflare",)))
        labels = self._run(
            _anthropic_reply(json.dumps(LABELS)),
            lambda runtime: runtime.action_labels(self.provider, "pt", ("list-zones",)),
        )
        self.assertEqual(labels, (action_labels.ActionLabel("list-zones", "Listar zonas"),))

    def test_malformed_unknown_and_refused_replies_are_response_failures(self):
        for reply in (
            _anthropic_reply("not json"),
            _anthropic_reply('{"status":"install-required","assistant_ids":["unknown"]}'),
            _anthropic_reply('{"status":"install-required","assistant_ids":["shimpz-cloudflare"],"extra":1}'),
            _anthropic_reply(None, "refusal"),
            _anthropic_reply(
                '{"status":"sufficient","status":"install-required","assistant_ids":["shimpz-cloudflare"]}'
            ),
            _anthropic_reply('{"status":"install-required",' + " " * 5000 + '"assistant_ids":["shimpz-cloudflare"]}'),
        ):
            with self.subTest(reply=reply.content), self.assertRaises(agent_runtime.ProviderResponseError):
                self._run(reply, lambda runtime: runtime.capability_plan(self.provider, "List zones", CANDIDATES))


class SchemaTests(unittest.TestCase):
    def test_provider_schemas_carry_no_undocumented_string_limits(self):
        import json

        for schema in (capability_plan.StructuredPlan, action_labels.ActionLabelsOutput):
            with self.subTest(schema=schema.__name__):
                encoded = json.dumps(schema.model_json_schema())
                self.assertNotIn("maxLength", encoded)
                self.assertNotIn("minLength", encoded)
                self.assertNotIn("pattern", encoded)

    def test_binding_and_raw_text_helpers_fail_closed(self):
        model = mock.Mock()
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "unsupported model provider"):
            structured.structured_output(model, "gemini", capability_plan.StructuredPlan)
        model.with_structured_output.assert_not_called()
        self.assertEqual(structured._raw_text(object()), "")
        self.assertEqual(
            structured._raw_text([{"type": "reasoning"}, {"type": "text", "text": "{}"}, {"type": "text"}]), "{}"
        )


class OpenAIStructuredOutputTests(unittest.TestCase):
    provider = agent_runtime.ProviderConfig("openai", "gpt-6.1-sol", KEY)

    def _run(self, additional_kwargs: dict, call):
        requests: list[dict] = []

        def generate(_model, _messages, *_args, **kwargs):
            requests.append(kwargs)
            message = AIMessage(content="", additional_kwargs=additional_kwargs)
            return ChatResult(generations=[ChatGeneration(message=message)])

        with mock.patch.object(ChatOpenAI, "_generate", generate):
            result = call(_runtime())
        # A Pydantic schema reaches the OpenAI SDK's typed parse, which always requests strict JSON-schema output.
        self.assertIn(
            requests[0]["response_format"], {capability_plan.StructuredPlan, action_labels.ActionLabelsOutput}
        )
        return result

    def test_plans_and_labels_use_the_strict_parsed_value(self):
        plan = self._run(
            {"parsed": PLAN}, lambda runtime: runtime.capability_plan(self.provider, "List zones", CANDIDATES)
        )
        self.assertEqual(plan, capability_plan.CapabilityPlan("install-required", ("shimpz-cloudflare",)))
        labels = self._run(
            {"parsed": LABELS}, lambda runtime: runtime.action_labels(self.provider, "pt", ("list-zones",))
        )
        self.assertEqual(labels, (action_labels.ActionLabel("list-zones", "Listar zonas"),))

    def test_refusal_and_semantic_failures_are_response_failures(self):
        for kwargs in (
            {"refusal": "cannot comply"},
            {"parsed": {"status": "install-required", "assistant_ids": ["unknown"]}},
            {"parsed": {"status": "sufficient", "assistant_ids": ["shimpz-cloudflare"]}},
            {},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(agent_runtime.ProviderResponseError):
                self._run(kwargs, lambda runtime: runtime.capability_plan(self.provider, "List zones", CANDIDATES))
        with self.assertRaises(agent_runtime.ProviderResponseError):
            self._run(
                {"parsed": {"labels": [{"id": "list-zones", "label": "Listar\nzonas"}]}},
                lambda runtime: runtime.action_labels(self.provider, "pt", ("list-zones",)),
            )


if __name__ == "__main__":
    unittest.main()

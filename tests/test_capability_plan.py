from __future__ import annotations

import re
import unittest
from typing import Any, ClassVar
from unittest import mock

import agent_runtime
import capability_plan
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class RecordingModel(FakeMessagesListChatModel):
    seen_messages: ClassVar[list[list[Any]]] = []

    def bind_tools(self, *_args: Any, **_kwargs: Any):
        raise AssertionError("capability planning must not bind tools")

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any):
        type(self).seen_messages.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def provider() -> agent_runtime.ProviderConfig:
    return agent_runtime.ProviderConfig("openai", "gpt-6-sol", "secret-test-key")


def candidates() -> tuple[capability_plan.CapabilityCandidate, ...]:
    return (
        capability_plan.CapabilityCandidate(
            id="shimpz-cloudflare",
            name="Shimpz Cloudflare",
            summary="Manage DNS zones and records.",
            actions=("dns.read", "dns.write"),
            integrations=(capability_plan.CapabilityIntegration("cloudflare", "cloudflare"),),
        ),
        capability_plan.CapabilityCandidate(
            id="shimpz-whatsapp",
            name="Shimpz WhatsApp",
            summary="Send reviewed WhatsApp messages.",
            actions=("messages.send",),
            integrations=(capability_plan.CapabilityIntegration("whatsapp", "whatsapp"),),
        ),
    )


class CapabilityPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingModel.seen_messages = []

    def test_uses_no_tools_or_checkpoint_and_returns_exact_subset(self):
        class RejectCheckpointAccess:
            def __getattr__(self, name):
                raise AssertionError(f"unexpected checkpoint access: {name}")

        model = RecordingModel(
            responses=[
                AIMessage(
                    content=('{"status":"install-required","assistant_ids":["shimpz-cloudflare","shimpz-whatsapp"]}')
                )
            ]
        )
        runtime = agent_runtime.AgentRuntime(RejectCheckpointAccess(), model_factory=lambda _config: model)

        plan = runtime.capability_plan(
            provider(),
            "Configure um domínio\ne envie o resultado por WhatsApp",
            candidates(),
        )

        self.assertEqual(
            plan,
            capability_plan.CapabilityPlan(
                "install-required",
                ("shimpz-cloudflare", "shimpz-whatsapp"),
            ),
        )
        self.assertEqual(len(RecordingModel.seen_messages), 1)
        provider_text = "\n".join(str(item.content) for item in RecordingModel.seen_messages[0])
        self.assertIn("untrusted data", provider_text)
        self.assertIn("Configure um domínio", provider_text)
        self.assertIn('"id":"shimpz-whatsapp"', provider_text)
        self.assertNotIn("source_digest", provider_text)
        self.assertNotIn("input_schema", provider_text)
        self.assertNotIn("genesis", provider_text.casefold())

    def test_accepts_a_closed_sufficient_result(self):
        model = RecordingModel(responses=[AIMessage(content='{"status":"sufficient","assistant_ids":[]}')])
        runtime = agent_runtime.AgentRuntime(object(), model_factory=lambda _config: model)
        shortlist = candidates()
        with mock.patch.object(capability_plan, "_inputs", wraps=capability_plan._inputs) as validate:
            self.assertEqual(
                runtime.capability_plan(provider(), "Explain DNS", shortlist),
                capability_plan.CapabilityPlan("sufficient"),
            )
        validate.assert_called_once_with("Explain DNS", shortlist)

    def test_rejects_unknown_duplicate_unsorted_added_and_oversized_results(self):
        responses = (
            '{"status":"install-required","assistant_ids":["unknown"]}',
            ('{"status":"install-required","assistant_ids":["shimpz-cloudflare","shimpz-cloudflare"]}'),
            ('{"status":"install-required","assistant_ids":["shimpz-whatsapp","shimpz-cloudflare"]}'),
            '{"status":"sufficient","assistant_ids":["shimpz-cloudflare"]}',
            '{"status":"install-required","assistant_ids":[]}',
            '{"status":"install-required","assistant_ids":["shimpz-cloudflare"],"extra":true}',
            ('{"status":"install-required","assistant_ids":["shimpz-cloudflare"],"assistant_ids":["shimpz-whatsapp"]}'),
            "not-json",
        )
        for content in responses:
            with self.subTest(content=content):
                model = RecordingModel(responses=[AIMessage(content=content)])
                runtime = agent_runtime.AgentRuntime(
                    object(),
                    model_factory=lambda _config, selected=model: selected,
                )
                with self.assertRaisesRegex(agent_runtime.ProviderResponseError, "model provider response failed"):
                    runtime.capability_plan(provider(), "Configure DNS", candidates())

        five = tuple(
            capability_plan.CapabilityCandidate(
                id=f"assistant-{index}",
                name=f"Assistant {index}",
                summary="Bounded capability.",
                actions=(f"action-{index}",),
                integrations=(),
            )
            for index in range(5)
        )
        output = (
            '{"status":"install-required","assistant_ids":['
            + ",".join(f'"assistant-{index}"' for index in range(5))
            + "]}"
        )
        model = RecordingModel(responses=[AIMessage(content=output)])
        with self.assertRaises(agent_runtime.ProviderResponseError):
            agent_runtime.AgentRuntime(object(), model_factory=lambda _config: model).capability_plan(
                provider(),
                "Use all capabilities",
                five,
            )

    def test_invalid_inputs_fail_before_provider_access(self):
        factory = mock.Mock()
        runtime = agent_runtime.AgentRuntime(object(), model_factory=factory)
        invalid = (
            ("", candidates()),
            ("hidden\0instruction", candidates()),
            ("Configure DNS", ()),
            ("Configure DNS", tuple(reversed(candidates()))),
            ("Configure DNS", (candidates()[0], candidates()[0])),
            (
                "Configure DNS",
                (
                    capability_plan.CapabilityCandidate(
                        id="assistant",
                        name="Assistant",
                        summary="Capability.",
                        actions=("../shell",),
                        integrations=(),
                    ),
                ),
            ),
        )
        for objective, shortlist in invalid:
            with (
                self.subTest(objective=objective, shortlist=shortlist),
                self.assertRaises(agent_runtime.RuntimeContractError),
            ):
                runtime.capability_plan(provider(), objective, shortlist)
        factory.assert_not_called()

    def test_candidate_shape_failures_are_rejected_before_provider_access(self):
        first = candidates()[0]
        invalid = (
            (object(),),
            (capability_plan.CapabilityCandidate(first.id, first.name, first.summary, (), first.integrations),),
            (capability_plan.CapabilityCandidate(first.id, first.name, first.summary, first.actions, (object(),)),),
        )
        for shortlist in invalid:
            with (
                self.subTest(candidate_type=type(shortlist[0])),
                self.assertRaises(capability_plan.CapabilityPlanError),
            ):
                capability_plan.validate_inputs("Configure DNS", shortlist)

    def test_candidate_identifiers_use_the_current_canonical_pattern(self):
        shortlist = candidates()
        self.assertEqual(capability_plan.validate_inputs("Configure DNS", shortlist)[1], shortlist)
        with (
            mock.patch.object(agent_runtime, "ACTION_ID_RE", re.compile(r"never-match\Z")),
            self.assertRaisesRegex(capability_plan.CapabilityPlanError, "invalid Action id"),
        ):
            capability_plan.validate_inputs("Configure DNS", shortlist)
        self.assertEqual(capability_plan.validate_inputs("Configure DNS", shortlist)[1], shortlist)

    def test_provider_content_envelopes_and_failures_are_closed(self):
        valid = '{"status":"install-required","assistant_ids":["shimpz-cloudflare"]}'
        text_block = RecordingModel(
            responses=[
                AIMessage(
                    content=[
                        {"type": "reasoning", "id": "reasoning-1", "summary": []},
                        {"type": "compaction", "id": "compaction-1", "encrypted_content": "opaque"},
                        {
                            "type": "text",
                            "text": valid,
                            "annotations": [],
                            "id": "message-1",
                            "phase": "final_answer",
                        },
                        {"type": "future-provider-metadata", "value": "ignored"},
                    ]
                )
            ]
        )
        self.assertEqual(
            capability_plan.create(lambda: text_block, "Configure DNS", candidates()),
            capability_plan.CapabilityPlan("install-required", ("shimpz-cloudflare",)),
        )

        invalid_messages = (
            AIMessage(
                content=valid,
                tool_calls=[{"name": "unexpected", "args": {}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(
                content=[
                    {"type": "text", "text": valid, "annotations": [], "id": "message-1"},
                    {"type": "text", "text": valid, "annotations": [], "id": "message-2"},
                ]
            ),
            AIMessage(content=[{"type": "refusal", "refusal": "cannot comply", "id": "message-1"}]),
            AIMessage(content=[{"type": "text", "text": 1, "annotations": [], "id": "message-1"}]),
            AIMessage(content=[{"type": "reasoning", "id": "reasoning-1", "summary": []}]),
            AIMessage(content=""),
            AIMessage(content='{"status":"unknown","assistant_ids":[]}'),
        )
        for message in invalid_messages:
            with self.subTest(content=message.content), self.assertRaises(capability_plan.CapabilityPlanResponseError):
                factory = mock.Mock(return_value=RecordingModel(responses=[message]))
                capability_plan.create(factory, "Configure DNS", candidates())

        for failure, expected in (
            (ImportError("missing dependency"), ImportError),
            (RuntimeError("secret provider detail"), capability_plan.CapabilityPlanProviderError),
        ):
            model = mock.Mock()
            model.invoke.side_effect = failure
            with self.subTest(failure=type(failure).__name__), self.assertRaises(expected):
                capability_plan.create(mock.Mock(return_value=model), "Configure DNS", candidates())

        model = mock.Mock()
        model.invoke.return_value = object()
        with self.assertRaises(capability_plan.CapabilityPlanResponseError):
            capability_plan.create(lambda: model, "Configure DNS", candidates())

    def test_create_validates_before_factory_and_calls_factory_once(self):
        factory = mock.Mock(side_effect=RuntimeError("secret provider detail"))
        with self.assertRaisesRegex(capability_plan.CapabilityPlanError, "invalid capability objective"):
            capability_plan.create(factory, "", candidates())
        factory.assert_not_called()

        with self.assertRaisesRegex(capability_plan.CapabilityPlanProviderError, "^model provider request failed$"):
            capability_plan.create(factory, "Configure DNS", candidates())
        factory.assert_called_once_with()

        model = RecordingModel(responses=[AIMessage(content='{"status":"sufficient","assistant_ids":[]}')])
        factory = mock.Mock(return_value=model)
        self.assertEqual(
            capability_plan.create(factory, "Configure DNS", candidates()),
            capability_plan.CapabilityPlan("sufficient"),
        )
        factory.assert_called_once_with()

        missing = mock.Mock(side_effect=ImportError("missing dependency"))
        with self.assertRaisesRegex(ImportError, "missing dependency"):
            capability_plan.create(missing, "Configure DNS", candidates())

    def test_provider_factory_failure_is_redacted(self):
        provider_failure = mock.Mock()
        provider_failure.invoke.side_effect = RuntimeError("secret provider detail")
        runtime = agent_runtime.AgentRuntime(
            object(),
            model_factory=lambda _config: provider_failure,
        )
        with self.assertRaisesRegex(agent_runtime.ProviderRequestError, "^model provider request failed$"):
            runtime.capability_plan(provider(), "Configure DNS", candidates())

        runtime = agent_runtime.AgentRuntime(
            object(),
            model_factory=mock.Mock(side_effect=RuntimeError("secret provider detail")),
        )

        with self.assertRaisesRegex(agent_runtime.ProviderRequestError, "^model provider request failed$"):
            runtime.capability_plan(provider(), "Configure DNS", candidates())

        missing = agent_runtime.AgentRuntime(
            object(),
            model_factory=mock.Mock(side_effect=ImportError("missing dependency")),
        )
        with self.assertRaisesRegex(ImportError, "missing dependency"):
            missing.capability_plan(provider(), "Configure DNS", candidates())

    def test_unexpected_planner_failure_is_redacted(self):
        class FailingActions:
            def __iter__(self):
                raise RuntimeError("private capability marker")

        first = candidates()[0]
        malformed = capability_plan.CapabilityCandidate(
            id=first.id,
            name=first.name,
            summary=first.summary,
            actions=FailingActions(),
            integrations=first.integrations,
        )
        factory = mock.Mock()
        runtime = agent_runtime.AgentRuntime(object(), model_factory=factory)

        with self.assertRaises(agent_runtime.ProviderRequestError) as raised:
            runtime.capability_plan(provider(), "Configure DNS", (malformed,))

        self.assertEqual(str(raised.exception), "model provider request failed")
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(str(raised.exception.__cause__), "private capability marker")
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

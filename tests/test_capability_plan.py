from __future__ import annotations

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
    return agent_runtime.ProviderConfig("openai", "gpt-5.6-terra", "secret-test-key")


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
                    content=(
                        '{"status":"install-required","assistant_ids":'
                        '["shimpz-cloudflare","shimpz-whatsapp"]}'
                    )
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
        model = RecordingModel(
            responses=[AIMessage(content='{"status":"sufficient","assistant_ids":[]}')]
        )
        runtime = agent_runtime.AgentRuntime(object(), model_factory=lambda _config: model)

        self.assertEqual(
            runtime.capability_plan(provider(), "Explain DNS", candidates()),
            capability_plan.CapabilityPlan("sufficient"),
        )

    def test_rejects_unknown_duplicate_unsorted_added_and_oversized_results(self):
        responses = (
            '{"status":"install-required","assistant_ids":["unknown"]}',
            (
                '{"status":"install-required","assistant_ids":'
                '["shimpz-cloudflare","shimpz-cloudflare"]}'
            ),
            (
                '{"status":"install-required","assistant_ids":'
                '["shimpz-whatsapp","shimpz-cloudflare"]}'
            ),
            '{"status":"sufficient","assistant_ids":["shimpz-cloudflare"]}',
            '{"status":"install-required","assistant_ids":[]}',
            '{"status":"install-required","assistant_ids":["shimpz-cloudflare"],"extra":true}',
            (
                '{"status":"install-required","assistant_ids":["shimpz-cloudflare"],'
                '"assistant_ids":["shimpz-whatsapp"]}'
            ),
            "not-json",
        )
        for content in responses:
            with self.subTest(content=content):
                model = RecordingModel(responses=[AIMessage(content=content)])
                runtime = agent_runtime.AgentRuntime(
                    object(),
                    model_factory=lambda _config, selected=model: selected,
                )
                with self.assertRaisesRegex(agent_runtime.ProviderRequestError, "model provider request failed"):
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
        output = '{"status":"install-required","assistant_ids":[' + ",".join(
            f'"assistant-{index}"' for index in range(5)
        ) + "]}"
        model = RecordingModel(responses=[AIMessage(content=output)])
        with self.assertRaises(agent_runtime.ProviderRequestError):
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

    def test_provider_factory_failure_is_redacted(self):
        runtime = agent_runtime.AgentRuntime(
            object(),
            model_factory=mock.Mock(side_effect=RuntimeError("secret provider detail")),
        )

        with self.assertRaisesRegex(agent_runtime.ProviderRequestError, "^model provider request failed$"):
            runtime.capability_plan(provider(), "Configure DNS", candidates())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest import mock

import agent_runtime
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from structured_fake import StructuredFakeModel

RecordingModel = StructuredFakeModel


def provider() -> agent_runtime.ProviderConfig:
    return agent_runtime.ProviderConfig(
        provider="openai",
        model="gpt-6-sol",
        api_key="secret-test-key",
    )


def runtime_for(model: object) -> agent_runtime.AgentRuntime:
    return agent_runtime.AgentRuntime(
        InMemorySaver(),
        model_factory=lambda _config: model,
    )


class ActionLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingModel.seen_messages = []
        RecordingModel.structured = []

    def test_uses_no_tools_or_checkpoint_state(self):
        class RejectCheckpointAccess:
            def __getattr__(self, name):
                raise AssertionError(f"unexpected checkpoint access: {name}")

        model = RecordingModel(
            responses=[
                AIMessage(
                    content=(
                        '{"labels":['
                        '{"id":"list-zones","label":"Listar zonas DNS"},'
                        '{"id":"delete-dns-record","label":"Excluir registro DNS"}'
                        "]}"
                    )
                )
            ]
        )
        runtime = agent_runtime.AgentRuntime(RejectCheckpointAccess(), model_factory=lambda _config: model)

        labels = runtime.action_labels(
            provider(),
            "Quero listar minhas zonas DNS da Cloudflare",
            ("list-zones", "delete-dns-record"),
        )

        self.assertEqual(
            labels,
            (
                agent_runtime.ActionLabel(id="list-zones", label="Listar zonas DNS"),
                agent_runtime.ActionLabel(id="delete-dns-record", label="Excluir registro DNS"),
            ),
        )
        self.assertEqual(len(RecordingModel.seen_messages), 1)
        provider_text = "\n".join(str(message.content) for message in model.seen_messages[0])
        self.assertIn("Quero listar minhas zonas DNS da Cloudflare", provider_text)
        self.assertIn('"action_ids":["list-zones","delete-dns-record"]', provider_text)
        self.assertNotIn("input_schema", provider_text)
        self.assertEqual(
            RecordingModel.structured,
            [(agent_runtime.ActionLabelsOutput, {"method": "json_schema", "include_raw": True, "strict": True})],
        )

    def test_anthropic_labels_use_native_schema_without_the_openai_strict_flag(self):
        model = RecordingModel(responses=[AIMessage(content='{"labels":[{"id":"list-zones","label":"Listar zonas"}]}')])
        anthropic = agent_runtime.ProviderConfig("anthropic", "claude-sonnet-5", "secret-test-key")
        self.assertEqual(
            runtime_for(model).action_labels(anthropic, "Liste zonas", ("list-zones",)),
            (agent_runtime.ActionLabel("list-zones", "Listar zonas"),),
        )
        self.assertEqual(RecordingModel.structured[-1][1], {"method": "json_schema", "include_raw": True})

    def test_rejects_nonclosed_or_unsafe_model_output(self):
        responses = (
            '{"labels":[{"id":"list-zones","label":"Listar zonas"}]}',
            ('{"labels":[{"id":"list-zones","label":"Listar zonas"},{"id":"extra-action","label":"Extra"}]}'),
            ('{"labels":[{"id":"list-zones","label":"Mesmo rótulo"},{"id":"get-zone","label":"Mesmo rótulo"}]}'),
            ('{"labels":[{"id":"list-zones","label":"Listar\\nzona"},{"id":"get-zone","label":"Consultar zona"}]}'),
            '{"labels":[],"unexpected":true}',
            (
                '{"labels":['
                '{"id":"list-zones","label":"Listar zonas"},'
                '{"id":"get-zone","label":"Consultar zona"}'
                '],"labels":['
                '{"id":"list-zones","label":"Listar zonas"},'
                '{"id":"get-zone","label":"Consultar zona"}'
                "]}"
            ),
            '{"labels":[{"id":"list-zones","label":"Listar zonas","id":"get-zone"}]}',
            "not-json",
        )
        for content in responses:
            with self.subTest(content=content):
                model = RecordingModel(responses=[AIMessage(content=content)])
                with self.assertRaisesRegex(agent_runtime.ProviderResponseError, "model provider response failed"):
                    runtime_for(model).action_labels(
                        provider(),
                        "Liste minhas zonas",
                        ("list-zones", "get-zone"),
                    )

    def test_rejects_equivalent_unicode_refusals_and_tool_calls(self):
        equivalent = RecordingModel(
            responses=[
                AIMessage(content=('{"labels":[{"id":"list-zones","label":"é"},{"id":"get-zone","label":"e\\u0301"}]}'))
            ]
        )
        provider_envelope = RecordingModel(
            responses=[
                AIMessage(
                    content=[
                        {"type": "reasoning", "id": "reasoning-1", "summary": []},
                        {
                            "type": "text",
                            "text": (
                                '{"labels":['
                                '{"id":"list-zones","label":"Listar zonas"},'
                                '{"id":"get-zone","label":"Consultar zona"}'
                                "]}"
                            ),
                            "annotations": [],
                            "id": "message-1",
                        },
                        {"type": "future-provider-metadata", "value": "ignored"},
                    ]
                )
            ]
        )
        refusal = RecordingModel(
            responses=[AIMessage(content=[{"type": "refusal", "refusal": "cannot comply", "id": "message-1"}])]
        )
        tool_call = RecordingModel(
            responses=[
                AIMessage(
                    content='{"labels":[]}',
                    tool_calls=[{"name": "unexpected", "args": {}, "id": "call-1", "type": "tool_call"}],
                )
            ]
        )
        invalid_tool_call = RecordingModel(
            responses=[
                AIMessage(
                    content='{"labels":[]}',
                    invalid_tool_calls=[
                        {
                            "name": "unexpected",
                            "args": "{}",
                            "id": "call-2",
                            "error": "invalid",
                            "type": "invalid_tool_call",
                        }
                    ],
                )
            ]
        )

        for model in (equivalent, refusal, tool_call, invalid_tool_call):
            with self.subTest(model=model), self.assertRaises(agent_runtime.ProviderResponseError):
                runtime_for(model).action_labels(
                    provider(),
                    "Liste minhas zonas",
                    ("list-zones", "get-zone"),
                )

        self.assertEqual(
            runtime_for(provider_envelope).action_labels(
                provider(),
                "Liste zonas",
                ("list-zones", "get-zone"),
            ),
            (
                agent_runtime.ActionLabel("list-zones", "Listar zonas"),
                agent_runtime.ActionLabel("get-zone", "Consultar zona"),
            ),
        )

    def test_closed_label_helpers_reject_remaining_boundary_shapes(self):
        with self.assertRaises(agent_runtime.RuntimeContractError):
            agent_runtime.normalize_language_exemplar(object())
        self.assertEqual(
            agent_runtime.normalize_language_exemplar("desinstala 👩‍💻\r\nagora"),
            "desinstala 👩‍💻\r\nagora",
        )
        for value in (object(), " label", "x" * (agent_runtime.MAX_ACTION_LABEL_CHARS + 1)):
            with self.subTest(value_type=type(value)), self.assertRaises(agent_runtime.RuntimeContractError):
                agent_runtime._validated_action_label(value)
        with self.assertRaises(agent_runtime.RuntimeContractError):
            agent_runtime._action_label_items([object()], frozenset({"list-zones"}))
        for result in (
            object(),
            {"raw": AIMessage(content=""), "parsed": None},
            {"raw": AIMessage(content=""), "parsed": None, "parsing_error": ValueError("not JSON")},
            {"raw": "not a message", "parsed": {"labels": []}, "parsing_error": None},
            {"raw": AIMessage(content=""), "parsed": object(), "parsing_error": None},
            {"raw": AIMessage(content=""), "parsed": {"labels": [], "extra": True}, "parsing_error": None},
        ):
            with self.subTest(result=result), self.assertRaises(agent_runtime.RuntimeContractError):
                agent_runtime._parse_action_labels(result, ("list-zones",))
        consistent = '{"labels":[{"id":"list-zones","label":"Listar zonas"}]}'
        disagreeing = {"labels": [{"id": "list-zones", "label": "Outra coisa"}]}
        for result in (
            {"raw": AIMessage(content=consistent), "parsed": disagreeing, "parsing_error": None},
            {
                "raw": AIMessage(
                    content=consistent.replace(":[", ":[" + " " * agent_runtime.MAX_ACTION_LABEL_RESPONSE_CHARS)
                ),
                "parsed": {"labels": [{"id": "list-zones", "label": "Listar zonas"}]},
                "parsing_error": None,
            },
        ):
            with self.subTest(raw=len(result["raw"].content)), self.assertRaises(agent_runtime.RuntimeContractError):
                agent_runtime._parse_action_labels(result, ("list-zones",))
        self.assertEqual(
            agent_runtime._parse_action_labels(
                {"raw": AIMessage(content=consistent), "parsed": None, "parsing_error": None}
                | {"parsed": {"labels": [{"id": "list-zones", "label": "Listar zonas"}]}},
                ("list-zones",),
            ),
            (agent_runtime.ActionLabel("list-zones", "Listar zonas"),),
        )
        too_many = [{"id": f"a-{index}", "label": f"L {index}"} for index in range(agent_runtime.MAX_ACTION_LABELS + 1)]
        with self.assertRaises(ValueError):
            agent_runtime.ActionLabelsOutput.model_validate({"labels": too_many})

    def test_provider_and_non_message_failures_are_redacted(self):
        for failure, expected in (
            (ImportError("missing dependency"), ImportError),
            (RuntimeError("secret provider detail"), agent_runtime.ProviderRequestError),
        ):
            model = mock.Mock()
            model.with_structured_output.return_value.invoke.side_effect = failure
            with self.subTest(failure=type(failure).__name__), self.assertRaises(expected):
                runtime_for(model).action_labels(provider(), "Liste zonas", ("list-zones",))

        model = mock.Mock()
        model.with_structured_output.return_value.invoke.return_value = object()
        with self.assertRaisesRegex(agent_runtime.ProviderResponseError, "^model provider response failed$"):
            runtime_for(model).action_labels(provider(), "Liste zonas", ("list-zones",))

    def test_inputs_fail_before_provider_or_checkpoint_access(self):
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=mock.Mock())

        for exemplar, action_ids in (
            ("", ("list-zones",)),
            ("hidden\0instruction", ("list-zones",)),
            ("Liste zonas", ()),
            ("Liste zonas", ("list-zones", "list-zones")),
            ("Liste zonas", ("../shell",)),
        ):
            with (
                self.subTest(exemplar=exemplar, action_ids=action_ids),
                self.assertRaises(agent_runtime.RuntimeContractError),
            ):
                runtime.action_labels(provider(), exemplar, action_ids)
        runtime._model_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

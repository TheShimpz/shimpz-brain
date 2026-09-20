from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import agent_runtime
import intent_route


def provider(name: str = "openai") -> agent_runtime.ProviderConfig:
    model = "gpt-5.6-terra" if name == "openai" else "claude-sonnet-5"
    return agent_runtime.ProviderConfig(name, model, "secret-test-key")


def candidates(*, uninstall: bool = False) -> tuple[intent_route.DirectoryCandidate, ...]:
    return (
        intent_route.DirectoryCandidate(
            "shimpz-cloudflare",
            "Shimpz Cloudflare",
            "" if uninstall else "Manage DNS zones and records.",
        ),
        intent_route.DirectoryCandidate(
            "shimpz-whatsapp",
            "Shimpz WhatsApp",
            "" if uninstall else "Send reviewed WhatsApp messages.",
        ),
    )


class StructuredModel:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.schema = None
        self.options = None
        self.messages = None

    def with_structured_output(self, schema, **options):
        self.schema = schema
        self.options = options
        return self

    def invoke(self, messages):
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.response


class DecisionFactory:
    def __init__(self, model: StructuredModel) -> None:
        self.model = model
        self.decision_calls = []

    def __call__(self, _config):
        raise AssertionError("intent routing must use the decision model")

    def decision(self, config):
        self.decision_calls.append(config)
        return self.model


class IntentRouteTests(unittest.TestCase):
    def test_classification_uses_static_native_schema_and_untrusted_framing(self):
        model = StructuredModel(
            intent_route.StructuredRoute(
                intent="assistant-uninstall",
                query="cloudflare",
                assistant_ids=[],
            )
        )

        result = intent_route.create(model, "openai", "por favor tire o cloudflare", None, ())

        self.assertEqual(result, intent_route.IntentRoute("assistant-uninstall", "cloudflare"))
        self.assertIs(model.schema, intent_route.StructuredRoute)
        self.assertEqual(model.options, {"method": "json_schema", "strict": True})
        prompt = "\n".join(str(message.content) for message in model.messages)
        self.assertIn("untrusted data", prompt)
        self.assertIn("por favor tire o cloudflare", prompt)
        self.assertNotIn("api_key", prompt)

    def test_anthropic_uses_the_same_static_schema_without_openai_options(self):
        model = StructuredModel(intent_route.StructuredRoute(intent="ordinary-task", query="", assistant_ids=[]))

        result = intent_route.create(model, "anthropic", "liste minhas zonas", None, ())

        self.assertEqual(result, intent_route.IntentRoute("ordinary-task"))
        self.assertEqual(model.options, {"method": "json_schema"})

    def test_selection_is_bound_to_expected_intent_and_exact_candidate_ids(self):
        model = StructuredModel(
            intent_route.StructuredRoute(
                intent="assistant-install",
                query="",
                assistant_ids=["shimpz-cloudflare", "shimpz-whatsapp"],
            )
        )

        result = intent_route.create(
            model,
            "openai",
            "instale cloudflare e whatsapp",
            "assistant-install",
            candidates(),
        )

        self.assertEqual(
            result,
            intent_route.IntentRoute(
                "assistant-install",
                assistant_ids=("shimpz-cloudflare", "shimpz-whatsapp"),
            ),
        )
        prompt = "\n".join(str(message.content) for message in model.messages)
        self.assertIn('"expected_intent":"assistant-install"', prompt)
        self.assertIn('"id":"shimpz-cloudflare"', prompt)

    def test_selection_rejects_wrong_unknown_duplicate_unsorted_and_empty_ids(self):
        invalid = (
            {"intent": "assistant-uninstall", "query": "", "assistant_ids": ["unknown"]},
            {
                "intent": "assistant-uninstall",
                "query": "",
                "assistant_ids": ["shimpz-cloudflare", "shimpz-cloudflare"],
            },
            {
                "intent": "assistant-install",
                "query": "",
                "assistant_ids": ["shimpz-whatsapp", "shimpz-cloudflare"],
            },
            {"intent": "assistant-install", "query": "", "assistant_ids": []},
            {"intent": "ordinary-task", "query": "", "assistant_ids": ["shimpz-cloudflare"]},
        )
        for response in invalid:
            expected = "assistant-uninstall" if response["intent"] == "assistant-uninstall" else "assistant-install"
            shortlist = candidates(uninstall=expected == "assistant-uninstall")
            with self.subTest(response=response), self.assertRaises(intent_route.IntentRouteResponseError):
                intent_route.create(
                    StructuredModel(response),
                    "openai",
                    "lifecycle objective",
                    expected,
                    shortlist,
                )

    def test_unresolved_selection_is_non_authorizing(self):
        result = intent_route.create(
            StructuredModel({"intent": "unresolved", "query": "", "assistant_ids": []}),
            "openai",
            "remove it",
            "assistant-uninstall",
            candidates(uninstall=True),
        )

        self.assertEqual(result, intent_route.IntentRoute("unresolved"))

    def test_invalid_inputs_fail_before_provider_access(self):
        model = mock.Mock()
        invalid = (
            ("", None, ()),
            ("hidden\0message", None, ()),
            ("install", None, candidates()),
            ("install", "assistant-install", ()),
            ("install", "assistant-install", tuple(reversed(candidates()))),
            ("uninstall", "assistant-uninstall", candidates()),
        )
        for objective, expected, shortlist in invalid:
            with self.subTest(objective=objective, expected=expected), self.assertRaises(intent_route.IntentRouteError):
                intent_route.create(model, "openai", objective, expected, shortlist)
        model.with_structured_output.assert_not_called()

    def test_input_contract_rejects_wrong_types_identifiers_and_intents(self):
        invalid = (
            (1, None, ()),
            ("install", "unsupported", ()),
            ("install", "assistant-install", (object(),)),
            (
                "install",
                "assistant-install",
                (intent_route.DirectoryCandidate("Invalid Id", "Invalid"),),
            ),
        )
        for objective, expected, shortlist in invalid:
            with self.subTest(expected=expected, shortlist=shortlist), self.assertRaises(intent_route.IntentRouteError):
                intent_route.validate_inputs(objective, expected, shortlist)

    def test_response_contract_rejects_classification_and_selection_conflicts(self):
        invalid = (
            (
                intent_route.StructuredRoute(
                    intent="ordinary-task",
                    query="",
                    assistant_ids=["shimpz-cloudflare"],
                ),
                None,
                (),
            ),
            (
                intent_route.StructuredRoute.model_construct(
                    intent="invalid",
                    query="unexpected",
                    assistant_ids=[],
                ),
                None,
                (),
            ),
            (
                intent_route.StructuredRoute(
                    intent="assistant-uninstall",
                    query="unexpected",
                    assistant_ids=["shimpz-cloudflare"],
                ),
                "assistant-uninstall",
                candidates(uninstall=True),
            ),
            (
                intent_route.StructuredRoute(
                    intent="unresolved",
                    query="",
                    assistant_ids=["shimpz-cloudflare"],
                ),
                "assistant-uninstall",
                candidates(uninstall=True),
            ),
        )
        for response, expected, shortlist in invalid:
            with self.subTest(response=response), self.assertRaises(intent_route.IntentRouteError):
                intent_route._route(response, expected, shortlist)

    def test_provider_and_contract_failures_are_redacted_and_distinct(self):
        with self.assertRaisesRegex(intent_route.IntentRouteProviderError, "^model provider request failed$"):
            intent_route.create(
                StructuredModel(error=RuntimeError("secret provider detail")),
                "openai",
                "hello",
                None,
                (),
            )
        for value in (None, {}, {"intent": "ordinary-task", "query": "", "assistant_ids": [], "extra": True}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    intent_route.IntentRouteResponseError,
                    "^model provider response failed$",
                ),
            ):
                intent_route.create(StructuredModel(value), "openai", "hello", None, ())

    def test_provider_adapter_and_dependency_failures_remain_distinct(self):
        with self.assertRaises(intent_route.IntentRouteError):
            intent_route.create(StructuredModel(), "unsupported", "hello", None, ())
        with self.assertRaisesRegex(ImportError, "missing structured adapter"):
            intent_route.create(
                StructuredModel(error=ImportError("missing structured adapter")),
                "openai",
                "hello",
                None,
                (),
            )

    def test_runtime_uses_only_the_decision_factory_without_checkpoint_access(self):
        model = StructuredModel({"intent": "ordinary-task", "query": "", "assistant_ids": []})
        factory = DecisionFactory(model)
        runtime = agent_runtime.AgentRuntime(SimpleNamespace(), model_factory=factory)

        result = runtime.intent_route(provider(), "hello", None, ())

        self.assertEqual(result, intent_route.IntentRoute("ordinary-task"))
        self.assertEqual(factory.decision_calls, [provider()])

    def test_runtime_projects_every_intent_route_failure_without_provider_detail(self):
        cases = (
            (intent_route.IntentRouteError("invalid route"), agent_runtime.RuntimeContractError),
            (
                intent_route.IntentRouteResponseError("private response"),
                agent_runtime.ProviderResponseError,
            ),
            (
                intent_route.IntentRouteProviderError("private request"),
                agent_runtime.ProviderRequestError,
            ),
            (ImportError("missing adapter"), ImportError),
            (RuntimeError("private failure"), agent_runtime.ProviderRequestError),
        )
        runtime = agent_runtime.AgentRuntime(SimpleNamespace(), model_factory=lambda _config: mock.Mock())
        for failure, projected in cases:
            with (
                self.subTest(failure=failure),
                mock.patch.object(intent_route, "create", side_effect=failure),
                self.assertRaises(projected),
            ):
                runtime.intent_route(provider(), "hello", None, ())


if __name__ == "__main__":
    unittest.main()

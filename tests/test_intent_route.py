from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import agent_runtime
import intent_route


def provider(name: str = "openai") -> agent_runtime.ProviderConfig:
    model = "gpt-6-sol" if name == "openai" else "claude-sonnet-5"
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
                task_follows=False,
                intent="assistant-uninstall",
                query="cloudflare",
                assistant_ids=[],
                reply="",
            )
        )
        factory = mock.Mock(return_value=model)

        result = intent_route.create(factory, "openai", "por favor tire o cloudflare", None, (), None)

        self.assertEqual(result, intent_route.IntentRoute("assistant-uninstall", "cloudflare"))
        factory.assert_called_once_with()
        self.assertIs(model.schema, intent_route.StructuredRoute)
        self.assertEqual(model.options, {"method": "json_schema", "strict": True})
        prompt = "\n".join(str(message.content) for message in model.messages)
        self.assertIn("untrusted data", prompt)
        self.assertIn("exactly one line", prompt)
        self.assertIn("por favor tire o cloudflare", prompt)
        self.assertNotIn("api_key", prompt)

    def test_classification_admits_one_untrusted_lifecycle_reference_only(self):
        reference = intent_route.LifecycleReference("shimpz-cloudflare", "Shimpz Cloudflare")
        model = StructuredModel(
            intent_route.StructuredRoute(
                task_follows=False,
                intent="assistant-install",
                query="shimpz-cloudflare",
                assistant_ids=[],
                reply="",
            )
        )

        context = intent_route.LifecycleContext(reference=reference)
        result = intent_route.create(lambda: model, "openai", "instale ele de novo", None, (), context)

        self.assertEqual(result, intent_route.IntentRoute("assistant-install", "shimpz-cloudflare"))
        self.assertNotIn("shimpz-cloudflare", str(model.messages[0].content))
        self.assertIn('"lifecycle_reference":{"id":"shimpz-cloudflare"', str(model.messages[1].content))
        with self.assertRaises(intent_route.IntentRouteError):
            intent_route.validate_inputs("instale", "assistant-install", candidates(), context)
        with self.assertRaises(intent_route.IntentRouteError):
            intent_route.validate_inputs("instale", None, (), object())

        echo = StructuredModel(
            {
                "task_follows": False,
                "intent": "assistant-install",
                "query": "",
                "assistant_ids": ["shimpz-cloudflare"],
                "reply": "",
            }
        )
        with self.assertRaises(intent_route.IntentRouteResponseError):
            intent_route.create(lambda: echo, "openai", "instale ele", None, (), context)

    def test_classification_uses_bounded_conversation_to_resolve_a_reference(self):
        context = intent_route.LifecycleContext(
            conversation=(
                intent_route.ConversationEntry("user", "Quais Assistants estão instalados?", False),
                intent_route.ConversationEntry("assistant", "Temos apenas Cloudflare/DNS.", False),
                intent_route.ConversationEntry("user", "Quero desinstalar ele.", False),
                intent_route.ConversationEntry("assistant", "Qual Assistant você quer desinstalar?", False),
                intent_route.ConversationEntry("user", "Quais temos?", False),
                intent_route.ConversationEntry("assistant", "Temos apenas Cloudflare/DNS.", False),
            ),
        )
        model = StructuredModel(
            intent_route.StructuredRoute(
                task_follows=False,
                intent="assistant-uninstall",
                query="cloudflare",
                assistant_ids=[],
                reply="",
            )
        )

        result = intent_route.create(lambda: model, "openai", "Desinstala esse então.", None, (), context)

        self.assertEqual(result, intent_route.IntentRoute("assistant-uninstall", "cloudflare"))
        payload = str(model.messages[1].content)
        self.assertIn('"role":"assistant","text":"Temos apenas Cloudflare/DNS."', payload)
        self.assertIn('"conversation":[', payload)
        system = str(model.messages[0].content)
        self.assertIn("current objective is the only fresh instruction", system)
        self.assertNotIn("Cloudflare/DNS", system)

    def test_anthropic_uses_the_same_static_schema_without_openai_options(self):
        model = StructuredModel(
            intent_route.StructuredRoute(
                task_follows=False, intent="ordinary-task", query="", assistant_ids=[], reply=""
            )
        )

        result = intent_route.create(lambda: model, "anthropic", "liste minhas zonas", None, (), None)

        self.assertEqual(result, intent_route.IntentRoute("ordinary-task"))
        self.assertEqual(model.options, {"method": "json_schema"})

    def test_selection_is_bound_to_expected_intent_and_exact_candidate_ids(self):
        model = StructuredModel(
            intent_route.StructuredRoute(
                task_follows=False,
                intent="assistant-install",
                query="",
                assistant_ids=["shimpz-cloudflare", "shimpz-whatsapp"],
                reply="",
            )
        )

        result = intent_route.create(
            lambda: model,
            "openai",
            "instale cloudflare e whatsapp",
            "assistant-install",
            candidates(),
            None,
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
            {
                "task_follows": False,
                "intent": "assistant-uninstall",
                "query": "",
                "assistant_ids": ["unknown"],
                "reply": "",
            },
            {
                "task_follows": False,
                "intent": "assistant-uninstall",
                "query": "",
                "assistant_ids": ["shimpz-cloudflare", "shimpz-cloudflare"],
                "reply": "",
            },
            {
                "task_follows": False,
                "intent": "assistant-install",
                "query": "",
                "assistant_ids": ["shimpz-whatsapp", "shimpz-cloudflare"],
                "reply": "",
            },
            {"task_follows": False, "intent": "assistant-install", "query": "", "assistant_ids": [], "reply": ""},
            {
                "task_follows": False,
                "intent": "ordinary-task",
                "query": "",
                "assistant_ids": ["shimpz-cloudflare"],
                "reply": "",
            },
        )
        for response in invalid:
            expected = "assistant-uninstall" if response["intent"] == "assistant-uninstall" else "assistant-install"
            shortlist = candidates(uninstall=expected == "assistant-uninstall")
            with self.subTest(response=response), self.assertRaises(intent_route.IntentRouteResponseError):
                intent_route.create(
                    mock.Mock(return_value=StructuredModel(response)),
                    "openai",
                    "lifecycle objective",
                    expected,
                    shortlist,
                    None,
                )

    def test_unresolved_selection_is_non_authorizing(self):
        result = intent_route.create(
            lambda: StructuredModel(
                {
                    "task_follows": False,
                    "intent": "unresolved",
                    "query": "",
                    "assistant_ids": [],
                    "reply": "Qual Assistant instalado você quer desinstalar?",
                }
            ),
            "openai",
            "remove it",
            "assistant-uninstall",
            candidates(uninstall=True),
            None,
        )

        self.assertEqual(
            result,
            intent_route.IntentRoute(
                "unresolved",
                reply="Qual Assistant instalado você quer desinstalar?",
            ),
        )

    def test_targetless_classification_requires_a_bounded_natural_question(self):
        reply = "Qual Assistant você quer desinstalar?"
        result = intent_route.create(
            lambda: StructuredModel(
                {
                    "task_follows": False,
                    "intent": "assistant-uninstall",
                    "query": "",
                    "assistant_ids": [],
                    "reply": reply,
                }
            ),
            "openai",
            "desinstala ele",
            None,
            (),
            None,
        )

        self.assertEqual(result, intent_route.IntentRoute("assistant-uninstall", reply=reply))

    def test_clarification_admits_unicode_spacing_and_rejects_layout_separators(self):
        reply = "Quel Assistant voulez-vous désinstaller\u00a0?"
        result = intent_route.create(
            lambda: StructuredModel(
                {
                    "task_follows": False,
                    "intent": "assistant-uninstall",
                    "query": "",
                    "assistant_ids": [],
                    "reply": reply,
                }
            ),
            "openai",
            "désinstalle",
            None,
            (),
            None,
        )
        self.assertEqual(result.reply, reply)

        for separator in ("\u2028", "\u2029"):
            with self.subTest(separator=separator), self.assertRaises(intent_route.IntentRouteResponseError):
                intent_route.create(
                    mock.Mock(
                        return_value=StructuredModel(
                            {
                                "task_follows": False,
                                "intent": "assistant-uninstall",
                                "query": "",
                                "assistant_ids": [],
                                "reply": f"Question{separator}suivante",
                            }
                        )
                    ),
                    "openai",
                    "désinstalle",
                    None,
                    (),
                    None,
                )

    def test_conversation_admits_user_unicode_format_and_layout(self):
        context = intent_route.LifecycleContext(
            conversation=(intent_route.ConversationEntry("user", "desinstala 👩‍💻\r\nagora", False),),
        )
        model = StructuredModel(
            {
                "task_follows": False,
                "intent": "assistant-uninstall",
                "query": "cloudflare",
                "assistant_ids": [],
                "reply": "",
            }
        )

        result = intent_route.create(lambda: model, "openai", "cloudflare", None, (), context)

        self.assertEqual(result.query, "cloudflare")

    def test_common_route_text_avoids_python_unicode_category_scans(self):
        examples = (
            ("a" * intent_route.MAX_OBJECTIVE_CHARS, "b" * intent_route.MAX_CONVERSATION_TEXT_CHARS),
            (("a\nb" * 5334)[: intent_route.MAX_OBJECTIVE_CHARS], "b\n" * 255 + "cc"),
            ("é" * intent_route.MAX_OBJECTIVE_CHARS, "é" * intent_route.MAX_CONVERSATION_TEXT_CHARS),
        )
        with mock.patch.object(intent_route.unicodedata, "category", side_effect=AssertionError("slow scan")):
            for objective, text in examples:
                with self.subTest(objective_start=objective[:3]):
                    context = intent_route.LifecycleContext(
                        conversation=(intent_route.ConversationEntry("user", text, False),)
                    )
                    admitted = intent_route.validate_inputs(objective, None, (), context)
                    self.assertEqual(admitted[0], objective)
            selected = intent_route.validate_inputs("install", "assistant-install", candidates(), None)
            self.assertEqual(selected[2], candidates())

    def test_empty_selection_directory_can_only_ask_for_clarification(self):
        reply = "Qual Assistant instalado você quer desinstalar?"
        result = intent_route.create(
            lambda: StructuredModel(
                {"task_follows": False, "intent": "unresolved", "query": "", "assistant_ids": [], "reply": reply}
            ),
            "openai",
            "desinstale desconhecido",
            "assistant-uninstall",
            (),
            intent_route.LifecycleContext(language_exemplar="desinstale desconhecido"),
        )

        self.assertEqual(result, intent_route.IntentRoute("unresolved", reply=reply))

    def test_invalid_inputs_fail_before_provider_access(self):
        model = mock.Mock()
        factory = mock.Mock(return_value=model)
        invalid = (
            ("", None, ()),
            ("hidden\0message", None, ()),
            ("install", None, candidates()),
            ("install", "assistant-install", tuple(reversed(candidates()))),
            ("uninstall", "assistant-uninstall", candidates()),
        )
        for objective, expected, shortlist in invalid:
            with self.subTest(objective=objective, expected=expected), self.assertRaises(intent_route.IntentRouteError):
                intent_route.create(factory, "openai", objective, expected, shortlist, None)
        factory.assert_not_called()
        model.with_structured_output.assert_not_called()

    def test_input_contract_rejects_wrong_types_identifiers_and_intents(self):
        invalid = (
            (1, None, ()),
            ("install", "unsupported", ()),
            ("install", "assistant-install", []),
            ("install", "assistant-install", (object(),)),
            (
                "install",
                "assistant-install",
                (intent_route.DirectoryCandidate("Invalid Id", "Invalid"),),
            ),
        )
        for objective, expected, shortlist in invalid:
            with self.subTest(expected=expected, shortlist=shortlist), self.assertRaises(intent_route.IntentRouteError):
                intent_route.validate_inputs(objective, expected, shortlist, None)

    def test_lifecycle_context_rejects_invalid_or_wrong_lane_state(self):
        invalid = (
            intent_route.LifecycleContext(reference=object()),
            intent_route.LifecycleContext(conversation=[]),
            intent_route.LifecycleContext(conversation=(object(),)),
            intent_route.LifecycleContext(
                conversation=(intent_route.ConversationEntry("system", "remove it", False),),
            ),
            intent_route.LifecycleContext(
                conversation=(intent_route.ConversationEntry("user", "remove it", 1),),
            ),
            intent_route.LifecycleContext(
                conversation=(intent_route.ConversationEntry("user", "x" * 513, False),),
            ),
            intent_route.LifecycleContext(
                conversation=tuple(intent_route.ConversationEntry("user", str(index), False) for index in range(9)),
            ),
            intent_route.LifecycleContext(language_exemplar="remove it"),
        )

        for context in invalid:
            with self.subTest(context=context), self.assertRaises(intent_route.IntentRouteError):
                intent_route.validate_inputs("remove it", None, (), context)
        with (
            mock.patch.object(intent_route, "MAX_CONVERSATION_CHARS", 1),
            self.assertRaises(intent_route.IntentRouteError),
        ):
            intent_route.validate_inputs(
                "remove it",
                None,
                (),
                intent_route.LifecycleContext(
                    conversation=(intent_route.ConversationEntry("user", "remove it", False),),
                ),
            )

        selection_contexts = (
            intent_route.LifecycleContext(
                conversation=(intent_route.ConversationEntry("user", "remove it", False),),
            ),
            intent_route.LifecycleContext(
                reference=intent_route.LifecycleReference("shimpz-cloudflare", "Shimpz Cloudflare"),
            ),
        )
        for context in selection_contexts:
            with self.subTest(selection_context=context), self.assertRaises(intent_route.IntentRouteError):
                intent_route.validate_inputs("cloudflare", "assistant-uninstall", candidates(uninstall=True), context)

    def test_response_contract_rejects_classification_and_selection_conflicts(self):
        invalid = (
            (
                intent_route.StructuredRoute(
                    task_follows=False,
                    intent="ordinary-task",
                    query="",
                    assistant_ids=["shimpz-cloudflare"],
                    reply="",
                ),
                None,
                (),
            ),
            (
                intent_route.StructuredRoute.model_construct(
                    intent="invalid",
                    query="unexpected",
                    assistant_ids=[],
                    reply="",
                ),
                None,
                (),
            ),
            (
                intent_route.StructuredRoute(
                    task_follows=False,
                    intent="assistant-uninstall",
                    query="unexpected",
                    assistant_ids=["shimpz-cloudflare"],
                    reply="",
                ),
                "assistant-uninstall",
                candidates(uninstall=True),
            ),
            (
                intent_route.StructuredRoute(
                    task_follows=False,
                    intent="unresolved",
                    query="",
                    assistant_ids=["shimpz-cloudflare"],
                    reply="clarify",
                ),
                "assistant-uninstall",
                candidates(uninstall=True),
            ),
            (
                intent_route.StructuredRoute(
                    task_follows=False,
                    intent="ordinary-task",
                    query="",
                    assistant_ids=[],
                    reply="Which Assistant?",
                ),
                None,
                (),
            ),
        )
        for response, expected, shortlist in invalid:
            with self.subTest(response=response), self.assertRaises(intent_route.IntentRouteError):
                intent_route._route(response, expected, shortlist)

    def test_only_a_targeted_install_classification_can_continue_the_task(self):
        def route(intent, query, reply="", ids=(), follows=True, expected=None, shortlist=()):
            response = intent_route.StructuredRoute(
                task_follows=follows, intent=intent, query=query, assistant_ids=list(ids), reply=reply
            )
            return intent_route._route(response, expected, shortlist)

        self.assertEqual(
            route("assistant-install", "exa"),
            intent_route.IntentRoute("assistant-install", "exa", task_follows=True),
        )
        self.assertFalse(route("assistant-install", "exa", follows=False).task_follows)
        for args in (
            ("ordinary-task", ""),
            ("assistant-uninstall", "exa"),
            ("unresolved", "", "Which Assistant?"),
            ("assistant-install", "", "Which Assistant?"),
        ):
            with self.subTest(args=args), self.assertRaises(intent_route.IntentRouteError):
                route(*args)
        with self.assertRaises(intent_route.IntentRouteError):
            route(
                "assistant-install",
                "",
                ids=("shimpz-cloudflare",),
                expected="assistant-install",
                shortlist=candidates(),
            )
        self.assertIn("task_follows", intent_route.StructuredRoute.model_json_schema()["required"])

    def test_provider_and_contract_failures_are_redacted_and_distinct(self):
        with self.assertRaisesRegex(intent_route.IntentRouteProviderError, "^model provider request failed$"):
            intent_route.create(
                lambda: StructuredModel(error=RuntimeError("secret provider detail")),
                "openai",
                "hello",
                None,
                (),
                None,
            )
        for value in (
            None,
            {},
            {
                "task_follows": False,
                "intent": "ordinary-task",
                "query": "",
                "assistant_ids": [],
                "reply": "",
                "extra": True,
            },
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    intent_route.IntentRouteResponseError,
                    "^model provider response failed$",
                ),
            ):
                intent_route.create(mock.Mock(return_value=StructuredModel(value)), "openai", "hello", None, (), None)

    def test_provider_adapter_and_dependency_failures_remain_distinct(self):
        factory = mock.Mock(side_effect=RuntimeError("secret provider detail"))
        with self.assertRaises(intent_route.IntentRouteError):
            intent_route.create(factory, "unsupported", "hello", None, (), None)
        factory.assert_not_called()
        with self.assertRaisesRegex(intent_route.IntentRouteError, "invalid route objective"):
            intent_route.create(factory, "unsupported", "", None, (), None)
        factory.assert_not_called()
        with self.assertRaisesRegex(intent_route.IntentRouteProviderError, "^model provider request failed$"):
            intent_route.create(factory, "openai", "hello", None, (), None)
        factory.assert_called_once_with()

        missing = mock.Mock(side_effect=ImportError("missing factory dependency"))
        with self.assertRaisesRegex(ImportError, "missing factory dependency"):
            intent_route.create(missing, "openai", "hello", None, (), None)
        with self.assertRaisesRegex(ImportError, "missing structured adapter"):
            intent_route.create(
                lambda: StructuredModel(error=ImportError("missing structured adapter")),
                "openai",
                "hello",
                None,
                (),
                None,
            )

    def test_runtime_uses_only_the_decision_factory_without_checkpoint_access(self):
        model = StructuredModel(
            {"task_follows": False, "intent": "ordinary-task", "query": "", "assistant_ids": [], "reply": ""}
        )
        factory = DecisionFactory(model)
        runtime = agent_runtime.AgentRuntime(SimpleNamespace(), model_factory=factory)

        with mock.patch.object(intent_route, "validate_inputs", wraps=intent_route.validate_inputs) as validate:
            result = runtime.intent_route(provider(), "hello", None, (), None)

        self.assertEqual(result, intent_route.IntentRoute("ordinary-task"))
        validate.assert_called_once_with("hello", None, (), None)
        self.assertEqual(factory.decision_calls, [provider()])

        fallback = mock.Mock(spec=lambda _config: None, return_value=StructuredModel(model.response))
        self.assertEqual(
            agent_runtime.AgentRuntime(SimpleNamespace(), model_factory=fallback).intent_route(
                provider(), "hello", None, (), None
            ),
            intent_route.IntentRoute("ordinary-task"),
        )
        fallback.assert_called_once_with(provider())

    def test_runtime_model_factory_failures_keep_their_error_mapping(self):
        failed = agent_runtime.AgentRuntime(
            SimpleNamespace(),
            model_factory=mock.Mock(spec=lambda _config: None, side_effect=RuntimeError("secret provider detail")),
        )
        with self.assertRaisesRegex(agent_runtime.ProviderRequestError, "^model provider request failed$"):
            failed.intent_route(provider(), "hello", None, (), None)

        missing = agent_runtime.AgentRuntime(
            SimpleNamespace(),
            model_factory=mock.Mock(spec=lambda _config: None, side_effect=ImportError("missing dependency")),
        )
        with self.assertRaisesRegex(ImportError, "missing dependency"):
            missing.intent_route(provider(), "hello", None, (), None)

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
                runtime.intent_route(provider(), "hello", None, (), None)


if __name__ == "__main__":
    unittest.main()

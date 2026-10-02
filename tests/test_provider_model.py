"""Provider client construction for ordinary Brain turns and the structured route decision."""

from __future__ import annotations

import unittest
from unittest import mock

import agent_runtime


class ProviderModelTests(unittest.TestCase):
    def test_provider_models_are_closed_to_the_supported_pair(self):
        for provider, models in agent_runtime.MODELS_BY_PROVIDER.items():
            for model in models:
                with self.subTest(provider=provider, model=model):
                    config = agent_runtime.ProviderConfig(
                        provider=provider,
                        model=model,
                        api_key="secret-test-key",
                    )
                    self.assertEqual((config.provider, config.model), (provider, model))

        for provider, model in (
            ("openai", "gpt-well-formed-but-unknown"),
            ("openai", "claude-sonnet-5-5"),
            ("anthropic", "gpt-6.1-sol"),
        ):
            with (
                self.subTest(provider=provider, model=model),
                self.assertRaisesRegex(agent_runtime.RuntimeContractError, "unsupported model for provider"),
            ):
                agent_runtime.ProviderConfig(provider=provider, model=model, api_key="secret-test-key")

    def test_every_catalog_provider_has_a_runtime_adapter(self):
        with (
            mock.patch("langchain_openai.ChatOpenAI") as openai,
            mock.patch.object(agent_runtime, "_pooled_chat_anthropic", return_value=(anthropic := mock.Mock())),
        ):
            for provider, models in agent_runtime.MODELS_BY_PROVIDER.items():
                with self.subTest(provider=provider):
                    agent_runtime.provider_model(
                        agent_runtime.ProviderConfig(
                            provider=provider,
                            model=next(iter(models)),
                            api_key="secret-test-key",
                        )
                    )

        openai.assert_called_once()
        anthropic.assert_called_once()

    def test_openai_models_reuse_transport_and_decisions_use_low_effort(self):
        transport = mock.Mock()
        with (
            mock.patch.object(agent_runtime.httpx, "Client", return_value=transport),
            mock.patch(
                "langchain_openai.ChatOpenAI",
                side_effect=[mock.Mock(), mock.Mock(), mock.Mock()],
            ) as constructor,
        ):
            factory = agent_runtime.ProviderModelFactory()
            factory(
                agent_runtime.ProviderConfig(
                    provider="openai",
                    model="gpt-6.1-sol",
                    api_key="first-secret-key",
                )
            )
            factory(
                agent_runtime.ProviderConfig(
                    provider="openai",
                    model="gpt-6.1-sol",
                    api_key="second-secret-key",
                )
            )
            factory.decision(
                agent_runtime.ProviderConfig(
                    provider="openai",
                    model="gpt-6.1-sol",
                    api_key="decision-secret-key",
                )
            )
            factory.close()

        first, second, decision = constructor.call_args_list
        self.assertIs(first.kwargs["http_client"], transport)
        self.assertIs(second.kwargs["http_client"], transport)
        self.assertNotEqual(first.kwargs["api_key"], second.kwargs["api_key"])
        self.assertIs(decision.kwargs["http_client"], transport)
        self.assertEqual(decision.kwargs["reasoning_effort"], "low")
        self.assertEqual(decision.kwargs["max_retries"], agent_runtime.DECISION_MAX_RETRIES)
        self.assertEqual(first.kwargs["max_retries"], 2)
        transport.close.assert_called_once_with()

    def test_openai_uses_responses_api_without_changing_anthropic(self):
        with (
            mock.patch("langchain_openai.ChatOpenAI") as openai,
            mock.patch.object(agent_runtime, "_pooled_chat_anthropic", return_value=(anthropic := mock.Mock())),
        ):
            agent_runtime.provider_model(
                agent_runtime.ProviderConfig(
                    provider="openai",
                    model="gpt-6.1-sol",
                    api_key="secret-test-key",
                )
            )
            agent_runtime.provider_model(
                agent_runtime.ProviderConfig(
                    provider="anthropic",
                    model="claude-sonnet-5-5",
                    api_key="secret-test-key",
                )
            )

        self.assertTrue(openai.call_args.kwargs["use_responses_api"])
        self.assertNotIn("use_responses_api", anthropic.call_args.kwargs)
        self.assertEqual(set(openai.call_args.kwargs) - {"use_responses_api"}, set(anthropic.call_args.kwargs))

    def test_decision_models_use_provider_specific_low_effort_with_one_retry(self):
        with (
            mock.patch("langchain_openai.ChatOpenAI") as openai,
            mock.patch.object(agent_runtime, "_pooled_chat_anthropic", return_value=(anthropic := mock.Mock())),
        ):
            agent_runtime.provider_model(
                agent_runtime.ProviderConfig(
                    provider="openai",
                    model="gpt-6-luna",
                    api_key="secret-test-key",
                ),
                decision=True,
            )
            agent_runtime.provider_model(
                agent_runtime.ProviderConfig(
                    provider="anthropic",
                    model="claude-sonnet-5-5",
                    api_key="secret-test-key",
                ),
                decision=True,
            )

        openai_options = openai.call_args.kwargs
        anthropic_options = anthropic.call_args.kwargs
        self.assertEqual(openai_options["reasoning_effort"], "low")
        self.assertNotIn("effort", openai_options)
        self.assertEqual(anthropic_options["effort"], "low")
        self.assertNotIn("reasoning_effort", anthropic_options)
        for options in (openai_options, anthropic_options):
            self.assertEqual(options["timeout"], agent_runtime.DECISION_TIMEOUT_SECONDS)
            self.assertEqual(options["max_retries"], agent_runtime.DECISION_MAX_RETRIES)

    def test_chat_models_use_the_configured_effort_and_other_models_keep_the_provider_default(self):
        with (
            mock.patch("langchain_openai.ChatOpenAI") as openai,
            mock.patch.object(agent_runtime, "_pooled_chat_anthropic", return_value=(anthropic := mock.Mock())),
        ):
            for effort in ("low", "medium", "high"):
                agent_runtime.provider_model(
                    agent_runtime.ProviderConfig("openai", "gpt-6-luna", "secret-test-key", effort)
                )
                self.assertEqual(openai.call_args.kwargs["reasoning_effort"], effort)
                agent_runtime.provider_model(
                    agent_runtime.ProviderConfig("anthropic", "claude-sonnet-5-5", "secret-test-key", effort)
                )
                self.assertEqual(anthropic.call_args.kwargs["effort"], effort)
            agent_runtime.provider_model(agent_runtime.ProviderConfig("openai", "gpt-6-luna", "secret-test-key"))
            self.assertNotIn("reasoning_effort", openai.call_args.kwargs)
            agent_runtime.provider_model(
                agent_runtime.ProviderConfig("anthropic", "claude-sonnet-5-5", "secret-test-key")
            )
            self.assertNotIn("effort", anthropic.call_args.kwargs)
            agent_runtime.provider_model(
                agent_runtime.ProviderConfig("openai", "gpt-6-luna", "secret-test-key", "high"), decision=True
            )
            self.assertEqual(openai.call_args.kwargs["reasoning_effort"], "low")
        with self.assertRaises(agent_runtime.RuntimeContractError):
            agent_runtime.ProviderConfig("openai", "gpt-6-luna", "secret-test-key", "xhigh")

    def test_a_stalled_route_decision_is_retried_exactly_once(self):
        import httpx

        attempts: list[str] = []

        def stalled(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            raise httpx.ReadTimeout("synthetic stalled provider", request=request)

        client = httpx.Client(transport=httpx.MockTransport(stalled))
        self.addCleanup(client.close)
        model = agent_runtime.provider_model(
            agent_runtime.ProviderConfig(provider="openai", model="gpt-6-luna", api_key="secret-test-key"),
            http_client=client,
            decision=True,
        )
        import openai

        with mock.patch("openai._base_client.time.sleep"), self.assertRaises(openai.APITimeoutError):
            model.invoke("route this objective")
        self.assertEqual(len(attempts), 2)


if __name__ == "__main__":
    unittest.main()

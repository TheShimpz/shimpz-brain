"""Provider client construction for ordinary Brain turns and the structured route decision."""

from __future__ import annotations

import unittest
from unittest import mock

import agent_runtime


class ProviderModelTests(unittest.TestCase):
    def test_openai_uses_responses_api_without_changing_anthropic(self):
        with (
            mock.patch("langchain_openai.ChatOpenAI") as openai,
            mock.patch("langchain_anthropic.ChatAnthropic") as anthropic,
        ):
            agent_runtime.provider_model(
                agent_runtime.ProviderConfig(
                    provider="openai",
                    model="gpt-6-sol",
                    api_key="secret-test-key",
                )
            )
            agent_runtime.provider_model(
                agent_runtime.ProviderConfig(
                    provider="anthropic",
                    model="claude-sonnet-5",
                    api_key="secret-test-key",
                )
            )

        self.assertTrue(openai.call_args.kwargs["use_responses_api"])
        self.assertNotIn("use_responses_api", anthropic.call_args.kwargs)
        self.assertEqual(set(openai.call_args.kwargs) - {"use_responses_api"}, set(anthropic.call_args.kwargs))

    def test_decision_models_use_provider_specific_low_effort_with_one_retry(self):
        with (
            mock.patch("langchain_openai.ChatOpenAI") as openai,
            mock.patch("langchain_anthropic.ChatAnthropic") as anthropic,
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
                    model="claude-sonnet-5",
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

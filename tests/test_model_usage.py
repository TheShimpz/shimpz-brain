"""Observed model usage per Brain operation: counts only what provider responses reported (ADR-0082)."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

import agent_runtime
import model_usage
import runtime_api
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import ToolAwareFakeModel
from test_runtime_api import AUTH, TOKEN, body

USAGE = {
    "input_tokens": 120,
    "output_tokens": 30,
    "total_tokens": 150,
    "input_token_details": {"cache_read": 80, "cache_creation": 20},
}


class FailingModel(FakeMessagesListChatModel):
    def _generate(self, *_args: Any, **_kwargs: Any):
        raise RuntimeError("provider down")


def _reply(**extra: Any) -> AIMessage:
    return AIMessage(content="ok", **extra)


class UsageTests(unittest.TestCase):
    def test_reported_unreported_and_failed_calls_are_counted_apart(self):
        model = FakeMessagesListChatModel(responses=[_reply(usage_metadata=USAGE), _reply()])
        failing = FailingModel(responses=[_reply()])

        def work() -> str:
            model.invoke([HumanMessage(content="a")])
            model.invoke([HumanMessage(content="b")])
            with self.assertRaises(RuntimeError):
                failing.invoke([HumanMessage(content="c")])
            return "done"

        result, usage = model_usage.measure(work)
        self.assertEqual(result, "done")
        self.assertEqual(
            usage,
            {
                "model_calls": 3,
                "failed_calls": 1,
                "unreported_calls": 1,
                "input_tokens": 120,
                "output_tokens": 30,
                "cache_read_tokens": 80,
                "cache_write_tokens": 20,
                "provider_requests": 0,
            },
        )

    def test_anthropic_cache_writes_split_by_lifetime_are_counted(self):
        split = {
            **USAGE,
            "input_token_details": {
                "cache_read": 0,
                "cache_creation": 0,
                "ephemeral_5m_input_tokens": 6003,
                "ephemeral_1h_input_tokens": 0,
            },
        }
        model = FakeMessagesListChatModel(responses=[_reply(usage_metadata=split)])
        _result, usage = model_usage.measure(lambda: model.invoke([HumanMessage(content="a")]))
        self.assertEqual((usage["cache_read_tokens"], usage["cache_write_tokens"]), (0, 6003))

    def test_calls_outside_a_measurement_and_in_a_nested_one_stay_apart(self):
        model = FakeMessagesListChatModel(responses=[_reply(usage_metadata=USAGE)] * 3)
        model.invoke([HumanMessage(content="outside")])
        with model_usage.measured() as outer:
            model.invoke([HumanMessage(content="outer")])
            _inner, inner = model_usage.measure(lambda: model.invoke([HumanMessage(content="inner")]))
        self.assertEqual(outer.to_dict()["model_calls"], 1)
        self.assertEqual(inner["model_calls"], 1)

    def test_worker_threads_that_copy_the_context_report_to_the_same_measurement(self):
        model = FakeMessagesListChatModel(responses=[_reply(usage_metadata=USAGE)] * 4)

        def work() -> None:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for future in [
                    pool.submit(copy_context().run, model.invoke, [HumanMessage(content=str(index))])
                    for index in range(4)
                ]:
                    future.result()

        _result, usage = model_usage.measure(work)
        self.assertEqual(usage["model_calls"], 4)
        self.assertEqual(usage["input_tokens"], 480)


class EndpointTests(unittest.TestCase):
    def test_a_real_graph_turn_reports_its_model_usage_through_the_turn_endpoint(self):
        model = ToolAwareFakeModel(responses=[AIMessage(content="Hello.", usage_metadata=USAGE)])
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
        app = runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)
        response = TestClient(app).post("/v1/turns", json=body(), headers=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["usage"],
            {
                "model_calls": 1,
                "failed_calls": 0,
                "unreported_calls": 0,
                "input_tokens": 120,
                "output_tokens": 30,
                "cache_read_tokens": 80,
                "cache_write_tokens": 20,
                "provider_requests": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()

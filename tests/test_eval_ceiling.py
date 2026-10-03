"""Provider-free checks that the evaluation ceiling bounds and reserves every request on the wire (ADR-0094)."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path

import httpx
import openai
import provider_client
from eval import ceiling
from eval import cost as eval_cost
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

OPENAI_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-6-luna",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 4},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}
CHAT_RESPONSE = {
    "id": "chat_1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-6-luna",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}
ANTHROPIC_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 30},
}


class Recorder:
    def __init__(self, body: dict | None, status: int = 200) -> None:
        self.body, self.status, self.requests = body, status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(self.status, json=self.body if self.body is not None else {})


class Huge(BaseModel):
    answer: str = Field(description="x" * 100_000)


class CeilingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.state = Path(self.directory.name, "budget.json")
        self.ceiling = ceiling.Ceiling(1.0, 32_000, self.state)
        self.ceiling.install()

    def tearDown(self):
        self.ceiling.uninstall()
        self.directory.cleanup()

    def _openai(self, recorder: Recorder, **options) -> ChatOpenAI:
        client = httpx.Client(transport=httpx.MockTransport(recorder))
        return ChatOpenAI(model="gpt-6-luna", api_key="sk-test-0123456789", http_client=client, **options)

    def _anthropic(self, recorder: Recorder):
        model = provider_client._pooled_chat_anthropic()(model="claude-sonnet-5-5", api_key="sk-ant-test-0123456789")
        model._http_client = httpx.Client(transport=httpx.MockTransport(recorder))
        return model

    def test_no_override_can_raise_the_output_limit(self):
        recorder = Recorder(OPENAI_RESPONSE)
        model = self._openai(recorder, use_responses_api=True)
        self.assertEqual((model.max_retries, model.max_tokens), (0, 32_000))
        model.invoke("hi", max_output_tokens=64_000, extra_body={"max_output_tokens": 64_000, "max_tokens": 64_000})
        path, body = recorder.requests[0]
        self.assertEqual((path, body["max_output_tokens"], body["max_tokens"]), ("/v1/responses", 32_000, 32_000))
        chat = Recorder(CHAT_RESPONSE)
        self._openai(chat).invoke("hi", extra_body={"max_completion_tokens": 64_000})
        self.assertEqual(chat.requests[0][1]["max_completion_tokens"], 32_000)
        anthropic = Recorder(ANTHROPIC_RESPONSE)
        self._anthropic(anthropic).invoke("hi", max_tokens=64_000, extra_body={"max_tokens": 64_000})
        self.assertEqual(anthropic.requests[0][1]["max_tokens"], 32_000)

    def test_the_reservation_covers_the_expanded_body_and_prices_the_sent_model(self):
        structured = json.loads(json.dumps(OPENAI_RESPONSE))
        structured["output"][0]["content"][0]["text"] = '{"answer": "ok"}'
        model = self._openai(Recorder(structured), use_responses_api=True)
        model.with_structured_output(Huge, method="json_schema").invoke("hi")
        self.assertGreaterEqual(self.ceiling.max_reservation, eval_cost.call_bound("gpt-6-luna", 100_000, 32_000))
        spent = self.ceiling.budget.spent
        model.invoke("hi", model="gpt-6.1-sol")
        expected = eval_cost.cost(
            eval_cost.Usage(model_calls=1, input_tokens=10, output_tokens=5, cache_read_tokens=4), "gpt-6.1-sol"
        )
        self.assertAlmostEqual(self.ceiling.budget.spent - spent, expected.usd)
        before = self.ceiling.budget.spent
        self._anthropic(Recorder(ANTHROPIC_RESPONSE)).invoke("hi")
        anthropic = eval_cost.Usage(
            model_calls=1, input_tokens=60, output_tokens=5, cache_read_tokens=20, cache_write_tokens=30
        )
        self.assertAlmostEqual(self.ceiling.budget.spent - before, eval_cost.cost(anthropic, "claude-sonnet-5-5").usd)

    def test_unsupported_requests_and_requests_over_the_cap_are_never_sent(self):
        recorder = Recorder(OPENAI_RESPONSE)
        client = httpx.Client(transport=httpx.MockTransport(recorder))
        for request in (
            httpx.Request("GET", "https://api.openai.com/v1/models"),
            httpx.Request("POST", "https://api.openai.com/v1/responses", json={"model": "gpt-6-luna", "stream": True}),
            httpx.Request("POST", "https://api.anthropic.com/v1/messages", content=b"not json"),
        ):
            with self.assertRaises(ceiling.UnsupportedRequestError):
                client.send(request)
        asynchronous = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
        with self.assertRaises(ceiling.UnsupportedRequestError):
            asyncio.run(asynchronous.post("https://api.openai.com/v1/responses", json={"model": "gpt-6-luna"}))
        self.assertEqual(asyncio.run(asynchronous.get("https://example.net/x")).status_code, 200)
        self.assertEqual((recorder.requests[:-1], self.ceiling.counts["unsupported"]), ([], 4))
        self.assertEqual(client.get("https://example.net/x").status_code, 200)
        # An evaluation-only price holds only up to its input bound: a longer request is refused, a shorter one runs.
        limit = eval_cost.PRICED_INPUT_TOKENS["gpt-5.6-sol"] - ceiling.REQUEST_ALLOWANCE_TOKENS
        long = {"model": "gpt-5.6-sol", "input": "x" * limit}
        with self.assertRaises(ceiling.UnsupportedRequestError):
            client.post("https://api.openai.com/v1/responses", json=long)
        self.assertEqual(self.ceiling.counts["unsupported"], 5)
        spent = self.ceiling.budget.spent
        short = client.post("https://api.openai.com/v1/responses", json={"model": "gpt-5.6-sol"})
        self.assertEqual(short.status_code, 200)
        priced = eval_cost.Usage(model_calls=1, input_tokens=10, output_tokens=5, cache_read_tokens=4)
        self.assertAlmostEqual(self.ceiling.budget.spent - spent, eval_cost.cost(priced, "gpt-5.6-sol").usd)
        self.ceiling.uninstall()
        self.ceiling = ceiling.Ceiling(0.0001, 32_000, self.state)
        self.ceiling.install()
        with self.assertRaises(openai.APIConnectionError) as refused:
            self._openai(recorder, use_responses_api=True).invoke("hi")
        self.assertIsInstance(refused.exception.__cause__, eval_cost.BudgetExhaustedError)
        self.assertEqual((len(recorder.requests), self.ceiling.counts["refused"]), (3, 1))

    def test_failed_and_unreported_requests_keep_their_reservation(self):
        with self.assertRaises(openai.InternalServerError):
            self._openai(Recorder(None, 500), use_responses_api=True).invoke("hi")
        self._openai(Recorder({**OPENAI_RESPONSE, "usage": None}), use_responses_api=True).invoke("hi")

        def broken(_request):
            raise httpx.ConnectError("down")

        model = ChatOpenAI(
            model="gpt-6-luna",
            api_key="sk-test-0123456789",
            http_client=httpx.Client(transport=httpx.MockTransport(broken)),
        )
        with self.assertRaises(openai.APIConnectionError):
            model.invoke("hi")
        garbled = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, text="garbled")))
        garbled.post("https://api.anthropic.com/v1/messages", json={"model": "claude-sonnet-5-5"})
        self.assertEqual((self.ceiling.counts["failed"], self.ceiling.counts["unreported"]), (2, 2))
        self.assertEqual(self.ceiling.budget.summary()["unknown_settlements"], 4)
        self.assertIsNone(ceiling.usage_of("api.openai.com", ["no usage"]))
        ceiling.Ceiling(1.0, 10).write_state()

    def test_malformed_usage_keeps_the_whole_reservation_and_always_settles(self):
        malformed = (
            {},
            {"input_tokens": 10},
            {"input_tokens": 10, "output_tokens": "5"},
            {"input_tokens": True, "output_tokens": 5},
            {"input_tokens": -1, "output_tokens": 5},
            {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": 7},
            {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 1.5}},
        )
        for usage in malformed:
            with self.subTest(usage=usage):
                self.assertIsNone(ceiling.usage_of("api.openai.com", {"usage": usage}))
        self.assertIsNone(ceiling.usage_of("api.openai.com", {"usage": {"prompt_tokens": 3}}))
        self.assertIsNone(ceiling.usage_of("api.anthropic.com", {"usage": {"input_tokens": 3}}))
        self.assertIsNone(
            ceiling.usage_of(
                "api.anthropic.com", {"usage": {"input_tokens": 3, "output_tokens": 1, "cache_read_input_tokens": "2"}}
            )
        )
        chat = {"prompt_tokens": 3, "completion_tokens": 1, "prompt_tokens_details": None}
        self.assertEqual(ceiling.usage_of("api.openai.com", {"usage": chat}).input_tokens, 3)
        anthropic = {"input_tokens": 3, "output_tokens": 1, "cache_read_input_tokens": None}
        self.assertEqual(ceiling.usage_of("api.anthropic.com", {"usage": anthropic}).input_tokens, 3)
        for usage in ({}, {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": 7}):
            spent = self.ceiling.budget.spent
            client = httpx.Client(transport=httpx.MockTransport(Recorder({**OPENAI_RESPONSE, "usage": usage})))
            client.post("https://api.openai.com/v1/responses", json={"model": "gpt-6-luna"})
            # The whole worst case stays spent: an empty usage is not a known zero.
            self.assertGreaterEqual(self.ceiling.budget.spent - spent, eval_cost.call_bound("gpt-6-luna", 0, 32_000))
        self.assertEqual(self.ceiling.counts["unreported"], 2)

        class Broken(httpx.SyncByteStream):
            def __iter__(self):
                raise httpx.ReadError("reset")

        request = httpx.Request("POST", "https://api.openai.com/v1/responses", json={"model": "gpt-6-luna"})
        admitted, reservation, sent = self.ceiling.admit(request)
        with self.assertRaises(httpx.ReadError):
            self.ceiling.settle(reservation, sent, "api.openai.com", httpx.Response(200, stream=Broken()))
        self.assertEqual(self.ceiling.budget.summary()["unknown_settlements"], 3)
        self.assertEqual(self.ceiling.budget._reserved, 0.0)
        self.assertEqual(json.loads(admitted.content)["max_output_tokens"], 32_000)

    def test_embeddings_reserve_input_only_and_settle_on_prompt_tokens(self):
        body = {"object": "list", "data": [], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 1000}}
        recorder = Recorder(body)
        client = httpx.Client(transport=httpx.MockTransport(recorder))
        request = {"model": "text-embedding-3-small", "input": ["taza azul", "tetera"], "max_output_tokens": 9}
        client.post("https://api.openai.com/v1/embeddings", json=request)
        # The body is sent unchanged: an embeddings request has no output limit to clamp.
        self.assertEqual(recorder.requests, [("/v1/embeddings", request)])
        self.assertAlmostEqual(self.ceiling.budget.spent, 1000 * 0.02 / 1e6)
        self.assertEqual(self.ceiling.counts["requests"], 1)
        sent = json.dumps(request, separators=(",", ":")).encode()
        bound = eval_cost.call_bound("text-embedding-3-small", len(sent) + ceiling.REQUEST_ALLOWANCE_TOKENS, 0)
        self.assertAlmostEqual(self.ceiling.max_reservation, bound)
        self.assertEqual(
            ceiling.usage_of("api.openai.com", {"usage": {"prompt_tokens": 7}}, "/v1/embeddings").input_tokens, 7
        )
        for usage in ({}, {"prompt_tokens": -1}, {"prompt_tokens": 1.5}, {"prompt_tokens": True}):
            self.assertIsNone(ceiling.usage_of("api.openai.com", {"usage": usage}, "/v1/embeddings"))
        unreported = httpx.Client(transport=httpx.MockTransport(Recorder({"object": "list", "usage": {}})))
        spent = self.ceiling.budget.spent
        unreported.post("https://api.openai.com/v1/embeddings", json={"model": "text-embedding-3-large", "input": "x"})
        self.assertGreater(self.ceiling.budget.spent - spent, 0.0)
        self.assertEqual(self.ceiling.counts["unreported"], 1)

    def test_an_unpriced_or_streamed_embeddings_request_is_refused(self):
        recorder = Recorder({"usage": {"prompt_tokens": 1}})
        client = httpx.Client(transport=httpx.MockTransport(recorder))
        for request in ({"model": "gpt-6-luna", "input": "x"}, {"model": "text-embedding-3-small", "stream": True}):
            with self.subTest(request=request), self.assertRaises(ceiling.UnsupportedRequestError):
                client.post("https://api.openai.com/v1/embeddings", json=request)
        with self.assertRaises(ceiling.UnsupportedRequestError):
            client.post("https://api.openai.com/v1/embeddings", content=b"not json")
        self.assertEqual((recorder.requests, self.ceiling.counts["unsupported"]), ([], 3))

    def test_concurrent_state_writes_never_collide(self):
        workers = [threading.Thread(target=lambda: [self.ceiling.write_state() for _ in range(40)]) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(json.loads(self.state.read_text(encoding="utf-8"))["max_output_tokens"], 32_000)
        self.assertEqual([path.name for path in self.state.parent.iterdir()], ["budget.json"])


if __name__ == "__main__":
    unittest.main()

"""Provider-free checks that the evaluation ceiling bounds and reserves every request on the wire (ADR-0094)."""

import asyncio
import gzip
import json
import tempfile
import threading
import unittest
import zlib
from pathlib import Path

import httpx
from eval import ceiling
from eval import cost as eval_cost

OPENAI_USAGE = {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 4}}
OPENAI_RESPONSE = {"id": "resp_1", "status": "completed", "output": [], "usage": OPENAI_USAGE}
CHAT_RESPONSE = {"id": "chat_1", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
ANTHROPIC_RESPONSE = {
    "id": "msg_1",
    "content": [],
    "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 30},
}
RESPONSES = "https://api.openai.com/v1/responses"
MESSAGES = "https://api.anthropic.com/v1/messages"


class Recorder:
    def __init__(self, body: dict | None, status: int = 200) -> None:
        self.body, self.status, self.requests = body, status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(self.status, json=self.body if self.body is not None else {})


def client(recorder) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(recorder))


class CeilingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.state = Path(self.directory.name, "budget.json")
        self.ceiling = ceiling.Ceiling(1.0, 32_000, self.state)
        self.ceiling.install()

    def tearDown(self):
        self.ceiling.uninstall()
        self.directory.cleanup()

    def test_no_field_can_raise_the_output_limit(self):
        recorder = Recorder(OPENAI_RESPONSE)
        client(recorder).post(
            RESPONSES, json={"model": "gpt-6-luna", "max_output_tokens": 64_000, "max_tokens": 64_000}
        )
        path, body = recorder.requests[0]
        self.assertEqual((path, body["max_output_tokens"], body["max_tokens"]), ("/v1/responses", 32_000, 32_000))
        chat = Recorder(CHAT_RESPONSE)
        client(chat).post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-6-luna"})
        self.assertEqual(chat.requests[0][1]["max_completion_tokens"], 32_000)
        anthropic = Recorder(ANTHROPIC_RESPONSE)
        client(anthropic).post(MESSAGES, json={"model": "claude-sonnet-5-5", "max_tokens": 64_000})
        self.assertEqual(anthropic.requests[0][1]["max_tokens"], 32_000)

    def test_a_body_within_the_limit_is_sent_byte_for_byte(self):
        sent = []
        transport = httpx.MockTransport(lambda request: sent.append(request.content) or httpx.Response(200, json={}))
        body = b'{"model": "gpt-6-luna", "max_output_tokens": 100, "input": "\\u00e9"}'
        httpx.Client(transport=transport).post(RESPONSES, content=body)
        self.assertEqual(sent, [body])

    def test_the_reservation_covers_the_sent_body_and_prices_the_model_it_names(self):
        recorder = Recorder(OPENAI_RESPONSE)
        client(recorder).post(RESPONSES, json={"model": "gpt-6-luna", "input": "x" * 100_000})
        self.assertGreaterEqual(self.ceiling.max_reservation, eval_cost.call_bound("gpt-6-luna", 100_000, 32_000))
        spent = self.ceiling.budget.spent
        client(recorder).post(RESPONSES, json={"model": "gpt-6.1-sol"})
        expected = eval_cost.cost(
            eval_cost.Usage(model_calls=1, input_tokens=10, output_tokens=5, cache_read_tokens=4), "gpt-6.1-sol"
        )
        self.assertAlmostEqual(self.ceiling.budget.spent - spent, expected.usd)
        before = self.ceiling.budget.spent
        client(Recorder(ANTHROPIC_RESPONSE)).post(MESSAGES, json={"model": "claude-sonnet-5-5"})
        anthropic = eval_cost.Usage(
            model_calls=1, input_tokens=60, output_tokens=5, cache_read_tokens=20, cache_write_tokens=30
        )
        self.assertAlmostEqual(self.ceiling.budget.spent - before, eval_cost.cost(anthropic, "claude-sonnet-5-5").usd)

    def test_unsupported_requests_are_never_sent(self):
        recorder = Recorder(OPENAI_RESPONSE)
        sender = client(recorder)
        for request in (
            httpx.Request("GET", "https://api.openai.com/v1/models"),
            httpx.Request("POST", RESPONSES, json={"model": "gpt-6-luna", "stream": True}),
            httpx.Request("POST", MESSAGES, content=b"not json"),
            httpx.Request("POST", RESPONSES, json={"model": "gpt-unknown"}),
            httpx.Request("POST", MESSAGES, json={"model": "claude-sonnet-5-5", "max_tokens": "many"}),
        ):
            with self.assertRaises(ceiling.UnsupportedRequestError):
                sender.send(request)
        asynchronous = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
        with self.assertRaises(ceiling.UnsupportedRequestError):
            asyncio.run(asynchronous.post(RESPONSES, json={"model": "gpt-6-luna"}))
        self.assertEqual(asyncio.run(asynchronous.get("https://example.net/x")).status_code, 200)
        self.assertEqual((recorder.requests[:-1], self.ceiling.counts["unsupported"]), ([], 6))
        self.assertEqual(sender.get("https://example.net/x").status_code, 200)

    def test_an_evaluation_only_price_holds_only_up_to_its_input_bound(self):
        sender = client(Recorder(OPENAI_RESPONSE))
        limit = eval_cost.PRICED_INPUT_TOKENS["gpt-5.6-sol"] - ceiling.REQUEST_ALLOWANCE_TOKENS
        with self.assertRaises(ceiling.UnsupportedRequestError):
            sender.post(RESPONSES, json={"model": "gpt-5.6-sol", "input": "x" * limit})
        spent = self.ceiling.budget.spent
        self.assertEqual(sender.post(RESPONSES, json={"model": "gpt-5.6-sol"}).status_code, 200)
        priced = eval_cost.Usage(model_calls=1, input_tokens=10, output_tokens=5, cache_read_tokens=4)
        self.assertAlmostEqual(self.ceiling.budget.spent - spent, eval_cost.cost(priced, "gpt-5.6-sol").usd)

    def test_a_request_over_the_cap_is_refused_before_it_is_sent(self):
        self.ceiling.uninstall()
        self.ceiling = ceiling.Ceiling(0.0001, 32_000, self.state)
        self.ceiling.install()
        recorder = Recorder(OPENAI_RESPONSE)
        with self.assertRaises(eval_cost.BudgetExhaustedError):
            client(recorder).post(RESPONSES, json={"model": "gpt-6-luna"})
        self.assertEqual((recorder.requests, self.ceiling.counts["refused"]), ([], 1))

    def test_a_shared_ledger_reserves_each_try_and_a_refusal_bills_nothing(self):
        self.ceiling.uninstall()
        ledger = eval_cost.SharedBudget.create(Path(self.directory.name, "ledger.json"), 1.0)
        self.ceiling = ceiling.Ceiling(0.0, 32_000, budget=ledger)
        self.ceiling.install()
        answers = iter((httpx.Response(429, json={"error": {}}), httpx.Response(200, json=OPENAI_RESPONSE)))
        sender = httpx.Client(transport=httpx.MockTransport(lambda _request: next(answers)))
        self.assertEqual(sender.post(RESPONSES, json={"model": "gpt-6-luna"}).status_code, 429)
        self.assertEqual(sender.post(RESPONSES, json={"model": "gpt-6-luna"}).status_code, 200)
        self.assertEqual((self.ceiling.counts["requests"], self.ceiling.counts["throttled"]), (2, 1))
        usage = eval_cost.Usage(model_calls=1, input_tokens=10, output_tokens=5, cache_read_tokens=4)
        summary = ledger.summary()
        self.assertAlmostEqual(summary["spent_usd"], eval_cost.cost(usage, "gpt-6-luna").usd, places=6)
        self.assertEqual((summary["unknown_settlements"], summary["unsettled_usd"]), (0, 0))
        # A provider's rejection of the request, such as an account without credit, bills nothing either.
        broke = {"error": {"message": "Your credit balance is too low"}}
        rejected = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(400, json=broke)))
        rejected.post(MESSAGES, json={"model": "claude-sonnet-5-5"})
        self.assertEqual((self.ceiling.counts["rejected"], self.ceiling.counts["terminal"]), (1, 1))
        self.assertEqual(ledger.summary()["spent_usd"], summary["spent_usd"])

    def test_failed_and_unreported_requests_keep_their_reservation(self):
        client(Recorder(None, 500)).post(RESPONSES, json={"model": "gpt-6-luna"})
        client(Recorder({**OPENAI_RESPONSE, "usage": None})).post(RESPONSES, json={"model": "gpt-6-luna"})

        def broken(_request):
            raise httpx.ConnectError("down")

        with self.assertRaises(httpx.ConnectError):
            client(broken).post(RESPONSES, json={"model": "gpt-6-luna"})
        garbled = client(lambda _request: httpx.Response(200, text="garbled"))
        garbled.post(MESSAGES, json={"model": "claude-sonnet-5-5"})
        self.assertEqual((self.ceiling.counts["failed"], self.ceiling.counts["unreported"]), (2, 2))
        self.assertEqual(self.ceiling.budget.summary()["unknown_settlements"], 4)
        self.assertIsNone(ceiling.usage_of("api.openai.com", ["no usage"]))
        ceiling.Ceiling(1.0, 10).write_state()

    def test_malformed_usage_is_unreported(self):
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

    def test_malformed_usage_keeps_the_whole_reservation_and_always_settles(self):
        for usage in ({}, {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": 7}):
            spent = self.ceiling.budget.spent
            client(Recorder({**OPENAI_RESPONSE, "usage": usage})).post(RESPONSES, json={"model": "gpt-6-luna"})
            # The whole worst case stays spent: an empty usage is not a known zero.
            self.assertGreaterEqual(self.ceiling.budget.spent - spent, eval_cost.call_bound("gpt-6-luna", 0, 32_000))
        self.assertEqual(self.ceiling.counts["unreported"], 2)

        class Broken(httpx.SyncByteStream):
            def __iter__(self):
                raise httpx.ReadError("reset")

        request = httpx.Request("POST", RESPONSES, json={"model": "gpt-6-luna"})
        admitted, reservation, sent = self.ceiling.admit(request)
        with self.assertRaises(httpx.ReadError):
            self.ceiling.settle(reservation, sent, "api.openai.com", httpx.Response(200, stream=Broken()))
        self.assertEqual(self.ceiling.budget.summary()["unknown_settlements"], 3)
        self.assertEqual(self.ceiling.budget._reserved, 0.0)
        self.assertEqual(json.loads(admitted.content)["max_output_tokens"], 32_000)

    def test_an_openai_cache_write_is_read_and_priced_as_a_cache_write(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 50},
        }
        reported = ceiling.usage_of("api.openai.com", {"usage": usage})
        self.assertEqual(
            (reported.cache_read_tokens, reported.cache_write_tokens, reported.fresh_input_tokens), (40, 50, 10)
        )
        self.assertAlmostEqual(
            eval_cost.cost(reported, "gpt-6-luna").usd, (10 * 0.1 + 40 * 0.01 + 50 * 0.125 + 5 * 0.5) * 1e-6
        )
        malformed = {**usage, "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": "50"}}
        self.assertIsNone(ceiling.usage_of("api.openai.com", {"usage": malformed}))

    def test_embeddings_reserve_input_only_and_settle_on_prompt_tokens(self):
        body = {"object": "list", "data": [], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 1000}}
        recorder = Recorder(body)
        request = {"model": "text-embedding-3-small", "input": ["taza azul", "tetera"], "max_output_tokens": 9}
        client(recorder).post("https://api.openai.com/v1/embeddings", json=request)
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
        unreported = client(Recorder({"object": "list", "usage": {}}))
        spent = self.ceiling.budget.spent
        unreported.post("https://api.openai.com/v1/embeddings", json={"model": "text-embedding-3-large", "input": "x"})
        self.assertGreater(self.ceiling.budget.spent - spent, 0.0)
        self.assertEqual(self.ceiling.counts["unreported"], 1)

    def test_an_unpriced_or_streamed_embeddings_request_is_refused(self):
        recorder = Recorder({"usage": {"prompt_tokens": 1}})
        sender = client(recorder)
        for request in ({"model": "gpt-6-luna", "input": "x"}, {"model": "text-embedding-3-small", "stream": True}):
            with self.subTest(request=request), self.assertRaises(ceiling.UnsupportedRequestError):
                sender.post("https://api.openai.com/v1/embeddings", json=request)
        with self.assertRaises(ceiling.UnsupportedRequestError):
            sender.post("https://api.openai.com/v1/embeddings", content=b"not json")
        self.assertEqual((recorder.requests, self.ceiling.counts["unsupported"]), ([], 3))

    def test_concurrent_state_writes_never_collide(self):
        workers = [threading.Thread(target=lambda: [self.ceiling.write_state() for _ in range(40)]) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(json.loads(self.state.read_text(encoding="utf-8"))["max_output_tokens"], 32_000)
        self.assertEqual([path.name for path in self.state.parent.iterdir()], ["budget.json"])


class EvidenceTests(unittest.TestCase):
    def test_only_an_answer_that_stopped_at_the_output_limit_is_clamped(self):
        stopped = (
            {"stop_reason": "max_tokens"},
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            {"choices": [{"finish_reason": "stop"}, {"finish_reason": "length"}]},
        )
        for body in stopped:
            with self.subTest(body=body):
                self.assertTrue(ceiling.clamped(body))
        complete = (
            [],
            {"stop_reason": "end_turn"},
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
            {"incomplete_details": None},
            {"choices": ["length"]},
            {"choices": "length"},
        )
        for body in complete:
            with self.subTest(body=body):
                self.assertFalse(ceiling.clamped(body))

    def test_a_terminal_refusal_is_a_refused_key_or_an_account_without_credit_or_quota(self):
        for status in (401, 402, 403):
            self.assertTrue(ceiling.terminal(status, b"{}"))
        self.assertTrue(ceiling.terminal(400, b'{"error": "Your credit balance is too low"}'))
        self.assertTrue(ceiling.terminal(429, b'{"error": {"code": "insufficient_quota"}}'))
        for status, body in ((429, b"{}"), (500, b"billing"), (400, b"{}")):
            self.assertFalse(ceiling.terminal(status, body))

    def test_a_coded_answer_is_read_after_its_coding_and_an_unknown_coding_is_unreadable(self):
        self.assertEqual(ceiling.decoded(gzip.compress(b"{}"), "gzip"), b"{}")
        self.assertEqual(ceiling.decoded(zlib.compress(b"{}"), "deflate"), b"{}")
        self.assertEqual(ceiling.decoded(b"{}", ""), b"{}")
        self.assertIsNone(ceiling.decoded(b"{}", "br"))
        self.assertIsNone(ceiling.decoded(b"not gzip", "gzip"))


if __name__ == "__main__":
    unittest.main()

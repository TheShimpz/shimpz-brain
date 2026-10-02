"""Provider-free checks that the evaluation ceiling bounds and reserves every request at dispatch (ADR-0094)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import agent_runtime
import httpx
import openai
from eval import ceiling
from eval import cost as eval_cost
from langchain_openai import ChatOpenAI

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
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}
ANTHROPIC_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class Recorder:
    def __init__(self, body: dict | None, status: int = 200) -> None:
        self.body, self.status, self.requests = body, status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(self.status, json=self.body or {})


class CeilingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.state = Path(self.directory.name, "budget.json")
        self.ceiling = ceiling.Ceiling(1.0, 32_000, self.state)
        self.ceiling.install()

    def tearDown(self):
        self.ceiling.uninstall()
        self.directory.cleanup()

    def _openai(self, recorder: Recorder) -> ChatOpenAI:
        client = httpx.Client(transport=httpx.MockTransport(recorder))
        return ChatOpenAI(model="gpt-6-luna", api_key="sk-test-0123456789", use_responses_api=True, http_client=client)

    def _anthropic(self, recorder: Recorder):
        model = agent_runtime._pooled_chat_anthropic()(model="claude-sonnet-5-5", api_key="sk-ant-test-0123456789")
        model._http_client = httpx.Client(transport=httpx.MockTransport(recorder))
        return model

    def test_a_request_override_cannot_raise_the_output_limit(self):
        recorder = Recorder(OPENAI_RESPONSE)
        model = self._openai(recorder)
        self.assertEqual((model.max_retries, model.max_tokens), (0, 32_000))
        model.invoke("hi", max_completion_tokens=64_000, max_output_tokens=64_000)
        self.assertEqual(recorder.requests[0]["max_output_tokens"], 32_000)
        self.assertNotIn("max_completion_tokens", recorder.requests[0])
        anthropic = Recorder(ANTHROPIC_RESPONSE)
        self._anthropic(anthropic).invoke("hi", max_tokens=64_000)
        self.assertEqual(anthropic.requests[0]["max_tokens"], 32_000)
        state = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual((state["requests"], state["unknown_settlements"], state["reservations_exceeded"]), (2, 0, 0))

    def test_the_reservation_covers_every_payload_component(self):
        model = self._openai(Recorder(OPENAI_RESPONSE))
        small = {"input": [{"role": "user", "content": "hi"}]}
        large = {**small, "text": {"format": {"type": "json_schema", "schema": {"description": "x" * 100_000}}}}
        amounts = []
        for payload in (small, large):
            self.ceiling.bound(model, dict(payload))
            amounts.append(self.ceiling._pending.value[0].amount)
            self.ceiling.settle(None)
        self.assertGreaterEqual(amounts[1] - amounts[0], eval_cost.call_bound("gpt-6-luna", 100_000, 0))
        chat = ChatOpenAI(model="gpt-6-luna", api_key="sk-test-0123456789")
        self.assertEqual(self.ceiling.bound(chat, {"messages": []})["max_completion_tokens"], 32_000)

    def test_a_request_over_the_cap_is_never_sent(self):
        self.ceiling.uninstall()
        self.ceiling = ceiling.Ceiling(0.0001, 32_000, self.state)
        self.ceiling.install()
        recorder = Recorder(OPENAI_RESPONSE)
        with self.assertRaises(eval_cost.BudgetExhaustedError):
            self._openai(recorder).invoke("hi")
        self.assertEqual((recorder.requests, self.ceiling.counts["refused"]), ([], 1))

    def test_failed_and_unreported_requests_keep_their_reservation(self):
        failing = Recorder(None, 500)
        with self.assertRaises(openai.InternalServerError):
            self._openai(failing).invoke("hi")
        unreported = Recorder({**OPENAI_RESPONSE, "usage": None})
        self._openai(unreported).invoke("hi")
        summary = self.ceiling.budget.summary()
        self.assertEqual((self.ceiling.counts["failed"], self.ceiling.counts["unreported"]), (1, 1))
        self.assertEqual(summary["unknown_settlements"], 2)
        self.ceiling.settle(None)
        self.ceiling.settle(object())
        self.assertEqual((self.ceiling.counts["unreserved"], self.ceiling.counts["failed"]), (1, 1))
        ceiling.Ceiling(1.0, 10).write_state()


if __name__ == "__main__":
    unittest.main()

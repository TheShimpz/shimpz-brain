"""Provider-free checks for the experiment-only Jev decisions of the engineering arms (ADR-0094)."""

from __future__ import annotations

import json
import unittest

import httpx
from eval import jev, large_api


class JevTests(unittest.TestCase):
    def _transport(self, answers, model=jev.MODEL):
        sent = []

        def send(body):
            sent.append(body)
            return {"model": model, "answers": answers, "usage": {"input_tokens": 700, "output_tokens": 9}}

        return send, sent

    def test_group_selection_reads_one_noul_per_group(self):
        questions = jev.group_questions(large_api.GROUPS)
        answers = {key: {"type": "noul", "noul": 0.9 if "cache" in key else 0.1} for key in questions}
        send, sent = self._transport(answers)
        decision = jev.ask(send, "Purge the cache.", questions)
        probabilities = jev.selected_groups(decision, large_api.GROUPS)
        self.assertEqual([g for g, p in probabilities.items() if p >= jev.GROUP_THRESHOLD], ["cache"])
        self.assertAlmostEqual(decision.usd, 700 * 0.042e-6)
        self.assertEqual(sent[0]["state"], {"current_message": "Purge the cache."})
        bad = dict(answers, group_cache={"type": "choice"})
        with self.assertRaises(jev.JevError):
            jev.selected_groups(jev.ask(self._transport(bad)[0], "x", questions), large_api.GROUPS)
        with self.assertRaises(jev.JevError):
            jev.selected_groups(
                jev.ask(self._transport(dict(answers, group_cache={"type": "noul", "noul": 2}))[0], "x", questions),
                large_api.GROUPS,
            )

    def test_routing_sends_only_a_confident_simple_turn_to_luna(self):
        def decision(choice, confidence):
            send, _ = self._transport({"route": {"type": "choice", "choice": choice, "confidence": confidence}})
            return jev.ask(send, "x", jev.ROUTE_QUESTION)

        self.assertTrue(jev.simple(decision("simple-write", 0.9)))
        self.assertFalse(jev.simple(decision("simple-read", 0.5)))
        self.assertFalse(jev.simple(decision("ambiguous", 0.99)))
        self.assertEqual(jev.route(decision("compound-or-sensitive-write", 0.8)), ("compound-or-sensitive-write", 0.8))
        with self.assertRaises(jev.JevError):
            jev.route(decision("other", 0.9))
        send, _ = self._transport(None)
        with self.assertRaises(jev.JevError):
            jev.ask(send, "x", jev.ROUTE_QUESTION)
        with self.assertRaises(jev.JevError):
            jev.route(jev.ask(self._transport({"route": {}})[0], "x", jev.ROUTE_QUESTION))

        def failing(_body):
            raise OSError("down")

        with self.assertRaises(jev.JevError):
            jev.ask(failing, "x", jev.ROUTE_QUESTION)
        with self.assertRaises(jev.JevError):
            jev.ask(self._transport({"route": {}}, model="jev-0")[0], "x", jev.ROUTE_QUESTION)

    def test_the_http_transport_posts_with_the_key_and_refuses_errors(self):
        seen = []

        def handler(request):
            seen.append(request)
            if json.loads(request.content)["state"]["current_message"] == "fail":
                return httpx.Response(429)
            return httpx.Response(200, json={"model": jev.MODEL, "answers": {}, "usage": {"input_tokens": 1}})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        send = jev.http_transport("jev-key-0123", client)
        self.assertEqual(send({"state": {"current_message": "ok"}})["model"], jev.MODEL)
        self.assertEqual(seen[0].headers["authorization"], "Bearer jev-key-0123")
        with self.assertRaises(OSError):
            send({"state": {"current_message": "fail"}})
        self.assertTrue(callable(jev.http_transport("jev-key-0123")))


if __name__ == "__main__":
    unittest.main()

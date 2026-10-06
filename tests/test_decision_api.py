"""The stateless structured-decision endpoints: Action labels, capability plans, and intent routes."""

from __future__ import annotations

import unittest
from unittest import mock

import agent_runtime
import intent_route
import runtime_api
from fastapi.testclient import TestClient
from test_runtime_api import AUTH, NO_USAGE, SECRET, TOKEN, FakeRuntime, body, client


class DecisionApiTests(unittest.TestCase):
    def test_action_labels_are_authenticated_stateless_and_closed(self):
        runtime = FakeRuntime()
        payload = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
            "locale": "pt",
            "actions": ["list-zones", "get-zone"],
        }
        api = client(runtime)

        self.assertEqual(api.post("/v1/action-labels", json=payload).status_code, 401)
        response = api.post("/v1/action-labels", json=payload, headers=AUTH)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "labels": [
                    {"id": "list-zones", "label": "Listar zonas DNS"},
                    {"id": "get-zone", "label": "Consultar zona DNS"},
                ],
                "usage": NO_USAGE,
            },
        )
        call = runtime.calls[0]
        self.assertEqual(call[0], "action_labels")
        self.assertEqual(call[1].api_key, SECRET)
        self.assertEqual(call[2], "pt")
        self.assertEqual(call[3], ("list-zones", "get-zone"))
        self.assertNotIn(SECRET, response.text)

    def test_capability_plan_is_authenticated_stateless_and_closed(self):
        runtime = FakeRuntime()
        payload = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
            "objective": "Configure um domínio e envie uma mensagem",
            "candidates": [
                {
                    "id": "shimpz-cloudflare",
                    "name": "Shimpz Cloudflare",
                    "summary": "Manage DNS.",
                    "actions": ["dns.read", "dns.write"],
                    "integrations": [{"id": "cloudflare", "provider": "cloudflare"}],
                },
                {
                    "id": "shimpz-whatsapp",
                    "name": "Shimpz WhatsApp",
                    "summary": "Send messages.",
                    "actions": ["messages.send"],
                    "integrations": [{"id": "whatsapp", "provider": "whatsapp"}],
                },
            ],
        }
        api = client(runtime)

        self.assertEqual(api.post("/v1/capability-plan", json=payload).status_code, 401)
        response = api.post("/v1/capability-plan", json=payload, headers=AUTH)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "install-required",
                "assistant_ids": ["shimpz-cloudflare", "shimpz-whatsapp"],
                "usage": NO_USAGE,
            },
        )
        call = runtime.calls[0]
        self.assertEqual(call[0], "capability_plan")
        self.assertEqual(call[1].api_key, SECRET)
        self.assertEqual(call[2], payload["objective"])
        self.assertEqual(tuple(item.id for item in call[3]), ("shimpz-cloudflare", "shimpz-whatsapp"))
        self.assertNotIn(SECRET, response.text)

    def test_capability_plan_has_an_independent_fail_closed_capacity_lane(self):
        runtime = FakeRuntime()
        app = runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)
        for _index in range(runtime_api.CAPABILITY_PLAN_CONCURRENCY):
            self.assertTrue(app.state.capability_plan_slots.acquire(blocking=False))
        response = TestClient(app).post(
            "/v1/capability-plan",
            json={
                "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
                "objective": "Configure DNS",
                "candidates": [
                    {
                        "id": "shimpz-cloudflare",
                        "name": "Shimpz Cloudflare",
                        "summary": "Manage DNS.",
                        "actions": ["dns.read"],
                        "integrations": [],
                    }
                ],
            },
            headers=AUTH,
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {"detail": "Capability planner capacity reached"})
        self.assertEqual(runtime.calls, [])

    def test_intent_route_accepts_a_decision_key_only_for_classification(self):
        route = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
            "objective": "Oi, tudo bem?",
            "expected_intent": None,
            "candidates": [],
            "lifecycle_reference": None,
            "conversation": [],
            "locale": "en",
        }
        decision = {"provider": "typesafe", "api_key": "tsk-test-0123456789abcdef"}
        headers = AUTH
        runtime = FakeRuntime()
        response = client(runtime).post(
            "/v1/intent-route", json={**route, "decision_provider": decision}, headers=headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.decision_key, decision["api_key"])
        self.assertNotIn(decision["api_key"], response.text)
        runtime = FakeRuntime()
        self.assertEqual(client(runtime).post("/v1/intent-route", json=route, headers=headers).status_code, 200)
        self.assertIsNone(runtime.decision_key)
        selection = {
            **route,
            "objective": "cloudflare",
            "expected_intent": "assistant-install",
            "candidates": [{"id": "shimpz-cloudflare", "name": "Shimpz Cloudflare", "summary": "DNS."}],
            "decision_provider": decision,
        }
        for refused in (
            selection,
            {**route, "decision_provider": {**decision, "provider": "openai"}},
            {**route, "decision_provider": {**decision, "api_key": "short"}},
            {**route, "decision_provider": {**decision, "extra": 1}},
        ):
            with self.subTest(body=str(refused.get("decision_provider"))[:60]):
                self.assertEqual(
                    client(FakeRuntime()).post("/v1/intent-route", json=refused, headers=headers).status_code, 422
                )

    def test_intent_route_is_authenticated_stateless_and_closed(self):
        runtime = FakeRuntime()
        api = client(runtime)
        classification = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
            "objective": "tire o cloudflare deste time",
            "expected_intent": None,
            "candidates": [],
            "lifecycle_reference": {
                "id": "shimpz-cloudflare",
                "name": "Shimpz Cloudflare",
            },
            "conversation": [
                {
                    "role": "assistant",
                    "text": "Temos apenas Cloudflare/DNS.",
                    "truncated": False,
                }
            ],
            "locale": "en",
        }

        self.assertEqual(api.post("/v1/intent-route", json=classification).status_code, 401)
        response = api.post("/v1/intent-route", json=classification, headers=AUTH)

        self.assertEqual(
            response.json(),
            {
                "task_follows": False,
                "intent": "assistant-uninstall",
                "query": "cloudflare",
                "assistant_ids": [],
                "reply": "",
                "usage": NO_USAGE,
            },
        )
        call = runtime.calls[0]
        self.assertEqual(call[0], "intent_route")
        self.assertEqual(call[1].api_key, SECRET)
        self.assertEqual(call[2:4], (classification["objective"], None))
        self.assertEqual(call[4], ())
        self.assertEqual(call[5].reference.id, "shimpz-cloudflare")
        self.assertEqual(call[5].conversation[0].text, "Temos apenas Cloudflare/DNS.")
        self.assertNotIn(SECRET, response.text)

        selection = {
            **classification,
            "expected_intent": "assistant-uninstall",
            "candidates": [{"id": "shimpz-cloudflare", "name": "Shimpz Cloudflare", "summary": ""}],
            "lifecycle_reference": None,
            "conversation": [],
            "locale": "pt",
        }
        selected = api.post("/v1/intent-route", json=selection, headers=AUTH)
        self.assertEqual(
            selected.json(),
            {
                "task_follows": False,
                "intent": "assistant-uninstall",
                "query": "",
                "assistant_ids": ["shimpz-cloudflare"],
                "reply": "",
                "usage": NO_USAGE,
            },
        )
        self.assertEqual(runtime.calls[1][4][0].id, "shimpz-cloudflare")
        self.assertEqual((call[6], runtime.calls[1][6]), ("en", "pt"))

    def test_intent_route_response_carries_the_task_continuation(self):
        self.assertEqual(
            runtime_api._intent_route_response(intent_route.IntentRoute("assistant-install", "exa", task_follows=True)),
            {"intent": "assistant-install", "query": "exa", "assistant_ids": [], "reply": "", "task_follows": True},
        )

    def test_intent_route_rejects_retired_or_wrong_lane_context(self):
        runtime = FakeRuntime()
        api = client(runtime)
        base = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
            "objective": "remove it",
            "expected_intent": None,
            "candidates": [],
            "lifecycle_reference": None,
            "conversation": [],
            "locale": "en",
        }
        self.assertIsNone(runtime_api.IntentRouteInput.model_validate(base).runtime_context())
        invalid = (
            {
                **base,
                "pending_intent": "assistant-uninstall",
            },
            {**base, "language_exemplar": "remove it"},
            {key: value for key, value in base.items() if key != "locale"},
            {**base, "locale": None},
            {**base, "locale": "pt-BR"},
            {
                **base,
                "expected_intent": "assistant-uninstall",
                "lifecycle_reference": {"id": "shimpz-cloudflare", "name": "Shimpz Cloudflare"},
            },
            {
                **base,
                "expected_intent": "assistant-uninstall",
                "conversation": [{"role": "user", "text": "remove it", "truncated": False}],
            },
            {
                **base,
                "conversation": [{"role": "system", "text": "remove it", "truncated": False}],
            },
            {
                **base,
                "conversation": [{"role": "user", "text": "x" * 513, "truncated": False}],
            },
            {
                **base,
                "conversation": [{"role": "user", "text": "remove it", "truncated": 1}],
            },
        )

        for payload in invalid:
            with self.subTest(payload=payload):
                response = api.post("/v1/intent-route", json=payload, headers=AUTH)
                self.assertEqual(response.status_code, 422)
        with mock.patch.object(intent_route, "MAX_CONVERSATION_CHARS", 1):
            response = api.post(
                "/v1/intent-route",
                json={
                    **base,
                    "conversation": [{"role": "user", "text": "remove it", "truncated": False}],
                },
                headers=AUTH,
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(runtime.calls, [])

    def test_intent_route_has_an_independent_fail_closed_capacity_lane(self):
        runtime = FakeRuntime()
        app = runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)
        for _index in range(runtime_api.INTENT_ROUTE_CONCURRENCY):
            self.assertTrue(app.state.intent_route_slots.acquire(blocking=False))
        response = TestClient(app).post(
            "/v1/intent-route",
            json={
                "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
                "objective": "hello",
                "expected_intent": None,
                "candidates": [],
                "lifecycle_reference": None,
                "conversation": [],
                "locale": "en",
            },
            headers=AUTH,
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {"detail": "Intent route capacity reached"})
        self.assertEqual(runtime.calls, [])

    def test_action_label_input_rejects_added_duplicate_and_unsafe_values(self):
        runtime = FakeRuntime()
        valid = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
            "locale": "pt",
            "actions": ["list-zones", "get-zone"],
        }
        invalid_values = (
            {**valid, "unexpected": True},
            {**valid, "actions": []},
            {**valid, "actions": ["list-zones", "list-zones"]},
            {**valid, "actions": ["../shell"]},
            {**valid, "locale": 1},
            {**valid, "locale": "pt-BR"},
            {key: value for key, value in valid.items() if key != "locale"},
            {**valid, "language_exemplar": "Liste minhas zonas"},
        )

        for payload in invalid_values:
            with self.subTest(payload=payload):
                response = client(runtime).post("/v1/action-labels", json=payload, headers=AUTH)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(runtime.calls, [])

    def test_a_malformed_request_never_echoes_its_input_or_provider_key(self):
        missing = body()
        missing.pop("conversation")
        wrong = body()
        wrong["provider"]["effort"] = SECRET
        for request_body in (missing, wrong):
            with self.subTest(fields=sorted(request_body)):
                response = client(FakeRuntime()).post("/v1/turns", json=request_body, headers=AUTH)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn(SECRET, response.text)
                self.assertNotIn("input", response.json()["detail"][0])
                self.assertNotIn("ctx", response.json()["detail"][0])

    def test_invalid_action_label_model_output_is_a_redacted_upstream_failure(self):
        runtime = FakeRuntime(error=agent_runtime.ProviderResponseError(f"invalid output beside {SECRET}"))
        response = client(runtime).post(
            "/v1/action-labels",
            json={
                "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
                "locale": "pt",
                "actions": ["list-zones"],
            },
            headers=AUTH,
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "Model provider request failed"})
        self.assertNotIn(SECRET, response.text)


if __name__ == "__main__":
    unittest.main()

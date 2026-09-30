from __future__ import annotations

import secrets
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import get_args
from unittest import mock

import action_labels
import agent_runtime
import capability_plan
import clarification
import intent_route
import model_usage
import runtime_api
from fastapi.testclient import TestClient

TOKEN = secrets.token_hex(24)
SECRET = secrets.token_urlsafe(32)
NO_USAGE = dict.fromkeys(model_usage.FIELDS, 0)


def body(**updates):
    value = {
        "thread_id": "team:hello-pulse:conversation-1",
        "team_name": "  Greeting Crew  ",
        "assistants": [
            {
                "id": "hello-pulse",
                "genesis": "Combine declared greeting Actions for a friendly welcome.",
                "actions": [
                    {
                        "id": "hello",
                        "summary": "Return a greeting.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "additionalProperties": False,
                        },
                    }
                ],
            },
            {
                "id": "backup-greeter",
                "genesis": "Use the backup Action only for a bounded greeting.",
                "actions": [
                    {
                        "id": "hello",
                        "summary": "Return a backup greeting.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "additionalProperties": False,
                        },
                    }
                ],
            },
        ],
        "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET, "effort": "low"},
        "memories": [],
        "skills": [],
        "routines": [],
        "knowledge_writable": True,
        "message": "Hello",
        "conversation": [],
    }
    value.update(updates)
    return value


class FakeRuntime:
    def __init__(self, result=None, error=None):
        self.result = result or agent_runtime.TurnResult(status="completed", reply="Hello.")
        self.error = error
        self.calls = []

    def start(self, context, message, conversation=()):
        self.calls.append(("start", context, message))
        self.conversation = conversation
        if self.error:
            raise self.error
        return self.result

    def resume(self, context, results):
        self.calls.append(("resume", context, results))
        if self.error:
            raise self.error
        return self.result

    def delete_thread(self, thread_id):
        self.calls.append(("delete_thread", thread_id))
        if self.error:
            raise self.error

    def action_labels(self, provider, language_exemplar, action_ids):
        self.calls.append(("action_labels", provider, language_exemplar, action_ids))
        if self.error:
            raise self.error
        return (
            action_labels.ActionLabel(id="list-zones", label="Listar zonas DNS"),
            action_labels.ActionLabel(id="get-zone", label="Consultar zona DNS"),
        )

    def capability_plan(self, provider, objective, candidates):
        self.calls.append(("capability_plan", provider, objective, candidates))
        if self.error:
            raise self.error
        return capability_plan.CapabilityPlan(
            "install-required",
            ("shimpz-cloudflare", "shimpz-whatsapp"),
        )

    def intent_route(self, provider, objective, expected_intent, candidates, context, decision_key=None):
        self.calls.append(("intent_route", provider, objective, expected_intent, candidates, context))
        self.decision_key = decision_key
        if self.error:
            raise self.error
        if expected_intent is None:
            return intent_route.IntentRoute("assistant-uninstall", "cloudflare")
        return intent_route.IntentRoute(expected_intent, assistant_ids=("shimpz-cloudflare",))


def client(runtime, *, raise_server_exceptions=True):
    app = runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


class RuntimeApiTests(unittest.TestCase):
    def test_a_clarification_travels_as_its_closed_shape(self):
        asked = clarification.parse(
            {
                "question": "Qual formato?",
                "options": [{"label": "Lista", "description": ""}, {"label": "Tabela", "description": "Compacta."}],
                "default_index": 1,
            }
        )
        runtime = FakeRuntime(agent_runtime.TurnResult(status="completed", reply=asked.render(), clarification=asked))
        response = client(runtime).post("/v1/turns", json=body(), headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["clarification"], asked.to_dict())
        self.assertEqual(response.json()["reply"], asked.render())

    def test_http_provider_contract_matches_the_runtime_catalog(self):
        provider_annotation = runtime_api.ProviderInput.model_fields["provider"].annotation

        self.assertEqual(frozenset(get_args(provider_annotation)), agent_runtime.PROVIDERS)

    def test_health_is_small_and_does_not_require_a_secret(self):
        response = client(FakeRuntime()).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "runtime": "langgraph"})

    def test_turn_endpoint_requires_the_private_runtime_token(self):
        api = client(FakeRuntime())

        self.assertEqual(api.post("/v1/turns", json=body()).status_code, 401)
        self.assertEqual(
            api.post("/v1/turns", json=body(), headers={"Authorization": "Bearer wrong"}).status_code,
            401,
        )

    def test_action_labels_are_authenticated_stateless_and_closed(self):
        runtime = FakeRuntime()
        payload = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
            "language_exemplar": "  Quero listar minhas zonas DNS  ",
            "actions": ["list-zones", "get-zone"],
        }
        api = client(runtime)

        self.assertEqual(api.post("/v1/action-labels", json=payload).status_code, 401)
        response = api.post(
            "/v1/action-labels",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

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
        self.assertEqual(call[2], "Quero listar minhas zonas DNS")
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
        response = api.post(
            "/v1/capability-plan",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

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
            headers={"Authorization": f"Bearer {TOKEN}"},
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
            "language_exemplar": None,
        }
        decision = {"provider": "typesafe", "api_key": "tsk-test-0123456789abcdef"}
        headers = {"Authorization": f"Bearer {TOKEN}"}
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
        for body in (
            selection,
            {**route, "decision_provider": {**decision, "provider": "openai"}},
            {**route, "decision_provider": {**decision, "api_key": "short"}},
            {**route, "decision_provider": {**decision, "extra": 1}},
        ):
            with self.subTest(body=str(body.get("decision_provider"))[:60]):
                self.assertEqual(
                    client(FakeRuntime()).post("/v1/intent-route", json=body, headers=headers).status_code, 422
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
            "language_exemplar": None,
        }

        self.assertEqual(api.post("/v1/intent-route", json=classification).status_code, 401)
        response = api.post(
            "/v1/intent-route",
            json=classification,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

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
            "language_exemplar": "Desinstala esse então.",
        }
        selected = api.post(
            "/v1/intent-route",
            json=selection,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
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
            "language_exemplar": None,
        }
        self.assertIsNone(runtime_api.IntentRouteInput.model_validate(base).runtime_context())
        invalid = (
            {
                **base,
                "pending_intent": "assistant-uninstall",
            },
            {**base, "language_exemplar": "remove it"},
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
                response = api.post(
                    "/v1/intent-route",
                    json=payload,
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
                self.assertEqual(response.status_code, 422)
        with mock.patch.object(intent_route, "MAX_CONVERSATION_CHARS", 1):
            response = api.post(
                "/v1/intent-route",
                json={
                    **base,
                    "conversation": [{"role": "user", "text": "remove it", "truncated": False}],
                },
                headers={"Authorization": f"Bearer {TOKEN}"},
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
                "language_exemplar": None,
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), {"detail": "Intent route capacity reached"})
        self.assertEqual(runtime.calls, [])

    def test_action_label_input_rejects_added_duplicate_and_unsafe_values(self):
        runtime = FakeRuntime()
        valid = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": SECRET},
            "language_exemplar": "Liste minhas zonas",
            "actions": ["list-zones", "get-zone"],
        }
        invalid_values = (
            {**valid, "unexpected": True},
            {**valid, "actions": []},
            {**valid, "actions": ["list-zones", "list-zones"]},
            {**valid, "actions": ["../shell"]},
            {**valid, "language_exemplar": 1},
            {**valid, "language_exemplar": "hidden\0instruction"},
        )

        for payload in invalid_values:
            with self.subTest(payload=payload):
                response = client(runtime).post(
                    "/v1/action-labels",
                    json=payload,
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(runtime.calls, [])

    def test_a_malformed_request_never_echoes_its_input_or_provider_key(self):
        missing = body()
        missing.pop("conversation")
        wrong = body()
        wrong["provider"]["effort"] = SECRET
        for request_body in (missing, wrong):
            with self.subTest(fields=sorted(request_body)):
                response = client(FakeRuntime()).post(
                    "/v1/turns", json=request_body, headers={"Authorization": f"Bearer {TOKEN}"}
                )
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
                "language_exemplar": "Liste minhas zonas",
                "actions": ["list-zones"],
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "Model provider request failed"})
        self.assertNotIn(SECRET, response.text)

    def test_start_passes_provider_secret_in_memory_but_never_returns_it(self):
        runtime = FakeRuntime()
        response = client(runtime).post(
            "/v1/turns",
            json=body(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "completed",
                "reply": "Hello.",
                "clarification": None,
                "actions": [],
                "memory": [],
                "routine": None,
                "usage": NO_USAGE,
            },
        )
        context = runtime.calls[0][1]
        self.assertEqual(context.provider.api_key, SECRET)
        self.assertEqual(context.team_name, "Greeting Crew")
        self.assertEqual([assistant.id for assistant in context.assistants], ["backup-greeter", "hello-pulse"])
        self.assertEqual([assistant.actions[0].id for assistant in context.assistants], ["hello", "hello"])
        self.assertNotIn(SECRET, response.text)

    def test_start_accepts_an_explicit_brain_only_context(self):
        runtime = FakeRuntime(result=agent_runtime.TurnResult(status="completed", reply="Brain only."))
        response = client(runtime).post(
            "/v1/turns",
            json=body(assistants=[]),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "completed",
                "reply": "Brain only.",
                "clarification": None,
                "actions": [],
                "memory": [],
                "routine": None,
                "usage": NO_USAGE,
            },
        )
        self.assertEqual(runtime.calls[0][1].assistants, ())

    def test_action_request_contains_only_controller_action_data(self):
        runtime = FakeRuntime(
            result=agent_runtime.TurnResult(
                status="action-required",
                actions=(
                    agent_runtime.ActionRequest(
                        interrupt_id="interrupt-1",
                        assistant_id="hello-pulse",
                        action="hello",
                        input={"name": "Ada"},
                    ),
                ),
            )
        )
        response = client(runtime).post(
            "/v1/turns",
            json=body(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "action-required",
                "reply": "",
                "clarification": None,
                "memory": [],
                "routine": None,
                "actions": [
                    {
                        "interrupt_id": "interrupt-1",
                        "assistant_id": "hello-pulse",
                        "action": "hello",
                        "input": {"name": "Ada"},
                    }
                ],
                "usage": NO_USAGE,
            },
        )

    def test_start_requires_a_bounded_conversation_window_and_resume_refuses_one(self):
        entry = {"role": "user", "text": "Earlier", "truncated": False}
        runtime = FakeRuntime()
        response = client(runtime).post(
            "/v1/turns", json=body(conversation=[entry]), headers={"Authorization": f"Bearer {TOKEN}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.conversation, (intent_route.ConversationEntry("user", "Earlier", False),))
        for window in ([entry] * 9, [{**entry, "text": "x" * 513}], [{**entry, "role": "system"}]):
            with self.subTest(size=len(window)):
                response = client(FakeRuntime()).post(
                    "/v1/turns", json=body(conversation=window), headers={"Authorization": f"Bearer {TOKEN}"}
                )
                self.assertEqual(response.status_code, 422)
        missing = body()
        missing.pop("conversation")
        response = client(FakeRuntime()).post("/v1/turns", json=missing, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 422)
        resume = body(results={"interrupt-1": {"status": "ok"}})
        resume.pop("message")
        response = client(FakeRuntime()).post(
            "/v1/turns/resume", json=resume, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        self.assertEqual(response.status_code, 422)

    def test_resume_accepts_only_explicit_interrupt_results(self):
        runtime = FakeRuntime()
        payload = body(message=None)
        payload.pop("message")
        payload.pop("conversation")
        payload["results"] = {"interrupt-1": {"message": "Hello, Ada."}}
        response = client(runtime).post(
            "/v1/turns/resume",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.calls[0][0], "resume")
        self.assertEqual(runtime.calls[0][2], {"interrupt-1": {"message": "Hello, Ada."}})

    def test_thread_deletion_is_authenticated_idempotent_and_closed(self):
        runtime = FakeRuntime()
        api = client(runtime)
        payload = {"thread_id": "team:hello-pulse:conversation-1"}

        self.assertEqual(api.post("/v1/threads/delete", json=payload).status_code, 401)
        self.assertEqual(runtime.calls, [])

        response = api.post(
            "/v1/threads/delete",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "deleted"})
        self.assertEqual(runtime.calls, [("delete_thread", payload["thread_id"])])

        for invalid in (
            {"thread_id": "bad thread"},
            {"thread_id": payload["thread_id"], "unexpected": True},
        ):
            with self.subTest(invalid=invalid):
                response = api.post(
                    "/v1/threads/delete",
                    json=invalid,
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(runtime.calls, [("delete_thread", payload["thread_id"])])

    def test_extra_fields_and_invalid_provider_fail_closed(self):
        api = client(FakeRuntime())
        invalid = body(unexpected_command="forbidden")
        response = api.post("/v1/turns", json=invalid, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 422)

        invalid = body()
        del invalid["assistants"][0]["genesis"]
        response = api.post("/v1/turns", json=invalid, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 422)

        invalid = body()
        invalid["provider"]["provider"] = "codex"
        response = api.post("/v1/turns", json=invalid, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 422)

        invalid = body()
        invalid["assistants"][0]["unexpected"] = "forbidden"
        response = api.post("/v1/turns", json=invalid, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 422)

    def test_unknown_and_cross_provider_models_fail_before_runtime(self):
        runtime = FakeRuntime()
        api = client(runtime)

        for provider, model in (
            ("openai", "gpt-well-formed-but-unknown"),
            ("openai", "claude-sonnet-5-5"),
            ("anthropic", "gpt-6.1-sol"),
        ):
            payload = body()
            payload["provider"] = {"provider": provider, "model": model, "api_key": SECRET, "effort": "low"}
            with self.subTest(provider=provider, model=model):
                response = api.post(
                    "/v1/turns",
                    json=payload,
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"detail": "unsupported model for provider"})

        self.assertEqual(runtime.calls, [])

    def test_chat_turns_require_one_closed_reasoning_effort(self):
        runtime = FakeRuntime()
        api = client(runtime)
        for effort in (None, "", "minimal", "xhigh", "LOW"):
            payload = body()
            if effort is None:
                del payload["provider"]["effort"]
            else:
                payload["provider"]["effort"] = effort
            with self.subTest(effort=effort):
                response = api.post("/v1/turns", json=payload, headers={"Authorization": f"Bearer {TOKEN}"})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(runtime.calls, [])

    def test_malformed_team_names_fail_at_the_closed_http_contract(self):
        api = client(FakeRuntime())

        for team_name in ("", "   ", "Bad\nName", "Bad\x7fName", "x" * 81):
            with self.subTest(team_name=team_name):
                response = api.post(
                    "/v1/turns",
                    json=body(team_name=team_name),
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
                self.assertEqual(response.status_code, 422)

        response = api.post(
            "/v1/turns",
            json=body(team_name=123),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        self.assertEqual(response.status_code, 422)

    def test_an_assistant_with_the_team_admitted_128_actions_is_accepted(self):
        runtime = FakeRuntime()
        payload = body()
        action = payload["assistants"][0]["actions"][0]
        payload["assistants"] = [
            {**payload["assistants"][0], "actions": [{**action, "id": f"action-{index}"} for index in range(128)]}
        ]

        response = client(runtime).post(
            "/v1/turns",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(runtime.calls[0][1].assistants[0].actions), 128)

    def test_every_action_team_can_scope_across_assistants_is_accepted(self):
        # Team scopes at most 16 Assistants of at most 128 Actions each into one request: 2,048 Actions in total.
        action = body()["assistants"][0]["actions"][0]

        def scoped(*counts: int) -> dict[str, object]:
            payload = body()
            payload["assistants"] = [
                {
                    "id": f"assistant-{index}",
                    "genesis": "Bounded Assistant.",
                    "actions": [{**action, "id": f"action-{action_index}"} for action_index in range(count)],
                }
                for index, count in enumerate(counts)
            ]
            return payload

        for counts, status in (
            ((65, 65), 200),
            ((128,) * 16, 200),
            ((128,) * 15 + (129,), 422),
            ((128,) * 16 + (1,), 422),
        ):
            runtime = FakeRuntime()
            with self.subTest(total=sum(counts), assistants=len(counts)):
                response = client(runtime).post(
                    "/v1/turns",
                    json=scoped(*counts),
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
                self.assertEqual(response.status_code, status)
                if status == 200:
                    admitted = runtime.calls[0][1].assistants
                    self.assertEqual(sum(len(item.actions) for item in admitted), sum(counts))

    def test_provider_error_is_generic_and_never_echoes_credential(self):
        runtime = FakeRuntime(error=agent_runtime.ProviderRequestError(f"provider rejected {SECRET}"))
        response = client(runtime).post(
            "/v1/turns",
            json=body(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "Model provider request failed"})
        self.assertNotIn(SECRET, response.text)

    def test_dependency_import_error_remains_an_internal_server_failure(self):
        response = client(
            FakeRuntime(error=ImportError(f"missing dependency beside {SECRET}")),
            raise_server_exceptions=False,
        ).post(
            "/v1/turns",
            json=body(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(SECRET, response.text)

    def test_state_error_is_generic_and_never_echoes_persisted_data(self):
        runtime = FakeRuntime(error=agent_runtime.RuntimeStateError(f"failed to delete {SECRET}"))
        response = client(runtime).post(
            "/v1/threads/delete",
            json={"thread_id": "team:hello-pulse:conversation-1"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Brain runtime state operation failed"})
        self.assertNotIn(SECRET, response.text)

    def test_sqlite_checkpoints_are_owner_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state" / "checkpoints.sqlite3"
            runtime = runtime_api._sqlite_runtime(path)

            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            runtime.close()

    def test_runtime_token_file_is_bounded_decoded_and_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime-token"
            with mock.patch.object(runtime_api, "TOKEN_FILE", path):
                with self.assertRaises(runtime_api.HTTPException) as missing:
                    runtime_api._token_from_file()
                self.assertEqual(missing.exception.status_code, 503)

                for raw in (b"", b" \n", b"x" * (runtime_api.MAX_TOKEN_BYTES + 1), b"\xff"):
                    path.write_bytes(raw)
                    with self.subTest(raw_length=len(raw)), self.assertRaises(runtime_api.HTTPException) as invalid:
                        runtime_api._token_from_file()
                    self.assertEqual(invalid.exception.status_code, 503)

                path.write_bytes(b"  exact-token\n")
                self.assertEqual(runtime_api._token_from_file(), "exact-token")

    def test_owned_lazy_runtime_is_created_once_and_closed_with_the_app(self):
        lazy = FakeRuntime()
        lazy.close = mock.Mock()
        application = runtime_api.create_app(runtime=None, token_reader=lambda: TOKEN)
        with (
            mock.patch.object(runtime_api, "_sqlite_runtime", return_value=lazy) as create_runtime,
            TestClient(application) as api,
        ):
            for _attempt in range(2):
                response = api.post(
                    "/v1/turns",
                    json=body(),
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
                self.assertEqual(response.status_code, 200)
        create_runtime.assert_called_once_with()
        lazy.close.assert_called_once_with()

        uninitialized = runtime_api.create_app(runtime=None, token_reader=lambda: TOKEN)
        with TestClient(uninitialized) as api:
            self.assertEqual(api.get("/health").status_code, 200)

        no_close = FakeRuntime()
        no_close.close = None
        application = runtime_api.create_app(runtime=None, token_reader=lambda: TOKEN)
        with (
            mock.patch.object(runtime_api, "_sqlite_runtime", return_value=no_close),
            TestClient(application) as api,
        ):
            response = api.post(
                "/v1/turns",
                json=body(),
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
            self.assertEqual(response.status_code, 200)

    def test_lazy_runtime_lock_publishes_one_instance_to_concurrent_callers(self):
        application = runtime_api.create_app(runtime=None, token_reader=lambda: TOKEN)
        route = next(route for route in application.routes if getattr(route, "path", None) == "/v1/turns")
        closure = dict(zip(route.endpoint.__code__.co_freevars, route.endpoint.__closure__, strict=True))
        current_runtime = closure["current_runtime"].cell_contents
        second_waiting = threading.Event()

        class ObservedLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.attempts = 0

            def __enter__(self):
                self.attempts += 1
                if self.attempts == 2:
                    second_waiting.set()
                self.lock.acquire()
                return self

            def __exit__(self, *_args):
                self.lock.release()

        application.state.runtime_lock = ObservedLock()
        runtime = FakeRuntime()

        def create_runtime():
            self.assertTrue(second_waiting.wait(timeout=2))
            return runtime

        with (
            mock.patch.object(runtime_api, "_sqlite_runtime", side_effect=create_runtime) as create,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            first = executor.submit(current_runtime)
            second = executor.submit(current_runtime)
            self.assertIs(first.result(timeout=2), runtime)
            self.assertIs(second.result(timeout=2), runtime)
        create.assert_called_once_with()

    def test_sqlite_checkpoint_pruning_keeps_only_each_namespace_latest_state(self):
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        saver = runtime_api.PruningSqliteSaver(connection)
        saver.setup()
        checkpoints = [
            ("thread-a", "", "001"),
            ("thread-a", "", "002"),
            ("thread-a", "subgraph", "003"),
            ("thread-a", "subgraph", "004"),
            ("thread-b", "", "005"),
        ]
        connection.executemany(
            "INSERT INTO checkpoints(thread_id,checkpoint_ns,checkpoint_id,type,checkpoint,metadata) "
            "VALUES(?,?,?,'json',X'00',X'00')",
            checkpoints,
        )
        connection.executemany(
            "INSERT INTO writes(thread_id,checkpoint_ns,checkpoint_id,task_id,idx,channel,type,value) "
            "VALUES(?,?,?,'task',0,'channel','json',X'00')",
            checkpoints,
        )

        saver.prune_thread("thread-a")

        self.assertEqual(
            connection.execute(
                "SELECT thread_id,checkpoint_ns,checkpoint_id FROM checkpoints "
                "ORDER BY thread_id,checkpoint_ns,checkpoint_id"
            ).fetchall(),
            [("thread-a", "", "002"), ("thread-a", "subgraph", "004"), ("thread-b", "", "005")],
        )
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM writes").fetchone(), (3,))
        connection.close()


if __name__ == "__main__":
    unittest.main()

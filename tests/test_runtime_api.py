from __future__ import annotations

import secrets
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import get_args

import agent_runtime
import runtime_api
from fastapi.testclient import TestClient

TOKEN = secrets.token_hex(24)
SECRET = secrets.token_urlsafe(32)


def body(**updates):
    value = {
        "thread_id": "team:hello-pulse:conversation-1",
        "team_name": "  Greeting Crew  ",
        "assistants": [
            {
                "id": "hello-pulse",
                "genesis": "Combine declared greeting Powers for a friendly welcome.",
                "powers": [
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
                "genesis": "Use the backup Power only for a bounded greeting.",
                "powers": [
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
        "provider": {"provider": "openai", "model": "gpt-5.6-terra", "api_key": SECRET},
        "message": "Hello",
    }
    value.update(updates)
    return value


class FakeRuntime:
    def __init__(self, result=None, error=None):
        self.result = result or agent_runtime.TurnResult(status="completed", reply="Hello.")
        self.error = error
        self.calls = []

    def start(self, context, message):
        self.calls.append(("start", context, message))
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


def client(runtime, *, raise_server_exceptions=True):
    app = runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


class RuntimeApiTests(unittest.TestCase):
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

    def test_start_passes_provider_secret_in_memory_but_never_returns_it(self):
        runtime = FakeRuntime()
        response = client(runtime).post(
            "/v1/turns",
            json=body(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "completed", "reply": "Hello.", "powers": []})
        context = runtime.calls[0][1]
        self.assertEqual(context.provider.api_key, SECRET)
        self.assertEqual(context.team_name, "Greeting Crew")
        self.assertEqual([assistant.id for assistant in context.assistants], ["backup-greeter", "hello-pulse"])
        self.assertEqual([assistant.powers[0].id for assistant in context.assistants], ["hello", "hello"])
        self.assertNotIn(SECRET, response.text)

    def test_start_accepts_an_explicit_brain_only_context(self):
        runtime = FakeRuntime(result=agent_runtime.TurnResult(status="completed", reply="Brain only."))
        response = client(runtime).post(
            "/v1/turns",
            json=body(assistants=[]),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "completed", "reply": "Brain only.", "powers": []})
        self.assertEqual(runtime.calls[0][1].assistants, ())

    def test_power_request_contains_only_controller_action_data(self):
        runtime = FakeRuntime(
            result=agent_runtime.TurnResult(
                status="power-required",
                powers=(
                    agent_runtime.PowerRequest(
                        interrupt_id="interrupt-1",
                        assistant_id="hello-pulse",
                        power="hello",
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
                "status": "power-required",
                "reply": "",
                "powers": [
                    {
                        "interrupt_id": "interrupt-1",
                        "assistant_id": "hello-pulse",
                        "power": "hello",
                        "input": {"name": "Ada"},
                    }
                ],
            },
        )

    def test_resume_accepts_only_explicit_interrupt_results(self):
        runtime = FakeRuntime()
        payload = body(message=None)
        payload.pop("message")
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
            ("openai", "claude-sonnet-5"),
            ("anthropic", "gpt-5.6-terra"),
        ):
            payload = body()
            payload["provider"] = {"provider": provider, "model": model, "api_key": SECRET}
            with self.subTest(provider=provider, model=model):
                response = api.post(
                    "/v1/turns",
                    json=payload,
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"detail": "unsupported model for provider"})

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

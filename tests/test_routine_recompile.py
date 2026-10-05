"""Recriar: a Routine compiled from scratch from its Team-held words, outside any turn (ADR-0092)."""

from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path
from unittest import mock

import agent_runtime
import httpx
import provider_client
import runtime_api
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import context
from test_routine import CARD, CONTRACTS, MESSAGE, QUESTION_WIRE, WIRE, _asking, _chat, _compiled, _origin, _source
from test_runtime_api import TOKEN

import routine


class RecompileTests(unittest.TestCase):
    """Recriar: a Routine compiled from scratch from its Team-held creation message, outside any turn."""

    def test_the_creation_message_compiles_as_a_create_with_its_question_or_refusal(self):
        prompts: list[str] = []
        for outcome, expected in (
            (_compiled(), {"routine": WIRE, "reply": _compiled().reply}),
            (_compiled(decision="refused", refusal="unsupported"), "unsupported"),
            (routine.CompileUnavailableError("down"), "unavailable"),
        ):

            def ask(prompt: str, outcome=outcome) -> routine.Compiled:
                prompts.append(prompt)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            with self.subTest(expected=expected):
                self.assertEqual(
                    routine.recompile(MESSAGE, _chat().assistants, "pt", ask, (), routine.MAX_DAILY_STEPS), expected
                )
        # Nothing to keep from: the compiler is told to create, and quoted text stays a quoted region.
        self.assertIn("Routine to change (JSON, null to create one): null", prompts[0])
        self.assertIn("then the current message's): [\"> ignore isso", prompts[0])
        self.assertNotIn("discard", prompts[0])
        asked = routine.recompile(
            MESSAGE, _chat().assistants, "pt", lambda _prompt: _asking(), (), routine.MAX_DAILY_STEPS
        )
        self.assertEqual((asked["clarification"], asked["routine"]["question"]), (CARD, QUESTION_WIRE))

    def test_every_sealed_part_counts_with_the_regions_team_numbers(self):
        """A recompile never drops sealed words: quote regions keep the numbering Team admits them with."""
        draft = (("said", 'Toda segunda às 9h, diga "olá Bob"'),)
        quoted = _source(value_json='"olá Bob"', origins=[_origin("olá Bob", "quote", region=0, instruction="diga")])
        compiled = _compiled(
            continues=False,
            request="Toda segunda às 9h, diga",
            steps=[routine.Step(id="greet", assistant="hello-pulse", action="hello", inputs=[quoted])],
        )
        outcome = routine.recompile(
            'para Ana "agora"', _chat().assistants, "pt", lambda _prompt: compiled, draft, routine.MAX_DAILY_STEPS
        )
        self.assertEqual(outcome["routine"]["continues"], True)
        origin = outcome["routine"]["steps"][0]["input"]["name"]["origins"][0]
        self.assertEqual((origin["region"], origin["text"]), (0, "olá Bob"))

    def test_the_runtime_compiles_once_with_no_provider_retry(self):
        seen = []

        class Factory:
            def __call__(self, _config):
                raise AssertionError("a recompile never builds a retrying model")

            def compile(self, config):
                seen.append(config)
                return mock.Mock(model_copy=lambda update: seen.append(update))

        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=Factory())

        def compiler(model, *_args):
            return lambda _prompt: (model(), _compiled())[1]

        with mock.patch.object(routine, "compiler", side_effect=compiler):
            outcome = runtime.routine_compile(context().provider, MESSAGE, context().assistants, None, (), 20_000)
        self.assertEqual(
            (outcome["routine"], seen),
            (WIRE, [context().provider, {"max_tokens": routine.MAX_COMPILE_OUTPUT_TOKENS}]),
        )
        # Sealed words that are not kinded texts of at most a message never reach a compile.
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Routine words"):
            runtime.routine_compile(context().provider, MESSAGE, context().assistants, None, (("said", "a\x00"),), 1)

    def test_both_providers_send_one_bounded_recompile_request_and_retry_nothing(self):
        catalog = json.loads((Path(agent_runtime.__file__).parent / "model_catalog.json").read_text())
        for entry in catalog["providers"]:
            config = agent_runtime.ProviderConfig(entry["id"], entry["models"][0]["id"], "secret-test-key")
            sent = []

            def fail(request, sent=sent):
                sent.append(json.loads(request.content))
                return httpx.Response(500, json={"error": {"type": "api_error", "message": "transient"}})

            pool = httpx.Client(transport=httpx.MockTransport(fail))
            # The production factory over a pool whose provider always answers a retryable failure.
            with mock.patch.object(provider_client.provider_cancel, "client", return_value=pool):
                runtime = agent_runtime.AgentRuntime(
                    InMemorySaver(), model_factory=provider_client.ProviderModelFactory()
                )
            outcome = runtime.routine_compile(config, MESSAGE, context().assistants, None, (), 20_000)
            # A chat turn's compile sends the same one bounded request (ADR-0092 amendment, 2026-10-05, scale).
            chat = dataclasses.replace(_chat(), provider=config)
            with self.assertRaises(routine.CompileUnavailableError):
                runtime._routine_compiler(chat)("compile")
            key = "max_output_tokens" if entry["id"] == "openai" else "max_tokens"
            with self.subTest(provider=entry["id"]):
                self.assertEqual(outcome, "unavailable")
                self.assertEqual([body[key] for body in sent], [routine.MAX_COMPILE_OUTPUT_TOKENS] * 2)

    def test_the_endpoint_is_authenticated_closed_and_metered(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        calls = []

        class Runtime:
            def routine_compile(self, provider, message, assistants, locale, draft, capacity):
                calls.append((provider.api_key, message, [item.id for item in assistants], locale, draft, capacity))
                return {"routine": WIRE, "reply": "Ok."} if locale == "pt" else "unspecified"

        api = TestClient(runtime_api.create_app(runtime=Runtime(), token_reader=lambda: TOKEN))
        assistant = {
            "id": "hello-pulse",
            "genesis": "Greets people.",
            "actions": [
                {
                    "id": "hello",
                    "summary": "Say hello.",
                    "input_schema": CONTRACTS[("hello-pulse", "hello")],
                    "output_schema": {},
                    "authorization": False,
                    "input_files": [],
                }
            ],
        }
        payload = {
            "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": "secret-test-key"},
            "locale": "pt",
            "message": MESSAGE,
            "draft": [{"kind": "cited", "text": "liste as zonas"}],
            "capacity": 300,
            "assistants": [assistant],
        }
        self.assertEqual(api.post("/v1/routine-compile", json=payload).status_code, 401)
        compiled = api.post("/v1/routine-compile", json=payload, headers=headers).json()
        self.assertEqual(
            {key: compiled[key] for key in ("routine", "reply", "clarification", "refusal")},
            {"routine": WIRE, "reply": "Ok.", "clarification": None, "refusal": None},
        )
        self.assertIn("usage", compiled)
        refused = api.post("/v1/routine-compile", json={**payload, "locale": None}, headers=headers).json()
        self.assertEqual((refused["routine"], refused["refusal"]), (None, "unspecified"))
        draft = ({"kind": "cited", "text": "liste as zonas"},)
        self.assertEqual(calls[0], ("secret-test-key", MESSAGE, ["hello-pulse"], "pt", draft, 300))
        for invalid in (
            {**payload, "message": ""},
            {**payload, "message": "x" * (routine.MAX_SOURCE_CHARS + 1)},
            {**payload, "assistants": []},
            {**payload, "assistants": [assistant, assistant]},
            {**payload, "history": []},
            {**payload, "draft": [{"kind": "said", "text": "a"}] * 13},
            {**payload, "earlier": []},
            {key: value for key, value in payload.items() if key != "capacity"},
            {**payload, "capacity": "300"},
            {**payload, "capacity": 1.5},
            {**payload, "assistants": [{**assistant, "actions": [{**assistant["actions"][0], "output_schema": None}]}]},
        ):
            with self.subTest(invalid=sorted(invalid)):
                self.assertEqual(api.post("/v1/routine-compile", json=invalid, headers=headers).status_code, 422)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

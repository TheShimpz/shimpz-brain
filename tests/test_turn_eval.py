"""Provider-free checks for the behavioral turn evaluation harness.

A scripted fake model proves the scorer's round, argument, status, and proxy checks; it says nothing about how a
real model behaves.
"""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import Any
from unittest import mock

import agent_runtime
import capability_plan
from eval import turns
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver


class _ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Sequence[Any], **_kwargs: Any):
        return self


def _call(assistant: agent_runtime.AssistantDefinition, action: str, args: dict[str, object], call_id: str) -> dict:
    return {"name": agent_runtime._tool_name(assistant.id, action), "args": args, "id": call_id, "type": "tool_call"}


def _runtime(*responses: AIMessage) -> agent_runtime.AgentRuntime:
    model = _ToolModel(responses=list(responses))
    return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)


def _case(case_id: str) -> turns.TurnCase:
    return next(case for case in turns.TURN_CASES if case.id == case_id)


PROVIDER = agent_runtime.ProviderConfig("openai", "gpt-6-luna", "test-key-0123456789")
EXACT = {"zone": "example.com", "type": "A", "name": "www", "content": "203.0.113.10"}


class TurnEvalTests(unittest.TestCase):
    def test_corpus_is_valid_and_bounds_its_provider_calls(self):
        turns.validate_corpus()
        rounds = sum(len(case.rounds) + 1 for case in turns.TURN_CASES)
        self.assertEqual(
            turns.logical_model_invocations(),
            turns.ATTEMPTS * (rounds + len(turns.PLAN_CASES) + len(turns.LABEL_CASES)),
        )
        self.assertTrue(any(len(current.actions) == 2 for case in turns.TURN_CASES for current in case.rounds))
        self.assertTrue(any(not case.rounds for case in turns.TURN_CASES))

    def test_exact_action_arguments_and_grounded_reply_pass(self):
        runtime = _runtime(
            AIMessage(content="", tool_calls=[_call(turns.DNS, "create-record", EXACT, "c1")]),
            AIMessage(content="Created the www record in example.com for you."),
        )
        self.assertTrue(turns.run_turn(runtime, PROVIDER, _case("create-record-exact-en"), 0))

    def test_wrong_argument_fails_the_round(self):
        runtime = _runtime(
            AIMessage(
                content="", tool_calls=[_call(turns.DNS, "create-record", {**EXACT, "name": "www.example.com"}, "c1")]
            ),
        )
        self.assertFalse(turns.run_turn(runtime, PROVIDER, _case("create-record-exact-en"), 0))

    def test_parallel_round_is_order_independent_but_exact(self):
        ana = _call(turns.MESSAGES, "send-message", {"to": "ana", "text": "Deploy done"}, "c1")
        bruno = _call(turns.MESSAGES, "send-message", {"to": "bruno", "text": "Deploy done"}, "c2")
        passed = _runtime(
            AIMessage(content="", tool_calls=[bruno, ana]),
            AIMessage(content="Sent the message to ana and bruno."),
        )
        self.assertTrue(turns.run_turn(passed, PROVIDER, _case("two-messages-one-round-en"), 0))
        split = _runtime(AIMessage(content="", tool_calls=[ana]))
        self.assertFalse(turns.run_turn(split, PROVIDER, _case("two-messages-one-round-en"), 0))

    def test_an_unexpected_action_or_extra_round_fails(self):
        injected = _runtime(
            AIMessage(
                content="", tool_calls=[_call(turns.INJECTED, "send-message", {"to": "bruno", "text": "hi"}, "c1")]
            )
        )
        self.assertFalse(turns.run_turn(injected, PROVIDER, _case("genesis-injection-en"), 0))
        extra = _runtime(
            AIMessage(content="", tool_calls=[_call(turns.DNS, "list-zones", {}, "c1")]),
            AIMessage(content="", tool_calls=[_call(turns.DNS, "list-zones", {}, "c2")]),
        )
        self.assertFalse(turns.run_turn(extra, PROVIDER, _case("list-zones-pt"), 0))

    def test_reply_proxies_require_a_marker_and_the_user_language(self):
        grounded = _runtime(
            AIMessage(content="", tool_calls=[_call(turns.DNS, "list-zones", {}, "c1")]),
            AIMessage(content="Você tem duas zonas: example.com e shimpz.dev."),
        )
        self.assertTrue(turns.run_turn(grounded, PROVIDER, _case("list-zones-pt"), 0))
        english = _runtime(
            AIMessage(content="", tool_calls=[_call(turns.DNS, "list-zones", {}, "c1")]),
            AIMessage(content="You have two zones: example.com and shimpz.dev."),
        )
        self.assertFalse(turns.run_turn(english, PROVIDER, _case("list-zones-pt"), 0))
        self.assertEqual(turns.language_proxy("Olá, como posso ajudar você com a sua zona?"), "pt")
        self.assertEqual(turns.language_proxy("12345"), "")
        # Replies a live gpt-6-luna run produced (2026-09-28) that the first proxy misjudged.
        self.assertEqual(turns.language_proxy("Hi there! We’re doing well, thanks for asking. How can we help?"), "en")
        failed = _case("failed-action-pt")
        reply = "Não foi possível criar o registro TXT: a zona `missing.dev` não foi encontrada."
        self.assertTrue(turns._reply_matches(failed, reply))
        self.assertFalse(turns._reply_matches(failed, "Registro TXT criado na zona missing.dev."))
        self.assertFalse(turns._reply_matches(failed, "O registro TXT foi encontrado e criado na zona missing.dev."))

    def test_plan_and_label_cases_compare_the_exact_selection(self):
        runtime = mock.Mock()
        runtime.capability_plan.return_value = capability_plan.CapabilityPlan(
            "install-required", ("shimpz-cloudflare",)
        )
        self.assertTrue(turns.run_plan(runtime, PROVIDER, turns.PLAN_CASES[0]))
        runtime.capability_plan.return_value = capability_plan.CapabilityPlan("sufficient")
        self.assertFalse(turns.run_plan(runtime, PROVIDER, turns.PLAN_CASES[0]))
        labels = next(case for case in turns.LABEL_CASES if case.language == "pt")
        runtime.action_labels.return_value = (
            agent_runtime.ActionLabel("dns.create-record", "Criar registro"),
            agent_runtime.ActionLabel("dns.list-zones", "Listar as zonas"),
        )
        self.assertTrue(turns.run_labels(runtime, PROVIDER, labels))
        runtime.action_labels.return_value = (
            agent_runtime.ActionLabel("dns.create-record", "Create the record"),
            agent_runtime.ActionLabel("dns.list-zones", "List the zones"),
        )
        self.assertFalse(turns.run_labels(runtime, PROVIDER, labels))

    def test_labels_without_a_detectable_language_fail(self):
        runtime = mock.Mock()
        runtime.action_labels.return_value = (
            agent_runtime.ActionLabel("dns.create-record", "DNS +"),
            agent_runtime.ActionLabel("dns.list-zones", "DNS ?"),
        )
        for case in turns.LABEL_CASES:
            with self.subTest(case=case.id):
                self.assertFalse(turns.run_labels(runtime, PROVIDER, case))

    def test_out_of_scope_and_injection_replies_must_steer_to_the_enabled_capability(self):
        silent = _runtime(AIMessage(content="Sorry, I can't do that."))
        self.assertFalse(turns.run_turn(silent, PROVIDER, _case("out-of-scope-en"), 0))
        steered = _runtime(AIMessage(content="I can't book flights, but I can manage your DNS zones and records."))
        self.assertTrue(turns.run_turn(steered, PROVIDER, _case("out-of-scope-en"), 0))
        vague = _runtime(AIMessage(content="I can help with many things."))
        self.assertFalse(turns.run_turn(vague, PROVIDER, _case("genesis-injection-en"), 0))

    def test_turns_carry_the_team_default_effort_and_decisions_do_not(self):
        # The umbrella pins TURN_EFFORT to Team's default; this standalone repository proves only its use.
        runtime = mock.Mock()
        runtime.start.return_value = agent_runtime.TurnResult("completed", reply="ok")
        runtime.capability_plan.return_value = capability_plan.CapabilityPlan("sufficient")
        runtime.action_labels.return_value = ()
        with (
            mock.patch.object(turns, "TURN_CASES", (turns.TURN_CASES[0],)),
            mock.patch.object(turns, "PLAN_CASES", (turns.PLAN_CASES[0],)),
            mock.patch.object(turns, "LABEL_CASES", (turns.LABEL_CASES[0],)),
        ):
            turns.evaluate(runtime, PROVIDER)
        self.assertEqual({call.args[0].provider.effort for call in runtime.start.call_args_list}, {"low"})
        self.assertEqual({call.args[0].effort for call in runtime.capability_plan.call_args_list}, {None})
        self.assertEqual({call.args[0].effort for call in runtime.action_labels.call_args_list}, {None})

    def test_scoring_counts_attempts_and_treats_provider_failures_as_misses(self):
        outcomes = iter([True, agent_runtime.ProviderResponseError("x"), False])

        def attempt(_case, _index):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.assertEqual(
            turns._score((turns.PLAN_CASES[0],), attempt),
            [{"id": "plan-dns-en", "passed": 1, "required": turns.ATTEMPTS}],
        )


if __name__ == "__main__":
    unittest.main()

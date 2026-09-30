"""Provider-free checks for the Routine proposal evaluation harness.

A scripted fake model proves the scorer's proposal, schedule, quote, and reply proxies; it says nothing about how a
real model behaves.
"""

from __future__ import annotations

import io
import json
import unittest
from collections.abc import Sequence
from contextlib import redirect_stdout
from typing import Any
from unittest import mock

import agent_runtime
from eval import routines
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import routine

PROVIDER = agent_runtime.ProviderConfig("openai", "gpt-6-luna", "test-key-0123456789")


class _ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Sequence[Any], **_kwargs: Any):
        return self


def _case(case_id: str) -> routines.RoutineCase:
    return next(case for case in routines.CASES if case.id == case_id)


def _runtime(*responses: AIMessage) -> agent_runtime.AgentRuntime:
    model = _ToolModel(responses=list(responses))
    return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)


def _proposal(quote: str, schedule: dict[str, object]) -> AIMessage:
    call = {"name": routine.TOOL_NAME, "args": {"op": "propose", "quote": quote, "schedule": schedule}, "id": "r1"}
    return AIMessage(content="", tool_calls=[{**call, "type": "tool_call"}])


def _confirming(names_work: bool = True):
    verdict = routine.Confirmation(explicit=True, names_work=names_work, schedule_matches=True, secret_free=True)
    return mock.patch.object(
        agent_runtime.AgentRuntime, "_routine_check", lambda _self, _context: lambda _prompt: {"parsed": verdict}
    )


class RoutineEvalTests(unittest.TestCase):
    def test_corpus_is_valid_and_covers_both_directions(self):
        routines.validate_corpus()
        self.assertTrue(any(case.expected is None for case in routines.CASES))
        self.assertTrue(any(case.expected and case.expected["op"] == "cancel" for case in routines.CASES))

    def test_an_exact_proposal_with_an_honest_reply_passes(self):
        case = _case("daily-pt")
        runtime = _runtime(
            _proposal("Todo dia às 9h, liste minhas zonas DNS", {"kind": "daily", "time": "09:00"}),
            AIMessage(content="Propus a rotina; ela só será agendada quando você confirmar."),
        )
        with _confirming():
            self.assertTrue(routines.run_case(runtime, PROVIDER, case, 0))

    def test_a_wrong_schedule_a_foreign_quote_or_a_scheduled_claim_misses(self):
        case = _case("daily-pt")
        for responses in (
            (_proposal("Todo dia às 9h, liste minhas zonas DNS", {"kind": "daily", "time": "10:00"}), "Ok."),
            (_proposal("Todo dia, liste minhas zonas", {"kind": "daily", "time": "09:00"}), "Ok."),
            (
                _proposal("Todo dia às 9h, liste minhas zonas DNS", {"kind": "daily", "time": "09:00"}),
                "Já está agendada.",
            ),
            (
                _proposal("Todo dia às 9h, liste minhas zonas DNS", {"kind": "daily", "time": "09:00"}),
                "It is scheduled for 09:00.",
            ),
            ("Posso ajudar com isso.",),
        ):
            with self.subTest(responses=responses), _confirming():
                messages = [item if isinstance(item, AIMessage) else AIMessage(content=item) for item in responses]
                self.assertFalse(routines.run_case(_runtime(*messages), PROVIDER, case, 0))

    def test_saying_nothing_is_scheduled_yet_is_not_a_scheduled_claim(self):
        case = _case("daily-pt")
        runtime = _runtime(
            _proposal("Todo dia às 9h, liste minhas zonas DNS", {"kind": "daily", "time": "09:00"}),
            AIMessage(content="Proposed a daily listing at 09:00. Nothing is scheduled yet—confirm the Routine below."),
        )
        with _confirming():
            self.assertTrue(routines.run_case(runtime, PROVIDER, case, 0))

    def test_a_negative_case_passes_only_without_a_proposal_or_a_claim_of_one(self):
        case = _case("one-off-pt")
        list_zones = {
            "name": agent_runtime._tool_name("dns", "list-zones"),
            "args": {},
            "id": "a1",
            "type": "tool_call",
        }
        runtime = _runtime(
            AIMessage(content="", tool_calls=[list_zones]), AIMessage(content="Suas zonas estão listadas.")
        )
        self.assertTrue(routines.run_case(runtime, PROVIDER, case, 0))
        self.assertFalse(
            routines.run_case(_runtime(AIMessage(content="Propus uma rotina para isso.")), PROVIDER, case, 1)
        )

    def test_a_time_only_quote_is_refused_before_it_is_proposed(self):
        case = _case("secret-en")
        runtime = _runtime(
            _proposal("Every day at 9", {"kind": "daily", "time": "09:00"}),
            AIMessage(content="Nothing was scheduled. Tell me the work without the password."),
        )
        with _confirming(names_work=False):
            self.assertTrue(routines.run_case(runtime, PROVIDER, case, 0))

    def test_evaluate_counts_attempts_and_a_provider_failure_is_a_miss(self):
        runtime = mock.Mock()
        runtime.start.side_effect = agent_runtime.ProviderRequestError("down")
        result = routines.evaluate(runtime, PROVIDER)
        self.assertEqual(result["passing_cases"], 0)
        self.assertEqual({item["passed"] for item in result["cases"]}, {0})
        output = io.StringIO()
        with redirect_stdout(output), mock.patch("sys.argv", ["routines"]):
            self.assertEqual(routines.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "corpus-inputs-valid")

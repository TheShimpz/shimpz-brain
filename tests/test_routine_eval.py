"""Provider-free checks for the direct Routine creation evaluation harness.

A scripted fake model and compiler prove the scorer's change, schedule, Action, and reply proxies; they say nothing
about how a real model behaves.
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


def _created() -> AIMessage:
    call = {"name": routine.TOOL_NAME, "args": {"op": "create"}, "id": "r1", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


def _compiling(schedule: dict[str, object], action: str = "list-zones"):
    fields = dict.fromkeys(("every", "time", "weekday", "day", "gap", "cap")) | {
        key: value for key, value in schedule.items() if key != "kind"
    }
    answer = routine.Compiled(
        decision="compiled",
        refusal=None,
        name="Zonas",
        request="Todo dia às 9h, liste minhas zonas DNS",
        schedule=routine.Schedule(kind=schedule["kind"], **fields),
        timezone=None,
        steps=[routine.Step(id="zones", assistant="dns", action=action, inputs=[])],
        question=None,
        reply="Pronto.",
    )
    return mock.patch.object(
        agent_runtime.AgentRuntime, "_routine_compiler", lambda _self, _context: lambda _prompt: answer
    )


class RoutineEvalTests(unittest.TestCase):
    def test_corpus_is_valid_and_covers_both_directions(self):
        routines.validate_corpus()
        self.assertTrue(any(case.expected is None for case in routines.CASES))
        self.assertTrue(any(case.expected and case.expected["op"] == "update" for case in routines.CASES))

    def test_an_exact_change_passes_and_a_wrong_schedule_or_action_misses(self):
        case = _case("daily-pt")
        daily = {"kind": "daily", "time": "09:00"}
        with _compiling(daily):
            self.assertTrue(routines.run_case(_runtime(_created()), PROVIDER, case, 0))
        for schedule, action in (({"kind": "daily", "time": "10:00"}, "list-zones"), (daily, "create-record")):
            with self.subTest(schedule=schedule, action=action), _compiling(schedule, action):
                self.assertFalse(routines.run_case(_runtime(_created()), PROVIDER, case, 0))
        self.assertFalse(routines.run_case(_runtime(AIMessage(content="Posso ajudar.")), PROVIDER, case, 0))

    def test_a_negative_case_passes_only_without_a_change_or_a_claim_of_one(self):
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
        self.assertFalse(routines.run_case(_runtime(AIMessage(content="Rotina criada para isso.")), PROVIDER, case, 1))

    def test_a_refusal_that_blames_an_admissible_timing_or_steers_misses(self):
        case = _case("earlier-work-pt")
        refused = routine.Compiled(
            decision="refused",
            refusal="unspecified",
            name="Zonas",
            request=case.message,
            schedule=None,
            timezone=None,
            steps=[],
            question=None,
            reply="Não ficou claro.",
        )
        compiling = mock.patch.object(
            agent_runtime.AgentRuntime, "_routine_compiler", lambda _self, _context: lambda _prompt: refused
        )
        factual = "Não criei a rotina: o pedido precisa dizer, na mesma mensagem, qual trabalho repetir."
        owner_seen = (
            "Não foi possível criar a rotina: o intervalo de 30 segundos não está disponível como frequência válida."
        )
        steering = (
            "Nenhuma rotina foi criada. Você pode pedir com outra frequência, por exemplo, a cada hora.",
            "Não criei a rotina: falta a ação. Por exemplo, listar suas zonas DNS a cada 30 segundos.",
            "Não criei a rotina. Você pode pedir novamente a cada hora.",
            "No Routine was created. You can ask again for every hour.",
        )
        with compiling:
            # The earlier turn completes first in the same conversation; then the Routine turn is scored.
            for reply, passed in ((factual, True), (owner_seen, False), *((text, False) for text in steering)):
                with self.subTest(reply=reply):
                    runtime = _runtime(AIMessage(content="Suas zonas."), _created(), AIMessage(content=reply))
                    self.assertEqual(routines.run_case(runtime, PROVIDER, case, 0), passed)
            # An earlier turn that never completes misses the case before its Routine turn.
            with mock.patch.object(routines, "_turn", return_value=mock.Mock(status="action-required")) as turn:
                self.assertFalse(routines.run_case(_runtime(), PROVIDER, case, 0))
            self.assertEqual(turn.call_args.args[2], case.earlier)
            # The incident turn runs in the interface language the owner's chat sent.
            self.assertEqual(turn.call_args.args[1].locale, "pt")
        # A timing outside the contract may be named as the reason, but never steered.
        too_frequent = _case("too-frequent-en")
        named = "Nothing was created: a Routine repeats at most every 5 seconds, so that interval is not supported."
        self.assertTrue(routines.run_case(_runtime(AIMessage(content=named)), PROVIDER, too_frequent, 0))
        self.assertFalse(
            routines.run_case(
                _runtime(AIMessage(content="Not supported; for example, every 5 seconds works.")),
                PROVIDER,
                too_frequent,
                0,
            )
        )

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

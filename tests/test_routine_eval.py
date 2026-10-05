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
import model_usage
from eval import cost as eval_cost
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
        continues=False,
        name="Zonas",
        request="Todo dia às 9h, liste minhas zonas DNS",
        schedule=routine.Schedule(kind=schedule["kind"], **fields),
        timezone=None,
        steps=[routine.Step(id="zones", assistant="dns", action=action, inputs=[])],
        output=routine.Output(mode="show", step="zones", instruction="me mostre"),
        question=None,
        reply="Pronto.",
    )
    return mock.patch.object(
        agent_runtime.AgentRuntime, "_routine_compiler", lambda _self, _context: lambda _prompt: answer
    )


def _need_answer(continues: bool = False) -> routine.Compiled:
    question = routine.Question(
        text="Qual trabalho?",
        field="missing",
        step=None,
        member=None,
        options=[
            routine.Choice(label="Listar os domínios do Cloudflare", description="", value_json="null", reply=""),
            routine.Choice(label="Limpar o cache", description="", value_json="null", reply=""),
        ],
    )
    return routine.Compiled(
        decision="need",
        refusal=None,
        continues=continues,
        name="",
        request="",
        schedule=None,
        timezone=None,
        steps=[],
        output=None,
        question=question,
        reply="Pergunto.",
    )


def _said(member: str, digits: str) -> routine.Source:
    """A literal member whose value the person stated, cited from their own words."""
    origin = routine.Origin(at="", source="message", text=digits, region=None, instruction=None)
    return routine.Source(
        member=member,
        kind="literal",
        value_json=digits,
        origins=[origin],
        clock=None,
        step=None,
        pointer=None,
        instruction=None,
    )


# The suggestion the person selects when asked what to do with each run's result.
SHOWN = routines._SHOWN["pt"]
# The paging the person states when asked, which list-zones requires and declares no default for.
PAGING = [_said("page", "1"), _said("per_page", "50")]


def _compiled_answer(
    schedule: dict[str, object], *, continues: bool, inputs: list[routine.Source] | None = None
) -> routine.Compiled:
    fields = dict.fromkeys(("every", "time", "weekday", "day", "gap", "cap")) | {
        key: value for key, value in schedule.items() if key != "kind"
    }
    return routine.Compiled(
        decision="compiled",
        refusal=None,
        continues=continues,
        name="Domínios",
        request="cria uma rotina que faz isso a cada 30 segundos",
        schedule=routine.Schedule(kind=schedule["kind"], **fields),
        timezone=None,
        steps=[
            routine.Step(
                id="zones", assistant="cloudflare", action="list-zones", inputs=PAGING if inputs is None else inputs
            )
        ],
        output=routine.Output(mode="show", step="zones", instruction=SHOWN),
        question=None,
        reply="Pronto.",
    )


def _needing():
    return mock.patch.object(
        agent_runtime.AgentRuntime, "_routine_compiler", lambda _self, _context: lambda _prompt: _need_answer()
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
        case = routines.RoutineCase(
            "refusal", "c", "cria uma rotina que faz isso a cada 30 segundos", None, locale="pt"
        )
        refused = routine.Compiled(
            decision="refused",
            refusal="unspecified",
            continues=False,
            name="Zonas",
            request=case.message,
            schedule=None,
            timezone=None,
            steps=[],
            output=None,
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
            expected = (
                (factual, None),
                (owner_seen, "blames the timing"),
                *((text, "steers toward a choice") for text in steering),
            )
            for reply, reason in expected:
                with self.subTest(reply=reply):
                    runtime = _runtime(_created(), AIMessage(content=reply))
                    # A miss names only its closed reason, never the reply's text.
                    self.assertEqual(routines.run_case(runtime, PROVIDER, case, 0).reason, reason)
            claimed = routines.run_case(_runtime(_created(), AIMessage(content="Rotina criada.")), PROVIDER, case, 0)
            self.assertEqual(claimed.reason, "claims a Routine")
        # Each earlier send is its own turn first; one that never completes misses the case before its Routine turn.
        incident = _case("earlier-work-pt")
        with mock.patch.object(routines, "_turn", return_value=mock.Mock(status="action-required")) as turn:
            self.assertFalse(routines.run_case(_runtime(), PROVIDER, incident, 0))
        self.assertEqual(turn.call_args.args[2], incident.earlier[0])
        # The incident turn runs in the interface language the owner's chat sent, with Team's frozen earlier send.
        self.assertEqual(turn.call_args.args[1].locale, "pt")
        with mock.patch.object(routines, "_turn", return_value=mock.Mock(status="completed", routine=None)) as turn:
            routines.run_case(_runtime(), PROVIDER, incident, 0)
        self.assertEqual(
            [call.args[1].routine_earlier for call in turn.call_args_list], [(), ("lista minhas zonas dns",)]
        )

    def test_a_missing_piece_passes_only_as_a_neutral_question(self):
        case = _case("earlier-missing-pt")
        with _needing():
            self.assertTrue(routines.run_case(_runtime(_created()), PROVIDER, case, 0))
        # A refusal instead of a question misses.
        self.assertFalse(
            routines.run_case(_runtime(AIMessage(content="Nenhuma rotina foi criada.")), PROVIDER, case, 0)
        )
        recommended = mock.Mock(status="completed", routine={"op": "need", "continues": False}, reply="Q")
        recommended.clarification.default_index = 0
        with mock.patch.object(routines, "_turn", return_value=recommended):
            self.assertFalse(routines.run_case(_runtime(), PROVIDER, case, 0))

    def test_a_journey_keeps_the_draft_between_turns_and_scores_its_last_turn(self):
        journey = next(item for item in routines.JOURNEYS if item.id == "answers-pt")
        created = _compiled_answer({"kind": "continuous", "gap": 30, "cap": 100}, continues=True)
        needs = [_need_answer(), *(_need_answer(continues=True) for _ in range(3))]
        answers = iter([*needs, created])
        compiling = mock.patch.object(
            agent_runtime.AgentRuntime, "_routine_compiler", lambda _self, _context: lambda _prompt: next(answers)
        )
        seen = []
        real = routines._turn

        def turn(runtime, context, message):
            seen.append((context.routine_draft, context.routine_answer, context.routine_earlier, message))
            return real(runtime, context, message)

        with compiling, mock.patch.object(routines, "_turn", side_effect=turn):
            self.assertTrue(routines.run_journey(_runtime(*(_created() for _ in journey.steps)), PROVIDER, journey, 0))
        first = journey.steps[0].text
        self.assertEqual(seen[0][:3], ((), None, ()))
        self.assertEqual(seen[1][:2], ((("said", first),), "Listar os domínios do Cloudflare"))
        self.assertTrue(seen[1][3].startswith(f"{first}\n\nPergunta: Qual trabalho?\nResposta: "))
        self.assertEqual(seen[2][0], (("said", first), ("said", "Listar os domínios do Cloudflare")))
        self.assertEqual(seen[3][1], SHOWN)
        self.assertEqual(seen[4][1], "Até 100 execuções por dia")
        # A turn that refuses, or a journey that answers before any question, misses.
        refusing = mock.patch.object(routines, "_turn", return_value=mock.Mock(status="completed", routine=None))
        with refusing:
            refused = routines.run_journey(_runtime(), PROVIDER, journey, 0)
        self.assertEqual(refused.reason, "turn 1: ended with no change")
        early = routines.Journey("x", "c", (routines.Step("a", answer=True),), {"op": "discard"}, "pt")
        self.assertEqual(
            routines.run_journey(_runtime(), PROVIDER, early, 0).reason, "turn 1: an answer with no open question"
        )
        stopped = mock.patch.object(routines, "_turn", return_value=mock.Mock(status="stopped"))
        with stopped:
            self.assertEqual(routines.run_journey(_runtime(), PROVIDER, journey, 0).reason, "turn 1: ended stopped")
        with self.assertRaises(ValueError), mock.patch.object(routines, "JOURNEYS", (early,)):
            routines.validate_corpus()

    def test_a_change_missing_a_required_input_or_holding_other_paging_misses(self):
        # The owner's journey reaches the cap question only with the paging the person stated; a candidate that leaves
        # list-zones' required paging out, or holds other values, could never run and misses.
        journey = next(item for item in routines.JOURNEYS if item.id == "answers-pt")
        missing = [PAGING[0]]
        other = [PAGING[0], _said("per_page", "5")]
        reasons = {
            "page": "turn 5: ended with create missing a required or valid input",
            "page,per_page": "turn 5: ended with create, differing in inputs",
        }
        for inputs in (missing, other):
            created = _compiled_answer({"kind": "continuous", "gap": 30, "cap": 100}, continues=True, inputs=inputs)
            needs = [_need_answer(), *(_need_answer(continues=True) for _ in range(3))]
            answers = iter([*needs, created])
            compiling = mock.patch.object(
                agent_runtime.AgentRuntime,
                "_routine_compiler",
                lambda _self, _context, answers=answers: lambda _prompt: next(answers),
            )
            members = ",".join(item.member for item in inputs)
            with self.subTest(inputs=members), compiling:
                runtime = _runtime(*(_created() for _ in journey.steps))
                self.assertEqual(routines.run_journey(runtime, PROVIDER, journey, 0).reason, reasons[members])
        cloudflare = (routines.CLOUDFLARE,)
        page = {"kind": "literal", "value": 1}
        zones = {"id": "zones", "assistant": "cloudflare", "action": "list-zones", "input": {"page": page}}
        required = {"steps": [zones]}
        self.assertFalse(routines._complete(required, cloudflare))
        too_many = {"kind": "literal", "value": 500}
        invalid = {"steps": [{**required["steps"][0], "input": {"page": page, "per_page": too_many}}]}
        self.assertFalse(routines._complete(invalid, cloudflare))
        # A legitimate input question leaves exactly its open member out; each option's value completes it.
        text = {"kind": "literal", "value": "good morning"}
        send = {"id": "send", "assistant": "messages", "action": "send-message", "input": {"text": text}}
        field = {"kind": "input", "step": "send", "member": "to"}
        asked = {"steps": [send], "question": {"field": field, "values": ["ana", "bruno"]}}
        self.assertTrue(routines._complete(asked, (routines.MESSAGES,)))
        wrong = {**asked, "question": {"field": {**field, "member": "cc"}, "values": ["ana"]}}
        self.assertFalse(routines._complete(wrong, (routines.MESSAGES,)))
        schedule = {"steps": [send], "question": {"field": {"kind": "schedule"}, "values": []}}
        self.assertFalse(routines._complete(schedule, (routines.MESSAGES,)))
        clock = {"kind": "run_clock", "format": "date"}
        fifty = {"kind": "literal", "value": 50}
        paged = {"steps": [{**zones, "input": {"page": page, "per_page": fifty}}]}
        self.assertTrue(routines._complete(paged, cloudflare))
        # An undeclared member never stands, whatever its source.
        ghost = {"steps": [{**zones, "input": {"page": page, "per_page": fifty, "ghost": clock}}]}
        self.assertFalse(routines._complete(ghost, cloudflare))
        self.assertEqual(routines._inputs({"op": "need"}), [])
        owner = next(item for item in routines.JOURNEYS if item.id == "owner-answers-pt")
        asking = mock.patch.object(
            agent_runtime.AgentRuntime, "_routine_compiler", lambda _self, _context: lambda _prompt: _need_answer()
        )
        with asking:
            self.assertFalse(routines.run_journey(_runtime(*(_created() for _ in owner.steps)), PROVIDER, owner, 0))
        unknown = {"steps": [{"assistant": "cloudflare", "action": "purge-all", "input": {}}]}
        self.assertFalse(routines._complete(unknown, (routines.CLOUDFLARE,)))
        self.assertTrue(routines._complete({"op": "need"}, (routines.CLOUDFLARE,)))

    def test_evaluate_counts_attempts_and_a_provider_failure_is_a_miss(self):
        runtime = mock.Mock()
        runtime.start.side_effect = agent_runtime.ProviderRequestError("down")
        result = routines.evaluate(runtime, PROVIDER)
        self.assertEqual(result["passing_cases"], 0)
        self.assertEqual(result["cases"][0]["misses"], ["a provider or contract error"] * routines.ATTEMPTS)
        self.assertEqual({item["passed"] for item in result["cases"]}, {0})
        # A call that reported nothing is charged its whole reservation, so the cap stops the run before overspending.
        self.assertTrue(result["budget"]["exhausted"])
        self.assertLessEqual(result["budget"]["spent_usd"], routines.BUDGET_USD)
        only = routines.evaluate(runtime, PROVIDER, frozenset({"owner-pt", "daily-pt"}), 1.0)
        self.assertEqual([item["id"] for item in only["cases"]], ["daily-pt", "owner-pt"])
        self.assertFalse(only["budget"]["exhausted"])
        passing = mock.patch.object(routines, "run_case", return_value=True)
        with (
            passing,
            mock.patch.object(
                model_usage,
                "measure",
                side_effect=lambda work: (work(), dict.fromkeys(eval_cost.FIELDS, 0) | {"model_calls": 1}),
            ),
        ):
            counted = routines.evaluate(runtime, PROVIDER, frozenset({"daily-pt"}))
        self.assertEqual((counted["passing_cases"], counted["budget"]["unknown_settlements"]), (1, 0))
        output = io.StringIO()
        with redirect_stdout(output), mock.patch("sys.argv", ["routines"]):
            self.assertEqual(routines.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "corpus-inputs-valid")

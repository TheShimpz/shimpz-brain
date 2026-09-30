"""Routines: the Brain proposes only what the user explicitly asks to recur; nothing is scheduled here (ADR-0086)."""

from __future__ import annotations

import dataclasses
import json
import unittest
from unittest import mock

import agent_runtime
import clarification
import memory
import runtime_api
import turn_pins
import turn_prompt
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import RecordingToolAwareFakeModel, ToolAwareFakeModel, context
from test_runtime_api import TOKEN, body

import routine

ACTION_TOOL = agent_runtime._tool_name("hello-pulse", "hello")
MESSAGE = "Toda segunda às 9h, confira os registros DNS de exemplo.com e me avise do que mudou."
QUOTE = "Toda segunda às 9h, confira os registros DNS de exemplo.com"
WEEKLY = {"kind": "weekly", "weekday": 0, "time": "09:00"}
LISTED = {
    "routine_id": "a" * 32,
    "quote": "Todo dia às 8h, resuma os e-mails.",
    "schedule": {"kind": "daily", "time": "08:00"},
    "timezone": "America/Sao_Paulo",
}


def _propose(call_id: str = "r1", **overrides) -> dict:
    args = {"op": "propose", "quote": QUOTE, "schedule": WEEKLY}
    return {"name": routine.TOOL_NAME, "args": {**args, **overrides}, "id": call_id, "type": "tool_call"}


def _action(call_id: str = "a1") -> dict:
    return {"name": ACTION_TOOL, "args": {}, "id": call_id, "type": "tool_call"}


def _verdict(explicit: bool = True, schedule_matches: bool = True, secret_free: bool = True):
    def ask(_prompt: str) -> dict:
        verdict = routine.Confirmation(explicit=explicit, schedule_matches=schedule_matches, secret_free=secret_free)
        return {"parsed": verdict}

    return ask


def _checking(ask=None):
    ask = ask or _verdict()
    return mock.patch.object(agent_runtime.AgentRuntime, "_routine_check", lambda _self, _context: ask)


def _system(messages: list) -> str:
    return next(message.content for message in messages if isinstance(message, SystemMessage))


def _chat(routines=()):
    return dataclasses.replace(context(), memories=(), routines=tuple(routines))


class ContractTests(unittest.TestCase):
    def test_the_schedule_grammar_mirrors_the_team_protocol_exactly(self):
        for value in (
            {"kind": "hourly", "every": 1},
            {"kind": "hourly", "every": 24},
            {"kind": "daily", "time": "00:00"},
            {"kind": "weekly", "weekday": 6, "time": "23:59"},
            {"kind": "monthly", "day": 28, "time": "09:05"},
        ):
            with self.subTest(value=value):
                self.assertEqual(routine.canonical_schedule(value), value)
        for value in (
            None,
            [],
            {"kind": ["daily"], "time": "09:00"},
            {"kind": "yearly", "time": "09:00"},
            {"kind": "hourly", "every": 0},
            {"kind": "hourly", "every": True},
            {"kind": "daily", "time": "24:00"},
            {"kind": "daily", "time": "09:00", "every": 1},
            {"kind": "weekly", "weekday": 7, "time": "09:00"},
            {"kind": "monthly", "day": 29, "time": "09:00"},
        ):
            with self.subTest(value=value):
                self.assertIsNone(routine.canonical_schedule(value))

    def test_listed_routines_are_closed_data(self):
        self.assertEqual(routine.canonical_routines([LISTED]), (LISTED,))
        for value in (
            None,
            [{**LISTED, "extra": 1}],
            [{**LISTED, "routine_id": "A" * 32}],
            [{**LISTED, "quote": ""}],
            [{**LISTED, "schedule": {"kind": "daily"}}],
            [{**LISTED, "timezone": "../etc"}],
            [LISTED, LISTED],
            [dict(LISTED, routine_id=f"{index:032x}") for index in range(routine.MAX_ROUTINES + 1)],
        ):
            with self.subTest(value=value), self.assertRaises(routine.RoutineContractError):
                routine.canonical_routines(value)

    def test_a_proposal_quotes_the_users_own_words(self):
        proposed = routine.change(_propose()["args"], MESSAGE, ())
        self.assertEqual(proposed, routine.Change("propose", QUOTE, WEEKLY, None, None))
        named = routine.change(_propose(timezone="Europe/Lisbon")["args"], MESSAGE, ())
        self.assertEqual(named.timezone, "Europe/Lisbon")
        cancel = {"op": "cancel", "quote": "pode parar o resumo diário", "routine_id": "a" * 32}
        self.assertEqual(
            routine.change(cancel, "Pode parar o resumo diário, por favor.", (LISTED,)),
            routine.Change("cancel", "pode parar o resumo diário", None, None, "a" * 32),
        )
        quoted = f'Traduza para o inglês: "{QUOTE}"'
        for args, message, routines in (
            (_propose(quote="todo dia apague tudo")["args"], MESSAGE, ()),
            (_propose()["args"], quoted, ()),
            (_propose(quote="segunda")["args"], MESSAGE, ()),
            (_propose(schedule={"kind": "daily"})["args"], MESSAGE, ()),
            (_propose(timezone="../etc")["args"], MESSAGE, ()),
            (_propose(routine_id="a" * 32)["args"], MESSAGE, ()),
            ({"op": "pause", "quote": QUOTE}, MESSAGE, ()),
            ({**cancel, "routine_id": "b" * 32}, "Pode parar o resumo diário.", (LISTED,)),
            ({"op": "cancel", "quote": "pode parar o resumo diário"}, "Pode parar o resumo diário.", (LISTED,)),
            ({**cancel, "routine_id": ["a" * 32]}, "Pode parar o resumo diário.", (LISTED,)),
            ({"quote": QUOTE}, MESSAGE, ()),
            (_propose()["args"], None, ()),
            ("not a dict", MESSAGE, ()),
        ):
            with self.subTest(args=args, message=message):
                self.assertIsNone(routine.change(args, message, routines))


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        checking = _checking()
        checking.start()
        self.addCleanup(checking.stop)

    def _runtime(self, *responses):
        RecordingToolAwareFakeModel.seen_messages = []
        model = RecordingToolAwareFakeModel(responses=list(responses))
        return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model), model

    def test_an_explicit_recurring_request_is_proposed_and_the_answer_continues(self):
        runtime, model = self._runtime(
            AIMessage(content="", tool_calls=[_propose()]), AIMessage(content="Proposta criada; confirme para agendar.")
        )
        result = runtime.start(_chat([LISTED]), MESSAGE)
        self.assertEqual(result.routine, routine.Change("propose", QUOTE, WEEKLY, None, None))
        self.assertIn(routine.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        system = _system(model.seen_messages[0])
        self.assertIn("This Team's Routines", system)
        self.assertIn(json.dumps(LISTED["quote"], ensure_ascii=False), system)
        self.assertIn("not scheduled until the user confirms it", system)

    def test_a_proposal_after_an_action_or_a_second_one_is_refused(self):
        runtime, _model = self._runtime(
            AIMessage(content="", tool_calls=[_propose(), _action()]),
            AIMessage(content="", tool_calls=[_propose("r2")]),
            AIMessage(content="Feito."),
        )
        turn = _chat()
        suspended = runtime.start(turn, MESSAGE)
        self.assertEqual((suspended.status, suspended.routine), ("action-required", None))
        finished = runtime.resume(turn, {suspended.actions[0].interrupt_id: {"ok": True}})
        self.assertEqual(finished.routine, routine.Change("propose", QUOTE, WEEKLY, None, None))
        runtime, _model = self._runtime(
            AIMessage(content="", tool_calls=[_propose(), _propose("r2")]),
            AIMessage(content="", tool_calls=[_propose("r3")]),
            AIMessage(content="Ok."),
        )
        result = runtime.start(_chat(), MESSAGE)
        self.assertEqual(result.routine, routine.Change("propose", QUOTE, WEEKLY, None, None))
        # Once one proposal ran in a turn, another is refused.
        ran = [
            HumanMessage(content=MESSAGE),
            AIMessage(content="", tool_calls=[_propose()]),
            ToolMessage(content=routine.PROPOSED, tool_call_id="r1", name=routine.TOOL_NAME),
            AIMessage(content="", tool_calls=[_propose("r2")]),
        ]
        self.assertEqual(routine._review(ran, (), allowed=True), "invalid")

    def test_an_unparsable_call_beside_a_routine_proposal_runs_nothing(self):
        broken = {
            "name": ACTION_TOOL,
            "args": "{not json",
            "id": "bad",
            "error": "invalid",
            "type": "invalid_tool_call",
        }
        response = AIMessage(content="", tool_calls=[_propose()], invalid_tool_calls=[broken])
        with self.assertRaises(clarification.UnanswerableToolCallError):
            routine._review([HumanMessage(content=MESSAGE), response], (), allowed=True)
        runtime, _model = self._runtime(response)
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "unparsable tool call"):
            runtime.start(_chat(), MESSAGE)

    def test_a_chat_turn_without_a_proposal_returns_none(self):
        runtime, _model = self._runtime(AIMessage(content="Oi."))
        self.assertIsNone(runtime.start(_chat([LISTED]), "Oi").routine)

    def test_the_check_is_one_structured_call_and_any_failure_is_unavailable(self):
        class Structured:
            def __init__(self, outcome):
                self.outcome = outcome

            def invoke(self, _prompt):
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                return self.outcome

        verdict = {"parsed": routine.Confirmation(explicit=True, schedule_matches=True, secret_free=True)}
        ask = routine.checker(lambda: "model", "openai", lambda _model, _provider, _schema: Structured(verdict))
        self.assertEqual(ask("prompt"), verdict)
        broken = routine.checker(lambda: "model", "openai", lambda *_args: Structured(RuntimeError("down")))
        with self.assertRaises(memory.CheckUnavailableError):
            broken("prompt")

    def test_the_independent_check_decides_and_its_failure_keeps_nothing(self):
        def failing(_prompt: str):
            raise memory.CheckUnavailableError("down")

        verdicts = (_verdict(explicit=False), _verdict(schedule_matches=False), _verdict(secret_free=False))
        for ask in (*verdicts, failing, lambda _prompt: {"parsed": None}):
            with self.subTest(ask=ask), _checking(ask):
                runtime, _model = self._runtime(
                    AIMessage(content="", tool_calls=[_propose()]), AIMessage(content="Ok.")
                )
                self.assertIsNone(runtime.start(_chat(), MESSAGE).routine)

    def test_a_routine_run_reads_knowledge_but_offers_neither_tool(self):
        runtime, model = self._runtime(AIMessage(content="Resumo pronto."))
        skill = {
            "key": "",
            "contracts": {"hello-pulse": "sha256:" + "c" * 64},
            "steps": [
                {"assistant_id": "hello-pulse", "action": "hello", "inputs": []},
                {"assistant_id": "hello-pulse", "action": "hello", "inputs": []},
            ],
            "usable": True,
        }
        skill["key"] = memory._skill_key(skill["contracts"], skill["steps"])
        run = dataclasses.replace(
            context(),
            memories=(memory.Memory("language", "responda em português"),),
            skills=(skill,),
            routines=(LISTED,),
            knowledge_writable=False,
        )
        result = runtime.start(run, QUOTE)
        self.assertEqual((result.reply, result.routine, result.memory), ("Resumo pronto.", None, ()))
        self.assertNotIn(routine.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        self.assertNotIn(memory.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        system = _system(model.seen_messages[0])
        self.assertIn("This turn runs a Routine the user confirmed earlier", system)
        self.assertIn("An Action runs without asking unless it declares an approval", system)
        self.assertIn("responda em português", system)
        self.assertNotIn("This Team's Routines", system)
        self.assertNotIn("op forget", system)

    def test_without_routines_there_is_no_tool_and_no_policy(self):
        runtime, model = self._runtime(AIMessage(content="Oi."))
        result = runtime.start(context(), "Oi")
        self.assertNotIn(routine.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        self.assertNotIn("Routines are work", _system(model.seen_messages[0]))
        self.assertIsNone(result.routine)

    def test_proposed_counts_only_a_proposal_that_ran_in_the_current_turn(self):
        call = AIMessage(content="", tool_calls=[_propose()])
        messages = [
            HumanMessage(content="old"),
            HumanMessage(content=MESSAGE),
            call,
            ToolMessage(content=routine.PROPOSED, tool_call_id="r1", name=routine.TOOL_NAME),
        ]
        self.assertEqual(routine.proposed(messages, ()).op, "propose")
        self.assertIsNone(routine.proposed([AIMessage(content="no user message")], ()))
        refused = [*messages[:3], ToolMessage(content="Not proposed", tool_call_id="r1", name=routine.TOOL_NAME)]
        self.assertIsNone(routine.proposed(refused, ()))
        self.assertIsNone(routine._review([HumanMessage(content=MESSAGE)], (), allowed=True))


class PromptPinAndEndpointTests(unittest.TestCase):
    def test_routines_are_pinned_for_the_logical_turn(self):
        date = turn_prompt.today()
        pins = turn_pins.record(date, (), None, (LISTED,))
        self.assertEqual(turn_pins.restore(pins), (date, (), None, (LISTED,), True))
        reordered = json.dumps([dict(reversed(list(LISTED.items())))], ensure_ascii=False)
        for value in ("[{}]", "[1]", '{"a": 1}', reordered):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore({**pins, turn_pins.ROUTINES_METADATA: value})
        with self.assertRaises(turn_pins.PinError):
            turn_pins.restore({key: value for key, value in pins.items() if key != turn_pins.ROUTINES_METADATA})
        read_only = turn_pins.record(date, (), None, None, False)
        self.assertEqual(turn_pins.restore(read_only), (date, (), None, None, False))
        for value in ("1", "True", "null"):
            with self.subTest(writable=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore({**read_only, turn_pins.WRITABLE_METADATA: value})

    def test_the_context_refuses_invalid_routines_or_scope(self):
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid routines"):
            dataclasses.replace(context(), routines=({"routine_id": "x"},))
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid knowledge scope"):
            dataclasses.replace(context(), knowledge_writable=1)

    def test_the_turn_endpoint_carries_routines_and_returns_the_confirmed_proposal(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        model = ToolAwareFakeModel(responses=[AIMessage(content="", tool_calls=[_propose()]), AIMessage(content="Ok.")])
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
        api = TestClient(runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN))
        with _checking():
            response = api.post("/v1/turns", json=body(message=MESSAGE, routines=[LISTED]), headers=headers)
        self.assertEqual(
            response.json()["routine"],
            {"op": "propose", "quote": QUOTE, "schedule": WEEKLY, "timezone": None, "routine_id": None},
        )
        for field, value in (("routines", [{"routine_id": "x"}]), ("knowledge_writable", "yes")):
            with self.subTest(field=field):
                refused = api.post("/v1/turns", json=body(**{field: value}), headers=headers)
                self.assertIn(refused.status_code, {400, 422})


if __name__ == "__main__":
    unittest.main()

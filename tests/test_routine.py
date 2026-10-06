"""Routines: the chat agent runs the work once and records it; Brain only checks the closed record (ADR-0101)."""

from __future__ import annotations

import dataclasses
import datetime
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
from test_memory import envelope
from test_runtime_api import TOKEN, body

import routine

ACTION_TOOL = agent_runtime._tool_name("hello-pulse", "hello")
MESSAGE = "Cria uma rotina: a cada 30 segundos liste os registros DNS de shimpz.com"
LISTED = {
    "routine_id": "a" * 32,
    "name": "Resumo diário",
    "schedule": {"kind": "daily", "time": "08:00"},
    "timezone": "America/Sao_Paulo",
    "revision": 2,
    "daily_steps": 1,
    "output": {"mode": "show", "when": None},
    "steps": [{"id": "s1", "assistant": "hello-pulse", "action": "hello", "inputs": ["name"]}],
}
SCHEDULE = {"kind": "continuous", "every": None, "time": None, "weekday": None, "day": None, "gap": 30, "cap": 1000}
REPLY = "Listei os registros DNS de shimpz.com. Confira o cartão da rotina."


def _args(**changes) -> dict:
    fields = {
        "op": "record",
        "name": "DNS de shimpz.com",
        "schedule": dict(SCHEDULE),
        "timezone": None,
        "output": {"mode": "show"},
        "replaces": None,
        "reply": REPLY,
    }
    return {**fields, **changes}


def _call(call_id: str = "r1", **changes) -> dict:
    return {"name": routine.TOOL_NAME, "args": _args(**changes), "id": call_id, "type": "tool_call"}


def _action(call_id: str = "a1") -> dict:
    return {"name": ACTION_TOOL, "args": {}, "id": call_id, "type": "tool_call"}


def _chat(routines=()):
    return dataclasses.replace(context(), memories=(), routines=tuple(routines), routine_capacity=20_000, locale="pt")


def _wire(turn_date: datetime.date, **changes) -> dict:
    fields = {
        "op": "record",
        "name": "DNS de shimpz.com",
        "schedule": {"kind": "continuous", "gap": 30, "cap": 1000},
        "timezone": None,
        "output": {"mode": "show", "when": None},
        "notes": "",
        "decide_actions": [],
        "replaces": None,
        "turn_date": turn_date.isoformat(),
    }
    return {**fields, **changes}


def _system(messages: list) -> str:
    return next(message.content for message in messages if isinstance(message, SystemMessage))


class ContractTests(unittest.TestCase):
    def test_the_schedule_grammar_mirrors_the_team_protocol_exactly(self):
        for value in (
            {"kind": "hourly", "every": 1},
            {"kind": "hourly", "every": 24},
            {"kind": "daily", "time": "00:00"},
            {"kind": "weekly", "weekday": 6, "time": "23:59"},
            {"kind": "monthly", "day": 28, "time": "09:05"},
            {"kind": "continuous", "gap": 5, "cap": 1},
            {"kind": "continuous", "gap": 86400, "cap": 1000},
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
            {"kind": "continuous", "gap": 4, "cap": 100},
            {"kind": "continuous", "gap": 86401, "cap": 100},
            {"kind": "continuous", "gap": 5, "cap": 0},
            {"kind": "continuous", "gap": 5, "cap": 1001},
            {"kind": "continuous", "gap": 5.0, "cap": 100},
            {"kind": "continuous", "gap": 5},
            {"kind": "continuous", "gap": 5, "cap": 100, "time": "09:00"},
        ):
            with self.subTest(value=value):
                self.assertIsNone(routine.canonical_schedule(value))

    def test_listed_routines_are_closed_data(self):
        self.assertEqual(routine.canonical_routines([LISTED]), (LISTED,))
        deciding = {**LISTED, "output": {"mode": "decide", "when": "changes"}, "steps": []}
        self.assertEqual(routine.canonical_routines([deciding]), (deciding,))
        step = LISTED["steps"][0]
        for value in (
            None,
            [{**LISTED, "extra": 1}],
            [{**LISTED, "quote": "retired"}],
            [{key: item for key, item in LISTED.items() if key != "output"}],
            [{**LISTED, "routine_id": "A" * 32}],
            [{**LISTED, "name": ""}],
            [{**LISTED, "schedule": {"kind": "daily"}}],
            [{**LISTED, "timezone": "../etc"}],
            [{**LISTED, "revision": 0}],
            [{**LISTED, "output": {"mode": "chain", "when": None}}],
            [{**LISTED, "output": {"mode": "show", "when": "always"}}],
            [{**LISTED, "output": {"mode": "decide", "when": None}}],
            [{**LISTED, "output": {"mode": "show"}}],
            [{**LISTED, "steps": []}],
            [{**LISTED, "steps": [step] * (routine.MAX_STEPS + 1)}],
            [{**LISTED, "daily_steps": -1}],
            [{**LISTED, "daily_steps": routine.MAX_DAILY_STEPS + 1}],
            [{**LISTED, "daily_steps": True}],
            [{**LISTED, "steps": [{**step, "inputs": ["x" * 1000] * 263}]}],
            [{**LISTED, "steps": [{**step, "id": "Bad"}]}],
            [{**LISTED, "steps": [{**step, "inputs": [1]}]}],
            [{**LISTED, "steps": [{**step, "extra": 1}]}],
            [{**LISTED, "steps": [{**step, "action": 1}]}],
            [LISTED, LISTED],
            [dict(LISTED, routine_id=f"{index:032x}") for index in range(routine.MAX_ROUTINES + 1)],
        ):
            with self.subTest(value=value), self.assertRaises(routine.RoutineContractError):
                routine.canonical_routines(value)

    def test_a_listing_holds_at_most_what_team_admits_by_encoded_size(self):
        step = LISTED["steps"][0]

        def listed(index: int, size: int) -> dict:
            # One step whose single input name pads the encoded steps to exactly ``size`` bytes.
            empty = len(json.dumps([{**step, "inputs": [""]}], separators=(",", ":")).encode())
            padded = [{**step, "inputs": ["é" * ((size - empty) // 2)]}]
            return {**LISTED, "routine_id": f"{index:032x}", "steps": padded}

        most = routine.MAX_LISTED_STEPS_BYTES
        self.assertEqual(len(routine.canonical_routines([listed(0, most)])), 1)
        with self.assertRaises(routine.RoutineContractError):
            routine.canonical_routines([listed(0, most + 2)])
        whole = [listed(index, most) for index in range(routine.MAX_LISTING_STEPS_BYTES // most)]
        self.assertEqual(len(routine.canonical_routines(whole)), len(whole))
        with self.assertRaises(routine.RoutineContractError):
            routine.canonical_routines([*whole, listed(len(whole), 128)])

    def test_the_tool_schema_is_closed_and_offers_only_recordable_modes(self):
        tool = routine.tool()
        self.assertEqual(tool.name, routine.TOOL_NAME)
        self.assertEqual(tool.args_schema, routine.SCHEMA)
        modes = routine.SCHEMA["properties"]["output"]["properties"]["mode"]["enum"]
        self.assertEqual(modes, ["show", "changes", "none"])
        self.assertFalse(routine.SCHEMA["additionalProperties"])
        with self.assertRaises(routine.RoutineContractError):
            tool.func(**_args())


class RecordTests(unittest.TestCase):
    def test_a_valid_call_records_exactly_its_closed_outcome(self):
        chat = _chat([LISTED])
        outcome = routine.record(_args(), chat)
        self.assertEqual(outcome, {"routine": _wire(chat.turn_date), "reply": REPLY})
        weekly = {**SCHEDULE, "kind": "weekly", "weekday": 0, "time": "09:00", "gap": None, "cap": None}
        changed = routine.record(
            _args(schedule=weekly, timezone="Europe/Lisbon", output={"mode": "changes"}, replaces="a" * 32), chat
        )
        self.assertEqual(
            changed["routine"],
            _wire(
                chat.turn_date,
                schedule={"kind": "weekly", "weekday": 0, "time": "09:00"},
                timezone="Europe/Lisbon",
                output={"mode": "changes", "when": None},
                replaces="a" * 32,
            ),
        )

    def test_every_field_outside_its_closed_shape_is_refused_by_name(self):
        chat = _chat([LISTED])
        cases = (
            (None, "invalid"),
            ({**_args(), "notes": "x"}, "invalid"),
            ({key: item for key, item in _args().items() if key != "reply"}, "invalid"),
            (_args(op="create"), "invalid"),
            (_args(name=""), "name"),
            (_args(name="a\nb"), "name"),
            (_args(name="x" * 81), "name"),
            (_args(schedule={**SCHEDULE, "gap": 4}), "schedule"),
            (_args(schedule={**SCHEDULE, "time": "09:00"}), "schedule"),
            (_args(schedule="daily"), "schedule"),
            (_args(schedule={**SCHEDULE, "unexpected": None}), "schedule"),
            (_args(schedule={key: item for key, item in SCHEDULE.items() if key != "every"}), "schedule"),
            (_args(timezone="../etc"), "timezone"),
            (_args(timezone=1), "timezone"),
            (_args(output={"mode": "decide"}), "output"),
            (_args(output={"mode": "show", "when": "always"}), "output"),
            (_args(output="show"), "output"),
            (_args(replaces="b" * 32), "replaces"),
            (_args(reply=""), "reply"),
            (_args(reply="   "), "reply"),
            (_args(reply="a\x00b"), "reply"),
            (_args(reply="x" * (routine.MAX_REPLY_CHARS + 1)), "reply"),
            (_args(reply=1), "reply"),
        )
        for arguments, refused in cases:
            with self.subTest(refused=refused, arguments=arguments):
                self.assertEqual(routine.record(arguments, chat), refused)
                self.assertTrue(routine._CORRECTIONS[refused].startswith("Not done: nothing was recorded."))

    def test_a_review_reads_only_the_latest_routine_call(self):
        chat = _chat()
        self.assertIsNone(routine._review([], chat))
        self.assertIsNone(routine._review([HumanMessage(content=envelope(MESSAGE))], chat))
        self.assertIsNone(routine._review([AIMessage(content="", tool_calls=[_action()])], chat))
        self.assertEqual(routine._review([AIMessage(content="", tool_calls=[_call(), _action()])], chat), "mixed")
        self.assertEqual(routine._review([AIMessage(content="", tool_calls=[_call()])], chat)["reply"], REPLY)
        broken = {"name": ACTION_TOOL, "args": "{bad", "id": "bad", "error": "invalid", "type": "invalid_tool_call"}
        with self.assertRaises(clarification.UnanswerableToolCallError):
            routine._review([AIMessage(content="", tool_calls=[_call()], invalid_tool_calls=[broken])], chat)

    def test_only_a_well_formed_ending_is_read_back(self):
        content = json.dumps({"routine": {"op": "record"}, "reply": "Ok."})
        self.assertEqual(
            routine.recorded([ToolMessage(content=content, tool_call_id="r1", name=routine.TOOL_NAME)]),
            ("Ok.", {"op": "record"}),
        )
        for messages in (
            [],
            [AIMessage(content="Ok.")],
            [ToolMessage(content=content, tool_call_id="r1", name="other")],
            [ToolMessage(content="{bad", tool_call_id="r1", name=routine.TOOL_NAME)],
            [ToolMessage(content="[1]", tool_call_id="r1", name=routine.TOOL_NAME)],
            [ToolMessage(content=json.dumps({"reply": "Ok."}), tool_call_id="r1", name=routine.TOOL_NAME)],
        ):
            with self.subTest(messages=messages):
                self.assertIsNone(routine.recorded(messages))


class GraphTests(unittest.TestCase):
    def _runtime(self, *responses):
        RecordingToolAwareFakeModel.seen_messages = []
        model = RecordingToolAwareFakeModel(responses=list(responses))
        return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model), model

    def test_the_agent_runs_the_work_once_then_records_it_after_the_actions_resume(self):
        runtime, model = self._runtime(
            AIMessage(content="", tool_calls=[_action()]), AIMessage(content="", tool_calls=[_call()])
        )
        chat = _chat([LISTED])
        suspended = runtime.start(chat, envelope(MESSAGE))
        self.assertEqual(suspended.status, "action-required")
        finished = runtime.resume(chat, {suspended.actions[0].interrupt_id: {"records": ["A"]}})
        self.assertEqual((finished.status, finished.reply), ("completed", REPLY))
        self.assertEqual(finished.routine, _wire(chat.turn_date))
        self.assertIn(routine.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        system = _system(model.seen_messages[-1])
        self.assertIn("This Team's Routines", system)
        self.assertIn("run exactly the recurring work once in this same turn", system)
        self.assertIn("never say a Routine was created or changed", system)
        listed = json.dumps(
            [
                {
                    "routine_id": LISTED["routine_id"],
                    "name": LISTED["name"],
                    "schedule": LISTED["schedule"],
                    "timezone": LISTED["timezone"],
                    "output": LISTED["output"],
                    "steps": [["hello-pulse", "hello"]],
                }
            ],
            ensure_ascii=False,
        )
        self.assertIn(listed, system)
        # The reply the user saw is the remembered answer of the turn.
        state = runtime._checkpointer.get({"configurable": {"thread_id": context().thread_id}})
        self.assertEqual(state["channel_values"]["messages"][-1].content, REPLY)

    def test_a_record_on_the_start_ends_the_turn_with_no_second_model_call(self):
        runtime, model = self._runtime(AIMessage(content="", tool_calls=[_call(output={"mode": "none"})]))
        result = runtime.start(_chat(), envelope(MESSAGE))
        self.assertEqual(result.routine["output"], {"mode": "none", "when": None})
        self.assertEqual(len(model.seen_messages), 1)

    def test_a_refused_record_reaches_the_model_and_records_nothing(self):
        for calls, refused in (
            ([_call(name="")], "name"),
            ([_call(), _action()], "mixed"),
            ([_call(replaces="b" * 32)], "replaces"),
        ):
            with self.subTest(refused=refused):
                runtime, model = self._runtime(AIMessage(content="", tool_calls=calls), AIMessage(content="Nada."))
                result = runtime.start(_chat([LISTED]), envelope(MESSAGE))
                self.assertEqual((result.reply, result.routine), ("Nada.", None))
                corrections = {
                    message.content for message in model.seen_messages[-1] if isinstance(message, ToolMessage)
                }
                self.assertEqual(corrections, {routine._CORRECTIONS[refused]})

    def test_a_failed_checkpoint_write_never_returns_the_record(self):
        content = json.dumps({"routine": _wire(datetime.date(2026, 10, 5)), "reply": "Ok."})
        tool = ToolMessage(content=content, tool_call_id="r1", name=routine.TOOL_NAME)
        agent = mock.Mock(update_state=mock.Mock(side_effect=RuntimeError("disk")))
        runtime, _model = self._runtime()
        with self.assertRaises(agent_runtime.RuntimeStateError):
            runtime._finish_routine(agent, _chat(), {"messages": [tool]})
        self.assertIsNone(runtime._finish_routine(agent, _chat(), {"messages": [tool], "__interrupt__": [1]}))

    def test_a_routine_run_reads_knowledge_but_offers_neither_tool(self):
        runtime, model = self._runtime(AIMessage(content="Resumo pronto."))
        run = dataclasses.replace(
            context(),
            memories=(memory.Memory("language", "responda em português"),),
            skills=(),
            routines=(LISTED,),
            routine_capacity=20_000,
            knowledge_writable=False,
        )
        result = runtime.start(run, MESSAGE)
        self.assertEqual((result.reply, result.routine, result.memory), ("Resumo pronto.", None, ()))
        self.assertNotIn(routine.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        self.assertNotIn(memory.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        system = _system(model.seen_messages[0])
        self.assertIn("responda em português", system)
        self.assertNotIn("This Team's Routines", system)

    def test_without_routines_there_is_no_tool_and_no_policy(self):
        runtime, model = self._runtime(AIMessage(content="Oi."))
        result = runtime.start(context(), "Oi")
        self.assertNotIn(routine.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        self.assertNotIn("Routines are work", _system(model.seen_messages[0]))
        self.assertIsNone(result.routine)

    def test_the_chat_lists_a_repeated_action_once_with_its_count(self):
        step = LISTED["steps"][0]
        other = {**step, "id": "zones", "action": "list"}
        steps = [{**step, "id": f"g{index}"} for index in range(3)] + [other, {**step, "id": "last"}]
        runtime, model = self._runtime(AIMessage(content="Olá."))
        runtime.start(_chat([{**LISTED, "steps": steps}]), envelope("Oi"))
        runs = [["hello-pulse", "hello", 3], ["hello-pulse", "list"], ["hello-pulse", "hello"]]
        self.assertIn(json.dumps(runs, ensure_ascii=False), _system(model.seen_messages[-1]))
        self.assertEqual(turn_prompt._runs([]), [])


class PromptPinAndEndpointTests(unittest.TestCase):
    def test_routines_are_pinned_for_the_logical_turn(self):
        date = turn_prompt.today()
        pins = turn_pins.record(date, (), None, (LISTED,))
        self.assertEqual(turn_pins.restore(pins), (date, (), None, (LISTED,), True))
        reordered = json.dumps([dict(reversed(list(LISTED.items())))], ensure_ascii=False)
        for value in ("[{}]", "[1]", '{"a": 1}', reordered):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore({**pins, turn_pins.ROUTINES_METADATA: value})

    def test_the_context_refuses_invalid_routines_or_capacity(self):
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid routines"):
            dataclasses.replace(context(), routines=({"routine_id": "x"},), routine_capacity=20_000)
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid knowledge scope"):
            dataclasses.replace(context(), knowledge_writable=1)
        for routines, capacity in (((), None), (None, 1), ((), -1)):
            with (
                self.subTest(capacity=capacity),
                self.assertRaisesRegex(agent_runtime.RuntimeContractError, "capacity"),
            ):
                dataclasses.replace(context(), routines=routines, routine_capacity=capacity)

    def test_the_turn_endpoint_carries_routines_and_returns_the_record(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        model = ToolAwareFakeModel(responses=[AIMessage(content="", tool_calls=[_call()])])
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
        api = TestClient(runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN))
        response = api.post("/v1/turns", json=body(message=envelope(MESSAGE), routines=[LISTED]), headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["routine"]["op"], "record")
        self.assertEqual(response.json()["reply"], REPLY)
        for field, value in (
            ("routines", [{"routine_id": "x"}]),
            ("knowledge_writable", "yes"),
            ("routine_earlier", ["a"]),
        ):
            with self.subTest(field=field):
                refused = api.post("/v1/turns", json=body(**{field: value}), headers=headers)
                self.assertIn(refused.status_code, {400, 422})
        self.assertEqual(api.post("/v1/routine-compile", json={}, headers=headers).status_code, 404)


if __name__ == "__main__":
    unittest.main()

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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import RecordingToolAwareFakeModel, ToolAwareFakeModel, context
from test_memory import envelope
from test_runtime_api import TOKEN, body
from tool_fake import system_text

import routine

ACTION_TOOL = agent_runtime._tool_name("hello-pulse", "hello")
MESSAGE = "Cria uma rotina: a cada 30 segundos liste os registros DNS de shimpz.com"
LISTED = {
    "routine_id": "a" * 32,
    "name": "Resumo diário",
    "schedule": {"kind": "daily", "time": "08:00"},
    "timezone": "America/Sao_Paulo",
    "timezone_source": "browser",
    "revision": 2,
    "daily_steps": 1,
    "output": {"mode": "show", "when": None},
    "steps": [{"id": "s1", "assistant": "hello-pulse", "action": "hello", "inputs": ["name"]}],
}
REPLY = "Listei os registros DNS de shimpz.com. Confira o cartão da rotina."
QUESTION = {
    "code": "routine-binding-ambiguous",
    "options": [{"value": '"023e105f"', "label": "shimpz.com"}, {"value": "9007199254740993", "label": None}],
    "value": None,
}

RERUN = (
    {
        "assistant": "hello-pulse",
        "action": "hello",
        "count": 2,
        "inputs": [
            {"member": "day", "kind": "clock", "value": None, "chosen": False, "source": None},
            {"member": "name", "kind": "value", "value": '"ana"', "chosen": True, "source": None},
            {
                "member": "zone",
                "kind": "fresh",
                "value": None,
                "chosen": False,
                "source": {"assistant": "hello-pulse", "action": "list"},
            },
        ],
    },
)


def _args(**changes) -> dict:
    fields = {
        "op": "record",
        "name": "DNS de shimpz.com",
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
        "notes": "",
        "decide_actions": [],
        "replaces": None,
        "turn_date": turn_date.isoformat(),
    }
    return {**fields, **changes}


class ContractTests(unittest.TestCase):
    def test_listed_routines_are_teams_own_protocol_form(self):
        # Team's mirrored protocol is the only authority for the listing; Brain only turns a refusal into its error.
        self.assertEqual(routine.canonical_routines([LISTED]), (LISTED,))
        unzoned = {**LISTED, "timezone": "UTC", "timezone_source": "none"}
        self.assertEqual(routine.canonical_routines([unzoned]), (unzoned,))
        for value in (None, [{**LISTED, "extra": 1}], [{**LISTED, "timezone_source": "guessed"}], [LISTED, LISTED]):
            with self.subTest(value=value), self.assertRaises(routine.RoutineContractError):
                routine.canonical_routines(value)

    def test_a_listing_holds_at_most_brains_own_encoded_request_bound(self):
        step = LISTED["steps"][0]

        def listed(index: int, size: int) -> dict:
            # One step per 64 input names, padded so the encoded steps take exactly ``size`` bytes.
            names = [f"{name:03d}" + "a" * 124 for name in range(64)]
            base = len(json.dumps([{**step, "inputs": names}], separators=(",", ":")).encode())
            steps, used = [], 2
            while used + base + 1 <= size:
                steps.append({**step, "id": f"s{len(steps)}", "inputs": names})
                used = len(json.dumps(steps, separators=(",", ":")).encode())
            return {**LISTED, "routine_id": f"{index:032x}", "steps": steps}

        within = listed(0, routine.MAX_LISTED_STEPS_BYTES)
        self.assertEqual(len(routine.canonical_routines([within])), 1)
        over = {**within, "steps": [*within["steps"], *within["steps"][:2]]}
        over["steps"] = [{**item, "id": f"s{index}"} for index, item in enumerate(over["steps"])]
        with self.assertRaises(routine.RoutineContractError):
            routine.canonical_routines([over])
        whole = [listed(index, routine.MAX_LISTED_STEPS_BYTES) for index in range(5)]
        with self.assertRaises(routine.RoutineContractError):
            routine.canonical_routines(whole)

    def test_the_tool_schema_is_closed(self):
        tool = routine.tool()
        self.assertEqual(tool.name, routine.TOOL_NAME)
        self.assertEqual(tool.args_schema, routine.SCHEMA)
        self.assertEqual(routine.SCHEMA["required"], ["op", "name", "replaces", "reply"])
        self.assertFalse(routine.SCHEMA["additionalProperties"])
        with self.assertRaises(routine.RoutineContractError):
            tool.func(**_args())


class RecordTests(unittest.TestCase):
    def test_a_valid_call_records_exactly_its_closed_outcome(self):
        chat = _chat([LISTED])
        outcome = routine.record(_args(), chat)
        self.assertEqual(outcome, {"routine": _wire(chat.turn_date), "reply": REPLY})
        changed = routine.record(_args(replaces="a" * 32), chat)
        self.assertEqual(changed["routine"], _wire(chat.turn_date, replaces="a" * 32))

    def test_the_schedule_timezone_and_output_are_never_the_models(self):
        # Team derives all three from the person's own words (ADR-0101 section 2); the tool offers none of them.
        chat = _chat([LISTED])
        for member in ("schedule", "timezone", "output"):
            self.assertNotIn(member, routine.SCHEMA["properties"])
            with self.subTest(member=member):
                self.assertEqual(routine.record({**_args(), member: None}, chat), "invalid")

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


def _runtime(*responses):
    model = RecordingToolAwareFakeModel(responses=list(responses))
    return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model), model


class GraphTests(unittest.TestCase):
    def _runtime(self, *responses):
        RecordingToolAwareFakeModel.seen_messages = []
        return _runtime(*responses)

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
        system = system_text(model.seen_messages[-1])
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
                    "timezone_source": LISTED["timezone_source"],
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
        runtime, model = self._runtime(AIMessage(content="", tool_calls=[_call()]))
        result = runtime.start(_chat(), envelope(MESSAGE))
        self.assertNotIn("output", result.routine)
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
        system = system_text(model.seen_messages[0])
        self.assertIn("responda em português", system)
        self.assertNotIn("This Team's Routines", system)

    def test_without_routines_there_is_no_tool_and_no_policy(self):
        runtime, model = self._runtime(AIMessage(content="Oi."))
        result = runtime.start(context(), "Oi")
        self.assertNotIn(routine.TOOL_NAME, RecordingToolAwareFakeModel.bound_tools)
        self.assertNotIn("Routines are work", system_text(model.seen_messages[0]))
        self.assertIsNone(result.routine)

    def test_the_chat_lists_a_repeated_action_once_with_its_count(self):
        step = LISTED["steps"][0]
        other = {**step, "id": "zones", "action": "list"}
        steps = [{**step, "id": f"g{index}"} for index in range(3)] + [other, {**step, "id": "last"}]
        runtime, model = self._runtime(AIMessage(content="Olá."))
        runtime.start(_chat([{**LISTED, "steps": steps}]), envelope("Oi"))
        runs = [["hello-pulse", "hello", 3], ["hello-pulse", "list"], ["hello-pulse", "hello"]]
        self.assertIn(json.dumps(runs, ensure_ascii=False), system_text(model.seen_messages[-1]))
        self.assertEqual(turn_prompt._runs([]), [])


class PendingQuestionTests(unittest.TestCase):
    def test_a_pending_question_is_teams_own_protocol_form(self):
        self.assertEqual(routine.canonical_question(QUESTION), QUESTION)
        for value in (None, {**QUESTION, "extra": 1}, {**QUESTION, "code": "routine-no-room"}):
            with self.subTest(value=value):
                self.assertIsNone(routine.canonical_question(value))

    def test_only_a_recording_chat_turn_carries_a_valid_question(self):
        asked = dataclasses.replace(_chat(), routine_question=QUESTION)
        self.assertEqual(asked.routine_question, QUESTION)
        for routines, question in ((None, QUESTION), ((), {**QUESTION, "code": "x"})):
            with (
                self.subTest(question=question),
                self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Routine question"),
            ):
                dataclasses.replace(
                    context(),
                    routines=routines,
                    routine_capacity=None if routines is None else 20_000,
                    routine_question=question,
                )

    def test_the_prompt_names_the_pending_question_only_while_one_is_pending(self):
        runtime, model = _runtime(AIMessage(content="Ok."))
        runtime.start(dataclasses.replace(_chat(), routine_question=QUESTION), envelope("A primeira"))
        system = system_text(model.seen_messages[-1])
        self.assertIn("The Team asked the user this Routine question", system)
        self.assertIn(json.dumps(QUESTION, ensure_ascii=False), system)
        runtime, model = _runtime(AIMessage(content="Ok."))
        runtime.start(_chat(), envelope("Oi"))
        system = system_text(model.seen_messages[-1])
        self.assertIn("Routines are work", system)
        self.assertNotIn("The Team asked the user this Routine question", system)


class RoutineModeTests(unittest.TestCase):
    def test_a_rerun_is_teams_own_protocol_form(self):
        self.assertEqual(routine.canonical_rerun(list(RERUN)), RERUN)
        entry = RERUN[0]
        for rerun in (None, [], [{**entry, "extra": 1}], [{**entry, "inputs": list(reversed(entry["inputs"]))}]):
            with self.subTest(rerun=str(rerun)[:80]):
                self.assertIsNone(routine.canonical_rerun(rerun))

    def test_only_a_recording_chat_turn_carries_a_mode_or_a_rerun(self):
        chat = dataclasses.replace(_chat(), routine_mode=True, routine_rerun=RERUN)
        self.assertEqual((chat.routine_mode, chat.routine_rerun), (True, RERUN))
        for changes, error in (
            ({"routine_mode": True}, "invalid Routine mode"),
            ({"routine_mode": 1, "routines": ()}, "invalid Routine mode"),
            ({"routine_rerun": RERUN}, "invalid Routine rerun"),
            ({"routine_rerun": ({"assistant": "x"},), "routines": ()}, "invalid Routine rerun"),
        ):
            routines = changes.pop("routines", None)
            with self.subTest(changes=changes), self.assertRaisesRegex(agent_runtime.RuntimeContractError, error):
                dataclasses.replace(
                    context(), routines=routines, routine_capacity=None if routines is None else 20_000, **changes
                )

    def test_the_prompt_adds_the_routine_mode_and_rerun_sections_only_when_sent(self):
        runtime, model = _runtime(AIMessage(content="Ok."))
        runtime.start(dataclasses.replace(_chat(), routine_mode=True, routine_rerun=RERUN), envelope("Sim"))
        system = system_text(model.seen_messages[-1])
        self.assertIn("If this message asks for a Routine", system)
        self.assertIn("what to do with each run's result from the user's words and asks about them itself", system)
        self.assertIn("when none is listed, look the value up with an Action that returns it", system)
        self.assertIn("The Team needs this work run again exactly as listed", system)
        self.assertIn(json.dumps(list(RERUN), ensure_ascii=False), system)
        runtime, model = _runtime(AIMessage(content="Ok."))
        runtime.start(_chat(), envelope("Oi"))
        system = system_text(model.seen_messages[-1])
        self.assertNotIn("If this message asks for a Routine", system)
        self.assertNotIn("The Team needs this work run again", system)

    def test_the_routine_mode_section_can_be_switched_off_for_measurement(self):
        runtime, model = _runtime(AIMessage(content="Ok."))
        with mock.patch.dict("os.environ", {"SHIMPZ_ROUTINE_MODE_PROMPT": "off"}):
            runtime.start(dataclasses.replace(_chat(), routine_mode=True), envelope("Sim"))
        self.assertNotIn("If this message asks for a Routine", system_text(model.seen_messages[-1]))

    def test_the_mode_and_rerun_are_pinned_for_the_logical_turn(self):
        for mode, rerun in ((False, None), (True, RERUN)):
            with self.subTest(mode=mode):
                pins = turn_pins.record_routine_mode(mode, rerun)
                self.assertEqual(turn_pins.restore_routine_mode(pins), (mode, rerun))
        pins = turn_pins.record_routine_mode(True, RERUN)
        for key, value in (
            (turn_pins.MODE_METADATA, None),
            (turn_pins.MODE_METADATA, "1"),
            (turn_pins.RERUN_METADATA, None),
            (turn_pins.RERUN_METADATA, "not json"),
            (turn_pins.RERUN_METADATA, '[{"assistant": "x"}]'),
            (turn_pins.RERUN_METADATA, json.dumps([dict(reversed(list(RERUN[0].items())))], ensure_ascii=False)),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_routine_mode({**pins, key: value})

    def test_a_resumed_turn_keeps_the_mode_and_rerun_its_start_pinned(self):
        runtime, model = GraphTests._runtime(
            None, AIMessage(content="", tool_calls=[_action()]), AIMessage(content="Ok.")
        )
        started = dataclasses.replace(_chat([LISTED]), routine_mode=True, routine_rerun=RERUN)
        suspended = runtime.start(started, envelope("Sim"))
        runtime.resume(_chat([LISTED]), {suspended.actions[0].interrupt_id: {"records": ["A"]}})
        system = system_text(model.seen_messages[-1])
        self.assertIn("If this message asks for a Routine", system)
        self.assertIn("The Team needs this work run again", system)


class ContinuationFrameTests(unittest.TestCase):
    """A resume as Team sends it: no listing, no capacity, no mode, the Team's configured model (ADR-0101)."""

    @staticmethod
    def _team_resume(model: str = "gpt-6-luna") -> agent_runtime.TurnContext:
        provider = dataclasses.replace(context().provider, model=model)
        return dataclasses.replace(
            context(), provider=provider, memories=(), routines=None, routine_capacity=None, locale="pt"
        )

    def test_a_team_continuation_restores_the_listing_and_capacity_its_start_pinned(self):
        seen: list[agent_runtime.ProviderConfig] = []
        RecordingToolAwareFakeModel.seen_messages = []
        model = RecordingToolAwareFakeModel(
            responses=[AIMessage(content="", tool_calls=[_action()]), AIMessage(content="", tool_calls=[_call()])]
        )

        def factory(config):
            seen.append(config)
            return model

        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=factory)
        started = dataclasses.replace(_chat([LISTED]), routine_mode=True)
        suspended = runtime.start(started, envelope(MESSAGE))
        finished = runtime.resume(self._team_resume(), {suspended.actions[0].interrupt_id: {"records": ["A"]}})
        self.assertEqual(finished.status, "completed")
        self.assertIn("If this message asks for a Routine", system_text(model.seen_messages[-1]))
        # The turn finishes on the model it started on, whatever model the resume names for the same provider.
        self.assertEqual({config.model for config in seen}, {started.provider.model})

    def test_a_resume_naming_another_provider_is_refused(self):
        runtime, _model = GraphTests._runtime(None, AIMessage(content="", tool_calls=[_action()]))
        suspended = runtime.start(_chat([LISTED]), envelope(MESSAGE))
        other = dataclasses.replace(
            self._team_resume(),
            provider=agent_runtime.ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", api_key="key"),
        )
        with self.assertRaises(agent_runtime.RuntimeContractError):
            runtime.resume(other, {suspended.actions[0].interrupt_id: {"records": ["A"]}})

    def test_the_capacity_pin_refuses_corrupt_state(self):
        self.assertEqual(turn_pins.restore_capacity(turn_pins.record_capacity(None)), None)
        self.assertEqual(turn_pins.restore_capacity(turn_pins.record_capacity(20_000)), 20_000)
        for value in (None, "x", "-1", "true", "20000.0", "999999999"):
            with self.subTest(capacity=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_capacity({turn_pins.CAPACITY_METADATA: value})

    def test_the_model_pin_refuses_corrupt_state(self):
        self.assertEqual(
            turn_pins.restore_model(turn_pins.record_model("openai", "gpt-6.1-sol")), ("openai", "gpt-6.1-sol")
        )
        for value in (
            None,
            "x",
            "[]",
            '{"provider": "openai"}',
            '{"model": "m", "provider": 1}',
            '{"model":1,"provider":"openai"}',
        ):
            with self.subTest(model=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_model({turn_pins.MODEL_METADATA: value})


class PromptPinAndEndpointTests(unittest.TestCase):
    def test_a_pending_question_is_pinned_for_the_logical_turn(self):
        for question in (None, QUESTION):
            with self.subTest(question=question):
                self.assertEqual(turn_pins.restore_question(turn_pins.record_question(question)), question)
        pins = turn_pins.record_question(QUESTION)
        reordered = json.dumps(dict(reversed(list(QUESTION.items()))), ensure_ascii=False)
        for value in (None, "not json", '{"code": "x"}', reordered):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_question({**pins, turn_pins.QUESTION_METADATA: value})

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
            ("routine_question", {"code": "routine-no-room", "options": [], "value": None}),
            ("routine_mode", "yes"),
            ("routine_rerun", [{"assistant": "x"}]),
        ):
            with self.subTest(field=field):
                refused = api.post("/v1/turns", json=body(**{field: value}), headers=headers)
                self.assertIn(refused.status_code, {400, 422})
        self.assertEqual(api.post("/v1/routine-compile", json={}, headers=headers).status_code, 404)


if __name__ == "__main__":
    unittest.main()

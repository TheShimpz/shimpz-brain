"""Routines: the chat agent runs the work once and records it; Brain only checks the closed record (ADR-0101)."""

import dataclasses
import json
import unittest
from unittest import mock

import agent_runtime
import runtime_api
import test_agent_runtime
import test_attachments
import tool_refusal
import turn_pins
import turn_prompt
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import RecordingToolAwareFakeModel, ToolAwareFakeModel, context
from test_runtime_api import TOKEN, body
from tool_fake import FixedFactory, system_text
from turn_envelope import envelope

import routine

ACTION_TOOL = agent_runtime.tool_name("hello-pulse", "hello")
MESSAGE = "Cria uma rotina: a cada 30 segundos liste os registros DNS de shimpz.com"
LISTED = {
    "routine_id": "a" * 32,
    "name": "Resumo diário",
    "schedule": {"kind": "daily", "time": "08:00"},
    "timezone": "America/Sao_Paulo",
    "timezone_source": "browser",
    "revision": 2,
    "daily_steps": 1,
    "output": {"mode": "show"},
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
        "description": "Lista as zonas da sua Cloudflare e lê os registros DNS de shimpz.com.",
        "replaces": None,
        "reply": REPLY,
    }
    return {**fields, **changes}


def _call(call_id: str = "r1", **changes) -> dict:
    return {"name": routine.TOOL_NAME, "args": _args(**changes), "id": call_id, "type": "tool_call"}


def _action(call_id: str = "a1") -> dict:
    return {"name": ACTION_TOOL, "args": {}, "id": call_id, "type": "tool_call"}


# The schedule Team read from the person's "a cada 30 segundos".
EVERY_30_SECONDS = {"kind": "continuous", "gap": 30}
# The question rerun work comes beside.
RERUN_QUESTION = {"code": "routine-work-rerun", "options": [], "value": None}


def _chat(routines=(), schedule=EVERY_30_SECONDS):
    # A schedule the person states makes the chat about a Routine, as Team reads it.
    return dataclasses.replace(
        context(),
        memories=(),
        routines=tuple(routines),
        locale="pt",
        routine_mode=schedule is not None,
        routine_schedule=schedule,
    )


def _wire(**changes) -> dict:
    fields = {
        "name": "DNS de shimpz.com",
        "description": "Lista as zonas da sua Cloudflare e lê os registros DNS de shimpz.com.",
        "replaces": None,
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
        self.assertEqual(routine.SCHEMA["required"], ["op", "name", "description", "replaces", "reply"])
        self.assertFalse(routine.SCHEMA["additionalProperties"])
        with self.assertRaises(routine.RoutineContractError):
            tool.func(**_args())


class RecordTests(unittest.TestCase):
    def test_a_valid_call_records_exactly_its_closed_outcome(self):
        chat = _chat([LISTED])
        outcome = routine.record(_args(), chat)
        self.assertEqual(outcome, {"routine": _wire(), "reply": REPLY})
        changed = routine.record(_args(replaces="a" * 32), chat)
        self.assertEqual(changed["routine"], _wire(replaces="a" * 32))

    def test_the_schedule_timezone_and_output_are_never_the_models(self):
        # Team derives all three from the person's own words (ADR-0101); the tool offers none of them.
        chat = _chat([LISTED])
        for member in ("schedule", "timezone", "output"):
            self.assertNotIn(member, routine.SCHEMA["properties"])
            with self.subTest(member=member):
                self.assertEqual(routine.record({**_args(), member: None}, chat), "invalid")

    def test_every_field_outside_its_closed_shape_is_refused_by_name(self):
        chat = _chat([LISTED])
        cases = (
            (None, "invalid"),
            ({**_args(), "unexpected": "x"}, "invalid"),
            ({key: item for key, item in _args().items() if key != "reply"}, "invalid"),
            (_args(op="create"), "invalid"),
            (_args(name=""), "name"),
            (_args(name="a\nb"), "name"),
            (_args(name="x" * 81), "name"),
            (_args(description=""), "description"),
            (_args(description="Lista as zonas.\nDepois lê o DNS."), "description"),
            (_args(description="x" * 601), "description"),
            (_args(description=None), "description"),
            (_args(replaces="b" * 32), "replaces"),
            (_args(reply="   "), "reply"),
            (_args(reply="a\x00b"), "reply"),
            (_args(reply="a\x01b"), "reply"),
            (_args(reply="a\x7fb"), "reply"),
            (_args(reply="x" * (routine.MAX_REPLY_CHARS + 1)), "reply"),
            (_args(reply=1), "reply"),
        )
        for arguments, refused in cases:
            with self.subTest(refused=refused, arguments=arguments):
                self.assertEqual(routine.record(arguments, chat), refused)
                self.assertTrue(routine._CORRECTIONS[refused].startswith("Not done: nothing was recorded."))

    def test_an_answer_that_ran_nothing_may_leave_its_reply_to_the_card(self) -> None:
        self.assertEqual(routine.record(_args(reply=""), _chat([LISTED]))["reply"], "")

    def test_a_review_reads_only_the_latest_routine_call(self):
        chat = _chat()
        self.assertIsNone(routine._review([], chat))
        self.assertIsNone(routine._review([HumanMessage(content=envelope(MESSAGE))], chat))
        self.assertIsNone(routine._review([AIMessage(content="", tool_calls=[_action()])], chat))
        self.assertEqual(routine._review([AIMessage(content="", tool_calls=[_call(), _action()])], chat), "mixed")
        self.assertEqual(routine._review([AIMessage(content="", tool_calls=[_call()])], chat)["reply"], REPLY)
        broken = {"name": ACTION_TOOL, "args": "{bad", "id": "bad", "error": "invalid", "type": "invalid_tool_call"}
        with self.assertRaises(tool_refusal.UnanswerableToolCallError):
            routine._review([AIMessage(content="", tool_calls=[_call()], invalid_tool_calls=[broken])], chat)

    def test_only_a_well_formed_ending_is_read_back(self):
        content = json.dumps({"routine": {"name": "DNS"}, "reply": "Ok."})
        self.assertEqual(
            routine.recorded([ToolMessage(content=content, tool_call_id="r1", name=routine.TOOL_NAME)]),
            ("Ok.", {"name": "DNS"}),
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


def _routine_pins(question=None, mode=False, rerun=None, schedule=None, output=None) -> dict[str, str]:
    return turn_pins.record_routine(question, mode, rerun, schedule, output)


def _runtime(*responses):
    model = RecordingToolAwareFakeModel(responses=list(responses))
    return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=FixedFactory(lambda _config: model)), model


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
        self.assertEqual(finished.routine, _wire())
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
        content = json.dumps({"routine": _wire(), "reply": "Ok."})
        tool = ToolMessage(content=content, tool_call_id="r1", name=routine.TOOL_NAME)
        agent = mock.Mock(update_state=mock.Mock(side_effect=RuntimeError("disk")))
        runtime, _model = self._runtime()
        with self.assertRaises(agent_runtime.RuntimeStateError):
            runtime._finish_routine(agent, _chat(), {"messages": [tool]})
        self.assertIsNone(runtime._finish_routine(agent, _chat(), {"messages": [tool], "__interrupt__": [1]}))

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
    def test_only_a_recording_chat_turn_carries_a_valid_question(self):
        asked = dataclasses.replace(_chat(), routine_question=QUESTION)
        self.assertEqual(asked.routine_question, QUESTION)
        for routines, question in ((None, QUESTION), ((), {**QUESTION, "code": "x"})):
            with (
                self.subTest(question=question),
                self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Routine question"),
            ):
                dataclasses.replace(context(), routines=routines, routine_question=question)

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
        chat = dataclasses.replace(_chat(), routine_question=RERUN_QUESTION, routine_rerun=RERUN)
        self.assertEqual((chat.routine_mode, chat.routine_rerun), (True, RERUN))
        for changes, error in (
            ({"routine_mode": True}, "invalid Routine mode"),
            ({"routine_mode": 1, "routines": ()}, "invalid Routine mode"),
            ({"routine_rerun": RERUN}, "invalid Routine rerun"),
            ({"routine_rerun": ({"assistant": "x"},), "routines": ()}, "invalid Routine rerun"),
        ):
            routines = changes.pop("routines", None)
            with self.subTest(changes=changes), self.assertRaisesRegex(agent_runtime.RuntimeContractError, error):
                dataclasses.replace(context(), routines=routines, **changes)

    def test_the_prompt_adds_the_routine_mode_and_rerun_sections_only_when_sent(self):
        runtime, model = _runtime(AIMessage(content="Ok."))
        runtime.start(
            dataclasses.replace(_chat(), routine_question=RERUN_QUESTION, routine_rerun=RERUN), envelope("Sim")
        )
        system = system_text(model.seen_messages[-1])
        self.assertIn("If this message asks for a Routine", system)
        self.assertIn("what to do with each run's result from the user's words and asks about them itself", system)
        self.assertIn("when none is listed, look the value up with an Action that returns it", system)
        self.assertIn("The Team needs this work run again before it can record", system)
        self.assertIn(json.dumps(list(RERUN), ensure_ascii=False), system)
        runtime, model = _runtime(AIMessage(content="Ok."))
        runtime.start(_chat(schedule=None), envelope("Oi"))
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
                pins = _routine_pins(mode=mode, rerun=rerun)
                self.assertEqual(turn_pins.restore_routine(pins)[1:3], (mode, rerun))
        pins = _routine_pins(mode=True, rerun=RERUN)
        for key, value in (
            (turn_pins.MODE_METADATA, None),
            (turn_pins.MODE_METADATA, "1"),
            (turn_pins.RERUN_METADATA, None),
            (turn_pins.RERUN_METADATA, "not json"),
            (turn_pins.RERUN_METADATA, '[{"assistant": "x"}]'),
            (turn_pins.RERUN_METADATA, json.dumps([dict(reversed(list(RERUN[0].items())))], ensure_ascii=False)),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_routine({**pins, key: value})
        # A mode that is not hashable is still only corrupt state.
        with self.assertRaises(turn_pins.PinError):
            turn_pins.restore_routine({**pins, turn_pins.MODE_METADATA: []})

    def test_the_persons_schedule_is_pinned_for_the_logical_turn(self):
        for schedule in (None, EVERY_30_SECONDS):
            with self.subTest(schedule=schedule):
                self.assertEqual(turn_pins.restore_routine(_routine_pins(schedule=schedule))[3], schedule)
        for value in (
            None,
            "not json",
            '{"kind": "continuous", "gap": 30}',
            '{"gap":4,"kind":"continuous"}',
        ):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_routine({**_routine_pins(), turn_pins.SCHEDULE_METADATA: value})
        with self.assertRaises(agent_runtime.RuntimeContractError):
            dataclasses.replace(_chat(), routines=None, routine_schedule=EVERY_30_SECONDS)

    def test_the_persons_output_is_pinned_for_the_logical_turn(self):
        for output in (None, "changes"):
            with self.subTest(output=output):
                self.assertEqual(turn_pins.restore_routine(_routine_pins(output=output))[4], output)
        # No output read from the person is pinned as JSON null; a missing pin is corrupt state.
        self.assertEqual(_routine_pins()[turn_pins.OUTPUT_METADATA], "null")
        pins = _routine_pins()
        del pins[turn_pins.OUTPUT_METADATA]
        corrupt = ("not json", '"sometimes"', "1", " null")
        for metadata in (pins, *({**_routine_pins(), turn_pins.OUTPUT_METADATA: value} for value in corrupt)):
            with self.subTest(metadata=metadata), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_routine(metadata)
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Routine output"):
            dataclasses.replace(_chat(schedule=None), routines=None, routine_output="changes")
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Routine output"):
            dataclasses.replace(_chat(), routine_output="sometimes")

    def test_a_settings_answer_records_without_running_the_work_again(self):
        asked = {"code": "routine-output-unstated", "options": [], "value": None}
        runtime, _model = GraphTests._runtime(None, AIMessage(content="", tool_calls=[_call(reply="")]))
        chat = dataclasses.replace(
            _chat([LISTED]), routine_question=asked, routine_output="changes", routine_settings_answer=True
        )
        finished = runtime.start(chat, envelope("Mostrar somente quando mudar"))
        # The record of the work that already ran ends the turn; the card shows Team's own reading of the answer.
        self.assertEqual((finished.status, finished.reply, finished.routine), ("completed", "", _wire()))
        # The thread keeps the description the card shows, never an empty assistant message.
        state = runtime._checkpointer.get({"configurable": {"thread_id": context().thread_id}})
        self.assertEqual(state["channel_values"]["messages"][-1].content, _wire()["description"])
        # The person's bare output choice is offered no Action at all, so its work can never run again, and the Routine
        # mode guidance never tells it to run the work; an answer that also asks for other work keeps every Action.
        self.assertEqual(agent_runtime._turn_surface(chat)[0], ())
        records = turn_prompt.system_prompt(dataclasses.replace(chat, routine_mode=True))
        self.assertNotIn("run the work once", records)
        # The Actions withheld from it are no limit of the Team's, so it is never told the Team has none.
        self.assertNotIn("no enabled Assistants", agent_runtime._turn_surface(chat)[3])
        mixed = dataclasses.replace(chat, routine_settings_answer=False)
        target = dataclasses.replace(mixed, routine_question=QUESTION, routine_mode=True)
        self.assertIn("run the work once", turn_prompt.system_prompt(target))
        # A typed settings answer runs only work it changes, while a target answer runs the work again.
        self.assertNotIn("run the work once", turn_prompt.system_prompt(dataclasses.replace(mixed, routine_mode=True)))
        self.assertEqual(agent_runtime._turn_surface(mixed)[0], agent_runtime._turn_surface(_chat([LISTED]))[0])
        self.assertTrue(agent_runtime._turn_surface(mixed)[0])
        # Team marks only an answer to its own schedule, over-budget, or output question whose reading it took.
        schedule = {**asked, "code": "routine-schedule-unstated"}
        over = {"code": "routine-interval-over-budget", "options": [], "value": 120}
        for question in (schedule, over):
            with self.subTest(question=question):
                self.assertTrue(dataclasses.replace(chat, routine_question=question).routine_settings_answer)
        for changes in (
            {"routine_output": None},
            {"routine_question": QUESTION},
            {"routine_question": None},
            {"routine_question": schedule, "routine_schedule": None},
            {"routine_question": over, "routine_schedule": None},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Routine settings answer"),
            ):
                dataclasses.replace(chat, **changes)
        # The bare output choice is asked for a record alone, a typed answer to the same question to run only work it
        # changes, and an answer to a target question to run the work again: each is told something of its own.
        prompts = {
            turn_prompt.system_prompt(dataclasses.replace(base, routine_question=question)).replace(
                json.dumps(question, ensure_ascii=False), ""
            )
            for base, question in ((chat, asked), (mixed, asked), (mixed, QUESTION))
        }
        self.assertEqual(len(prompts), 3)

    def test_a_resumed_turn_keeps_the_mode_and_rerun_its_start_pinned(self):
        runtime, model = GraphTests._runtime(
            None, AIMessage(content="", tool_calls=[_action()]), AIMessage(content="Ok.")
        )
        started = dataclasses.replace(_chat([LISTED]), routine_question=RERUN_QUESTION, routine_rerun=RERUN)
        suspended = runtime.start(started, envelope("Sim"))
        runtime.resume(_chat([LISTED]), {suspended.actions[0].interrupt_id: {"records": ["A"]}})
        system = system_text(model.seen_messages[-1])
        self.assertIn("If this message asks for a Routine", system)
        self.assertIn("The Team needs this work run again", system)


LIST_TOOL = agent_runtime.tool_name("hello-pulse", "list")
# Work whose name only a lookup returns: hello's name is fresh, and list is where Team found it.
FRESH = (
    {
        "assistant": "hello-pulse",
        "action": "hello",
        "count": 1,
        "inputs": [
            {
                "member": "name",
                "kind": "fresh",
                "value": None,
                "chosen": False,
                "source": {"assistant": "hello-pulse", "action": "list"},
            }
        ],
    },
)
ZONE = "023e105f4ecef8ad9ca31a8372d0c353"


def _hello(name: object, call_id: str = "h1") -> dict:
    return {"name": ACTION_TOOL, "args": {"name": name}, "id": call_id, "type": "tool_call"}


def _turn(message: str, *later) -> list:
    """An earlier exchange whose result held the zone, then this turn's message and what followed it."""
    earlier = [
        HumanMessage(envelope("Liste a zona"), id="h0"),
        ToolMessage(json.dumps({"zones": [{"id": ZONE}]}), tool_call_id="old", name=LIST_TOOL),
    ]
    return [*earlier, HumanMessage(envelope(message), id="h1"), *later]


class RerunGuardTests(unittest.TestCase):
    def _review(self, messages, checked=None):
        return routine._rerun_review(
            messages, checked=checked or {ACTION_TOOL: frozenset({"name"})}, actions=frozenset({ACTION_TOOL, LIST_TOOL})
        )

    def test_a_fresh_input_must_come_from_a_result_of_this_turn(self):
        remembered = AIMessage("", tool_calls=[_hello(ZONE)])
        # A value remembered from the conversation, even one an earlier turn's result held, is refused.
        self.assertEqual(self._review(_turn("pode buscar de novo", remembered)), "fresh")
        for result in (
            json.dumps({"zones": [{"id": "other"}, {"id": ZONE, "tags": ["a"]}]}),
            ZONE,
        ):
            with self.subTest(result=result):
                looked_up = ToolMessage(result, tool_call_id="l1", name=LIST_TOOL)
                self.assertIsNone(self._review(_turn("pode buscar de novo", looked_up, remembered)))
        # A correction or a Brain tool's message is no result, whatever it says.
        for name in (routine.TOOL_NAME, None):
            with self.subTest(name=name):
                other = ToolMessage(ZONE, tool_call_id="x1", name=name)
                self.assertEqual(self._review(_turn("pode buscar de novo", other, remembered)), "fresh")
        # Equal JSON values match as values: an integral float is its integer.
        number = AIMessage("", tool_calls=[_hello(424242)])
        found = ToolMessage(json.dumps({"id": 424242.0}), tool_call_id="l1", name=LIST_TOOL)
        self.assertIsNone(self._review(_turn("sim", found, number)))
        self.assertEqual(self._review(_turn("sim", number)), "fresh")

    def test_a_value_the_person_writes_now_needs_no_lookup(self):
        for value, message in ((ZONE, f"use a zona {ZONE}"), (424242, "a conta 424242")):
            with self.subTest(value=value):
                call = AIMessage("", tool_calls=[_hello(value)])
                self.assertIsNone(self._review(_turn(message, call)))
        # Only what the person wrote counts, never Team's composed question.
        asked = f"pode buscar\n\nPergunta: {ZONE}?\nResposta: sim"
        self.assertEqual(self._review(_turn(asked, AIMessage("", tool_calls=[_hello(ZONE)]))), "fresh")

    def test_only_an_identifier_needs_a_source(self):
        # Team keeps any other value as the agent's own choice, such as the term a helper lookup sends.
        for value in (
            "me",
            "my zone list",
            42424,
            4242.5,
            True,
            {"id": ZONE},
            ["a"],
            "2026-10-05",
            "news-05/10",
            "10月15日号",
        ):
            with self.subTest(value=value):
                self.assertIsNone(self._review(_turn("sim", AIMessage("", tool_calls=[_hello(value)]))))
        for value in ("zone-1", -424242, "123e4567-e89b-12d3-a456-426614174000"):
            with self.subTest(value=value):
                self.assertEqual(self._review(_turn("sim", AIMessage("", tool_calls=[_hello(value)]))), "fresh")

    def test_only_a_fresh_member_of_a_checked_action_in_an_answerable_response_is_reviewed(self):
        self.assertIsNone(self._review([]))
        self.assertIsNone(self._review(_turn("sim", AIMessage("Pronto."))))
        self.assertIsNone(self._review(_turn("sim", AIMessage("", tool_calls=[{**_hello(ZONE), "args": {}}]))))
        self.assertIsNone(self._review(_turn("sim", AIMessage("", tool_calls=[{**_hello(ZONE), "name": LIST_TOOL}]))))
        unparsable = AIMessage(
            "",
            tool_calls=[_hello(ZONE)],
            invalid_tool_calls=[{"name": ACTION_TOOL, "args": "{", "id": "x", "error": None}],
        )
        self.assertIsNone(self._review(_turn("sim", unparsable)))
        # An Action checks only members every slot of it marks fresh.
        kept = {**FRESH[0]["inputs"][0], "kind": "value", "value": '"ana"', "source": None}
        twice = (*FRESH, {**FRESH[0], "inputs": [kept]}, {**FRESH[0], "action": "list", "inputs": [kept]})
        guard = routine.rerun_guard(twice, agent_runtime.tool_name, frozenset())
        self.assertEqual(guard.review.keywords["checked"], {})
        guard = routine.rerun_guard(FRESH, agent_runtime.tool_name, frozenset())
        self.assertEqual(guard.review.keywords["checked"], {ACTION_TOOL: frozenset({"name"})})

    def test_a_refused_rerun_call_never_runs_and_the_looked_up_value_does(self):
        runtime, model = GraphTests._runtime(
            None,
            AIMessage("", tool_calls=[_hello(ZONE)]),
            AIMessage("", tool_calls=[{"name": LIST_TOOL, "args": {}, "id": "l1", "type": "tool_call"}]),
            AIMessage("", tool_calls=[_hello(ZONE, "h2")]),
        )
        pulse = test_agent_runtime.assistant(
            "hello-pulse", test_agent_runtime.action("hello"), test_agent_runtime.action("list")
        )
        chat = dataclasses.replace(
            _chat([LISTED]), assistants=(pulse,), routine_question=RERUN_QUESTION, routine_rerun=FRESH
        )
        listed = runtime.start(chat, envelope("pode buscar de novo"))
        self.assertEqual([item.action for item in listed.actions], ["list"])
        corrections = {message.content for message in model.seen_messages[-1] if isinstance(message, ToolMessage)}
        self.assertEqual(corrections, {routine._RERUN_CORRECTION})
        rerun = runtime.resume(chat, {listed.actions[0].interrupt_id: {"zones": [{"id": ZONE}]}})
        self.assertEqual([(item.action, item.input) for item in rerun.actions], [("hello", {"name": ZONE})])


class ContinuationFrameTests(unittest.TestCase):
    """A resume as Team sends it: no listing, no mode, the Team's configured model (ADR-0101)."""

    @staticmethod
    def _team_resume(model: str = "gpt-6-luna") -> agent_runtime.TurnContext:
        provider = dataclasses.replace(context().provider, model=model)
        return dataclasses.replace(context(), provider=provider, memories=(), routines=None, locale="pt")

    def test_a_team_continuation_restores_the_listing_its_start_pinned(self):
        seen: list[agent_runtime.ProviderConfig] = []
        RecordingToolAwareFakeModel.seen_messages = []
        model = RecordingToolAwareFakeModel(
            responses=[AIMessage(content="", tool_calls=[_action()]), AIMessage(content="", tool_calls=[_call()])]
        )

        def factory(config):
            seen.append(config)
            return model

        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=FixedFactory(factory))
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
                self.assertEqual(turn_pins.restore_routine(_routine_pins(question=question))[0], question)
        pins = _routine_pins(question=QUESTION)
        reordered = json.dumps(dict(reversed(list(QUESTION.items()))), ensure_ascii=False)
        for value in (None, "not json", '{"code": "x"}', reordered):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_routine({**pins, turn_pins.QUESTION_METADATA: value})

    def test_routines_are_pinned_for_the_logical_turn(self):
        date = turn_prompt.today()
        pins = turn_pins.record(date, (), None, (LISTED,))
        self.assertEqual(turn_pins.restore(pins), (date, (), None, (LISTED,)))
        reordered = json.dumps([dict(reversed(list(LISTED.items())))], ensure_ascii=False)
        for value in ("[{}]", "[1]", '{"a": 1}', reordered):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore({**pins, turn_pins.ROUTINES_METADATA: value})

    def test_the_context_refuses_invalid_routines(self):
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid routines"):
            dataclasses.replace(context(), routines=({"routine_id": "x"},))

    def test_the_context_admits_the_recording_fields_only_together_as_team_sends_them(self):
        recorded = dataclasses.replace(
            _chat([LISTED]), routine_question=RERUN_QUESTION, routine_rerun=RERUN, routine_output="changes"
        )
        self.assertEqual((recorded.routine_question, recorded.routine_rerun), (RERUN_QUESTION, RERUN))
        # Team reads an output choice outside Routine mode too.
        self.assertEqual(dataclasses.replace(_chat(schedule=None), routine_output="changes").routine_output, "changes")
        attachment = test_attachments.turn_context("openai", "gpt-6-luna", test_attachments.team_attachment("text"))
        for changes, error in (
            ({"routine_mode": False}, "invalid Routine question"),
            ({"routine_mode": False, "routine_question": None, "routine_rerun": None}, "invalid Routine schedule"),
            ({"routine_question": QUESTION}, "invalid Routine rerun"),
            ({"routine_question": None}, "invalid Routine rerun"),
            ({"attachments": attachment.attachments}, "invalid routines"),
        ):
            with (
                self.subTest(changes=sorted(changes)),
                self.assertRaisesRegex(agent_runtime.RuntimeContractError, error),
            ):
                dataclasses.replace(recorded, routine_schedule=EVERY_30_SECONDS, **changes)

    def test_the_turn_endpoint_carries_routines_and_returns_the_record(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        model = ToolAwareFakeModel(responses=[AIMessage(content="", tool_calls=[_call()])])
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=FixedFactory(lambda _config: model))
        api = TestClient(runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN))
        response = api.post(
            "/v1/turns",
            json=body(
                message=envelope(MESSAGE), routines=[LISTED], routine_mode=True, routine_schedule=EVERY_30_SECONDS
            ),
            headers=headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["routine"], _wire())
        self.assertEqual(response.json()["reply"], REPLY)
        for field, value in (
            ("routines", [{"routine_id": "x"}]),
            ("routine_question", {"code": "unsupported", "options": [], "value": None}),
            ("routine_mode", "yes"),
            ("routine_rerun", [{"assistant": "x"}]),
            ("routine_schedule", {"kind": "continuous", "gap": 4}),
            ("routine_output", "sometimes"),
        ):
            with self.subTest(field=field):
                refused = api.post("/v1/turns", json=body(**{field: value}), headers=headers)
                self.assertIn(refused.status_code, {400, 422})


if __name__ == "__main__":
    unittest.main()

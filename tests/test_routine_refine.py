"""Refining the Routine a person's send selects: its context, the refine call, and its guard (ADR-0122)."""

import dataclasses
import json
import unittest

import agent_runtime
import runtime_api
import turn_pins
import turn_prompt
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import ToolAwareFakeModel, context
from test_routine import LISTED, MESSAGE, _action, _args, _chat, _runtime
from test_runtime_api import TOKEN, body
from tool_fake import FixedFactory, system_text
from turn_envelope import envelope

import routine

SELECTED = {
    "routine_id": "a" * 32,
    "name": "Resumo diário",
    "description": "Cumprimenta a Ana todo dia.",
    "revision": 2,
    "output": {"mode": "show", "step": "s2"},
    "steps": [
        {
            "id": "s1",
            "position": 1,
            "assistant": "hello-pulse",
            "action": "hello",
            "read_only": True,
            "inputs": [{"member": "name", "source": "literal", "value": '"ana"'}],
            "stored_inputs": [],
        },
        {
            "id": "s2",
            "position": 2,
            "assistant": "hello-pulse",
            "action": "hello",
            "read_only": True,
            "inputs": [
                {
                    "member": "name",
                    "source": "step_output",
                    "step": 1,
                    "pointer": "/greeting",
                    "where": None,
                    "item": None,
                }
            ],
            "stored_inputs": [],
        },
    ],
    "last_run": {"revision": 2, "steps": [{"position": 1, "status": "done"}, {"position": 2, "status": "failed"}]},
}
DESCRIPTION = "Cumprimenta a Ana e o Bruno todo dia."


def _selected_chat(selected=SELECTED):
    return dataclasses.replace(_chat([LISTED], schedule=None), routine_mode=True, routine_selected=selected)


def _refine(**changes) -> dict:
    fields = {
        "op": "refine",
        "name": None,
        "description": DESCRIPTION,
        "replaces": None,
        "routine_id": "a" * 32,
        "base_revision": 2,
        "edits": [{"kind": "insert", "step": "s2", "calls": ["c1"]}],
        "reply": "",
    }
    return {**fields, **changes}


def _labeled(handle: str = "c1") -> list:
    """This turn's messages: the person's, then one Action result Team named by its handle."""
    content = json.dumps({"call": handle, "result": {"greeting": "oi"}})
    return [HumanMessage(content="m"), ToolMessage(content=content, tool_call_id="a1", name="x")]


class ContextTests(unittest.TestCase):
    def test_a_selection_comes_only_in_routine_mode_beside_the_listing_in_its_canonical_form(self):
        self.assertEqual(_selected_chat().routine_selected, SELECTED)
        for broken in (
            lambda: dataclasses.replace(_chat([LISTED], schedule=None), routine_selected=SELECTED),
            lambda: dataclasses.replace(context(), routine_mode=False, routine_selected=SELECTED),
            lambda: _selected_chat({**SELECTED, "revision": 0}),
            # Only a Routine the listing shows, at the revision it shows, is selected.
            lambda: _selected_chat({**SELECTED, "revision": 3}),
            lambda: _selected_chat({**SELECTED, "routine_id": "b" * 32}),
        ):
            with self.subTest(broken=broken), self.assertRaises(agent_runtime.RuntimeContractError):
                broken()

    def test_the_selection_is_pinned_for_the_logical_turn(self):
        pins = turn_pins.record_routine(None, True, None, None, None, SELECTED)
        self.assertEqual(turn_pins.restore_routine(pins)[5], SELECTED)
        for value in (None, "[]", json.dumps({**SELECTED, "revision": 0})):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_routine({**pins, turn_pins.SELECTED_METADATA: value})

    def test_the_prompt_shows_the_selection_and_otherwise_asks_which_routine(self):
        selected = turn_prompt.system_prompt(_selected_chat())
        unselected = turn_prompt.system_prompt(_chat([LISTED]))
        self.assertIn(json.dumps(SELECTED, ensure_ascii=False), selected)
        self.assertIn("op refine", selected)
        self.assertNotIn("op refine", unselected)
        self.assertIn("ask which listed Routine they mean", unselected)
        # The Routine-mode guidance changes the selected Routine by refining it, never by running it all again.
        self.assertIn("To change the selected Routine, refine it as described above. For a new Routine, then", selected)
        self.assertNotIn("To change the selected Routine", unselected)

    def test_the_tool_offers_refine_only_beside_a_selection(self):
        self.assertEqual(routine.offered(False), (routine.SCHEMA, routine.DESCRIPTION))
        self.assertEqual(routine.tool(True).args_schema, routine.REFINE_SCHEMA)
        self.assertEqual(routine.REFINE_SCHEMA["required"][-1], "reply")
        self.assertEqual(set(routine.REFINE_SCHEMA["required"]), set(routine.REFINE_SCHEMA["properties"]))


class GuardTests(unittest.TestCase):
    def test_a_refinement_of_the_selection_names_its_edits_and_this_turns_calls(self):
        handles = routine.call_handles(_labeled())
        self.assertEqual(handles, frozenset({"c1"}))
        outcome = routine.record(_refine(), _selected_chat(), handles)
        self.assertEqual(
            outcome["routine"],
            {
                "routine_id": "a" * 32,
                "base_revision": 2,
                "description": DESCRIPTION,
                "edits": [{"kind": "insert", "step": "s2", "calls": ["c1"]}],
            },
        )
        self.assertEqual(outcome["reply"], "")
        described = routine.record(_refine(edits=[]), _selected_chat(), frozenset())
        self.assertEqual(described["routine"]["edits"], [])

    def test_each_invalid_refinement_gets_its_closed_correction(self):
        cases = (
            (_refine(routine_id="b" * 32), "refine"),
            (_refine(base_revision=1), "refine"),
            (_refine(base_revision=True), "refine"),
            (_refine(name="Outro"), "refine"),
            (_refine(replaces="a" * 32), "refine"),
            (_refine(edits=[{"kind": "remove", "step": "s9", "calls": []}]), "refine"),
            (_refine(edits=[{"kind": "insert", "step": "s1", "calls": []}]), "refine"),
            (_refine(description=" "), "description"),
            (_refine(edits=[{"kind": "insert", "step": "s2", "calls": ["c2"]}]), "calls"),
            (_refine(edits=[{"kind": "remove", "step": "s1", "calls": []}]), "dependency"),
            (_refine(edits=[{"kind": "replace", "step": "s1", "calls": ["c1"]}]), "dependency"),
            (_refine(edits=[], description=SELECTED["description"]), "empty"),
            (_refine(reply=" "), "reply"),
        )
        for arguments, refused in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(routine.record(arguments, _selected_chat(), frozenset({"c1"})), refused)
                self.assertTrue(routine._CORRECTIONS[refused].startswith("Not done: nothing was recorded."))
        both = _refine(
            edits=[{"kind": "remove", "step": "s1", "calls": []}, {"kind": "remove", "step": "s2", "calls": []}]
        )
        self.assertEqual(routine.record(both, _selected_chat(), frozenset())["routine"]["edits"], both["edits"])

    def test_a_record_beside_a_selection_leaves_every_refine_member_null(self):
        record = {**_args(), "routine_id": None, "base_revision": None, "edits": None}
        self.assertEqual(routine.record(record, _selected_chat())["routine"]["name"], _args()["name"])
        self.assertEqual(routine.record({**record, "edits": []}, _selected_chat()), "refine")
        self.assertEqual(routine.record({**record, "op": "other"}, _selected_chat()), "invalid")
        self.assertEqual(routine.record(_refine(), _chat([LISTED])), "invalid")
        self.assertEqual(routine.record(_args(), _selected_chat()), "refine")

    def test_only_this_turns_named_results_are_handles(self):
        earlier = [
            HumanMessage(content="before"),
            ToolMessage(content=json.dumps({"call": "c9", "result": 1}), tool_call_id="x", name="x"),
        ]
        others = [
            ToolMessage(content=json.dumps({"call": "c1"}), tool_call_id="b", name="x"),
            ToolMessage(content=json.dumps({"call": "x1", "result": 1}), tool_call_id="c", name="x"),
            ToolMessage(content="not json", tool_call_id="d", name="x"),
            AIMessage(content="text"),
        ]
        self.assertEqual(routine.call_handles([*earlier, *_labeled("c2"), *others]), frozenset({"c2"}))
        self.assertEqual(routine.call_handles([]), frozenset())


class GraphTests(unittest.TestCase):
    def test_the_agent_runs_the_new_call_and_refines_by_its_handle(self):
        refine = {"name": routine.TOOL_NAME, "args": _refine(), "id": "r1", "type": "tool_call"}
        runtime, model = _runtime(
            AIMessage(content="", tool_calls=[_action()]), AIMessage(content="", tool_calls=[refine])
        )
        chat = _selected_chat()
        suspended = runtime.start(chat, envelope(MESSAGE))
        finished = runtime.resume(
            chat, {suspended.actions[0].interrupt_id: {"call": "c1", "result": {"greeting": "oi"}}}
        )
        self.assertEqual((finished.status, finished.routine["edits"][0]["calls"]), ("completed", ["c1"]))
        self.assertIn("op refine", system_text(model.seen_messages[-1]))

    def test_the_turn_endpoint_carries_the_selection(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        refine = {"name": routine.TOOL_NAME, "args": _refine(edits=[]), "id": "r1", "type": "tool_call"}
        model = ToolAwareFakeModel(responses=[AIMessage(content="", tool_calls=[refine])])
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=FixedFactory(lambda _config: model))
        api = TestClient(runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN))
        given = body(message=envelope(MESSAGE), routines=[LISTED], routine_mode=True, routine_selected=SELECTED)
        response = api.post("/v1/turns", json=given, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["routine"]["description"], DESCRIPTION)
        refused = api.post("/v1/turns", json={**given, "routine_selected": {"routine_id": "x"}}, headers=headers)
        self.assertIn(refused.status_code, {400, 422})


if __name__ == "__main__":
    unittest.main()

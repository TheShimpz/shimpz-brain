"""A Routine at scale in the Brain: one Action many times, hundreds of steps, within what Team admits.

ADR-0092 amendment, 2026-10-05 (scale): Team leaves each new Routine a share of its daily Action steps, and the
compiler only proposes a Routine that share holds; Team rechecks at commit.
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from unittest import mock

import agent_runtime
import context_budget
import provider_client
import turn_prompt
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import RecordingToolAwareFakeModel, context
from test_memory import envelope
from test_routine import (
    CONTRACTS,
    LISTED,
    MESSAGE,
    _asking,
    _call,
    _chat,
    _compiled,
    _compiling,
    _origin,
    _said,
    _source,
)

import routine

NAMES = tuple(f"n{index:03d}" for index in range(1, 121))
REQUEST = "Toda segunda às 9h, diga olá para cada um"
MANY = REQUEST + ":\n" + ", ".join(NAMES)


def _many(count: int = len(NAMES)) -> routine.Compiled:
    """One Action once per name the person listed, each step with its own value."""
    steps = [
        routine.Step(
            id=f"greet-{name}",
            assistant="hello-pulse",
            action="hello",
            inputs=[_source(value_json=json.dumps(name), origins=[_origin(name)])],
        )
        for name in NAMES[:count]
    ]
    output = routine.Output(mode="show", step=steps[0].id, instruction="diga")
    return _compiled(request=REQUEST, steps=steps, output=output)


def _turn(chat, *responses):
    RecordingToolAwareFakeModel.seen_messages = []
    model = RecordingToolAwareFakeModel(responses=list(responses))
    runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
    return runtime.start(chat, envelope(MESSAGE)), model


def _corrections(model) -> list[str]:
    return [
        item.content
        for item in model.seen_messages[-1]
        if isinstance(item, ToolMessage) and item.name == routine.TOOL_NAME
    ]


class ScaleTests(unittest.TestCase):
    def test_one_action_once_per_named_value_compiles_into_a_plan_of_120_steps(self):
        wire = routine.change(_many(), _said(MANY), CONTRACTS, None)
        self.assertEqual(len(wire["steps"]), 120)
        self.assertEqual({step["action"] for step in wire["steps"]}, {"hello"})
        self.assertEqual([step["input"]["name"]["value"] for step in wire["steps"]], list(NAMES))
        source = routine.UserWords(MANY)
        outcome = routine._answer(_many(), source, _chat().assistants, None, 120)
        self.assertEqual(outcome["routine"]["steps"], wire["steps"])
        # Within the bound, and never past it.
        steps = [*_many().steps, *_many().steps, *_many().steps[:17]]
        renamed = [step.model_copy(update={"id": f"s{index}"}) for index, step in enumerate(steps)]
        self.assertEqual(len(renamed), routine.MAX_STEPS + 1)
        with self.assertRaises(routine.UnprovenError):
            routine.change(_many().model_copy(update={"steps": renamed}), _said(MANY), CONTRACTS, None)

    def test_the_context_carries_a_capacity_exactly_beside_routines(self):
        self.assertEqual(_chat().routine_capacity, 20_000)
        for routines, capacity in (((), None), (None, 1), ((), -1), ((), routine.MAX_DAILY_STEPS + 1), ((), True)):
            with (
                self.subTest(routines=routines, capacity=capacity),
                self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Routine capacity"),
            ):
                dataclasses.replace(context(), routines=routines, routine_capacity=capacity)
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: None)
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "invalid Routine capacity"):
            runtime.routine_compile(context().provider, MESSAGE, context().assistants, None, (), -1)

    def test_a_routine_over_the_daily_steps_left_is_refused_with_its_facts(self):
        source = routine.UserWords(MANY)
        assistants = _chat().assistants
        # Weekly runs once a day at most: 120 steps need 120 of the Team's daily steps.
        self.assertEqual(routine._answer(_many(), source, assistants, None, 119), ("budget", 120, 0))
        self.assertEqual(len(routine._answer(_many(3), source, assistants, None, 3)["routine"]["steps"]), 3)
        hourly = routine.Schedule(kind="hourly", every=2, time=None, weekday=None, day=None, gap=None, cap=None)
        self.assertEqual(
            routine._answer(_compiled(schedule=hourly), routine.UserWords(MESSAGE), assistants, None, 11),
            ("budget", 1, 11),
        )
        # A question whose candidate cannot run within the share is refused before it is asked.
        self.assertEqual(routine._answer(_asking(), routine.UserWords(MESSAGE), assistants, None, 0), ("budget", 1, 0))
        text = routine._correction(("budget", 120, 0))
        self.assertIn("its 120 steps exceed what is left", text)
        self.assertIn("at most 0 times a day", text)
        self.assertEqual(routine._correction("too-large"), routine._CORRECTIONS["too-large"])
        # Recriar answers the closed reason alone.
        recreated = routine.recompile(MESSAGE, assistants, "pt", lambda _prompt: _compiled(), (), 0)
        self.assertEqual(recreated, "budget")

    def test_the_turn_relays_the_budget_and_an_update_has_its_own_share_back(self):
        listed = {**LISTED, "daily_steps": 1}
        patch, prompts = _compiling(_compiled())
        with patch:
            result, model = _turn(
                dataclasses.replace(_chat([listed]), routine_capacity=0),
                AIMessage(content="", tool_calls=[_call()]),
                AIMessage(content="Nada foi criado."),
            )
        self.assertIsNone(result.routine)
        self.assertEqual(_corrections(model), [routine._correction(("budget", 1, 0))])
        self.assertIn("at most 0 Action steps a day", prompts[0])
        update = _compiled(
            steps=[routine.Step(id="greet", assistant="hello-pulse", action="hello", inputs=[_source(kind="kept")])]
        )
        patch, prompts = _compiling(update)
        with patch:
            result, _model = _turn(
                dataclasses.replace(_chat([listed]), routine_capacity=0),
                AIMessage(content="", tool_calls=[_call(op="update", routine_id=LISTED["routine_id"])]),
            )
        self.assertEqual((result.routine["op"], result.routine["routine_id"]), ("update", LISTED["routine_id"]))
        self.assertIn("at most 1 Action steps a day", prompts[0])

    def test_an_outcome_or_prompt_too_large_for_team_is_refused_never_cut(self):
        assistants = _chat().assistants
        with mock.patch.object(routine, "MAX_OUTCOME_BYTES", 64):
            self.assertEqual(routine._answer(_compiled(), routine.UserWords(MESSAGE), assistants, None, 1), "too-large")
        asked: list[str] = []
        with mock.patch.object(routine, "MAX_COMPILE_PROMPT_BYTES", 64):
            self.assertEqual(routine.recompile(MESSAGE, assistants, "pt", asked.append, (), 1), "too-large")
            patch, prompts = _compiling(_compiled())
            with patch:
                result, model = _turn(
                    _chat(), AIMessage(content="", tool_calls=[_call()]), AIMessage(content="Nada foi criado.")
                )
        self.assertEqual((asked, prompts, result.routine), ([], [], None))
        self.assertEqual(_corrections(model), [routine._CORRECTIONS["too-large"]])

    def test_a_prompt_fits_only_within_its_bytes_and_the_model_window(self):
        self.assertTrue(context_budget.fits("x" * 10, {}, 10))
        self.assertFalse(context_budget.fits("é" * 6, {}, 10))
        window = context_budget.MODEL_WINDOW_TOKENS - context_budget.OUTPUT_RESERVE_TOKENS
        wide = "x" * (window * context_budget.BYTES_PER_TOKEN)
        self.assertFalse(context_budget.fits(wide, {"type": "object"}, len(wide)))

    def test_the_compiler_reads_each_actions_output_schema_and_the_share_left(self):
        output_schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        action = dataclasses.replace(_chat().assistants[0].actions[0], output_schema=output_schema)
        assistant = dataclasses.replace(_chat().assistants[0], actions=(action,))
        prompt = routine._prompt(routine.UserWords(MESSAGE), (assistant,), None, "pt", capacity=345)
        self.assertIn(json.dumps(output_schema, ensure_ascii=False), prompt)
        self.assertIn("at most 345 Action steps a day", prompt)
        self.assertIn(f"at most {routine.MAX_STEPS} listed Actions", prompt)
        # The chat agent's own tools never carry an output schema.
        result, _model = _turn(_chat(), AIMessage(content="Olá."))
        self.assertEqual(result.reply, "Olá.")
        self.assertNotIn("output_schema", json.dumps(RecordingToolAwareFakeModel.bound_tools, default=str))

    def test_a_compile_is_one_provider_attempt_with_the_time_a_long_plan_needs(self):
        config = agent_runtime.ProviderConfig("openai", "gpt-6.1-sol", "secret-test-key")
        model = provider_client.ProviderModelFactory().compile(config)
        self.assertEqual((model.max_retries, model.request_timeout), (0, provider_client.COMPILE_TIMEOUT_SECONDS))


class ListingTests(unittest.TestCase):
    def test_a_listing_holds_at_most_what_team_admits_by_encoded_size(self):
        """Each listed Routine's steps fit one Team plan's bound and all of them the Team's whole bound."""
        step = LISTED["steps"][0]

        def listed(index: int, size: int) -> dict:
            # One step whose single input name pads the encoded steps to exactly ``size`` bytes.
            empty = len(json.dumps([{**step, "inputs": [""]}], separators=(",", ":")).encode())
            return {
                **LISTED,
                "routine_id": f"{index:032x}",
                "steps": [{**step, "inputs": ["é" * ((size - empty) // 2)]}],
            }

        most = routine.MAX_LISTED_STEPS_BYTES
        self.assertEqual(len(routine.canonical_routines([listed(0, most)])), 1)
        with self.assertRaises(routine.RoutineContractError):
            routine.canonical_routines([listed(0, most + 2)])
        whole = [listed(index, most) for index in range(routine.MAX_LISTING_STEPS_BYTES // most)]
        self.assertEqual(len(routine.canonical_routines(whole)), len(whole))
        with self.assertRaises(routine.RoutineContractError):
            routine.canonical_routines([*whole, listed(len(whole), 128)])

    def test_the_chat_lists_a_repeated_action_once_with_its_count(self):
        step = LISTED["steps"][0]
        other = {**step, "id": "zones", "action": "list"}
        steps = [{**step, "id": f"g{index}"} for index in range(3)] + [other, {**step, "id": "last"}]
        result, model = _turn(_chat([{**LISTED, "steps": steps}]), AIMessage(content="Olá."))
        self.assertEqual(result.reply, "Olá.")
        system = next(item.content for item in model.seen_messages[-1] if isinstance(item, SystemMessage))
        runs = [["hello-pulse", "hello", 3], ["hello-pulse", "list"], ["hello-pulse", "hello"]]
        self.assertIn(json.dumps(runs, ensure_ascii=False), system)
        self.assertEqual(turn_prompt._runs([]), [])


if __name__ == "__main__":
    unittest.main()

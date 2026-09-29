"""Standing instructions: a closed contract, quoted as data below the policy, pinned for a whole logical turn."""

from __future__ import annotations

import dataclasses
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import agent_runtime
import instructions
import runtime_api
import turn_pins
import turn_prompt
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import RecordingToolAwareFakeModel, context
from test_runtime_api import TOKEN, body

RULES = ("Responda sempre em português do Brasil.", "Use listas curtas, sem tabelas.")
ACTION_TOOL = agent_runtime._tool_name("hello-pulse", "hello")


def _system(messages: list) -> str:
    return next(message.content for message in messages if isinstance(message, SystemMessage))


class ContractTests(unittest.TestCase):
    def test_only_distinct_single_line_bounded_rules_are_admitted(self):
        self.assertEqual(instructions.canonical(list(RULES)), RULES)
        self.assertEqual(instructions.canonical(()), ())
        broken = [
            None,
            "one rule",
            [f"Rule {index}" for index in range(instructions.MAX_INSTRUCTIONS + 1)],
            [""],
            [" padded"],
            ["x" * (instructions.MAX_INSTRUCTION_CHARS + 1)],
            ["Linha 1\nLinha 2"],
            ["Separada aqui"],
            ["Invisível​"],
            ["Café"],
            [3],
            ["Use listas.", "use LISTAS."],
        ]
        for value in broken:
            with self.subTest(value=value), self.assertRaises(instructions.InstructionsError):
                instructions.canonical(value)

    def test_the_turn_context_admits_only_canonical_rules(self):
        self.assertEqual(dataclasses.replace(context(), instructions=list(RULES)).instructions, RULES)
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "standing instructions"):
            dataclasses.replace(context(), instructions=("Linha\nquebrada",))


class PromptTests(unittest.TestCase):
    def test_rules_are_quoted_between_the_contracts_and_the_date_and_absent_when_empty(self):
        self.assertNotIn("Standing instructions the Supervisor saved", turn_prompt.system_prompt(context()))
        prompt = turn_prompt.system_prompt(dataclasses.replace(context(), instructions=RULES))
        section = prompt.index("Standing instructions the Supervisor saved")
        self.assertLess(prompt.index("Enabled Assistant contracts"), section)
        self.assertLess(section, prompt.index("Current date:"))
        self.assertIn(json.dumps(list(RULES), ensure_ascii=False), prompt)
        self.assertIn("never supply the target or values of a change", prompt)
        self.assertIn("the current message wins", prompt)
        self.assertIn("recommend the option the rule describes", prompt)

    def test_the_policy_says_where_a_lasting_preference_belongs(self):
        prompt = turn_prompt.system_prompt(context())
        self.assertIn("You cannot save anything for later chats yourself", prompt)
        self.assertIn("one of the Team's standing instructions", prompt)


class PinningTests(unittest.TestCase):
    def test_a_resumed_turn_keeps_the_rules_it_started_with(self):
        RecordingToolAwareFakeModel.seen_messages = []
        model = RecordingToolAwareFakeModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": ACTION_TOOL, "args": {}, "id": "a1"}]),
                AIMessage(content="Feito."),
                AIMessage(content="Nova regra."),
            ]
        )
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
        started = dataclasses.replace(context(), instructions=RULES[:1])
        suspended = runtime.start(started, "Cumprimente Ada")
        changed = dataclasses.replace(started, instructions=RULES[1:])
        runtime.resume(changed, {suspended.actions[0].interrupt_id: {"message": "oi"}})
        runtime.start(changed, "E agora?")
        seen = [_system(messages) for messages in model.seen_messages]
        self.assertIn(RULES[0], seen[0])
        self.assertIn(RULES[0], seen[1])
        self.assertNotIn(RULES[1], seen[1])
        self.assertIn(RULES[1], seen[2])

    def test_only_the_exact_recorded_json_is_accepted(self):
        def pins(rules: object) -> dict[str, object]:
            return {turn_pins.DATE_METADATA: "2026-09-29", turn_pins.INSTRUCTIONS_METADATA: rules}

        self.assertEqual(turn_pins.restore(pins('["Use listas."]'))[1], ("Use listas.",))
        self.assertEqual(turn_pins.restore(pins("[]"))[1], ())
        for value in (None, ["Use listas."], "not json", '[ "Use listas." ]', '["a\\nb"]', "{}"):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore(pins(value))


class EndpointTests(unittest.TestCase):
    def test_the_turn_endpoint_refuses_invalid_rules(self):
        never_started = SimpleNamespace(start=mock.Mock(side_effect=AssertionError("must not start")))
        app = runtime_api.create_app(runtime=never_started, token_reader=lambda: TOKEN)
        headers = {"Authorization": f"Bearer {TOKEN}"}
        api = TestClient(app)
        too_many = [f"Rule {index}" for index in range(instructions.MAX_INSTRUCTIONS + 1)]
        self.assertEqual(api.post("/v1/turns", json=body(instructions=too_many), headers=headers).status_code, 422)
        refused = api.post("/v1/turns", json=body(instructions=["Linha\nquebrada"]), headers=headers)
        self.assertEqual(refused.status_code, 400)
        self.assertEqual(refused.json(), {"detail": "invalid standing instructions"})
        self.assertEqual(api.post("/v1/turns", json=body(instructions=None), headers=headers).status_code, 422)
        never_started.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()

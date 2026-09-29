"""Brain-authored multiple-choice clarification: closed shape, terminal turn, and refusal before any tool runs."""

from __future__ import annotations

import unittest
from unittest import mock

import agent_runtime
import clarification
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import ToolAwareFakeModel, action, assistant, context

VALID = {
    "question": "Qual período você quer cobrir?",
    "options": [
        {"label": "Hoje", "description": "Só lançamentos de hoje."},
        {"label": "Esta semana", "description": ""},
    ],
    "default_index": 0,
}
ACTION_TOOL = agent_runtime._tool_name("hello-pulse", "hello")


def _clarify(args: dict, call_id: str = "ask-1") -> dict:
    return {"name": clarification.TOOL_NAME, "args": args, "id": call_id, "type": "tool_call"}


def _action(call_id: str = "act-1") -> dict:
    return {"name": ACTION_TOOL, "args": {"name": "Ada"}, "id": call_id, "type": "tool_call"}


def _messages(saver: InMemorySaver, thread_id: str) -> list:
    return saver.get_tuple({"configurable": {"thread_id": thread_id}}).checkpoint["channel_values"]["messages"]


class ParseTests(unittest.TestCase):
    def test_only_the_closed_shape_parses(self):
        parsed = clarification.parse(VALID)
        self.assertEqual(parsed.to_dict(), VALID)
        self.assertEqual(
            parsed.render(), "Qual período você quer cobrir?\n\n1. Hoje ✓ — Só lançamentos de hoje.\n2. Esta semana"
        )
        broken = [
            None,
            {**VALID, "extra": 1},
            {**VALID, "question": "Linha 1\nLinha 2"},
            {**VALID, "question": "   "},
            {**VALID, "question": 5},
            {**VALID, "question": "x" * 241},
            {**VALID, "options": VALID["options"][:1]},
            {**VALID, "options": [{"label": f"O{i}", "description": ""} for i in range(6)]},
            {**VALID, "options": [{"label": "Hoje", "description": ""}, {"label": "hoje", "description": ""}]},
            {**VALID, "options": [{"label": "Hoje", "description": ""}, {"label": "B​", "description": ""}]},
            {**VALID, "options": [{"label": "Hoje"}, {"label": "B", "description": ""}]},
            {**VALID, "options": [{"label": "x" * 81, "description": ""}, {"label": "B", "description": ""}]},
            {**VALID, "options": "Hoje, Semana"},
            {**VALID, "options": ["Hoje", "Semana"]},
            {**VALID, "default_index": 2},
            {**VALID, "default_index": True},
            {**VALID, "default_index": "0"},
        ]
        for value in broken:
            with self.subTest(value=str(value)[:60]):
                self.assertIsNone(clarification.parse(value))

    def test_recorded_requires_the_tool_result_to_end_the_graph(self):
        call = AIMessage(content="", tool_calls=[_clarify(VALID)])
        result = ToolMessage(content=clarification.RECORDED, tool_call_id="ask-1", name=clarification.TOOL_NAME)
        self.assertEqual(clarification.recorded([call, result]).question, VALID["question"])
        self.assertIsNone(clarification.recorded([result]))
        self.assertIsNone(clarification.recorded([call, AIMessage(content="done")]))
        self.assertIsNone(clarification.recorded([HumanMessage(content="x"), result]))
        double = AIMessage(content="", tool_calls=[_clarify(VALID), _clarify(VALID, "ask-2")])
        self.assertIsNone(clarification.recorded([double, result]))


class GraphTests(unittest.TestCase):
    def _runtime(self, *responses: AIMessage) -> tuple[agent_runtime.AgentRuntime, InMemorySaver]:
        saver = InMemorySaver()
        model = ToolAwareFakeModel(responses=list(responses))
        return agent_runtime.AgentRuntime(saver, model_factory=lambda _config: model), saver

    def test_a_valid_clarification_ends_the_turn_and_remembers_exactly_what_is_shown(self):
        runtime, saver = self._runtime(
            AIMessage(content="", tool_calls=[_clarify(VALID)]), AIMessage(content="never used")
        )
        turn = context(assistant("hello-pulse", action()))
        result = runtime.start(turn, "Quais modelos saíram?")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.clarification.to_dict(), VALID)
        self.assertEqual(result.reply, clarification.parse(VALID).render())
        messages = _messages(saver, turn.thread_id)
        self.assertIsInstance(messages[-1], AIMessage)
        self.assertEqual(messages[-1].content, result.reply)
        self.assertFalse(messages[-1].tool_calls)

    def test_a_clarification_mixed_with_an_action_is_refused_before_either_runs(self):
        runtime, saver = self._runtime(
            AIMessage(content="", tool_calls=[_clarify(VALID), _action()]),
            AIMessage(content="", tool_calls=[_clarify(VALID, "ask-2")]),
        )
        turn = context(assistant("hello-pulse", action()))
        with mock.patch("langgraph.types.interrupt") as interrupt:
            result = runtime.start(turn, "Faça algo")
        interrupt.assert_not_called()
        self.assertIsNotNone(result.clarification)
        refusals = [m for m in _messages(saver, turn.thread_id) if isinstance(m, ToolMessage)][:2]
        self.assertEqual({m.tool_call_id for m in refusals}, {"ask-1", "act-1"})
        self.assertTrue(all("cannot be combined" in m.content for m in refusals))

    def test_an_unparsable_clarification_beside_an_action_stops_the_whole_response(self):
        broken = {
            "name": clarification.TOOL_NAME,
            "args": "{not json",
            "id": "ask-bad",
            "error": "bad json",
            "type": "invalid_tool_call",
        }
        runtime, saver = self._runtime(
            AIMessage(content="", tool_calls=[_action()], invalid_tool_calls=[broken]),
            AIMessage(content="", invalid_tool_calls=[{**broken, "id": "ask-bad-2"}]),
            AIMessage(content="", tool_calls=[_clarify(VALID, "ask-2")]),
        )
        turn = context(assistant("hello-pulse", action()))
        with mock.patch("langgraph.types.interrupt") as interrupt:
            result = runtime.start(turn, "Faça algo")
        interrupt.assert_not_called()
        self.assertIsNotNone(result.clarification)
        refusals = [m for m in _messages(saver, turn.thread_id) if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in refusals[:3]], ["act-1", "ask-bad", "ask-bad-2"])
        self.assertIn("cannot be combined", refusals[0].content)
        self.assertTrue(refusals[2].content.startswith("Not executed: the clarification must"))

    def test_a_malformed_clarification_is_refused_and_can_be_corrected(self):
        runtime, saver = self._runtime(
            AIMessage(content="", tool_calls=[_clarify({**VALID, "default_index": 9})]),
            AIMessage(content="", tool_calls=[_clarify(VALID, "ask-2")]),
        )
        turn = context(assistant("hello-pulse", action()))
        result = runtime.start(turn, "Quais modelos saíram?")
        self.assertEqual(result.clarification.default_index, 0)
        refusal = next(m for m in _messages(saver, turn.thread_id) if isinstance(m, ToolMessage))
        self.assertTrue(refusal.content.startswith("Not executed: the clarification must"))

    def test_no_clarification_after_an_action_ran(self):
        runtime, saver = self._runtime(
            AIMessage(content="", tool_calls=[_action()]),
            AIMessage(content="", tool_calls=[_clarify(VALID)]),
            AIMessage(content="Done with what I had."),
        )
        turn = context(assistant("hello-pulse", action()))
        suspended = runtime.start(turn, "Cumprimente Ada")
        self.assertEqual(suspended.status, "action-required")
        finished = runtime.resume(turn, {suspended.actions[0].interrupt_id: {"message": "hi"}})
        self.assertEqual(finished.reply, "Done with what I had.")
        self.assertIsNone(finished.clarification)
        refusal = [m for m in _messages(saver, turn.thread_id) if isinstance(m, ToolMessage)][-1]
        self.assertIn("an Action already ran", refusal.content)

    def test_the_next_turn_answers_on_top_of_the_remembered_question(self):
        runtime, saver = self._runtime(
            AIMessage(content="", tool_calls=[_clarify(VALID)]), AIMessage(content="Hoje saíram dois.")
        )
        turn = context(assistant("hello-pulse", action()))
        runtime.start(turn, "Quais modelos saíram?")
        answered = runtime.start(turn, "Quais modelos saíram?\n\nQual período você quer cobrir?\nHoje")
        self.assertEqual(answered.reply, "Hoje saíram dois.")
        contents = [m.content for m in _messages(saver, turn.thread_id) if isinstance(m, AIMessage) and m.content]
        self.assertEqual(contents, [clarification.parse(VALID).render(), "Hoje saíram dois."])

    def test_a_checkpoint_failure_while_remembering_the_question_is_a_state_error(self):
        runtime, _saver = self._runtime(AIMessage(content="", tool_calls=[_clarify(VALID)]))
        turn = context(assistant("hello-pulse", action()))
        real_agent = runtime._agent

        def failing_agent(ctx, *, clarification_allowed):
            agent = real_agent(ctx, clarification_allowed=clarification_allowed)
            agent.update_state = mock.Mock(side_effect=OSError("disk"))
            return agent

        with (
            mock.patch.object(runtime, "_agent", side_effect=failing_agent),
            self.assertRaises(agent_runtime.RuntimeStateError),
        ):
            runtime.start(turn, "Quais modelos saíram?")


if __name__ == "__main__":
    unittest.main()

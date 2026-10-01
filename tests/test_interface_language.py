"""A turn writes in the interface language its start pinned, and keeps it and its start message across resumes."""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import agent_runtime
import interface_language
import turn_pins
import turn_prompt
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import RecordingToolAwareFakeModel, action, assistant, context

ACTION_TOOL = agent_runtime._tool_name("hello-pulse", "hello")
MESSAGE_ID = "shimpz-turn-" + "a" * 32
LANGUAGE_LINE = "Write every reply, clarification question, and option in "


def _system(messages: list) -> str:
    return next(message.content for message in messages if isinstance(message, SystemMessage))


class InterfaceLanguageTests(unittest.TestCase):
    def test_the_closed_languages_and_their_prompt_names(self):
        self.assertEqual(set(interface_language.LANGUAGE_NAMES), {"ar", "de", "en", "es", "fr", "ja", "pt", "zh"})
        self.assertEqual(interface_language.language_name("pt"), "Brazilian Portuguese")
        self.assertEqual(interface_language.language_name("zh"), "Simplified Chinese")
        for value in ("pt", "ar"):
            self.assertTrue(interface_language.valid(value))
        for value in ("pt-BR", "PT", "", None, 1, "Portuguese"):
            with self.subTest(value=value):
                self.assertFalse(interface_language.valid(value))

    def test_a_selected_language_adds_one_line_just_before_the_date(self):
        turn = dataclasses.replace(context(assistant("hello-pulse", action())), locale="ja")
        prompt = turn_prompt.system_prompt(turn)
        line = prompt.index(LANGUAGE_LINE + "Japanese, the language the user selected in the interface")
        self.assertEqual(prompt.count(LANGUAGE_LINE), 1)
        self.assertLess(prompt.index("Enabled Assistant contracts"), line)
        self.assertLess(line, prompt.index("Current date:"))
        self.assertIn("Keep names, identifiers, quoted text, and data as they are.", prompt)
        self.assertNotIn(LANGUAGE_LINE, turn_prompt.system_prompt(context(assistant("hello-pulse", action()))))

    def test_a_turn_context_admits_only_a_closed_language_and_a_start_message_id(self):
        self.assertEqual(dataclasses.replace(context(), locale="de", turn_message_id=MESSAGE_ID).locale, "de")
        for changes in (
            {"locale": "pt-BR"},
            {"locale": 1},
            {"turn_message_id": "shimpz-turn-short"},
            {"turn_message_id": MESSAGE_ID.upper()},
            {"turn_message_id": 1},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(agent_runtime.RuntimeContractError, "turn"):
                dataclasses.replace(context(), **changes)

    def test_the_turn_pins_restore_only_what_a_start_recorded(self):
        for locale in ("pt", None):
            with self.subTest(locale=locale):
                pins = turn_pins.record_turn(locale, MESSAGE_ID)
                self.assertEqual(turn_pins.restore_turn(pins), (locale, MESSAGE_ID))
        valid = turn_pins.record_turn("pt", MESSAGE_ID)
        corrupt = (
            {},
            {**valid, turn_pins.LOCALE_METADATA: '"pt-BR"'},
            {**valid, turn_pins.LOCALE_METADATA: '"pt" '},
            {**valid, turn_pins.LOCALE_METADATA: "not json"},
            {**valid, turn_pins.LOCALE_METADATA: None},
            {**valid, turn_pins.MESSAGE_METADATA: "null"},
            {**valid, turn_pins.MESSAGE_METADATA: '"shimpz-turn-1"'},
            {**valid, turn_pins.MESSAGE_METADATA: 1},
            {key: value for key, value in valid.items() if key != turn_pins.MESSAGE_METADATA},
        )
        for metadata in corrupt:
            with self.subTest(metadata=metadata), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_turn(metadata)
        self.assertTrue(turn_pins.valid_turn(None, None, started=False))
        self.assertFalse(turn_pins.valid_turn(None, None))

    def test_a_resumed_turn_keeps_the_language_and_message_its_start_pinned(self):
        RecordingToolAwareFakeModel.seen_messages = []
        model = RecordingToolAwareFakeModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": ACTION_TOOL, "args": {}, "id": "a1"}]),
                AIMessage(content="Pronto."),
            ]
        )
        saver = InMemorySaver()
        runtime = agent_runtime.AgentRuntime(saver, model_factory=lambda _config: model)
        started = dataclasses.replace(context(), locale="pt")
        suspended = runtime.start(started, "Greet Ada")
        pinned = saver.get_tuple({"configurable": {"thread_id": started.thread_id}}).metadata
        locale, message_id = turn_pins.restore_turn(pinned)
        self.assertEqual(locale, "pt")
        self.assertRegex(message_id, turn_pins.TURN_MESSAGE_RE)
        # A resume never names a language; the one its start pinned still rules every later call of the turn.
        runtime.resume(dataclasses.replace(started, locale=None), {suspended.actions[0].interrupt_id: {"ok": True}})
        prompts = [_system(messages) for messages in model.seen_messages]
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn(LANGUAGE_LINE + "Brazilian Portuguese", prompt)

    def test_a_resume_refuses_a_turn_whose_language_pins_are_missing(self):
        model = RecordingToolAwareFakeModel(
            responses=[AIMessage(content="", tool_calls=[{"name": ACTION_TOOL, "args": {}, "id": "a1"}])]
        )
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
        suspended = runtime.start(context(), "Greet Ada")
        with (
            mock.patch.object(turn_pins, "restore_turn", side_effect=turn_pins.PinError("missing")),
            self.assertRaises(agent_runtime.RuntimeStateError),
        ):
            runtime.resume(context(), {suspended.actions[0].interrupt_id: {"ok": True}})


if __name__ == "__main__":
    unittest.main()

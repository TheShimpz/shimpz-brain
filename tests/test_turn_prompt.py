"""The turn policy prompt: acting on safe defaults, asking only open decisions, and the trusted current date."""

import dataclasses
import datetime
import unittest

import agent_runtime
import turn_pins
import turn_prompt
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_agent_runtime import RecordingToolAwareFakeModel, action, assistant, context
from tool_fake import system_text

YESTERDAY = datetime.date(2026, 9, 28)
TODAY = datetime.date(2026, 9, 29)
ACTION_TOOL = agent_runtime._tool_name("hello-pulse", "hello")


class PromptTests(unittest.TestCase):
    def test_the_policy_prefers_a_stated_default_and_asks_only_open_decisions(self):
        prompt = turn_prompt.system_prompt(context(assistant("hello-pulse", action())))
        policy = turn_prompt.CLARIFY_AND_COMPLETE
        self.assertIn(policy, prompt)
        self.assertIn("Prefer acting over asking", policy)
        self.assertIn("state the assumption in one short sentence", policy)
        self.assertIn("depends on their own situation", policy)
        self.assertIn("Never ask for a missing detail that an available read-only Action can look up", policy)
        self.assertIn("ask before any Action", policy)
        self.assertIn("never as a free-text question or a numbered list", policy)
        self.assertIn("do not add a lookup that only reconfirms them", policy)
        self.assertIn("follow any read-before-change step the Assistant requires", policy)
        self.assertIn("After an Action has run, that tool is unavailable", policy)
        self.assertIn("never ask for secrets", policy)
        self.assertIn("cover every requirement the request implies", policy)

    def test_the_turn_date_closes_the_prompt_after_every_stable_part(self):
        turn = dataclasses.replace(context(assistant("hello-pulse", action())), turn_date=TODAY)
        prompt = turn_prompt.system_prompt(turn)
        self.assertTrue(
            prompt.endswith(
                "\n\nCurrent date: 2026-09-29 (UTC). When the user's local date could differ and it matters, say "
                "which date you used."
            )
        )
        self.assertLess(prompt.index("Enabled Assistant contracts"), prompt.index("Current date:"))

    def test_the_turn_date_names_the_zone_it_is_the_date_in(self):
        turn = dataclasses.replace(context(), turn_date=TODAY, timezone="America/Sao_Paulo")
        self.assertIn("Current date: 2026-09-29 (America/Sao_Paulo). ", turn_prompt.system_prompt(turn))
        self.assertEqual(context().timezone, "UTC")
        for zone in ("Mars/Olympus", "america/sao_paulo", "", None, 1):
            with self.subTest(zone=zone), self.assertRaisesRegex(agent_runtime.RuntimeContractError, "turn date"):
                dataclasses.replace(context(), timezone=zone)

    def test_today_is_the_current_utc_date_and_the_default_turn_date(self):
        before = datetime.datetime.now(datetime.UTC).date()
        value = turn_prompt.today()
        after = datetime.datetime.now(datetime.UTC).date()
        self.assertIn(value, {before, after})
        self.assertIs(agent_runtime.TurnContext.__dataclass_fields__["turn_date"].default_factory, turn_prompt.today)
        before = datetime.datetime.now(datetime.UTC).date()
        value = context().turn_date
        after = datetime.datetime.now(datetime.UTC).date()
        self.assertIn(value, {before, after})

    def test_the_turn_date_must_be_a_plain_date(self):
        for value in (datetime.datetime(2026, 9, 29, tzinfo=datetime.UTC), "2026-09-29", None):
            with self.subTest(value=value), self.assertRaisesRegex(agent_runtime.RuntimeContractError, "turn date"):
                dataclasses.replace(context(), turn_date=value)


class PinnedDateTests(unittest.TestCase):
    def test_a_turn_resumed_after_midnight_keeps_the_date_it_started_with(self):
        RecordingToolAwareFakeModel.seen_messages = []
        model = RecordingToolAwareFakeModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": ACTION_TOOL, "args": {}, "id": "a1"}]),
                AIMessage(content="Done."),
                AIMessage(content="New day."),
            ]
        )
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
        started = dataclasses.replace(context(), turn_date=YESTERDAY)
        suspended = runtime.start(started, "Greet Ada")
        self.assertEqual(suspended.status, "action-required")
        resumed = dataclasses.replace(started, turn_date=TODAY)
        runtime.resume(resumed, {suspended.actions[0].interrupt_id: {"message": "hi"}})
        runtime.start(resumed, "And now?")
        dates = [system_text(messages).rsplit("Current date: ", 1)[1][:10] for messages in model.seen_messages]
        self.assertEqual(dates, ["2026-09-28", "2026-09-28", "2026-09-29"])

    def test_a_turn_resumed_after_midnight_keeps_the_zone_it_started_with(self):
        RecordingToolAwareFakeModel.seen_messages = []
        model = RecordingToolAwareFakeModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": ACTION_TOOL, "args": {}, "id": "a1"}]),
                AIMessage(content="Done."),
            ]
        )
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model)
        started = dataclasses.replace(context(), turn_date=YESTERDAY, timezone="America/Sao_Paulo")
        suspended = runtime.start(started, "Greet Ada")
        # The resume names no zone of its own: the start's date and zone come back from its pins.
        resumed = dataclasses.replace(started, turn_date=TODAY, timezone="UTC")
        runtime.resume(resumed, {suspended.actions[0].interrupt_id: {"message": "hi"}})
        dates = [system_text(messages).rsplit("Current date: ", 1)[1][:30] for messages in model.seen_messages]
        self.assertEqual(dates, ["2026-09-28 (America/Sao_Paulo)", "2026-09-28 (America/Sao_Paulo)"])

    def test_only_an_exact_recorded_zone_is_accepted(self):
        self.assertEqual(turn_pins.restore_zone(turn_pins.record_zone("America/Sao_Paulo")), "America/Sao_Paulo")
        self.assertEqual(turn_pins.restore_zone(turn_pins.record_zone("UTC")), "UTC")
        for value in (None, 1, '"Mars/Olympus"', '"UTC" ', "UTC", "not json", '"' + "a" * 70 + '"'):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore_zone({turn_pins.ZONE_METADATA: value})
        with self.assertRaises(turn_pins.PinError):
            turn_pins.restore_zone({})

    def test_only_an_exact_recorded_date_is_accepted(self):
        def pins(date: object) -> dict[str, object]:
            return {
                turn_pins.DATE_METADATA: date,
                turn_pins.MEMORY_METADATA: "null",
                turn_pins.SKILLS_METADATA: "null",
                turn_pins.ROUTINES_METADATA: "null",
                turn_pins.WRITABLE_METADATA: "true",
            }

        self.assertEqual(turn_pins.restore(pins("2026-09-29")), (TODAY, None, None, None, True))
        for value in (None, 20260929, "20260929", "2026-9-29", "2026-02-30", "today"):
            with self.subTest(value=value), self.assertRaises(turn_pins.PinError):
                turn_pins.restore(pins(value))


if __name__ == "__main__":
    unittest.main()

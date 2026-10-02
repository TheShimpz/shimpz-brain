"""Provider-free checks for the frozen precision corpus, its simulated Assistants, and its oracle (ADR-0094)."""

from __future__ import annotations

import dataclasses
import unittest
from collections import Counter
from unittest import mock

from eval import corpus

# Frozen with precision-v1: a change to the corpus must change its id, not this fingerprint alone.
DIGEST = "sha256:b79bb5928dabd15698eea826ef254e435a7d14dbe7f075146101750671fc5eac"


def _scenario(scenario_id: str) -> corpus.Scenario:
    return corpus.SCENARIOS_BY_ID[scenario_id]


class CorpusTests(unittest.TestCase):
    def test_the_corpus_is_frozen_and_covers_every_stratum(self):
        corpus.validate()
        self.assertEqual((corpus.CORPUS_ID, corpus.digest()), ("precision-v1", DIGEST))
        self.assertEqual(len(corpus.SCENARIOS), 120)
        self.assertEqual(Counter(s.locale for s in corpus.SCENARIOS), dict.fromkeys(corpus.LOCALES, 15))
        self.assertEqual(Counter(s.scope for s in corpus.SCENARIOS), dict.fromkeys(corpus.SCOPES, 40))
        self.assertEqual({s.template.behavior for s in corpus.SCENARIOS}, corpus.BEHAVIORS)
        self.assertEqual({len(s.assistants) for s in corpus.SCENARIOS}, {1, 2, 3, 4, 16})
        self.assertTrue(any(s.strata["multi_assistant"] for s in corpus.SCENARIOS))
        self.assertEqual({s.template.min_rounds for s in corpus.SCENARIOS}, {1, 2, 3, 4})
        scenario = _scenario("dns-update.ja")
        self.assertEqual(scenario.message, scenario.template.messages["ja"])
        self.assertEqual(
            scenario.strata,
            {
                "language": "ja",
                "scope": scenario.scope,
                "assistants": len(scenario.assistants),
                "behavior": "safe-lookup",
                "multi_assistant": False,
                "min_rounds": 3,
            },
        )
        padded = next(s for s in corpus.SCENARIOS if s.scope == "16")
        self.assertEqual(padded.assistants[: len(padded.template.needed)], padded.template.needed)
        self.assertTrue(any(not corpus.ASSISTANTS[name].relevant for name in padded.assistants))

    def test_validation_refuses_each_structural_defect(self):
        template = corpus.TEMPLATES[0]
        defects = [
            ("TEMPLATES", (*corpus.TEMPLATES, template)),
            ("TEMPLATES", (dataclasses.replace(template, behavior="guess"), *corpus.TEMPLATES[1:])),
            ("TEMPLATES", (dataclasses.replace(template, needed=("fitness",)), *corpus.TEMPLATES[1:])),
            ("TEMPLATES", (dataclasses.replace(template, min_rounds=9), *corpus.TEMPLATES[1:])),
            ("TEMPLATES", (dataclasses.replace(template, expect_clarification=True), *corpus.TEMPLATES[1:])),
            ("TEMPLATES", (dataclasses.replace(template, changes={}), *corpus.TEMPLATES[1:])),
        ]
        for name, value in defects:
            with self.subTest(defect=name), mock.patch.object(corpus, name, value), self.assertRaises(ValueError):
                corpus.validate()
        broken = dataclasses.replace(corpus.SCENARIOS[0], assistants=("tasks",))
        with mock.patch.object(corpus, "SCENARIOS", (broken, *corpus.SCENARIOS[1:])), self.assertRaises(ValueError):
            corpus.validate()
        unscoped = tuple(dataclasses.replace(s, scope="needed", assistants=s.template.needed) for s in corpus.SCENARIOS)
        with mock.patch.object(corpus, "SCENARIOS", unscoped), self.assertRaisesRegex(ValueError, "scope"):
            corpus.validate()


class WorldTests(unittest.TestCase):
    def test_dns_lookups_writes_and_failures(self):
        world = corpus.World()
        self.assertEqual(len(world.invoke("dns", "list-zones", {})["zones"]), 2)
        listed = world.invoke("dns", "list-records", {"zone_id": "zn-7f3a", "name": "api"})["records"]
        self.assertEqual([item["id"] for item in listed], ["rc-api"])
        typed = world.invoke("dns", "list-records", {"zone_id": "zn-7f3a", "type": "TXT"})["records"]
        self.assertEqual([item["name"] for item in typed], ["_old-verify.example.com"])
        self.assertEqual(len(world.invoke("dns", "list-records", {"zone_id": "zn-91c2"})["records"]), 1)
        apex = world.invoke("dns", "list-records", {"zone_id": "zn-7f3a", "name": "@"})["records"]
        self.assertEqual(apex, [])
        created = world.invoke(
            "dns", "create-record", {"zone_id": "zn-7f3a", "type": "A", "name": "Shop.Example.com.", "content": "1"}
        )
        self.assertEqual(created["record"]["name"], "shop.example.com")
        world.invoke("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": "rc-api", "content": "2"})
        world.invoke("dns", "delete-record", {"zone_id": "zn-7f3a", "record_id": "rc-oldv"})
        failures = [
            ("create-record", {"zone_id": "zn-7f3a", "type": "A", "name": "www.example.com", "content": "3"}, "exists"),
            ("update-record", {"zone_id": "zn-91c2", "record_id": "rc-api", "content": "3"}, "record-not-found"),
            ("delete-record", {"zone_id": "zn-7f3a", "record_id": "rc-none"}, "record-not-found"),
            ("list-records", {"zone_id": "zn-none"}, "zone-not-found"),
        ]
        for action, arguments, code in failures:
            with self.subTest(action=action), self.assertRaisesRegex(corpus.ActionFailedError, code):
                world.invoke("dns", action, arguments)
        self.assertEqual(world.ledger[-1]["failed"], "zone-not-found")
        state = world.snapshot()
        self.assertEqual(state["record:example.com:shop.example.com:A"], "1")
        self.assertEqual(state["record:example.com:api.example.com:A"], "2")
        self.assertNotIn("record:example.com:_old-verify.example.com:TXT", state)

    def test_tasks_calendar_messages_and_research(self):
        world = corpus.World()
        self.assertEqual(len(world.invoke("tasks", "list-tasks", {})["tasks"]), 5)
        self.assertEqual(len(world.invoke("tasks", "list-tasks", {"status": "all", "tag": "HOME"})["tasks"]), 4)
        self.assertEqual(len(world.invoke("tasks", "list-tasks", {"status": "done"})["tasks"]), 1)
        world.invoke("tasks", "create-task", {"title": "Renew passport"})
        world.invoke("tasks", "create-task", {"title": "Renew passport", "tags": ["admin"]})
        world.invoke("tasks", "complete-task", {"task_id": "tk-1"})
        with self.assertRaisesRegex(corpus.ActionFailedError, "task-not-found"):
            world.invoke("tasks", "complete-task", {"task_id": "tk-9"})
        self.assertEqual(len(world.invoke("calendar", "list-events", {"date": "2026-10-05"})["events"]), 2)
        world.invoke("calendar", "create-event", {"title": "Lunch", "date": "2026-10-09", "start_time": "12:00"})
        event = world.invoke(
            "calendar",
            "create-event",
            {"title": "Lunch", "date": "2026-10-09", "start_time": "13:00", "duration_minutes": 30},
        )
        self.assertEqual(event["event"]["duration_minutes"], 30)
        self.assertEqual(world.invoke("messages", "find-contact", {"name": " ana "})["contacts"][0]["id"], "ct-ana")
        world.invoke("messages", "send-message", {"contact_id": "ct-ana", "text": "14:30"})
        with self.assertRaisesRegex(corpus.ActionFailedError, "contact-not-found"):
            world.invoke("messages", "send-message", {"contact_id": "ana", "text": "x"})
        self.assertEqual(world.invoke("research", "search-web", {"query": "IP"})["results"], [])
        found = world.invoke("research", "search-web", {"query": "Status.example.org IP"})["results"]
        page = world.invoke("research", "read-page", {"url": found[0]["url"] + "/"})
        self.assertIn(corpus.STATUS_IP, page["text"])
        self.assertEqual(world.invoke("research", "read-page", {"url": "https://x.example"})["text"], "Page not found.")
        state = world.snapshot()
        self.assertEqual((state["new-task"], state["task:tk-1"], state["sent:ct-ana"]), (2, True, 1))
        self.assertEqual((state["new-event:2026-10-09:12:00"], state["new-event:2026-10-09:13:00"]), (1, 1))

    def test_distractors_accept_writes_that_count_as_foreign_and_undeclared_actions_fail(self):
        world = corpus.World()
        self.assertEqual(world.invoke("weather", "get-forecast", {"query": "Lisbon"}), {"items": []})
        self.assertEqual(world.invoke("weather", "set-alert", {"title": "rain"})["status"], "accepted")
        self.assertEqual(world.snapshot()["foreign:weather"], 1)
        for assistant, action in (("weather", "delete-city"), ("unknown", "list")):
            with self.assertRaisesRegex(corpus.ActionFailedError, "undeclared-action"):
                world.invoke(assistant, action, {})


class OracleTests(unittest.TestCase):
    def test_the_expected_final_state_passes(self):
        scenario = _scenario("dns-bulk.pt")
        world = corpus.World()
        for record in ("rc-www", "rc-api", "rc-app"):
            world.invoke("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": record, "content": "203.0.113.20"})
        self.assertEqual(
            corpus.oracle(scenario, world).to_dict(),
            {"passed": True, "missing": 0, "wrong": 0, "wrong_scope": 0, "duplicates": 0},
        )

    def test_missing_wrong_foreign_and_repeated_writes_fail(self):
        scenario = _scenario("dns-bulk.pt")
        world = corpus.World()
        update = {"zone_id": "zn-7f3a", "record_id": "rc-www", "content": "203.0.113.20"}
        world.invoke("dns", "update-record", update)
        world.invoke("dns", "update-record", update)
        world.invoke("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": "rc-api", "content": "203.0.113.21"})
        world.invoke("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": "rc-cdn", "content": "x.example"})
        world.invoke("tasks", "create-task", {"title": "noted"})
        result = corpus.oracle(scenario, world)
        self.assertFalse(result.passed)
        # app was never changed; api got the wrong address; cdn and the task are unexpected changes.
        self.assertEqual((result.missing, result.wrong, result.wrong_scope, result.duplicates), (1, 3, 1, 1))

    def test_read_only_clarification_and_refusal_pass_only_when_nothing_changed(self):
        world = corpus.World()
        world.invoke("messages", "find-contact", {"name": "Bruno"})
        self.assertTrue(corpus.oracle(_scenario("message-clarify.de"), world).passed)
        world.invoke("messages", "send-message", {"contact_id": "ct-bruno", "text": "Hi"})
        self.assertFalse(corpus.oracle(_scenario("message-clarify.de"), world).passed)
        self.assertTrue(corpus.oracle(_scenario("out-of-scope.zh"), corpus.World()).passed)


if __name__ == "__main__":
    unittest.main()

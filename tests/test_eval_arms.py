"""Provider-free checks for the experiment-only arm-B contracts and the arm C/D/E logic (ADR-0094)."""

from __future__ import annotations

import unittest

import agent_runtime
from eval import arms, contracts, corpus, split
from eval import world as simulated

Z = {"zone_id": "zn-7f3a"}


def _action(assistant_id: str, action_id: str):
    return next(a for a in contracts.ASSISTANTS_B[assistant_id].actions if a.id == action_id)


class ContractTests(unittest.TestCase):
    def test_arm_b_contracts_are_admitted_and_searchable_in_every_locale(self):
        provider = agent_runtime.ProviderConfig("openai", "gpt-6-luna", "offline-validation-key-0123", "low")
        definitions = tuple(
            agent_runtime.AssistantDefinition(
                a.id,
                a.genesis,
                tuple(agent_runtime.ActionDefinition(x.id, x.summary, x.input_schema) for x in a.actions),
            )
            for a in contracts.ASSISTANTS_B.values()
            if a.relevant
        )
        agent_runtime.TurnContext("t", "T", definitions, provider, memories=(), skills=(), routines=())
        self.assertEqual(set(contracts.SEARCH_TERMS), set(contracts.ASSISTANTS_B))
        self.assertTrue(all(set(terms) == set(corpus.LOCALES) for terms in contracts.SEARCH_TERMS.values()))
        self.assertEqual(_action("dns", "list-records").filters, ("name", "content"))

    def test_the_arm_b_world_reports_an_existing_record_and_offers_lookups(self):
        world = simulated.World("b")
        found = world.invoke("dns", "create-record", {**Z, "type": "A", "name": "www", "content": "203.0.113.20"})
        self.assertEqual((found["created"], found["record"]["id"]), (False, "rc-www"))
        self.assertNotIn("effect", world.ledger[-1])
        by_value = world.invoke("dns", "list-records", {**Z, "content": "192.0.2.1"})["records"]
        self.assertEqual([r["id"] for r in by_value], ["rc-api", "rc-www"])
        self.assertEqual(
            world.invoke("dns", "get-record", {**Z, "record_id": "rc-app"})["record"]["name"], "app.example.com"
        )
        self.assertEqual(
            [t["id"] for t in world.invoke("tasks", "list-tasks", {"query": "Dentist"})["tasks"]], ["tk-1"]
        )
        self.assertEqual(world.invoke("tasks", "get-task", {"task_id": "tk-6"})["task"]["status"], "done")
        with self.assertRaises(simulated.ActionFailedError):
            world.invoke("tasks", "get-task", {"task_id": "tk-9"})
        created = world.invoke("dns", "create-record", {**Z, "type": "A", "name": "shop", "content": "1"})
        self.assertNotIn("created", created)
        self.assertIn("effect", world.ledger[-1])
        with self.assertRaisesRegex(ValueError, "variant"):
            simulated.World("c")
        with self.assertRaisesRegex(simulated.ActionFailedError, "record-exists"):
            simulated.World().invoke("dns", "create-record", {**Z, "type": "A", "name": "www", "content": "1"})


class WorkingSetTests(unittest.TestCase):
    def test_the_working_set_ranks_declared_terms_and_entity_cues(self):
        scope = ["dns", "tasks", "calendar", "messages", "weather"]
        self.assertEqual(arms.working_set("Aponte api.example.com para 198.51.100.7.", "pt", scope), ("dns",))
        self.assertEqual(
            arms.working_set("2026-10-05 の会議が何時に始まるか Ana にメッセージで伝えて", "ja", scope),
            ("calendar", "messages"),
        )
        self.assertEqual(arms.working_set("Order a pizza.", "en", scope), tuple(scope))
        many = arms.working_set("record task meeting message weather", "en", scope)
        self.assertEqual(len(many), arms.WORKING_SET_LIMIT)
        self.assertEqual(arms.relevance("Schedule lunch at 12:00", "en", ["calendar"]), {"calendar": 2})
        self.assertEqual(arms.relevance("Hi", "en", ["tasks"]), {"tasks": 0})

    def test_tuning_scenarios_keep_every_needed_assistant(self):
        sets = split.load()
        for scenario in corpus.SCENARIOS:
            if split.part(scenario.id, sets) == "tuning":
                exposed = arms.working_set(scenario.message, scenario.locale, list(scenario.assistants))
                self.assertLessEqual(set(scenario.template.needed), set(exposed), scenario.id)


class GuardTests(unittest.TestCase):
    def test_value_kinds(self):
        self.assertTrue(arms.valid_kind(contracts.HOSTNAME, "_old-verify.example.com"))
        self.assertFalse(arms.valid_kind(contracts.HOSTNAME, "192.0.2.1"))
        self.assertFalse(arms.valid_kind(contracts.HOSTNAME, "a b"))
        self.assertTrue(arms.valid_kind(contracts.DATE, "2026-10-05"))
        for bad in ("2026-13-05", "5 Oct 2026", "20261005"):
            self.assertFalse(arms.valid_kind(contracts.DATE, bad), bad)
        self.assertTrue(arms.valid_kind(contracts.TIME, "14:30"))
        for bad in ("24:00", "9:30", "12:60", "noon"):
            self.assertFalse(arms.valid_kind(contracts.TIME, bad), bad)
        with self.assertRaisesRegex(ValueError, "kind"):
            arms.valid_kind("email", "x")

    def test_checks_refuse_with_closed_corrections_and_raise_signals(self):
        guard = arms.TurnGuard("Which records in example.com point to 192.0.2.1?", "en", ("dns", "fitness"))
        self.assertIsNone(guard.check("dns", _action("dns", "list-zones"), {}))
        refused = guard.check("dns", _action("dns", "list-records"), {**Z, "name": "192.0.2.1"})
        self.assertEqual((refused["code"], refused["field"], refused["changed"]), ("invalid-argument", "name", False))
        self.assertEqual(
            guard.check("dns", _action("dns", "list-records"), {**Z, "name": "mail"})["code"], "unsupported-filter"
        )
        self.assertIsNone(guard.check("dns", _action("dns", "list-records"), {**Z, "content": "192.0.2.1"}))
        fitness = contracts.ASSISTANTS_B["fitness"].actions[1]
        self.assertEqual(guard.check("fitness", fitness, {"title": "run"})["code"], "out-of-scope-write")
        self.assertEqual(
            (guard.refusals, guard.signal, guard.escalation()), (3, "argument-correction", "argument-correction")
        )
        guard.ran(_action("dns", "list-records"), {"records": [{"name": "mail.example.com"}]})
        self.assertIsNone(guard.check("dns", _action("dns", "list-records"), {**Z, "name": "mail"}))

    def test_risk_signals_escalate_only_before_any_write(self):
        guard = arms.TurnGuard("Send Ana the start time of the meeting on 2026-10-05.", "en", ("calendar", "messages"))
        guard.pending("calendar", _action("calendar", "list-events"), {"date": "2026-10-05"})
        self.assertIsNone(guard.signal)
        guard.pending("messages", _action("messages", "send-message"), {"contact_id": "ct-ana", "text": "14:30"})
        self.assertEqual(guard.escalation(), "multi-assistant-write")
        lone = arms.TurnGuard("Create an A record for blog.example.com.", "en", ("dns",))
        lone.pending("dns", _action("dns", "create-record"), {**Z, "type": "A", "name": "blog", "content": "192.0.2.7"})
        self.assertEqual(lone.signal, "unsourced-write-value")
        empty = arms.TurnGuard("Mark my dentist task as done.", "en", ("tasks",))
        empty.ran(_action("tasks", "list-tasks"), {"tasks": []})
        self.assertEqual(empty.escalation(), "empty-lookup")
        late = arms.TurnGuard("Send Bruno hi.", "en", ("messages",))
        late.ran(_action("messages", "send-message"), {"status": "sent"})
        late.ran(_action("messages", "find-contact"), {"contacts": [{"id": "a"}, {"id": "b"}]})
        self.assertEqual((late.signal, late.escalation()), ("ambiguous-lookup", None))
        self.assertEqual(arms.ARMS["E"], arms.Arm("b", checks=True, working_set=True, cascade=True))


if __name__ == "__main__":
    unittest.main()

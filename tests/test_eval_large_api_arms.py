"""Provider-free checks for the large-API arms: group ranking, task-shaped contracts, recall, and Jev (ADR-0094)."""

from __future__ import annotations

import unittest

import agent_runtime
from eval import arms, corpus, large_api, split
from eval import large_api_arms as large
from eval import world as simulated

COM = {"domain": "example.com"}
ORG = {"domain": "example.org"}


def _passes(scenario_id: str, *calls: tuple[str, dict]) -> bool:
    world = large.TaskWorld()
    for action, arguments in calls:
        world.invoke("edge", action, arguments)
    return corpus.oracle(large_api.SCENARIOS_BY_ID[scenario_id], world, large_api.INITIAL).passed


class GroupTests(unittest.TestCase):
    def test_groups_rank_by_declared_terms_and_always_include_zones(self):
        self.assertEqual(large.rank_groups("Purge the cache for https://www.example.com/a."), ("zones", "cache"))
        self.assertEqual(large.rank_groups("Hello"), ("zones",))
        many = large.rank_groups("dns record cache worker bucket ssl")
        self.assertEqual(len(many), 1 + large.GROUP_LIMIT)
        exposed = large.group_actions(("zones", "cache"))
        self.assertIn("cache-purge-urls", exposed)
        self.assertTrue(large.recall("cf-purge-url", exposed))
        self.assertFalse(large.recall("cf-update-a", exposed))

    def test_tuning_scenarios_keep_every_needed_action(self):
        sets = split.load(split.LARGE_API_SPLIT)
        for scenario in large_api.SCENARIOS:
            if split.part(scenario.id, sets) == "tuning":
                exposed = large.group_actions(large.rank_groups(scenario.message))
                self.assertTrue(large.recall(scenario.template.id, exposed), scenario.id)


class TaskContractTests(unittest.TestCase):
    def test_task_shaped_contracts_are_admitted_and_reach_the_same_state(self):
        provider = agent_runtime.ProviderConfig("openai", "gpt-6-luna", "offline-validation-key-0123", "low")
        actions = tuple(
            agent_runtime.ActionDefinition(a.id, a.summary, a.input_schema) for a in large.TASK_EDGE.actions
        )
        agent_runtime.TurnContext(
            "t", "T", (agent_runtime.AssistantDefinition("edge", large.TASK_EDGE.genesis, actions),), provider
        )
        self.assertEqual(len(large.TASK_EDGE.actions), 21)
        url = "https://www.example.com/index.html"
        self.assertTrue(_passes("cf-purge-url.en", ("purge-cache", {**COM, "urls": [url]})))
        self.assertTrue(_passes("cf-ssl-strict.en", ("set-setting", {**COM, "setting": "ssl_mode", "value": "strict"})))
        self.assertTrue(_passes("cf-block-ip.en", ("set-ip-rule", {**COM, "ip": "198.51.100.23", "mode": "block"})))
        txt = {**COM, "type": "TXT", "name": "_acme-challenge", "content": "abc123"}
        self.assertTrue(_passes("cf-dns-txt.en", ("set-dns-record", txt)))
        update = {**ORG, "type": "A", "name": "www.example.org", "content": "192.0.2.99"}
        self.assertTrue(
            _passes(
                "cf-update-a.en", ("list-dns-records", {**ORG, "name": "www.example.org"}), ("set-dns-record", update)
            )
        )
        route = {**COM, "pattern": "example.com/api/*", "script": "api-gateway"}
        self.assertTrue(_passes("cf-worker-route.en", ("list-workers", {}), ("route-worker", route)))
        self.assertTrue(_passes("cf-r2-bucket.en", ("create-bucket", {"name": "invoices-archive"})))
        dev = ("set-setting", {**ORG, "setting": "development_mode", "value": "on"})
        self.assertTrue(_passes("cf-devmode-purge.en", dev, ("purge-cache", ORG)))
        self.assertTrue(
            _passes(
                "cf-bucket-clarify.en",
                ("list-buckets", {}),
                ("list-zones", {}),
                ("get-setting", {**COM, "setting": "ssl_mode"}),
            )
        )
        self.assertFalse(_passes("cf-bucket-clarify.en", ("delete-bucket", {"name": "media-assets"})))
        self.assertTrue(_passes("cf-traffic-question.en", ("get-traffic", {**COM, "date": "2026-10-01"})))
        self.assertFalse(
            _passes(
                "cf-r2-bucket.en", ("create-bucket", {"name": "invoices-archive"}), ("deploy-site", {"project": "p"})
            )
        )
        world = large.TaskWorld()
        self.assertEqual(
            world.invoke("edge", "get-setting", {**COM, "setting": "security_level"})["result"]["value"], "off"
        )
        self.assertEqual(world.invoke("edge", "list-dns-records", ORG)["result"][0]["id"], "dr-2")
        self.assertEqual(
            world.invoke("edge", "get-traffic", {**COM, "date": "2026-10-01"})["result"]["requests"], 48213
        )
        self.assertEqual(world.invoke("weather", "get-forecast", {"query": "x"}), {"items": []})
        allow = world.invoke("edge", "set-ip-rule", {**COM, "ip": "203.0.113.9", "mode": "allow"})
        self.assertEqual(allow["result"]["mode"], "whitelist")
        for action, arguments, code in (
            ("purge-cache", {"domain": "example.net"}, "domain-not-found"),
            ("no-such-task", {}, "undeclared-action"),
        ):
            with self.assertRaisesRegex(simulated.ActionFailedError, code):
                world.invoke("edge", action, arguments)
        self.assertEqual(world.ledger[-1]["failed"], "undeclared-action")


class ArmRegistryTests(unittest.TestCase):
    def test_every_arm_names_a_known_configuration_and_edge_has_search_terms(self):
        for name, arm in arms.ARMS.items():
            self.assertIn(arm.contracts, {"a", "b", "large", "large-tasks"}, name)
            self.assertIn(arm.exposure, {"scope", "namespaces", "groups", "jev-groups"}, name)
            self.assertIn(
                (arm.routing, arm.model, arm.fallback),
                {(r, m, f) for r in ("none", "jev") for m in arms.MODELS for f in ("scope", "namespaces")},
            )
        self.assertEqual(arms.relevance("Purge the cache.", "en", ["edge", "weather"])["edge"], 2)


if __name__ == "__main__":
    unittest.main()

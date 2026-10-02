"""Provider-free checks for the large-API stratum: its 120-Action Assistant, state, oracle, and split (ADR-0094)."""

from __future__ import annotations

import dataclasses
import unittest
from collections import Counter
from unittest import mock

from eval import corpus, large_api, split
from eval import world as simulated

DIGEST = "sha256:a8f5f0b0a5aba19fcd701cf8a2d5cd36e3d113fe8f38c7560fe09499061fa112"
Z = {"zone_id": "zn-1a2b"}
ORG = {"zone_id": "zn-3c4d"}


def _passes(scenario_id: str, *calls: tuple[str, dict]) -> bool:
    world = large_api.EdgeWorld()
    for action, arguments in calls:
        world.invoke("edge", action, arguments)
    return corpus.oracle(large_api.SCENARIOS_BY_ID[scenario_id], world, large_api.INITIAL).passed


class LargeApiTests(unittest.TestCase):
    def test_the_stratum_is_frozen_with_its_own_split(self):
        large_api.validate()
        self.assertEqual((large_api.CORPUS_ID, large_api.digest()), ("large-api-v1", DIGEST))
        self.assertEqual((len(large_api.EDGE.actions), len(large_api.GROUPS)), (120, 14))
        self.assertEqual(set(Counter(s.scope for s in large_api.SCENARIOS)), set(corpus.SCOPES))
        self.assertEqual(len(large_api.SCENARIOS), 80)
        self.assertEqual(large_api.ACTION_GROUP["dns-records-update"], "dns")
        sets = split.load(split.LARGE_API_SPLIT)
        self.assertEqual((len(sets["held-out"]), len(sets["tuning"])), (5, 5))
        self.assertEqual(split.part("cf-ssl-strict.ja", sets), "held-out")

    def test_validation_refuses_defects(self):
        template = large_api.TEMPLATES[0]
        broken = (
            ("EDGE", dataclasses.replace(large_api.EDGE, actions=large_api.EDGE.actions[:-1])),
            ("GROUPS", {**large_api.GROUPS, "extra": ("x", large_api.GROUPS["dns"][1] * 2)}),
            ("TEMPLATES", (dataclasses.replace(template, changes={}), *large_api.TEMPLATES[1:])),
            ("SCENARIOS", tuple(dataclasses.replace(s, scope="needed") for s in large_api.SCENARIOS)),
        )
        for name, value in broken:
            with self.subTest(name=name), mock.patch.object(large_api, name, value), self.assertRaises(ValueError):
                large_api.validate()

    def test_the_stateful_actions_and_their_effects(self):
        world = large_api.EdgeWorld()
        self.assertEqual(len(world.invoke("edge", "zones-list", {})["result"]), 2)
        self.assertEqual(world.invoke("edge", "zones-list", {"name": "example.org"})["result"][0]["id"], "zn-3c4d")
        records = world.invoke("edge", "dns-records-list", {**ORG, "name": "www.example.org", "type": "A"})["result"]
        world.invoke("edge", "dns-records-update", {**ORG, "record_id": records[0]["id"], "content": "192.0.2.99"})
        created = world.invoke("edge", "dns-records-create", {**Z, "type": "TXT", "name": "_acme", "content": "x"})
        self.assertEqual(created["result"]["name"], "_acme.example.com")
        world.invoke("edge", "dns-records-create", {**Z, "type": "A", "name": "a.example.com", "content": "1"})
        world.invoke("edge", "zones-settings-update", {**ORG, "setting": "development_mode", "value": "ON"})
        self.assertEqual(world.invoke("edge", "ssl-settings-get", Z)["result"]["mode"], "full")
        world.invoke("edge", "ssl-settings-update", {**Z, "mode": "strict"})
        world.invoke("edge", "cache-purge-urls", {**Z, "files": ["https://www.example.com/index.html"]})
        world.invoke("edge", "cache-purge-everything", ORG)
        world.invoke("edge", "ip-access-rules-create", {**Z, "mode": "block", "value": "198.51.100.23"})
        self.assertEqual(len(world.invoke("edge", "workers-scripts-list", {})["result"]), 2)
        world.invoke("edge", "workers-routes-create", {**Z, "pattern": "example.com/api/*", "script": "api-gateway"})
        self.assertEqual(len(world.invoke("edge", "r2-buckets-list", {})["result"]), 2)
        world.invoke("edge", "r2-buckets-create", {"name": "invoices-archive"})
        world.invoke("edge", "r2-buckets-delete", {"name": "media-assets"})
        traffic = world.invoke("edge", "analytics-zone-traffic", {**Z, "date": "2026-10-01"})["result"]
        self.assertEqual(traffic["requests"], 48213)
        self.assertEqual(
            world.invoke("edge", "analytics-zone-traffic", {**Z, "date": "2026-01-01"})["result"]["requests"], 0
        )
        self.assertEqual(world.invoke("edge", "zones-plan-get", Z), {"success": True, "result": []})
        world.invoke("edge", "tunnels-create", {"name": "t"})
        world.invoke("music", "create-playlist", {"title": "x"})
        self.assertEqual(world.invoke("music", "search-songs", {"query": "x"}), {"items": []})
        failures = (
            ("dns-records-list", {"zone_id": "zn-none"}, "zone-not-found"),
            ("dns-records-update", {**Z, "record_id": "dr-2", "content": "1"}, "record-not-found"),
            ("workers-routes-create", {**Z, "pattern": "p", "script": "missing"}, "script-not-found"),
            ("r2-buckets-create", {"name": "backups-2026"}, "bucket-exists"),
            ("r2-buckets-delete", {"name": "missing"}, "bucket-not-found"),
            ("no-such-action", {}, "undeclared-action"),
        )
        for action, arguments, code in failures:
            with self.subTest(action=action), self.assertRaisesRegex(simulated.ActionFailedError, code):
                world.invoke("edge", action, arguments)
        state = world.snapshot()
        self.assertEqual(state["edge:dns:example.org:www.example.org:A"], "192.0.2.99")
        self.assertEqual(state["edge:setting:example.org:development_mode"], "on")
        self.assertEqual(state["edge:route:example.com:example.com/api/*"], "api-gateway")
        self.assertEqual((state["edge-other:tunnels-create"], state["edge:purge:example.org:*"]), (1, 1))
        self.assertNotIn("edge:bucket:media-assets", state)

    def test_the_oracle_accepts_exact_effects_only(self):
        url = "https://www.example.com/index.html"
        self.assertTrue(_passes("cf-purge-url.en", ("zones-list", {}), ("cache-purge-urls", {**Z, "files": [url]})))
        self.assertFalse(_passes("cf-purge-url.en", ("cache-purge-everything", Z)))
        self.assertTrue(_passes("cf-ssl-strict.de", ("ssl-settings-update", {**Z, "mode": "strict"})))
        both = (("zones-settings-update", {**ORG, "setting": "development_mode", "value": "on"}),)
        self.assertFalse(_passes("cf-devmode-purge.fr", *both))
        self.assertTrue(_passes("cf-devmode-purge.fr", *both, ("cache-purge-everything", ORG)))
        self.assertTrue(_passes("cf-bucket-clarify.zh", ("r2-buckets-list", {})))
        self.assertFalse(_passes("cf-bucket-clarify.zh", ("r2-buckets-delete", {"name": "media-assets"})))
        self.assertFalse(
            _passes(
                "cf-r2-bucket.pt",
                ("r2-buckets-create", {"name": "invoices-archive"}),
                ("tunnels-create", {"name": "t"}),
            )
        )


if __name__ == "__main__":
    unittest.main()

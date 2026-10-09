"""Provider-free checks for the long-journey skill evaluation harness.

A scripted fake model proves the simulator, the outcome check, the skill the harness learns, and the budget stop; it
says nothing about how a real model behaves.
"""

import unittest
from unittest import mock

import agent_runtime
import clarification
import memory
import model_usage
from eval import journeys
from langchain_core.messages import AIMessage
from tool_fake import PROVIDER
from tool_fake import call as _call
from tool_fake import runtime as _runtime


def _scenario(scenario_id: str) -> journeys.Scenario:
    return next(scenario for scenario in journeys.SCENARIOS if scenario.id == scenario_id)


ZONE = {"zone_id": journeys.ZONE_ID}
WWW = {**ZONE, "type": "A", "name": "www.exemplo.com", "content": "198.51.100.7"}


def _update(record_id: str) -> agent_runtime.AgentRuntime:
    return _runtime(
        AIMessage(content="", tool_calls=[_call(journeys.DNS, "list-zones", {}, "c1")]),
        AIMessage(content="", tool_calls=[_call(journeys.DNS, "list-dns-records", {**ZONE, "name": "www"}, "c2")]),
        AIMessage(
            content="", tool_calls=[_call(journeys.DNS, "replace-dns-record", {**WWW, "record_id": record_id}, "c3")]
        ),
        AIMessage(content="Updated www.exemplo.com."),
    )


class JourneyEvalTests(unittest.TestCase):
    def test_scenarios_are_valid_and_none_succeeds_without_an_action(self):
        journeys.validate_scenarios()
        self.assertTrue(any(len(scenario.assistants) == 2 for scenario in journeys.SCENARIOS))
        self.assertTrue(all(len(scenario.requests) >= 3 for scenario in journeys.SCENARIOS))

    def test_a_listed_record_replaced_by_its_id_succeeds_and_records_only_the_structure(self):
        scenario = _scenario("dns-update")
        outcome = journeys.run_request(
            _update(journeys._record_id("www.exemplo.com")), PROVIDER, scenario, scenario.requests[0], "j:1", []
        )
        self.assertTrue(outcome.succeeded)
        self.assertEqual((outcome.ended, outcome.rounds, outcome.repeated_writes), ("reply", 3, 0))
        self.assertEqual(journeys.repeated_writes([("ensure-dns-record", WWW)] * 2 + [("list-zones", {})] * 2), 1)
        self.assertEqual(
            outcome.steps,
            (
                ("dns", "list-zones", (), False),
                ("dns", "list-dns-records", ("name", "zone_id"), False),
                ("dns", "replace-dns-record", ("content", "name", "record_id", "type", "zone_id"), False),
            ),
        )
        self.assertEqual(outcome.failed_calls, 0)
        self.assertNotIn("198.51.100.7", repr(outcome.steps))

    def test_a_write_to_another_zone_fails_and_changes_nothing(self):
        other = {**WWW, "zone_id": "f" * 32}
        self.assertEqual(journeys._simulate("ensure-dns-record", other, {}), {"error": "zone-not-found"})
        self.assertEqual(journeys.records([("ensure-dns-record", other)]), {})
        self.assertIn("results", journeys._simulate("search-web", {"query": "ip"}, {"service": "a.example"}))

    def test_only_a_lookup_naming_the_service_reveals_its_address(self):
        facts = _scenario("research-then-record").requests[0].facts
        page = f"https://{facts['service']}/ip"
        self.assertEqual(journeys._simulate("search-web", {"query": "current ip address"}, facts), {"results": []})
        found = journeys._simulate("search-web", {"query": f"{facts['service'].upper()} ip"}, facts)
        self.assertEqual([result["url"] for result in found["results"]], [page])
        pages = journeys._simulate("read-pages", {"urls": ["https://other.example/ip", page]}, facts)["pages"]
        self.assertEqual([facts["ip"] in item["text"] for item in pages], [False, True])
        self.assertEqual(journeys._simulate("read-pages", {"urls": page}, facts), {"pages": []})

    def test_blind_rules_surface_only_as_errors_and_failed_calls_never_count_as_writes(self):
        scenario = _scenario("blind-update-by-id")
        self.assertEqual(journeys._simulate("ensure-dns-record", WWW, {}, scenario.rules), {"error": "record-exists"})
        self.assertIn("record", journeys._simulate("ensure-dns-record", WWW, {}))
        absolute = _scenario("blind-absolute-names").rules
        self.assertEqual(journeys._simulate("ensure-dns-record", WWW, {}, absolute)["error"], "invalid-name")
        self.assertIn(
            "record", journeys._simulate("ensure-dns-record", {**WWW, "name": "www.exemplo.com."}, {}, absolute)
        )
        runtime = _runtime(
            AIMessage(content="", tool_calls=[_call(journeys.DNS, "ensure-dns-record", WWW, "c1")]),
            AIMessage(content="Updated www.exemplo.com."),
        )
        outcome = journeys.run_request(runtime, PROVIDER, scenario, scenario.requests[0], "j:4", [])
        self.assertEqual((outcome.succeeded, outcome.failed_calls), (False, 1))
        self.assertEqual(outcome.steps[0][3], True)

    def test_a_replace_with_an_unknown_record_id_changes_nothing(self):
        scenario = _scenario("dns-update")
        self.assertEqual(
            journeys._simulate("replace-dns-record", {**WWW, "record_id": "f" * 32}, {}), {"error": "record-not-found"}
        )
        outcome = journeys.run_request(_update("f" * 32), PROVIDER, scenario, scenario.requests[0], "j:2", [])
        self.assertFalse(outcome.succeeded)

    def test_either_valid_write_counts_and_a_question_before_acting_fails(self):
        self.assertEqual(
            journeys.records([("ensure-dns-record", {**WWW, "name": "www"})]), {"www.exemplo.com": "198.51.100.7"}
        )
        cname = ("ensure-dns-record", {**WWW, "type": "CNAME"})
        self.assertEqual(journeys.records([cname]), {"CNAME www.exemplo.com": "198.51.100.7"})
        update = _scenario("dns-update").requests[0]
        self.assertTrue(update.succeeded([("ensure-dns-record", WWW)]))
        # An extra, unrequested write fails the request even when the requested record is right.
        self.assertFalse(update.succeeded([("ensure-dns-record", WWW), cname]))
        self.assertFalse(update.succeeded([("ensure-dns-record", WWW), ("ensure-dns-record", {**WWW, "name": "api"})]))
        question = {
            "question": "Which address?",
            "options": [{"label": "Look it up", "description": ""}, {"label": "I will say", "description": ""}],
            "default_index": 0,
        }
        runtime = _runtime(
            AIMessage(content="", tool_calls=[{"name": clarification.TOOL_NAME, "args": question, "id": "q1"}])
        )
        scenario = _scenario("migrate-origin")
        outcome = journeys.run_request(runtime, PROVIDER, scenario, scenario.requests[0], "j:3", [])
        self.assertEqual((outcome.succeeded, outcome.ended, outcome.rounds), (False, "clarification", 0))

    def test_learned_skills_are_what_the_brain_admits_and_stay_the_newest_eight(self):
        steps = [["dns", "list-zones", []], ["dns", "ensure-dns-record", ["content", "name", "type", "zone_id"]]]
        skill = journeys.learned_skill((journeys.DNS,), steps)
        self.assertEqual(memory.canonical_skills([skill]), (skill,))
        self.assertIsNone(journeys.learned_skill((journeys.DNS,), steps[:1]))
        self.assertEqual(journeys.remember([skill], None), [skill])
        others = [
            journeys.learned_skill((journeys.DNS,), [["dns", "list-zones", []]] * count)
            for count in range(2, 2 + memory.MAX_SKILLS)
        ]
        kept: list[dict[str, object]] = []
        for learned in [skill, *others]:
            kept = journeys.remember(kept, learned)
        self.assertEqual(len(kept), memory.MAX_SKILLS)
        self.assertNotIn(skill, kept)
        kept = journeys.remember(kept, others[0])
        self.assertEqual((len(kept), kept[-1]), (memory.MAX_SKILLS, others[0]))
        memory.canonical_skills(kept)

    def test_evaluation_stops_before_spending_past_its_budget(self):
        stopped = journeys.evaluate(_runtime(), PROVIDER, 0.0)
        self.assertEqual((stopped["stopped"], stopped["seed"]), ("budget", "journeys"))
        self.assertEqual(journeys.evaluate(_runtime(), PROVIDER, 0.0, "bulk-records", "s")["stopped"], "budget")

    def test_each_request_reserves_before_dispatch_and_unknown_usage_keeps_its_reservation(self):
        floor = journeys.eval_cost.call_bound("gpt-6-luna", *journeys.REQUEST_RESERVE_TOKENS)
        runtime = _runtime(AIMessage(content="Done."))
        report = journeys.evaluate(runtime, PROVIDER, 100 * floor, "dns-update", "s")
        self.assertNotIn("stopped", report)
        for mode in journeys.MODES:
            summary = report[f"dns-update/{mode}/cost"]
            # The fake model reports no usage, so every request's cost is unknown rather than zero.
            self.assertEqual((summary["attempts"], summary["unknown_usage_attempts"]), (3, 3))
            self.assertFalse(summary["usd_known"])
        self.assertEqual(report["budget"]["unknown_settlements"], 9)
        self.assertAlmostEqual(report["budget"]["spent_usd"], round(9 * floor, 6))
        stopped = journeys.evaluate(_runtime(AIMessage(content="Done.")), PROVIDER, 2.5 * floor, "dns-update", "s")
        self.assertEqual(stopped["stopped"], "budget")
        self.assertEqual(stopped["budget"]["unknown_settlements"], 2)

    def test_mode_order_is_random_per_scenario_and_recorded(self):
        orders = {
            tuple(journeys.evaluate(_runtime(), PROVIDER, 0.0, "dns-update", f"seed-{index}")["dns-update/order"])
            for index in range(12)
        }
        self.assertGreater(len(orders), 1)
        self.assertTrue(all(sorted(order) == sorted(journeys.MODES) for order in orders))

    def test_an_outcome_prices_cache_categories_and_labels_unreported_usage_unknown(self):
        scenario = _scenario("dns-update")
        usage = {**dict.fromkeys(model_usage.FIELDS, 0), "model_calls": 4, "input_tokens": 1000, "output_tokens": 10}
        usage |= {"cache_read_tokens": 400, "cache_write_tokens": 100}
        with mock.patch.object(model_usage, "measure", lambda work: (work(), usage)):
            outcome = journeys.run_request(
                _update(journeys._record_id("www.exemplo.com")), PROVIDER, scenario, scenario.requests[0], "j:5", []
            )
        # Luna: 500 fresh + 400 cache reads at a tenth + 100 unreported writes as input, 10 output.
        self.assertAlmostEqual(outcome.usd, (500 + 40 + 100) * 0.1e-6 + 10 * 0.5e-6)
        self.assertEqual((outcome.cache_write_tokens, outcome.usage_known), (100, True))
        with mock.patch.object(model_usage, "measure", lambda work: (work(), usage | {"unreported_calls": 1})):
            outcome = journeys.run_request(
                _update(journeys._record_id("www.exemplo.com")), PROVIDER, scenario, scenario.requests[0], "j:6", []
            )
        self.assertEqual((outcome.unreported_model_calls, outcome.usage_known), (1, False))


if __name__ == "__main__":
    unittest.main()

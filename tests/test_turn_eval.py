"""Provider-free and binary-free checks for the behavioral turn evaluation harness.

A scripted Brain proves the scorer's round, argument, status, and proxy checks; it says nothing about how a real model
or the real Brain behaves.
"""

import unittest
from unittest import mock

from brain_script import ONE_CALL, PROVIDER, ScriptedBrain, usage
from eval import brain_api, turns


def _case(case_id: str) -> turns.TurnCase:
    return next(case for case in turns.TURN_CASES if case.id == case_id)


EXACT = {"zone": "example.com", "type": "A", "name": "www", "content": "203.0.113.10"}
ANA = ("messages", "send-message", {"to": "ana", "text": "Deploy done"})
BRUNO = ("messages", "send-message", {"to": "bruno", "text": "Deploy done"})
ZONES = ("dns", "list-zones", {})


def turn(brain: ScriptedBrain, case_id: str, index: int = 0) -> bool:
    return turns.run_turn(brain, PROVIDER, _case(case_id), index, lambda _counts: None)


def outcome(brain: ScriptedBrain, case_id: str, index: int = 0) -> bool:
    return turns.run_outcome(brain, PROVIDER, _case(case_id), index, lambda _counts: None)


class TurnEvalTests(unittest.TestCase):
    def test_corpus_is_valid_and_bounds_its_provider_calls(self):
        turns.validate_corpus()
        rounds = sum(len(case.rounds) + 1 for case in turns.TURN_CASES)
        self.assertEqual(
            turns.logical_model_invocations(),
            turns.ATTEMPTS * (2 * rounds + len(turns.PLAN_CASES)),
        )
        self.assertTrue(any(len(current.actions) == 2 for case in turns.TURN_CASES for current in case.rounds))
        self.assertTrue(any(not case.rounds for case in turns.TURN_CASES))

    def test_exact_action_arguments_and_grounded_reply_pass(self):
        brain = ScriptedBrain([("dns", "create-record", EXACT)], "Created the www record in example.com for you.")
        self.assertTrue(turn(brain, "create-record-exact-en"))
        (_kind, context, sent), (_resume, _context, results) = brain.calls
        self.assertEqual(
            (context.thread_id, sent["message"]),
            ("eval:create-record-exact-en:0", _case("create-record-exact-en").message),
        )
        self.assertEqual(results, {"i0-0": {"status": "ok", "output": {}}})

    def test_wrong_argument_fails_the_round(self):
        brain = ScriptedBrain([("dns", "create-record", {**EXACT, "name": "www.example.com"})])
        self.assertFalse(turn(brain, "create-record-exact-en"))

    def test_parallel_round_is_order_independent_but_exact(self):
        self.assertTrue(
            turn(ScriptedBrain([BRUNO, ANA], "Sent the message to ana and bruno."), "two-messages-one-round-en")
        )
        self.assertFalse(turn(ScriptedBrain([ANA]), "two-messages-one-round-en"))

    def test_an_unexpected_action_or_extra_round_fails(self):
        self.assertFalse(
            turn(ScriptedBrain([("messages", "send-message", {"to": "bruno", "text": "hi"})]), "genesis-injection-en")
        )
        self.assertFalse(turn(ScriptedBrain([ZONES], [ZONES]), "list-zones-pt"))

    def test_outcomes_score_final_writes_whatever_the_round_structure(self):
        create = ("dns", "create-record", EXACT)
        # An extra lookup round fails the contract test but is a correct outcome.
        self.assertTrue(outcome(ScriptedBrain([ZONES], [create], "Done."), "create-record-exact-en"))
        self.assertFalse(outcome(ScriptedBrain([create], [create], "Done."), "create-record-exact-en", 1))
        wrong = ("dns", "create-record", {**EXACT, "name": "api"})
        self.assertFalse(outcome(ScriptedBrain([wrong]), "create-record-exact-en", 2))
        self.assertTrue(outcome(ScriptedBrain([ANA], [BRUNO], "Ok."), "two-messages-one-round-en"))
        self.assertFalse(outcome(ScriptedBrain([ANA], "Only ana."), "two-messages-one-round-en", 1))
        self.assertTrue(outcome(ScriptedBrain("Hi!"), "greeting-en"))
        with mock.patch.object(turns, "OUTCOME_ROUNDS", 2):
            self.assertFalse(outcome(ScriptedBrain([ZONES], [ZONES], [ZONES]), "list-zones-pt"))

    def test_reply_proxies_require_a_marker_and_the_user_language(self):
        grounded = ScriptedBrain([ZONES], "Você tem duas zonas: example.com e shimpz.dev.")
        self.assertTrue(turn(grounded, "list-zones-pt"))
        english = ScriptedBrain([ZONES], "You have two zones: example.com and shimpz.dev.")
        self.assertFalse(turn(english, "list-zones-pt"))
        self.assertEqual(turns.language_proxy("Olá, como posso ajudar você com a sua zona?"), "pt")
        self.assertEqual(turns.language_proxy("12345"), "")
        # Replies a live gpt-6-luna run produced (2026-09-28) that the first proxy misjudged.
        self.assertEqual(turns.language_proxy("Hi there! We’re doing well, thanks for asking. How can we help?"), "en")
        failed = _case("failed-action-pt")
        reply = "Não foi possível criar o registro TXT: a zona `missing.dev` não foi encontrada."
        self.assertTrue(turns._reply_matches(failed, reply))
        self.assertFalse(turns._reply_matches(failed, "Registro TXT criado na zona missing.dev."))
        self.assertFalse(turns._reply_matches(failed, "O registro TXT foi encontrado e criado na zona missing.dev."))

    def test_plan_cases_compare_the_exact_selection(self):
        brain = mock.Mock()
        brain.capability_plan.return_value = {
            "status": "install-required",
            "assistant_ids": ["shimpz-cloudflare"],
            "usage": ONE_CALL,
        }
        recorded = []
        self.assertTrue(turns.run_plan(brain, PROVIDER, turns.PLAN_CASES[0], recorded.append))
        self.assertEqual(recorded, [ONE_CALL])
        candidates = brain.capability_plan.call_args.args[2]
        self.assertEqual(set(candidates[0]), {"id", "name", "summary", "actions", "integrations"})
        brain.capability_plan.return_value = {"status": "sufficient", "assistant_ids": [], "usage": ONE_CALL}
        self.assertFalse(turns.run_plan(brain, PROVIDER, turns.PLAN_CASES[0], recorded.append))

    def test_out_of_scope_and_injection_replies_must_steer_to_the_enabled_capability(self):
        self.assertFalse(turn(ScriptedBrain("Sorry, I can't do that."), "out-of-scope-en"))
        steered = "I can't book flights, but I can manage your DNS zones and records."
        self.assertTrue(turn(ScriptedBrain(steered), "out-of-scope-en"))
        self.assertFalse(turn(ScriptedBrain("I can help with many things."), "genesis-injection-en"))


class ScoringTests(unittest.TestCase):
    def test_scoring_counts_attempts_and_treats_refusals_and_provider_failures_as_misses(self):
        outcomes = iter([True, brain_api.BrainError("/v1/turns", 502), brain_api.BrainError("/v1/turns", 400)])

        def attempt(_case, _index, _record):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        [scored] = turns._score((turns.PLAN_CASES[0],), attempt, "gpt-6-luna")
        self.assertEqual((scored["id"], scored["passed"], scored["required"]), ("plan-dns-en", 1, turns.ATTEMPTS))
        # A failed operation reported no usage, so those attempts' cost is unknown.
        self.assertEqual((scored["cost"]["attempts"], scored["cost"]["unknown_usage_attempts"]), (3, 2))

    def test_a_harness_or_brain_failure_is_never_scored_as_a_miss(self):
        for status in (401, 422, 503, None):

            def attempt(_case, _index, _record, status=status):
                raise brain_api.BrainError("/v1/turns", status)

            with self.subTest(status=status), self.assertRaises(brain_api.BrainError):
                turns._score((turns.PLAN_CASES[0],), attempt, "gpt-6-luna")

    def test_each_attempt_is_priced_from_its_reported_usage_and_unreported_usage_is_unknown(self):
        reports = iter([[ONE_CALL], [ONE_CALL | {"unreported_calls": 1}], [ONE_CALL, ONE_CALL]])

        def attempt(_case, index, record):
            for counts in next(reports):
                record(counts)
            return index != 1

        [scored] = turns._score((turns.PLAN_CASES[0],), attempt, "gpt-6-luna")
        self.assertEqual(scored["passed"], 2)
        self.assertAlmostEqual(scored["cost"]["usd"], 4 * (1000 * 0.1e-6 + 10 * 0.5e-6), places=6)
        self.assertEqual((scored["cost"]["usd_known"], scored["cost"]["unknown_usage_attempts"]), (False, 1))

    def test_a_report_scores_every_section_with_its_effort(self):
        brain = ScriptedBrain("Hi!", "Hi!", counts=usage())
        with (
            mock.patch.object(turns, "TURN_CASES", (turns.TURN_CASES[1],)),
            mock.patch.object(turns, "PLAN_CASES", ()),
            mock.patch.object(turns, "ATTEMPTS", 1),
        ):
            report = turns.evaluate(brain, PROVIDER)
        self.assertEqual((report["usd"], report["usd_known"], report["turn_effort"]), (0, True, "low"))
        self.assertEqual((report["cases"], report["passing_cases"], report["passing_outcomes"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()

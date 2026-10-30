"""Provider-free and binary-free checks for the fixed intent-route semantic evaluation."""

import unittest
from unittest import mock

from brain_script import ONE_CALL, PROVIDER
from eval import brain_api
from eval import intent_route as semantics


def _route(intent: str, **fields: object) -> dict[str, object]:
    answer = {"intent": intent, "query": "", "assistant_ids": [], "reply": "", "task_follows": False}
    return {**answer, **fields, "usage": ONE_CALL}


class IntentRouteSemanticsTests(unittest.TestCase):
    def test_corpus_admits_every_case_without_provider_access(self):
        semantics.validate_corpus()
        self.assertEqual(
            {case.intent for case in semantics.CASES},
            {"ordinary-task", "assistant-install", "assistant-uninstall", "unresolved"},
        )
        installs = [case for case in semantics.CASES if case.intent == "assistant-install" and case.target]
        self.assertEqual({case.task_follows for case in installs}, {False, True})

    def test_oracle_requires_the_exact_task_continuation(self):
        case = next(item for item in semantics.CASES if item.id == "install-then-search-pt")
        self.assertTrue(semantics._matches(case, semantics.Route("assistant-install", "exa", task_follows=True)))
        self.assertFalse(semantics._matches(case, semantics.Route("assistant-install", "exa")))
        only = next(item for item in semantics.CASES if item.id == "install-pt")
        self.assertFalse(
            semantics._matches(only, semantics.Route("assistant-install", "cloudflare", task_follows=True))
        )

    def test_oracle_rejects_wrong_or_mixed_lifecycle_targets(self):
        case = next(item for item in semantics.CASES if item.id == "explicit-over-reference")
        self.assertTrue(semantics._matches(case, semantics.Route("assistant-install", "whatsapp")))
        self.assertFalse(semantics._matches(case, semantics.Route("assistant-install", "cloudflare")))
        self.assertFalse(semantics._matches(case, semantics.Route("assistant-install", "whatsapp cloudflare")))

    def test_each_case_sends_teams_request_for_its_mode_and_context(self):
        selection = semantics.request(next(item for item in semantics.CASES if item.id == "select-uninstall"))
        self.assertEqual((selection.expected_intent, selection.locale), ("assistant-uninstall", "en"))
        self.assertEqual([item["id"] for item in selection.candidates], ["shimpz-cloudflare", "shimpz-whatsapp"])
        pronoun = semantics.request(next(item for item in semantics.CASES if item.id == "reference-pronoun"))
        self.assertEqual(pronoun.lifecycle_reference, {"id": "shimpz-cloudflare", "name": "Shimpz Cloudflare"})
        self.assertIsNone(pronoun.expected_intent)
        answer = semantics.request(next(item for item in semantics.CASES if item.id == "answer-after-guidance"))
        self.assertEqual(
            answer.conversation[0], {"role": "user", "text": "Quero desinstalar um Assistant.", "truncated": False}
        )
        self.assertEqual(
            semantics.request(next(item for item in semantics.CASES if item.id == "ordinary-pt")).locale, "pt"
        )

    def test_runner_counts_each_attempt_and_reports_only_case_identifiers(self):
        case = next(item for item in semantics.CASES if item.id == "select-uninstall")
        client = mock.Mock()
        client.intent_route.side_effect = [
            _route("assistant-uninstall", assistant_ids=["shimpz-whatsapp"]),
            _route("unresolved", reply="Which Assistant?"),
            brain_api.BrainError("/v1/intent-route", 502),
        ]
        with mock.patch.object(semantics, "CASES", (case,)):
            result = semantics.evaluate(client, PROVIDER)
        self.assertEqual(result["cases"], [{"id": "select-uninstall", "passed": 1, "rejected": 1, "required": 3}])
        self.assertEqual(result["passing_cases"], 0)
        self.assertEqual(client.intent_route.call_count, semantics.ATTEMPTS)

    def test_a_refused_request_stops_the_run_instead_of_counting_a_miss(self):
        client = mock.Mock()
        client.intent_route.side_effect = brain_api.BrainError("/v1/intent-route", 400)
        with mock.patch.object(semantics, "CASES", semantics.CASES[:1]), self.assertRaises(brain_api.BrainError):
            semantics.evaluate(client, PROVIDER)


if __name__ == "__main__":
    unittest.main()

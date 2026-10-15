"""Provider-free checks for the fixed intent-route semantic evaluation."""

import unittest
from unittest import mock

import agent_runtime
import intent_route
from eval import intent_route as semantics


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
        self.assertTrue(
            semantics._matches(case, intent_route.IntentRoute("assistant-install", "exa", task_follows=True))
        )
        self.assertFalse(semantics._matches(case, intent_route.IntentRoute("assistant-install", "exa")))
        only = next(item for item in semantics.CASES if item.id == "install-pt")
        self.assertFalse(
            semantics._matches(only, intent_route.IntentRoute("assistant-install", "cloudflare", task_follows=True))
        )

    def test_oracle_rejects_wrong_or_mixed_lifecycle_targets(self):
        case = next(item for item in semantics.CASES if item.id == "explicit-over-reference")
        self.assertTrue(semantics._matches(case, intent_route.IntentRoute("assistant-install", "whatsapp")))
        self.assertFalse(semantics._matches(case, intent_route.IntentRoute("assistant-install", "cloudflare")))
        self.assertFalse(semantics._matches(case, intent_route.IntentRoute("assistant-install", "whatsapp cloudflare")))

    def test_runner_counts_each_attempt_and_reports_only_case_identifiers(self):
        case = next(item for item in semantics.CASES if item.id == "select-uninstall")
        with (
            mock.patch.object(semantics, "CASES", (case,)),
            mock.patch.object(
                intent_route,
                "create",
                side_effect=[
                    intent_route.IntentRoute("assistant-uninstall", assistant_ids=("shimpz-whatsapp",)),
                    intent_route.IntentRoute("unresolved", reply="Which Assistant?"),
                    agent_runtime.ProviderResponseError("model provider response failed"),
                ],
            ) as route,
        ):
            result = semantics.evaluate(mock.Mock(), mock.Mock())
        self.assertEqual(result["cases"], [{"id": "select-uninstall", "passed": 1, "rejected": 1, "required": 3}])
        self.assertEqual(result["passing_cases"], 0)
        self.assertEqual(route.call_count, semantics.ATTEMPTS)


if __name__ == "__main__":
    unittest.main()

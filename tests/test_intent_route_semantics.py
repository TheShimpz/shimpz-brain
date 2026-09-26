"""Provider-free checks for the fixed intent-route semantic evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
                    intent_route.IntentRoute("assistant-uninstall", assistant_ids=("shimpz-whatsapp",)),
                ],
            ) as route,
        ):
            result = semantics.evaluate(mock.Mock(), mock.Mock())
        self.assertEqual(result["cases"], [{"id": "select-uninstall", "passed": 2, "required": 3}])
        self.assertEqual(result["passing_cases"], 0)
        self.assertEqual(route.call_count, semantics.ATTEMPTS)

    def test_key_reader_rejects_symlink_and_wide_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"
            path.write_bytes(b"a" * 16 + b"\n")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                semantics._key(path)
            path.chmod(0o600)
            self.assertEqual(semantics._key(path), "a" * 16)
            alias = Path(directory) / "alias"
            alias.symlink_to(path)
            with self.assertRaises(OSError):
                semantics._key(alias)


if __name__ == "__main__":
    unittest.main()

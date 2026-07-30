from __future__ import annotations

import copy
import importlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
EGRESS = ROOT / "egress"
sys.path.insert(0, str(EGRESS))
app = importlib.import_module("app")
policy = importlib.import_module("policy")


class BrainEgressPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = json.loads((ROOT / "model_catalog.json").read_text(encoding="utf-8"))

    def _write(self, value: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_canonical_catalog_derives_only_exact_provider_hosts(self) -> None:
        hosts = policy.load_provider_hosts(ROOT / "model_catalog.json")

        self.assertEqual(hosts, frozenset({"api.anthropic.com", "api.openai.com"}))
        self.assertTrue(app.permitted("api.openai.com", 443, hosts))
        self.assertTrue(app.permitted("API.OPENAI.COM.", 443, hosts))
        self.assertFalse(app.permitted("sub.api.openai.com", 443, hosts))
        self.assertFalse(app.permitted("api.openai.com", 80, hosts))
        self.assertFalse(app.permitted("*", 443, hosts))

    def test_missing_and_malformed_catalogs_are_refused(self) -> None:
        missing = Path(tempfile.gettempdir()) / "shimpz-missing-brain-egress-catalog"
        missing.unlink(missing_ok=True)
        with self.assertRaises(policy.ProviderPolicyError):
            policy.load_provider_hosts(missing)
        with self.assertRaises(policy.ProviderPolicyError):
            policy.load_provider_hosts(self._write(b"{"))

    def test_inexact_empty_duplicate_and_wildcard_policies_are_refused(self) -> None:
        invalid_catalogs = []

        inexact = copy.deepcopy(self.catalog)
        inexact["unexpected"] = True
        invalid_catalogs.append(inexact)

        empty = copy.deepcopy(self.catalog)
        empty["providers"] = []
        invalid_catalogs.append(empty)

        duplicate = copy.deepcopy(self.catalog)
        duplicate["providers"][1]["id"] = duplicate["providers"][0]["id"]
        invalid_catalogs.append(duplicate)

        wildcard = copy.deepcopy(self.catalog)
        wildcard["providers"][0]["credential_validation"]["host"] = "*"
        invalid_catalogs.append(wildcard)

        invalid_default = copy.deepcopy(self.catalog)
        invalid_default["default_provider"] = "missing"
        invalid_catalogs.append(invalid_default)

        for candidate in invalid_catalogs:
            with self.subTest(candidate=candidate), self.assertRaises(policy.ProviderPolicyError):
                policy.load_provider_hosts(self._write(candidate))

    def test_startup_exits_before_binding_when_policy_is_unavailable(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                app.policy,
                "load_provider_hosts",
                side_effect=policy.ProviderPolicyError("synthetic policy failure"),
            ),
            mock.patch.object(app, "Server") as server,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as caught,
        ):
            app.main()

        self.assertEqual(caught.exception.code, 1)
        server.assert_not_called()
        self.assertEqual(stderr.getvalue(), "brain-egress: provider policy is unavailable; refusing to start\n")


if __name__ == "__main__":
    unittest.main()

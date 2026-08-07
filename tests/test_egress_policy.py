from __future__ import annotations

import copy
import importlib
import io
import json
import stat
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

    def test_catalog_field_types_and_bounds_fail_closed(self) -> None:
        candidates = []
        invalid_title = copy.deepcopy(self.catalog)
        invalid_title["providers"][0]["title"] = ""
        candidates.append(invalid_title)
        invalid_models = copy.deepcopy(self.catalog)
        invalid_models["providers"][0]["models"] = []
        candidates.append(invalid_models)
        invalid_model_id = copy.deepcopy(self.catalog)
        invalid_model_id["providers"][0]["models"][0]["id"] = "INVALID"
        candidates.append(invalid_model_id)
        invalid_price = copy.deepcopy(self.catalog)
        invalid_price["providers"][0]["models"][0]["input_usd_per_million_cents"] = True
        candidates.append(invalid_price)
        invalid_default_model = copy.deepcopy(self.catalog)
        invalid_default_model["providers"][0]["default_model"] = "missing"
        candidates.append(invalid_default_model)
        invalid_schema = copy.deepcopy(self.catalog)
        invalid_schema["schema"] = True
        candidates.append(invalid_schema)

        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(policy.ProviderPolicyError):
                policy.load_provider_hosts(self._write(candidate))

    def test_catalog_file_metadata_must_be_regular_unique_and_bounded(self) -> None:
        path = self._write(self.catalog)
        metadata = path.stat()
        for replacement in (
            mock.Mock(st_mode=stat.S_IFDIR, st_nlink=1, st_size=metadata.st_size),
            mock.Mock(st_mode=stat.S_IFREG, st_nlink=2, st_size=metadata.st_size),
            mock.Mock(st_mode=stat.S_IFREG, st_nlink=1, st_size=0),
        ):
            with (
                self.subTest(replacement=replacement),
                mock.patch.object(Path, "stat", return_value=replacement),
                self.assertRaisesRegex(policy.ProviderPolicyError, "metadata is invalid"),
            ):
                policy.load_provider_hosts(path)

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

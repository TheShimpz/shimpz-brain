"""Provider-free checks that private evaluation files stay owner-only and outside every repository (ADR-0094)."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval import private


class PrivateFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_an_existing_file_is_made_owner_only_before_it_is_written(self):
        path = self.root / "transcript.jsonl"
        path.write_text("old\n", encoding="utf-8")
        path.chmod(0o644)
        private.write_private(path, "new\n")
        self.assertEqual((path.read_text(encoding="utf-8"), stat.S_IMODE(path.stat().st_mode)), ("new\n", 0o600))
        descriptor = private.open_private(path, append=True)
        os.write(descriptor, b"more\n")
        os.close(descriptor)
        self.assertEqual(path.read_text(encoding="utf-8"), "new\nmore\n")

    def test_links_repositories_and_foreign_files_are_refused(self):
        target = self.root / "target"
        target.write_text("keep", encoding="utf-8")
        (self.root / "link").symlink_to(target)
        (self.root / "hard").hardlink_to(target)
        (self.root / "linked-dir").symlink_to(self.root)
        repository = self.root / "repository"
        (repository / ".git").mkdir(parents=True)
        (repository / "nested").mkdir()
        refused = (
            self.root / "link",
            self.root / "hard",
            self.root / "linked-dir" / "file",
            repository / "transcript.jsonl",
            repository / "nested" / "transcript.jsonl",
            self.root / "missing" / "file",
            self.root / "nested" / ".." / "file",
        )
        for path in refused:
            with self.subTest(path=path.name), self.assertRaises(private.PrivateFileError):
                private.write_private(path, "secret")
        self.assertEqual(target.read_text(encoding="utf-8"), "keep")
        foreign = os.stat_result((stat.S_IFREG | 0o600, 0, 0, 1, os.getuid() + 1, 0, 0, 0, 0, 0))
        with mock.patch.object(private.os, "fstat", return_value=foreign), self.assertRaises(private.PrivateFileError):
            private.write_private(self.root / "foreign", "secret")


if __name__ == "__main__":
    unittest.main()

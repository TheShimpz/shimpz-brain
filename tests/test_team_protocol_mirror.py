"""The Brain's generated Team protocol mirror is exactly what its pins name, and only what the Brain consumes."""

import hashlib
import json
import re
import unittest
from pathlib import Path

from protocol.team.http.v1 import identifiers

MIRROR = Path(__file__).resolve().parents[1] / "protocol" / "team"
ROW = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)")
REPOSITORY = "https://github.com/TheShimpz/shimpz-teams"


def _files(directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


class TeamProtocolMirrorTests(unittest.TestCase):
    def test_the_mirror_holds_only_the_pinned_action_protocol_and_six_http_modules(self) -> None:
        self.assertEqual(
            _files(MIRROR),
            {
                "action/upstream.json",
                "action/v1/README.md",
                "action/v1/contract-files.sha256",
                "action/v1/schema.py",
                "http/upstream.json",
                "http/v1/identifiers.py",
                "http/v1/payload.py",
                "http/v1/phrase.py",
                "http/v1/purpose.py",
                "http/v1/routine.py",
                "http/v1/routine_context.py",
                "http/v1/routine_notice.py",
                "http/v1/routine_proposal.py",
                "http/v1/turn.py",
            },
        )

    def test_the_action_protocol_matches_its_manifest_and_pin(self) -> None:
        pin = json.loads((MIRROR / "action" / "upstream.json").read_bytes())
        manifest = (MIRROR / "action" / "v1" / "contract-files.sha256").read_bytes()
        self.assertEqual(set(pin), {"repository", "commit", "path", "tree", "contract_files_sha256"})
        self.assertEqual((pin["repository"], pin["path"]), (REPOSITORY, "protocol/action/v1"))
        self.assertRegex(pin["commit"], r"\A[0-9a-f]{40}\Z")
        self.assertEqual(hashlib.sha256(manifest).hexdigest(), pin["contract_files_sha256"])
        rows = [ROW.fullmatch(line) for line in manifest.decode("ascii").splitlines()]
        self.assertTrue(rows and all(rows))
        digests = {row[2]: row[1] for row in rows if row is not None}
        self.assertEqual(set(digests), _files(MIRROR / "action" / "v1") - {"contract-files.sha256"})
        for name, digest in digests.items():
            self.assertEqual(hashlib.sha256((MIRROR / "action" / "v1" / name).read_bytes()).hexdigest(), digest)

    def test_the_http_modules_match_their_pin_from_the_same_team_commit(self) -> None:
        pin = json.loads((MIRROR / "http" / "upstream.json").read_bytes())
        action = json.loads((MIRROR / "action" / "upstream.json").read_bytes())
        self.assertEqual(set(pin), {"repository", "commit", "path", "files"})
        self.assertEqual(
            (pin["repository"], pin["path"], pin["commit"]), (REPOSITORY, "protocol/http/v1", action["commit"])
        )
        self.assertEqual(set(pin["files"]), _files(MIRROR / "http" / "v1"))
        for name, digest in pin["files"].items():
            self.assertEqual(hashlib.sha256((MIRROR / "http" / "v1" / name).read_bytes()).hexdigest(), digest)

    def test_every_mirrored_identifier_rule_admits_exactly_its_kind(self) -> None:
        self.assertEqual(identifiers.canonical_team_id("team_1"), "team_1")
        for value in ("Team", "team-1", "t" * 41, None):
            self.assertIsNone(identifiers.canonical_team_id(value))
        self.assertEqual(identifiers.canonical_assistant_id("a" * 40), "a" * 40)
        self.assertIsNone(identifiers.canonical_assistant_id("a" * 41))
        self.assertEqual(identifiers.canonical_identifier("a" * 64), "a" * 64)
        self.assertIsNone(identifiers.canonical_identifier("a.b"))
        self.assertEqual(identifiers.canonical_action_id("a.b_c"), "a.b_c")
        self.assertIsNone(identifiers.canonical_action_id(7))


if __name__ == "__main__":
    unittest.main()

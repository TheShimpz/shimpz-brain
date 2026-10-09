"""Provider-free checks for the frozen tuning / held-out template split (ADR-0094)."""

import json
import tempfile
import unittest
from pathlib import Path

from eval import corpus, large_api, split


class SplitTests(unittest.TestCase):
    def test_the_frozen_split_follows_its_rule_and_covers_every_template_once(self):
        sets = split.load()
        self.assertEqual(sorted(sets["held-out"] + sets["tuning"]), sorted(t.id for t in corpus.TEMPLATES))
        self.assertEqual((len(sets["held-out"]), len(sets["tuning"])), (8, 7))
        held = [t for t in corpus.TEMPLATES if t.id in sets["held-out"]]
        self.assertTrue(any(len(t.needed) > 1 for t in held))
        self.assertEqual(split.part("dns-bulk.ja", sets), "held-out")
        self.assertEqual(split.part("dns-question.en", sets), "tuning")

    def test_the_large_api_split_is_frozen_by_the_same_rule(self):
        sets = split.load(split.LARGE_API_SPLIT)
        self.assertEqual(sorted(sets["held-out"] + sets["tuning"]), sorted(t.id for t in large_api.TEMPLATES))

    def test_an_edited_split_is_refused(self):
        data = json.loads(split.SPLIT.read_text(encoding="utf-8"))
        data["held-out"], data["tuning"] = data["tuning"], data["held-out"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "split.json")
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen split"):
                split.load(path)


if __name__ == "__main__":
    unittest.main()

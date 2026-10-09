"""Provider-free checks for whole-trajectory judging of precision-v3 episodes (ADR-0094)."""

import json
import tempfile
import unittest
from pathlib import Path

from eval.complex import judging
from eval.complex.model import Episode
from eval.complex.templates import TEMPLATES_BY_ID
from langchain_core.messages import AIMessage
from structured_fake import StructuredFakeModel

GOOD = judging.TrajectoryVerdict(
    turns_correct=True, unsupported_claim=False, honest_about_failures=True, language_matches=True, confident=True
)


def _trajectory(reply: str | None = "Done.") -> judging.Trajectory:
    template = TEMPLATES_BY_ID["s4-partial-bulk"]
    record = (
        {"assistant": "dns", "action": "update-record", "input": {}, "result": {}, "effect": "x", "secret": "y"},
        {"assistant": "dns", "action": "update-record", "input": {}, "failed": "unavailable"},
    )
    turns = (judging.Turn(0, "Point www, api, app.", None, record), judging.Turn(1, "What changed?", reply, ()))
    return judging.Trajectory(Episode(template, "ja"), turns)


class TrajectoryJudgeTests(unittest.TestCase):
    def test_the_judge_reads_every_turn_blinded_with_obligations_and_records(self):
        system, human = judging.prompt(_trajectory())
        body = json.loads(human.content)
        self.assertEqual(body["expected_reply_language"], "Japanese")
        steps = TEMPLATES_BY_ID["s4-partial-bulk"].steps
        self.assertEqual([turn["turn_obligation"] for turn in body["turns"]], [step.communicate for step in steps])
        self.assertIsNone(body["turns"][0]["assistant_reply"])
        self.assertEqual(body["turns"][0]["action_record"][1]["failed"], "unavailable")
        self.assertNotIn("secret", human.content)
        self.assertNotIn("effect", body["turns"][0]["action_record"][0])
        self.assertEqual([a["id"] for a in body["available_assistants"]], ["dns"])
        exposed = judging.Trajectory(_trajectory().episode, _trajectory().turns, ("tasks",))
        self.assertEqual(
            [a["id"] for a in json.loads(judging.prompt(exposed)[1].content)["available_assistants"]], ["tasks"]
        )
        text = (system.content + human.content).lower()
        for leaked in ("openai", "anthropic", "luna", "sonnet", " arm", "repetition", "baseline"):
            self.assertNotIn(leaked, text)

    def test_a_structured_verdict_is_parsed_and_a_bad_one_refused(self):
        model = StructuredFakeModel(responses=[AIMessage(content=GOOD.model_dump_json())])
        self.assertEqual(judging.judge(model, "openai", _trajectory()), GOOD)
        with self.assertRaises(judging.TrajectoryJudgeError):
            judging.judge(StructuredFakeModel(responses=[AIMessage(content="{}")]), "anthropic", _trajectory())

    def test_success_needs_every_criterion_and_the_tiebreak_settles_doubt(self):
        self.assertTrue(judging.succeeded(GOOD))
        for change in (
            {"turns_correct": False},
            {"unsupported_claim": True},
            {"honest_about_failures": False},
            {"language_matches": False},
        ):
            self.assertFalse(judging.succeeded(GOOD.model_copy(update=change)))
        unsure = GOOD.model_copy(update={"confident": False})
        second = GOOD.model_copy(update={"turns_correct": False})
        calls = []

        def tiebreak(trajectory):
            calls.append(trajectory)
            return second

        self.assertEqual(judging.decide(lambda _t: GOOD, tiebreak, _trajectory(), True).final, GOOD)
        self.assertEqual(calls, [])
        self.assertEqual(judging.decide(lambda _t: unsure, tiebreak, _trajectory(), True).final, second)
        disagree = judging.decide(lambda _t: GOOD, tiebreak, _trajectory(), False)
        self.assertEqual((disagree.primary, disagree.tiebreak), (GOOD, second))

    def test_the_calibration_set_covers_every_criterion_both_ways_and_is_checked(self):
        items = judging.calibration_items()
        self.assertGreaterEqual(len(items), 16)
        for name in judging.CRITERIA:
            self.assertEqual({expected[name] for _id, _t, expected in items}, {True, False}, name)
        self.assertEqual(len({item_id for item_id, _t, _e in items}), len(items))
        data = json.loads(judging.CALIBRATION.read_text(encoding="utf-8"))
        bad_cases = (
            {**data, "corpus": "precision-v2"},
            {**data, "version": "trajectory-v0"},
            {**data, "items": [{**data["items"][0], "expected": {"turns_correct": True}}]},
            {**data, "items": [{**data["items"][0], "expected": {**data["items"][0]["expected"], "turns_correct": 1}}]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "calibration.json")
            for bad in bad_cases:
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(bad=str(bad)[:40]), self.assertRaises(ValueError):
                    judging.calibration_items(path)
            path.write_text(json.dumps(data) + " ", encoding="utf-8")
            self.assertNotEqual(judging.identity(path), judging.identity())

    def test_agreement_counts_failures_and_reports_wilson_intervals(self):
        expected = dict.fromkeys(judging.CRITERIA, True) | {"unsupported_claim": False}
        summary = judging.agreement([(GOOD, expected), (None, expected)])
        self.assertEqual((summary["items"], summary["failed"]), (2, 1))
        self.assertEqual(summary["turns_correct"]["agree"], 1)
        self.assertEqual(judging.agreement([])["turns_correct"]["wilson95"], None)


if __name__ == "__main__":
    unittest.main()

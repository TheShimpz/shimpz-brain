"""Provider-free checks for whole-trajectory judging of precision-v3 episodes (ADR-0094)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from eval import judge, model_call
from eval.complex import judging
from eval.complex.model import Episode
from eval.complex.templates import TEMPLATES_BY_ID
from test_eval_judge import anthropic_answer, fake_client, openai_answer

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
        prompt = judging.prompt(_trajectory())
        body = json.loads(prompt.user)
        self.assertEqual(body["expected_reply_language"], "Japanese")
        steps = TEMPLATES_BY_ID["s4-partial-bulk"].steps
        self.assertEqual([turn["turn_obligation"] for turn in body["turns"]], [step.communicate for step in steps])
        self.assertIsNone(body["turns"][0]["assistant_reply"])
        self.assertEqual(body["turns"][0]["action_record"][1]["failed"], "unavailable")
        self.assertNotIn("secret", prompt.user)
        self.assertNotIn("effect", body["turns"][0]["action_record"][0])
        self.assertEqual([a["id"] for a in body["available_assistants"]], ["dns"])
        exposed = judging.Trajectory(_trajectory().episode, _trajectory().turns, ("tasks",))
        self.assertEqual([a["id"] for a in json.loads(judging.prompt(exposed).user)["available_assistants"]], ["tasks"])
        text = (prompt.system + prompt.user).lower()
        for leaked in ("openai", "anthropic", "luna", "sonnet", " arm", "repetition", "baseline"):
            self.assertNotIn(leaked, text)

    def test_a_structured_verdict_is_parsed_and_a_bad_one_refused(self):
        answers = [
            httpx.Response(200, json=openai_answer(GOOD.model_dump_json())),
            httpx.Response(200, json=anthropic_answer("{}")),
        ]
        with mock.patch.object(model_call.httpx, "Client", fake_client(lambda _request: answers.pop(0))):
            self.assertEqual(judging.judge(judge.judge_model("openai", "sk-test"), "openai", _trajectory()), GOOD)
            with self.assertRaises(judging.TrajectoryJudgeError):
                judging.judge(judge.judge_model("anthropic", "sk-test"), "anthropic", _trajectory())

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
        # Stored calibrations and verdicts carry this value; bump it only together with VERSION.
        self.assertEqual(judging.identity(), "sha256:046714111577da667a196f477239f1f0f415bb432269b68e0f13805494552bbe")

    def test_agreement_counts_failures_and_reports_wilson_intervals(self):
        expected = dict.fromkeys(judging.CRITERIA, True) | {"unsupported_claim": False}
        summary = judging.agreement([(GOOD, expected), (None, expected)])
        self.assertEqual((summary["items"], summary["failed"]), (2, 1))
        self.assertEqual(summary["turns_correct"]["agree"], 1)
        self.assertEqual(judging.agreement([])["turns_correct"]["wilson95"], None)


if __name__ == "__main__":
    unittest.main()

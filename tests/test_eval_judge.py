"""Provider-free checks for the blinded judges, their calibration sample, and the tiebreak (ADR-0094)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval import corpus, judge
from langchain_core.messages import AIMessage
from structured_fake import StructuredFakeModel

GOOD = judge.Verdict(
    reply_correct=True,
    unsupported_claim=False,
    asks_for_missing_information=False,
    language_matches=True,
    confident=True,
)


def _item(scenario_id: str = "dns-create.en", reply: str = "Created.") -> judge.Item:
    return judge.Item(
        corpus.SCENARIOS_BY_ID[scenario_id],
        ({"assistant": "dns", "action": "list-zones", "input": {}, "result": {}, "secret": "x"},),
        reply,
    )


class JudgeTests(unittest.TestCase):
    def test_the_judge_input_is_blinded_and_reference_grounded(self):
        system, human = judge.prompt(_item("dns-create.ja", "作成しました。"))
        body = json.loads(human.content)
        self.assertEqual(body["expected_reply_language"], "Japanese")
        self.assertEqual(body["reference_outcome"], corpus.SCENARIOS_BY_ID["dns-create.ja"].template.reference)
        self.assertEqual(
            body["action_record"], [{"assistant": "dns", "action": "list-zones", "input": {}, "result": {}}]
        )
        self.assertFalse(body["user_must_be_asked_for_missing_information"])
        scenario = corpus.SCENARIOS_BY_ID["dns-create.ja"]
        self.assertEqual([item["id"] for item in body["available_assistants"]], list(scenario.assistants))
        self.assertIn("create-record: Create one record", " ".join(body["available_assistants"][0]["actions"]))
        text = (system.content + human.content).lower()
        for leaked in ("openai", "anthropic", "luna", "sonnet", "arm", "repetition", "baseline"):
            self.assertNotIn(leaked, text.replace("alarm", ""))

    def test_a_structured_verdict_is_parsed_and_validated(self):
        StructuredFakeModel.structured.clear()
        model = StructuredFakeModel(responses=[AIMessage(content=GOOD.model_dump_json())])
        self.assertEqual(judge.judge(model, "openai", _item()), GOOD)
        self.assertEqual(StructuredFakeModel.structured[-1][0], judge.Verdict)
        with self.assertRaises(judge.JudgeError):
            judge.judge(StructuredFakeModel(responses=[AIMessage(content="{}")]), "anthropic", _item())

    def test_reply_success_requires_every_criterion_and_the_expected_clarification(self):
        self.assertTrue(judge.succeeded(GOOD, _item()))
        for change in (
            {"reply_correct": False},
            {"unsupported_claim": True},
            {"language_matches": False},
            {"asks_for_missing_information": True},
        ):
            self.assertFalse(judge.succeeded(GOOD.model_copy(update=change), _item()))
        asking = GOOD.model_copy(update={"asks_for_missing_information": True})
        self.assertTrue(judge.succeeded(asking, _item("dns-clarify.en")))
        self.assertFalse(judge.succeeded(GOOD, _item("dns-clarify.en")))

    def test_the_second_provider_decides_low_confidence_and_oracle_disagreement(self):
        other = GOOD.model_copy(update={"reply_correct": False})
        calls = []

        def tiebreak(item):
            calls.append(item)
            return other

        agreed = judge.decide(lambda _item: GOOD, tiebreak, _item(), oracle_passed=True)
        self.assertEqual((agreed.tiebreak, agreed.final, calls), (None, GOOD, []))
        disagreed = judge.decide(lambda _item: GOOD, tiebreak, _item(), oracle_passed=False)
        self.assertEqual((disagreed.primary, disagreed.tiebreak, disagreed.final), (GOOD, other, other))
        unsure = GOOD.model_copy(update={"confident": False})
        self.assertTrue(judge.needs_tiebreak(unsure, _item(), oracle_passed=True))
        # A correct reply flagged with an unsupported claim fails the task the oracle passed: the second judge decides.
        flagged = GOOD.model_copy(update={"unsupported_claim": True})
        self.assertTrue(judge.needs_tiebreak(flagged, _item(), oracle_passed=True))
        self.assertFalse(judge.needs_tiebreak(flagged, _item(), oracle_passed=False))
        self.assertEqual(len(calls), 1)


class CalibrationTests(unittest.TestCase):
    def test_the_adjudicated_sample_covers_every_criterion_both_ways(self):
        items = judge.calibration_items()
        self.assertEqual(len(items), 32)
        self.assertEqual(len({item_id for item_id, _, _ in items}), 32)
        for name in judge.CRITERIA:
            self.assertEqual({expected[name] for _, _, expected in items}, {True, False})
        self.assertEqual(len({item.scenario.locale for _, item, _ in items}), len(corpus.LOCALES))

    def test_a_sample_for_another_corpus_or_with_invalid_labels_is_refused(self):
        data = json.loads(judge.CALIBRATION.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "calibration.json")
            path.write_text(json.dumps({**data, "corpus": "precision-v0"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another corpus"):
                judge.calibration_items(path)
            data["items"][0]["expected"] = {"reply_correct": "yes"}
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid"):
                judge.calibration_items(path)

    def test_agreement_has_wilson_intervals_and_admits_only_a_clearly_agreeing_judge(self):
        items = judge.calibration_items()
        perfect = [(judge.Verdict(**expected, confident=True), expected) for _, _, expected in items]
        summary = judge.agreement(perfect)
        self.assertTrue(summary["admitted"])
        self.assertEqual((summary["items"], summary["failed"], summary["all_criteria"]["agree"]), (32, 0, 32))
        self.assertGreaterEqual(summary["reply_correct"]["wilson95"][0], 0.8)
        degraded = [(None, expected) for _, _, expected in items[:3]] + perfect[3:]
        summary = judge.agreement(degraded)
        self.assertEqual((summary["failed"], summary["unsupported_claim"]["agree"]), (3, 29))
        self.assertFalse(summary["admitted"])
        empty = judge.agreement([])
        self.assertEqual((empty["admitted"], empty["reply_correct"]["rate"]), (False, None))


if __name__ == "__main__":
    unittest.main()

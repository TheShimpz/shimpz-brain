"""Provider-free checks for judging journey attempts and building the sanitized report (ADR-0094)."""

from __future__ import annotations

import json
import runpy
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import model_usage
from eval import corpus, judge, precision, private
from eval import cost as eval_cost

GOOD = judge.Verdict(
    reply_correct=True,
    unsupported_claim=False,
    asks_for_missing_information=False,
    language_matches=True,
    confident=True,
)
BAD = GOOD.model_copy(update={"reply_correct": False})
USAGE = {**dict.fromkeys(eval_cost.FIELDS, 0), "model_calls": 2, "input_tokens": 1000, "output_tokens": 50}


def _attempt(scenario: str, arm: str = "a", repetition: int = 0, status: str = "completed", passed: bool = True):
    return {
        "campaign": "c",
        "provider": "openai",
        "model": "gpt-6-luna",
        "effort": "low",
        "arm": arm,
        "repetition": repetition,
        "scenario": scenario,
        "status": status,
        "failure": None,
        "clarification": False,
        "reply": "Done.",
        "ledger": [{"assistant": "dns", "action": "list-zones", "input": {}, "result": {}}],
        "oracle": {
            "passed": passed,
            "missing": 0,
            "wrong": int(not passed),
            "forbidden": 0,
            "wrong_scope": 0,
            "duplicates": 0,
        },
        "rounds": 2,
        "usage": dict(USAGE),
        "operations": 3,
        "usd": 0.001,
        "usage_known": True,
        "seconds_active": 1.0 + repetition,
        "seconds_wall": 2.0,
    }


def _decision(attempt, final=GOOD, tiebreak=None):
    return {
        "key": precision.attempt_key(attempt),
        "primary": GOOD.model_dump(),
        "tiebreak": None if tiebreak is None else tiebreak.model_dump(),
        "final": final.model_dump(),
    }


class JudgingTests(unittest.TestCase):
    def test_private_files_and_transcripts_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "t.jsonl")
            private.write_private(path, json.dumps({"a": 1}) + "\n\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(precision.read_jsonl(path), [{"a": 1}])
        attempt = _attempt("dns-create.en")
        self.assertEqual(precision.attempt_key(attempt), "c|openai|gpt-6-luna|a|0|dns-create.en")
        self.assertEqual(precision.judge_item(attempt).scenario.id, "dns-create.en")

    def test_metered_judgments_reserve_settle_and_keep_the_reservation_on_failure(self):
        budget = eval_cost.Budget(1.0)
        spend: list[eval_cost.Cost] = []
        with mock.patch.object(model_usage, "measure", lambda work: (work(), USAGE)):
            verdict = precision.metered("gpt-6-luna", lambda _item: GOOD, budget, spend)(
                precision.judge_item(_attempt("dns-create.en"))
            )
        self.assertEqual(verdict, GOOD)
        self.assertAlmostEqual(spend[0].usd, 1000 * 0.1e-6 + 50 * 0.5e-6)

        def fail(_item):
            raise judge.JudgeError("x")

        with self.assertRaises(judge.JudgeError):
            precision.metered("gpt-6-luna", fail, budget, spend)(precision.judge_item(_attempt("dns-create.en")))
        self.assertEqual(budget.summary()["unknown_settlements"], 1)
        with self.assertRaises(eval_cost.BudgetExhaustedError):
            precision.metered("gpt-6-luna", fail, eval_cost.Budget(0.0), spend)(
                precision.judge_item(_attempt("dns-create.en"))
            )

    def test_only_completed_attempts_are_judged_and_failed_judgments_stay_unjudged(self):
        attempts = [
            _attempt("dns-create.en"),
            _attempt("dns-create.pt", passed=False),
            _attempt("dns-create.de", status="turn-failed"),
            _attempt("dns-create.fr"),
        ]

        def primary(item):
            if item.scenario.locale == "fr":
                raise judge.JudgeError("x")
            return GOOD

        judged = precision.judge_attempts(attempts, primary, lambda _item: BAD, workers=2)
        self.assertEqual([item["key"].split("|")[-1] for item in judged], ["dns-create.en", "dns-create.pt"])
        # The primary called the failed-oracle attempt correct, so the second provider decided it.
        self.assertEqual((judged[0]["tiebreak"], judged[1]["final"]), (None, BAD.model_dump()))

    def test_calibration_reports_both_judges(self):
        def tiebreak(_item):
            raise eval_cost.BudgetExhaustedError("cap")

        expected = {item_id: labels for item_id, _, labels in judge.calibration_items()}
        by_reply = {item.reply: expected[item_id] for item_id, item, _ in judge.calibration_items()}
        summary = precision.calibrate(lambda item: judge.Verdict(**by_reply[item.reply], confident=True), tiebreak)
        self.assertTrue(summary["primary"]["admitted"])
        self.assertEqual((summary["tiebreak"]["failed"], summary["tiebreak"]["admitted"]), (32, False))


class ReportTests(unittest.TestCase):
    def test_outcomes(self):
        attempt = _attempt("dns-create.en")
        self.assertEqual(precision.outcome(_attempt("dns-create.en", status="budget-stopped"), None), "inconclusive")
        self.assertEqual(precision.outcome(_attempt("dns-create.en", status="brain-error"), None), "failure")
        self.assertEqual(precision.outcome(_attempt("dns-create.en", passed=False), None), "failure")
        self.assertEqual(precision.outcome(attempt, None), "inconclusive")
        self.assertEqual(precision.outcome(attempt, _decision(attempt)), "success")
        self.assertEqual(precision.outcome(attempt, _decision(attempt, BAD)), "failure")

    def test_the_report_is_sanitized_and_pairs_only_complete_pairs(self):
        attempts, judged = [], []
        for index, scenario in enumerate(s.id for s in corpus.SCENARIOS[:24]):
            for repetition in range(3):
                for arm in ("a", "b"):
                    attempt = _attempt(scenario, arm, repetition, passed=(index + repetition) % 4 != 0)
                    attempts.append(attempt)
                    judged.append(_decision(attempt, tiebreak=BAD if index == 0 else None))
        attempts.append(_attempt(corpus.SCENARIOS[30].id, "a", 0, status="budget-stopped"))
        attempts.append(_attempt(corpus.SCENARIOS[31].id, "b", 0, status="turn-failed"))
        report = precision.build_report(attempts, judged, {"seed": "s", "identical_arms": True})
        text = json.dumps(report)
        for leaked in ("Done.", "list-zones", '"ledger"', '"reply"', '"input"'):
            self.assertNotIn(leaked, text)
        self.assertEqual(report["schema"], precision.REPORT_SCHEMA)
        self.assertEqual(report["corpus"]["digest"], corpus.digest())
        run = report["runs"][0]
        self.assertEqual((run["arm"], run["conclusive"], run["inconclusive"]), ("a", 72, 1))
        self.assertEqual(run["attempts"]["budget-stopped"], 1)
        self.assertEqual(run["successes"], 54)
        self.assertIsNotNone(run["success"]["low"])
        self.assertIsNotNone(run["pass^3"]["mean"])
        self.assertIsNone(run["pass^5"]["mean"])
        self.assertEqual(run["cost"]["attempts"], 72)
        self.assertEqual(set(run["strata"]), set(precision.STRATA))
        self.assertEqual(run["usage"]["fresh_input_tokens"], 73 * 1000)
        paired = report["paired"][0]
        self.assertEqual((paired["complete_pairs"], paired["incomplete_pairs"]), (72, 2))
        self.assertEqual(paired["difference"]["mean"], 0.0)
        self.assertIsNotNone(report["pooled_identical_arms"][0]["pass^5"]["mean"])
        self.assertEqual(report["judges"]["tiebreaks"], 6)
        self.assertEqual(report["judges"]["tiebreak_changed_reply_correct"], 6)
        self.assertEqual(precision.build_report(attempts, judged, {})["pooled_identical_arms"], [])
        single = precision.build_report(attempts[:1], [], {})
        self.assertEqual((single["paired"], single["runs"][0]["mean_rounds"]), ([], 2.0))
        self.assertIsNone(precision._group([], "s", "empty")["mean_rounds"])


class CommandTests(unittest.TestCase):
    def test_validate_admits_every_scenario_in_brain(self):
        result = precision.validate()
        self.assertEqual((result["scenarios"], result["calibration_items"]), (120, 32))
        scenario = corpus.SCENARIOS_BY_ID["status-migrate.en"]
        self.assertEqual(len(precision.brain_assistants(scenario)), len(scenario.assistants))

    def test_the_module_runs_as_a_command(self):
        with (
            mock.patch("sys.argv", ["precision", "validate"]),
            mock.patch("builtins.print"),
            self.assertRaises(SystemExit) as raised,
        ):
            runpy.run_module("eval.precision", run_name="__main__")
        self.assertEqual(raised.exception.code, 0)

    def test_commands_write_private_judgments_and_a_report(self):
        attempt = _attempt("dns-create.en")
        structured = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "t.jsonl").write_text(json.dumps(attempt) + "\n", encoding="utf-8")
            (root / "key").write_text("sk-test-0123456789abcdef\n", encoding="utf-8")
            (root / "key").chmod(0o600)
            (root / "meta.json").write_text(json.dumps({"seed": "s"}), encoding="utf-8")
            keys = ["--key-file", str(root / "key"), "--tiebreak-key-file", str(root / "key"), "--cap", "1"]
            with (
                mock.patch.object(judge, "judge_model", return_value=structured),
                mock.patch.object(judge, "judge", return_value=GOOD) as judged,
                mock.patch.object(model_usage, "measure", lambda work: (work(), USAGE)),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(precision.main(["validate"]), 0)
                out = root / "judged.jsonl"
                self.assertEqual(
                    precision.main(["judge", "--transcript", str(root / "t.jsonl"), "--out", str(out), *keys]), 0
                )
                self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o600)
                calibration = root / "calibration.json"
                self.assertEqual(precision.main(["calibrate", "--out", str(calibration), *keys]), 0)
                self.assertIn("judge_budget", json.loads(calibration.read_text(encoding="utf-8")))
            self.assertEqual({call.args[1] for call in judged.call_args_list}, {"openai", "anthropic"})
            report = root / "report.json"
            arguments = ["report", "--transcript", str(root / "t.jsonl"), "--judged", str(out), "--out", str(report)]
            self.assertEqual(
                precision.main([*arguments, "--meta", str(root / "meta.json"), "--calibration", str(calibration)]), 0
            )
            self.assertIn("judge_calibration", json.loads(report.read_text(encoding="utf-8"))["meta"])
            self.assertEqual(precision.main(arguments), 0)
            with mock.patch("sys.stderr"):
                self.assertEqual(
                    precision.main(["judge", "--transcript", str(root / "missing"), "--out", str(out), *keys]), 2
                )


if __name__ == "__main__":
    unittest.main()

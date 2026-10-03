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
from eval import arms, corpus, judge, large_api, precision, private, split
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
        "judge_identity": judge.identity(),
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

    def test_metered_judgments_record_what_each_call_reported(self):
        spend: list[eval_cost.Cost] = []
        with mock.patch.object(model_usage, "measure", lambda work: (work(), USAGE)):
            verdict = precision.metered("gpt-6-luna", lambda _item: GOOD, spend)(
                precision.judge_item(_attempt("dns-create.en"))
            )
        self.assertEqual(verdict, GOOD)
        self.assertAlmostEqual(spend[0].usd, 1000 * 0.1e-6 + 50 * 0.5e-6)

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
        undispatched = _attempt(corpus.SCENARIOS[32].id, "a", 0, status="budget-stopped")
        attempts.append({**undispatched, "operations": 0, "usd": 0.0, "usage": dict.fromkeys(eval_cost.FIELDS, 0)})
        attempts.append(_attempt(corpus.SCENARIOS[31].id, "b", 0, status="turn-failed"))
        report = precision.build_report(attempts, judged, {"seed": "s", "identical_arms": True})
        text = json.dumps(report)
        for leaked in ("Done.", "list-zones", '"ledger"', '"reply"', '"input"'):
            self.assertNotIn(leaked, text)
        self.assertEqual(report["schema"], precision.REPORT_SCHEMA)
        self.assertEqual(
            report["corpora"], [{"id": corpus.CORPUS_ID, "digest": corpus.digest(), "scenarios": len(corpus.SCENARIOS)}]
        )
        run = report["runs"][0]
        self.assertEqual((run["arm"], run["conclusive"], run["inconclusive"]), ("a", 72, 2))
        self.assertEqual(run["attempts"]["budget-stopped"], 2)
        self.assertEqual(run["successes"], 54)
        self.assertIsNotNone(run["success"]["low"])
        self.assertIsNotNone(run["pass^3"]["mean"])
        self.assertIsNone(run["pass^5"]["mean"])
        # The paid budget-stopped attempt counts toward cost; an undispatched one does not.
        self.assertEqual((run["cost"]["attempts"], run["cost"]["inconclusive_usd"]), (73, 0.001))
        self.assertEqual(set(run["strata"]), set(precision.STRATA))
        self.assertEqual(run["usage"]["fresh_input_tokens"], 73 * 1000)
        paired = report["paired"][0]
        self.assertEqual((paired["complete_pairs"], paired["incomplete_pairs"]), (72, 3))
        self.assertEqual(paired["difference"]["mean"], 0.0)
        self.assertIsNotNone(report["pooled_identical_arms"][0]["pass^5"]["mean"])
        self.assertEqual(report["judges"]["tiebreaks"], 6)
        self.assertEqual(report["judges"]["tiebreak_changed_reply_correct"], 6)
        self.assertEqual(precision.build_report(attempts, judged, {})["pooled_identical_arms"], [])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            precision.build_report([*attempts, attempts[0]], judged, {})
        # A second campaign of the same scenarios is reported apart, never merged into the first one's pairs.
        other = [{**attempt, "campaign": "d"} for attempt in attempts[:4]]
        merged = precision.build_report([*attempts, *other], judged, {"seed": "s"})
        self.assertEqual([item["campaign"] for item in merged["paired"]], ["c", "d"])
        self.assertEqual(merged["paired"][0]["complete_pairs"], 72)
        sweep = [{**attempt, "arm": "high"} for attempt in attempts if attempt["arm"] == "b"]
        swept = precision.build_report([*attempts, *sweep], judged, {"seed": "s", "baseline_arm": "b"})
        self.assertEqual([(i["baseline"], i["candidate"]) for i in swept["paired"]], [("b", "a"), ("b", "high")])
        single = precision.build_report(attempts[:1], [], {})
        self.assertEqual((single["paired"], single["runs"][0]["mean_rounds"]), ([], 2.0))
        self.assertIsNone(precision._group([], "s", "empty")["mean_rounds"])


class CorpusIdentityTests(unittest.TestCase):
    def test_a_report_names_exactly_the_corpora_its_attempts_ran(self):
        large = [_attempt(scenario.id, "L1") for scenario in large_api.SCENARIOS[:2]]
        only_large = precision.build_report(large, [], {"seed": "s"})
        self.assertEqual(
            only_large["corpora"],
            [{"id": large_api.CORPUS_ID, "digest": large_api.digest(), "scenarios": len(large_api.SCENARIOS)}],
        )
        both = precision.build_report([*large, _attempt(corpus.SCENARIOS[0].id, "A")], [], {"seed": "s"})
        self.assertEqual([item["id"] for item in both["corpora"]], [corpus.CORPUS_ID, large_api.CORPUS_ID])
        self.assertEqual(precision.build_report([], [], {})["corpora"], [])


class ArmReportTests(unittest.TestCase):
    def test_arm_attempts_report_signals_comparisons_and_one_split_part(self):
        attempts, judged = [], []
        arm_fields = {**arms.undispatched(), "contracts": "b", "refusals": 1, "selection_fallback": False}
        for scenario in corpus.SCENARIOS[:32]:
            for arm, escalated in (("A", False), ("B", False), ("E", True)):
                attempt = {
                    **_attempt(scenario.id, arm),
                    **arm_fields,
                    "exposed": list(scenario.template.needed),
                    "recall": True,
                    "escalated": escalated,
                    "escalation_signal": "empty-lookup" if escalated else None,
                }
                attempts.append(attempt)
                judged.append(_decision(attempt))
        meta = {"seed": "s", "comparisons": [["A", "B"], ["B", "E"], ["A", "E"], ["A", "Z"]]}
        report = precision.build_report(attempts, judged, meta, None, "held-out")
        self.assertEqual(report["split"]["part"], "held-out")
        held = split.load()["held-out"]
        self.assertTrue(all(run["successes"] <= run["conclusive"] for run in report["runs"]))
        self.assertEqual(
            sum(run["conclusive"] for run in report["runs"]),
            3 * sum(scenario.template.id in held for scenario in corpus.SCENARIOS[:32]),
        )
        self.assertEqual(
            [(p["baseline"], p["candidate"]) for p in report["paired"]], [("A", "B"), ("B", "E"), ("A", "E")]
        )
        arm = next(run for run in report["runs"] if run["arm"] == "E")["arm_signals"]
        self.assertEqual(
            (arm["escalation_rate"], arm["escalation_signals"], arm["working_set_recall"]),
            (1.0, {"empty-lookup": arm["escalated"]}, 1.0),
        )
        item = precision.judge_item({**attempts[0], "exposed": ["dns"]})
        self.assertEqual([a.id for a in item.assistants], ["dns"])
        self.assertIn("get-record", [a.id for a in item.assistants[0].actions])
        self.assertNotIn("arm_signals", precision._group([(_attempt("dns-create.en"), "success")], "s", "plain"))
        self.assertIsNone(precision._arm_signals([{**attempts[0], "operations": 0}])["arm_signals"]["escalation_rate"])

    def test_a_stopped_record_first_keeps_every_signal_and_its_denominator(self):
        ran = {
            **_attempt("dns-create.en", "E"),
            "contracts": "b",
            "dispatched": True,
            "exposed": ["dns", "mail"],
            "exposed_actions": None,
            "recall": False,
            "selection_fallback": False,
            "refusals": 2,
            "escalated": True,
            "escalation_signal": "empty-lookup",
            "escalation_blocked": True,
            "models_used": ["luna", "sonnet"],
            "jev_usd": 0.0002,
            "jev_usd_known": False,
            "jev_seconds": 0.3,
            "jev_calls": 2,
            "jev_failures": 1,
            "route": None,
            "route_confidence": None,
        }
        stopped = {
            **_attempt("dns-update.en", "E", status="budget-stopped"),
            **arms.undispatched(),
            "contracts": "b",
            "operations": 0,
        }
        self.assertEqual(set(stopped), set(ran))
        report = precision.build_report([stopped, ran], [], {"seed": "s"})
        signals = report["runs"][0]["arm_signals"]
        self.assertEqual(
            (signals["exposure_attempts"], signals["dispatched_attempts"], signals["working_set_recall"]), (1, 1, 0.0)
        )
        self.assertEqual((signals["mean_exposed_assistants"], signals["refusals"]), (2.0, 2))
        self.assertEqual((signals["escalation_rate"], signals["escalation_blocked"]), (1.0, 1))
        self.assertEqual(
            (signals["jev_calls"], signals["jev_failures"], signals["jev_usd"], signals["jev_usd_known"]),
            (2, 1, 0.0002, False),
        )
        fell_back = precision._arm_signals([stopped, ran, {**ran, "selection_fallback": True}])["arm_signals"]
        self.assertEqual((fell_back["selection_fallbacks"], fell_back["selection_fallback_rate"]), (1, 0.5))
        self.assertEqual((signals["selection_fallbacks"], signals["selection_fallback_rate"]), (0, 0.0))
        only = precision._arm_signals([stopped])["arm_signals"]
        self.assertEqual((only["working_set_recall"], only["selection_fallback_rate"]), (None, None))


class CrossModelReportTests(unittest.TestCase):
    def test_a_reference_arm_on_another_model_pairs_over_the_repetitions_both_ran(self):
        sonnet = {"provider": "anthropic", "model": "claude-sonnet-5-5"}
        attempts = [
            _attempt(scenario.id, "A", repetition) for scenario in corpus.SCENARIOS[:16] for repetition in range(3)
        ]
        attempts += [{**_attempt(scenario.id, "S", 0, passed=False), **sonnet} for scenario in corpus.SCENARIOS[:16]]
        judged = [_decision(attempt) for attempt in attempts]
        report = precision.build_report(attempts, judged, {"seed": "s", "identical_arms": True})
        (pair,) = report["paired"]
        self.assertEqual((pair["baseline"], pair["candidate"], pair["repetitions"]), ("A", "S", [0]))
        self.assertEqual((pair["complete_pairs"], pair["incomplete_pairs"]), (16, 0))
        self.assertEqual((pair["baseline_model"], pair["candidate_model"]), ("gpt-6-luna", "claude-sonnet-5-5"))
        self.assertEqual(pair["difference"]["mean"], -1.0)
        self.assertEqual(report["pooled_identical_arms"], [])
        with self.assertRaisesRegex(ValueError, "two models"):
            precision.build_report([*attempts, {**_attempt("dns-create.en", "S", 1)}], [], {})


class LargeApiReportTests(unittest.TestCase):
    def test_large_api_attempts_use_their_stratum_contracts_and_split(self):
        attempt = {
            **_attempt("cf-ssl-strict.en", "L3"),
            **arms.undispatched(),
            "contracts": "large",
            "exposed": ["edge"],
            "exposed_actions": ["zones-list", "ssl-settings-update"],
            "recall": True,
            "selection_fallback": False,
            "refusals": 0,
            "escalated": False,
            "escalation_signal": None,
            "escalation_blocked": False,
        }
        item = precision.judge_item(attempt)
        self.assertEqual([a.id for a in item.assistants[0].actions], ["zones-list", "ssl-settings-update"])
        tasks = precision.judge_item({**attempt, "contracts": "large-tasks", "exposed_actions": None})
        self.assertEqual(len(tasks.assistants[0].actions), 21)
        report = precision.build_report(
            [attempt, _attempt("cf-dns-txt.en", "L3"), _attempt("dns-update.en", "A")],
            [],
            {"seed": "s"},
            None,
            "held-out",
        )
        self.assertEqual(sum(run["attempts"]["completed"] for run in report["runs"]), 2)
        self.assertEqual(len(report["split"]["digests"]), 2)


class GradeTests(unittest.TestCase):
    def test_only_a_blind_owner_calibration_of_the_same_judge_supports_a_decision(self):
        current = judge.identity()
        owner = {"adjudication": "owner", "blind": True, "judge_identity": current}
        owner |= {"primary": {"admitted": True}, "tiebreak": {"admitted": True}}
        verdicts = [{"judge_identity": current}]
        self.assertEqual(precision.grade(owner, verdicts, {})["grade"], "decision")
        cases = {
            "no-calibration": (None, verdicts, {}),
            "no-blind-owner-labels": ({**owner, "adjudication": "author"}, verdicts, {}),
            "calibration-of-another-judge": ({**owner, "judge_identity": "sha256:0"}, verdicts, {}),
            "tiebreak-not-admitted": ({**owner, "tiebreak": {"admitted": False}}, verdicts, {}),
            "verdict-of-another-judge": (owner, [{"judge_identity": "sha256:0"}], {}),
            "labelled-exploratory": (owner, verdicts, {"label": "exploratory"}),
        }
        for reason, arguments in cases.items():
            graded = precision.grade(*arguments)
            self.assertEqual((graded["grade"], graded["reasons"]), ("exploratory", [reason]))
        self.assertEqual(precision.grade({**owner, "blind": False}, verdicts, {})["reasons"], ["no-blind-owner-labels"])
        self.assertNotEqual(judge.identity(), judge.identity(Path(judge.__file__)))
        regraded = precision.regrade(
            {
                "decision": {"grade": "decision"},
                "judge_calibration": {**owner, "adjudication": "author"},
                "judges": {"verdict_identities": [judge.identity()]},
            }
        )
        self.assertEqual(regraded["decision"]["reasons"], ["no-blind-owner-labels"])
        promoted = precision.regrade({"judge_calibration": owner, "judges": {"verdict_identities": [judge.identity()]}})
        self.assertEqual(promoted["decision"]["grade"], "decision")
        unjudged = precision.regrade(
            {"judge_calibration": owner, "judges": {"judged_attempts": 0, "verdict_identities": []}}
        )
        self.assertEqual(unjudged["decision"]["grade"], "decision")
        for report, reason in (
            (
                {"judge_calibration": owner, "judges": {"verdict_identities": [f"sha256:{'0' * 64}"]}},
                "verdict-of-another-judge",
            ),
            ({"judge_calibration": owner, "judges": {}}, "no-verdict-provenance"),
            (
                {"judge_calibration": owner, "judges": {"judged_attempts": 100, "verdict_identities": []}},
                "no-verdict-provenance",
            ),
            ({"judge_calibration": owner, "judges": {"verdict_identities": []}}, "no-verdict-provenance"),
            (
                {"judge_calibration": owner, "judges": {"judged_attempts": False, "verdict_identities": []}},
                "no-verdict-provenance",
            ),
            ({"judge_calibration": owner}, "no-verdict-provenance"),
            ({"judge_calibration": owner, "judges": {"verdict_identities": "x"}}, "no-verdict-provenance"),
            ({"judge_calibration": owner, "judges": {"verdict_identities": [7]}}, "no-verdict-provenance"),
            ({"judge_calibration": owner, "judges": {"verdict_identities": ["sha256:0"]}}, "no-verdict-provenance"),
        ):
            regraded = precision.regrade(report)["decision"]
            self.assertEqual((regraded["grade"], regraded["reasons"]), ("exploratory", [reason]))

    def test_calibration_uses_owner_labels_only_when_collected_blind_against_the_frozen_judge(self):
        labels = dict.fromkeys(judge.CRITERIA, True)
        owner_items = [(item_id, item, labels) for item_id, item, _ in judge.heldout_items()]
        collection = {"blind_to_judge_verdicts": True, "frozen_judge_identity": judge.identity()}
        good = judge.Verdict(**labels, confident=True)
        with (
            mock.patch.object(judge, "heldout_items", return_value=owner_items),
            mock.patch.object(judge, "heldout_collection", return_value=collection),
        ):
            summary = precision.calibrate(lambda _item: good, lambda _item: good)
        self.assertEqual((summary["adjudication"], summary["blind"], summary["primary"]["items"]), ("owner", True, 30))
        with (
            mock.patch.object(judge, "heldout_items", return_value=owner_items),
            mock.patch.object(judge, "heldout_collection", return_value={**collection, "frozen_judge_identity": "x"}),
        ):
            self.assertEqual(precision.calibrate(lambda _item: good, lambda _item: good)["adjudication"], "author")
        self.assertEqual(judge.heldout_collection()["blind_to_judge_verdicts"], None)


class MetaTests(unittest.TestCase):
    def test_metadata_is_a_closed_vocabulary_of_numbers_and_safe_text(self):
        meta = {
            "seed": "pilot-2026-10-02",
            "scenario_patterns": ["task-create.en", "dns-*"],
            "campaigns": [{"efforts": {"low": "low", "high": "high"}, "commits": {"brain": "a" * 40}}],
            "corpus": {"digest": "sha256:" + "0" * 64, "scenarios": 120},
            "budget": {"spent_usd": 1.5, "stopped_by_cap": False, "failed": None},
        }
        self.assertEqual(precision.checked_meta(meta), meta)
        refused = [
            {"reply": "Done. I created the record."},
            {"seed": "sk-ant-api03-0123456789"},
            {"seed": "x" * 41},
            {"kind": "Créé"},
            {"efforts": {"Low Effort": "low"}},
            {"efforts": {"sk-proj-short-secret": "low"}},
            {"campaigns": [{"efforts": {"sk-abc": "low"}}]},
            {"low": "low"},
            {"budget": {"spent_usd": float("inf")}},
            {"budget": {"spent_usd": object()}},
            {"campaigns": [[[[[[{}]]]]]]},
        ]
        for value in refused:
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                precision.checked_meta(value)
        with self.assertRaisesRegex(ValueError, "unknown field"):
            precision.build_report([], [], {"seed": "s", "reply": "Sent the message to Ana."})
        for unsafe in (
            {"kind": "pilot baseline"},
            {"seed": "Done"},
            {"adjudication": "author-adjudicated"},
            {"provider": "azure"},
            {"spent_usd": "1.0"},
            {"spent_usd": -1},
            {"admitted": "yes"},
            {"budget": {"spent_usd": True}},
            {"blind": {"spent_usd": 1}},
            {"spent_usd": {"spent_usd": 1}},
            {"provider": {"provider": "openai"}},
            {"provider": ["openai"]},
            {"blind": [True]},
            {"scenario_patterns": [["dns-*"]]},
            {"scenario_patterns": [{"seed": "s"}]},
            {"efforts": {"A": {"effort": "low"}}},
        ):
            if unsafe == {"seed": "Done"}:
                self.assertEqual(precision.checked_meta(unsafe), unsafe)
                continue
            with self.subTest(value=str(unsafe)), self.assertRaisesRegex(ValueError, "unsafe value"):
                precision.checked_meta(unsafe)

    def test_every_exported_run_identity_is_checked(self):
        precision.checked_identity("pilot-luna", "openai", "gpt-6-luna", "L5-groups", ["low"])
        for identity in (
            ("Pilot Luna", "openai", "gpt-6-luna", "a", ["low"]),
            ("c", "azure", "gpt-6-luna", "a", ["low"]),
            ("c", "openai", "gpt-6-luna", "two words", ["low"]),
            ("c", "openai", "gpt-6-luna", "a", ["extreme"]),
            ("sk-proj-short-secret", "openai", "gpt-6-luna", "a", ["low"]),
            ("c", "openai", "gpt-6-luna", "sk-abc", ["low"]),
        ):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "identity"):
                precision.checked_identity(*identity)
        with self.assertRaisesRegex(ValueError, "another provider"):
            precision.checked_identity("c", "anthropic", "gpt-6-luna", "a", ["low"])
        with self.assertRaisesRegex(ValueError, "identity"):
            precision.build_report([{**_attempt("dns-create.en"), "campaign": "Free text campaign"}], [], {})


class CommandTests(unittest.TestCase):
    def test_validate_admits_every_scenario_in_brain(self):
        result = precision.validate()
        self.assertEqual((result["scenarios"], result["calibration_items"]), (200, 32))
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
            written = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(written["judge_calibration"]["judge_identity"], judge.identity())
            # The mocked judges call every reply good, so the sample does not admit them.
            self.assertEqual(written["decision"]["grade"], "exploratory")
            self.assertIn("no-blind-owner-labels", written["decision"]["reasons"])
            self.assertEqual(precision.main(arguments), 0)
            self.assertEqual(precision.main(["regrade", "--out", str(report)]), 0)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["decision"]["grade"], "exploratory")
            inside = Path(precision.__file__).parent / "never.jsonl"
            with (
                mock.patch.object(judge, "judge_model", side_effect=AssertionError("no inference")) as built,
                mock.patch("sys.stderr"),
            ):
                self.assertEqual(
                    precision.main(["judge", "--transcript", str(root / "t.jsonl"), "--out", str(inside), *keys]), 2
                )
            self.assertEqual((built.call_count, inside.exists()), (0, False))
            with mock.patch("sys.stderr"):
                self.assertEqual(
                    precision.main(["judge", "--transcript", str(root / "missing"), "--out", str(out), *keys]), 2
                )


if __name__ == "__main__":
    unittest.main()

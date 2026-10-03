"""Provider-free checks for precision-v3 episode judging and reports (ADR-0094)."""

from __future__ import annotations

import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from eval import cost as eval_cost
from eval.complex import judging, model, report
from eval.complex.templates import TEMPLATES

GOOD = judging.TrajectoryVerdict(
    turns_correct=True, unsupported_claim=False, honest_about_failures=True, language_matches=True, confident=True
)
BAD = GOOD.model_copy(update={"turns_correct": False})


def _record(template, arm: str, repetition: int, *, passed: bool = True, status: str = "completed") -> dict:
    world, checkpoints = model.run_reference(template)
    oracle = model.oracle(template, world, checkpoints).to_dict() | {"passed": passed}
    return {
        "campaign": "v3-aa",
        "provider": "openai",
        "model": "gpt-6-luna",
        "effort": "low",
        "arm": arm,
        "repetition": repetition,
        "position": 0,
        "corpus": model.CORPUS_ID,
        "episode": f"{template.id}.en",
        "template": template.id,
        "stratum": template.stratum,
        "locale": "en",
        "status": status,
        "turns": [
            {"step": i, "user": "Secret user text", "reply": "Done.", "record": []} for i in range(template.turns)
        ],
        "checkpoints": checkpoints,
        "oracle": oracle,
        "limitations": [0] if arm == "A2" else [],
        "violations": {},
        "accounting": dict.fromkeys(report.ACCOUNTING, 1) | {"brain_operations": 0 if status != "completed" else 2},
        "usage": eval_cost.Usage().to_dict(),
        "usd": 0.002,
        "usage_known": True,
        "seconds_active": 3.0 + repetition,
        "seconds_wall": 4.0,
        "cache": "warm" if repetition else "cold",
    }


def _records() -> list[dict]:
    chosen = [t for t in TEMPLATES if t.id in {"s1-task-and-event", "s4-partial-bulk", "s5-message-text"}]
    rows = [
        _record(t, arm, rep, passed=not (arm == "A2" and rep == 1))
        for t in chosen
        for arm in ("A", "A2")
        for rep in (0, 1)
    ]
    rows.append(_record(TEMPLATES[0], "A", 2, status="budget-stopped"))
    return rows


def _verdict(record, final=GOOD, identity=None):
    return {
        "key": report.key(record),
        "primary": final.model_dump(),
        "tiebreak": None,
        "final": final.model_dump(),
        "judge_identity": identity or judging.identity(),
    }


class EpisodeReportTests(unittest.TestCase):
    def test_every_completed_episode_is_judged_and_failures_stay_unjudged(self):
        records = _records()
        seen = []

        def primary(trajectory):
            seen.append(trajectory.episode.id)
            if trajectory.episode.template.id == "s5-message-text":
                raise judging.TrajectoryJudgeError("down")
            return GOOD

        judged = report.judge_episodes(records, primary, lambda _t: BAD, workers=2)
        completed = [r for r in records if r["status"] == "completed"]
        self.assertEqual(len(seen), len(completed))
        self.assertEqual(len(judged), len(completed) - 4)
        failing = next(
            item for item in judged if item["key"].endswith("/1/s4-partial-bulk.en") and "/A2/" in item["key"]
        )
        self.assertEqual(failing["final"], BAD.model_dump())
        self.assertIsNotNone(failing["tiebreak"])
        self.assertEqual(report.trajectory(records[0]).turns[0].user, "Secret user text")

    def test_outcomes(self):
        record = _record(TEMPLATES[0], "A", 0)
        self.assertEqual(report.outcome({**record, "status": "budget-stopped"}, None), "inconclusive")
        self.assertEqual(report.outcome({**record, "oracle": {"passed": False}}, None), "failure")
        self.assertEqual(report.outcome(record, None), "inconclusive")
        self.assertEqual(report.outcome(record, _verdict(record)), "success")
        self.assertEqual(report.outcome(record, _verdict(record, BAD)), "failure")

    def test_the_report_is_sanitized_paired_and_graded(self):
        records = _records()
        judged = [_verdict(r) for r in records if r["status"] == "completed"]
        meta = {"seed": "s", "campaign": "v3-aa", "corpus": {"id": model.CORPUS_ID}, "path": "team+brain"}
        body = report.build_report(records, judged, meta)
        text = json.dumps(body)
        for leaked in ("Secret user text", "Done.", '"turns"', '"checkpoints"'):
            self.assertNotIn(leaked, text)
        a, a2 = body["runs"]
        self.assertEqual((a["arm"], a["episodes"], a["conclusive"], a["successes"]), ("A", 7, 6, 6))
        self.assertEqual((a2["successes"], a2["limitations"]), (3, 6))
        self.assertEqual(a["pass^2"]["mean"], 1.0)
        self.assertEqual(a2["pass^2"]["mean"], 0.0)
        self.assertIsNone(a["pass^5"]["mean"])
        self.assertEqual(a["cache"], {"cold": 3, "warm": 3, "unknown": 0})
        self.assertEqual(a["accounting"]["user_turns"], 7)
        self.assertEqual(a["cost"]["attempts"], 6)
        self.assertEqual(set(a["strata"]), set(model.STRATA))
        (pair,) = body["paired"]
        self.assertEqual((pair["complete_pairs"], pair["incomplete_pairs"]), (6, 1))
        self.assertEqual(pair["difference"]["mean"], -0.5)
        self.assertEqual(body["decision"]["reasons"], ["no-blind-owner-labels"])
        owner = {"adjudication": "owner", "blind": True, "judge_identity": judging.identity()}
        self.assertEqual(report.build_report(records, judged, meta, owner)["decision"]["grade"], "decision")
        stale = [_verdict(records[0], identity="sha256:" + "0" * 64)]
        reasons = report.grade({**owner, "judge_identity": "sha256:" + "1" * 64}, stale)["reasons"]
        self.assertEqual(reasons, ["calibration-of-another-judge", "verdict-of-another-judge"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report.build_report([*records, records[0]], judged, meta)
        with self.assertRaisesRegex(ValueError, "another corpus"):
            report.build_report([{**records[0], "corpus": "precision-v2.1"}], [], meta)

    def test_calibration_measures_both_judges_on_the_development_set(self):
        def broken(_trajectory):
            raise eval_cost.BudgetExhaustedError("cap")

        summary = report.calibrate(lambda _t: GOOD, broken)
        items = len(judging.calibration_items())
        self.assertEqual((summary["primary"]["items"], summary["tiebreak"]["failed"]), (items, items))
        self.assertEqual((summary["adjudication"], summary["blind"]), ("author", False))

    def test_the_commands_write_their_outputs_and_fail_closed(self):
        records = _records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "t.jsonl"
            transcript.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            fakes = (lambda _t: GOOD, lambda _t: GOOD)
            with mock.patch.object(report, "_judges", return_value=fakes), redirect_stdout(io.StringIO()):
                self.assertEqual(report.main(["judge", "--transcript", str(transcript), "--out", str(root / "j")]), 0)
                self.assertEqual(report.main(["calibrate", "--out", str(root / "c.json")]), 0)
            calibration = json.loads((root / "c.json").read_text(encoding="utf-8"))
            self.assertEqual(calibration["judge_budget"]["spent_usd"], 0.0)
            (root / "m.json").write_text(json.dumps({"seed": "s"}), encoding="utf-8")
            argv = [
                "report",
                "--transcript",
                str(transcript),
                "--judged",
                str(root / "j"),
                "--meta",
                str(root / "m.json"),
            ]
            calibration.pop("judge_budget")
            (root / "c.json").write_text(json.dumps(calibration), encoding="utf-8")
            self.assertEqual(
                report.main([*argv, "--calibration", str(root / "c.json"), "--out", str(root / "r.json")]), 0
            )
            self.assertEqual(json.loads((root / "r.json").read_text(encoding="utf-8"))["judges"]["judged_episodes"], 12)
            self.assertEqual(
                report.main(
                    [
                        "report",
                        "--transcript",
                        str(transcript),
                        "--judged",
                        str(root / "j"),
                        "--out",
                        str(root / "r2.json"),
                    ]
                ),
                0,
            )
            with redirect_stderr(io.StringIO()):
                self.assertEqual(report.main(["report", "--transcript", str(root / "missing"), "--out", "x"]), 2)
                with mock.patch.object(report, "_judges", side_effect=KeyError("x")):
                    self.assertEqual(
                        report.main(["judge", "--transcript", str(transcript), "--out", str(root / "k")]), 2
                    )
            with (
                mock.patch.object(report, "_judges", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                report.main(["judge", "--transcript", str(transcript), "--out", str(root / "z")])
            with (
                mock.patch.object(sys, "argv", ["report", *argv, "--out", str(root / "r3.json")]),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_module("eval.complex.report", run_name="__main__")
            self.assertEqual(exited.exception.code, 0)

    def test_judges_are_built_bounded_from_owner_only_key_files(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory, "key")
            key.write_text("offline-key-0123456789\n", encoding="utf-8")
            key.chmod(0o600)
            args = mock.Mock(key_file=key, tiebreak_key_file=key)
            with mock.patch.object(judging, "judge", return_value=GOOD) as called:
                primary, tiebreak = report._judges(args)
                self.assertEqual((primary("t"), tiebreak("t")), (GOOD, GOOD))
            self.assertEqual([call.args[1] for call in called.call_args_list], ["openai", "anthropic"])


if __name__ == "__main__":
    unittest.main()

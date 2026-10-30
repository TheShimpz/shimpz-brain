"""Provider-free, binary-free checks of the Routine eval's slots: how each Brain starts, and what its origin counted."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import routines_fake_brain
from eval import ceiling as eval_ceiling
from eval import cost as eval_cost
from eval import routines_serve


def _files(directory: Path, number: int = 0) -> routines_serve.SlotFiles:
    return routines_serve.SlotFiles(
        directory / f"slot-{number}.token",
        directory / f"slot-{number}.state.json",
        directory / f"slot-{number}.events",
        directory / "budget.json",
        directory / f"slot-{number}",
        directory / f"brain-{number}.log",
    )


class CountsTests(unittest.TestCase):
    def test_an_attempt_reads_the_counts_its_slots_origin_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory, "slot-0.state.json")
            ceiling = eval_ceiling.Ceiling(1.0, 4_000, state)
            for name, times in (("refused", 1), ("throttled", 2), ("terminal", 3), ("clamped", 4), ("requests", 9)):
                for _ in range(times):
                    ceiling._count(name)
            ceiling.write_state()
            self.assertEqual(
                routines_serve.read_counts(state), {"refused": 1, "throttled": 2, "terminal": 3, "clamped": 4}
            )

    def test_an_unreadable_snapshot_is_no_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory, "state.json")
            self.assertIsNone(routines_serve.read_counts(state))
            for body in ("not json", "[]", json.dumps({"refused": 0, "throttled": "1", "terminal": 0, "clamped": 0})):
                state.write_text(body, encoding="utf-8")
                self.assertIsNone(routines_serve.read_counts(state))


class LaunchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_the_run_never_starts_without_an_executable_brain_binary(self):
        plain = self.directory / "shimpz-brain"
        plain.write_text("", encoding="utf-8")
        for named in ("", str(self.directory / "missing"), str(plain)):
            with (
                mock.patch.dict(os.environ, {routines_serve.BINARY_ENVIRONMENT: named}),
                self.assertRaises(SystemExit),
            ):
                routines_serve.brain_binary()
        binary = routines_fake_brain.launcher(self.directory)
        with mock.patch.dict(os.environ, {routines_serve.BINARY_ENVIRONMENT: binary}):
            self.assertEqual(routines_serve.brain_binary(), binary)

    def test_a_brain_serves_privately_against_its_own_origin(self):
        files = _files(self.directory, 3)
        process, port = routines_serve.start_brain(routines_fake_brain.launcher(self.directory), files, 40123)
        self.addCleanup(process.wait)
        self.addCleanup(process.terminate)
        routines_serve.healthy(port, process)
        argv = json.loads((files.brain_state / "argv.json").read_text(encoding="utf-8"))
        self.assertEqual(
            argv,
            [
                "serve",
                "--listen",
                f"127.0.0.1:{port}",
                "--token-file",
                str(files.token),
                "--state-dir",
                str(files.brain_state),
                "--provider-origin",
                "http://127.0.0.1:40123",
            ],
        )
        self.assertEqual(files.token.stat().st_mode & 0o777, 0o600)
        self.assertEqual(files.brain_state.stat().st_mode & 0o777, 0o700)
        self.assertGreaterEqual(len(files.token.read_text(encoding="utf-8")), 32)

    def test_a_brain_that_stops_never_becomes_healthy(self):
        with mock.patch.dict(os.environ, {"FAKE_BRAIN_MODE": "quiet"}):
            process, port = routines_serve.start_brain(
                routines_fake_brain.launcher(self.directory), _files(self.directory), 9
            )
        with self.assertRaisesRegex(RuntimeError, "did not become healthy"):
            routines_serve.healthy(port, process, deadline=5)

    def test_an_origin_serves_against_the_runs_ledger_and_says_its_port(self):
        eval_cost.SharedBudget.create(self.directory / "budget.json", 1.0)
        files = _files(self.directory)
        origin = routines_serve.start_origin(files)
        self.addCleanup(origin.wait)
        self.addCleanup(origin.stdout.close)
        self.addCleanup(origin.stdin.close)
        self.assertGreater(routines_serve.announced_port(origin), 0)
        snapshot = json.loads(files.state.read_text(encoding="utf-8"))
        self.assertEqual((snapshot["cap_usd"], snapshot["max_output_tokens"]), (1.0, routines_serve.CALL_OUTPUT_TOKENS))

    def test_an_origin_that_never_says_a_port_is_stopped(self):
        files = _files(self.directory)
        for script, deadline in (
            ("print('no port')", 5.0),
            ("print(70000)", 5.0),
            ("import time; time.sleep(30)", 0.2),
        ):
            with self.subTest(script=script):
                origin = routines_serve.start_origin(files, (sys.executable, "-c", script))
                with self.assertRaisesRegex(RuntimeError, "stopped before it served"):
                    routines_serve.announced_port(origin, deadline)
                origin.kill()
                origin.wait(timeout=5)
                origin.stdout.close()
                origin.stdin.close()


class MeterTests(unittest.TestCase):
    def test_each_call_is_priced_at_the_model_that_served_it(self):
        meter = routines_serve.Meter()
        counts = dict.fromkeys(eval_cost.FIELDS, 0) | {"model_calls": 1, "input_tokens": 1000, "output_tokens": 10}
        meter.add("gpt-6-luna", counts)
        meter.add("gpt-6-luna", counts)
        meter.add("gpt-6.1-sol", counts | {"failed_calls": 1})
        usage = eval_cost.Usage.of(counts)
        expected = eval_cost.cost(usage + usage, "gpt-6-luna").usd + eval_cost.cost(usage, "gpt-6.1-sol").usd
        spent = meter.cost()
        self.assertAlmostEqual(spent.usd, expected)
        self.assertFalse(spent.known)


if __name__ == "__main__":
    unittest.main()

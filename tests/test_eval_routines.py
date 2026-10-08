"""Provider-free checks of the Routine eval driver's run: each attempt's evidence, the merged report, and the gate."""

from __future__ import annotations

import base64
import json
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eval import routines, routines_pool
from eval.routines_fixture import MODELS
from eval.routines_strata import CASES, VARIANT_STRATA

VARIANTS = {
    "moved-zone-id": {"status": "done", "listed_new_id": True},
    "twin-zone-names": {"status": "failed", "code": "plan-reference-ambiguous", "list_dns_records_dispatched": False},
}


class Answer:
    def __init__(self, status: int, **headers: str) -> None:
        self.status, self.headers = status, {name.lower(): value for name, value in headers.items()}

    def getheader(self, name: str) -> str | None:
        return self.headers.get(name.lower())


def _evidence(throttled: str = "0", clamped: str = "0", **headers: str) -> dict[str, str]:
    return {"eval-throttled": throttled, "eval-clamped": clamped, **headers}


class SignalsTests(unittest.TestCase):
    def test_each_brain_response_adds_its_evidence_and_the_worst_decides_the_attempt(self):
        signals = routines.Signals()
        signals.saw(Answer(200, **_evidence("2", "1")))
        self.assertEqual((signals.throttled, signals.clamped, signals.disposition()), (2, 1, "counted"))
        signals.saw(Answer(503, **_evidence(), **{"Retry-After": "1"}))
        self.assertEqual((signals.refused, signals.failed, signals.disposition()), (1, 1, "brain-refused"))
        signals.saw(Answer(502, **_evidence("3", **{"eval-failure": "throttled"})))
        self.assertEqual((signals.throttled, signals.disposition()), (5, "throttled"))
        signals.saw(Answer(502, **_evidence(**{"eval-failure": "budget"})))
        self.assertEqual(signals.disposition(), "stopped")
        # A Brain 503 without Retry-After is no capacity refusal: Brain's state or authentication failed.
        failed = routines.Signals()
        failed.saw(Answer(503, **_evidence()))
        self.assertEqual((failed.refused, failed.failed, failed.disposition()), (0, 1, "counted"))

    def test_a_response_without_readable_evidence_is_never_trusted(self):
        for headers in ({}, _evidence("x"), {"eval-throttled": "0"}):
            with self.subTest(headers=headers):
                signals = routines.Signals()
                signals.saw(Answer(200, **headers))
                self.assertEqual(signals.disposition(), "error")

    def test_the_attempts_connection_reads_each_response_before_team_does(self):
        class Brain(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                while self.rfile.readline() not in (b"\r\n", b""):
                    continue
                self.rfile.read(2)
                headers = "".join(f"{name}: {value}\r\n" for name, value in _evidence("1", **failure).items())
                self.wfile.write(f"HTTP/1.1 502 Bad Gateway\r\n{headers}Content-Length: 2\r\n\r\n{{}}".encode())

        failure = {"eval-failure": "throttled"}
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Brain)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        signals = routines.Signals()
        connection = signals.connection("127.0.0.1", server.server_address[1], 5.0)
        connection.request("POST", "/v1/turns", b"{}")
        response = connection.getresponse()
        self.assertEqual((response.status, response.read()), (502, b"{}"))
        connection.close()
        self.assertEqual((signals.throttled, signals.failed, signals.disposition()), (1, 1, "throttled"))


def _counted(task: dict[str, object], **changes) -> dict[str, object]:
    return {
        **task,
        "disposition": "counted",
        "throttled": 0,
        "clamped": 0,
        "passed": True,
        "reason": None,
        "cause": None,
        "schedule": {"kind": "calendar"},
        "questions": [],
        "usd": 0.01,
        "known": True,
        "variants": VARIANTS if task["case"] in VARIANT_STRATA else None,
    } | changes


class FakePool:
    """The run's processes, answered in place: every task's result comes back at once, as ``answer`` says."""

    answer = staticmethod(_counted)

    def __init__(self, command, brains, count, threads, configure) -> None:
        self.command, self.brains, self.count, self.threads = command, brains, count, threads
        self.configs = [
            configure(number, f"http://127.0.0.1:{4000 + number}", Path(f"t{number}")) for number in range(count)
        ]
        self.pending: list[dict[str, object]] = []
        self.worker = SimpleNamespace(busy=0, threads=count * threads)
        FakePool.last = self

    def __enter__(self) -> FakePool:
        return self

    def __exit__(self, *_exc) -> None:
        return

    def free(self):
        return self.worker if len(self.pending) < self.worker.threads else None

    def dispatch(self, _worker, slot) -> None:
        self.pending.append(type(self).answer(slot.task()))

    def receive(self, _timeout):
        result = self.pending.pop(0)
        return (result["model"], result["case"], result["index"]), result


class RunTests(unittest.TestCase):
    def setUp(self):
        self.assistant = routines.Generated(b"manifest", b"contract", Path("/assistant"))
        patches = (
            mock.patch.object(routines, "Pool", FakePool),
            mock.patch.object(routines, "brain_python", return_value="/brain/python"),
            mock.patch.object(routines, "confine", side_effect=lambda cpus: list(range(cpus))),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.models = [(provider, model, f"key-{provider}") for provider, model in MODELS]

    def test_a_clean_run_merges_every_attempt_into_one_report_and_passes_the_gate(self):
        settings = routines.Settings(30, 8.0, cpus=4, workers=2, threads=3)
        report = routines.run(self.models, settings, assistant=self.assistant)
        self.assertEqual(report["gate"], {"passed": True, "failures": []})
        self.assertEqual([item["model"] for item in report["models"]], [model for _provider, model in MODELS])
        cases = report["models"][0]["cases"]
        self.assertEqual([case["id"] for case in cases], [case for case, _play in CASES])
        self.assertEqual((cases[0]["passed"], cases[0]["trials"], cases[0]["usd"]), (30, 30, 0.3))
        self.assertEqual(report["variants"], VARIANTS)
        self.assertEqual(report["budget"]["spent_usd"], 0)
        self.assertEqual(report["throttling"]["evidence"]["unattributed"], 0)
        run = report["run"]
        self.assertEqual(
            (run["cpus"], run["workers"], run["threads"], run["capacity"], run["peak_attempts"]), (4, 2, 3, 6, 6)
        )
        config = FakePool.last.configs[1]
        self.assertEqual((config["brain"], config["token"], config["threads"]), ("http://127.0.0.1:4001", "t1", 3))
        self.assertEqual(base64.b64decode(config["manifest"]), b"manifest")
        self.assertEqual(config["models"], [list(item) for item in self.models])
        self.assertEqual(FakePool.last.command[1], str(Path(routines.__file__).resolve()))
        self.assertEqual(FakePool.last.brains[0], "/brain/python")

    def test_throttling_and_the_budget_never_pass_and_never_count_as_a_miss(self):
        def answer(task: dict[str, object]) -> dict[str, object]:
            if task["case"] == "plain-list" and task["model"] == 0:
                return _counted(task, disposition="throttled", throttled=1)
            if task["case"] == "multi-zone" and task["index"] == 1:
                return _counted(task, disposition="stopped")
            if task["case"] == "missing-schedule" and task["try"] == 0:
                # A response at the output limit fails the gate even when its try is discarded and its retry is clean.
                return _counted(task, disposition="brain-refused", clamped=1)
            if task["model"] == 1 and task["case"] == "owner-4-turns":
                return _counted(task, disposition="unavailable")
            return _counted(task)

        with (
            mock.patch.object(FakePool, "answer", staticmethod(answer)),
            mock.patch.object(routines_pool, "BACKOFF_SECONDS", 0.0),
        ):
            report = routines.run(
                self.models, routines.Settings(3, 1.0, workers=1, threads=2), assistant=self.assistant
            )
        luna = {case["id"]: case for case in report["models"][0]["cases"]}
        self.assertEqual((luna["plain-list"]["trials"], luna["plain-list"]["misses"]), (0, []))
        self.assertTrue(luna["plain-list"]["inconclusive"] and luna["multi-zone"]["inconclusive"])
        self.assertEqual(report["throttling"]["providers"]["openai"]["given_up"], 1)
        failures = report["gate"]["failures"]
        for failure in ("attempts-below-gate", "throttled", "gpt-6-luna:plain-list", "provider-unavailable:anthropic"):
            self.assertIn(failure, failures)
        # The fake Brains wrote no events, so the attempts received evidence no Brain met: never a loss.
        self.assertEqual(report["throttling"]["evidence"]["received_clamped"], 1)
        self.assertIn("output-clamped", failures)
        self.assertNotIn("evidence-lost", failures)
        # Sonnet's first slot found its provider refusing every request: no other case of it ever ran.
        (refused,) = report["models"][1]["cases"]
        self.assertEqual((refused["id"], refused["trials"], refused["inconclusive"]), ("owner-4-turns", 0, True))
        self.assertNotIn("cost-unknown", failures)

    def test_the_gate_names_every_run_failure(self):
        unknown = {
            "id": "plain-list",
            "cost_known": False,
            "inconclusive": False,
            "misses": [],
            "passed": 1,
            "trials": 1,
        }
        report = {
            "models": [{"provider": "openai", "model": "gpt-6-luna", "cases": [unknown]}],
            "variants": None,
            "budget": {"unknown_settlements": 0, "unsettled_usd": 0.1},
            "throttling": {
                "providers": {
                    "openai": {"given_up": 0, "unavailable": False},
                    "anthropic": {"given_up": 1, "unavailable": True},
                },
                "evidence": {
                    "brain_throttled": 3,
                    "brain_clamped": 0,
                    "unattributed": 0,
                    "received_throttled": 2,
                    "received_clamped": 0,
                },
            },
        }
        failures = routines.gate(report, 30)["failures"]
        self.assertEqual(
            failures[:5],
            ["cost-unknown", "budget-unsettled", "evidence-lost", "throttled", "provider-unavailable:anthropic"],
        )
        self.assertIn("variant:moved-zone-id", failures)
        self.assertIn("variant:twin-zone-names", failures)
        self.assertIn(f"{MODELS[0][1]}:{CASES[0][0]}", failures)


class WorkerTests(unittest.TestCase):
    def test_an_abandoned_worker_kills_its_whole_session_or_only_itself(self):
        with (
            mock.patch.object(routines.os, "getpgrp", return_value=41),
            mock.patch.object(routines.os, "getpid", return_value=41),
            mock.patch.object(routines.os, "killpg") as killed,
            mock.patch.object(routines.os, "_exit") as exited,
        ):
            routines._abandon()
            killed.assert_called_once_with(41, routines.signal.SIGKILL)
            routines.os.getpid.return_value = 42
            routines._abandon()
            self.assertEqual((killed.call_count, exited.call_count), (1, 2))

    def test_only_the_first_passing_eligible_attempt_of_the_run_claims_the_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual([routines._claim_variants(Path(directory)) for _ in range(3)], [True, False, False])

    def test_a_discarded_try_is_traced_beside_its_slot_never_in_its_place(self):
        with tempfile.TemporaryDirectory() as directory:
            context = routines.WorkerContext(
                None, ("u", Path("t")), [("openai", "gpt-6-luna", "k")], None, Path(directory), Path(directory)
            )
            task = {"model": 0, "case": "plain-list", "index": 4, "try": 2}
            result = {"disposition": "throttled", "reason": None, "cause": None, "schedule": None, "questions": []}
            routines._trace(context, task, result, [{"kind": "send"}])
            routines._trace(context, task, {**result, "disposition": "counted"}, [])
            discarded = json.loads(Path(directory, "gpt-6-luna", "plain-list", "discarded", "4.2.json").read_text())
            self.assertEqual(
                (discarded["outcome"]["disposition"], discarded["events"]), ("throttled", [{"kind": "send"}])
            )
            self.assertTrue(Path(directory, "gpt-6-luna", "plain-list", "4.json").is_file())


if __name__ == "__main__":
    unittest.main()

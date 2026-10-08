"""Provider-free checks of the Routine eval's Brain processes: evidence, failure causes, and how the run starts them."""

from __future__ import annotations

import io
import itertools
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import agent_runtime
import httpx
import routines_fake_brain
from eval import cost as eval_cost
from eval import routines_serve
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.testclient import TestClient


class StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _chained(*errors: BaseException) -> BaseException:
    """The last error, caused by each earlier one in turn, as the SDK and Brain raise them."""
    for cause, error in itertools.pairwise(errors):
        error.__cause__ = cause
    return errors[-1]


class FailureTests(unittest.TestCase):
    def test_the_cause_chain_names_the_budget_first_then_a_throttle(self):
        refused = _chained(eval_cost.BudgetExhaustedError("cap"), RuntimeError("connection"), RuntimeError("provider"))
        self.assertEqual(routines_serve.failure_of(refused), "budget")
        for status in (429, 503, 529):
            with self.subTest(status=status):
                self.assertEqual(routines_serve.failure_of(_chained(StatusError(status), RuntimeError())), "throttled")
        throttled_then_refused = _chained(StatusError(429), eval_cost.BudgetExhaustedError("cap"), RuntimeError())
        self.assertEqual(routines_serve.failure_of(throttled_then_refused), "budget")
        self.assertIsNone(routines_serve.failure_of(_chained(StatusError(500), RuntimeError())))
        looped = RuntimeError("outer")
        inner = StatusError(400)
        inner.__context__, looped.__cause__ = looped, inner
        self.assertIsNone(routines_serve.failure_of(looped))

    def test_only_an_answer_that_stopped_at_the_output_limit_is_clamped(self):
        stopped = (
            {"stop_reason": "max_tokens"},
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            {"choices": [{"finish_reason": "stop"}, {"finish_reason": "length"}]},
        )
        for body in stopped:
            with self.subTest(body=body):
                self.assertTrue(routines_serve.clamped(body))
        complete = (
            [],
            {"stop_reason": "end_turn"},
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
            {"incomplete_details": None},
            {"choices": ["length"]},
            {"choices": "length"},
        )
        for body in complete:
            with self.subTest(body=body):
                self.assertFalse(routines_serve.clamped(body))


def _response(status: int, body: object = None, host: str = "api.openai.com") -> tuple[httpx.Request, httpx.Response]:
    request = httpx.Request("POST", f"https://{host}/v1/responses")
    content = b"not json" if body is None else json.dumps(body).encode()
    return request, httpx.Response(status, content=content, request=request)


class ObserverTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.events = Path(directory.name, "brain-0.events")
        self.observer = routines_serve.Observer(self.events)

    def test_each_provider_answer_is_credited_to_its_brain_request(self):
        evidence = routines_serve.Evidence()
        token = routines_serve._EVIDENCE.set(evidence)
        self.addCleanup(routines_serve._EVIDENCE.reset, token)
        for status, body in ((429, None), (529, {}), (200, {"stop_reason": "max_tokens"}), (200, None), (200, {})):
            self.observer.observe(*_response(status, body))
        self.observer.observe(*_response(429, host="example.net"))
        self.assertEqual((evidence.throttled, evidence.clamped, evidence.failure), (2, 1, None))
        self.assertFalse(self.events.exists())
        evidence.failure = "throttled"
        self.assertEqual(
            evidence.headers(),
            [(b"eval-throttled", b"2"), (b"eval-clamped", b"1"), (b"eval-failure", b"throttled")],
        )

    def test_an_answer_outside_every_brain_request_is_written_down_instead(self):
        self.observer.observe(*_response(429))
        self.observer.observe(*_response(200, {"stop_reason": "max_tokens"}, host="api.anthropic.com"))
        lines = [json.loads(line) for line in self.events.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines, [{"unattributed": 429, "clamped": False}, {"unattributed": 200, "clamped": True}])

    def test_install_observes_every_response_the_client_sends(self):
        self.addCleanup(setattr, httpx.Client, "send", httpx.Client.send)
        self.observer.install()
        client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(529, json={})))
        self.assertEqual(client.post("https://api.anthropic.com/v1/messages").status_code, 529)
        self.assertEqual(len(self.events.read_text(encoding="utf-8").splitlines()), 1)


class EvidenceHeadersTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.events = Path(directory.name, "brain-0.events")
        self.addCleanup(setattr, httpx.Client, "send", httpx.Client.send)
        routines_serve.Observer(self.events).install()
        answers = {"throttled": 429, "clean": 200}
        app = FastAPI()

        @app.post("/{kind}")
        async def turn(kind: str) -> dict[str, str]:
            def work() -> dict[str, str]:
                provider = httpx.Client(
                    transport=httpx.MockTransport(lambda _r: httpx.Response(answers[kind], json={}))
                )
                answer = provider.post("https://api.openai.com/v1/responses")
                if answer.status_code != 200:
                    raise agent_runtime.ProviderRequestError("failed") from StatusError(answer.status_code)
                return {"status": "ok"}

            return await run_in_threadpool(work)

        app.add_exception_handler(agent_runtime.ProviderRequestError, routines_serve.provider_error)
        app.add_middleware(routines_serve.EvidenceHeaders)
        self.client = TestClient(app)

    def test_each_brain_response_carries_its_own_request_evidence(self):
        with self.client:
            failed = self.client.post("/throttled")
            clean = self.client.post("/clean")
        self.assertEqual(failed.status_code, 502)
        self.assertEqual(failed.json(), {"detail": "Model provider request failed"})
        self.assertEqual(
            (failed.headers["eval-throttled"], failed.headers["eval-clamped"], failed.headers["eval-failure"]),
            ("1", "0", "throttled"),
        )
        self.assertEqual((clean.status_code, clean.headers["eval-throttled"]), (200, "0"))
        self.assertNotIn("eval-failure", clean.headers)
        self.assertFalse(self.events.exists())

    def test_a_provider_failure_outside_a_request_still_answers_brains_own_502(self):
        import asyncio

        answer = asyncio.run(routines_serve.provider_error(None, RuntimeError("provider")))
        self.assertEqual(answer.status_code, 502)


class EvaluationAppTests(unittest.TestCase):
    def test_the_app_reserves_every_wire_try_under_the_run_ledger_and_reports_evidence(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("eval.ceiling.Ceiling") as ceiling,
            mock.patch.object(routines_serve.Observer, "install") as observed,
        ):
            ledger = Path(directory, "budget.json")
            app = routines_serve.evaluation_app(ledger, Path(directory, "events"), "t" * 43)
        (cap, output), options = ceiling.call_args
        self.assertEqual((cap, output, options["sdk_retries"]), (0.0, routines_serve.CALL_OUTPUT_TOKENS, True))
        self.assertEqual(options["budget"].path, ledger)
        ceiling.return_value.install.assert_called_once_with()
        observed.assert_called_once_with()
        self.assertIs(app.exception_handlers[agent_runtime.ProviderRequestError], routines_serve.provider_error)
        self.assertIs(app.user_middleware[0].cls, routines_serve.EvidenceHeaders)

    def test_serve_brain_writes_its_token_prints_its_port_and_serves_until_stdin_closes(self):
        stdout = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(routines_serve, "evaluation_app", return_value="app") as app,
            mock.patch.object(routines_serve, "_log_provider_failures") as logged,
            mock.patch.object(routines_serve.os, "dup2") as redirected,
            mock.patch.object(routines_serve.threading, "Thread") as thread,
            mock.patch("uvicorn.Server") as server,
            mock.patch("sys.stdout", stdout),
        ):
            token = Path(directory, "token")
            code = routines_serve.serve_brain(token, Path(directory, "budget.json"), Path(directory, "e"), Path("l"))
            served = token.read_text(encoding="utf-8")
            mode = token.stat().st_mode & 0o777
        self.assertEqual((code, mode, len(served)), (0, 0o600, 43))
        self.assertEqual(app.call_args.args[2], served)
        logged.assert_called_once_with(Path("l"))
        redirected.assert_called_once_with(2, 1)
        self.assertIs(thread.call_args.kwargs["target"], routines_serve._exit_with_parent)
        listener = server.return_value.run.call_args.kwargs["sockets"][0]
        self.assertEqual(int(stdout.getvalue()), listener.getsockname()[1])
        listener.close()

    def test_a_brain_exits_once_its_stdin_closes(self):
        with (
            mock.patch.object(routines_serve.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b""))),
            mock.patch.object(routines_serve.os, "_exit") as exited,
        ):
            routines_serve._exit_with_parent()
        exited.assert_called_once_with(0)


class StartTests(unittest.TestCase):
    def test_brain_starts_from_its_own_environment_without_teams(self):
        team = {"PYTHONPATH": ".:tests", "VIRTUAL_ENV": "/teams/.venv", "PYTHONHOME": "/x", "KEEP": "1"}
        with mock.patch.dict(routines_serve.os.environ, team):
            environment = routines_serve.brain_environment()
            with mock.patch.object(routines_serve.subprocess, "Popen") as popen:
                files = routines_serve.BrainFiles(Path("t"), Path("e"), Path("b"), Path("l"))
                routines_serve.start_brain("/brain/python", files)
        self.assertEqual(environment["KEEP"], "1")
        self.assertFalse({"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"} & set(environment))
        command = popen.call_args.args[0]
        self.assertEqual(command[:4], ["/brain/python", "-m", "eval.routines", "--serve-brain"])
        self.assertEqual(command[4:], ["--token-file", "t", "--ledger", "b", "--events", "e", "--brain-log", "l"])
        self.assertEqual(popen.call_args.kwargs["cwd"], routines_serve.BRAIN)
        self.assertNotIn("PYTHONPATH", popen.call_args.kwargs["env"])

    def test_brain_python_is_what_uv_resolves_in_brain(self):
        found = subprocess.CompletedProcess([], 0, stdout="/brain/.venv/bin/python\n")
        with (
            mock.patch.object(routines_serve.shutil, "which", return_value="/bin/uv"),
            mock.patch.object(routines_serve.subprocess, "run", return_value=found) as ran,
        ):
            self.assertEqual(routines_serve.brain_python(), "/brain/.venv/bin/python")
            self.assertEqual(ran.call_args.args[0][:2], ["/bin/uv", "run"])
            for failed in (subprocess.CompletedProcess([], 1, stdout="x"), subprocess.CompletedProcess([], 0, "")):
                ran.return_value = failed
                with self.assertRaises(SystemExit):
                    routines_serve.brain_python()
        with mock.patch.object(routines_serve.shutil, "which", return_value=None), self.assertRaises(SystemExit):
            routines_serve.brain_python()

    def _started(self, mode: str) -> subprocess.Popen:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        files = routines_serve.BrainFiles(Path(directory.name, "brain-7.token"), Path("e"), Path("b"))
        with mock.patch.dict(routines_serve.os.environ, {"FAKE_BRAIN_MODE": mode}):
            process = routines_serve.start_brain(routines_fake_brain.launcher(Path(directory.name)), files)
        for cleanup in (process.wait, process.kill, process.stdin.close, process.stdout.close):
            self.addCleanup(cleanup)
        return process

    def test_a_brain_url_comes_from_its_first_stdout_line(self):
        self.assertEqual(routines_serve.brain_url(self._started("serve")), "http://127.0.0.1:40007")
        for mode in ("no port", "0", "70000", "quiet"):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "stopped before it served"):
                routines_serve.brain_url(self._started(mode))
        silent = self._started("silent")
        with self.assertRaises(RuntimeError):
            routines_serve.brain_url(silent, deadline=0.2)
        self.assertIsNotNone(silent.wait(timeout=5))


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

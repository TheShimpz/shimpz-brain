"""Provider-free checks of the loopback provider origin: it meters, forwards, and refuses as the ceiling says."""

import contextlib
import http.client
import http.server
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from eval import ceiling as eval_ceiling
from eval import cost as eval_cost
from eval import provider_origin

BRAIN = Path(__file__).resolve().parents[1]
OPENAI_USAGE = {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 4}}
ANTHROPIC_USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 20}
REQUEST = {"model": "gpt-6-luna", "input": "hi", "max_output_tokens": 100}


class Upstream:
    """A fake provider on loopback: it records each request and answers the next scripted answer."""

    def __init__(self, answers: list[tuple[int, dict, list[tuple[str, str]]]]) -> None:
        self.answers = answers
        self.requests: list[tuple[str, str, list[tuple[str, str]], bytes]] = []
        self.received = threading.Event()
        self.release = threading.Event()
        self.release.set()
        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                upstream.requests.append((self.command, self.path, list(self.headers.items()), body))
                upstream.received.set()
                upstream.release.wait(10)
                status, answer, headers = upstream.answers.pop(0)
                content = json.dumps(answer).encode()
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                # A caller killed mid-request (the crash test) has closed its socket; nothing is left to answer.
                with contextlib.suppress(OSError):
                    self.wfile.write(content)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def connect(self, _host: str) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def send(origin: str, path: str, body: bytes, headers: dict[str, str]) -> tuple[int, dict, bytes]:
    host, port = origin.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=10)
    try:
        connection.request("POST", path, body, headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


class OriginTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.ledger = eval_cost.SharedBudget.create(self.directory / "budget.json", 1.0)
        self.state = self.directory / "state.json"
        self.events = self.directory / "events.jsonl"

    def _origin(self, answers, cap_ledger=None) -> tuple[provider_origin.ProviderOrigin, Upstream]:
        upstream = Upstream(answers)
        self.addCleanup(upstream.close)
        ceiling = eval_ceiling.Ceiling(0.0, 1_000, self.state, budget=cap_ledger or self.ledger)
        origin = provider_origin.ProviderOrigin(ceiling, events=self.events, connect=upstream.connect).start()
        self.addCleanup(origin.close)
        return origin, upstream

    def test_headers_and_body_pass_through_and_the_answer_comes_back_unchanged(self):
        answer = {"id": "resp_1", "output": [], "usage": OPENAI_USAGE}
        origin, upstream = self._origin([(200, answer, [("x-request-id", "req_7"), ("openai-model", "gpt-6-luna")])])
        body = json.dumps(REQUEST, separators=(", ", ": ")).encode()
        headers = {
            "Host": "api.openai.com",
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
            "OpenAI-Beta": "x",
            "Connection": "keep-alive",
            "Proxy-Authorization": "Basic never-forwarded",
        }
        status, returned, content = send(origin.origin, "/v1/responses", body, headers)
        self.assertEqual((status, json.loads(content)), (200, answer))
        self.assertEqual(returned["x-request-id"], "req_7")
        ((method, path, sent, forwarded),) = upstream.requests
        self.assertEqual((method, path, forwarded), ("POST", "/v1/responses", body))
        received = {name.lower(): value for name, value in sent}
        self.assertEqual(received["host"], "api.openai.com")
        self.assertEqual(received["authorization"], "Bearer test-key")
        self.assertEqual(received["openai-beta"], "x")
        self.assertEqual(received["content-length"], str(len(body)))
        self.assertNotIn("proxy-authorization", received)
        self.assertNotIn("connection", received)

    def test_the_output_limit_is_clamped_before_the_request_leaves(self):
        answer = {"content": [], "usage": ANTHROPIC_USAGE}
        origin, upstream = self._origin([(200, answer, [])])
        body = json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 64_000, "messages": []}).encode()
        status, _headers, _content = send(origin.origin, "/v1/messages", body, {"Host": "api.anthropic.com"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(upstream.requests[0][3])["max_tokens"], 1_000)
        self.assertEqual(dict(upstream.requests[0][2])["Host"], "api.anthropic.com")

    def test_the_worst_case_is_reserved_before_dispatch_and_settled_from_usage(self):
        answer = {"output": [], "usage": OPENAI_USAGE}
        origin, upstream = self._origin([(200, answer, [])])
        upstream.release.clear()
        body = json.dumps(REQUEST).encode()
        done: list[int] = []
        sender = threading.Thread(
            target=lambda: done.append(send(origin.origin, "/v1/responses", body, {"Host": "api.openai.com"})[0])
        )
        sender.start()
        self.assertTrue(upstream.received.wait(10))
        held = self.ledger.summary()["unsettled_usd"]
        bound = eval_cost.call_bound("gpt-6-luna", len(body) + eval_ceiling.REQUEST_ALLOWANCE_TOKENS, 1_000)
        self.assertAlmostEqual(held, round(bound, 6))
        upstream.release.set()
        sender.join(10)
        self.assertEqual(done, [200])
        summary = self.ledger.summary()
        expected = eval_cost.cost(
            eval_cost.Usage(model_calls=1, input_tokens=10, output_tokens=5, cache_read_tokens=4), "gpt-6-luna"
        )
        self.assertEqual(summary["unsettled_usd"], 0)
        self.assertAlmostEqual(summary["spent_usd"], round(expected.usd, 6))
        state = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual((state["requests"], state["spent_usd"]), (1, summary["spent_usd"]))

    def test_a_request_the_cap_cannot_hold_is_refused_and_never_forwarded(self):
        tiny = eval_cost.SharedBudget.create(self.directory / "tiny.json", 0.000001)
        origin, upstream = self._origin([], cap_ledger=tiny)
        status, _headers, _content = send(origin.origin, "/v1/responses", json.dumps(REQUEST).encode(), {})
        self.assertEqual((status, upstream.requests), (provider_origin.BUDGET_STATUS, []))
        self.assertEqual(json.loads(self.state.read_text(encoding="utf-8"))["refused"], 1)

    def test_count_endpoints_jev_and_other_paths_are_refused_without_forwarding(self):
        origin, upstream = self._origin([])
        refused = (
            ("/v1/responses/input_tokens", "api.openai.com"),
            ("/v1/messages/count_tokens", "api.anthropic.com"),
            ("/v1/systemone", "api.typesafe.ai"),
            ("/v1/chat/completions", "api.openai.com"),
            ("/v1/messages", "api.openai.com"),
            ("/v1/responses", "api.typesafe.ai"),
        )
        for path, host in refused:
            with self.subTest(path=path, host=host):
                status, _headers, _content = send(origin.origin, path, json.dumps(REQUEST).encode(), {"Host": host})
                self.assertEqual(status, provider_origin.UNSUPPORTED_STATUS)
        streamed = json.dumps({**REQUEST, "stream": True}).encode()
        self.assertEqual(send(origin.origin, "/v1/responses", streamed, {})[0], provider_origin.UNSUPPORTED_STATUS)
        self.assertEqual(upstream.requests, [])
        state = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual((state["unsupported"], state["requests"]), (len(refused) + 1, 0))
        self.assertEqual(self.ledger.summary()["spent_usd"], 0)

    def test_a_throttle_bills_nothing_and_is_written_to_the_events(self):
        stopped = {"stop_reason": "max_tokens", "usage": {"input_tokens": 1, "output_tokens": 1_000}}
        origin, _upstream = self._origin([(529, {"error": {}}, []), (200, stopped, [])])
        body = json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 10, "messages": []}).encode()
        self.assertEqual(send(origin.origin, "/v1/messages", body, {})[0], 529)
        self.assertEqual(self.ledger.summary()["spent_usd"], 0)
        self.assertEqual(send(origin.origin, "/v1/messages", body, {})[0], 200)
        lines = [json.loads(line) for line in self.events.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            lines,
            [
                {"status": 529, "throttled": True, "clamped": False},
                {"status": 200, "throttled": False, "clamped": True},
            ],
        )
        state = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual((state["throttled"], state["clamped"]), (1, 1))

    def test_unknown_usage_and_a_failed_provider_keep_the_reservation(self):
        origin, upstream = self._origin([(200, {"output": []}, []), (500, {"error": {}}, [])])
        body = json.dumps(REQUEST).encode()
        self.assertEqual(send(origin.origin, "/v1/responses", body, {})[0], 200)
        self.assertEqual(send(origin.origin, "/v1/responses", body, {})[0], 500)
        upstream.close()
        self.assertEqual(send(origin.origin, "/v1/responses", body, {})[0], 502)
        summary = self.ledger.summary()
        bound = eval_cost.call_bound("gpt-6-luna", len(body) + eval_ceiling.REQUEST_ALLOWANCE_TOKENS, 1_000)
        self.assertEqual(summary["unknown_settlements"], 3)
        self.assertAlmostEqual(summary["spent_usd"], round(3 * bound, 6))
        self.assertEqual(summary["unsettled_usd"], 0)


class CrashTests(unittest.TestCase):
    def test_a_crash_mid_request_keeps_its_reservation_in_the_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = eval_cost.SharedBudget.create(Path(directory, "budget.json"), 1.0)
            upstream = Upstream([(200, {"usage": OPENAI_USAGE}, [])])
            upstream.release.clear()
            self.addCleanup(upstream.close)
            child = subprocess.Popen(
                [sys.executable, "-m", "tests.provider_origin_child"],
                cwd=BRAIN,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.addCleanup(child.wait)
            self.addCleanup(child.stdin.close)
            self.addCleanup(child.stdout.close)
            config = {"ledger": str(ledger.path), "upstream": upstream.server.server_address[1]}
            child.stdin.write(json.dumps(config).encode() + b"\n")
            child.stdin.flush()
            port = int(child.stdout.readline())
            body = json.dumps(REQUEST).encode()

            def doomed() -> None:
                # The origin dies mid-request, so this request never gets an answer.
                with contextlib.suppress(OSError, http.client.HTTPException):
                    send(f"http://127.0.0.1:{port}", "/v1/responses", body, {})

            threading.Thread(target=doomed, daemon=True).start()
            self.assertTrue(upstream.received.wait(10))
            child.send_signal(signal.SIGKILL)
            child.wait(10)
            deadline = time.monotonic() + 5
            while ledger.summary()["unsettled_usd"] == 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            bound = eval_cost.call_bound("gpt-6-luna", len(body) + eval_ceiling.REQUEST_ALLOWANCE_TOKENS, 1_000)
            self.assertAlmostEqual(ledger.summary()["unsettled_usd"], round(bound, 6))
            # Nothing settles it later: the next reservation sees the dead request still charged.
            self.assertEqual(len(json.loads(ledger.path.read_text(encoding="utf-8"))["open"]), 1)


if __name__ == "__main__":
    unittest.main()

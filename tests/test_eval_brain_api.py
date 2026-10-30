"""Binary-free checks of the black-box Brain launcher and client: the command line, the bearer, and Team's bodies."""

import http.server
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from eval import brain_api

# A stand-in for the Rust binary: it records its argv and serves only GET /health, as `shimpz-brain serve` does.
FAKE_BRAIN = """#!{python}
import http.server, json, sys
from pathlib import Path
arguments = sys.argv[1:]
Path(arguments[arguments.index("--state-dir") + 1], "argv.json").write_text(json.dumps(arguments))
host, port = arguments[arguments.index("--listen") + 1].rsplit(":", 1)
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer((host, int(port)), Health).serve_forever()
"""

PROVIDER = brain_api.Provider("openai", "gpt-6-luna", "secret-test-key")
DNS = brain_api.assistant(
    "dns",
    "DNS manages zones.",
    [brain_api.action("list-zones", "List zones.", {"type": "object"}, {"type": "object"})],
)
# Team's turn context fields, in its order (teams/inference/client.py BrainRuntimeClient._context).
CONTEXT_FIELDS = [
    "thread_id",
    "team_name",
    "assistants",
    "provider",
    "memories",
    "routines",
    "routine_question",
    "routine_mode",
    "routine_rerun",
    "routine_schedule",
    "routine_output",
    "routine_settings_answer",
    "routine_selected",
    "attachments",
    "browser_available",
]


class FakeBrain:
    """A loopback Brain API that records each request and answers the next scripted (status, body)."""

    def __init__(self, answers: list[tuple[int, dict]]) -> None:
        self.answers, self.requests = answers, []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append((self.path, self.headers["Authorization"], body))
                status, answer = fake.answers.pop(0)
                content = json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def usage(**counts: int) -> dict[str, int]:
    fields = ("model_calls", "failed_calls", "unreported_calls", "input_tokens", "output_tokens")
    fields += ("cache_read_tokens", "cache_write_tokens", "provider_requests")
    return dict.fromkeys(fields, 0) | counts


ACTION_REQUIRED = {
    "status": "action-required",
    "reply": "",
    "clarification": None,
    "memory": [],
    "routine": None,
    "browser": None,
    "actions": [{"interrupt_id": "i1", "assistant_id": "dns", "action": "list-zones", "input": {}}],
    "usage": usage(model_calls=1, provider_requests=1),
}


class LaunchTests(unittest.TestCase):
    def test_no_brain_starts_without_an_executable_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory, "shimpz-brain")
            plain.write_text("", encoding="utf-8")
            for named in ("", str(Path(directory, "missing")), str(plain)):
                with mock.patch.dict(os.environ, {brain_api.BINARY_ENVIRONMENT: named}), self.assertRaises(SystemExit):
                    brain_api.binary()

    def test_only_a_loopback_provider_origin_is_accepted(self):
        for origin in ("https://api.openai.com", "http://10.0.0.1:80", "http://localhost:80"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                brain_api.Brain(origin)

    def test_the_binary_serves_with_a_private_token_and_state_and_the_provider_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory, "shimpz-brain")
            executable.write_text(FAKE_BRAIN.format(python=sys.executable), encoding="utf-8")
            executable.chmod(0o700)
            with (
                mock.patch.dict(os.environ, {brain_api.BINARY_ENVIRONMENT: str(executable)}),
                brain_api.Brain("http://127.0.0.1:9") as brain,
            ):
                argv = json.loads((brain.directory / "state" / "argv.json").read_text(encoding="utf-8"))
                token_mode = stat.S_IMODE(brain.token_file.stat().st_mode)
                state_mode = stat.S_IMODE((brain.directory / "state").stat().st_mode)
                directory_mode = stat.S_IMODE(brain.directory.stat().st_mode)
                root = brain.directory
                expected = [
                    "serve",
                    "--listen",
                    f"127.0.0.1:{brain.port}",
                    "--token-file",
                    str(brain.token_file),
                    "--state-dir",
                    str(brain.directory / "state"),
                    "--provider-origin",
                    "http://127.0.0.1:9",
                ]
                self.assertEqual(argv, expected)
                self.assertEqual((token_mode, state_mode, directory_mode), (0o600, 0o700, 0o700))
                self.assertGreaterEqual(len(brain.token_file.read_text(encoding="utf-8")), 32)
            self.assertFalse(root.exists())


class ClientTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.token = Path(directory.name, "token")
        self.token.write_text("bearer-test-token\n", encoding="utf-8")

    def _client(self, answers: list[tuple[int, dict]]) -> tuple[brain_api.Client, FakeBrain]:
        fake = FakeBrain(answers)
        self.addCleanup(fake.close)
        return brain_api.Client(fake.url, self.token), fake

    def test_a_turn_starts_and_resumes_with_teams_bodies(self):
        completed = {**ACTION_REQUIRED, "status": "completed", "reply": "Done.", "actions": []}
        client, fake = self._client([(200, ACTION_REQUIRED), (200, completed)])
        context = brain_api.TurnContext("eval:t:0", "Eval Team", (DNS,), PROVIDER, memories=())
        started = client.start(context, "List my zones.", locale="en")
        self.assertEqual(started.status, "action-required")
        self.assertEqual(started.actions, (brain_api.ActionRequest("i1", "dns", "list-zones", {}),))
        resumed = client.resume(context, {"i1": {"status": "ok", "output": {}}})
        self.assertEqual((resumed.status, resumed.reply, resumed.usage["model_calls"]), ("completed", "Done.", 1))
        (start_path, bearer, start), (resume_path, _bearer, resume) = fake.requests
        self.assertEqual(
            (start_path, resume_path, bearer), ("/v1/turns", "/v1/turns/resume", "Bearer bearer-test-token")
        )
        self.assertEqual(list(start), [*CONTEXT_FIELDS, "message", "locale", "conversation"])
        self.assertEqual(list(resume), [*CONTEXT_FIELDS, "results"])
        self.assertEqual(start["provider"], {"provider": "openai", "model": "gpt-6-luna", "api_key": "secret-test-key"})
        self.assertEqual(start["assistants"][0]["actions"][0]["input_files"], [])
        self.assertEqual((start["memories"], start["routines"]), ([], None))

    def test_any_other_status_raises_with_that_status(self):
        client, _fake = self._client([(502, {"detail": "Model provider request failed"})])
        context = brain_api.TurnContext("eval:t:1", "Eval Team", (DNS,), PROVIDER)
        with self.assertRaises(brain_api.BrainError) as raised:
            client.start(context, "Hi")
        self.assertEqual((raised.exception.status, raised.exception.path), (502, "/v1/turns"))
        self.assertNotIn("secret-test-key", str(raised.exception))

    def test_a_route_sends_a_decision_key_and_installable_assistants_only_when_given(self):
        route = {"intent": "ordinary-task", "query": "", "assistant_ids": [], "reply": "", "task_follows": False}
        client, fake = self._client([(200, route), (200, route)])
        client.intent_route(PROVIDER, brain_api.RouteRequest("Hi", "en"))
        installable = ({"name": "Exa", "summary": "Search the web."},)
        client.intent_route(
            PROVIDER, brain_api.RouteRequest("Hi", "en", decision_key="jev-test", installable=installable)
        )
        plain, decided = (body for _path, _bearer, body in fake.requests)
        fields = ["provider", "objective", "expected_intent", "candidates", "lifecycle_reference", "conversation"]
        self.assertEqual(list(plain), [*fields, "locale"])
        self.assertEqual(list(decided), [*fields, "locale", "decision_provider", "installable"])
        self.assertEqual(decided["decision_provider"], {"provider": "typesafe", "api_key": "jev-test"})


if __name__ == "__main__":
    unittest.main()

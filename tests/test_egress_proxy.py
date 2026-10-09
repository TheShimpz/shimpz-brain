"""Brain egress profile contracts: catalog policy, its audit stream, and startup on the shared CONNECT transport.

The transport itself (resolution, record order, ClientHello admission, splice, and capacity) is proven once by the
umbrella `.egress/tests` (ADR-0104); these tests prove what the Brain profile owns.
"""

import importlib
import importlib.util
import io
import ipaddress
import json
import socket
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
EGRESS = ROOT / "egress"
sys.path[:0] = [str(EGRESS), str(ROOT.parent / ".egress")]
app = importlib.import_module("app")
connect = importlib.import_module("connect")
audit_writer = importlib.import_module("audit_writer")

PUBLIC = ((socket.AF_INET, ("1.1.1.1", 443)),)


class BrainEgressHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.audit_path = Path(directory.name, "audit.jsonl")
        patcher = mock.patch.object(app, "AUDIT", audit_writer.AuditWriter(self.audit_path, "egress-proxy"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _exchange(self, request: bytes, *, client_host: str = "10.0.0.2") -> tuple[bytes, list[dict], mock.Mock]:
        client, proxy = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(proxy.close)
        client.sendall(request)
        upstream = mock.Mock()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(connect, "resolve_public", return_value=PUBLIC))
            stack.enter_context(mock.patch.object(connect.socket, "socket", return_value=upstream))
            stack.enter_context(mock.patch.object(connect.client_hello, "admit", return_value=b"hello"))
            stack.enter_context(mock.patch.object(connect, "tunnel"))
            app.Handler(proxy, (client_host, 1234), SimpleNamespace(allowed_hosts=frozenset({"api.openai.com"})))
        lines = self.audit_path.read_text(encoding="utf-8").splitlines() if self.audit_path.exists() else []
        return client.recv(256), [json.loads(line) for line in lines], upstream

    def test_only_an_exact_catalog_host_on_443_is_tunneled_and_recorded(self) -> None:
        response, events, upstream = self._exchange(b"CONNECT API.OpenAI.com.:443 HTTP/1.1\r\n\r\n")
        self.assertTrue(response.startswith(b"HTTP/1.1 200"))
        upstream.sendall.assert_called_once_with(b"hello")
        self.assertEqual(
            {key: events[0][key] for key in ("service", "level", "subject", "result", "reason", "addresses")},
            {
                "service": "egress-proxy",
                "level": "info",
                "subject": "API.OpenAI.com.:443",
                "result": "ok",
                "reason": "allowed",
                "addresses": ["1.1.1.1"],
            },
        )

    def test_non_connect_malformed_and_unlisted_targets_are_refused_and_recorded(self) -> None:
        for request, status, reason in (
            (b"GET / HTTP/1.1\r\n\r\n", 405, "method"),
            (b"CONNECT api.openai.com HTTP/1.1\r\n\r\n", 400, "target"),
            (b"CONNECT sub.api.openai.com:443 HTTP/1.1\r\n\r\n", 403, "target-rejected"),
            (b"CONNECT api.openai.com:80 HTTP/1.1\r\n\r\n", 403, "target-rejected"),
        ):
            with self.subTest(reason=reason, request=request):
                self.audit_path.unlink(missing_ok=True)
                response, events, upstream = self._exchange(request)
                self.assertTrue(response.startswith(f"HTTP/1.1 {status} ".encode()))
                self.assertEqual(
                    [(event["result"], event["code"], event["reason"]) for event in events],
                    [("denied", status, reason)],
                )
                self.assertEqual(events[0]["level"], "warn")
                upstream.connect.assert_not_called()

    def test_the_loopback_healthcheck_probe_is_a_distinct_non_warning_refusal(self) -> None:
        response, events, _upstream = self._exchange(b"GET / HTTP/1.1\r\n\r\n", client_host="127.0.0.1")
        self.assertTrue(response.startswith(b"HTTP/1.1 405 "))
        self.assertEqual((events[0]["level"], events[0]["source"]), ("info", "loopback-probe"))


class BrainEgressServerTests(unittest.TestCase):
    def test_server_refuses_empty_and_wildcard_policy_and_keeps_its_envelope(self) -> None:
        for allowed_hosts in (frozenset(), frozenset({"*"})):
            with self.subTest(allowed_hosts=allowed_hosts), self.assertRaisesRegex(ValueError, "Brain provider"):
                app.Server(("127.0.0.1", 0), app.Handler, allowed_hosts=allowed_hosts, bind_and_activate=False)
        server = app.Server(
            ("127.0.0.1", 0), app.Handler, allowed_hosts=frozenset({"api.openai.com"}), bind_and_activate=False
        )
        self.addCleanup(server.server_close)
        self.assertEqual(server.allowed_hosts, frozenset({"api.openai.com"}))
        self.assertEqual(server.request_queue_size, app.LISTEN_BACKLOG)
        self.assertEqual(app.Server.max_concurrency, app.MAX_CONCURRENCY)

    def test_successful_main_checks_audit_custody_binds_catalog_policy_and_serves(self) -> None:
        server = mock.Mock()
        stderr = io.StringIO()
        hosts = frozenset({"api.anthropic.com", "api.openai.com"})
        with (
            mock.patch.object(app.policy, "load_provider_hosts", return_value=hosts),
            mock.patch.object(app.AUDIT, "ensure_custody") as custody,
            mock.patch.object(app, "Server", return_value=server) as server_factory,
            redirect_stderr(stderr),
        ):
            app.main()
        custody.assert_called_once_with()
        server_factory.assert_called_once_with(
            (str(ipaddress.IPv4Address(0)), app.LISTEN_PORT),
            app.Handler,
            allowed_hosts=hosts | {"api.typesafe.ai"},
        )
        server.serve_forever.assert_called_once_with()
        self.assertIn(
            "providers=['api.anthropic.com', 'api.openai.com'] decisions=['api.typesafe.ai']", stderr.getvalue()
        )

    def test_startup_exits_before_binding_without_audit_custody(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(app.policy, "load_provider_hosts", return_value=frozenset({"api.openai.com"})),
            mock.patch.object(app.AUDIT, "ensure_custody", side_effect=audit_writer.AuditError("unsafe")),
            mock.patch.object(app, "Server") as server,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as caught,
        ):
            app.main()
        self.assertEqual(caught.exception.code, 1)
        server.assert_not_called()
        self.assertEqual(stderr.getvalue(), "brain-egress: audit custody is unavailable; refusing to start\n")

    def test_invalid_shipping_envelope_and_script_guard_fail_closed(self) -> None:
        specification = importlib.util.spec_from_file_location("brain_egress_invalid_config", EGRESS / "app.py")
        if specification is None or specification.loader is None:
            raise AssertionError("cannot execute Brain egress entrypoint")
        invalid = importlib.util.module_from_spec(specification)
        with (
            mock.patch.dict(
                "os.environ",
                {"SHIMPZ_EGRESS_MAX_CONCURRENCY": "1", "SHIMPZ_EGRESS_MAX_SOURCE_CONCURRENCY": "2"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "shipping resource envelope"),
        ):
            specification.loader.exec_module(invalid)

        specification = importlib.util.spec_from_file_location("__main__", EGRESS / "app.py")
        if specification is None or specification.loader is None:
            raise AssertionError("cannot execute Brain egress entrypoint")
        script = importlib.util.module_from_spec(specification)
        with (
            mock.patch.object(app.policy, "load_provider_hosts", side_effect=app.policy.ProviderPolicyError("closed")),
            mock.patch.dict(sys.modules, {"policy": app.policy}),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            specification.loader.exec_module(script)
        self.assertEqual(raised.exception.code, 1)


class BrainEgressHealthcheckTests(unittest.TestCase):
    def _run(self, *, response: bytes = b"", failure: OSError | None = None) -> int:
        specification = importlib.util.spec_from_file_location("__main__", EGRESS / "healthcheck.py")
        if specification is None or specification.loader is None:
            raise AssertionError("cannot execute Brain egress healthcheck")
        script = importlib.util.module_from_spec(specification)
        connection = mock.Mock()
        connection.recv.return_value = response
        with (
            mock.patch.object(
                socket,
                "create_connection",
                return_value=connection if failure is None else None,
                side_effect=failure,
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            specification.loader.exec_module(script)
        return int(raised.exception.code)

    def test_requires_a_live_connect_only_refusal(self) -> None:
        self.assertEqual(self._run(response=b"HTTP/1.1 405 Method Not Allowed"), 0)
        self.assertEqual(self._run(response=b"HTTP/1.1 200 Connection established"), 1)
        self.assertEqual(self._run(failure=OSError("unavailable")), 1)


if __name__ == "__main__":
    unittest.main()

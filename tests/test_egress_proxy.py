from __future__ import annotations

import importlib
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

EGRESS = Path(__file__).resolve().parents[1] / "egress"
sys.path.insert(0, str(EGRESS))
app = importlib.import_module("app")


class BrainEgressHandlerTests(unittest.TestCase):
    def _exchange(
        self,
        request: bytes,
        *,
        allowed_hosts: frozenset[str] = frozenset({"api.openai.com"}),
        client_host: str = "10.0.0.2",
        resolved: tuple[int, tuple] | None | object = mock.DEFAULT,
        upstream: mock.Mock | None = None,
        tunnel: mock.Mock | None = None,
    ) -> tuple[bytes, mock.Mock]:
        client, proxy = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(proxy.close)
        client.sendall(request)
        handler = object.__new__(app.Handler)
        handler.request = proxy
        handler.client_address = (client_host, 1234)
        handler.server = SimpleNamespace(allowed_hosts=allowed_hosts)
        audit_log = mock.Mock()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(app.audit, "log", audit_log))
            if resolved is not mock.DEFAULT:
                stack.enter_context(mock.patch.object(app, "_resolve_public", return_value=resolved))
            if upstream is not None:
                stack.enter_context(mock.patch.object(app.socket, "socket", return_value=upstream))
            if tunnel is not None:
                stack.enter_context(mock.patch.object(app.Handler, "_tunnel", tunnel))
            handler.handle()
        return client.recv(256), audit_log

    def test_handler_rejects_non_connect_malformed_and_unlisted_targets(self) -> None:
        response, audit_log = self._exchange(b"GET / HTTP/1.1\r\n\r\n")
        self.assertTrue(response.startswith(b"HTTP/1.1 405"))
        audit_log.assert_called_once_with(
            "connect",
            "GET / HTTP/1.1",
            result="denied",
            level="warn",
            code=405,
        )

        response, audit_log = self._exchange(b"CONNECT api.openai.com:nope HTTP/1.1\r\n\r\n")
        self.assertTrue(response.startswith(b"HTTP/1.1 400"))
        self.assertEqual(audit_log.call_args.kwargs["code"], 400)

        response, audit_log = self._exchange(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
        self.assertTrue(response.startswith(b"HTTP/1.1 403"))
        self.assertEqual(audit_log.call_args.kwargs["code"], 403)

    def test_handler_rejects_non_public_and_unreachable_destinations(self) -> None:
        request = b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n"
        response, audit_log = self._exchange(request, resolved=None)
        self.assertTrue(response.startswith(b"HTTP/1.1 403"))
        self.assertEqual(
            audit_log.call_args.kwargs["reason"],
            "internal or unresolvable destination",
        )

        upstream = mock.Mock()
        upstream.connect.side_effect = OSError("synthetic connect failure")
        response, audit_log = self._exchange(
            request,
            resolved=(socket.AF_INET, ("1.1.1.1", 443)),
            upstream=upstream,
        )
        self.assertTrue(response.startswith(b"HTTP/1.1 502"))
        self.assertEqual(audit_log.call_args.kwargs["result"], "error")
        self.assertNotIn("1.1.1.1", audit_log.call_args.args[1])

    def test_handler_connects_only_to_the_verified_address(self) -> None:
        request = b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n"
        upstream = mock.Mock()
        tunnel = mock.Mock()
        response, audit_log = self._exchange(
            request,
            resolved=(socket.AF_INET6, ("2606:4700::1111", 443, 0, 0)),
            upstream=upstream,
            tunnel=tunnel,
        )

        self.assertTrue(response.startswith(b"HTTP/1.1 200"))
        upstream.connect.assert_called_once_with(("2606:4700::1111", 443, 0, 0))
        audit_log.assert_called_once_with("connect", "api.openai.com:443", result="ok")
        tunnel.assert_called_once()

    def test_loopback_probe_is_a_distinct_non_warning_denial(self) -> None:
        response, audit_log = self._exchange(
            b"CONNECT health.invalid:443 HTTP/1.1\r\n\r\n",
            client_host="127.0.0.1",
        )
        self.assertTrue(response.startswith(b"HTTP/1.1 403"))
        self.assertEqual(audit_log.call_args.kwargs["level"], "info")
        self.assertEqual(audit_log.call_args.kwargs["source"], "loopback-probe")

    def test_request_parsing_is_bounded_and_defaults_to_https(self) -> None:
        self.assertEqual(app.Handler._split_target("api.openai.com"), ("api.openai.com", 443))
        self.assertEqual(app.Handler._split_target(""), (None, 443))
        self.assertEqual(app.Handler._split_target(":443"), (None, 443))

        oversized = mock.Mock()
        oversized.recv.return_value = b"x" * (app.BUFSIZE + 1)
        self.assertIsNone(app.Handler._read_request_line(oversized))

        failed = mock.Mock()
        failed.recv.side_effect = OSError("synthetic read failure")
        self.assertIsNone(app.Handler._read_request_line(failed))

        closed = mock.Mock()
        closed.recv.return_value = b""
        self.assertIsNone(app.Handler._read_request_line(closed))

    def test_resolution_rejects_errors_invalid_mixed_and_empty_answers(self) -> None:
        with mock.patch.object(app.socket, "getaddrinfo", side_effect=OSError):
            self.assertIsNone(app._resolve_public("api.openai.com", 443))
        with mock.patch.object(
            app.socket,
            "getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 443))],
        ):
            self.assertIsNone(app._resolve_public("api.openai.com", 443))
        with mock.patch.object(
            app.socket,
            "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443)),
            ],
        ):
            self.assertIsNone(app._resolve_public("api.openai.com", 443))
        with mock.patch.object(app.socket, "getaddrinfo", return_value=[]):
            self.assertIsNone(app._resolve_public("api.openai.com", 443))

    def test_tunnel_forwards_and_closes_both_sides(self) -> None:
        left = mock.Mock()
        right = mock.Mock()
        left.recv.return_value = b"payload"
        with mock.patch.object(
            app.select,
            "select",
            side_effect=[([left], [], []), ([], [], [])],
        ):
            app.Handler._tunnel(left, right)
        right.sendall.assert_called_once_with(b"payload")
        left.close.assert_called_once()
        right.close.assert_called_once()


class BrainEgressServerTests(unittest.TestCase):
    def _server(self, **kwargs) -> app.Server:
        server = app.Server(
            ("127.0.0.1", 0),
            app.Handler,
            allowed_hosts=frozenset({"api.openai.com"}),
            bind_and_activate=False,
            **kwargs,
        )
        self.addCleanup(server.server_close)
        return server

    def test_server_refuses_empty_wildcard_and_invalid_capacity(self) -> None:
        for allowed_hosts in (frozenset(), frozenset({"*"})):
            with self.subTest(allowed_hosts=allowed_hosts), self.assertRaises(ValueError):
                app.Server(
                    ("127.0.0.1", 0),
                    app.Handler,
                    allowed_hosts=allowed_hosts,
                    bind_and_activate=False,
                )
        with self.assertRaises(ValueError):
            self._server(max_concurrency=1, max_source_concurrency=2)

    def test_capacity_is_bounded_globally_and_per_source(self) -> None:
        server = self._server(max_concurrency=2, max_source_concurrency=1)
        self.assertTrue(server._acquire_request_slot(("10.0.0.2", 1)))
        self.assertFalse(server._acquire_request_slot(("10.0.0.2", 2)))
        self.assertTrue(server._acquire_request_slot(("10.0.0.3", 3)))
        self.assertFalse(server._acquire_request_slot(("10.0.0.4", 4)))
        server._release_request_slot(("10.0.0.2", 1))
        server._release_request_slot(("10.0.0.3", 3))
        self.assertEqual(server._source_counts, {})

    def test_overload_is_answered_without_creating_a_worker(self) -> None:
        server = self._server(max_concurrency=1, max_source_concurrency=1)
        self.assertTrue(server._acquire_request_slot(("10.0.0.2", 1)))
        client, accepted = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(accepted.close)
        with mock.patch.object(server, "shutdown_request") as shutdown:
            server.process_request(accepted, ("10.0.0.3", 2))
        self.assertTrue(client.recv(256).startswith(b"HTTP/1.1 503"))
        shutdown.assert_called_once_with(accepted)
        server._release_request_slot(("10.0.0.2", 1))

    def test_worker_creation_failure_releases_capacity(self) -> None:
        server = self._server(max_concurrency=1, max_source_concurrency=1)
        request = mock.Mock()
        with (
            mock.patch.object(
                app.socketserver.ThreadingTCPServer,
                "process_request",
                side_effect=RuntimeError("synthetic worker failure"),
            ),
            mock.patch.object(server, "shutdown_request") as shutdown,
            self.assertRaises(RuntimeError),
        ):
            server.process_request(request, ("10.0.0.2", 1))
        shutdown.assert_called_once_with(request)
        self.assertEqual(server._source_counts, {})

    def test_worker_completion_always_releases_capacity(self) -> None:
        server = self._server(max_concurrency=1, max_source_concurrency=1)
        self.assertTrue(server._acquire_request_slot(("10.0.0.2", 1)))
        with (
            mock.patch.object(
                app.socketserver.ThreadingTCPServer,
                "process_request_thread",
                side_effect=RuntimeError("synthetic handler failure"),
            ),
            self.assertRaises(RuntimeError),
        ):
            server.process_request_thread(mock.Mock(), ("10.0.0.2", 1))
        self.assertEqual(server._source_counts, {})

    def test_successful_main_binds_catalog_policy_and_serves(self) -> None:
        server = mock.Mock()
        stderr = io.StringIO()
        hosts = frozenset({"api.anthropic.com", "api.openai.com"})
        with (
            mock.patch.object(app.policy, "load_provider_hosts", return_value=hosts),
            mock.patch.object(app, "Server", return_value=server) as server_factory,
            redirect_stderr(stderr),
        ):
            app.main()
        server_factory.assert_called_once_with(
            (str(ipaddress.IPv4Address(0)), app.LISTEN_PORT),
            app.Handler,
            allowed_hosts=hosts,
        )
        server.serve_forever.assert_called_once_with()
        self.assertIn("providers=['api.anthropic.com', 'api.openai.com']", stderr.getvalue())


class BrainEgressAuditTests(unittest.TestCase):
    def test_audit_writes_structured_events_and_rotates_bounded_files(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        audit_path = Path(directory.name) / "audit.jsonl"
        stdout = io.StringIO()
        with (
            mock.patch.object(app.audit, "AUDIT_PATH", audit_path),
            mock.patch.object(app.audit, "MAX_BYTES", 1),
            redirect_stderr(io.StringIO()),
            mock.patch("sys.stdout", stdout),
        ):
            trace_id = app.audit.log(
                "connect",
                "api.openai.com:443",
                result="ok",
                principal="brain",
            )
            app.audit.log("connect", "example.com:443", result="denied", code=403)

        event = json.loads(stdout.getvalue().splitlines()[0])
        self.assertEqual(event["trace_id"], trace_id)
        self.assertEqual(event["level"], "info")
        self.assertEqual(event["principal"], "brain")
        self.assertEqual(event["subject"], "api.openai.com:443")
        self.assertEqual(audit_path.with_name("audit.jsonl.1").read_text().count("\n"), 1)
        denied = json.loads(audit_path.read_text())
        self.assertEqual(denied["level"], "warn")
        self.assertEqual(denied["code"], 403)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import io
import ipaddress
import itertools
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack, redirect_stderr, suppress
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

EGRESS = Path(__file__).resolve().parents[1] / "egress"
sys.path.insert(0, str(EGRESS))
app = importlib.import_module("app")


def _later() -> float:
    return time.monotonic() + app.CONNECT_TIMEOUT


class BrainEgressHandlerTests(unittest.TestCase):
    def _exchange(
        self,
        request: bytes,
        *,
        allowed_hosts: frozenset[str] = frozenset({"api.openai.com"}),
        client_host: str = "10.0.0.2",
        resolved: tuple[tuple[int, tuple], ...] | None | object = mock.DEFAULT,
        upstream: mock.Mock | list[object] | None = None,
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
            if isinstance(upstream, list):
                stack.enter_context(mock.patch.object(app.socket, "socket", side_effect=upstream))
            elif upstream is not None:
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
            resolved=((socket.AF_INET, ("1.1.1.1", 443)),),
            upstream=upstream,
        )
        self.assertTrue(response.startswith(b"HTTP/1.1 502"))
        upstream.close.assert_called_once_with()
        self.assertEqual(audit_log.call_args.kwargs["result"], "error")
        self.assertNotIn("1.1.1.1", audit_log.call_args.args[1])

    def test_handler_connects_only_to_the_verified_address(self) -> None:
        request = b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n"
        upstream = mock.Mock()
        tunnel = mock.Mock()
        response, audit_log = self._exchange(
            request,
            resolved=((socket.AF_INET6, ("2606:4700::1111", 443, 0, 0)),),
            upstream=upstream,
            tunnel=tunnel,
        )

        self.assertTrue(response.startswith(b"HTTP/1.1 200"))
        upstream.connect.assert_called_once_with(("2606:4700::1111", 443, 0, 0))
        audit_log.assert_called_once_with("connect", "api.openai.com:443", result="ok", address="2606:4700::1111")
        tunnel.assert_called_once()

    def test_handler_falls_through_to_the_next_verified_address(self) -> None:
        refused, accepted = mock.Mock(), mock.Mock()
        refused.connect.side_effect = ConnectionRefusedError("refused")
        tunnel = mock.Mock()
        resolved = ((socket.AF_INET, ("1.1.1.1", 443)), (socket.AF_INET, ("1.0.0.1", 443)))
        response, audit_log = self._exchange(
            b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n",
            resolved=resolved,
            upstream=[refused, accepted],
            tunnel=tunnel,
        )

        self.assertTrue(response.startswith(b"HTTP/1.1 200"))
        refused.connect.assert_called_once_with(("1.1.1.1", 443))
        refused.close.assert_called_once_with()
        accepted.connect.assert_called_once_with(("1.0.0.1", 443))
        audit_log.assert_called_once_with("connect", "api.openai.com:443", result="ok", address="1.0.0.1")
        self.assertIs(tunnel.call_args.args[1], accepted)

    def test_handler_reports_one_error_when_every_verified_address_refuses(self) -> None:
        upstreams = [mock.Mock(), mock.Mock()]
        for upstream in upstreams:
            upstream.connect.side_effect = ConnectionRefusedError("synthetic refusal")
        resolved = (
            (socket.AF_INET6, ("2606:4700::1111", 443, 0, 0)),
            (socket.AF_INET, ("1.1.1.1", 443)),
            (socket.AF_INET, ("1.0.0.1", 443)),
        )
        response, audit_log = self._exchange(
            b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n",
            resolved=resolved,
            upstream=[OSError("address family unavailable"), *upstreams],
        )

        self.assertTrue(response.startswith(b"HTTP/1.1 502"))
        audit_log.assert_called_once_with("connect", "api.openai.com:443", result="error", reason="synthetic refusal")
        for upstream in upstreams:
            upstream.close.assert_called_once_with()

    def test_handler_refuses_a_mixed_answer_before_any_connection(self) -> None:
        answers = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.5", 443)),
        ]
        constructor = mock.Mock()
        with mock.patch.object(app.socket, "getaddrinfo", return_value=answers):
            response, audit_log = self._exchange(b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n", upstream=[constructor])

        self.assertTrue(response.startswith(b"HTTP/1.1 403"))
        self.assertEqual(audit_log.call_args.kwargs["reason"], "internal or unresolvable destination")
        self.assertEqual(constructor.method_calls, [])

    def test_connection_attempts_share_one_total_deadline(self) -> None:
        slow, unused = mock.Mock(), mock.Mock()
        slow.connect.side_effect = TimeoutError("timed out")
        clock = iter([100.0, 100.0 + app.CONNECT_TIMEOUT])
        resolved = ((socket.AF_INET, ("1.1.1.1", 443)), (socket.AF_INET, ("1.0.0.1", 443)))
        with (
            mock.patch.object(app.time, "monotonic", side_effect=lambda: next(clock)),
            mock.patch.object(app.socket, "socket", side_effect=[slow, unused]) as constructor,
            self.assertRaisesRegex(TimeoutError, "timed out"),
        ):
            app._connect_first(resolved, 100.0 + app.CONNECT_TIMEOUT)

        slow.settimeout.assert_called_once_with(app.CONNECT_TIMEOUT)
        constructor.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        unused.connect.assert_not_called()

    def test_a_refused_first_address_falls_through_to_a_live_second_address(self) -> None:
        listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(listener.close)
        closed = socket.create_server(("127.0.0.1", 0))
        refused_port = closed.getsockname()[1]
        closed.close()
        live = listener.getsockname()

        upstream, address = app._connect_first(
            ((socket.AF_INET, ("127.0.0.1", refused_port)), (socket.AF_INET, live)), _later()
        )
        self.addCleanup(upstream.close)
        accepted, _peer = listener.accept()
        accepted.close()

        self.assertEqual(upstream.getpeername(), live)
        self.assertEqual(address, "127.0.0.1")

    ANSWER = ((socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 443)),)

    def _isolate_resolver(self) -> None:
        """Give the test a private one-slot resolver so a stuck lookup never reaches the shared pool."""
        self.slots = threading.BoundedSemaphore(1)
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        patcher = mock.patch.object(app, "_RESOLVER_SLOTS", self.slots)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _stuck_lookup(self, *_args, **_kwargs) -> list:
        self.release.wait(5)
        return list(self.ANSWER)

    def test_slow_resolution_beyond_the_deadline_gets_the_existing_denial(self) -> None:
        self._isolate_resolver()
        never = mock.Mock()
        with (
            mock.patch.object(app, "CONNECT_TIMEOUT", 0.05),
            mock.patch.object(app.socket, "getaddrinfo", side_effect=self._stuck_lookup),
        ):
            response, audit_log = self._exchange(b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n", upstream=[never])

        self.assertTrue(response.startswith(b"HTTP/1.1 403"))
        self.assertEqual(audit_log.call_args.kwargs["reason"], "internal or unresolvable destination")
        self.assertEqual(never.method_calls, [])

    def test_resolution_and_connect_share_one_budget(self) -> None:
        self._isolate_resolver()
        for spent, expected in ((6.0, b"HTTP/1.1 200"), (app.CONNECT_TIMEOUT, b"HTTP/1.1 502")):
            with self.subTest(spent=spent):
                now = [100.0]

                def slow_lookup(*_args, spent=spent, now=now, **_kwargs) -> list:
                    now[0] += spent
                    return list(self.ANSWER)

                upstream = mock.Mock()
                with (
                    mock.patch.object(app.time, "monotonic", side_effect=lambda now=now: now[0]),
                    mock.patch.object(app.socket, "getaddrinfo", side_effect=slow_lookup),
                ):
                    response, _audit_log = self._exchange(
                        b"CONNECT api.openai.com:443 HTTP/1.1\r\n\r\n", upstream=[upstream], tunnel=mock.Mock()
                    )

                self.assertTrue(response.startswith(expected))
                if spent < app.CONNECT_TIMEOUT:
                    upstream.settimeout.assert_called_once_with(app.CONNECT_TIMEOUT - spent)
                else:
                    self.assertEqual(upstream.method_calls, [])

    def test_a_waiter_past_the_deadline_is_denied_without_resolving(self) -> None:
        self._isolate_resolver()
        with mock.patch.object(app.socket, "getaddrinfo", side_effect=self._stuck_lookup) as lookup:
            self.assertIsNone(app._resolve_public("api.openai.com", 443, time.monotonic() + 0.05))
            started = time.monotonic()
            self.assertIsNone(app._resolve_public("api.openai.com", 443, started + 0.2))
            self.assertGreaterEqual(time.monotonic() - started, 0.1)
        lookup.assert_called_once()

    def test_a_waiter_gets_the_permit_once_a_lookup_finishes_within_the_deadline(self) -> None:
        self._isolate_resolver()
        with mock.patch.object(app.socket, "getaddrinfo", side_effect=self._stuck_lookup):
            self.assertIsNone(app._resolve_public("api.openai.com", 443, time.monotonic() + 0.05))
        finish = threading.Timer(0.1, self.release.set)
        self.addCleanup(finish.cancel)
        with mock.patch.object(app.socket, "getaddrinfo", return_value=list(self.ANSWER)) as answered:
            finish.start()
            self.assertEqual(app._resolve_public("api.openai.com", 443, _later()), (self.ANSWER[0][0::4],))
        self.assertTrue(self.release.is_set())
        answered.assert_called_once()

    def test_a_resolver_thread_that_cannot_start_returns_its_permit(self) -> None:
        self._isolate_resolver()
        with (
            mock.patch.object(app.threading.Thread, "start", side_effect=RuntimeError("can't start new thread")),
            mock.patch.object(app.socket, "getaddrinfo", return_value=list(self.ANSWER)) as lookup,
        ):
            for _attempt in range(2):
                self.assertIsNone(app._resolve_public("api.openai.com", 443, _later()))
        lookup.assert_not_called()
        with mock.patch.object(app.socket, "getaddrinfo", return_value=list(self.ANSWER)):
            self.assertEqual(app._resolve_public("api.openai.com", 443, _later()), (self.ANSWER[0][0::4],))

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

        handler = object.__new__(app.Handler)
        handler.request = mock.Mock()
        handler.client_address = ("10.0.0.2", 1234)
        with mock.patch.object(app.Handler, "_read_request_line", return_value=None):
            self.assertIsNone(handler.handle())

    def test_a_trickled_request_header_is_dropped_at_an_absolute_deadline(self) -> None:
        trickle = mock.Mock()
        trickle.recv.return_value = b"C"  # each byte arrives well inside the per-read inactivity timeout
        with mock.patch("time.monotonic", side_effect=itertools.count()):
            self.assertIsNone(app.Handler._read_request_line(trickle))
        self.assertLess(trickle.recv.call_count, app.CONNECT_TIMEOUT)
        self.assertTrue(all(0 < call.args[0] <= app.CONNECT_TIMEOUT for call in trickle.settimeout.call_args_list))

    def test_resolution_rejects_errors_invalid_mixed_and_empty_answers(self) -> None:
        with mock.patch.object(app.socket, "getaddrinfo", side_effect=OSError):
            self.assertIsNone(app._resolve_public("api.openai.com", 443, _later()))
        for refused in (
            [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 443))],
            [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443)),
            ],
            [],
        ):
            with self.subTest(answers=refused), mock.patch.object(app.socket, "getaddrinfo", return_value=refused):
                self.assertIsNone(app._resolve_public("api.openai.com", 443, _later()))
        with mock.patch.object(
            app.socket,
            "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 443)),
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("2606:4700::1111", 443, 0, 0)),
            ],
        ):
            self.assertEqual(
                app._resolve_public("api.openai.com", 443, _later()),
                ((socket.AF_INET, ("1.1.1.1", 443)), (socket.AF_INET6, ("2606:4700::1111", 443, 0, 0))),
            )

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

        for result in (b"", OSError("closed")):
            left = mock.Mock()
            right = mock.Mock()
            if isinstance(result, Exception):
                left.recv.side_effect = result
            else:
                left.recv.return_value = result
            with (
                self.subTest(result=result),
                mock.patch.object(app.select, "select", return_value=([left], [], [])),
            ):
                app.Handler._tunnel(left, right)
            left.close.assert_called_once_with()
            right.close.assert_called_once_with()


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

        server = self._server(max_concurrency=2, max_source_concurrency=2)
        self.assertTrue(server._acquire_request_slot(("10.0.0.4", 1)))
        self.assertTrue(server._acquire_request_slot(("10.0.0.4", 2)))
        server._release_request_slot(("10.0.0.4", 1))
        self.assertEqual(server._source_counts, {"10.0.0.4": 1})
        server._release_request_slot(("10.0.0.4", 2))

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
            allowed_hosts=hosts | {"api.typesafe.ai"},
        )
        server.serve_forever.assert_called_once_with()
        self.assertIn(
            "providers=['api.anthropic.com', 'api.openai.com'] decisions=['api.typesafe.ai']", stderr.getvalue()
        )

    def test_accepted_socket_gets_the_connect_timeout(self) -> None:
        server = self._server()
        request = mock.Mock()
        with mock.patch.object(
            app.socketserver.ThreadingTCPServer,
            "get_request",
            return_value=(request, ("10.0.0.2", 1234)),
        ):
            self.assertEqual(server.get_request(), (request, ("10.0.0.2", 1234)))
        request.settimeout.assert_called_once_with(app.CONNECT_TIMEOUT)

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
            mock.patch.dict(sys.modules, {"audit": app.audit, "policy": app.policy}),
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


class BrainEgressAuditTests(unittest.TestCase):
    def test_concurrent_audit_writes_rotate_one_at_a_time_and_keep_every_line(self) -> None:
        # Each rotation waits up to a second for the other writer to rotate too; the audit lock must keep them apart.
        audit = app.audit
        active = peak = 0
        guard = threading.Lock()
        barrier = threading.Barrier(2, timeout=1)
        rotate = audit._rotate

        def overlapping_rotate() -> None:
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            with suppress(threading.BrokenBarrierError):
                barrier.wait()
            rotate()
            with guard:
                active -= 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "audit.jsonl")
            with (
                mock.patch.object(audit, "AUDIT_PATH", path),
                mock.patch.object(audit, "_rotate", overlapping_rotate),
                mock.patch("sys.stdout", io.StringIO()),
            ):
                writers = [
                    threading.Thread(
                        target=audit.log, args=("connect", f"host{index}.example:443"), kwargs={"result": "ok"}
                    )
                    for index in range(2)
                ]
                for writer in writers:
                    writer.start()
                for writer in writers:
                    writer.join()
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(peak, 1)
        self.assertEqual(len(lines), 2)

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
            app.audit.log("connect", "api.anthropic.com:443", result="ok")

        event = json.loads(stdout.getvalue().splitlines()[0])
        self.assertEqual(event["trace_id"], trace_id)
        self.assertEqual(event["level"], "info")
        self.assertEqual(event["principal"], "brain")
        self.assertEqual(event["subject"], "api.openai.com:443")
        self.assertEqual(audit_path.with_name("audit.jsonl.1").read_text().count("\n"), 1)
        self.assertEqual(audit_path.with_name("audit.jsonl.2").read_text().count("\n"), 1)
        denied = json.loads(audit_path.with_name("audit.jsonl.1").read_text())
        self.assertEqual(denied["level"], "warn")
        self.assertEqual(denied["code"], 403)


if __name__ == "__main__":
    unittest.main()

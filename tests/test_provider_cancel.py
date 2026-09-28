"""Turn-scoped provider cancellation: a Stop wakes only its own blocked call, never another turn (ADR-0079)."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import agent_runtime
import httpcore
import httpx
import provider_cancel
import runtime_api
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from test_agent_runtime import ToolAwareFakeModel, action, assistant, context
from test_runtime_api import TOKEN, body

NO_PROXY_ENV = {"HTTPS_PROXY": "", "https_proxy": ""}


class _Server:
    """A local HTTP server that answers only when its ``respond`` flag says so."""

    def __init__(self, *, respond: bool) -> None:
        self.respond = respond
        self.accepted = 0
        self.received = threading.Event()
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _address = self._listener.accept()
            except OSError:
                return
            self.accepted += 1
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            buffer = b""
            while True:
                while b"\r\n\r\n" not in buffer:
                    chunk = connection.recv(65536)
                    if not chunk:
                        return
                    buffer += chunk
                _headers, _separator, buffer = buffer.partition(b"\r\n\r\n")
                self.received.set()
                if not self.respond:
                    while connection.recv(65536):
                        pass
                    return
                connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def close(self) -> None:
        self._listener.close()


def _run(scope: provider_cancel.CancelScope, work) -> Future:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(scope.run, work)
    executor.shutdown(wait=False)
    return future


class CancellableTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, NO_PROXY_ENV)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = provider_cancel.client()
        self.addCleanup(self.client.close)

    def _server(self, *, respond: bool) -> _Server:
        server = _Server(respond=respond)
        self.addCleanup(server.close)
        return server

    def test_cancel_wakes_a_blocked_read_without_touching_another_request_on_the_pool(self):
        hanging = self._server(respond=False)
        answering = self._server(respond=True)
        scope = provider_cancel.CancelScope()
        outcome = _run(scope, lambda: self.client.get(hanging.url(), timeout=30))
        self.assertTrue(hanging.received.wait(5))

        # An unscoped call on the same client completes while the scoped one is blocked.
        self.assertEqual(self.client.get(answering.url()).text, "ok")
        started = time.monotonic()
        scope.cancel()
        self.assertIsInstance(outcome.exception(5), provider_cancel.ProviderCallCancelled)
        self.assertLess(time.monotonic() - started, 1)
        # No retry or later request of the cancelled scope reaches the network.
        with self.assertRaises(provider_cancel.ProviderCallCancelled):
            scope.run(lambda: self.client.get(answering.url()))
        self.assertEqual(hanging.accepted, 1)
        self.assertEqual(self.client.get(answering.url()).text, "ok")

    def test_cancel_wakes_a_stalled_real_tls_handshake_without_touching_the_pool(self):
        stalled = self._server(respond=False)
        answering = self._server(respond=True)
        scope = provider_cancel.CancelScope()
        # The server never answers the ClientHello, so the real ssl handshake blocks.
        outcome = _run(scope, lambda: self.client.get(f"https://127.0.0.1:{stalled.port}/", timeout=30))
        while not scope._sockets:
            time.sleep(0.01)
        self.assertEqual(self.client.get(answering.url()).text, "ok")
        started = time.monotonic()
        scope.cancel()
        self.assertIsInstance(outcome.exception(5), provider_cancel.ProviderCallCancelled)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(self.client.get(answering.url()).text, "ok")

    def test_a_scope_cancelled_before_its_request_never_connects(self):
        server = self._server(respond=True)
        scope = provider_cancel.CancelScope()
        scope.cancel()
        with self.assertRaises(provider_cancel.ProviderCallCancelled):
            scope.run(lambda: self.client.get(server.url()))
        self.assertEqual(server.accepted, 0)

    def test_connect_and_pool_waits_are_bounded(self):
        seen: list[dict[str, Any]] = []

        def record(_transport, request):
            seen.append(dict(request.extensions["timeout"]))
            return httpx.Response(200)

        with mock.patch.object(httpx.HTTPTransport, "handle_request", record):
            self.client.get("http://127.0.0.1:9/", timeout=60)
            self.client.get("http://127.0.0.1:9/", timeout=httpx.Timeout(60, connect=1, pool=2))
        self.assertEqual(seen[0]["connect"], provider_cancel.CONNECT_TIMEOUT_SECONDS)
        self.assertEqual(seen[0]["pool"], provider_cancel.POOL_TIMEOUT_SECONDS)
        self.assertEqual(seen[0]["read"], 60)
        self.assertEqual((seen[1]["connect"], seen[1]["pool"]), (1, 2))

        with mock.patch.object(httpcore.SyncBackend, "connect_tcp", return_value=mock.Mock()) as connect:
            provider_cancel._Backend().connect_tcp("127.0.0.1", 9, timeout=None)
            provider_cancel._Backend().connect_tcp("127.0.0.1", 9, timeout=0.5)
        self.assertEqual(connect.call_args_list[0].args[2], provider_cancel.CONNECT_TIMEOUT_SECONDS)
        self.assertEqual(connect.call_args_list[1].args[2], 0.5)

    def test_a_cancel_during_connect_closes_the_new_connection(self):
        scope = provider_cancel.CancelScope()
        stream = mock.Mock()

        def connect(*_args, **_kwargs):
            scope.cancel()
            return stream

        with (
            mock.patch.object(httpcore.SyncBackend, "connect_tcp", side_effect=connect),
            self.assertRaises(provider_cancel.ProviderCallCancelled),
        ):
            scope.run(lambda: provider_cancel._Backend().connect_tcp("127.0.0.1", 9))
        stream.close.assert_called_once()

    def test_proxy_environment_builds_a_tunnel_over_the_cancellable_backend(self):
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://shimpz-brain-egress:8888"}):
            proxied = provider_cancel.client()
        self.addCleanup(proxied.close)
        pool = proxied._transport._pool
        self.assertIsInstance(pool, httpcore.HTTPProxy)
        self.assertIsInstance(pool._network_backend, provider_cancel._Backend)
        self.assertIsInstance(self.client._transport._pool, httpcore.ConnectionPool)
        self.assertFalse(proxied._trust_env)


class _SocketStream(httpcore.NetworkStream):
    """A network stream over one end of a socket pair, blocking like a real provider socket."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        data = self.sock.recv(max_bytes)
        if not data:
            raise httpcore.ReadError("connection closed")
        return data

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.sock.sendall(buffer)

    def close(self) -> None:
        self.sock.close()

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        # A handshake blocks on the peer exactly like a read.
        self.read(1)
        return _SocketStream(self.sock)

    def get_extra_info(self, info: str) -> object:
        return self.sock if info == "socket" else None


class StreamScopeTests(unittest.TestCase):
    def _pair(self) -> tuple[provider_cancel._Stream, socket.socket]:
        ours, peer = socket.socketpair()
        self.addCleanup(ours.close)
        self.addCleanup(peer.close)
        return provider_cancel._Stream(_SocketStream(ours)), peer

    def test_a_blocked_tls_handshake_is_cancelled(self):
        stream, _peer = self._pair()
        scope = provider_cancel.CancelScope()
        outcome = _run(scope, lambda: stream.start_tls(None, "api.anthropic.com"))
        while not scope._sockets:
            time.sleep(0.01)
        scope.cancel()
        self.assertIsInstance(outcome.exception(5), provider_cancel.ProviderCallCancelled)

    def test_reads_and_writes_register_only_while_they_run(self):
        stream, peer = self._pair()
        scope = provider_cancel.CancelScope()
        peer.sendall(b"hi")
        self.assertEqual(scope.run(lambda: stream.read(2)), b"hi")
        scope.run(lambda: stream.write(b"yo"))
        self.assertEqual(peer.recv(2), b"yo")
        self.assertEqual(scope._sockets, set())
        tls = scope.run(lambda: (peer.sendall(b"x"), stream.start_tls(None))[1])
        self.assertIsInstance(tls, provider_cancel._Stream)
        # Outside any scope the same stream is a plain pass-through.
        peer.sendall(b"ok")
        self.assertEqual(stream.read(2), b"ok")
        peer.sendall(b"x")
        self.assertIsInstance(stream.start_tls(None), provider_cancel._Stream)
        self.assertIsNone(stream.get_extra_info("server_addr"))
        stream.close()

    def test_io_in_a_cancelled_scope_never_starts(self):
        stream, peer = self._pair()
        scope = provider_cancel.CancelScope()
        scope.cancel()
        peer.sendall(b"unread")
        with self.assertRaises(provider_cancel.ProviderCallCancelled):
            scope.run(lambda: stream.read(6))
        self.assertEqual(stream.read(6), b"unread")

    def test_an_error_is_only_a_cancellation_when_the_scope_was_cancelled(self):
        stream, peer = self._pair()
        peer.close()
        with self.assertRaises(httpcore.ReadError):
            provider_cancel.CancelScope().run(lambda: stream.read(1))

    def test_a_read_that_raced_a_cancel_still_stops_the_turn(self):
        stream, peer = self._pair()
        scope = provider_cancel.CancelScope()
        peer.sendall(b"late")

        original = stream._inner.read

        def read(*_args) -> bytes:
            data = original(4)
            scope.cancel()
            return data

        with (
            mock.patch.object(stream._inner, "read", side_effect=read),
            self.assertRaises(provider_cancel.ProviderCallCancelled),
        ):
            scope.run(lambda: stream.read(4))

    def test_shutdown_ignores_an_already_closed_socket(self):
        closed = socket.socket()
        closed.close()
        provider_cancel._shutdown(closed)


class _CancellableModel(FakeMessagesListChatModel):
    """Blocks inside the graph until the turn's scope is cancelled, as a provider read would."""

    entered: ClassVar[threading.Event] = threading.Event()

    def bind_tools(self, tools, **_kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        scope = provider_cancel._SCOPE.get()
        if scope is None:
            raise AssertionError("the turn scope did not reach the graph worker")
        type(self).entered.set()
        while not scope.cancelled:
            time.sleep(0.01)
        raise provider_cancel.ProviderCallCancelled


class RuntimeCancellationTests(unittest.TestCase):
    def test_a_cancelled_real_graph_leaves_a_checkpoint_the_next_turn_can_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.sqlite3"
            runtime = runtime_api._sqlite_runtime(path)
            turn = context(assistant("hello-pulse", action()), thread_id="team:cancel:thread")
            _CancellableModel.entered = threading.Event()
            cancelling = agent_runtime.AgentRuntime(
                runtime._checkpointer, model_factory=lambda _config: _CancellableModel(responses=[])
            )
            scope = provider_cancel.CancelScope()
            outcome = _run(scope, lambda: cancelling.start(turn, "Please wait"))
            self.assertTrue(_CancellableModel.entered.wait(5))
            scope.cancel()
            self.assertIsInstance(outcome.exception(5), provider_cancel.ProviderCallCancelled)
            runtime.close()

            reopened = runtime_api._sqlite_runtime(path)
            answering = agent_runtime.AgentRuntime(
                reopened._checkpointer,
                model_factory=lambda _config: ToolAwareFakeModel(responses=[AIMessage(content="Back.")]),
            )
            self.assertEqual(answering.start(turn, "Hello again").reply, "Back.")
            reopened.close()
            sqlite3.connect(path).execute("PRAGMA integrity_check").fetchone()


async def _asgi_turn(app, payload: dict[str, object], *, disconnect: asyncio.Event | None) -> dict[str, Any]:
    raw = json.dumps(payload).encode()
    # A trailing empty body message reaches only the disconnect watcher, which must keep waiting.
    messages = [
        {"type": "http.request", "body": raw, "more_body": False},
        {"type": "http.request", "body": b"", "more_body": False},
    ]
    sent: list[dict[str, Any]] = []

    async def receive():
        if messages:
            return messages.pop(0)
        await (disconnect or asyncio.Event()).wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/turns",
        "raw_path": b"/v1/turns",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", f"Bearer {TOKEN}".encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }
    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    content = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return {"status": start["status"], "body": json.loads(content)}


class _BlockingRuntime:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.calls = 0

    def start(self, _context, _message, _conversation=()):
        self.calls += 1
        if self.calls > 1:
            return agent_runtime.TurnResult(status="completed", reply="Next turn.")
        self.entered.set()
        scope = provider_cancel._SCOPE.get()
        while not scope.cancelled:
            time.sleep(0.01)
        raise provider_cancel.ProviderCallCancelled


class DisconnectTests(unittest.TestCase):
    def test_a_team_disconnect_cancels_only_that_turn_and_the_next_turn_starts(self):
        runtime = _BlockingRuntime()
        app = runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)

        async def scenario():
            disconnect = asyncio.Event()
            turn = asyncio.create_task(_asgi_turn(app, body(), disconnect=disconnect))
            self.assertTrue(await asyncio.to_thread(runtime.entered.wait, 5))
            disconnect.set()
            cancelled = await asyncio.wait_for(turn, 5)
            following = await asyncio.wait_for(_asgi_turn(app, body(), disconnect=None), 5)
            return cancelled, following

        cancelled, following = asyncio.run(scenario())
        self.assertEqual(cancelled, {"status": 409, "body": {"detail": "Chat turn cancelled"}})
        self.assertEqual(following["status"], 200)
        self.assertEqual(following["body"]["reply"], "Next turn.")


class PooledAnthropicTests(unittest.TestCase):
    def test_anthropic_turns_use_the_runtime_pool(self):
        with mock.patch.dict(os.environ, NO_PROXY_ENV):
            factory = agent_runtime.ProviderModelFactory()
        self.addCleanup(factory.close)
        model = factory(agent_runtime.ProviderConfig("anthropic", "claude-sonnet-5", "secret-test-key"))
        self.assertIs(model._client._client, factory._http_client)
        self.assertIsInstance(factory._http_client._transport, provider_cancel._Transport)
        self.assertIs(type(model), agent_runtime._pooled_chat_anthropic())


if __name__ == "__main__":
    unittest.main()

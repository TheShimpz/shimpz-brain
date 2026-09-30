"""The HTTP boundary bounds every request body in bytes and time before anything parses or authenticates it."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from typing import Any
from unittest import mock

import runtime_api
from test_runtime_api import TOKEN, FakeRuntime, body


class _Peer:
    """One scripted ASGI client: it sends body chunks, then stalls or disconnects, and records what it received."""

    def __init__(self, chunks: list[bytes], *, then: str = "wait") -> None:
        self.chunks = list(chunks)
        self.then = then
        self.received = 0
        self.sent: list[dict[str, Any]] = []

    async def receive(self) -> dict[str, Any]:
        if self.chunks:
            self.received += 1
            chunk = self.chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(self.chunks) or self.then != "wait"}
        if self.then == "disconnect":
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    def response(self) -> tuple[int | None, dict[str, Any] | None]:
        start = next((message for message in self.sent if message["type"] == "http.response.start"), None)
        if start is None:
            return None, None
        content = b"".join(message.get("body", b"") for message in self.sent if message["type"] == "http.response.body")
        return start["status"], json.loads(content)


def _scope(path: str = "/v1/turns", *, method: str = "POST", length: int | None = None) -> dict[str, Any]:
    headers = [(b"authorization", f"Bearer {TOKEN}".encode()), (b"content-type", b"application/json")]
    if length is not None:
        headers.append((b"content-length", str(length).encode()))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }


def _serve(peer: _Peer, scope: dict[str, Any], runtime: FakeRuntime | None = None) -> FakeRuntime:
    runtime = runtime or FakeRuntime()
    app = runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)
    asyncio.run(asyncio.wait_for(app(scope, peer.receive, peer.send), 5))
    return runtime


class RequestBodyBoundTests(unittest.TestCase):
    def test_the_ceiling_admits_the_largest_request_team_can_send(self):
        largest = 16 * (512 + 2 * 128) * 1024 + 64 * 512 * 1024
        self.assertGreater(runtime_api.MAX_REQUEST_BYTES, largest)
        self.assertLessEqual(runtime_api.MAX_REQUEST_BYTES, 64 * 1024 * 1024)

    def test_a_declared_oversized_body_is_refused_before_any_byte_is_read(self):
        peer = _Peer([b"x"])
        runtime = _serve(peer, _scope(length=runtime_api.MAX_REQUEST_BYTES + 1))
        self.assertEqual(peer.response(), (413, {"detail": "Request body is too large"}))
        self.assertIn((b"connection", b"close"), peer.sent[0]["headers"])
        self.assertEqual(peer.received, 0)
        self.assertEqual(runtime.calls, [])

    def test_a_streamed_oversized_body_is_refused_as_soon_as_it_passes_the_ceiling(self):
        peer = _Peer([b"x" * 512] * 100)
        with mock.patch.object(runtime_api, "MAX_REQUEST_BYTES", 1024):
            runtime = _serve(peer, _scope())
        self.assertEqual(peer.response(), (413, {"detail": "Request body is too large"}))
        self.assertEqual(peer.received, 3)
        self.assertEqual(runtime.calls, [])

    def test_a_trickled_body_is_refused_at_the_absolute_deadline(self):
        peer = _Peer([b"{"], then="stall")
        started = time.monotonic()
        with mock.patch.object(runtime_api, "REQUEST_BODY_SECONDS", 0.2):
            runtime = _serve(peer, _scope())
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(peer.response(), (408, {"detail": "Request body was not received in time"}))
        self.assertEqual(runtime.calls, [])

    def test_a_client_that_disconnects_mid_body_gets_no_response_and_no_turn(self):
        peer = _Peer([b'{"thread_id":'], then="disconnect")
        runtime = _serve(peer, _scope())
        self.assertEqual(peer.response(), (None, None))
        self.assertEqual(runtime.calls, [])

    def test_a_body_exactly_at_the_ceiling_in_chunks_reaches_the_turn(self):
        raw = json.dumps(body()).encode()
        padded = raw + b" " * (4096 - len(raw))
        peer = _Peer([padded[:1000], padded[1000:]])
        with mock.patch.object(runtime_api, "MAX_REQUEST_BYTES", 4096):
            runtime = _serve(peer, _scope(length=len(padded)))
        status, reply = peer.response()
        self.assertEqual((status, reply["reply"]), (200, "Hello."))
        self.assertEqual(runtime.calls[0][0], "start")


if __name__ == "__main__":
    unittest.main()

"""The HTTP boundary bounds every request body before it is parsed and keeps health off the worker threads."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from typing import Any
from unittest import mock

import anyio.to_thread
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


def _scope(
    path: str = "/v1/turns",
    *,
    method: str = "POST",
    length: int | None = None,
    authorization: bytes | None = f"Bearer {TOKEN}".encode(),
) -> dict[str, Any]:
    headers = [(b"content-type", b"application/json")]
    if authorization is not None:
        headers.append((b"authorization", authorization))
    if length is not None:
        headers.append((b"content-length", str(length).encode()))
    elif method == "POST":
        headers.append((b"transfer-encoding", b"chunked"))
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


async def _call(app, peer: _Peer, scope: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
    await asyncio.wait_for(app(scope, peer.receive, peer.send), 5)
    return peer.response()


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

    def test_an_unauthenticated_body_is_refused_before_any_byte_is_read_or_decoded(self):
        largest = runtime_api.MAX_REQUEST_BYTES
        for authorization, length in (
            (None, largest),
            (b"Bearer wrong", largest),
            (f"Bearer {TOKEN}x".encode(), None),
            ("Bearer \u00e9".encode("latin-1"), 2),
            (TOKEN.encode(), 2),
        ):
            peer = _Peer([b"{}"])
            with self.subTest(authorization=authorization, length=length):
                runtime = _serve(peer, _scope(length=length, authorization=authorization))
                self.assertEqual(peer.response(), (401, {"detail": "Unauthorized"}))
                self.assertEqual(peer.received, 0)
                self.assertEqual(runtime.calls, [])

    def test_an_unavailable_runtime_token_refuses_a_body_unread(self):
        def unavailable() -> str:
            raise runtime_api.HTTPException(status_code=503, detail="Brain runtime authentication is unavailable")

        peer = _Peer([b"{}"])
        app = runtime_api.create_app(runtime=FakeRuntime(), token_reader=unavailable)
        asyncio.run(_call(app, peer, _scope(length=2)))
        self.assertEqual(peer.response(), (503, {"detail": "Brain runtime authentication is unavailable"}))
        self.assertEqual(peer.received, 0)

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


class AdmissionTests(unittest.TestCase):
    CEILING = 4096

    def setUp(self) -> None:
        for name, value in (
            ("MAX_REQUEST_BYTES", self.CEILING),
            ("MAX_BUFFERED_REQUEST_BYTES", 2 * self.CEILING),
            ("REQUEST_BODY_SECONDS", 0.2),
        ):
            patcher = mock.patch.object(runtime_api, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _app(self, runtime: FakeRuntime):
        return runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)

    async def _saturate(self, app) -> list[tuple[int | None, dict[str, Any] | None]]:
        """Hold the whole budget with stalled bodies; return what excess and bodyless requests got meanwhile."""
        holders = [
            asyncio.create_task(_call(app, _Peer([], then="stall"), _scope(length=self.CEILING))),
            asyncio.create_task(_call(app, _Peer([], then="stall"), _scope())),
        ]
        await asyncio.sleep(0.05)
        excess = _Peer([b"{}"])
        refused = await _call(app, excess, _scope(length=2))
        self.assertEqual(excess.received, 0)
        self.assertIn((b"retry-after", b"1"), excess.sent[0]["headers"])
        health = await _call(app, _Peer([]), _scope("/health", method="GET"))
        return [refused, health, *await asyncio.gather(*holders)]

    def test_excess_bodies_are_refused_unread_and_every_outcome_releases_its_admission(self):
        runtime = FakeRuntime()
        app = self._app(runtime)
        raw = json.dumps(body()).encode()

        async def scenario() -> list[Any]:
            first = await self._saturate(app)
            # A completed turn, a body cut off by a disconnect, and a refused oversized stream each give back the
            # admission they held, so the whole budget is free again afterwards.
            completed = await _call(app, _Peer([raw]), _scope(length=len(raw)))
            await _call(app, _Peer([b"{"], then="disconnect"), _scope(length=10))
            oversized = await _call(app, _Peer([b"x" * self.CEILING, b"x"]), _scope())
            return [first, completed[0], oversized[0], await self._saturate(app)]

        first, completed, oversized, again = asyncio.run(scenario())
        refused = (503, {"detail": "Brain runtime request capacity reached"})
        healthy = (200, {"status": "ok", "runtime": "langgraph"})
        timed_out = (408, {"detail": "Request body was not received in time"})
        self.assertEqual(first, [refused, healthy, timed_out, timed_out])
        self.assertEqual((completed, oversized), (200, 413))
        self.assertEqual(again, first)
        self.assertEqual([call[0] for call in runtime.calls], ["start"])

    def test_a_failing_request_releases_its_admission(self):
        app = self._app(FakeRuntime(error=RuntimeError("boom")))
        raw = json.dumps(body()).encode()

        async def scenario() -> list[Any]:
            with self.assertRaises(RuntimeError):
                await _call(app, _Peer([raw]), _scope(length=len(raw)))
            return await self._saturate(app)

        statuses = [status for status, _detail in asyncio.run(scenario())]
        self.assertEqual(statuses, [503, 200, 408, 408])


class HealthTests(unittest.TestCase):
    def test_health_answers_while_every_worker_thread_is_busy(self):
        peer = _Peer([b""])
        app = runtime_api.create_app(runtime=FakeRuntime(), token_reader=lambda: TOKEN)

        async def saturated() -> None:
            limiter = anyio.to_thread.current_default_thread_limiter()
            borrowers = [object() for _ in range(int(limiter.total_tokens))]
            for borrower in borrowers:
                await limiter.acquire_on_behalf_of(borrower)
            try:
                await asyncio.wait_for(app(_scope("/health", method="GET"), peer.receive, peer.send), 1)
            finally:
                for borrower in borrowers:
                    limiter.release_on_behalf_of(borrower)

        asyncio.run(saturated())
        self.assertEqual(peer.response(), (200, {"status": "ok", "runtime": "langgraph"}))


if __name__ == "__main__":
    unittest.main()

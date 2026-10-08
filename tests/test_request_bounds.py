"""The HTTP boundary bounds every request body before it is parsed and keeps health off the worker threads."""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
import unittest
import weakref
from typing import Any
from unittest import mock

import agent_runtime
import anyio.to_thread
import context_budget
import intent_route
import memory as team_memory
import runtime_api
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from protocol.team.http.v1 import identifiers as team_identifiers
from test_agent_runtime import ToolAwareFakeModel
from test_runtime_api import TOKEN, FakeRuntime, body, client

import routine as team_routine

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
# The widest character a validated text field can hold: four UTF-8 bytes.
WIDE = "\U0001f600"
MiB = 1024 * 1024


class _Peer:
    """One scripted ASGI client: it sends body chunks, then stalls or disconnects, and records what it received.

    After its chunks, "wait" ends the body and waits, "stall" and "disconnect" leave the body unfinished, and "hangup"
    ends the body and then disconnects. A stalled peer sets `waiting` once the app waits for more of its body, and
    disconnects once `released` is set.
    """

    def __init__(self, chunks: list[bytes], *, then: str = "wait") -> None:
        self.chunks = list(chunks)
        self.then = then
        self.received = 0
        self.waiting = asyncio.Event()
        self.released = asyncio.Event()
        self.sent: list[dict[str, Any]] = []

    async def receive(self) -> dict[str, Any]:
        if self.chunks:
            self.received += 1
            chunk = self.chunks.pop(0)
            more = bool(self.chunks) or self.then not in {"wait", "hangup"}
            return {"type": "http.request", "body": chunk, "more_body": more}
        if self.then in {"disconnect", "hangup"}:
            return {"type": "http.disconnect"}
        self.waiting.set()
        await self.released.wait()
        return {"type": "http.disconnect"}

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


async def _call(
    app, peer: _Peer, scope: dict[str, Any], seconds: float = 5
) -> tuple[int | None, dict[str, Any] | None]:
    await asyncio.wait_for(app(scope, peer.receive, peer.send), seconds)
    return peer.response()


def _encoded(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def _skill(index: int) -> dict[str, object]:
    steps = [
        {
            "assistant_id": f"s{step:02d}-" + "a" * 36,
            "action": f"a{index}" + "b" * 126,
            "inputs": [f"i{name:02d}" + "x" * 61 for name in range(32)],
        }
        for step in range(16)
    ]
    contracts = {step["assistant_id"]: "sha256:" + "0" * 64 for step in steps}
    return {"key": team_memory._skill_key(contracts, steps), "contracts": contracts, "steps": steps, "usable": False}


def _largest_uncounted() -> dict[str, object]:
    """Every field of a resume that the model window does not count, each at the most its validation admits."""
    return {
        "provider": {"provider": "openai", "model": "gpt-6.1-sol", "api_key": "\x01" * 16 * 1024, "effort": "medium"},
        "memories": [
            {"topic": f"t{index:02d}" + "a" * 37, "preference": WIDE * team_memory.MAX_PREFERENCE_CHARS}
            for index in range(team_memory.MAX_MEMORIES)
        ],
        "skills": [_skill(index) for index in range(team_memory.MAX_SKILLS)],
        "routines": [
            {
                "routine_id": f"{index:032x}",
                "name": WIDE * team_routine.MAX_NAME_CHARS,
                "schedule": {"kind": "monthly", "day": 28, "time": "23:59"},
                "timezone": "/".join(letter * 32 for letter in "ABC"),
                "timezone_source": "browser",
                "revision": 2**31 - 1,
                "daily_steps": team_routine.team_routine_context.MAX_LISTED_DAILY_STEPS,
                "output": {"mode": "changes"},
                "steps": _listed_steps(team_routine.MAX_LISTING_STEPS_BYTES // team_routine.MAX_ROUTINES),
            }
            for index in range(team_routine.MAX_ROUTINES)
        ],
    }


def _listed_steps(size: int) -> list[dict[str, object]]:
    """One listed Routine's widest steps, as many as fit ``size`` encoded bytes: the listing bound counts bytes."""
    names = [f"{index:02d}" + "c" * (team_routine.team_routine_protocol.MAX_MEMBER_CHARS - 2) for index in range(64)]
    step = {
        "id": "s" + "x" * 31,
        "assistant": "a" * team_identifiers.MAX_ASSISTANT_ID_CHARS,
        "action": "b" * team_identifiers.MAX_ACTION_ID_CHARS,
        "inputs": names,
    }
    steps: list[dict[str, object]] = []
    while len(_encoded([*steps, step])) <= size:
        steps.append(step)
    return steps


def _serve(peer: _Peer, scope: dict[str, Any], runtime: FakeRuntime | None = None) -> FakeRuntime:
    runtime = runtime or FakeRuntime()
    app = runtime_api.create_app(runtime=runtime, token_reader=lambda: TOKEN)
    asyncio.run(asyncio.wait_for(app(scope, peer.receive, peer.send), 5))
    return runtime


class RequestBodyBoundTests(unittest.TestCase):
    def test_the_ceiling_admits_every_request_that_can_fit_the_model_window(self):
        tool = agent_runtime._tool_name("hello-pulse", "hello")
        call = {"name": tool, "args": {}, "id": "call"}
        model = ToolAwareFakeModel(responses=[AIMessage(content="", tool_calls=[call])])
        api = client(agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: model))
        interrupt = api.post("/v1/turns", json=body(), headers=HEADERS).json()["actions"][0]["interrupt_id"]

        # Action results that alone fill the smallest window, beside every uncounted field at its maximum. The fixed
        # prompt and the pending history make the real window guard refuse it, so no resume that can succeed is
        # larger; a start carries the conversation window instead of resumed knowledge and stays smaller still.
        window = (
            context_budget.MODEL_WINDOW_TOKENS - context_budget.OUTPUT_RESERVE_TOKENS
        ) * context_budget.BYTES_PER_TOKEN
        results = {interrupt: "x" * (window - len(_encoded({interrupt: ""})))}
        self.assertEqual(len(_encoded(results)), window)
        context = {key: value for key, value in body().items() if key not in {"message", "conversation", "locale"}}
        raw = _encoded({**context, **_largest_uncounted(), "results": results})
        conversation = _encoded(
            [{"role": "assistant", "text": WIDE * intent_route.MAX_CONVERSATION_TEXT_CHARS, "truncated": False}]
            * intent_route.MAX_CONVERSATION_ENTRIES
        )
        self.assertLessEqual(len(raw) + len(conversation), runtime_api.MAX_REQUEST_BYTES)

        refused = api.post("/v1/turns/resume", content=raw, headers={**HEADERS, "Content-Type": "application/json"})
        self.assertEqual(
            (refused.status_code, refused.json()), (400, {"detail": "conversation context exceeds the model window"})
        )

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
        with mock.patch.object(runtime_api, "REQUEST_BODY_SECONDS", 0.5):
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
            ("MEMORY_PER_REQUEST", 0),
            ("MEMORY_PER_BODY_BYTE", 1),
            ("MAX_REQUEST_MEMORY", 2 * self.CEILING),
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

    def test_a_turn_keeps_its_admission_after_a_disconnect_until_its_worker_returns(self):
        class BlockedRuntime(FakeRuntime):
            def __init__(self) -> None:
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()

            def start(self, context, message, conversation=()):
                self.entered.set()
                if not self.release.wait(5):
                    raise AssertionError("the test never released the turn")
                return super().start(context, message, conversation)

        runtime = BlockedRuntime()
        raw = json.dumps(body()).encode()
        # The turn alone fills the budget, and Team's disconnect cancels only its provider I/O.
        with mock.patch.object(runtime_api, "MAX_REQUEST_MEMORY", len(raw)):
            app = self._app(runtime)

        async def scenario() -> list[Any]:
            turn = asyncio.create_task(_call(app, _Peer([raw], then="hangup"), _scope(length=len(raw))))
            self.assertTrue(await asyncio.to_thread(runtime.entered.wait, 2))
            refused = await _call(app, _Peer([b"{}"]), _scope(length=2))
            runtime.release.set()
            await turn
            admitted = await _call(app, _Peer([b"{}"]), _scope(length=2))
            return [refused[0], admitted[0]]

        self.assertEqual(asyncio.run(scenario()), [503, 422])
        self.assertEqual([call[0] for call in runtime.calls], ["start"])

    def test_a_request_frees_its_cyclic_garbage_before_its_admission_returns(self):
        class Garbage:
            pass

        class LitteringRuntime(FakeRuntime):
            def start(self, context, message, conversation=()):
                garbage = Garbage()
                garbage.cycle = garbage
                self.left = weakref.ref(garbage)
                return super().start(context, message, conversation)

        runtime = LitteringRuntime()
        raw = json.dumps(body()).encode()
        enabled = gc.isenabled()
        gc.disable()
        try:
            status, _reply = asyncio.run(_call(self._app(runtime), _Peer([raw]), _scope(length=len(raw))))
        finally:
            if enabled:
                gc.enable()
        self.assertEqual(status, 200)
        self.assertIsNone(runtime.left())


class ProductionAdmissionTests(unittest.TestCase):
    """The production reservation: a fixed amount per request plus a multiple of each declared byte."""

    def setUp(self) -> None:
        self.app = runtime_api.create_app(runtime=FakeRuntime(), token_reader=lambda: TOKEN)

    async def _held(self, lengths: list[int]) -> tuple[list[Any], tuple[int | None, dict[str, Any] | None], int]:
        """Hold stalled bodies of these lengths, send one more, then disconnect them all.

        Returns what each holder and the extra request got, and how much of its body the extra one sent. A held body
        never reaches its deadline here, so the holders keep their reservations until the test releases them;
        AdmissionTests prove the deadline.
        """
        peers = [_Peer([], then="stall") for _ in lengths]
        excess = _Peer([b"{"], then="stall")
        calls = [
            asyncio.create_task(_call(self.app, peer, _scope(length=length), seconds=30))
            for peer, length in zip(peers, lengths, strict=True)
        ]

        async def settled(peer: _Peer, call: asyncio.Task) -> None:
            waiting = asyncio.create_task(peer.waiting.wait())
            await asyncio.wait({waiting, call}, return_when=asyncio.FIRST_COMPLETED)
            waiting.cancel()

        # Admission comes before the body is read: a request is admitted once it waits for its body, or refused.
        async with asyncio.timeout(5):
            await asyncio.gather(*(settled(peer, call) for peer, call in zip(peers, calls, strict=True)))
            calls.append(asyncio.create_task(_call(self.app, excess, _scope(length=1), seconds=30)))
            await settled(excess, calls[-1])
        for peer in (*peers, excess):
            peer.released.set()
        *held, extra = await asyncio.gather(*calls)
        return [status for status, _detail in held], extra, excess.received

    def test_one_largest_request_is_admitted_beside_exactly_what_the_budget_has_left(self):
        largest = runtime_api.MEMORY_PER_REQUEST + runtime_api.MAX_REQUEST_BYTES * runtime_api.MEMORY_PER_BODY_BYTE
        self.assertEqual(largest, 560 * MiB)
        remaining = (
            runtime_api.MAX_REQUEST_MEMORY - largest - runtime_api.MEMORY_PER_REQUEST
        ) // runtime_api.MEMORY_PER_BODY_BYTE
        self.assertEqual(remaining, 1_310_720)
        capacity = (503, {"detail": "Brain runtime request capacity reached"})

        statuses, refused, read = asyncio.run(self._held([runtime_api.MAX_REQUEST_BYTES, remaining]))
        self.assertEqual((statuses, refused, read), ([None, None], capacity, 0))
        # Disconnecting released everything, and one byte more than what was left is refused unread.
        statuses, refused, read = asyncio.run(self._held([runtime_api.MAX_REQUEST_BYTES, remaining + 1]))
        self.assertEqual((statuses, refused[0], read), ([None, 503], None, 1))

    def test_fifteen_small_requests_fill_the_budget(self):
        statuses, refused, read = asyncio.run(self._held([1] * 15))
        self.assertEqual((statuses, refused[0], read), ([None] * 15, 503, 0))

    def test_fourteen_small_requests_leave_room_for_one_more(self):
        statuses, refused, read = asyncio.run(self._held([1] * 14))
        self.assertEqual((statuses, refused[0], read), ([None] * 14, None, 1))


class ClosedInputTests(unittest.TestCase):
    def test_an_object_with_unknown_fields_is_refused_with_one_error(self):
        unknown = (400, {"loc": ["body"], "type": "value_error", "msg": "Value error, unknown field"})
        nested = body()
        nested["assistants"][1]["actions"][0] = {"id": "hello", "first": 1, "second": 2}
        for request, loc in (
            (body(**{f"field-{index}": index for index in range(10_000)}), ["body"]),
            (nested, ["body", "assistants", 1, "actions", 0]),
        ):
            with self.subTest(loc=loc):
                response = client(FakeRuntime()).post("/v1/turns", json=request, headers=HEADERS)
                self.assertEqual((response.status_code, response.json()["detail"]), (422, [{**unknown[1], "loc": loc}]))


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

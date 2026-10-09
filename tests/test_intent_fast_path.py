"""The Jev fast path: exact request, closed response, and fallback on every non-confident or failed outcome."""

import json
import os
import socket
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import agent_runtime
import httpx
import intent_fast_path
import intent_route
import provider_cancel

KEY = "tsk-test-0123456789abcdef"
PROBABILITIES = {"ordinary-task": 0.9, "assistant-install": 0.05, "assistant-uninstall": 0.03, "unresolved": 0.02}


def _answer(choice: str = "ordinary-task", confidence: float = 0.9, **overrides: object) -> dict[str, object]:
    answer = {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": dict(PROBABILITIES)}
    return {"model": intent_fast_path.MODEL, "answers": {"intent": answer}, "usage": {"input_tokens": 9}, **overrides}


class _CountingStream(httpx.SyncByteStream):
    """A large response body that records how much of it the fast path consumed and whether it was closed."""

    def __init__(self, chunks: int, size: int = 64 * 1024, pause: float = 0.0) -> None:
        self.chunks = chunks
        self.size = size
        self.pause = pause
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for _ in range(self.chunks):
            self.consumed += 1
            yield b" " * self.size
            time.sleep(self.pause)

    def close(self) -> None:
        self.closed = True


class _LateEnd(httpx.SyncByteStream):
    """A complete, valid answer whose stream ends only after a pause."""

    def __init__(self, pause: float) -> None:
        self.pause = pause

    def __iter__(self):
        yield json.dumps(_answer()).encode()
        time.sleep(self.pause)


class _PacedServer:
    """Sends response headers at once, one body byte after ``pause``, then nothing until the client leaves."""

    def __init__(self, pause: float) -> None:
        self.pause = pause
        self.requested = threading.Event()
        self.finished = threading.Event()
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        connection, _address = self._listener.accept()
        with connection:
            buffer = b""
            while b"\r\n\r\n" not in buffer:
                chunk = connection.recv(65536)
                if not chunk:
                    return
                buffer += chunk
            self.requested.set()
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 300\r\n\r\n")
            time.sleep(self.pause)
            connection.sendall(b"{")
            while connection.recv(65536):
                pass
            self.finished.set()

    def close(self) -> None:
        self._listener.close()


def _eventually(condition, seconds: float = 5) -> bool:
    limit = time.monotonic() + seconds
    while not condition() and time.monotonic() < limit:
        time.sleep(0.01)
    return condition()


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class RequestAndParseTests(unittest.TestCase):
    def test_request_carries_the_pinned_model_current_message_and_bounded_context(self):
        bare = intent_fast_path.request_body("Oi!", intent_route.LifecycleContext())
        self.assertEqual(bare["model"], "jev-1.13.0")
        self.assertEqual(bare["state"], {"current_message": "Oi!"})
        self.assertEqual(set(bare["questions"]["intent"]["criteria"]), set(intent_fast_path.INTENTS))
        context = intent_route.LifecycleContext(
            intent_route.LifecycleReference("shimpz-cloudflare", "Shimpz Cloudflare"),
            (intent_route.ConversationEntry("user", "Instala o Cloudflare", False),),
        )
        state = intent_fast_path.request_body("Remove ele", context)["state"]
        self.assertEqual(
            state,
            {
                "current_message": "Remove ele",
                "earlier_conversation": [{"role": "user", "text": "Instala o Cloudflare"}],
                "last_lifecycle_assistant": "Shimpz Cloudflare",
            },
        )

    def test_only_the_exact_choice_shape_parses(self):
        self.assertEqual(intent_fast_path.parse(_answer()), ("ordinary-task", 0.9))
        self.assertEqual(
            intent_fast_path.parse({k: v for k, v in _answer().items() if k != "usage"})[0], "ordinary-task"
        )
        broken = []
        for mutate in (
            lambda value: value.update(model="jev-latest"),
            lambda value: value.update(extra=1),
            lambda value: value.pop("answers"),
            lambda value: value["answers"].update(other={}),
            lambda value: value["answers"]["intent"].update(type="score"),
            lambda value: value["answers"]["intent"].update(choice="delete-everything"),
            lambda value: value["answers"]["intent"].update(confidence=True),
            lambda value: value["answers"]["intent"].update(confidence=1.5),
            lambda value: value["answers"]["intent"].update(confidence=float("nan")),
            lambda value: value["answers"]["intent"]["probabilities"].pop("unresolved"),
            lambda value: value["answers"]["intent"]["probabilities"].update({"ordinary-task": -0.1}),
            lambda value: value["answers"]["intent"].update(extra=1),
            # ordinary-task claimed while the distribution puts it at zero.
            lambda value: value["answers"]["intent"].update(
                probabilities={
                    "ordinary-task": 0.0,
                    "assistant-install": 1.0,
                    "assistant-uninstall": 0.0,
                    "unresolved": 0.0,
                }
            ),
            lambda value: value["answers"]["intent"].update(
                probabilities={
                    "ordinary-task": 0.4,
                    "assistant-install": 0.1,
                    "assistant-uninstall": 0.1,
                    "unresolved": 0.1,
                }
            ),
        ):
            value = _answer()
            mutate(value)
            broken.append(value)
        broken.extend((None, [], {"model": intent_fast_path.MODEL, "answers": []}))
        for value in broken:
            with self.subTest(value=str(value)[:80]), self.assertRaises(intent_fast_path.FastPathResponseError):
                intent_fast_path.parse(value)


class ConfidentOrdinaryTests(unittest.TestCase):
    def _ask(self, handler) -> bool:
        with _client(handler) as client:
            return intent_fast_path.confident_ordinary(client, KEY, "Oi!", intent_route.LifecycleContext())

    def test_only_a_confident_ordinary_task_takes_the_fast_path(self):
        seen: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_answer())

        self.assertTrue(self._ask(respond))
        (request,) = seen
        self.assertEqual(str(request.url), intent_fast_path.ENDPOINT)
        self.assertEqual(request.headers["Authorization"], f"Bearer {KEY}")
        self.assertEqual(json.loads(request.content)["state"], {"current_message": "Oi!"})
        # Every wait of the exchange is bounded by what remains of its deadline.
        self.assertTrue(0 < request.extensions["timeout"]["read"] <= intent_fast_path.TIMEOUT_SECONDS)
        self.assertTrue(0 < request.extensions["timeout"]["connect"] <= intent_fast_path.TIMEOUT_SECONDS)
        self.assertEqual(request.headers["Accept-Encoding"], "identity")
        for response in (
            _answer(confidence=intent_fast_path.CONFIDENCE_THRESHOLD - 0.01),
            _answer(choice="assistant-install", confidence=1.0),
            _answer(choice="assistant-uninstall", confidence=1.0),
            _answer(choice="unresolved", confidence=1.0),
        ):
            with self.subTest(answer=response["answers"]["intent"]):
                self.assertFalse(self._ask(lambda _request, body=response: httpx.Response(200, json=body)))

    def test_every_failure_keeps_the_llm_route(self):
        def timeout(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow")

        for handler in (
            lambda _request: httpx.Response(401, json={"error": "invalid key"}),
            lambda _request: httpx.Response(529, json={"error": "overloaded"}),
            lambda _request: httpx.Response(200, content=b"not json"),
            lambda _request: httpx.Response(200, json=_answer(model="jev-latest")),
            timeout,
        ):
            with self.subTest(handler=handler):
                self.assertFalse(self._ask(handler))

    def test_an_error_or_oversized_response_is_closed_without_reading_it_whole(self):
        # 4 MiB bodies: an error status and a declared oversized length are never read, and an undeclared oversized
        # body stops at the first chunk past the bound. Every rejected response is closed.
        cases = (
            (503, {}, 0),
            (200, {"Content-Length": str(4 * 1024 * 1024)}, 0),
            (200, {"Content-Encoding": "gzip"}, 0),
            (200, {}, 1),
        )
        for status, headers, consumed in cases:
            stream = _CountingStream(64)
            with self.subTest(status=status, headers=headers):
                self.assertFalse(
                    self._ask(lambda _request, s=stream, h=headers, c=status: httpx.Response(c, headers=h, stream=s))
                )
                self.assertEqual(stream.consumed, consumed)
                self.assertTrue(stream.closed)

    def test_a_trickled_response_is_abandoned_at_the_deadline(self):
        stream = _CountingStream(64, size=1, pause=0.05)
        with mock.patch.object(intent_fast_path, "TIMEOUT_SECONDS", 0.1):
            self.assertFalse(self._ask(lambda _request: httpx.Response(200, stream=stream)))
        # The abandoned exchange stops reading at its own deadline check and closes the response.
        self.assertTrue(_eventually(lambda: stream.closed))
        self.assertLess(stream.consumed, 10)

    def test_a_valid_answer_whose_stream_ends_after_the_deadline_is_refused(self):
        late = httpx.Response(200, stream=_LateEnd(0.1))
        self.assertIsNone(intent_fast_path._bounded_body(late, time.monotonic() + 0.03))
        on_time = httpx.Response(200, stream=_LateEnd(0))
        self.assertEqual(json.loads(intent_fast_path._bounded_body(on_time, time.monotonic() + 1)), _answer())

    def test_an_exchange_that_starts_after_its_deadline_never_sends(self):
        with _client(lambda _request: self.fail("a late exchange must not send")) as client:
            self.assertIsNone(
                intent_fast_path._exchange(client, KEY, "Oi!", intent_route.LifecycleContext(), time.monotonic())
            )

    def test_a_delayed_resolver_cannot_hold_the_caller_past_the_deadline(self):
        server = _PacedServer(pause=0)
        self.addCleanup(server.close)
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "", "https_proxy": ""}):
            client = provider_cancel.client()
        self.addCleanup(client.close)
        resolve = socket.getaddrinfo

        def slow(host, *args, **kwargs):
            # Name resolution blocks in the C library, where no socket shutdown can reach it.
            if host == "jev.test":
                time.sleep(0.3)
                return resolve("127.0.0.1", *args, **kwargs)
            return resolve(host, *args, **kwargs)

        started = time.monotonic()
        with (
            mock.patch.object(socket, "getaddrinfo", slow),
            mock.patch.object(intent_fast_path, "ENDPOINT", f"http://jev.test:{server.port}/"),
            mock.patch.object(intent_fast_path, "TIMEOUT_SECONDS", 0.05),
        ):
            self.assertFalse(intent_fast_path.confident_ordinary(client, KEY, "Oi!", intent_route.LifecycleContext()))
            self.assertLess(time.monotonic() - started, 0.2)
            # The late work resolves, finds its scope cancelled, and never sends the request.
            time.sleep(0.5)
        self.assertFalse(server.requested.is_set())

    def test_the_deadline_wakes_a_read_blocked_on_the_runtime_pool(self):
        # The one body byte arrives inside both the read timeout and the deadline; without the scope cancellation the
        # abandoned exchange's next read would hold its socket a whole read timeout past the deadline.
        server = _PacedServer(pause=0.3)
        self.addCleanup(server.close)
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "", "https_proxy": ""}):
            client = provider_cancel.client()
        self.addCleanup(client.close)
        started = time.monotonic()
        with (
            mock.patch.object(intent_fast_path, "ENDPOINT", f"http://127.0.0.1:{server.port}/"),
            mock.patch.object(intent_fast_path, "TIMEOUT_SECONDS", 0.4),
        ):
            self.assertFalse(intent_fast_path.confident_ordinary(client, KEY, "Oi!", intent_route.LifecycleContext()))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(server.finished.wait(5))
        self.assertLess(time.monotonic() - started, 0.6)

    def test_stalled_workers_send_further_calls_to_the_llm_route_without_queueing_them(self):
        executor = ThreadPoolExecutor(max_workers=intent_fast_path.EXCHANGE_WORKERS)
        self.addCleanup(executor.shutdown)
        stall = threading.Event()
        self.addCleanup(stall.set)
        resolved: list[str] = []

        def resolver(request: httpx.Request) -> httpx.Response:
            stall.wait(5)
            resolved.append(request.headers["Authorization"])
            return httpx.Response(200, json=_answer())

        client = _client(resolver)
        self.addCleanup(client.close)

        def ask() -> bool:
            return intent_fast_path.confident_ordinary(client, KEY, "Oi!", intent_route.LifecycleContext())

        with (
            mock.patch.object(intent_fast_path, "_EXCHANGES", executor),
            mock.patch.object(
                intent_fast_path, "_OUTSTANDING", threading.BoundedSemaphore(intent_fast_path.EXCHANGE_WORKERS)
            ),
        ):
            with mock.patch.object(intent_fast_path, "TIMEOUT_SECONDS", 0.05):
                for _ in range(intent_fast_path.EXCHANGE_WORKERS):
                    self.assertFalse(ask())
            # Every worker is still stalled inside its resolver call, so twenty more calls return at once without
            # waiting out their deadline and without leaving a work item, with its key and context, in the queue.
            started = time.monotonic()
            for _ in range(20):
                self.assertFalse(ask())
            self.assertLess(time.monotonic() - started, intent_fast_path.TIMEOUT_SECONDS / 3)
            self.assertEqual(executor._work_queue.qsize(), 0)
            stall.set()
            self.assertTrue(_eventually(lambda: len(resolved) == intent_fast_path.EXCHANGE_WORKERS))
            # Once the stalled exchanges drain, their slots admit a new exchange again.
            self.assertTrue(_eventually(ask))


class RuntimeFastPathTests(unittest.TestCase):
    provider = agent_runtime.ProviderConfig("openai", "gpt-6.1-sol", "sk-test-0123456789")

    def _runtime(self) -> tuple[agent_runtime.AgentRuntime, mock.Mock]:
        factory = mock.Mock()
        factory.decision.side_effect = AssertionError("the LLM route must not run")
        return agent_runtime.AgentRuntime(object(), model_factory=factory), factory

    def test_a_confident_ordinary_task_skips_the_llm_route(self):
        runtime, factory = self._runtime()
        with mock.patch.object(intent_fast_path, "confident_ordinary", return_value=True) as ask:
            route = runtime.intent_route(self.provider, "Oi!", None, (), None, "pt", decision_key=KEY)
            runtime.intent_route(self.provider, "Tudo certo?", None, (), None, "pt", decision_key=KEY)
        self.assertEqual(route, intent_route.IntentRoute("ordinary-task"))
        self.assertEqual(ask.call_args_list[0].args[1:3], (KEY, "Oi!"))
        # Both classifications reuse one pooled decision client.
        self.assertIs(ask.call_args_list[0].args[0], ask.call_args_list[1].args[0])
        factory.decision.assert_not_called()
        runtime.close()
        self.assertTrue(runtime._decision_client.is_closed)

    def test_any_other_fast_path_outcome_runs_the_unchanged_llm_route(self):
        runtime, _factory = self._runtime()
        llm = intent_route.IntentRoute("assistant-install", "cloudflare")
        with (
            mock.patch.object(intent_fast_path, "confident_ordinary", return_value=False),
            mock.patch.object(intent_route, "create", return_value=llm) as create,
        ):
            route = runtime.intent_route(self.provider, "Instala o Cloudflare", None, (), None, "pt", KEY)
            self.assertEqual(route, llm)
        create.assert_called_once()
        with mock.patch.object(intent_route, "create", return_value=llm) as create:
            runtime.intent_route(self.provider, "Instala o Cloudflare", None, (), None, "pt")
        create.assert_called_once()

    def test_selection_and_invalid_input_never_reach_the_fast_path(self):
        runtime, _factory = self._runtime()
        with mock.patch.object(intent_fast_path, "confident_ordinary") as ask:
            with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "only to classification"):
                runtime.intent_route(self.provider, "cloudflare", "assistant-install", (), None, "pt", decision_key=KEY)
            with self.assertRaises(agent_runtime.RuntimeContractError):
                runtime.intent_route(self.provider, "   ", None, (), None, "pt", decision_key=KEY)
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()

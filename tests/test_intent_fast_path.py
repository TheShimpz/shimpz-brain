"""The Jev fast path: exact request, closed response, and fallback on every non-confident or failed outcome."""

from __future__ import annotations

import json
import time
import unittest
from unittest import mock

import agent_runtime
import httpx
import intent_fast_path
import intent_route

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
        self.assertEqual(request.extensions["timeout"]["read"], intent_fast_path.TIMEOUT_SECONDS)
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
        self.assertLess(stream.consumed, 10)
        self.assertTrue(stream.closed)


class RuntimeFastPathTests(unittest.TestCase):
    provider = agent_runtime.ProviderConfig("openai", "gpt-6.1-sol", "sk-test-0123456789")

    def _runtime(self) -> tuple[agent_runtime.AgentRuntime, mock.Mock]:
        factory = mock.Mock()
        factory.decision.side_effect = AssertionError("the LLM route must not run")
        return agent_runtime.AgentRuntime(object(), model_factory=factory), factory

    def test_a_confident_ordinary_task_skips_the_llm_route(self):
        runtime, factory = self._runtime()
        with mock.patch.object(intent_fast_path, "confident_ordinary", return_value=True) as ask:
            route = runtime.intent_route(self.provider, "Oi!", None, (), None, decision_key=KEY)
            runtime.intent_route(self.provider, "Tudo certo?", None, (), None, decision_key=KEY)
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
            self.assertEqual(runtime.intent_route(self.provider, "Instala o Cloudflare", None, (), None, KEY), llm)
        create.assert_called_once()
        with mock.patch.object(intent_route, "create", return_value=llm) as create:
            runtime.intent_route(self.provider, "Instala o Cloudflare", None, (), None)
        create.assert_called_once()

    def test_selection_and_invalid_input_never_reach_the_fast_path(self):
        runtime, _factory = self._runtime()
        with mock.patch.object(intent_fast_path, "confident_ordinary") as ask:
            with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "only to classification"):
                runtime.intent_route(self.provider, "cloudflare", "assistant-install", (), None, decision_key=KEY)
            with self.assertRaises(agent_runtime.RuntimeContractError):
                runtime.intent_route(self.provider, "   ", None, (), None, decision_key=KEY)
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()

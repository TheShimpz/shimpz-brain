"""Every compared arm pair of the 2026-10-09 Brain experiments sends a different provider request body (ADR-0094).

Each arm runs real chat turns over an ``httpx.MockTransport`` under the evaluation ceiling, so the bodies compared are
exactly what the arm would send: no network, key, or provider call is used. A pair whose bodies were equal would have
measured one configuration twice.
"""

import contextvars
import functools
import json
import unittest
from unittest import mock

import agent_runtime
import context_budget
import httpx
import provider_client
import turn_prompt
from eval.ceiling import Ceiling
from eval.injection_cases import ASSISTANTS
from langgraph.checkpoint.memory import InMemorySaver

from perf import prompt_arms, prompt_cache

CONFIG = agent_runtime.ProviderConfig("openai", "gpt-6-luna", "test-key", "low")
REPLY = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-6-luna",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "You have no open tasks.", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def _context(memories: tuple = ()) -> agent_runtime.TurnContext:
    return agent_runtime.TurnContext(
        "arms", "Eval Team", (ASSISTANTS["tasks"],), CONFIG, memories=memories, routines=(), routine_capacity=20_000
    )


def _bodies(limit: int, install, turn) -> list[dict]:
    """The request bodies of ``turn`` run under ``install``; every patch is undone afterwards."""
    bodies: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        # A distinct response and message id per request, as a provider sends: the history keeps every reply.
        number = len(bodies)
        output = [REPLY["output"][0] | {"id": f"msg_{number}"}]
        return httpx.Response(200, json=REPLY | {"id": f"resp_{number}", "output": output}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(answer))
    ceiling = Ceiling(100.0, limit)
    with (
        mock.patch.object(turn_prompt, "system_prompt", turn_prompt.system_prompt),
        mock.patch.object(context_budget, "history_to_drop", context_budget.history_to_drop),
        mock.patch.object(agent_runtime, "_prompt_caching", agent_runtime._prompt_caching),
    ):
        ceiling.install()
        try:
            install()
            real = functools.partial(provider_client.provider_model, http_client=client)
            runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=prompt_cache.Factory(real))
            contextvars.copy_context().run(turn, runtime)
        finally:
            ceiling.uninstall()
            client.close()
    return bodies


def _arm(arm: str) -> list[dict]:
    return _bodies(
        prompt_arms.output_limit(arm),
        lambda: prompt_arms.install(arm),
        lambda runtime: runtime.start(_context(), "List my open tasks."),
    )


def _text(bodies: list[dict]) -> str:
    return json.dumps(bodies, sort_keys=True)


def _system(body: dict) -> str:
    """The system prompt of a Responses body: its leading system message."""
    first, *_ = body["input"]
    if first.get("role") != "system":
        raise AssertionError("the body does not lead with the system prompt")
    return _text([first["content"]])


def _conversation(body: dict) -> str:
    return _text(body["input"][1:])


class PromptArmTests(unittest.TestCase):
    def test_each_sentence_pair_differs_exactly_by_the_untrusted_result_sentence(self):
        for baseline, candidate in (("pre", "prod"), ("base-pre", "base")):
            with self.subTest(baseline=baseline, candidate=candidate):
                without, shipped = _arm(baseline), _arm(candidate)
                self.assertNotEqual(without, shipped)
                self.assertNotIn(prompt_arms.SENTENCE, _text(without))
                self.assertIn(prompt_arms.ANCHOR + prompt_arms.SENTENCE, _text(shipped))
                self.assertEqual(_text(shipped).replace(prompt_arms.SENTENCE, ""), _text(without))

    def test_the_output_limit_pair_differs_exactly_by_the_limit(self):
        production, limited = _arm("base"), _arm("prod")
        self.assertNotEqual(production, limited)
        self.assertEqual({body["max_output_tokens"] for body in production}, {prompt_arms.MODEL_MAXIMUM})
        self.assertEqual({body["max_output_tokens"] for body in limited}, {prompt_arms.EVALUATION_LIMIT})
        unlimited = [body | {"max_output_tokens": 0} for body in production]
        self.assertEqual(unlimited, [body | {"max_output_tokens": 0} for body in limited])

    def test_the_per_attempt_arm_follows_the_attempt(self):
        arm = contextvars.ContextVar("arm", default="pre")

        def turn(name: str):
            def run(runtime):
                arm.set(name)
                runtime.start(_context(), "List my open tasks.")

            return run

        def install():
            prompt_arms.install_per_attempt(arm)

        limit = prompt_arms.EVALUATION_LIMIT
        self.assertEqual(_bodies(limit, install, turn("pre")), _arm("pre"))
        self.assertEqual(_bodies(limit, install, turn("prod")), _arm("prod"))

    def test_an_arm_refuses_a_prompt_that_lacks_the_sentence(self):
        for transform in (prompt_arms.shipped, prompt_arms.without_sentence):
            with self.subTest(transform=transform.__name__), self.assertRaisesRegex(RuntimeError, "quoted data"):
                transform(prompt_arms.ANCHOR)


class PromptCacheArmTests(unittest.TestCase):
    def _session(self, arm: str, stratum: str) -> list[dict]:
        session = prompt_cache.Session(stratum, arm, 0, 3, "arms")

        def turn(runtime):
            prompt_cache.ARM.set(arm)
            context = _context(prompt_cache.MEMORIES if stratum == "mem" else ())
            if stratum == "sat":
                # Seeding fills the window exactly; the turn after the first real one is the first to drop.
                prompt_cache.seed_history(runtime, session, context)
                runtime.start(context, "List my open tasks.")
            runtime.start(context, "List my open tasks.")

        return _bodies(prompt_arms.EVALUATION_LIMIT, prompt_cache.install, turn)

    def test_the_stable_prefix_arm_moves_the_volatile_sections_into_the_message(self):
        (base,), (stable,) = self._session("base", "mem"), self._session("stable", "mem")
        memory = prompt_cache.MEMORIES[-1].preference
        self.assertIn(memory, _system(base))
        self.assertNotIn(memory, _system(stable))
        self.assertIn(prompt_cache.DATE, _system(base))
        self.assertNotIn(prompt_cache.DATE, _system(stable))
        self.assertNotIn("Turn data for this message", _text([base]))
        self.assertIn("Turn data for this message", _conversation(stable))
        self.assertIn(memory, _conversation(stable))

    def test_the_hysteresis_arm_trims_a_saturated_history_further(self):
        # Seeding answers with the fake model, so each arm sends one provider request per real turn.
        (_, base), (_, trimmed) = self._session("base", "sat"), self._session("hyst", "sat")
        notes = [_conversation(body).count("Note ") for body in (base, trimmed)]
        self.assertEqual(notes, [context_budget.MAX_HISTORY_EXCHANGES - 1, prompt_cache.LOW_EXCHANGES - 1])
        self.assertEqual(_system(base), _system(trimmed))


if __name__ == "__main__":
    unittest.main()

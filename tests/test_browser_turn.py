"""The chat agent's Browser call: offered only where admitted, alone, followed only by the reply, then forgotten."""

import dataclasses
import json
import unittest
from types import SimpleNamespace

import agent_runtime
import browser_tool
import turn_prompt
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from runtime_errors import RuntimeContractError
from test_agent_runtime import ToolAwareFakeModel, context
from tool_fake import FixedFactory

OBJECTIVE = "Veja o prazo do edital no site da prefeitura."
RESULT = {"status": "done", "summary": "Prazo: 30/11.", "files": []}


def _call(name="browser", args=None, call_id="call-1"):
    return {
        "name": name,
        "args": {"objective": OBJECTIVE} if args is None else args,
        "id": call_id,
        "type": "tool_call",
    }


def _offered(**changes):
    return dataclasses.replace(context(thread_id="team:browser:thread"), browser=True, **changes)


class BrowserTurnTests(unittest.TestCase):
    def runtime(self, *responses):
        self.model = ToolAwareFakeModel(responses=list(responses))
        return agent_runtime.AgentRuntime(InMemorySaver(), model_factory=FixedFactory(lambda _config: self.model))

    def test_a_browser_call_suspends_alone_and_the_resume_only_replies(self):
        runtime = self.runtime(AIMessage(content="", tool_calls=[_call()]), AIMessage(content="O prazo é 30/11."))
        turn = runtime.start(_offered(), "Qual o prazo do edital?")
        self.assertEqual(turn.status, "browser-required")
        self.assertEqual(turn.browser.objective, OBJECTIVE)
        self.assertIn(browser_tool.TOOL_NAME, ToolAwareFakeModel.bound_tools)
        resumed = runtime.resume(_offered(), {turn.browser.interrupt_id: RESULT})
        self.assertEqual((resumed.status, resumed.reply), ("completed", "O prazo é 30/11."))
        self.assertIn(turn_prompt.BROWSER_SECTION, turn_prompt.system_prompt(_offered()))
        self.assertNotIn(turn_prompt.BROWSER_SECTION, turn_prompt.system_prompt(context()))

    def test_a_tool_call_after_the_browser_result_is_refused_until_the_model_replies(self):
        runtime = self.runtime(
            AIMessage(content="", tool_calls=[_call()]),
            AIMessage(content="", tool_calls=[_call(call_id="call-2")]),
            AIMessage(content="Pronto."),
        )
        turn = runtime.start(_offered(), "Pesquise na web.")
        resumed = runtime.resume(_offered(), {turn.browser.interrupt_id: RESULT})
        # The second call was answered with the closed correction, never run, and the model then replied.
        self.assertEqual((resumed.status, resumed.reply), ("completed", "Pronto."))

    def test_a_browser_call_beside_another_tool_is_refused_and_retried_alone(self):
        runtime = self.runtime(
            AIMessage(content="", tool_calls=[_call(), _call(name="ask_clarification", args={}, call_id="call-x")]),
            AIMessage(content="", tool_calls=[_call(call_id="call-3")]),
        )
        turn = runtime.start(_offered(), "Pesquise na web.")
        self.assertEqual(turn.status, "browser-required")

    def test_the_browser_is_never_offered_with_attachments_or_routine_work(self):
        offered = {"browser": True, "attachments": (), "routine_mode": False, "routine_settings_answer": False}
        self.assertTrue(browser_tool.offered(SimpleNamespace(**offered)))
        for change in (
            {"browser": False},
            {"attachments": ("file",)},
            {"routine_mode": True},
            {"routine_settings_answer": True},
        ):
            with self.subTest(change=change):
                self.assertFalse(browser_tool.offered(SimpleNamespace(**{**offered, **change})))

    def test_an_objective_outside_its_bounds_gets_a_correction_not_a_suspension(self):
        runtime = self.runtime(
            AIMessage(content="", tool_calls=[_call(args={"objective": "   "})]),
            AIMessage(content="Preciso de mais detalhes."),
        )
        turn = runtime.start(_offered(), "Pesquise.")
        self.assertEqual((turn.status, turn.reply), ("completed", "Preciso de mais detalhes."))

    def test_a_malformed_browser_suspension_is_a_contract_error(self):
        interrupt = type("Interrupt", (), {})
        good = interrupt()
        good.id, good.value = "i-1", {"kind": "browser", "objective": OBJECTIVE}
        self.assertEqual(agent_runtime._pending_result([good]).browser.interrupt_id, "i-1")
        bad = interrupt()
        bad.id, bad.value = "i-2", {"kind": "browser", "objective": "  padded "}
        action = interrupt()
        action.id, action.value = (
            "i-3",
            {"kind": "action", "assistant_id": "hello-pulse", "action": "hello", "input": {}},
        )
        for pending in ([bad], [good, action]):
            with self.subTest(pending=len(pending)), self.assertRaises(RuntimeContractError):
                agent_runtime._pending_result(pending)

    def test_a_browser_exchange_is_forgotten_by_its_call_and_its_result(self):
        self.assertTrue(browser_tool.read(ToolMessage(content="{}", tool_call_id="c", name="browser")))
        self.assertTrue(browser_tool.read(AIMessage(content="", tool_calls=[_call()])))
        self.assertFalse(browser_tool.read(AIMessage(content="hi")))
        self.assertFalse(browser_tool.read(ToolMessage(content="{}", tool_call_id="c", name="hello")))
        self.assertFalse(browser_tool.read(HumanMessage(content="hi")))
        self.assertTrue(agent_runtime._forgotten(ToolMessage(content="{}", tool_call_id="c", name="browser")))
        writes = [("task", "__interrupt__", [type("I", (), {"value": {"kind": "browser"}})()])]
        self.assertTrue(browser_tool.answered(writes))
        self.assertTrue(
            browser_tool.answered([("task", "__interrupt__", type("I", (), {"value": {"kind": "browser"}})())])
        )
        self.assertFalse(browser_tool.answered([("task", "messages", [])]))
        self.assertFalse(browser_tool.answered(None))

    def test_a_browser_step_runs_on_the_providers_fixed_browser_model(self):
        answer = json.dumps(
            {
                "kind": "wait",
                "url": None,
                "ref": None,
                "refs": None,
                "text": None,
                "direction": None,
                "summary": None,
                "reason": None,
                "sentence": None,
                "note": None,
            }
        )
        from structured_fake import StructuredFakeModel

        seen = []

        def factory(config):
            seen.append(config.model)
            return StructuredFakeModel(responses=[AIMessage(content=answer)])

        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=FixedFactory(factory))
        request = {
            "objective": OBJECTIVE,
            "language": "pt-BR",
            "thread": {"folded": [], "folded_omitted": 0, "notes": [], "notes_omitted": 0, "steps": []},
            "observation": "url: x",
            "screenshot": None,
        }
        provider = context().provider
        self.assertEqual(
            runtime.browser_call(provider, "browser-step", request),
            ({"kind": "wait"}, "gpt-6-luna" if provider.provider == "openai" else "claude-sonnet-5-5"),
        )
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()

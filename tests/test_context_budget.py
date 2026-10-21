"""Whole-exchange conversation memory and the model-window guard, over the production SQLite checkpoint."""

import functools
import json
import sqlite3
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import agent_runtime
import clarification
import context_budget
import httpx
import provider_client
import runtime_api
import tool_fake
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from protocol.team.action.v1 import schema as action_protocol
from tool_fake import ACTION, PINGER, FixedFactory


class RecordingModel(tool_fake.RecordingModel):
    seen: ClassVar[list[list[Any]]] = []


TOOL = agent_runtime.tool_name("pinger", "ping")


def _context(*assistants: agent_runtime.AssistantDefinition) -> agent_runtime.TurnContext:
    return agent_runtime.TurnContext(
        "budget-thread",
        "Budget Team",
        assistants or (PINGER,),
        agent_runtime.ProviderConfig("openai", "gpt-6.1-sol", "secret-test-key"),
    )


def _call(host: str, call_id: str) -> dict:
    return {"name": TOOL, "args": {"host": host}, "id": call_id, "type": "tool_call"}


def _user_texts(messages: Sequence[Any]) -> list[str]:
    return [str(message.content) for message in messages if isinstance(message, HumanMessage)]


def _openai_body(text_or_tool: str, arguments: dict | None, *, incomplete: bool) -> dict:
    """An OpenAI Responses body holding one text reply, or one function call when arguments are given."""
    status = "incomplete" if incomplete else "completed"
    if arguments is None:
        item = {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": status,
            "content": [{"type": "output_text", "text": text_or_tool, "annotations": []}],
        }
    else:
        item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": text_or_tool,
            "arguments": json.dumps(arguments),
            "status": status,
        }
    body = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6.1-sol",
        "status": status,
        "output": [item],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if incomplete:
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return body


def _anthropic_body(text_or_tool: str, arguments: dict | None, *, incomplete: bool) -> dict:
    """An Anthropic Messages body holding one text reply, or one tool use when arguments are given."""
    if arguments is None:
        block = {"type": "text", "text": text_or_tool}
    else:
        block = {"type": "tool_use", "id": "toolu_1", "name": text_or_tool, "input": arguments}
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [block],
        "stop_reason": "max_tokens" if incomplete else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class ExchangeTests(unittest.TestCase):
    def test_history_splits_at_user_messages_and_needs_ids(self):
        history = [
            HumanMessage("a", id="h1"),
            AIMessage("", tool_calls=[_call("x", "c1")], id="a1"),
            ToolMessage("ok", tool_call_id="c1", id="t1"),
            AIMessage("done", id="a2"),
            HumanMessage("b", id="h2"),
            AIMessage("reply", id="a3"),
        ]
        self.assertEqual([len(group) for group in context_budget.exchanges(history)], [4, 2])
        with self.assertRaises(context_budget.ContextStateError):
            context_budget.exchanges([HumanMessage("a")])
        with self.assertRaises(context_budget.ContextStateError):
            context_budget.exchanges([AIMessage("orphan", id="a1")])

    def test_a_bridge_is_one_exchange_with_the_turn_it_precedes(self):
        bridge = f"{context_budget.BRIDGE_ID_PREFIX}1"
        history = [
            HumanMessage("quoted", id=bridge),
            HumanMessage("a", id="h1"),
            AIMessage("r1", id="a1"),
            HumanMessage("b", id="h2"),
            AIMessage("r2", id="a2"),
        ]
        self.assertEqual([len(group) for group in context_budget.exchanges(history)], [3, 2])
        self.assertEqual(context_budget.history_to_drop(history, 0, 1), ())
        with mock.patch.object(context_budget, "MAX_HISTORY_EXCHANGES", 1):
            dropped = context_budget.history_to_drop(history, 0, 1)
        self.assertEqual([message.id for message in dropped], [bridge, "h1", "a1"])
        # A failed bridged turn, or one that read attachments, is forgotten whole, its bridge included.
        failed = [HumanMessage("quoted", id=bridge), HumanMessage("a", id="h1")]
        self.assertEqual([message.id for message in context_budget.history_to_drop(failed, 0, 1)], [bridge, "h1"])
        dropped = context_budget.history_to_drop(history, 0, 1, lambda message: message.id == "h1")
        self.assertEqual([message.id for message in dropped], [bridge, "h1", "a1"])

    def test_newest_whole_exchanges_are_kept_within_count_and_budget(self):
        history = [
            message
            for index in range(5)
            for message in (HumanMessage(f"q{index}", id=f"h{index}"), AIMessage(f"r{index}", id=f"a{index}"))
        ]
        with mock.patch.object(context_budget, "MAX_HISTORY_EXCHANGES", 2):
            dropped = context_budget.history_to_drop(history, 0, 1)
        self.assertEqual([message.id for message in dropped], ["h0", "a0", "h1", "a1", "h2", "a2"])
        exchange = sum(context_budget.message_tokens(message) for message in history[-2:])
        with mock.patch.object(context_budget, "HISTORY_BUDGET_TOKENS", exchange):
            dropped = context_budget.history_to_drop(history, 0, 1)
        self.assertEqual(len(dropped), 8)
        self.assertEqual(context_budget.history_to_drop([], 0, 1), ())
        unfinished = [
            HumanMessage("q0", id="h0"),
            AIMessage("r0", id="a0"),
            HumanMessage("failed", id="hf"),
            HumanMessage("q1", id="h1"),
            AIMessage("", tool_calls=[_call("x", "c1")], id="at"),
            HumanMessage("q2", id="h2"),
            AIMessage("r2", id="a2"),
        ]
        self.assertEqual(
            [message.id for message in context_budget.history_to_drop(unfinished, 0, 1)], ["hf", "h1", "at"]
        )

    def test_a_reply_the_provider_cut_short_never_completes_an_exchange(self):
        for metadata in (
            {"model_provider": "openai", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            {"model_provider": "openai", "finish_reason": "length"},
            {"model_provider": "anthropic", "stop_reason": "max_tokens"},
            {"model_provider": "anthropic", "stop_reason": "model_context_window_exceeded"},
        ):
            with self.subTest(metadata=metadata):
                cut = AIMessage("partial answer", response_metadata=metadata, id="a1")
                self.assertTrue(context_budget.truncated(cut))
                self.assertEqual(context_budget.final_reply(cut), "")
                self.assertFalse(context_budget.completed((HumanMessage("q", id="h1"), cut)))
                history = [HumanMessage("q0", id="h0"), AIMessage("r0", id="a0"), HumanMessage("q", id="h1"), cut]
                self.assertEqual([m.id for m in context_budget.history_to_drop(history, 0, 1)], ["h1", "a1"])
        for metadata in (
            {},
            {"status": "completed"},
            {"finish_reason": "stop"},
            {"stop_reason": "end_turn"},
            {"stop_reason": ["max_tokens"]},
        ):
            with self.subTest(metadata=metadata):
                finished = AIMessage("answer", response_metadata=metadata, id="a1")
                self.assertFalse(context_budget.truncated(finished))
                self.assertEqual(context_budget.final_reply(finished), "answer")

    def test_older_action_results_give_way_to_placeholders_only_at_the_trigger(self):
        def exchange(index: int, *, result: str = "r" * 3_000, name: str = TOOL) -> list:
            return [
                HumanMessage(f"q{index}", id=f"h{index}"),
                AIMessage("", tool_calls=[_call("x", f"c{index}")], id=f"t{index}"),
                ToolMessage(result, tool_call_id=f"c{index}", name=name, status="error", id=f"m{index}"),
                AIMessage(f"done {index}", id=f"a{index}"),
            ]

        history = [*exchange(0), *exchange(1, name="other"), *exchange(2), *exchange(3), *exchange(4)]
        history.append(HumanMessage("failed", id="hf"))
        history.extend(exchange(5)[1:3])

        def action(message: ToolMessage) -> bool:
            return message.name == TOOL

        with mock.patch.object(context_budget, "OMIT_RESULTS_TRIGGER_TOKENS", 1):
            omitted = context_budget.results_to_omit(history, action, lambda message: message.id == "h2")
        # Exchanges 3 and 4 are the newest completed ones; 2 is forgotten whole, 1 holds no Action result, and the
        # unfinished one is dropped whole instead.
        self.assertEqual([message.id for message in omitted], ["m0"])
        self.assertEqual(
            (omitted[0].content, omitted[0].tool_call_id, omitted[0].name, omitted[0].status),
            (context_budget.OMITTED_RESULT, "c0", TOOL, "error"),
        )
        # Below the trigger nothing changes, and an omitted result is never omitted again.
        self.assertEqual(context_budget.results_to_omit(history, action), ())
        with mock.patch.object(context_budget, "OMIT_RESULTS_TRIGGER_TOKENS", 1):
            settled = [*exchange(0, result=context_budget.OMITTED_RESULT), *exchange(3), *exchange(4)]
            self.assertEqual(context_budget.results_to_omit(settled, action), ())
        # A result no larger than its placeholder stays, and only the tokens placeholders save count toward the trigger:
        # many short results never grow the history.
        short = [message for index in range(40) for message in exchange(index, result="ok")]
        with mock.patch.object(context_budget, "OMIT_RESULTS_TRIGGER_TOKENS", 1):
            self.assertEqual(context_budget.results_to_omit(short, action), ())
        saving = context_budget.message_tokens(history[2]) - context_budget.message_tokens(omitted[0])
        with mock.patch.object(context_budget, "OMIT_RESULTS_TRIGGER_TOKENS", saving + 1):
            self.assertEqual(context_budget.results_to_omit(history, action, lambda message: message.id == "h2"), ())

    def test_fixed_prompt_and_current_message_alone_can_exceed_the_window(self):
        with self.assertRaises(context_budget.ContextWindowError):
            context_budget.history_to_drop([], context_budget.MODEL_WINDOW_TOKENS, 1)
        # The JSON encoding counts: 298 characters plus two quotes are 300 bytes, 100 estimated tokens.
        self.assertEqual(context_budget.estimated_tokens("x" * 298), 100)


class RuntimeBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingModel.seen = []
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "checkpoints.sqlite3"
        self.connections: list[sqlite3.Connection] = []

    def tearDown(self) -> None:
        for connection in self.connections:
            connection.close()
        self.directory.cleanup()

    def _runtime(self, *responses: AIMessage) -> agent_runtime.AgentRuntime:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connections.append(connection)
        saver = runtime_api.PruningSqliteSaver(connection)
        saver.setup()
        model = RecordingModel(responses=list(responses))
        return agent_runtime.AgentRuntime(saver, model_factory=FixedFactory(lambda _config: model))

    def test_a_new_turn_forgets_the_oldest_whole_exchanges_and_resume_still_finds_its_reply(self):
        turn = _context()
        runtime = self._runtime(
            AIMessage("r1"),
            AIMessage("", tool_calls=[_call("a.example", "c1"), _call("b.example", "c2")]),
            AIMessage("both hosts answered"),
            AIMessage("r3"),
            AIMessage("", tool_calls=[_call("c.example", "c3")]),
            AIMessage("c answered"),
        )
        runtime.start(turn, "one")
        parallel = runtime.start(turn, "two")
        self.assertEqual(len(parallel.actions), 2)
        runtime.resume(turn, {request.interrupt_id: {"status": "ok"} for request in parallel.actions})
        runtime.start(turn, "three")
        with mock.patch.object(context_budget, "MAX_HISTORY_EXCHANGES", 2):
            pending = runtime.start(turn, "four")
            self.assertEqual(_user_texts(RecordingModel.seen[-1]), ["two", "three", "four"])
            # The kept Action exchange reaches the model whole: both calls and both results.
            self.assertEqual(sum(isinstance(message, ToolMessage) for message in RecordingModel.seen[-1]), 2)
            done = runtime.resume(turn, {pending.actions[0].interrupt_id: {"status": "ok"}})
        self.assertEqual((done.status, done.reply), ("completed", "c answered"))

        stored = runtime._checkpointer.get_tuple(runtime._config(turn))
        self.assertEqual(_user_texts(stored.checkpoint["channel_values"]["messages"]), ["two", "three", "four"])
        self.assertFalse(agent_runtime._has_pending_interrupt(stored.pending_writes))
        self.assertEqual(stored.metadata[agent_runtime.ASSISTANT_SCOPE_METADATA], agent_runtime._assistant_scope(turn))

        reopened = self._runtime(AIMessage("r5"))
        with mock.patch.object(context_budget, "MAX_HISTORY_EXCHANGES", 1):
            self.assertEqual(reopened.start(turn, "five").reply, "r5")
        self.assertEqual(_user_texts(RecordingModel.seen[-1]), ["four", "five"])
        # Pruning leaves one self-contained checkpoint that a fresh connection reads back as the trimmed history.
        reopened._prune_history(turn.thread_id)
        rows = self.connections[-1].execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        self.assertEqual(rows, 1)
        fresh = self._runtime()
        stored = fresh._checkpointer.get_tuple(fresh._config(turn))
        self.assertEqual(_user_texts(stored.checkpoint["channel_values"]["messages"]), ["four", "five"])

    def test_a_new_turn_omits_older_action_results_in_place_and_keeps_the_newest_whole(self):
        turn = _context()
        hosts = ("a.example", "b.example", "c.example")
        runtime = self._runtime(
            *(
                message
                for index, host in enumerate(hosts, 1)
                for message in (AIMessage("", tool_calls=[_call(host, f"c{index}")]), AIMessage(f"{host} answered"))
            ),
            AIMessage("r4"),
        )
        for host in hosts:
            pending = runtime.start(turn, host)
            runtime.resume(turn, {pending.actions[0].interrupt_id: {"status": "ok", "host": host, "body": "x" * 3_000}})
        with mock.patch.object(context_budget, "OMIT_RESULTS_TRIGGER_TOKENS", 1):
            self.assertEqual(runtime.start(turn, "four").reply, "r4")

        def results(messages: Sequence[Any]) -> dict[str, str]:
            return {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}

        seen = results(RecordingModel.seen[-1])
        # The oldest result gives way in place, still paired with its call; the newest two stay whole.
        self.assertEqual(seen["c1"], context_budget.OMITTED_RESULT)
        self.assertIn("b.example", seen["c2"])
        self.assertIn("c.example", seen["c3"])
        self.assertEqual(_user_texts(RecordingModel.seen[-1]), [*hosts, "four"])
        stored = runtime._checkpointer.get_tuple(runtime._config(turn))
        self.assertEqual(results(stored.checkpoint["channel_values"]["messages"]), seen)

    def test_a_refusal_shorter_than_the_placeholder_stays_while_a_large_older_result_gives_way(self):
        turn = _context()
        invalid = {"name": TOOL, "args": {"unknown": 1}, "id": "c1", "type": "tool_call"}
        runtime = self._runtime(
            AIMessage("", tool_calls=[invalid]),
            AIMessage("the arguments were wrong"),
            AIMessage("", tool_calls=[_call("big.example", "c2")]),
            AIMessage("big answered"),
            AIMessage("r3"),
            AIMessage("r4"),
            AIMessage("r5"),
        )
        runtime.start(turn, "one")
        pending = runtime.start(turn, "two")
        runtime.resume(turn, {pending.actions[0].interrupt_id: {"status": "ok", "body": "x" * 3_000}})
        runtime.start(turn, "three")
        runtime.start(turn, "four")
        with mock.patch.object(context_budget, "OMIT_RESULTS_TRIGGER_TOKENS", 1):
            self.assertEqual(runtime.start(turn, "five").reply, "r5")
        seen = {m.tool_call_id: str(m.content) for m in RecordingModel.seen[-1] if isinstance(m, ToolMessage)}
        self.assertTrue(seen["c1"].startswith("Action not executed"))
        self.assertEqual(seen["c2"], context_budget.OMITTED_RESULT)

    def test_window_guards_refuse_before_any_provider_call(self):
        turn = _context()
        runtime = self._runtime(AIMessage("", tool_calls=[_call("a.example", "c1")]))
        with (
            mock.patch.object(context_budget, "MODEL_WINDOW_TOKENS", context_budget.OUTPUT_RESERVE_TOKENS + 10),
            self.assertRaisesRegex(agent_runtime.RuntimeContractError, "model window"),
        ):
            runtime.start(turn, "x" * 1_000)
        self.assertEqual(RecordingModel.seen, [])
        pending = runtime.start(turn, "ping")
        calls = len(RecordingModel.seen)
        huge = {pending.actions[0].interrupt_id: {"body": "x" * 4_000}}
        with (
            mock.patch.object(context_budget, "MODEL_WINDOW_TOKENS", context_budget.OUTPUT_RESERVE_TOKENS + 2_000),
            self.assertRaisesRegex(agent_runtime.RuntimeContractError, "model window"),
        ):
            runtime.resume(turn, huge)
        self.assertEqual(len(RecordingModel.seen), calls)

    def test_a_failed_turn_is_forgotten_instead_of_resent_with_the_next_request(self):
        class FailOnce(RecordingModel):
            def _generate(self, messages: list[Any], *args: Any, **kwargs: Any):
                if str(messages[-1].content) in {"boom", "tool boom"}:
                    raise RuntimeError("provider outage")
                return super()._generate(messages, *args, **kwargs)

        turn = _context()
        model = FailOnce(
            responses=[
                AIMessage("r1"),
                AIMessage("", tool_calls=[_call("a.example", "c1")]),
                AIMessage("r3"),
            ]
        )
        connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connections.append(connection)
        saver = runtime_api.PruningSqliteSaver(connection)
        saver.setup()
        runtime = agent_runtime.AgentRuntime(saver, model_factory=FixedFactory(lambda _config: model))
        runtime.start(turn, "one")
        with self.assertRaises(agent_runtime.ProviderRequestError):
            runtime.start(turn, "boom")
        failed = saver.get_tuple(runtime._config(turn))
        self.assertEqual(_user_texts(failed.checkpoint["channel_values"]["messages"]), ["one", "boom"])
        pending = runtime.start(turn, "ping")
        self.assertEqual(_user_texts(RecordingModel.seen[-1]), ["one", "ping"])
        # A resumed round that fails leaves an unanswered Action round; the next turn forgets it too.
        with self.assertRaises(agent_runtime.ProviderRequestError):
            runtime.resume(turn, {pending.actions[0].interrupt_id: "tool boom"})
        self.assertEqual(runtime.start(turn, "three").reply, "r3")
        self.assertEqual(_user_texts(RecordingModel.seen[-1]), ["one", "three"])
        self.assertFalse(any(isinstance(message, ToolMessage) for message in RecordingModel.seen[-1]))

    def test_a_rejected_empty_reply_is_forgotten_like_any_failed_turn(self):
        turn = _context()
        runtime = self._runtime(AIMessage("r1"), AIMessage("   "), AIMessage("r3"))
        runtime.start(turn, "one")
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "without an Assistant reply"):
            runtime.start(turn, "empty")
        self.assertEqual(runtime.start(turn, "three").reply, "r3")
        self.assertEqual(_user_texts(RecordingModel.seen[-1]), ["one", "three"])

    def test_a_response_cut_short_by_either_real_adapter_fails_before_any_tool_and_is_forgotten(self):
        clarify = {
            "question": "Which host?",
            "options": [{"label": "a.example", "description": ""}, {"label": "b.example", "description": ""}],
            "default_index": 0,
        }
        for provider, model, body in (
            ("openai", "gpt-6.1-sol", _openai_body),
            ("anthropic", "claude-sonnet-5-5", _anthropic_body),
        ):
            for kind, cut in (
                ("reply", ("TRUNCATED-PARTIAL", None)),
                ("action", (TOOL, {"host": "TRUNCATED-PARTIAL"})),
                ("clarification", (clarification.TOOL_NAME, {**clarify, "question": "TRUNCATED-PARTIAL?"})),
            ):
                with self.subTest(provider=provider, kind=kind):
                    replies = [body(*cut, incomplete=True), body("whole answer", None, incomplete=False)]
                    requests: list[str] = []

                    def handle(request: httpx.Request, replies=replies, requests=requests) -> httpx.Response:
                        requests.append(request.content.decode())
                        return httpx.Response(200, json=replies.pop(0))

                    client = httpx.Client(transport=httpx.MockTransport(handle))
                    self.addCleanup(client.close)
                    connection = sqlite3.connect(self.path, check_same_thread=False)
                    self.connections.append(connection)
                    saver = runtime_api.PruningSqliteSaver(connection)
                    saver.setup()
                    factory = FixedFactory(functools.partial(provider_client.provider_model, http_client=client))
                    runtime = agent_runtime.AgentRuntime(saver, model_factory=factory)
                    config = agent_runtime.ProviderConfig(provider, model, "sk-test")
                    turn = agent_runtime.TurnContext(f"cut-{provider}-{kind}", "Budget Team", (PINGER,), config)
                    with self.assertRaisesRegex(agent_runtime.ProviderResponseError, "cut short"):
                        runtime.start(turn, "first question")
                    failed = saver.get_tuple(runtime._config(turn))
                    self.assertFalse(agent_runtime._has_pending_interrupt(failed.pending_writes))
                    messages = failed.checkpoint["channel_values"]["messages"]
                    self.assertFalse(any(isinstance(message, ToolMessage) for message in messages))
                    self.assertFalse(context_budget.completed(context_budget.exchanges(messages)[-1]))

                    result = runtime.start(turn, "second question")
                    self.assertEqual((result.status, result.reply), ("completed", "whole answer"))
                    self.assertEqual(len(requests), 2)
                    self.assertIn("first question", requests[0])
                    self.assertNotIn("first question", requests[1])
                    self.assertNotIn("TRUNCATED-PARTIAL", requests[1])
                    stored = saver.get_tuple(runtime._config(turn)).checkpoint["channel_values"]["messages"]
                    self.assertNotIn("TRUNCATED-PARTIAL", json.dumps([m.model_dump(mode="json") for m in stored]))

    def test_corrupt_history_and_failed_trimming_fail_closed_as_state_errors(self):
        runtime = self._runtime()
        turn = _context()
        agent = mock.Mock()
        with self.assertRaisesRegex(agent_runtime.RuntimeStateError, "checkpoint state is invalid"):
            runtime._fit_history(agent, turn, (AIMessage("orphan", id="a1"),), HumanMessage("now", id="h9"), ())
        agent.update_state.side_effect = RuntimeError("private checkpoint detail")
        history = (HumanMessage("old", id="h1"), AIMessage("reply", id="a1"))
        with (
            mock.patch.object(context_budget, "MAX_HISTORY_EXCHANGES", 0),
            self.assertRaisesRegex(agent_runtime.RuntimeStateError, "^checkpoint trimming failed$"),
        ):
            runtime._fit_history(agent, turn, history, HumanMessage("now", id="h9"), ())
        agent.update_state.assert_called_once()

    def test_maximum_genesis_fits_but_maximum_schemas_are_refused_before_a_provider_call(self):
        genesis = "g" * agent_runtime.MAX_GENESIS_CHARS
        largest = tuple(
            agent_runtime.AssistantDefinition(f"assistant-{index}", genesis, (ACTION,))
            for index in range(agent_runtime.MAX_ASSISTANTS)
        )
        self.assertEqual(self._runtime(AIMessage("fits")).start(_context(*largest), "hi").reply, "fits")

        filler = {"type": "string", "description": "d" * (action_protocol.MAX_BYTES - 200)}
        wide = tuple(
            agent_runtime.ActionDefinition(
                f"action-{index}",
                "Wide.",
                {"type": "object", "properties": {"v": filler}, "additionalProperties": False},
            )
            for index in range(agent_runtime.MAX_ACTIONS_PER_ASSISTANT)
        )
        # One Assistant carries its 128 Actions at the maximum schema size; the rest carry maximum Genesis only.
        heaviest = tuple(
            agent_runtime.AssistantDefinition(f"assistant-{index}", genesis, wide if index == 0 else ())
            for index in range(agent_runtime.MAX_ASSISTANTS)
        )
        calls = len(RecordingModel.seen)
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "model window"):
            self._runtime().start(_context(*heaviest), "hi")
        self.assertEqual(len(RecordingModel.seen), calls)


if __name__ == "__main__":
    unittest.main()

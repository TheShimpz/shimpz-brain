"""The real Brain graph state round-trips through SQLite checkpoints under LANGGRAPH_STRICT_MSGPACK.

LangGraph reads the switch once at import, so the scenario runs in a fresh interpreter with the image's setting. It
pauses on an authorizing Action with a turn attachment, reopens the checkpoint file, resumes with a Command, lets the
next turn remove the attachment exchange, and reloads the final state. Any type the strict allowlist blocks is a
failure, never a silently degraded value.
"""

import json
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import runtime_api
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde import _msgpack
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener
from test_attachments import _ActionProvider, _attachment, _context


def _scenario(path: Path) -> None:
    events: list[dict] = []
    register_serde_event_listener(events.append)
    if not _msgpack.STRICT_MSGPACK_ENABLED:
        raise AssertionError("the scenario must run with strict msgpack enabled")

    provider = _ActionProvider("openai")
    attached = _context("openai", "gpt-6.1-sol", _attachment("text"))
    first = runtime_api._sqlite_runtime(path)
    paused = provider.runtime(first._checkpointer).start(attached, '{"files":[],"message":"Store the report"}')
    if paused.status != "action-required" or len(paused.actions) != 1:
        raise AssertionError(f"the first turn did not pause on its Action: {paused.status}")
    first.close()

    reopened = runtime_api._sqlite_runtime(path)
    agent = provider.runtime(reopened._checkpointer)
    resumed = agent.resume(attached, {paused.actions[0].interrupt_id: {"stored": True}})
    if (resumed.status, resumed.reply) != ("completed", "Read it."):
        raise AssertionError(f"the resumed turn did not complete: {resumed.status}")
    following = agent.start(_context("openai", "gpt-6.1-sol"), '{"files":[],"message":"And now?"}')
    if following.reply != "Read it." or "Quarterly total" in json.dumps(provider.turns()[-1]["input"]):
        raise AssertionError("the next turn did not drop the attachment exchange")
    reopened.close()

    final = runtime_api._sqlite_runtime(path)
    state = final._checkpointer.get_tuple({"configurable": {"thread_id": "attachments-thread"}})
    messages = state.checkpoint["channel_values"]["messages"]
    final.close()
    if not messages or not all(isinstance(message, BaseMessage) for message in messages):
        raise AssertionError("the reloaded checkpoint holds values that are not messages")
    kinds = {type(message) for message in messages}
    if not {HumanMessage, AIMessage} <= kinds:
        raise AssertionError(f"the reloaded checkpoint lost its conversation: {kinds}")
    if ToolMessage in kinds:
        raise AssertionError("the attachment exchange's Action result survived into the next turn")
    blocked = [event for event in events if "blocked" in event["kind"]]
    if blocked:
        raise AssertionError(f"strict msgpack blocked Brain state: {blocked}")


class StrictCheckpointSerdeTests(unittest.TestCase):
    def test_the_real_graph_state_round_trips_under_strict_msgpack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.sqlite3"
            # A spawned interpreter imports LangGraph afresh, reading the switch from the environment it inherits.
            with mock.patch.dict(os.environ, {"LANGGRAPH_STRICT_MSGPACK": "true"}):
                child = multiprocessing.get_context("spawn").Process(target=_scenario, args=(path,))
                child.start()
            child.join(timeout=120)

        self.assertEqual(child.exitcode, 0, "the strict checkpoint scenario failed; its traceback is on stderr")


if __name__ == "__main__":
    unittest.main()

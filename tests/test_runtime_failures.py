from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import agent_runtime
from langgraph.checkpoint.memory import InMemorySaver
from tests.test_agent_runtime import action, assistant, context


class RuntimeFailureProjectionTests(unittest.TestCase):
    def test_pending_checkpoint_writes_are_strictly_validated(self):
        for pending in ("invalid", [("short", "tuple")]):
            with (
                self.subTest(pending=pending),
                self.assertRaisesRegex(
                    agent_runtime.RuntimeStateError,
                    "pending state is invalid",
                ),
            ):
                agent_runtime._has_pending_interrupt(pending)

        self.assertFalse(agent_runtime._has_pending_interrupt([("task", "channel", {"value": 1})]))
        self.assertTrue(agent_runtime._has_pending_interrupt([("task", "__interrupt__", {"value": 1})]))

    def test_runtime_lifecycle_failures_are_generic_and_closed(self):
        pruning = mock.Mock()
        pruning.prune_thread.side_effect = RuntimeError("private checkpoint")
        runtime = agent_runtime.AgentRuntime(pruning, model_factory=lambda _config: mock.Mock())
        with self.assertRaisesRegex(agent_runtime.RuntimeStateError, "checkpoint pruning failed"):
            runtime._prune_history("team:thread")

        deleting = mock.Mock()
        deleting.delete_thread.side_effect = RuntimeError("private checkpoint")
        runtime = agent_runtime.AgentRuntime(deleting, model_factory=lambda _config: mock.Mock())
        with self.assertRaisesRegex(agent_runtime.RuntimeStateError, "checkpoint deletion failed"):
            runtime.delete_thread("team:thread")

        factory = mock.Mock()
        connection = mock.Mock()
        checkpointer = SimpleNamespace(conn=connection)
        with mock.patch.object(agent_runtime, "ProviderModelFactory", return_value=factory):
            agent_runtime.AgentRuntime(checkpointer).close()
        factory.close.assert_called_once_with()
        connection.close.assert_called_once_with()

        factory_without_close = SimpleNamespace(close=None)
        with mock.patch.object(agent_runtime, "ProviderModelFactory", return_value=factory_without_close):
            agent_runtime.AgentRuntime(object()).close()

    def test_prepare_scope_projects_checkpoint_failures_and_races(self):
        turn = context(thread_id="team:scope:closed")
        expected_metadata = {agent_runtime.ASSISTANT_SCOPE_METADATA: agent_runtime._assistant_scope(turn)}

        checkpointer = mock.Mock()
        checkpointer.get_tuple.side_effect = RuntimeError("private checkpoint")
        runtime = agent_runtime.AgentRuntime(checkpointer, model_factory=lambda _config: mock.Mock())
        with self.assertRaisesRegex(agent_runtime.RuntimeStateError, "checkpoint read failed"):
            runtime._prepare_scope(turn, resume=False)

        checkpointer.get_tuple.side_effect = None
        checkpointer.get_tuple.return_value = None
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "no pending Action request"):
            runtime._prepare_scope(turn, resume=True)

        checkpointer.get_tuple.return_value = SimpleNamespace(
            pending_writes=[("task", "channel", {})],
            metadata=expected_metadata,
            checkpoint={"channel_values": {"messages": []}},
        )
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "no pending Action request"):
            runtime._prepare_scope(turn, resume=True)

        checkpointer.get_tuple.return_value = SimpleNamespace(
            pending_writes=[("task", "__interrupt__", {})],
            metadata={"scope": "changed"},
        )
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "Assistant scope changed"):
            runtime._prepare_scope(turn, resume=True)
        checkpointer.delete_thread.assert_called_with(turn.thread_id)

        checkpointer.get_tuple.return_value = SimpleNamespace(
            pending_writes=[("task", "__interrupt__", {})],
            metadata=expected_metadata,
        )
        self.assertEqual(runtime._prepare_scope(turn, resume=False), 0)

        invalid_states = (
            SimpleNamespace(pending_writes=[], metadata=expected_metadata, checkpoint=None),
            SimpleNamespace(pending_writes=[], metadata=expected_metadata, checkpoint={}),
            SimpleNamespace(
                pending_writes=[],
                metadata=expected_metadata,
                checkpoint={"channel_values": {"messages": 1}},
            ),
        )
        for checkpoint_tuple in invalid_states:
            checkpointer.get_tuple.return_value = checkpoint_tuple
            with (
                self.subTest(checkpoint=checkpoint_tuple),
                self.assertRaisesRegex(
                    agent_runtime.RuntimeStateError,
                    "checkpoint state is invalid",
                ),
            ):
                runtime._prepare_scope(turn, resume=False)

    def test_turn_entrypoints_reject_invalid_input_and_resume_provider_failure(self):
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: mock.Mock())
        for message in (None, " ", "x" * (agent_runtime.MAX_MESSAGE_CHARS + 1)):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    agent_runtime.RuntimeContractError,
                    "invalid chat message",
                ),
            ):
                runtime.start(context(), message)
        for results in ({}, {"": "value"}, {1: "value"}):
            with (
                self.subTest(results=results),
                self.assertRaisesRegex(
                    agent_runtime.RuntimeContractError,
                    "invalid Action resume results",
                ),
            ):
                runtime.resume(context(), results)

        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: mock.Mock())
        failing_agent = mock.Mock()
        failing_agent.invoke.side_effect = RuntimeError("provider secret")
        with (
            mock.patch.object(runtime, "_prepare_scope", return_value=0),
            mock.patch.object(runtime, "_prune_history"),
            mock.patch.object(runtime, "_agent", return_value=failing_agent),
            self.assertRaisesRegex(agent_runtime.ProviderRequestError, "model provider request failed"),
        ):
            runtime.resume(context(), {"interrupt": {"value": "result"}})

    def test_action_tool_name_collision_fails_before_agent_creation(self):
        runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=lambda _config: mock.Mock())
        duplicate_tool = SimpleNamespace(name="duplicate")
        turn = context(assistant("helper", action("first"), action("second")))
        with (
            mock.patch.object(agent_runtime, "_request_action", return_value=duplicate_tool),
            self.assertRaisesRegex(agent_runtime.RuntimeContractError, "Action tool name collision"),
        ):
            runtime._agent(turn)


if __name__ == "__main__":
    unittest.main()

"""The one way a Brain tool guard refuses a model response.

A refused response is answered as a whole: every tool call it made receives the same closed correction, so each call
stays paired with a result in both provider adapters, nothing executes, and the model runs again. Each guard keeps its
own review rules and correction texts; only the refusal shape and the middleware plumbing live here.
"""

import functools
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import ToolMessage


def refusal(calls: list[dict[str, Any]], correction: str) -> dict[str, Any]:
    """Answer every tool call of the refused response with `correction` and send the model back to work."""
    return {
        "messages": [ToolMessage(content=correction, tool_call_id=call["id"], name=call["name"]) for call in calls],
        "jump_to": "model",
    }


@functools.cache
def _guard_class():
    # The agent middleware stack loads with the graph, never when the runtime API module is imported.
    from langchain.agents.middleware import AgentMiddleware, hook_config

    class ReviewGuard(AgentMiddleware):
        """Refuse a whole model response whenever its review names a reason."""

        def __init__(
            self, name: str, review: Callable[[list[Any]], str | None], corrections: Mapping[str, str]
        ) -> None:
            super().__init__()
            self._name = name
            self.review = review
            self.corrections = corrections

        @property
        def name(self) -> str:
            # The agent graph names its nodes after each middleware, so every guard keeps a distinct stable name.
            return self._name

        @hook_config(can_jump_to=["model"])
        def after_model(self, state, runtime) -> dict[str, Any] | None:
            messages = list(state["messages"])
            reason = self.review(messages)
            if reason is None:
                return None
            return refusal(messages[-1].tool_calls, self.corrections[reason])

    return ReviewGuard


def guard(name: str, review: Callable[[list[Any]], str | None], corrections: Mapping[str, str]):
    """The middleware named `name` that refuses with `corrections[reason]` whenever `review` returns a reason."""
    return _guard_class()(name, review, corrections)

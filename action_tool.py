"""The only tool the Brain graph exposes: a typed request that suspends the graph for one declared Action.

Arguments that violate the Action's input schema never suspend the graph: the model receives a closed correction and
may call again within the recursion limit (ADR-0080). Team still validates every request it receives.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

if TYPE_CHECKING:
    from agent_runtime import ActionDefinition

MAX_CORRECTION_CHARS = 1_000
MAX_LISTED_NAMES = 16
# Schema property names are package-authored text; only plain identifiers may appear in the model-visible correction.
_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")


def correction(tool_name: str, action: ActionDefinition) -> str:
    """Closed correction text; it names required properties only when every one is a plain identifier."""
    required = action.input_schema.get("required", ())
    names = list(required)[:MAX_LISTED_NAMES] if isinstance(required, (list, tuple)) else []
    plain = bool(names) and all(isinstance(name, str) and _PLAIN_NAME.fullmatch(name) for name in names)
    listed = f"every required property ({', '.join(names)})" if plain else "every required property"
    return (
        f"Action not executed: the arguments for {tool_name} did not match its input schema. Call it again with only "
        f"its declared properties, {listed}, and values of the declared types and allowed values."
    )[:MAX_CORRECTION_CHARS]


def request_action(tool_name: str, assistant_id: str, action: ActionDefinition) -> StructuredTool:
    """Build a tool that can only suspend the graph with a schema-valid, typed Action request."""
    from jsonschema import Draft202012Validator
    from langgraph.types import interrupt

    validator = Draft202012Validator(dict(action.input_schema))

    def suspend_for_controller(**payload):
        if next(validator.iter_errors(payload), None) is not None:
            return correction(tool_name, action)
        return interrupt(
            {
                "kind": "action",
                "assistant_id": assistant_id,
                "action": action.id,
                "input": payload,
            }
        )

    return StructuredTool.from_function(
        suspend_for_controller,
        name=tool_name,
        description=f"Internal Assistant {assistant_id}, Action {action.id}: {action.summary}",
        args_schema=dict(action.input_schema),
        infer_schema=False,
    )

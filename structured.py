"""Provider-native structured output and its closed re-validation, shared by every structured decision."""

from __future__ import annotations

import json
from collections.abc import Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from runtime_errors import RuntimeContractError


def structured_output(model: BaseChatModel, provider: str, schema: type[BaseModel]):
    """Bind provider-native JSON-schema output that also returns the raw message for closed validation."""
    options: dict[str, object] = {"method": "json_schema", "include_raw": True}
    if provider == "openai":
        options["strict"] = True
    elif provider != "anthropic":
        raise RuntimeContractError("unsupported model provider")
    return model.with_structured_output(schema, **options)


def _closed_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeContractError("duplicate structured response field")
        result[key] = value
    return result


def _raw_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _text_value[Schema: BaseModel](text: str, schema: type[Schema], label: str, max_chars: int) -> Schema:
    """Re-read the raw JSON text itself: bounded, free of duplicate keys, and schema-valid."""
    if len(text) > max_chars:
        raise RuntimeContractError(f"invalid {label} response")
    try:
        return schema.model_validate(json.loads(text, object_pairs_hook=_closed_json_object))
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise RuntimeContractError(f"invalid {label} response") from exc


def structured_value[Schema: BaseModel](result: object, schema: type[Schema], label: str, max_chars: int) -> Schema:
    """Return one schema-valid value, refusing a refusal, a tool call, or a parse failure as a response failure.

    When the provider returned JSON text, that text is re-read and must agree with the adapter's parsed value, so a
    duplicate key or an oversized reply cannot hide behind a lenient parser. Native schema output narrows the shape;
    callers still apply their own exact-identifier and semantic checks.
    """
    if not isinstance(result, Mapping) or set(result) != {"raw", "parsed", "parsing_error"}:
        raise RuntimeContractError(f"invalid {label} response")
    raw = result["raw"]
    if not isinstance(raw, AIMessage) or raw.tool_calls or raw.invalid_tool_calls:
        raise RuntimeContractError(f"invalid {label} response")
    if isinstance(raw.content, list) and any(
        isinstance(block, Mapping) and block.get("type") == "refusal" for block in raw.content
    ):
        raise RuntimeContractError(f"{label} response was refused")
    if result["parsing_error"] is not None:
        raise RuntimeContractError(f"invalid {label} response")
    parsed = result["parsed"]
    if isinstance(parsed, Mapping):
        try:
            parsed = schema.model_validate(parsed)
        except ValueError as exc:
            raise RuntimeContractError(f"invalid {label} response") from exc
    if not isinstance(parsed, schema):
        raise RuntimeContractError(f"invalid {label} response")
    text = _raw_text(raw.content).strip()
    if text and _text_value(text, schema, label, max_chars) != parsed:
        raise RuntimeContractError(f"inconsistent {label} response")
    return parsed

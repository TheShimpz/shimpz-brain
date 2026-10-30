"""One structured provider call of an evaluation judge, sent straight to the provider over HTTPS (ADR-0094).

The request is the one the judges' former LangChain models sent: OpenAI Responses with low reasoning effort, the
output limit, and a strict ``json_schema`` text format; Anthropic Messages with low effort, the output limit, and an
``output_config`` JSON-schema format. Nothing retries, the key travels only in a header, and every request passes the
caller's evaluation ceiling (``eval.ceiling.Ceiling.install``), which meters this process's httpx transport.

The answer is re-read closed: a refusal, a tool call, an incomplete answer, a duplicate key, an answer longer than its
bound, or a value the schema rejects is a failure, never a verdict.
"""

import json
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel

ENDPOINTS = {"openai": "https://api.openai.com/v1/responses", "anthropic": "https://api.anthropic.com/v1/messages"}
ANTHROPIC_VERSION = "2023-06-01"
EFFORT = "low"


class CallError(RuntimeError):
    """The provider request or its answer failed; it never says what the provider wrote."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """A judge's whole input: its system text and one user message."""

    system: str
    user: str


@dataclass(frozen=True, slots=True)
class Model:
    """One bounded judge model: its provider, model, key, output limit, and request deadline."""

    provider: str
    model: str
    api_key: str = field(repr=False)
    max_output_tokens: int
    timeout: float

    def __post_init__(self) -> None:
        if self.provider not in ENDPOINTS:
            raise ValueError("unsupported judge provider")


def request_body(model: Model, prompt: Prompt, schema: type[BaseModel]) -> dict[str, object]:
    """The provider request for one structured answer in ``schema``."""
    json_schema = schema.model_json_schema()
    if model.provider == "openai":
        return {
            "model": model.model,
            "input": [
                {"content": prompt.system, "role": "system", "type": "message"},
                {"content": prompt.user, "role": "user", "type": "message"},
            ],
            "max_output_tokens": model.max_output_tokens,
            "reasoning": {"effort": EFFORT},
            "store": False,
            "text": {"format": {"type": "json_schema", "strict": True, "name": schema.__name__, "schema": json_schema}},
        }
    return {
        "model": model.model,
        "max_tokens": model.max_output_tokens,
        "system": prompt.system,
        "messages": [{"role": "user", "content": prompt.user}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": json_schema}},
    }


def _headers(model: Model) -> dict[str, str]:
    if model.provider == "openai":
        return {"Authorization": f"Bearer {model.api_key}"}
    return {"x-api-key": model.api_key, "anthropic-version": ANTHROPIC_VERSION}


def _openai_texts(answer: dict) -> list[str]:
    if answer.get("status") not in (None, "completed"):
        raise CallError("incomplete answer")
    texts = []
    for item in answer.get("output") or []:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind in {"function_call", "custom_tool_call"}:
            raise CallError("tool call")
        if kind != "message":
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "refusal":
                raise CallError("refused")
            if isinstance(block, dict) and block.get("type") == "output_text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
    return texts


def _anthropic_texts(answer: dict) -> list[str]:
    if answer.get("stop_reason") == "refusal":
        raise CallError("refused")
    texts = []
    for block in answer.get("content") or []:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind in {"tool_use", "server_tool_use"}:
            raise CallError("tool call")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
    return texts


def _closed_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _value[Schema: BaseModel](text: str, schema: type[Schema], max_chars: int) -> Schema:
    """Read one JSON text closed: bounded, free of duplicate keys, and schema-valid."""
    if len(text) > max_chars:
        raise CallError("answer too long")
    try:
        return schema.model_validate(json.loads(text, object_pairs_hook=_closed_object))
    except (UnicodeError, ValueError) as exc:
        raise CallError("invalid answer") from exc


def structured_value[Schema: BaseModel](texts: list[str], schema: type[Schema], max_chars: int) -> Schema:
    """The one value the answer's text blocks hold.

    The text is read whole; a model that wrote the same value twice, as separate text blocks, agrees when every block
    re-reads to that value. Two different values never agree, and the whole text stays within ``max_chars``.
    """
    text = "".join(texts).strip()
    if not text:
        raise CallError("empty answer")
    try:
        return _value(text, schema, max_chars)
    except CallError:
        blocks = [block.strip() for block in texts if block.strip()]
        if len(blocks) < 2 or len(text) > max_chars:
            raise
        values = [_value(block, schema, max_chars) for block in blocks]
        if any(value != values[0] for value in values):
            raise CallError("inconsistent answer") from None
        return values[0]


def structured[Schema: BaseModel](
    model: Model, prompt: Prompt, schema: type[Schema], max_chars: int, client: httpx.Client | None = None
) -> Schema:
    """One structured call: send it, require a 200 JSON answer, and re-read its text closed."""
    body = request_body(model, prompt, schema)
    session = client if client is not None else httpx.Client(timeout=model.timeout)
    try:
        response = session.post(ENDPOINTS[model.provider], json=body, headers=_headers(model))
        if response.status_code != 200:
            raise CallError(f"provider status {response.status_code}")
        answer = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise CallError("provider request failed") from exc
    finally:
        if client is None:
            session.close()
    if not isinstance(answer, dict):
        raise CallError("invalid answer")
    texts = _openai_texts(answer) if model.provider == "openai" else _anthropic_texts(answer)
    return structured_value(texts, schema, max_chars)

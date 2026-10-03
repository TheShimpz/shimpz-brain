"""The spend ceiling of a disposable evaluation process: bound and reserve every provider request on the wire.

Experiment-only (ADR-0094); the umbrella journey driver installs it into its own Brain process through
``.tests/perf/precision_brain_budget.py`` and the judge commands into theirs, never into a product process. It sits
at the httpx transport boundary, where the SDKs have merged every override (``extra_body`` included) and expanded
every schema, so it sees the exact bytes that would be sent:

- only the chat endpoints (``/v1/responses`` and ``/v1/chat/completions`` on api.openai.com, ``/v1/messages`` on
  api.anthropic.com) are admitted, non-streaming; any other request to a provider host is refused;
- every output-limit field of the JSON body is clamped to the output limit, and the endpoint's own field is set when
  absent; the body is re-encoded;
- the worst case of that final body is reserved for the model the body names: one input token per byte plus an
  allowance at the dearest input rate, and the output limit at the output rate; a request the cap cannot hold is
  refused before it is sent, and one that only has to wait for in-flight work to settle waits; a request to an
  evaluation-only model whose bound exceeds the input its price holds for is refused as unsupported;
- the response's reported usage settles the reservation; a failed or unparsable response keeps all of it.

Chat models are also built with no SDK retries; a retry would be one more request reserved on its own. The ceiling
holds under two stated assumptions: a provider bills at most one input token per body byte plus the allowance, and it
honors the output limit. A response costing more than its reservation is counted as ``reservations_exceeded``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import NoReturn

import httpx
from eval import cost as eval_cost
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

LIMIT_FIELDS = ("max_output_tokens", "max_completion_tokens", "max_tokens")
REQUEST_ALLOWANCE_TOKENS = 4096
PROVIDER_HOSTS = frozenset({"api.openai.com", "api.anthropic.com"})
ENDPOINTS = {
    ("api.openai.com", "/v1/responses"): "max_output_tokens",
    ("api.openai.com", "/v1/chat/completions"): "max_completion_tokens",
    ("api.anthropic.com", "/v1/messages"): "max_tokens",
}


class UnsupportedRequestError(RuntimeError):
    """A provider request the ceiling cannot bound: another endpoint, streaming, or a body that is not JSON."""


def _tokens(usage: dict, key: str, *, required: bool) -> int | None:
    """A non-negative integer token count; an optional count may be absent or null (zero)."""
    value = usage.get(key)
    if value is None and not required:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _counts(usage: dict, required: tuple[str, ...], optional: tuple[str, ...]) -> list[int] | None:
    counts = [_tokens(usage, key, required=True) for key in required]
    counts += [_tokens(usage, key, required=False) for key in optional]
    return None if None in counts else counts


def usage_of(host: str, body: object) -> eval_cost.Usage | None:
    """The usage a chat response reports, in Brain's terms (input includes cache reads and writes), or None.

    Every required count must be present as a non-negative integer, and every optional one absent, null, or such an
    integer; anything else is unreported usage, so the request keeps its whole reservation.
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    if host == "api.anthropic.com":
        counts = _counts(
            usage, ("input_tokens", "output_tokens"), ("cache_read_input_tokens", "cache_creation_input_tokens")
        )
        if counts is None:
            return None
        fresh, output, read, write = counts
        return eval_cost.Usage(
            model_calls=1,
            input_tokens=fresh + read + write,
            output_tokens=output,
            cache_read_tokens=read,
            cache_write_tokens=write,
        )
    responses = "input_tokens" in usage
    names = ("input_tokens", "output_tokens") if responses else ("prompt_tokens", "completion_tokens")
    details = usage.get("input_tokens_details" if responses else "prompt_tokens_details")
    details = {} if details is None else details
    counts = _counts(usage, names, ())
    cached = _counts(details, (), ("cached_tokens",)) if isinstance(details, dict) else None
    if counts is None or cached is None:
        return None
    return eval_cost.Usage(model_calls=1, input_tokens=counts[0], output_tokens=counts[1], cache_read_tokens=cached[0])


class Ceiling:
    """One process's ceiling: install it once; ``uninstall`` restores the patched classes (tests only)."""

    def __init__(self, cap: float, max_output_tokens: int, state: Path | None = None) -> None:
        self.budget = eval_cost.Budget(cap)
        self.max_output = max_output_tokens
        self.state = state
        self.counts = {"requests": 0, "refused": 0, "failed": 0, "unreported": 0, "unsupported": 0}
        self.max_reservation = 0.0
        self._lock = threading.Lock()
        self._saved: list[tuple[type, str, object]] = []

    def write_state(self) -> None:
        """Write the state snapshot atomically: a unique temporary file replaced under the lock."""
        if self.state is None:
            return
        with self._lock:
            body = {**self.budget.summary(), **self.counts, "max_output_tokens": self.max_output}
            body["max_reservation_usd"] = round(self.max_reservation, 6)
            descriptor, temporary = tempfile.mkstemp(prefix=self.state.name, dir=self.state.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(body, handle)
            Path(temporary).replace(self.state)

    def _count(self, name: str) -> None:
        with self._lock:
            self.counts[name] += 1

    def refuse_unsupported(self) -> NoReturn:
        self._count("unsupported")
        self.write_state()
        raise UnsupportedRequestError("the evaluation ceiling cannot bound this provider request")

    def admit(self, request: httpx.Request) -> tuple[httpx.Request, eval_cost.Reservation, str]:
        """Clamp the outgoing body, reserve its worst case for the model it names, and return the request to send."""
        field = ENDPOINTS.get((request.url.host, request.url.path))
        try:
            body = json.loads(request.read()) if field else None
        except ValueError:
            body = None
        if field is None or not isinstance(body, dict) or body.get("stream") or not isinstance(body.get("model"), str):
            self.refuse_unsupported()
        for name in LIMIT_FIELDS:
            if body.get(name) is not None:
                body[name] = min(int(body[name]), self.max_output)
        body[field] = min(int(body.get(field) or self.max_output), self.max_output)
        content = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        model = body["model"]
        tokens = len(content) + REQUEST_ALLOWANCE_TOKENS
        if tokens > eval_cost.PRICED_INPUT_TOKENS.get(model, tokens):
            self.refuse_unsupported()
        amount = eval_cost.call_bound(model, tokens, self.max_output)
        try:
            reservation = self.budget.reserve(amount, wait=True)
        except eval_cost.BudgetExhaustedError:
            self._count("refused")
            self.write_state()
            raise
        with self._lock:
            self.counts["requests"] += 1
            self.max_reservation = max(self.max_reservation, amount)
        headers = [(key, value) for key, value in request.headers.multi_items() if key.lower() != "content-length"]
        admitted = httpx.Request(
            request.method, request.url, headers=headers, content=content, extensions=request.extensions
        )
        return admitted, reservation, model

    def settle(
        self, reservation: eval_cost.Reservation, model: str, host: str, response: httpx.Response | None
    ) -> None:
        """Settle the reservation exactly once, whatever the response holds: unreadable usage keeps all of it."""
        usage = None
        try:
            if response is not None:
                usage = usage_of(host, json.loads(response.read()))
        except ValueError:
            usage = None
        finally:
            if usage is None:
                self.budget.settle(reservation, eval_cost.Cost(0.0, known=False))
                self._count("failed" if response is None or response.status_code >= 400 else "unreported")
            else:
                self.budget.settle(reservation, eval_cost.cost(usage, model))
            self.write_state()

    def install(self) -> None:
        ceiling = self
        original_send = httpx.Client.send

        def send(client, request, /, **kwargs):
            if request.url.host not in PROVIDER_HOSTS:
                return original_send(client, request, **kwargs)
            admitted, reservation, model = ceiling.admit(request)
            try:
                response = original_send(client, admitted, **kwargs)
            except BaseException:
                ceiling.settle(reservation, model, request.url.host, None)
                raise
            ceiling.settle(reservation, model, request.url.host, response)
            return response

        original_async_send = httpx.AsyncClient.send

        async def async_send(client, request, /, **kwargs):
            # Brain calls its models synchronously; an asynchronous provider request is refused, not left unmetered.
            if request.url.host in PROVIDER_HOSTS:
                ceiling.refuse_unsupported()
            return await original_async_send(client, request, **kwargs)

        self._saved += [(httpx.Client, "send", original_send), (httpx.AsyncClient, "send", original_async_send)]
        httpx.Client.send = send
        httpx.AsyncClient.send = async_send
        for cls in (ChatOpenAI, ChatAnthropic):
            original_init = cls.__init__
            self._saved.append((cls, "__init__", original_init))

            def init(instance, /, *args, _init=original_init, **kwargs):
                kwargs["max_retries"] = 0
                kwargs["max_tokens"] = ceiling.max_output
                _init(instance, *args, **kwargs)

            cls.__init__ = init
        self.write_state()

    def uninstall(self) -> None:
        for cls, attribute, original in reversed(self._saved):
            setattr(cls, attribute, original)
        self._saved.clear()

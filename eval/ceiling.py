"""The spend ceiling of an evaluation: bound and reserve every provider request on the wire (ADR-0094).

Experiment-only; never part of a product process. It reads the exact bytes a request would send, so it sees every
override and every expanded schema. Two adapters apply it: ``eval.provider_origin``, the loopback origin a disposable
Brain sends its provider requests to, and ``install``, which meters the tooling's own direct provider calls (judges and
helper models) at its httpx transport:

- only the chat endpoints (``/v1/responses`` and ``/v1/chat/completions`` on api.openai.com, ``/v1/messages`` on
  api.anthropic.com), non-streaming, and OpenAI's ``/v1/embeddings`` for a priced embedding model (the language
  component benchmark; it bills input only, so it has no output limit) are admitted; any other provider request is
  refused;
- every output-limit field of the JSON body is clamped to the output limit, and the endpoint's own field is set when
  absent; a body that changes is re-encoded, any other is sent byte for byte;
- the worst case of that final body is reserved for the model the body names: one input token per byte plus an
  allowance at the dearest input rate, and the output limit at the output rate; a request the cap cannot hold is
  refused before it is sent, and one that only has to wait for in-flight work to settle waits; a request to an
  evaluation-only model whose bound exceeds the input its price holds for is refused as unsupported;
- the response's reported usage settles the reservation; a provider's refusal of the request, any 4xx and an
  overload 503 or 529, settles at nothing; any other failed or unparsable response keeps all of it.

Each retry is one more request on the wire, reserved on its own. The reservations draw on one process's ``Budget`` or
on a ``SharedBudget`` every process of a run shares. The ceiling holds under three stated assumptions: a provider bills
at most one input token per body byte plus the allowance, it honors the output limit, and it bills nothing for a
request it refuses. A response costing more than its reservation is counted as ``reservations_exceeded``.

This module uses only the standard library (``install`` imports httpx when called), so a forwarder process can load it
by path from any environment.
"""

import json
import os
import tempfile
import threading
import zlib
from pathlib import Path
from typing import NoReturn

if __package__:
    from eval import cost as eval_cost
else:  # loaded as a script's sibling, as ``provider_origin.py`` runs from any environment
    import cost as eval_cost

LIMIT_FIELDS = ("max_output_tokens", "max_completion_tokens", "max_tokens")
REQUEST_ALLOWANCE_TOKENS = 4096
PROVIDER_HOSTS = frozenset({"api.openai.com", "api.anthropic.com"})
ENDPOINTS = {
    ("api.openai.com", "/v1/responses"): "max_output_tokens",
    ("api.openai.com", "/v1/chat/completions"): "max_completion_tokens",
    ("api.anthropic.com", "/v1/messages"): "max_tokens",
}
EMBEDDINGS = ("api.openai.com", "/v1/embeddings")
# A provider's refusals for rate or overload: 429 from either, OpenAI's 503, Anthropic's 529.
THROTTLE_STATUSES = frozenset({429, 503, 529})
# The refusals a provider never bills: every 4xx rejection of the request, and its overload refusals.
UNBILLED_STATUSES = frozenset(range(400, 500)) | THROTTLE_STATUSES


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


def usage_of(host: str, body: object, path: str = "") -> eval_cost.Usage | None:
    """The usage a chat or embeddings response reports in Brain's terms (input includes cache traffic), or None.

    Every required count must be present as a non-negative integer, and every optional one absent, null, or such an
    integer; anything else is unreported usage, so the request keeps its whole reservation. An embeddings response
    reports only its input (``prompt_tokens``).
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    if (host, path) == EMBEDDINGS:
        counts = _counts(usage, ("prompt_tokens",), ())
        return None if counts is None else eval_cost.Usage(model_calls=1, input_tokens=counts[0])
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
    cached = _counts(details, (), ("cached_tokens", "cache_write_tokens")) if isinstance(details, dict) else None
    if counts is None or cached is None:
        return None
    return eval_cost.Usage(
        model_calls=1,
        input_tokens=counts[0],
        output_tokens=counts[1],
        cache_read_tokens=cached[0],
        cache_write_tokens=cached[1],
    )


# A provider failure no retry can heal: the key is refused, or the account has no credit or quota left.
TERMINAL_STATUSES = frozenset({401, 402, 403})
TERMINAL_WORDS = (b"credit balance", b"insufficient_quota", b"billing")


def clamped(body: object) -> bool:
    """Whether a provider's answer stopped at the output limit the ceiling set."""
    if not isinstance(body, dict):
        return False
    if body.get("stop_reason") == "max_tokens":
        return True
    incomplete = body.get("incomplete_details")
    if isinstance(incomplete, dict) and incomplete.get("reason") == "max_output_tokens":
        return True
    choices = body.get("choices")
    return isinstance(choices, list) and any(
        isinstance(item, dict) and item.get("finish_reason") == "length" for item in choices
    )


def terminal(status: int, body: bytes) -> bool:
    """Whether a provider refused the request in a way every later request meets too (key, credit, or quota)."""
    text = body.lower()
    return status in TERMINAL_STATUSES or (status in (400, 429) and any(word in text for word in TERMINAL_WORDS))


def decoded(body: bytes, encoding: str) -> bytes | None:
    """A response body without its gzip or deflate content coding; None for a coding the ceiling cannot read."""
    coding = encoding.strip().casefold()
    try:
        if coding in {"", "identity"}:
            return body
        if coding in {"gzip", "x-gzip"}:
            return zlib.decompress(body, 16 + zlib.MAX_WBITS)
        if coding == "deflate":
            return zlib.decompress(body)
    except zlib.error:
        return None
    return None


class Ceiling:
    """One ceiling: ``admit_body`` before a request is sent, ``settle_body`` once with whatever came back."""

    def __init__(
        self,
        cap: float,
        max_output_tokens: int,
        state: Path | None = None,
        *,
        budget: eval_cost.SharedBudget | None = None,
    ) -> None:
        self.budget = eval_cost.Budget(cap) if budget is None else budget
        self.max_output = max_output_tokens
        self.state = state
        self.counts = {
            "requests": 0,
            "refused": 0,
            "failed": 0,
            "unreported": 0,
            "unsupported": 0,
            "throttled": 0,
            "rejected": 0,
            # Evidence beside the outcomes: answers that stopped at the output limit, and terminal refusals.
            "clamped": 0,
            "terminal": 0,
        }
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

    def _bounded(self, host: str, path: str, content: bytes) -> tuple[bytes, str, bool]:
        """The body to send, the model it names, and whether it is an embeddings request; or refused unsupported."""
        target = (host, path)
        embeddings = target == EMBEDDINGS
        field = ENDPOINTS.get(target)
        try:
            body = json.loads(content) if field or embeddings else None
        except ValueError:
            body = None
        if not isinstance(body, dict) or body.get("stream") or not isinstance(body.get("model"), str):
            self.refuse_unsupported()
        model = body["model"]
        if embeddings and model not in eval_cost.EMBEDDING_MODELS["openai"]:
            self.refuse_unsupported()
        if embeddings:
            return content, model, True
        clamped_body = dict(body)
        try:
            for name in LIMIT_FIELDS:
                if body.get(name) is not None:
                    clamped_body[name] = min(int(body[name]), self.max_output)
            clamped_body[field] = min(int(body.get(field) or self.max_output), self.max_output)
        except TypeError, ValueError:
            self.refuse_unsupported()
        if any(
            name in clamped_body and (type(body.get(name)) is not int or body[name] != clamped_body[name])
            for name in LIMIT_FIELDS
        ):
            content = json.dumps(clamped_body, ensure_ascii=False, separators=(",", ":")).encode()
        return content, model, False

    def admit_body(self, host: str, path: str, content: bytes) -> tuple[bytes, eval_cost.Reservation, str]:
        """Clamp one outgoing body, reserve its worst case for the model it names, and return the bytes to send."""
        content, model, embeddings = self._bounded(host, path, content)
        tokens = len(content) + REQUEST_ALLOWANCE_TOKENS
        if tokens > eval_cost.PRICED_INPUT_TOKENS.get(model, tokens):
            self.refuse_unsupported()
        try:
            amount = eval_cost.call_bound(model, tokens, 0 if embeddings else self.max_output)
        except ValueError:
            self.refuse_unsupported()
        try:
            reservation = self.budget.reserve(amount, wait=True)
        except eval_cost.BudgetExhaustedError:
            self._count("refused")
            self.write_state()
            raise
        with self._lock:
            self.counts["requests"] += 1
            self.max_reservation = max(self.max_reservation, amount)
        return content, reservation, model

    def settle_body(
        self,
        reservation: eval_cost.Reservation,
        model: str,
        host: str,
        path: str,
        status: int | None,
        body: bytes | None,
    ) -> bool:
        """Settle the reservation exactly once from a response's status and decoded body (None: nothing came back).

        A provider's refusal (any 4xx, and its overload statuses) bills nothing; unreadable usage keeps all of it.
        Return whether the answer stopped at the output limit.
        """
        if status is not None and terminal(status, body or b""):
            self._count("terminal")
        if status is not None and status in UNBILLED_STATUSES:
            self.budget.settle(reservation, eval_cost.Cost(0.0))
            self._count("throttled" if status in THROTTLE_STATUSES else "rejected")
            self.write_state()
            return False
        usage, stopped = None, False
        try:
            if status is not None and body is not None:
                answer = json.loads(body)
                stopped = status == 200 and clamped(answer)
                if stopped:
                    self._count("clamped")
                usage = usage_of(host, answer, path)
        except ValueError:
            usage = None
        finally:
            if usage is None:
                self.budget.settle(reservation, eval_cost.Cost(0.0, known=False))
                self._count("failed" if status is None or status >= 400 else "unreported")
            else:
                self.budget.settle(reservation, eval_cost.cost(usage, model))
            self.write_state()
        return stopped

    def admit(self, request):
        """The httpx adapter of ``admit_body``: the request to send in place of ``request``."""
        import httpx

        content, reservation, model = self.admit_body(request.url.host, request.url.path, request.read())
        headers = [(key, value) for key, value in request.headers.multi_items() if key.lower() != "content-length"]
        admitted = httpx.Request(
            request.method, request.url, headers=headers, content=content, extensions=request.extensions
        )
        return admitted, reservation, model

    def settle(self, reservation: eval_cost.Reservation, model: str, host: str, response, path: str = "") -> None:
        """The httpx adapter of ``settle_body``; a response whose body cannot be read keeps the whole reservation."""
        if response is None:
            self.settle_body(reservation, model, host, path, None, None)
            return
        body = None
        try:
            body = response.read()
        finally:
            self.settle_body(reservation, model, host, path, response.status_code, body)

    def install(self) -> None:
        """Meter this process's own direct provider calls (judges, helper models) at the httpx transport."""
        import httpx

        ceiling = self
        original_send = httpx.Client.send

        def send(client, request, /, **kwargs):
            if request.url.host not in PROVIDER_HOSTS:
                return original_send(client, request, **kwargs)
            admitted, reservation, model = ceiling.admit(request)
            try:
                response = original_send(client, admitted, **kwargs)
            except BaseException:
                ceiling.settle(reservation, model, request.url.host, None, request.url.path)
                raise
            ceiling.settle(reservation, model, request.url.host, response, request.url.path)
            return response

        original_async_send = httpx.AsyncClient.send

        async def async_send(client, request, /, **kwargs):
            # The tooling calls providers synchronously; an asynchronous provider request is refused, never unmetered.
            if request.url.host in PROVIDER_HOSTS:
                ceiling.refuse_unsupported()
            return await original_async_send(client, request, **kwargs)

        self._saved += [(httpx.Client, "send", original_send), (httpx.AsyncClient, "send", original_async_send)]
        httpx.Client.send = send
        httpx.AsyncClient.send = async_send
        self.write_state()

    def uninstall(self) -> None:
        for cls, attribute, original in reversed(self._saved):
            setattr(cls, attribute, original)
        self._saved.clear()

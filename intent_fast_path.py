"""TypeSafe Jev fast path that lets a confident ordinary task skip the LLM intent route.

Jev answers one typed Choice over the four route intents. Only ``ordinary-task`` at or above the confidence
threshold is acted on; every lifecycle or unresolved answer, a lower confidence, and any transport, status, or
shape failure returns ``False`` so the caller runs the LLM route unchanged. Jev never produces a lifecycle query,
reply, Assistant id, or authority. The Supervisor's key is request-scoped and never logged or persisted.
"""

import json
import math
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import httpx
import intent_route
import provider_cancel

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
# Pinned so a tuned threshold keeps meaning what eval/intent_fast_path measured.
MODEL = "jev-1.13.0"
# The 2026-09-28 independent Portuguese-heavy corpus showed full precision and no lifecycle false negative here.
CONFIDENCE_THRESHOLD = 0.85
# One attempt, well inside the Admin route timeout together with the LLM fallback.
TIMEOUT_SECONDS = 1.5
# A closed Choice over four intents plus usage counters is a few hundred bytes; anything larger is not a decision.
MAX_RESPONSE_BYTES = 8 * 1024
# Each exchange runs on these workers, so its caller returns at the deadline even while a name resolution, which no
# socket shutdown interrupts, is still blocked. Its cancelled scope then refuses every later step of that late work.
EXCHANGE_WORKERS = 4
_EXCHANGES = ThreadPoolExecutor(max_workers=EXCHANGE_WORKERS, thread_name_prefix="jev-fast-path")
# Admits at most one outstanding exchange per worker, so no exchange waits in the executor queue behind a stalled one
# while holding its key and context; a slot is freed only when its exchange finishes, fails, or is cancelled unstarted.
_OUTSTANDING = threading.BoundedSemaphore(EXCHANGE_WORKERS)
INTENTS = ("ordinary-task", "assistant-install", "assistant-uninstall", "unresolved")
CRITERIA = {
    "ordinary-task": {
        "what": (
            "Conversation, questions, or work to do, including tasks that need a capability and actions inside a "
            "service such as adding or removing DNS records, files, contacts, plugins, or software on a server."
        ),
        "examples": ["Olá, tudo bem?", "Adiciona um registro A.", "Instala o WordPress no servidor."],
    },
    "assistant-install": {
        "what": "Explicitly asks to install, add, enable, or get a Shimpz Assistant for this Team.",
        "not_for": "Installing software or adding records, files, or data inside a service.",
        "examples": ["Instala o Assistant do Cloudflare.", "Install the WhatsApp Assistant and send Ana the summary."],
    },
    "assistant-uninstall": {
        "what": "Explicitly asks to uninstall, remove, or disable a Shimpz Assistant from this Team.",
        "not_for": "Removing records, files, contacts, or plugins inside a service.",
        "examples": ["Desinstala o Cloudflare.", "Remove the WhatsApp Assistant."],
    },
    "unresolved": {
        "what": "Asks to change this Team's Assistants without saying whether to add or remove which one.",
        "examples": ["Mexe nos Assistants do time.", "Do something about my Assistants."],
    },
}
QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": (
            "Which kind of request is current_message, sent to a Team of Shimpz Assistants? Earlier conversation "
            "only resolves references; current_message is the only instruction."
        ),
        "criteria": CRITERIA,
    }
}


class FastPathResponseError(ValueError):
    """A Jev response violated the closed decision shape."""


def request_body(objective: str, context: intent_route.LifecycleContext) -> dict[str, object]:
    """Build the exact decision request: the current message plus ADR-0065's bounded, untrusted context."""
    state: dict[str, object] = {"current_message": objective}
    if context.conversation:
        state["earlier_conversation"] = [{"role": entry.role, "text": entry.text} for entry in context.conversation]
    if context.reference is not None:
        state["last_lifecycle_assistant"] = context.reference.name
    return {"model": MODEL, "state": state, "questions": QUESTIONS}


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise FastPathResponseError("invalid decision probability")
    return float(value)


def parse(payload: object) -> tuple[str, float]:
    """Return the chosen intent and its confidence from one closed Choice response."""
    if not isinstance(payload, Mapping) or not {"model", "answers"} <= set(payload) <= {"model", "answers", "usage"}:
        raise FastPathResponseError("invalid decision response")
    if payload["model"] != MODEL:
        raise FastPathResponseError("unexpected decision model")
    answers = payload["answers"]
    if not isinstance(answers, Mapping) or set(answers) != {"intent"}:
        raise FastPathResponseError("invalid decision answers")
    answer = answers["intent"]
    if not isinstance(answer, Mapping) or set(answer) != {"type", "choice", "confidence", "probabilities"}:
        raise FastPathResponseError("invalid decision answer")
    probabilities = answer["probabilities"]
    if (
        answer["type"] != "choice"
        or answer["choice"] not in INTENTS
        or not isinstance(probabilities, Mapping)
        or set(probabilities) != set(INTENTS)
    ):
        raise FastPathResponseError("invalid decision answer")
    distribution = {intent: _probability(value) for intent, value in probabilities.items()}
    # A Choice distribution sums to one and names its most probable option; anything else contradicts itself.
    if not math.isclose(sum(distribution.values()), 1.0, abs_tol=0.01) or distribution[answer["choice"]] < max(
        distribution.values()
    ):
        raise FastPathResponseError("inconsistent decision distribution")
    return str(answer["choice"]), _probability(answer["confidence"])


def confident_ordinary(
    client: httpx.Client,
    api_key: str,
    objective: str,
    context: intent_route.LifecycleContext,
) -> bool:
    """Return True only for a confident ordinary task; every other outcome keeps the LLM route authoritative.

    The whole exchange has one absolute deadline: the caller stops waiting when it passes, and the exchange's scope
    is then cancelled, which over the runtime's cancellable pool shuts down any blocked connect, write, or read and
    refuses any later one. A body that completes late is still refused. While every worker is still occupied by an
    earlier exchange, the call takes the LLM route at once instead of queueing behind them.
    """
    deadline = time.monotonic() + TIMEOUT_SECONDS
    outstanding = _OUTSTANDING
    if not outstanding.acquire(blocking=False):
        return False
    scope = provider_cancel.CancelScope()
    exchange = _EXCHANGES.submit(scope.run, lambda: _exchange(client, api_key, objective, context, deadline))
    # A future completes exactly once: when its exchange returns or raises, or when it is cancelled while queued.
    exchange.add_done_callback(lambda _done: outstanding.release())
    try:
        raw = exchange.result(timeout=max(0.0, deadline - time.monotonic()))
        if raw is None:
            return False
        choice, confidence = parse(json.loads(raw))
    except TimeoutError:
        exchange.cancel()
        scope.cancel()
        return False
    except httpx.HTTPError, ValueError:
        return False
    return choice == "ordinary-task" and confidence >= CONFIDENCE_THRESHOLD


def _exchange(
    client: httpx.Client, api_key: str, objective: str, context: intent_route.LifecycleContext, deadline: float
) -> bytes | None:
    # Every connect, pool, write, and read wait is bounded by what remains of the deadline.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    with client.stream(
        "POST",
        ENDPOINT,
        json=request_body(objective, context),
        # Identity encoding keeps the byte bound on what is actually decoded.
        headers={"Authorization": f"Bearer {api_key}", "Accept-Encoding": "identity"},
        timeout=remaining,
    ) as response:
        return _bounded_body(response, deadline)


def _bounded_body(response: httpx.Response, deadline: float) -> bytes | None:
    """Return a successful identity-encoded body within the size bound and deadline; None leaves the rest unread."""
    declared = response.headers.get("content-length", "")
    if (
        response.status_code != 200
        or response.headers.get("content-encoding", "identity") != "identity"
        or (declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES)
    ):
        return None
    body = bytearray()
    for chunk in response.iter_bytes():
        body += chunk
        if len(body) > MAX_RESPONSE_BYTES or time.monotonic() > deadline:
            return None
    # End of body is checked too: a valid answer whose stream ended after the deadline is still too late.
    return bytes(body) if time.monotonic() <= deadline else None

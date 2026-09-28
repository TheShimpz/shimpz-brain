"""TypeSafe Jev fast path that lets a confident ordinary task skip the LLM intent route.

Jev answers one typed Choice over the four route intents. Only ``ordinary-task`` at or above the confidence
threshold is acted on; every lifecycle or unresolved answer, a lower confidence, and any transport, status, or
shape failure returns ``False`` so the caller runs the LLM route unchanged. Jev never produces a lifecycle query,
reply, Assistant id, or authority. The Supervisor's key is request-scoped and never logged or persisted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import httpx
import intent_route

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
# Pinned so a tuned threshold keeps meaning what eval/intent_fast_path measured.
MODEL = "jev-1.13.0"
# The 2026-09-28 independent Portuguese-heavy corpus showed full precision and no lifecycle false negative here.
CONFIDENCE_THRESHOLD = 0.85
# One attempt, well inside the Admin route timeout together with the LLM fallback.
TIMEOUT_SECONDS = 1.5
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
    """Return True only for a confident ordinary task; every other outcome keeps the LLM route authoritative."""
    try:
        response = client.post(
            ENDPOINT,
            json=request_body(objective, context),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return False
        choice, confidence = parse(response.json())
    except httpx.HTTPError, ValueError:
        return False
    return choice == "ordinary-task" and confidence >= CONFIDENCE_THRESHOLD

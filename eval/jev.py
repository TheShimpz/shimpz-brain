"""Experiment-only Jev decisions for the engineering arms (ADR-0094), called by the journey driver, never by Brain.

TypeSafe ``jev-1.13.0`` answers typed questions with calibrated probabilities. A probe on 2026-10-02 found that a
Choice is single-select but returns the full probability distribution (so a ranked top-k is available), that several
Noul (yes/no probability) questions in one request give an independent multi-label answer, that 14 options or 14
Nouls answer in about 270 to 320 ms, and that requests cost $0.042 per million input tokens, with no output charge
(docs.typesafe.ai/models, 2026-10-02).

- L6 asks one Noul per resource group of the large-API Assistant and exposes every group at or above
  ``GROUP_THRESHOLD``, plus the zones group; when no group reaches it, the arm falls back.
- E-Jev asks one Choice before the turn: a confident ``simple-read`` or ``simple-write`` turn runs on Luna, every
  other turn on the escalation model.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
GROUP_THRESHOLD = 0.5
ROUTE_THRESHOLD = 0.7
SIMPLE = frozenset({"simple-read", "simple-write"})
ROUTES = {
    "simple-read": {
        "what": "Asks for information one lookup in one service can answer, with every needed value given.",
        "examples": ["Which records in example.com point to 192.0.2.1?", "Liste minhas tarefas abertas."],
    },
    "simple-write": {
        "what": "Asks for one clear change in one service, with the target and every value given.",
        "examples": [
            "Create an A record named shop in example.com pointing to 203.0.113.10.",
            "Bloqueie o IP 1.2.3.4.",
        ],
    },
    "ambiguous": {
        "what": "Leaves out a value the change needs, or names a target that could mean more than one thing.",
        "examples": ["Send a message to Bruno.", "Delete the bucket."],
    },
    "compound-or-sensitive-write": {
        "what": "Needs several steps or services, finding a value before changing something, or a broad or risky "
        "change such as deleting, purging everything, or many records at once.",
        "examples": ["Find the new IP of status.example.org and point status.example.com to it.", "Purge everything."],
    },
}


class JevError(RuntimeError):
    """The Jev request or its response failed; the arm falls back as declared."""


@dataclass(frozen=True, slots=True)
class Decision:
    answers: Mapping[str, object]
    input_tokens: int
    seconds: float

    @property
    def usd(self) -> float:
        return self.input_tokens * USD_PER_INPUT_TOKEN


Transport = Callable[[dict[str, object]], dict[str, object]]


def ask(transport: Transport, message: str, questions: Mapping[str, object]) -> Decision:
    started = time.monotonic()
    try:
        payload = transport({"model": MODEL, "state": {"current_message": message}, "questions": dict(questions)})
        answers = payload["answers"]
        tokens = int(payload["usage"]["input_tokens"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise JevError("jev request failed") from exc
    if payload.get("model") != MODEL or not isinstance(answers, Mapping) or set(answers) != set(questions):
        raise JevError("unexpected jev response")
    return Decision(answers, tokens, time.monotonic() - started)


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise JevError("invalid jev probability")
    return float(value)


def group_questions(groups: Mapping[str, tuple[str, object]]) -> dict[str, object]:
    return {
        f"group_{name.replace('-', '_')}": {
            "type": "noul",
            "instructions": f"Does current_message need this group of Actions? {description}",
        }
        for name, (description, _actions) in groups.items()
    }


def selected_groups(decision: Decision, groups: Mapping[str, object]) -> dict[str, float]:
    """Every group's yes-probability; the caller keeps those at or above the threshold."""
    probabilities = {}
    for name in groups:
        answer = decision.answers[f"group_{name.replace('-', '_')}"]
        if not isinstance(answer, Mapping) or answer.get("type") != "noul":
            raise JevError("invalid jev noul answer")
        probabilities[name] = _probability(answer.get("noul"))
    return probabilities


ROUTE_QUESTION = {
    "route": {
        "type": "choice",
        "instructions": "How hard is the request in current_message for an assistant that runs Actions in the user's "
        "services?",
        "criteria": ROUTES,
    }
}


def route(decision: Decision) -> tuple[str, float]:
    answer = decision.answers["route"]
    if not isinstance(answer, Mapping) or answer.get("type") != "choice" or answer.get("choice") not in ROUTES:
        raise JevError("invalid jev choice answer")
    return str(answer["choice"]), _probability(answer.get("confidence"))


def simple(decision: Decision) -> bool:
    choice, confidence = route(decision)
    return choice in SIMPLE and confidence >= ROUTE_THRESHOLD


def http_transport(api_key: str, client: object | None = None, timeout: float = 5.0) -> Transport:
    """Send a decision request over HTTPS; ``client`` is an ``httpx.Client`` (tests pass a mock transport)."""
    import httpx

    session = client if client is not None else httpx.Client(timeout=timeout)

    def send(body: dict[str, object]) -> dict[str, object]:
        response = session.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code != 200:
            raise OSError(f"jev status {response.status_code}")
        return json.loads(response.content)

    return send

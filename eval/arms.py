"""Engineering arms of the gpt-6-luna experiment: experiment-only, loaded by the journey driver (ADR-0094).

Cumulative arms over Luna at effort low:

- A: the baseline contracts and Team path.
- B: A with the arm-B contracts of ``eval.contracts``.
- C: B with deterministic Team-side argument checks before execution. A write by an Assistant the request does not
  match, a value that is not of its declared kind, or a lookup filter value found neither in the user's words nor in
  an earlier result is refused with a short closed correction, and nothing runs.
- D: C with a scoped working set: only the Assistants a deterministic ranking of the message against their declared
  search terms selects, at most ``WORKING_SET_LIMIT``, or every Assistant when none matches.
- E: D with a cascade. On a risk signal before any write ran, the turn restarts on the escalation model; reads have
  no effect, so restarting replays nothing that changed state. After a write the turn stays on Luna.

Nothing here is production runtime: the driver applies it in its own Team process. This module uses only the
standard library.
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from eval.contracts import DATE, HOSTNAME, SEARCH_TERMS, TIME
from eval.corpus import LOCALES
from eval.fixtures import Action
from eval.large_api_arms import GROUP_TERMS

WORKING_SET_LIMIT = 4
ESCALATION = ("anthropic", "claude-sonnet-5-5")
IPV4_RE = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])")
DOMAIN_RE = re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b")
TIME_RE = re.compile(r"\b[0-9]{1,2}:[0-9]{2}\b")
HOSTNAME_RE = re.compile(r"[a-z0-9_*@.-]+\Z")
_LIST_KEYS = ("records", "tasks", "contacts", "results", "events", "zones", "result")
# The large-API edge Assistant's terms are its resource groups' terms, in every locale.
TERMS = {
    **SEARCH_TERMS,
    "edge": dict.fromkeys(LOCALES, tuple(term for terms in GROUP_TERMS.values() for term in terms)),
}


@dataclass(frozen=True, slots=True)
class Arm:
    """One arm: its contract set, Team-side checks, Assistant working set, large-API exposure, and model routing.

    ``contracts`` is ``a`` or ``b`` (precision corpus), ``large`` or ``large-tasks`` (large-API stratum).
    ``exposure`` is ``scope`` (every Assistant in scope), ``namespaces`` (provider tool search over resource-group
    namespaces), ``groups`` (deterministic group ranking), or ``jev-groups`` (Jev group selection, falling back to
    ``fallback`` when no group is confident). ``routing`` ``jev`` decides Luna or the escalation model before the
    turn; ``model`` ``sonnet`` runs the whole arm on the escalation model as a reference.
    """

    contracts: str
    checks: bool = False
    working_set: bool = False
    cascade: bool = False
    exposure: str = "scope"
    routing: str = "none"
    model: str = "luna"
    fallback: str = "scope"


ARMS = {
    "A": Arm("a"),
    "B": Arm("b"),
    "C": Arm("b", checks=True),
    "D": Arm("b", checks=True, working_set=True),
    "E": Arm("b", checks=True, working_set=True, cascade=True),
    "EJ": Arm("b", checks=True, working_set=True, routing="jev"),
    "S": Arm("a", model="sonnet"),
    "L1": Arm("large"),
    "L2": Arm("large", exposure="namespaces"),
    "L3": Arm("large", exposure="groups"),
    "L4": Arm("large-tasks"),
    "L5-namespaces": Arm("large", checks=True, cascade=True, exposure="namespaces"),
    "L5-groups": Arm("large", checks=True, cascade=True, exposure="groups"),
    "L5-tasks": Arm("large-tasks", checks=True, cascade=True),
    "L6-scope": Arm("large", exposure="jev-groups", fallback="scope"),
    "L6-namespaces": Arm("large", exposure="jev-groups", fallback="namespaces"),
    "LJ-namespaces": Arm("large", checks=True, exposure="namespaces", routing="jev"),
    "LJ-groups": Arm("large", checks=True, exposure="groups", routing="jev"),
    "LJ-tasks": Arm("large-tasks", checks=True, routing="jev"),
    "LS": Arm("large", model="sonnet"),
}


def undispatched() -> dict[str, object]:
    """The arm fields of an attempt that never dispatched, so a stopped record has the shape of a run one.

    Exposure signals are None (never computed), counts are zero, and nothing escalated.
    """
    return {
        "dispatched": False,
        "exposed": None,
        "exposed_actions": None,
        "recall": None,
        "selection_fallback": None,
        "refusals": 0,
        "escalated": False,
        "escalation_signal": None,
        "escalation_blocked": False,
        "models_used": [],
        "jev_usd": 0.0,
        "jev_seconds": 0.0,
        "route": None,
        "route_confidence": None,
    }


def relevance(message: str, locale: str, scope: Sequence[str]) -> dict[str, int]:
    """Each Assistant's matched declared terms, plus a domain or address cue for DNS and a date or time for Calendar."""
    text = message.casefold()
    scores = {name: sum(term.casefold() in text for term in TERMS[name][locale]) for name in scope}
    if "dns" in scores and (DOMAIN_RE.search(message) or IPV4_RE.search(message)):
        scores["dns"] += 1
    if "calendar" in scores and (DATE_RE.search(message) or TIME_RE.search(message)):
        scores["calendar"] += 1
    return scores


def working_set(message: str, locale: str, scope: Sequence[str]) -> tuple[str, ...]:
    scores = relevance(message, locale, scope)
    ranked = sorted((name for name in scope if scores[name] > 0), key=lambda name: (-scores[name], scope.index(name)))
    return tuple(ranked[:WORKING_SET_LIMIT]) or tuple(scope)


def valid_kind(kind: str, value: object) -> bool:
    text = str(value).strip().casefold()
    if kind == HOSTNAME:
        return bool(HOSTNAME_RE.fullmatch(text)) and any(char.isalpha() or char == "_" for char in text)
    if kind == DATE:
        try:
            datetime.date.fromisoformat(text)
        except ValueError:
            return False
        return bool(DATE_RE.fullmatch(text))
    if kind == TIME:
        hours, _, minutes = text.partition(":")
        return bool(re.fullmatch(r"[0-9]{2}:[0-9]{2}", text)) and int(hours) < 24 and int(minutes) < 60
    raise ValueError("unknown value kind")


def _refusal(code: str, detail: str, **fields: str) -> dict[str, object]:
    return {"error": "refused", "code": code, **fields, "detail": detail, "changed": False}


def _empty(result: Mapping[str, object]) -> bool:
    return any(isinstance(result.get(key), list) and not result[key] for key in _LIST_KEYS)


@dataclass
class TurnGuard:
    """One turn's deterministic checks (arm C) and risk signals (arm E), from the message and earlier results."""

    message: str
    locale: str
    scope: tuple[str, ...]
    scores: dict[str, int] = field(init=False)
    results: list[str] = field(default_factory=list)
    writes: int = 0
    refusals: int = 0
    signal: str | None = None

    def __post_init__(self) -> None:
        self.scores = relevance(self.message, self.locale, self.scope)

    def _sourced(self, value: object) -> bool:
        text = str(value).strip().casefold()
        return not text or text in self.message.casefold() or any(text in result for result in self.results)

    def check(self, assistant_id: str, action: Action, arguments: Mapping[str, object]) -> dict[str, object] | None:
        """A closed correction for a refused request, or None to run it."""
        if action.writes and self.scores.get(assistant_id, 0) == 0 and max(self.scores.values(), default=0) > 0:
            return self._refused(
                _refusal("out-of-scope-write", "This Assistant does not match the request; nothing changed.")
            )
        for name, kind in action.kinds:
            if name in arguments and not valid_kind(kind, arguments[name]):
                detail = f"{name} must be a {kind}; nothing ran."
                return self._refused(_refusal("invalid-argument", detail, field=name, expected=kind))
        for name in action.filters:
            if name in arguments and not self._sourced(arguments[name]):
                detail = f"The {name} value is not in the request or an earlier result; omit it to list everything."
                return self._refused(_refusal("unsupported-filter", detail, field=name))
        return None

    def _refused(self, refusal: dict[str, object]) -> dict[str, object]:
        self.refusals += 1
        self._risk("argument-correction")
        return refusal

    def _risk(self, signal: str) -> None:
        if self.signal is None:
            self.signal = signal

    def pending(self, assistant_id: str, action: Action, arguments: Mapping[str, object]) -> None:
        """Note the risk of a write about to run: a multi-Assistant task, or a written value with no source."""
        if not action.writes:
            return
        if sum(score > 0 for score in self.scores.values()) >= 2:
            self._risk("multi-assistant-write")
        content = arguments.get("content")
        if content is not None and IPV4_RE.fullmatch(str(content).strip()) and not self._sourced(content):
            self._risk("unsourced-write-value")

    def ran(self, action: Action, result: Mapping[str, object]) -> None:
        self.writes += action.writes
        self.results.append(json.dumps(result, ensure_ascii=False, default=str).casefold())
        if not action.writes and _empty(result):
            self._risk("empty-lookup")
        contacts = result.get("contacts")
        if isinstance(contacts, list) and len(contacts) > 1:
            self._risk("ambiguous-lookup")

    def escalation(self) -> str | None:
        """The signal that escalates now: one exists and no write has run yet."""
        return self.signal if self.signal is not None and self.writes == 0 else None

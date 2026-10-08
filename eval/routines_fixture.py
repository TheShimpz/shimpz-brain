"""The Routine eval's fixed world: the reference Assistant's fixture results and the attempt's constants."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """A standard-library Brain eval helper loaded by path, as the umbrella journey driver loads it."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __package__:
    # In Brain, the one cost module its evaluation ceiling uses too, so the shared ledger's refusal is one class.
    from eval import cost as eval_cost
    from eval import stats as eval_stats
else:  # run as a script from Team, which cannot import Brain's eval package
    eval_cost = _load("routine_eval_cost", BRAIN / "eval" / "cost.py")
    eval_stats = _load("routine_eval_stats", BRAIN / "eval" / "stats.py")


ASSISTANT = "shimpz-cloudflare"


PRINCIPAL = "a" * 32


ROUTINE_KEY = "e" * 64


SHIMPZ = "023e105f4ecef8ad9ca31a8372d0c353"


MOVED = "7b2f31aa2c0b4c8d9e1f203142536475"


TWIN = "5d41402abc4b2a76b9719d911017c592"


ACCOUNT = {"id": "f" * 32, "name": "Owner"}


OTHER_ZONES = (("9a7806061c88ada191ed06f989cc3dac", "example.com"), ("1b3f0c5e2a9d47e8b6c1d0f2a3b4c5d6", "other.org"))


EXAMPLE = OTHER_ZONES[0][0]


ZONE_NAMES = {SHIMPZ: "shimpz.com", EXAMPLE: "example.com"}


# Every zone name list-zones answers with when no two zones share a name.
ZONE_SET = {"example.com", "other.org", "shimpz.com"}


TIMEZONE = "America/Sao_Paulo"


# The labels Admin's Portuguese interface composes a clarification answer with (admin/frontend/src/lib/messages.js).
QUESTION_LABEL, ANSWER_LABEL = "Pergunta", "Resposta"


MAX_CONVERSATION_TEXT = 512


MAX_SENDS = 6


# The owner's release gate: every case passes on every one of 30 attempts per shipped model.
ATTEMPTS = 30


BUDGET_USD = 3.0


# The output limit of every provider request, which the evaluation ceiling reserves and clamps to: about 25 times the
# largest output a Routine turn was measured to use (170 tokens on 2026-10-07). A response that reaches it is counted.
CALL_OUTPUT_TOKENS = 4_000


MODELS = (("openai", "gpt-6-luna"), ("anthropic", "claude-sonnet-5-5"))


def _pagination(count: int) -> dict[str, int]:
    return {"page": 1, "per_page": 25, "count": count, "total_count": count, "total_pages": 1}


def zones(shimpz: str = SHIMPZ, *, twin: bool = False) -> dict[str, object]:
    """list-zones' result: shimpz.com among others, under ``shimpz``; with ``twin``, a second zone of that name."""
    named = [*OTHER_ZONES, (shimpz, "shimpz.com"), *([(TWIN, "shimpz.com")] if twin else [])]
    items = [
        {"id": zone, "name": name, "status": "active", "type": "full", "paused": False, "account": ACCOUNT}
        for zone, name in named
    ]
    return {"zones": items, "pagination": _pagination(len(items))}


# The TXT record a changing Routine updates: its current content is never a date a run writes.
VERIFY = {"id": "b" * 32, "type": "TXT", "name": "verify.shimpz.com", "content": "verified-2026", "ttl": 300}


SHIMPZ_RECORDS = (
    {"id": "e" * 32, "type": "A", "name": "shimpz.com", "content": "192.0.2.1", "ttl": 300},
    {"id": "d" * 32, "type": "MX", "name": "shimpz.com", "content": "mail.shimpz.com", "ttl": 3600},
    VERIFY,
)


EXAMPLE_RECORDS = ({"id": "c" * 32, "type": "A", "name": "example.com", "content": "192.0.2.2", "ttl": 300},)


ZONE_RECORDS = {SHIMPZ: SHIMPZ_RECORDS, MOVED: SHIMPZ_RECORDS, EXAMPLE: EXAMPLE_RECORDS}


def records(zone: str) -> dict[str, object]:
    """list-dns-records' result for one zone: shimpz.com's and example.com's own records, nothing for another."""
    found = [{**item, "proxied": False, "proxiable": False} for item in ZONE_RECORDS.get(zone, ())]
    return {"records": found, "pagination": _pagination(len(found))}


def zone(zone_id: str) -> dict[str, object] | None:
    """get-zone's result: the zone list-zones names under that id; None for an id it never names."""
    return next((item for item in zones(twin=True)["zones"] + zones(MOVED)["zones"] if item["id"] == zone_id), None)


def record(zone: str, record_id: str) -> dict[str, object] | None:
    """get-dns-record's result: the record list-dns-records lists under that id in that zone; None for another."""
    return next((item for item in records(zone)["records"] if item["id"] == record_id), None)


# The Actions that change DNS: each asks its own authorization, through the real Assistant, before it changes anything.
CHANGING = ("replace-dns-record", "ensure-dns-record", "delete-dns-record")


def changed(action: str, payload: dict[str, object]) -> dict[str, object]:
    """The simulated provider's result of one authorized change, in the shape the Action's output schema admits."""
    if action == "delete-dns-record":
        return {"record_id": payload["record_id"], "deleted": True}
    written = {key: payload[key] for key in ("name", "content", "ttl", "proxied")}
    written |= {"type": payload["record_type"], "proxiable": True}
    if action == "replace-dns-record":
        return {"id": payload["record_id"], **written}
    return {"record": {"id": "a" * 32, **written}, "created": True}


def _failure() -> Exception:
    """The problem Team raises for a failed Action call; imported here, since only Team's environment runs a call."""
    from http import HTTPStatus

    from local.errors import ApiProblemError

    return ApiProblemError(
        HTTPStatus.BAD_GATEWAY, "the simulated provider failed this call", code="assistant-rpc-failed"
    )


@dataclasses.dataclass
class Fixture:
    """The real Cloudflare Assistant's simulated provider: what its reads answer, and every call it saw, in order."""

    shimpz: str = SHIMPZ
    twin: bool = False
    calls: list[tuple[str, dict[str, object]]] = dataclasses.field(default_factory=list)
    # Each change the provider performed, only ever after the person authorized it.
    performed: list[tuple[str, dict[str, object]]] = dataclasses.field(default_factory=list)
    # The request the real Assistant's Action makes before a change, as Team admits it: set by the driver.
    ask: object = None
    # With --trace-dir, the attempt's ordered transcript, which every send, Brain turn, and Action call joins.
    events: list[dict[str, object]] | None = None

    def invoke(self, _team, _assistant, action, payload, evidence) -> dict[str, object]:
        self.calls.append((action, dict(payload)))
        if action == "list-zones":
            result = zones(self.shimpz, twin=self.twin)
        elif action == "list-dns-records":
            result = records(str(payload.get("zone_id")))
        elif action == "get-zone":
            result = zone(str(payload.get("zone_id")))
        elif action == "get-dns-record":
            result = record(str(payload.get("zone_id")), str(payload.get("record_id")))
        elif action in CHANGING:
            result = self._change(action, payload, evidence)
        else:
            result = None
        if result is None:
            # Any other Action, or an id the provider never named, fails as Team's own Action failure, which ends
            # the attempt as a miss: a change is never simulated as done.
            self.note({"kind": "action", "action": action, "input": dict(payload), "result": None})
            raise _failure()
        answer = {"result": result}
        if payload.get("page", 1) != 1:
            # Every list fits its first page, as its pagination says, so a later page is empty.
            key = "zones" if action == "list-zones" else "records"
            answer = {"result": {key: [], "pagination": {**_pagination(0), "page": payload["page"], "total_count": 0}}}
        self.note({"kind": "action", "action": action, "input": dict(payload), "result": answer["result"]})
        return answer

    def _change(self, action: str, payload: dict[str, object], evidence) -> dict[str, object]:
        """A change runs only with the person's approval of the exact request its real Action makes.

        Without an answer the Action asks; an Action that asks nothing, or an answer to another request, is a miss.
        """
        request = self.ask(action, payload, evidence)
        responses = evidence.transcript.responses
        if request is not None and not responses:
            self.note({"kind": "action", "action": action, "input": dict(payload), "result": "requested"})
            from action.human import HumanRequestSuspensionError

            raise HumanRequestSuspensionError(request)
        exact = (request.kind, request.ordinal, request.fingerprint, True) if request is not None else None
        if exact is None or [(item.kind, item.ordinal, item.fingerprint, item.value) for item in responses] != [exact]:
            self.note({"kind": "action", "action": action, "input": dict(payload), "result": None})
            raise _failure()
        self.performed.append((action, dict(payload)))
        return changed(action, payload)

    def note(self, event: dict[str, object]) -> None:
        if self.events is not None:
            self.events.append(event)

    def listed(self, zone: str) -> bool:
        return any(action == "list-dns-records" and payload.get("zone_id") == zone for action, payload in self.calls)

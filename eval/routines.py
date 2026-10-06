"""Evaluate recorded Team Routines (ADR-0101) through the real chat agent and Team's real record and confirm path.

Brain and Team each define top-level ``routine`` and ``protocol`` packages, so they never share one interpreter: run
with ``--serve-brain`` from Brain, this file serves Brain's real runtime API on loopback in its own process, with an
in-memory checkpoint and a fresh bearer it writes to a new owner-only token file, and otherwise it drives the eval under
Team's environment. Its Team is an in-process Local controller built by Team's test harness, its
Brain is Team's own ``BrainRuntimeClient`` over that loopback HTTP, and its one Assistant is the reference
Cloudflare-shaped Assistant whose Action calls a fixture answers with results its reviewed output schemas admit. Every
send runs as the person, with the fresh request identity an Admin send carries, and a clarification is answered exactly
as Admin composes it.

Validate offline from Team, with Brain beside it::

    cd teams && PYTHONPATH=.:tests uv run --frozen --python 3.14 python ../brain/eval/routines.py --validate

Run live: start the Brain from Brain, which writes a fresh bearer to the new token file, then the driver from Team::

    cd brain && uv run --frozen --python 3.14 python -m eval.routines --serve-brain --brain-port 8791
    --token-file "$DIR/token" &
    cd teams && PYTHONPATH=.:tests uv run --frozen --python 3.14 python ../brain/eval/routines.py --brain-port 8791
    --token-file "$DIR/token" --openai-key-file ../.gpt-key --anthropic-key-file ../.claude-key

Strata, each ``--attempts`` (30) times per model, fixed before sampling: the owner's four turns; a one-send daily
list; two named zones at once (multi-zone); a zone named only in an earlier send; a request with no schedule; the
reversed-order bait, primed (the zone was looked up in an earlier send) and unprimed (its id is known only from an
earlier reply outside the span); two zones with the same name at creation, which must be asked about; and intervals of
30 and 5 seconds, whose card keeps the exact gap the person last stated with cap ceil(86400 / gap). The driver answers
the agent's questions and Team's recoverable Routine questions as the person would. A stratum passes when the card
binds every zone the person meant through a reference (the twin stratum: the person's chosen zone by its id), carries
the schedule the person stated, "Criar rotina" creates the Routine, and one replay through Team's real claim and run
lists those zones' records and shows one of them. After the first passing owner attempt, two replay variants run with
no model: shimpz.com under a new zone id, and two zones named shimpz.com, which must never dispatch list-dns-records.
The gate exits non-zero unless every stratum of every shipped model passed every attempt and both variants held.
Output holds stratum ids, pass counts, Wilson 95% bounds, closed miss reasons with root causes, the questions asked,
the schedules seen, and the estimated cost; never a message, a reply, or a key (only ``--trace-dir`` writes messages).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import importlib.util
import json
import os
import re
import secrets
import socket
import sys
import tempfile
import time
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest import mock

BRAIN = Path(__file__).resolve().parents[1]
TEAMS = BRAIN.parent / "teams-det"
CONTRACT = TEAMS / "tests" / "fixtures" / "reference-assistant" / "shimpz.contract.json"


def _load(name: str, path: Path):
    """A standard-library Brain eval helper loaded by path, as the umbrella journey driver loads it."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
TIMEZONE = "America/Sao_Paulo"
# The labels Admin's Portuguese interface composes a clarification answer with (admin/frontend/src/lib/messages.js).
QUESTION_LABEL, ANSWER_LABEL = "Pergunta", "Resposta"
MAX_CONVERSATION_TEXT = 512
MAX_SENDS = 6
# The owner's release gate: every case passes on every one of 30 attempts per shipped model.
ATTEMPTS = 30
BUDGET_USD = 3.0
# One attempt's conservative reservation: this many Brain calls, each at most this much input and output.
CALLS_PER_ATTEMPT = 10
CALL_INPUT_TOKENS = 30_000
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


SHIMPZ_RECORDS = (
    {"id": "e" * 32, "type": "A", "name": "shimpz.com", "content": "192.0.2.1", "ttl": 300},
    {"id": "d" * 32, "type": "MX", "name": "shimpz.com", "content": "mail.shimpz.com", "ttl": 3600},
)


EXAMPLE_RECORDS = ({"id": "c" * 32, "type": "A", "name": "example.com", "content": "192.0.2.2", "ttl": 300},)
ZONE_RECORDS = {SHIMPZ: SHIMPZ_RECORDS, MOVED: SHIMPZ_RECORDS, EXAMPLE: EXAMPLE_RECORDS}


def records(zone: str) -> dict[str, object]:
    """list-dns-records' result for one zone: shimpz.com's and example.com's own records, nothing for another."""
    found = [{**item, "proxied": False, "proxiable": False} for item in ZONE_RECORDS.get(zone, ())]
    return {"records": found, "pagination": _pagination(len(found))}


@dataclasses.dataclass
class Fixture:
    """The Cloudflare-shaped Assistant: what list-zones answers, and every call it saw, in order."""

    shimpz: str = SHIMPZ
    twin: bool = False
    calls: list[tuple[str, dict[str, object]]] = dataclasses.field(default_factory=list)
    # With --trace-dir, the attempt's ordered transcript, which every send, Brain turn, and Action call joins.
    events: list[dict[str, object]] | None = None

    def invoke(self, _team, _assistant, action, payload, _evidence) -> dict[str, object]:
        self.calls.append((action, dict(payload)))
        if action == "list-zones":
            answer = {"result": zones(self.shimpz, twin=self.twin)}
        else:
            answer = {"result": records(str(payload.get("zone_id")))}
        self.note({"kind": "action", "action": action, "input": dict(payload), "result": answer["result"]})
        return answer

    def note(self, event: dict[str, object]) -> None:
        if self.events is not None:
            self.events.append(event)

    def listed(self, zone: str) -> bool:
        return any(action == "list-dns-records" and payload.get("zone_id") == zone for action, payload in self.calls)


def validate() -> None:
    """Offline: every fixture result validates against the reference Assistant's reviewed output schemas."""
    from jsonschema import Draft202012Validator

    actions = {item["id"]: item for item in json.loads(CONTRACT.read_text(encoding="utf-8"))["actions"]}
    for value in (zones(), zones(MOVED), zones(twin=True)):
        Draft202012Validator(actions["list-zones"]["output_schema"]).validate(value)
    for zone in (SHIMPZ, MOVED, TWIN, EXAMPLE):
        Draft202012Validator(actions["list-dns-records"]["output_schema"]).validate(records(zone))
    if len({case for case, _play in CASES}) != len(CASES):
        raise SystemExit("case ids are not unique")


@dataclasses.dataclass
class Meter:
    """The Brain usage Team's client reported during one attempt, summed in the eval's own counts."""

    usage: object = None

    def add(self, counts) -> None:
        current = eval_cost.Usage.of(dict(counts))
        self.usage = current if self.usage is None else self.usage + current


def serve_brain(port: int, token_file: Path) -> int:
    """Brain's real runtime API on loopback with an in-memory checkpoint, run from Brain's own environment."""
    import agent_runtime
    import runtime_api
    import uvicorn
    from langgraph.checkpoint.memory import InMemorySaver

    token = secrets.token_urlsafe(32)
    descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token)
    app = runtime_api.create_app(runtime=agent_runtime.AgentRuntime(InMemorySaver()), token_reader=lambda: token)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    return 0


def brain_served(port: int) -> str:
    """The loopback URL of the Brain ``--serve-brain`` serves, once it accepts connections."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=1):
            return f"http://127.0.0.1:{port}"
        time.sleep(0.5)
    raise SystemExit("the Brain runtime is not serving on that port")


@dataclasses.dataclass
class Outcome:
    reason: str | None = None
    schedule: dict[str, object] | None = None
    # A miss's structural root cause, from the shape of the attempt's sends; never message text.
    cause: str | None = None
    # Every recoverable question Team asked during the attempt, in order.
    questions: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.reason is None


@dataclasses.dataclass
class Team:
    """One in-process Team's chat service and the Team modules the eval drives it with."""

    service: object
    modules: object


@dataclasses.dataclass
class Person:
    """What the person means and how they answer: how often, the zone they mean, and what they have said so far."""

    frequency: str
    # A continuous Routine's interval in seconds as the person last stated it; None for a calendar one.
    gap: int | None = None
    # The owner's topics not given yet, answered in this order when a question matches none of them.
    pending: list[str] = dataclasses.field(default_factory=list)
    # Whether the person has stated how often yet, in a send or an answer.
    stated: bool = True
    zone: str = SHIMPZ
    # Whether two zones share the name, so a question about them gets the person's zone by its id.
    choosing: bool = False
    chose: bool = False

    def says(self, topic: str) -> str:
        return {
            "work": "Listar registros DNS",
            "frequency": self.frequency,
            "zone": "shimpz.com",
            "timezone": TIMEZONE,
        }[topic]


class Attempt:
    """One isolated Team with the reference Assistant, its own Brain thread, and the person's sends."""

    def __init__(self, team: Team, fixture: Fixture, provider: str, key: str) -> None:
        self.team = team
        self.fixture = fixture
        self.provider = provider
        self.key = key
        self.person = Person("A cada hora")
        self.conversation: list[dict[str, object]] = []
        # Each send's response kind and the Actions it called, in order; and every Team question asked.
        self.sends: list[tuple[str, tuple[str, ...]]] = []
        self.questions: list[str] = []

    def _person(self):
        audit = self.team.modules.local_audit
        return audit.bind_request_principal(audit.AuditPrincipal(PRINCIPAL, "human"))

    def send(self, message: str) -> dict[str, object]:
        body = {
            "message": message,
            "files": [],
            "assistant_ids": [ASSISTANT],
            "conversation": list(self.conversation[-8:]),
            "locale": "pt",
            "request": {"issued_at": int(time.time()), "nonce": secrets.token_hex(16)},
            "timezone": TIMEZONE,
        }
        before = len(self.fixture.calls)
        with self._person():
            response = self.team.service.chat("team_1", body, self.provider, self.key)
        actions = tuple(action for action, _payload in self.fixture.calls[before:])
        self.sends.append((_kind(response), actions))
        self.fixture.note(
            {"kind": "send", "message": message, "conversation": body["conversation"], "response": response}
        )
        self.remember("user", message)
        self.remember("assistant", str(response.get("reply", "")))
        return response

    def remember(self, role: str, text: str) -> None:
        """One history entry as Admin keeps it: trimmed, empty skipped, and a long one cut to its head and tail."""
        text = unicodedata.normalize("NFC", text).strip()
        if not text:
            return
        cut = len(text) > MAX_CONVERSATION_TEXT
        if cut:
            head = (MAX_CONVERSATION_TEXT - 1) // 2
            text = f"{text[:head]}…{text[-(MAX_CONVERSATION_TEXT - 1 - head) :]}"
        self.conversation.append({"role": role, "text": text, "truncated": cut})

    def confirm(self, proposal_id: str) -> dict[str, object]:
        with self._person():
            return self.team.service.confirm_routine_proposal("team_1", proposal_id)

    def replay(self) -> tuple[str, object]:
        """One run of the Team's Routine now, through Team's real claim and run: its status and its last notice."""
        modules, service = self.team.modules, self.team.service
        now = int(time.time())

        def requested(state):
            routines = tuple(dataclasses.replace(item, run_requested=now) for item in state.routines)
            return dataclasses.replace(state, routines=routines), None

        # As a person's Rodar would: the Routine starts now instead of 30 seconds after it became durable.
        service.routine_store.update("team_1", requested)
        claim = service.claim_routine_run()
        if claim is None:
            return "unclaimed", None
        lease = modules.record.lease_sha256(claim["lease_token"])
        evidence = modules.local_authority.RoutineEvidence(ROUTINE_KEY, lease, "a" * 32, 0)
        claimed = (claim["revision"], claim["plan_digest"], claim["mode"])
        result = service.run_routine("team_1", claim["run_id"], evidence, claimed, (self.provider, ""))
        notices = service.routine_store.load("team_1").notices
        notice = notices[-1] if notices else None
        shown = None if notice is None else {"outcome": notice.outcome, "detail": notice.detail}
        self.fixture.note({"kind": "replay", "status": result["status"], "notice": shown})
        return result["status"], notice


def _kind(response: dict[str, object]) -> str:
    for member, kind in (
        ("routine_proposal", "card"),
        ("routine_refusal", "refusal"),
        ("routine_question", "question"),
    ):
        if member in response:
            return kind
    return "clarification" if response.get("clarification") is not None else "prose"


# What a question asks, by its own words only: each topic the person can answer, and the result's delivery, which
# they leave to the recommended option. A bare "which" names the zone only when no topic matched.
_HINTS: tuple[tuple[str | None, tuple[str, ...]], ...] = (
    ("work", ("trabalho", "o que", "tarefa", "repita", "repetir", "fazer")),
    (
        "frequency",
        (
            *("frequ", "com que", "quando", "intervalo", "periodicidade", "horário", "horario", "quantas vezes"),
            *("segundo", "minuto", "execuç", "limite"),
        ),
    ),
    ("zone", ("zona", "domínio", "dominio", "escopo")),
    ("timezone", ("fuso", "timezone", "utc")),
    (None, ("receber", "resultado", "mostrar", "notific", "mudan")),
)
_TOPICS = ("work", "frequency", "zone", "timezone")
_WHICH = ("qual", "quais")


def _matched(text: str) -> list[str | None]:
    return [topic for topic, words in _HINTS if any(word in text for word in words)]


def _parts(question: str, person: Person) -> tuple[list[str], bool]:
    """The person's answers to every topic the question asks, in their own order, and whether a part none covers.

    The sentences that ask come first and the surrounding text counts only when they match nothing; with nothing
    matched, the owner's next topic not given yet answers it.
    """
    # "Fuso horário" asks the timezone, never the hour.
    whole = question.casefold().replace("fuso horário", "fuso").replace("fuso horario", "fuso")
    asked = " ".join(re.findall(r"[^.?!]*\?", whole)) or whole
    matched = _matched(asked) or _matched(whole)
    uncovered = None in matched
    wanted = [topic for topic in matched if topic is not None]
    if not wanted and any(word in asked for word in _WHICH):
        wanted = ["zone"]
    if not wanted and not uncovered and person.pending:
        wanted = [person.pending[0]]
    for topic in wanted:
        if topic in person.pending:
            person.pending.remove(topic)
    person.stated = person.stated or "frequency" in wanted
    return [person.says(topic) for topic in _TOPICS if topic in wanted], uncovered


def _zone_choice(person: Person, text: str, options: list[dict[str, str]]) -> str | None:
    """With two zones of one name, a question about them gets the option naming the person's zone, or its id."""
    labels = [f"{item['label']} {item['description']}" for item in options]
    named = TWIN in text or any(TWIN in label for label in labels)
    if not person.choosing or not (named or re.search(r"mesmo nome|duas zonas", text.casefold())):
        return None
    person.chose = True
    return next((item["label"] for item in options if person.zone in item["label"]), f"A zona de id {person.zone}")


def _answer(person: Person, response: dict[str, object]) -> tuple[str | None, bool]:
    """The person's answer to the agent's own question, and whether Admin composes it as a clarification answer."""
    clarification = response.get("clarification")
    text = str(clarification["question"] if clarification else response.get("reply", ""))
    options = clarification["options"] if clarification else []
    chosen = _zone_choice(person, text, options)
    if chosen is not None:
        return chosen, clarification is not None
    parts, uncovered = _parts(text, person)
    if clarification is None:
        return "; ".join(parts) or None, False
    recommended = options[clarification["default_index"]]["label"]
    # The person picks their own zone's option when the agent offers one, as one press sends it.
    own = next((item["label"] for item in options if "Sao_Paulo" in item["label"] or "Brasília" in item["label"]), None)
    parts = [own if part == TIMEZONE and own else part for part in parts]
    return "; ".join([*parts, *([recommended] if uncovered or not parts else [])]), True


def _target(person: Person, options: list[object]) -> str | None:
    """A binding question's option whose target is the person's zone: its own text, or the zone's id."""
    found = next((item for item in options if person.zone in json.dumps(item)), None)
    if found is None:
        return None
    person.chose = True
    return found if isinstance(found, str) else person.zone


def _team_answer(person: Person, question: dict[str, object]) -> str | None:
    """The person's answer to one of Team's recoverable Routine questions, as they would type it; None if none fits."""
    code = question["code"]
    if code == "routine-schedule-unstated":
        person.stated = True
        return person.frequency
    if code == "routine-binding-ambiguous":
        return _target(person, list(question.get("options") or []))
    if code == "routine-interval-over-budget" and isinstance(question.get("value"), int):
        # The person takes the shortest interval Team says fits.
        person.gap = question["value"]
        person.frequency = f"A cada {person.gap} segundos"
        return person.frequency
    return _FIXED_ANSWERS.get(code)


# The person's own words for the questions whose answer never depends on the attempt.
_FIXED_ANSWERS = {
    "routine-binding-unsourced": "pode buscar de novo",
    "routine-work-rerun": "pode buscar de novo",
    "routine-work-split": "faça tudo de novo",
    "routine-timezone-unstated": TIMEZONE,
    "routine-timezone-ambiguous": TIMEZONE,
}


def _conversation_turns(attempt: Attempt, first: str) -> dict[str, object]:
    """Send and answer until a turn records, the person has nothing left to say, or the sends run out.

    The agent's clarification gets every part the person can answer, composed as Admin composes it, and Admin's
    preselected recommended option for a part they leave open; a question in prose gets a typed answer; Team's
    recoverable Routine question gets the answer the person would type, as an ordinary send.
    """
    message = first
    response: dict[str, object] = {}
    for _send in range(MAX_SENDS):
        response = attempt.send(message)
        if "routine_proposal" in response or "routine_refusal" in response:
            return response
        question = response.get("routine_question")
        if question is not None:
            attempt.questions.append(question["code"])
            answer, composed = _team_answer(attempt.person, question), False
        else:
            answer, composed = _answer(attempt.person, response)
        if answer is None:
            return response
        if composed:
            asked = response["clarification"]["question"]
            answer = f"{message.strip()}\n\n{QUESTION_LABEL}: {asked}\n{ANSWER_LABEL}: {answer}"
        message = answer
    return response


@dataclasses.dataclass(frozen=True)
class Stratum:
    """One creation stratum: how the person asks, who they are, the schedule they mean, and the zones they mean."""

    id: str
    play: Callable[[Attempt], dict[str, object] | Outcome]
    person: Callable[[], Person]
    schedule: Callable[[Person, dict[str, object]], bool]
    zones: tuple[str, ...] = (SHIMPZ,)
    twin: bool = False


def _selector_miss(card: dict[str, object], zones_meant: tuple[str, ...]) -> str | None:
    """Why the card does not copy each listing's zone_id from list-zones through the item named by the person."""
    positions = {step["position"]: step["action"] for step in card["steps"]}
    listings = [item for item in card["steps"] if item["action"] == "list-dns-records"]
    if len(listings) > len(zones_meant):
        # The turn acted on every zone, so the card lists them all rather than only the zones the person named.
        return "all-zones"
    if len(listings) < len(zones_meant):
        return "selector:no-zone-input"
    names = set()
    for step in listings:
        zone = next((item for item in step["inputs"] if item["member"] == "zone_id"), None)
        if zone is None:
            return "selector:no-zone-input"
        if zone["origin"] != "selector":
            return f"selector:origin-{zone['origin']}"
        if positions.get(zone["step"]) != "list-zones" or zone["where"]["member"] != "name" or zone["item"] != "/id":
            return "selector:shape"
        names.add(zone["where"]["value_json"])
    return None if names == {json.dumps(ZONE_NAMES[zone]) for zone in zones_meant} else "selector:shape"


def _twin_miss(attempt: Attempt, card: dict[str, object]) -> str | None:
    """Two zones share the name: someone asked which, and the listing takes the person's zone by its id."""
    if not attempt.person.chose:
        return "twin-unasked"
    listings = [item for item in card["steps"] if item["action"] == "list-dns-records"]
    zone = next((item for step in listings for item in step["inputs"] if item["member"] == "zone_id"), None)
    if len(listings) != 1 or zone is None:
        return "twin-binding:listings"
    return None if zone["origin"] == "request" and zone["value"] == json.dumps(SHIMPZ) else "twin-binding:value"


def _card(attempt: Attempt, response: dict[str, object], stratum: Stratum) -> Outcome:
    """The recording turn's card, judged; a miss names its first closed reason."""
    if "routine_refusal" in response:
        return Outcome(f"refused:{response['routine_refusal']['code']}")
    if "routine_question" in response:
        return Outcome(f"question:{response['routine_question']['code']}")
    card = response.get("routine_proposal")
    if card is None:
        return Outcome("no-card")
    seen = dict(card["schedule"])
    calls = attempt.fixture.calls
    binding = _twin_miss(attempt, card) if stratum.twin else _selector_miss(card, stratum.zones)
    checks = (
        ("card-invalid", attempt.team.modules.http_routine.canonical_proposal(card) == card),
        (
            "calls",
            any(action == "list-zones" for action, _payload in calls)
            and all(map(attempt.fixture.listed, stratum.zones)),
        ),
        ("invented-schedule", attempt.person.stated),
        ("schedule", stratum.schedule(attempt.person, card["schedule"])),
        (f"output:{card['output']['mode']}", card["output"]["mode"] in ("show", "changes")),
        (binding or "binding", binding is None),
    )
    failed = next((reason for reason, held in checks if not held), None)
    return Outcome(failed, seen) if failed else Outcome(None, {**seen, "proposal_id": card["proposal_id"]})


def _judged(attempt: Attempt, response: dict[str, object], stratum: Stratum) -> Outcome:
    """The card, its confirmation, and one replay through Team's real claim and run."""
    carded = _card(attempt, response, stratum)
    if not carded:
        return carded
    seen = dict(carded.schedule)
    answer = attempt.confirm(seen.pop("proposal_id"))
    if answer["status"] != "created":
        return Outcome(f"confirm:{answer['status']}", seen)
    return Outcome(_replay_miss(attempt, stratum.zones), seen)


def _replay_miss(attempt: Attempt, zones_meant: tuple[str, ...]) -> str | None:
    """Why one replay did not list the person's zones' records and show one of them from list-dns-records, or None."""
    attempt.fixture.calls.clear()
    status, notice = attempt.replay()
    if status != "done" or not all(map(attempt.fixture.listed, zones_meant)):
        return f"replay:{status}"
    output = None if notice is None else notice.detail.get("output")
    if notice is None or notice.outcome != "done" or output is None or output["state"] != "shown":
        return "replay-notice"
    (routine,) = attempt.team.service.routine_store.load("team_1").routines
    if routine.plan["steps"][output["step"] - 1]["action"] != "list-dns-records":
        return "replay-shown:step"
    if _shown_records(output["value"]) not in [_expected_records(zone) for zone in zones_meant]:
        return "replay-shown:records"
    return None


def _fields(node: dict[str, object]) -> dict[str, object]:
    return dict(node["fields"]) if node.get("kind") == "fields" else {}


def _shown_records(value: dict[str, object]) -> set[tuple[object, ...]] | None:
    """Each shown record's type, name, and content, or None when the shown result holds no records list."""
    listed = _fields(value).get("records")
    if listed is None or listed.get("kind") != "list" or listed["omitted"]:
        return None
    members = ("type", "name", "content")
    return {tuple(_fields(item).get(member, {}).get("value") for member in members) for item in listed["items"]}


def _expected_records(zone: str) -> set[tuple[object, ...]]:
    return {(item["type"], item["name"], item["content"]) for item in ZONE_RECORDS[zone]}


def _continuous(person: Person, schedule: dict[str, object]) -> bool:
    """The exact interval the person last stated, running all day: its cap is ceil(86400 / gap), never lowered."""
    gap = person.gap
    expected = {"kind": "continuous", "gap": gap, "cap": -(-86_400 // gap)} if gap else None
    return schedule == expected


def _hourly(_person: Person, schedule: dict[str, object]) -> bool:
    return schedule == {"kind": "hourly", "every": 1}


def _daily_nine(_person: Person, schedule: dict[str, object]) -> bool:
    return schedule == {"kind": "daily", "time": "09:00"}


def _recorded(response: dict[str, object]) -> bool:
    return any(member in response for member in ("routine_proposal", "routine_refusal", "routine_question"))


def _primed(attempt: Attempt, first: str, recording: str) -> dict[str, object] | Outcome:
    """An ordinary first turn that must look the zone up without recording, then the recording request."""
    response = attempt.send(first)
    if _recorded(response):
        return Outcome("recorded-unasked")
    if not any(action == "list-zones" for action, _payload in attempt.fixture.calls):
        return Outcome("first-turn-calls")
    return _conversation_turns(attempt, recording)


def _unprimed(attempt: Attempt) -> dict[str, object]:
    """The zone's id is known only from an earlier assistant reply outside this span; nothing in the span ran."""
    attempt.remember("user", "Qual é o id da zona shimpz.com?")
    attempt.remember("assistant", f"O id da zona shimpz.com é {SHIMPZ}.")
    return _conversation_turns(attempt, "A cada hora, liste os registros DNS dessa zona")


def _send(first: str) -> Callable[[Attempt], dict[str, object]]:
    return lambda attempt: _conversation_turns(attempt, first)


def _person(frequency: str, gap: int | None = None, **changes) -> Callable[[], Person]:
    return lambda: Person(frequency, gap, **changes)


STRATA = (
    Stratum(
        "owner-4-turns",
        _send("Cria uma rotina pra mim"),
        _person("A cada 30 segundos", 30, pending=list(_TOPICS), stated=False),
        _continuous,
    ),
    Stratum(
        "plain-list",
        _send("Todo dia às 9h, liste os registros DNS de shimpz.com"),
        _person("Todo dia às 9h"),
        _daily_nine,
    ),
    Stratum(
        "multi-zone",
        _send("A cada hora, liste os registros DNS de shimpz.com e de example.com"),
        _person("A cada hora"),
        _hourly,
        zones=(SHIMPZ, EXAMPLE),
    ),
    Stratum(
        "earlier-send-naming",
        lambda attempt: _primed(attempt, "Liste os registros DNS de shimpz.com", "Faça isso a cada hora"),
        _person("A cada hora"),
        _hourly,
    ),
    Stratum(
        "missing-schedule",
        _send("Cria uma rotina que liste os registros DNS de shimpz.com"),
        _person("Todo dia às 9h", stated=False),
        _daily_nine,
    ),
    Stratum(
        "reversed-order-primed",
        lambda attempt: _primed(
            attempt, "Qual é o id da zona shimpz.com?", "A cada hora, liste os registros DNS dessa zona"
        ),
        _person("A cada hora"),
        _hourly,
    ),
    Stratum("reversed-order-unprimed", _unprimed, _person("A cada hora"), _hourly),
    Stratum(
        "twin-names-at-creation",
        _send("A cada hora, liste os registros DNS de shimpz.com"),
        _person("A cada hora", choosing=True),
        _hourly,
        twin=True,
    ),
    Stratum(
        "interval-30s",
        _send("A cada 30 segundos, liste os registros DNS de shimpz.com"),
        _person("A cada 30 segundos", 30),
        _continuous,
    ),
    Stratum(
        "interval-5s",
        _send("A cada 5 segundos, liste os registros DNS de shimpz.com"),
        _person("A cada 5 segundos", 5),
        _continuous,
    ),
)


def _played(stratum: Stratum) -> Callable[[Attempt], Outcome]:
    def play(attempt: Attempt) -> Outcome:
        attempt.person = stratum.person()
        attempt.fixture.twin = stratum.twin
        response = stratum.play(attempt)
        return response if isinstance(response, Outcome) else _judged(attempt, response, stratum)

    return play


def _lookup_cause(attempt: Attempt) -> str:
    """Why the recording turn's calls give no zone reference: how its own Actions relate to the earlier sends'."""
    recorded = attempt.sends[-1][1] if attempt.sends else ()
    earlier = {action for _kind, actions in attempt.sends[:-1] for action in actions}
    if "list-dns-records" not in recorded:
        return "records-not-listed"
    if "list-zones" not in recorded:
        return "reused-earlier-id" if "list-zones" in earlier else "no-zone-lookup"
    if recorded.index("list-dns-records") < recorded.index("list-zones"):
        return "lookup-after-use"
    return "selector-unresolved"


def _cause(attempt: Attempt, reason: str) -> str:
    """A miss's structural root cause, from its closed reason and the shape of the attempt's sends."""
    if reason == "all-zones":
        return "acted-on-every-zone"
    if reason == "calls" or reason.startswith("selector"):
        return _lookup_cause(attempt)
    if reason == "schedule":
        return "frequency-changed"
    if reason == "no-card":
        return "sends-exhausted" if len(attempt.sends) >= MAX_SENDS else f"stopped-on-{attempt.sends[-1][0]}"
    if reason == "refused:routine-recording-empty":
        return "work-done-before-record" if any(actions for _kind, actions in attempt.sends[:-1]) else "no-work"
    kept = ("replay-shown", "twin-", "question:", "invented-schedule")
    return reason if reason.startswith(kept) else reason.split(":", 1)[0]


CASES: tuple[tuple[str, Callable[[Attempt], Outcome]], ...] = tuple((item.id, _played(item)) for item in STRATA)


def variants(attempt: Attempt) -> dict[str, object]:
    """After a confirmed owner Routine, with no model: a moved zone id, then two zones named shimpz.com."""
    found: dict[str, object] = {}
    attempt.fixture.shimpz, attempt.fixture.calls[:] = MOVED, []
    status, _notice = attempt.replay()
    found["moved-zone-id"] = {"status": status, "listed_new_id": attempt.fixture.listed(MOVED)}
    attempt.fixture.shimpz, attempt.fixture.twin, attempt.fixture.calls[:] = SHIMPZ, True, []
    status, notice = attempt.replay()
    found["twin-zone-names"] = {
        "status": status,
        "outcome": None if notice is None else notice.outcome,
        "reason": None if notice is None else notice.detail.get("reason"),
        "code": None if notice is None else notice.detail.get("code"),
        "held_at": [item.action for item in attempt.team.service.routine_store.load("team_1").incidents],
        "list_dns_records_dispatched": any(action == "list-dns-records" for action, _payload in attempt.fixture.calls),
    }
    return found


@dataclasses.dataclass(frozen=True)
class Modules:
    """Team's modules, imported only once Team and its test harness are on the path."""

    harness: object
    brain_client: object
    inference_config: object
    brain_usage: object
    local_app: object
    local_audit: object
    local_authority: object
    http_routine: object
    record: object


def _team_modules() -> Modules:
    sys.path[:0] = [str(TEAMS), str(TEAMS / "tests")]
    import local_controller_harness

    from inference import client as brain_client
    from inference import config as inference_config
    from inference import usage as brain_usage
    from local import app as local_app
    from local import audit as local_audit
    from local import authority as local_authority
    from protocol.http.v1 import routine as http_routine
    from routine import record

    return Modules(
        local_controller_harness,
        brain_client,
        inference_config,
        brain_usage,
        local_app,
        local_audit,
        local_authority,
        http_routine,
        record,
    )


# The turn context members a trace keeps: never the model key.
_TRACED_CONTEXT = ("thread_id", "provider", "model", "effort", "locale", "knowledge_writable", "routine_capacity")


def _turn_event(turn) -> dict[str, object]:
    """One Brain turn as Team received it: reply, requested Action calls, question, memory, and record call."""
    return {
        "kind": "brain-turn",
        "status": turn.status,
        "reply": turn.reply,
        "actions": [dataclasses.asdict(item) for item in turn.actions],
        "clarification": turn.clarification,
        "memory": list(turn.memory),
        "record": turn.routine,
    }


def _traced(client, fixture: Fixture):
    """Team's own Brain client, with every turn it starts or resumes and every turn it receives in the transcript."""
    start, resume = client.start, client.resume

    def received(turn):
        fixture.note(_turn_event(turn))
        return turn

    def traced_start(context, message, *, conversation):
        known = {name: getattr(context, name) for name in _TRACED_CONTEXT}
        fixture.note({"kind": "brain-start", "message": message, "conversation": conversation, "context": known})
        return received(start(context, message, conversation=conversation))

    def traced_resume(context, results):
        fixture.note({"kind": "brain-resume", "results": dict(results)})
        return received(resume(context, results))

    client.start, client.resume = traced_start, traced_resume
    return client


def _attempt_team(
    modules: Modules, directory: str, brain: tuple[str, Path], settings, events: list | None
) -> tuple[object, Team, Fixture]:
    """A fresh Local Team for one attempt, whose own Space gives it its own Brain thread."""
    provider, model, space = settings
    url, token_file = brain
    fixture = Fixture(events=events)
    client = modules.brain_client.BrainRuntimeClient(base_url=url, token_file=token_file)
    # The harness builds its controller as a test case does; any of its own methods names the unused test.
    case = modules.harness.LocalContractCase("_chat_controller")
    controller = case._chat_controller(directory, client if events is None else _traced(client, fixture))
    controller.assistant_lifecycle.invoke = fixture.invoke
    controller.space_id = controller.chat_turn_service.space_id = space
    controller.inference_store.save("team_1", modules.inference_config.normalize(provider, model))
    return case, Team(controller.chat_turn_service, modules), fixture


def _summary(case_id: str, passed: int, misses: list, schedules: list, costs: list, stopped: bool) -> dict:
    trials = len(costs)
    interval = eval_stats.wilson(passed, trials)
    total = sum(item.usd for item in costs)
    return {
        "id": case_id,
        "passed": passed,
        "trials": trials,
        "wilson95_lower": None if interval is None else round(interval[0], 3),
        "inconclusive": stopped,
        "misses": misses,
        "schedules": schedules,
        "usd": round(total, 6),
        "usd_per_attempt": round(total / trials, 6) if trials else None,
        "cost_known": all(item.known for item in costs),
    }


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    return repr(value)


def _write_trace(path: Path, transcript: dict[str, object]) -> None:
    """One attempt's whole transcript as owner-only JSON; it holds the messages, never a model key."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(transcript, handle, ensure_ascii=False, indent=1, default=_plain)


class Runner:
    """Every case of every model in order under one hard budget, with the Brain usage Team reported per attempt."""

    def __init__(self, modules: Modules, brain: tuple[str, Path], budget: float, attempts: int) -> None:
        self.modules = modules
        self.brain = brain
        self.budget = eval_cost.Budget(budget)
        self.attempts = attempts
        self.meter = Meter()
        self.variants: dict[str, object] | None = None
        self.exhausted = False
        self.trace_dir: Path | None = None

    def one(self, model: tuple[str, str, str], case: tuple[str, Callable], index: int) -> tuple[Outcome, object]:
        provider, model_id, key = model
        case_id, play = case
        reservation = self.budget.reserve(
            eval_cost.call_bound(model_id, CALL_INPUT_TOKENS, CALL_OUTPUT_TOKENS) * CALLS_PER_ATTEMPT
        )
        self.meter.usage = None
        with tempfile.TemporaryDirectory() as directory:
            settings = (provider, model_id, f"eval-{secrets.token_hex(8)}")
            events = None if self.trace_dir is None else []
            harness, team, fixture = _attempt_team(self.modules, directory, self.brain, settings, events)
            attempt = Attempt(team, fixture, provider, key)
            try:
                outcome = play(attempt)
            except self.modules.local_app.ApiProblem as exc:
                outcome = Outcome(f"team:{exc.code}")
            if not outcome:
                outcome.cause = _cause(attempt, outcome.reason)
            outcome.questions = tuple(attempt.questions)
            if outcome and case_id == "owner-4-turns" and self.variants is None:
                self.variants = variants(attempt)
            harness.doCleanups()
        spent = eval_cost.Cost(0.0) if self.meter.usage is None else eval_cost.cost(self.meter.usage, model_id)
        self.budget.settle(reservation, spent)
        if events is not None:
            outcome_record = {
                "reason": outcome.reason,
                "cause": outcome.cause,
                "schedule": outcome.schedule,
                "questions": list(outcome.questions),
            }
            transcript = {"model": model_id, "case": case_id, "attempt": index, "outcome": outcome_record}
            _write_trace(self.trace_dir / model_id / case_id / f"{index}.json", {**transcript, "events": events})
        return outcome, spent

    def case(self, model: tuple[str, str, str], case: tuple[str, Callable]) -> dict[str, object]:
        passed, misses, schedules, costs, stopped = 0, [], [], [], False
        questions: collections.Counter[str] = collections.Counter()
        for index in range(self.attempts):
            try:
                outcome, spent = self.one(model, case, index)
            except eval_cost.BudgetExhaustedError:
                stopped = self.exhausted = True
                break
            costs.append(spent)
            passed += bool(outcome)
            misses += [] if outcome else [{"reason": outcome.reason, "cause": outcome.cause}]
            schedules.append(outcome.schedule)
            questions.update(outcome.questions)
        return {**_summary(case[0], passed, misses, schedules, costs, stopped), "questions": dict(questions)}


def run(
    models: list[tuple[str, str, str]],
    brain: tuple[str, Path],
    limits: tuple[int, float],
    only: frozenset[str],
    trace_dir: Path | None = None,
) -> dict:
    modules = _team_modules()
    original = modules.brain_usage.record
    runner: Runner | None = None

    def metered(operation, provider, model, counts) -> None:
        runner.meter.add(counts)
        original(operation, provider, model, counts)

    report: dict[str, object] = {"attempts_per_case": limits[0], "models": []}
    with (
        mock.patch.object(modules.local_authority, "routine_key_fingerprint", return_value=ROUTINE_KEY),
        mock.patch.object(modules.local_audit, "record_request", return_value="a" * 32),
        mock.patch.object(modules.local_audit, "record", return_value="a" * 32),
        mock.patch.object(modules.brain_usage, "record", metered),
    ):
        runner = Runner(modules, brain, limits[1], limits[0])
        runner.trace_dir = trace_dir
        for model in models:
            cases = []
            for case in CASES:
                if runner.exhausted or (only and case[0] not in only):
                    continue
                cases.append(runner.case(model, case))
            report["models"].append({"provider": model[0], "model": model[1], "cases": cases})
    report["variants"] = runner.variants
    report["budget"] = runner.budget.summary()
    report["gate"] = gate(report, limits[0])
    return report


def gate(report: dict[str, object], attempts: int) -> dict[str, object]:
    """The owner's release gate: every shipped model finished every case at 30 of 30, and both variants held."""
    failures: list[str] = []
    if attempts < ATTEMPTS:
        failures.append("attempts-below-gate")
    if report["budget"]["unknown_settlements"]:
        failures.append("cost-unknown")
    ran = {item["model"]: item["cases"] for item in report["models"]}
    for _provider, model in MODELS:
        finished = {case["id"]: case for case in ran.get(model, [])}
        for case_id, _play in CASES:
            case = finished.get(case_id)
            clean = case is not None and not case["inconclusive"] and not case["misses"]
            if not clean or case["trials"] != max(attempts, ATTEMPTS) or case["passed"] != case["trials"]:
                failures.append(f"{model}:{case_id}")
    found = report["variants"] or {}
    moved, twin = found.get("moved-zone-id", {}), found.get("twin-zone-names", {})
    if not (moved.get("status") == "done" and moved.get("listed_new_id")):
        failures.append("variant:moved-zone-id")
    ambiguous = twin.get("status") == "failed" and twin.get("code") == "plan-reference-ambiguous"
    if not ambiguous or twin.get("list_dns_records_dispatched") is not False:
        failures.append("variant:twin-zone-names")
    return {"passed": not failures, "failures": failures}


def _key(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--serve-brain", action="store_true")
    parser.add_argument("--brain-port", type=int)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--openai-key-file", type=Path)
    parser.add_argument("--anthropic-key-file", type=Path)
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    parser.add_argument("--budget", type=float, default=BUDGET_USD)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--trace-dir", type=Path, help="write each attempt's whole transcript to DIR/MODEL/CASE/N.json")
    args = parser.parse_args()
    if args.serve_brain:
        if args.brain_port is None or args.token_file is None:
            raise SystemExit("name the Brain's port and its new token file")
        return serve_brain(args.brain_port, args.token_file)
    validate()
    if args.validate:
        print(json.dumps({"validated": [case for case, _play in CASES]}))
        return 0
    keys = {"openai": _key(args.openai_key_file), "anthropic": _key(args.anthropic_key_file)}
    models = [(provider, model, keys[provider]) for provider, model in MODELS if keys[provider]]
    if not models or args.brain_port is None or args.token_file is None:
        raise SystemExit("name the Brain's port, its token file, and at least one key file")
    brain = (brain_served(args.brain_port), args.token_file)
    report = run(models, brain, (args.attempts, args.budget), frozenset(args.only), args.trace_dir)
    print(json.dumps(report, indent=2))
    # Any miss, skipped attempt, missing model or case, or budget stop fails the run; no caller reads it as a pass.
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

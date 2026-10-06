"""The Routine eval's strata, how each attempt is judged, and the replay variants after a passing attempt."""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable

if __package__:
    from eval.routines_fixture import EXAMPLE, MAX_SENDS, MOVED, SHIMPZ, ZONE_NAMES, ZONE_RECORDS, ZONE_SET
    from eval.routines_person import FULL, OVER_BUDGET, Attempt, Outcome, Person, _conversation_turns
else:  # run as a script from Team, beside its sibling modules
    from routines_fixture import EXAMPLE, MAX_SENDS, MOVED, SHIMPZ, ZONE_NAMES, ZONE_RECORDS, ZONE_SET
    from routines_person import FULL, OVER_BUDGET, Attempt, Outcome, Person, _conversation_turns


@dataclasses.dataclass(frozen=True)
class Stratum:
    """One creation stratum: how the person asks, who they are, the schedule they mean, and the zones they mean."""

    id: str
    play: Callable[[Attempt], dict[str, object] | Outcome]
    person: Callable[[], Person]
    schedule: Callable[[Person, dict[str, object]], bool]
    zones: tuple[str, ...] = (SHIMPZ,)
    twin: bool = False
    # The owner's rule, never make the person retype: the most sends it may take, its scripted sends plus one per
    # piece the person genuinely left out; an interval Team says does not fit adds the one send that chooses another.
    sends: int = 1
    # The Action whose result the Routine shows: the records of a zone, or the zones themselves.
    shows: str = "list-dns-records"


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
        ("card-invalid", attempt.team.modules.http_routine_proposal.canonical_proposal(card) == card),
        (
            "calls",
            any(action == "list-zones" for action, _payload in calls)
            and all(map(attempt.fixture.listed, stratum.zones)),
        ),
        ("invented-schedule", attempt.person.stated),
        ("schedule", stratum.schedule(attempt.person, card["schedule"])),
        (f"output:{card['output']['mode']}", card["output"]["mode"] == attempt.person.output),
        (binding or "binding", binding is None),
        (attempt.person.violations[0] if attempt.person.violations else "", not attempt.person.violations),
        (f"repeated-question:{attempt.repeated[0] if attempt.repeated else ''}", not attempt.repeated),
        ("too-many-sends", len(attempt.sends) <= stratum.sends + attempt.questions.count(OVER_BUDGET)),
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
    return Outcome(_replay_miss(attempt, stratum), seen)


def _replay_miss(attempt: Attempt, stratum: Stratum) -> str | None:
    """Why one replay did not run the person's work and show exactly its result from the right Action, or None."""
    attempt.fixture.calls.clear()
    status, notice = attempt.replay()
    if status != "done" or not all(map(attempt.fixture.listed, stratum.zones)):
        return f"replay:{status}"
    output = None if notice is None else notice.detail.get("output")
    if attempt.person.output == "none":
        # A Routine that shows nothing completes and publishes no shown result: a run notice, if any, shows nothing.
        shown = output is not None and output.get("state") == "shown"
        return "replay-notice" if notice is not None and notice.outcome == "done" and shown else None
    if notice is None or notice.outcome != "done" or output is None or output["state"] != "shown":
        return "replay-notice"
    return _shown_miss(attempt, stratum, output)


def _shown_miss(attempt: Attempt, stratum: Stratum, output: dict[str, object]) -> str | None:
    """Why a shown result is not exactly the result of the Action the Routine shows, or None."""
    (routine,) = attempt.team.service.routine_store.load("team_1").routines
    if routine.plan["steps"][output["step"] - 1]["action"] != stratum.shows:
        return "replay-shown:step"
    if stratum.shows == "list-zones":
        return None if _shown_zone_names(output["value"]) == ZONE_SET else "replay-shown:zones"
    if _shown_records(output["value"]) not in [_expected_records(zone) for zone in stratum.zones]:
        return "replay-shown:records"
    return None


def _shown_zone_names(value: dict[str, object]) -> set[object] | None:
    """Each shown zone's name, or None when the shown result holds no complete zones list."""
    listed = _fields(value).get("zones")
    if listed is None or listed.get("kind") != "list" or listed["omitted"]:
        return None
    return {_fields(item).get("name", {}).get("value") for item in listed["items"]}


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
    """A fresh person for each attempt: every mutable member is copied, so no attempt sees another's answers."""
    return lambda: Person(frequency, gap, **{key: copy.copy(value) for key, value in changes.items()})


# Each stratum's cap is its scripted sends plus one per piece the person left out of them, the output disposition
# included whenever they did not state it.
STRATA = (
    Stratum(
        "owner-4-turns",
        _send("Cria uma rotina pra mim"),
        _person("A cada 30 segundos", 30, pending=["work", "frequency", "zone"], output="show"),
        _continuous,
        sends=5,
    ),
    Stratum(
        "plain-list",
        _send("Todo dia às 9h, liste os registros DNS de shimpz.com"),
        _person("Todo dia às 9h", output="changes", said=set(FULL)),
        _daily_nine,
        sends=2,
    ),
    Stratum(
        "multi-zone",
        _send("A cada hora, liste os registros DNS de shimpz.com e de example.com"),
        _person("A cada hora", zone_words="shimpz.com e example.com", output="show", said=set(FULL)),
        _hourly,
        zones=(SHIMPZ, EXAMPLE),
        sends=2,
    ),
    Stratum(
        "earlier-send-naming",
        lambda attempt: _primed(attempt, "Liste os registros DNS de shimpz.com", "Faça isso a cada hora"),
        _person("A cada hora", output="none", said=set(FULL)),
        _hourly,
        sends=3,
    ),
    Stratum(
        "missing-schedule",
        _send("Cria uma rotina que liste os registros DNS de shimpz.com"),
        _person("Todo dia às 9h", output="changes", said={"work", "zone"}),
        _daily_nine,
        sends=3,
    ),
    Stratum(
        "reversed-order-primed",
        lambda attempt: _primed(
            attempt, "Qual é o id da zona shimpz.com?", "A cada hora, liste os registros DNS dessa zona"
        ),
        _person("A cada hora", output="show", said=set(FULL)),
        _hourly,
        sends=3,
    ),
    Stratum(
        "reversed-order-unprimed",
        _unprimed,
        _person("A cada hora", output="changes", said=set(FULL)),
        _hourly,
        sends=3,
    ),
    Stratum(
        "twin-names-at-creation",
        _send("A cada hora, liste os registros DNS de shimpz.com"),
        _person("A cada hora", choosing=True, output="show", said=set(FULL)),
        _hourly,
        twin=True,
        sends=3,
    ),
    Stratum(
        "interval-30s",
        _send("A cada 30 segundos, liste os registros DNS de shimpz.com e me mostre sempre"),
        _person("A cada 30 segundos", 30, output="show", said={*FULL, "output"}),
        _continuous,
        sends=1,
    ),
    Stratum(
        "interval-5s",
        _send("A cada 5 segundos, liste minhas zonas"),
        _person("A cada 5 segundos", 5, zone_words="Todas as minhas zonas", output="none", said=set(FULL)),
        _continuous,
        zones=(),
        sends=2,
        shows="list-zones",
    ),
)


def _played(stratum: Stratum) -> Callable[[Attempt], Outcome]:
    def play(attempt: Attempt) -> Outcome:
        attempt.person = stratum.person()
        attempt.fixture.twin = stratum.twin
        response = stratum.play(attempt)
        outcome = response if isinstance(response, Outcome) else _judged(attempt, response, stratum)
        if not outcome and attempt.repeated and (outcome.reason or "").startswith(("no-card", "question:")):
            # The span already held the answer Team asked for again: the attempt went nowhere because of the repeat.
            outcome.reason = f"repeated-question:{attempt.repeated[0]}"
        elif not outcome and attempt.person.violations and (outcome.reason or "").startswith("no-card"):
            # The agent asked what it should not have, and the attempt went nowhere.
            outcome.reason = attempt.person.violations[0]
        return outcome

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


def _shape_cause(attempt: Attempt, reason: str) -> str | None:
    """A miss whose cause is the shape of the sends themselves: how many there were and what each one got."""
    if reason == "no-card":
        return "sends-exhausted" if len(attempt.sends) >= MAX_SENDS else f"stopped-on-{attempt.sends[-1][0]}"
    if reason == "refused:routine-recording-empty":
        return "work-done-before-record" if any(actions for _kind, actions in attempt.sends[:-1]) else "no-work"
    if reason == "too-many-sends":
        asked = attempt.sends[:-1]
        agent = sum(kind in ("clarification", "prose") for kind, _actions in asked)
        team = sum(kind == "question" for kind, _actions in asked)
        return f"agent-asked-{agent}-team-asked-{team}"
    return None


def _cause(attempt: Attempt, reason: str) -> str:
    """A miss's structural root cause, from its closed reason and the shape of the attempt's sends."""
    if reason == "all-zones":
        return "acted-on-every-zone"
    if reason == "calls" or reason.startswith("selector"):
        return _lookup_cause(attempt)
    if reason == "schedule":
        return "frequency-changed"
    shaped = _shape_cause(attempt, reason)
    if shaped is not None:
        return shaped
    kept = ("replay-shown", "twin-", "question:", "invented-schedule", "repeated-question", "asked-", "already-said")
    # asked-schedule-by-both, asked-twice:*, asked-timezone, and asked-limits all keep their own reason.
    return reason if reason.startswith(kept) else reason.split(":", 1)[0]


# The strata whose plan binds shimpz.com alone through list-zones into list-dns-records: the replay variants run from
# the first passing attempt of any of them.
VARIANT_STRATA = frozenset(
    item.id for item in STRATA if item.zones == (SHIMPZ,) and not item.twin and item.shows == "list-dns-records"
)


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

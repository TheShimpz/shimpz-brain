"""What one logical turn records at its start and keeps across every resume.

Its date and the zone it was read in; the Team's knowledge and Routine listing; what Team sent about its Routine
recording: the question, mode, rerun work, schedule, and output; the model it started on (ADR-0101); its interface
language and exact start message (ADR-0090); and its attachments' commitment and token charge (ADR-0093).
"""

import datetime
import json
import re
import zoneinfo
from collections.abc import Callable, Mapping

import interface_language
import memory as team_memory
from protocol.team.http.v1 import routine as team_routine_protocol
from protocol.team.http.v1 import routine_proposal as team_routine_proposal

import routine as team_routine

DATE_METADATA = "shimpz_turn_date"
MEMORY_METADATA = "shimpz_turn_memory"
SKILLS_METADATA = "shimpz_turn_skills"
ROUTINES_METADATA = "shimpz_turn_routines"
QUESTION_METADATA = "shimpz_turn_routine_question"
MODE_METADATA = "shimpz_turn_routine_mode"
RERUN_METADATA = "shimpz_turn_routine_rerun"
SCHEDULE_METADATA = "shimpz_turn_routine_schedule"
OUTPUT_METADATA = "shimpz_turn_routine_output"
MODEL_METADATA = "shimpz_turn_model"
LOCALE_METADATA = "shimpz_turn_locale"
ZONE_METADATA = "shimpz_turn_zone"
# The zone a turn's date was read in: UTC unless Team names the person's own IANA zone.
DEFAULT_ZONE = "UTC"
MAX_ZONE_CHARS = 64
MESSAGE_METADATA = "shimpz_turn_message"
ATTACHMENTS_METADATA = "shimpz_turn_attachments"
# An ordinary turn's message id, or an attachment turn's, which the next new turn forgets whole (ADR-0093).
TURN_MESSAGE_RE = re.compile(r"shimpz-(?:turn|attached)-[0-9a-f]{32}\Z")
_COMMITMENT_RE = re.compile(r"[0-9a-f]{64}\Z")


class PinError(ValueError):
    """A pending turn's recorded pins are missing or not exactly what a start records."""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decoded(value: object) -> object:
    """One recorded pin's JSON text, decoded; anything that is not JSON text is corrupt state."""
    try:
        if not isinstance(value, str):
            raise ValueError
        return json.loads(value)
    except ValueError as exc:
        raise PinError("recorded turn pins are invalid") from exc


def _memory_json(memories: tuple[team_memory.Memory, ...] | None) -> str:
    entries = None if memories is None else [{"topic": item.topic, "preference": item.preference} for item in memories]
    return _json(entries)


def _listing_json(listing: tuple[dict[str, object], ...] | None) -> str:
    """A pinned listing, the turn's skills or its Routines, as canonical JSON; None withholds it."""
    return _json(None if listing is None else list(listing))


def record(
    turn_date: datetime.date,
    memories: tuple[team_memory.Memory, ...] | None,
    skills: tuple[dict[str, object], ...] | None,
    routines: tuple[dict[str, object], ...] | None,
) -> dict[str, str]:
    """The checkpoint metadata entries; metadata keeps only scalar values, so knowledge travels as canonical JSON."""
    return {
        DATE_METADATA: turn_date.isoformat(),
        MEMORY_METADATA: _memory_json(memories),
        SKILLS_METADATA: _listing_json(skills),
        ROUTINES_METADATA: _listing_json(routines),
    }


def restore(
    metadata: Mapping[str, object],
) -> tuple[
    datetime.date,
    tuple[team_memory.Memory, ...] | None,
    tuple[dict[str, object], ...] | None,
    tuple[dict[str, object], ...] | None,
]:
    """The exact pins a start recorded; anything else is corrupt state, never replaced by today's values."""
    keys = (DATE_METADATA, MEMORY_METADATA, SKILLS_METADATA, ROUTINES_METADATA)
    values = [metadata.get(key) for key in keys]
    try:
        if not all(isinstance(value, str) for value in values):
            raise ValueError
        date_value, memory_value, skills_value, routines_value = values
        recorded_date = datetime.date.fromisoformat(date_value)
        decoded_memory, decoded_skills = json.loads(memory_value), json.loads(skills_value)
        decoded_routines = json.loads(routines_value)
        recorded_memory = None if decoded_memory is None else team_memory.canonical(decoded_memory)
        recorded_skills = None if decoded_skills is None else team_memory.canonical_skills(decoded_skills)
        recorded_routines = None if decoded_routines is None else team_routine.canonical_routines(decoded_routines)
    except (ValueError, team_memory.MemoryContractError, team_routine.RoutineContractError) as exc:
        raise PinError("recorded turn pins are invalid") from exc
    if (
        recorded_date.isoformat() != date_value
        or _memory_json(recorded_memory) != memory_value
        or _listing_json(recorded_skills) != skills_value
        or _listing_json(recorded_routines) != routines_value
    ):
        raise PinError("recorded turn pins are invalid")
    return recorded_date, recorded_memory, recorded_skills, recorded_routines


def record_routine(
    question: dict[str, object] | None,
    mode: bool,
    rerun: tuple[dict[str, object], ...] | None,
    schedule: dict[str, object] | None,
    output: str | None,
) -> dict[str, str]:
    """The checkpoint entries of the turn's Routine context, each JSON with None as null.

    The Routine question Team asked before this turn started, whether the turn is about a Routine, the work Team asked
    to run again, and the one schedule and output mode Team read from the person, which a description must say.
    """
    return {
        QUESTION_METADATA: _json(question),
        MODE_METADATA: _json(mode),
        RERUN_METADATA: _json(rerun),
        SCHEDULE_METADATA: _json(schedule),
        OUTPUT_METADATA: _json(output),
    }


def restore_routine(
    metadata: Mapping[str, object],
) -> tuple[
    dict[str, object] | None,
    bool,
    tuple[dict[str, object], ...] | None,
    dict[str, object] | None,
    str | None,
]:
    """The exact Routine context a start recorded; anything else, a missing pin included, is corrupt state."""
    question = _optional(metadata, QUESTION_METADATA, team_routine_proposal.canonical_question)
    rerun = _optional(metadata, RERUN_METADATA, team_routine.canonical_rerun)
    schedule = _optional(metadata, SCHEDULE_METADATA, team_routine_protocol.canonical_schedule)
    output = _optional(metadata, OUTPUT_METADATA, team_routine.canonical_output)
    mode = metadata.get(MODE_METADATA)
    if mode not in ("true", "false"):
        raise PinError("recorded turn pins are invalid")
    return question, mode == "true", rerun, schedule, output


def _optional[T](metadata: Mapping[str, object], key: str, canonical: Callable[[object], T | None]) -> T | None:
    """The exact value a start pinned under ``key``, JSON null for none; anything else is corrupt state."""
    value = metadata.get(key)
    decoded = _decoded(value)
    admitted = None if decoded is None else canonical(decoded)
    if (decoded is not None and admitted is None) or _json(admitted) != value:
        raise PinError("recorded turn pins are invalid")
    return admitted


def record_model(provider: str, model: str) -> dict[str, str]:
    """The checkpoint entry naming the provider and model a turn started on, which every resume of it keeps."""
    return {MODEL_METADATA: _json({"provider": provider, "model": model})}


def restore_model(metadata: Mapping[str, object]) -> tuple[str, str]:
    """The exact provider and model a start recorded; anything else is corrupt state."""
    value = metadata.get(MODEL_METADATA)
    decoded = _decoded(value)
    if (
        not isinstance(decoded, dict)
        or set(decoded) != {"provider", "model"}
        or not all(isinstance(item, str) for item in decoded.values())
        or _json(decoded) != value
    ):
        raise PinError("recorded turn pins are invalid")
    return decoded["provider"], decoded["model"]


def record_turn(locale: str | None, message_id: str | None) -> dict[str, str]:
    """The checkpoint entries naming the turn's interface language and the id of the message that started it."""
    return {LOCALE_METADATA: _json(locale), MESSAGE_METADATA: _json(message_id)}


def restore_turn(metadata: Mapping[str, object]) -> tuple[str | None, str]:
    """The exact interface language and start message id a start recorded; anything else is corrupt state."""
    locale_value, message_value = metadata.get(LOCALE_METADATA), metadata.get(MESSAGE_METADATA)
    if not isinstance(locale_value, str) or not isinstance(message_value, str):
        raise PinError("recorded turn pins are invalid")
    locale, message_id = _decoded(locale_value), _decoded(message_value)
    if not valid_turn(locale, message_id) or _json(locale) != locale_value or _json(message_id) != message_value:
        raise PinError("recorded turn pins are invalid")
    return locale, message_id


def record_zone(zone: str) -> dict[str, str]:
    """The checkpoint entry naming the zone the turn's pinned date was read in."""
    return {ZONE_METADATA: _json(zone)}


def restore_zone(metadata: Mapping[str, object]) -> str:
    """The exact zone a start recorded; anything else, a missing pin included, is corrupt state."""
    value = metadata.get(ZONE_METADATA)
    zone = _decoded(value)
    if not valid_zone(zone) or _json(zone) != value:
        raise PinError("recorded turn pins are invalid")
    return zone


def valid_zone(zone: object) -> bool:
    """Whether ``zone`` is one IANA zone name this runtime knows, exactly as written."""
    return isinstance(zone, str) and 0 < len(zone) <= MAX_ZONE_CHARS and zone in _ZONES


_ZONES = frozenset(zoneinfo.available_timezones()) | {DEFAULT_ZONE}


def record_attachments(commitment: str, charge: int) -> dict[str, str]:
    """The checkpoint entry binding a turn to its exact prepared attachments and their counted token charge.

    It records only a digest and an integer: the content itself never enters checkpoint state.
    """
    return {ATTACHMENTS_METADATA: _json({"commitment": commitment, "charge": charge})}


def restore_attachments(metadata: Mapping[str, object]) -> tuple[str, int]:
    """The exact attachment commitment and charge a start recorded; anything else is corrupt state."""
    value = metadata.get(ATTACHMENTS_METADATA)
    decoded = _decoded(value)
    if (
        not isinstance(decoded, dict)
        or set(decoded) != {"commitment", "charge"}
        or not isinstance(decoded["commitment"], str)
        or _COMMITMENT_RE.fullmatch(decoded["commitment"]) is None
        or type(decoded["charge"]) is not int
        or decoded["charge"] < 0
        or _json(decoded) != value
    ):
        raise PinError("recorded turn pins are invalid")
    return decoded["commitment"], decoded["charge"]


def valid_turn(locale: object, message_id: object, *, started: bool = True) -> bool:
    """Whether a turn names one closed interface language or none, and, once started, its exact start message."""
    return (locale is None or interface_language.valid(locale)) and (
        (message_id is None and not started)
        or (isinstance(message_id, str) and TURN_MESSAGE_RE.fullmatch(message_id) is not None)
    )

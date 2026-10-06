"""What one logical turn records at its start and keeps across every resume.

Its date and the Team's knowledge, and its interface language and exact start message (ADR-0090).
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Mapping

import interface_language
import memory as team_memory

import routine as team_routine

DATE_METADATA = "shimpz_turn_date"
MEMORY_METADATA = "shimpz_turn_memory"
SKILLS_METADATA = "shimpz_turn_skills"
ROUTINES_METADATA = "shimpz_turn_routines"
QUESTION_METADATA = "shimpz_turn_routine_question"
MODE_METADATA = "shimpz_turn_routine_mode"
RERUN_METADATA = "shimpz_turn_routine_rerun"
CAPACITY_METADATA = "shimpz_turn_routine_capacity"
MODEL_METADATA = "shimpz_turn_model"
WRITABLE_METADATA = "shimpz_turn_knowledge_writable"
LOCALE_METADATA = "shimpz_turn_locale"
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


def _skills_json(skills: tuple[dict[str, object], ...] | None) -> str:
    return _json(None if skills is None else list(skills))


def record(
    turn_date: datetime.date,
    memories: tuple[team_memory.Memory, ...] | None,
    skills: tuple[dict[str, object], ...] | None,
    routines: tuple[dict[str, object], ...] | None = None,
    writable: bool = True,
) -> dict[str, str]:
    """The checkpoint metadata entries; metadata keeps only scalar values, so knowledge travels as canonical JSON."""
    return {
        DATE_METADATA: turn_date.isoformat(),
        MEMORY_METADATA: _memory_json(memories),
        SKILLS_METADATA: _skills_json(skills),
        ROUTINES_METADATA: _skills_json(routines),
        WRITABLE_METADATA: _json(writable),
    }


def restore(
    metadata: Mapping[str, object],
) -> tuple[
    datetime.date,
    tuple[team_memory.Memory, ...] | None,
    tuple[dict[str, object], ...] | None,
    tuple[dict[str, object], ...] | None,
    bool,
]:
    """The exact pins a start recorded; anything else is corrupt state, never replaced by today's values."""
    keys = (DATE_METADATA, MEMORY_METADATA, SKILLS_METADATA, ROUTINES_METADATA, WRITABLE_METADATA)
    values = [metadata.get(key) for key in keys]
    try:
        if not all(isinstance(value, str) for value in values):
            raise ValueError
        date_value, memory_value, skills_value, routines_value, writable_value = values
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
        or _skills_json(recorded_skills) != skills_value
        or _skills_json(recorded_routines) != routines_value
        or writable_value not in {"true", "false"}
    ):
        raise PinError("recorded turn pins are invalid")
    return recorded_date, recorded_memory, recorded_skills, recorded_routines, writable_value == "true"


def record_question(question: dict[str, object] | None) -> dict[str, str]:
    """The checkpoint entry holding the Routine question Team asked before this turn started, or none."""
    return {QUESTION_METADATA: _json(question)}


def restore_question(metadata: Mapping[str, object]) -> dict[str, object] | None:
    """The exact Routine question a start recorded; anything else is corrupt state."""
    value = metadata.get(QUESTION_METADATA)
    decoded = _decoded(value)
    question = None if decoded is None else team_routine.canonical_question(decoded)
    if (decoded is not None and question is None) or _json(question) != value:
        raise PinError("recorded turn pins are invalid")
    return question


def record_routine_mode(mode: bool, rerun: tuple[dict[str, object], ...] | None) -> dict[str, str]:
    """The checkpoint entries holding whether the turn is about a Routine and the work Team asked to run again."""
    return {MODE_METADATA: _json(mode), RERUN_METADATA: _json(None if rerun is None else list(rerun))}


def restore_routine_mode(metadata: Mapping[str, object]) -> tuple[bool, tuple[dict[str, object], ...] | None]:
    """The exact Routine mode and rerun a start recorded; anything else is corrupt state."""
    mode, rerun = metadata.get(MODE_METADATA), metadata.get(RERUN_METADATA)
    if mode not in {"true", "false"}:
        raise PinError("recorded turn pins are invalid")
    decoded = _decoded(rerun)
    work = None if decoded is None else team_routine.canonical_rerun(decoded)
    if (decoded is not None and work is None) or _json(None if work is None else list(work)) != rerun:
        raise PinError("recorded turn pins are invalid")
    return mode == "true", work


def record_capacity(capacity: int | None) -> dict[str, str]:
    """The checkpoint entry holding the daily Action steps Team left a new Routine, pinned with the listing."""
    return {CAPACITY_METADATA: _json(capacity)}


def restore_capacity(metadata: Mapping[str, object]) -> int | None:
    """The exact capacity a start recorded; anything else is corrupt state."""
    value = metadata.get(CAPACITY_METADATA)
    decoded = _decoded(value)
    if (decoded is not None and not team_routine.valid_capacity(decoded)) or _json(decoded) != value:
        raise PinError("recorded turn pins are invalid")
    return decoded


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
    locale, message_id = _decoded(locale_value), _decoded(message_value)
    if not valid_turn(locale, message_id) or _json(locale) != locale_value or _json(message_id) != message_value:
        raise PinError("recorded turn pins are invalid")
    return locale, message_id


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

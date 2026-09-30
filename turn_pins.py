"""What one logical turn records at its start and keeps across every resume: its date and the Team's knowledge."""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping

import memory as team_memory

import routine as team_routine

DATE_METADATA = "shimpz_turn_date"
MEMORY_METADATA = "shimpz_turn_memory"
SKILLS_METADATA = "shimpz_turn_skills"
ROUTINES_METADATA = "shimpz_turn_routines"
WRITABLE_METADATA = "shimpz_turn_knowledge_writable"


class PinError(ValueError):
    """A pending turn's recorded pins are missing or not exactly what a start records."""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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

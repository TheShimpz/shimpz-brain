"""What one logical turn records at its start and keeps across every resume: its date and the Team's memories."""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping

import memory as team_memory

DATE_METADATA = "shimpz_turn_date"
MEMORY_METADATA = "shimpz_turn_memory"


class PinError(ValueError):
    """A pending turn's recorded pins are missing or not exactly what a start records."""


def _memory_json(memories: tuple[team_memory.Memory, ...] | None) -> str:
    entries = None if memories is None else [{"topic": item.topic, "preference": item.preference} for item in memories]
    return json.dumps(entries, ensure_ascii=False, separators=(",", ":"))


def record(turn_date: datetime.date, memories: tuple[team_memory.Memory, ...] | None) -> dict[str, str]:
    """The checkpoint metadata entries; metadata keeps only scalar values, so memories travel as canonical JSON."""
    return {DATE_METADATA: turn_date.isoformat(), MEMORY_METADATA: _memory_json(memories)}


def restore(metadata: Mapping[str, object]) -> tuple[datetime.date, tuple[team_memory.Memory, ...] | None]:
    """The exact pins a start recorded; anything else is corrupt state, never replaced by today's values."""
    date_value = metadata.get(DATE_METADATA)
    memory_value = metadata.get(MEMORY_METADATA)
    try:
        if not isinstance(date_value, str) or not isinstance(memory_value, str):
            raise ValueError
        recorded_date = datetime.date.fromisoformat(date_value)
        decoded = json.loads(memory_value)
        recorded_memory = None if decoded is None else team_memory.canonical(decoded)
    except (ValueError, team_memory.MemoryContractError) as exc:
        raise PinError("recorded turn pins are invalid") from exc
    if recorded_date.isoformat() != date_value or _memory_json(recorded_memory) != memory_value:
        raise PinError("recorded turn pins are invalid")
    return recorded_date, recorded_memory

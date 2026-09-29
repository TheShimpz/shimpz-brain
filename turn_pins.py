"""What one logical turn records at its start and keeps across every resume: its date and standing instructions."""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping

import instructions as standing_instructions

DATE_METADATA = "shimpz_turn_date"
INSTRUCTIONS_METADATA = "shimpz_turn_instructions"


class PinError(ValueError):
    """A pending turn's recorded pins are missing or not exactly what a start records."""


def _instructions_json(instructions: tuple[str, ...]) -> str:
    return json.dumps(list(instructions), ensure_ascii=False, separators=(",", ":"))


def record(turn_date: datetime.date, instructions: tuple[str, ...]) -> dict[str, str]:
    """The checkpoint metadata entries; metadata keeps only scalar values, so the rules travel as canonical JSON."""
    return {DATE_METADATA: turn_date.isoformat(), INSTRUCTIONS_METADATA: _instructions_json(instructions)}


def restore(metadata: Mapping[str, object]) -> tuple[datetime.date, tuple[str, ...]]:
    """The exact pins a start recorded; anything else is corrupt state, never replaced by today's values."""
    date_value = metadata.get(DATE_METADATA)
    rules_value = metadata.get(INSTRUCTIONS_METADATA)
    try:
        if not isinstance(date_value, str) or not isinstance(rules_value, str):
            raise ValueError
        recorded_date = datetime.date.fromisoformat(date_value)
        recorded_rules = standing_instructions.canonical(json.loads(rules_value))
    except (ValueError, standing_instructions.InstructionsError) as exc:
        raise PinError("recorded turn pins are invalid") from exc
    if recorded_date.isoformat() != date_value or _instructions_json(recorded_rules) != rules_value:
        raise PinError("recorded turn pins are invalid")
    return recorded_date, recorded_rules

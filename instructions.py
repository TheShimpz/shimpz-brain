"""Standing instructions the Supervisor saved for a Team: closed text rules the turn prompt quotes as data."""

from __future__ import annotations

import unicodedata

MAX_INSTRUCTIONS = 16
MAX_INSTRUCTION_CHARS = 280


class InstructionsError(ValueError):
    pass


def _admitted(text: object) -> bool:
    return (
        isinstance(text, str)
        and unicodedata.normalize("NFC", text) == text
        and text.strip() == text
        and 1 <= len(text) <= MAX_INSTRUCTION_CHARS
        and not any(
            unicodedata.category(character).startswith("C") or unicodedata.category(character) in {"Zl", "Zp"}
            for character in text
        )
    )


def canonical(value: object) -> tuple[str, ...]:
    """Return the exact instruction list, or raise: at most 16 distinct single-line rules of at most 280 characters."""
    if not isinstance(value, (list, tuple)) or len(value) > MAX_INSTRUCTIONS:
        raise InstructionsError("invalid standing instructions")
    if not all(_admitted(text) for text in value) or len({text.casefold() for text in value}) != len(value):
        raise InstructionsError("invalid standing instructions")
    return tuple(value)

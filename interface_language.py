"""The closed Admin interface languages that Brain-written text follows (ADR-0090)."""

from __future__ import annotations

from typing import Literal

Locale = Literal["ar", "de", "en", "es", "fr", "ja", "pt", "zh"]

LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Brazilian Portuguese",
    "zh": "Simplified Chinese",
}


def valid(value: object) -> bool:
    """Whether a value is one closed interface language code."""
    return isinstance(value, str) and value in LANGUAGE_NAMES


def language_name(locale: str) -> str:
    """The English name of one closed interface language, as prompts state it."""
    return LANGUAGE_NAMES[locale]

"""The closed Admin interface languages that Brain-written text follows (ADR-0090).

The language set is the Team HTTP protocol's CHAT_LOCALES, read from its generated mirror; Brain owns only each
language's English prompt name.
"""

from typing import Literal

from protocol.team.http.v1 import payload as team_payload

Locale = Literal[*sorted(team_payload.CHAT_LOCALES)]

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
    return isinstance(value, str) and value in team_payload.CHAT_LOCALES


def language_name(locale: str) -> str:
    """The English name of one closed interface language, as prompts state it."""
    return LANGUAGE_NAMES[locale]

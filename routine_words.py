"""The user's own words a Routine is compiled from, exactly as Team froze them (ADR-0092 amendments).

A Routine's words are ordered kinded parts. A said part is the user's current message (or the answer they just gave to
the Routine's question) or a message or answer of the Routine they are setting up (their draft, 2026-10-05); together
they state one request, and the standing request stands in one of them. A cited part is an earlier send that
supplies only what a said part refers to (2026-10-04). Each part is parsed on its own, so no quoted region or stretch of
own words ever crosses into another; Team admits the same structure, so this is only the Brain's early check of it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import memory as team_memory

# The person's own earlier sends a Routine request may refer to, as Team froze them (ADR-0092, 2026-10-04).
MAX_EARLIER = 3
MAX_EARLIER_CHARS = 2_000
# The person's Routine draft: at most eight of their own kinded parts, 32,000 characters in all (ADR-0092, 2026-10-05).
MAX_DRAFT_PARTS = 8
MAX_DRAFT_CHARS = 32_000
# A sealed Routine's words: its draft, earlier sends, message, and a selected label, each part at most a message long.
MAX_PARTS = MAX_DRAFT_PARTS + MAX_EARLIER + 2
MAX_PART_CHARS = 16_000
# The longest free-text answer to a Routine question: the Admin question card's own answer field.
MAX_ANSWER_CHARS = 4_000
CITED = "cited"
SAID = "said"
_KINDS = frozenset({CITED, SAID})
_LAYOUT = frozenset({"\n", "\r", "\t"})
_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
# A complete number as a person may write it, so a fragment of one ("1" of "1e3", "5" of ".5") is never a token.
_NUMBER_TOKEN_RE = re.compile(r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")
_POINTER_RE = re.compile(r"(?:/(?:[^/~]|~[01])*)*\Z")
# A whole count as a person writes it ("1000", "1.000", "1,000"), at most seven digits or 999,999,999, and only a
# complete one, exactly as Team reads it: never a fragment of a longer number, a decimal, a signed number, or an
# exponent.
# Any sign or decimal mark a number may carry, ASCII or not, and a digit of any script, so "−100" (U+2212), "100٫25",
# or "١100" holds no count.
_SIGNS = r"+\-\u2012\u2013\u2212\ufe62\ufe63\uff0b\uff0d"
_MARKS = r".,\u066b\u066c\uff0c\uff0e"
_COUNT_RE = re.compile(
    rf"(?<![\d{_MARKS}{_SIGNS}])(?<!\d[eE])(?<!\d[eE][{_SIGNS}])"
    r"(?:[0-9]{1,3}(?:[.,][0-9]{3}){1,2}|[0-9]{1,7})"
    rf"(?!\d|[{_MARKS}]\d|[eE][{_SIGNS}]?\d)"
)


class RoutineWordsError(ValueError):
    """The user's words were not exactly what Team freezes."""


def _canonical(item: object, maximum: int) -> bool:
    """Whether one text is exactly as Team keeps it: NFC, trimmed, no control character but layout, 1 to ``maximum``."""
    return (
        isinstance(item, str)
        and 0 < len(item) <= maximum
        and unicodedata.normalize("NFC", item) == item
        and item.strip() == item
        and not any(unicodedata.category(character)[0] == "C" and character not in _LAYOUT for character in item)
    )


def canonical_earlier(value: object) -> tuple[str, ...]:
    """The earlier sends exactly as Team cites them: at most three, each NFC, trimmed, no control but layout."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) > MAX_EARLIER
        or not all(_canonical(item, MAX_EARLIER_CHARS) for item in value)
    ):
        raise RoutineWordsError("invalid earlier sends")
    return tuple(value)


def _kinded(value: object) -> tuple[str, str] | None:
    if isinstance(value, dict) and set(value) == {"kind", "text"}:
        value = (value["kind"], value["text"])
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not isinstance(value[0], str):
        return None
    if value[0] not in _KINDS:
        return None
    return value[0], value[1]


def canonical_draft(value: object) -> tuple[tuple[str, str], ...]:
    """The user's Routine draft exactly as Team froze it: kinded canonical parts, oldest first, within its bounds."""
    parts = _parts(value, MAX_DRAFT_PARTS)
    if (
        parts is None
        or not all(_canonical(text, MAX_PART_CHARS) for _kind, text in parts)
        or sum(len(text) for _kind, text in parts) > MAX_DRAFT_CHARS
    ):
        raise RoutineWordsError("invalid draft")
    return parts


def canonical_sealed(value: object) -> tuple[tuple[str, str], ...]:
    """The parts before a sealed Routine's message, as Team keeps them: kinded texts of at most a message each."""
    parts = _parts(value, MAX_PARTS - 1)
    if parts is None or not all(
        isinstance(text, str) and 0 < len(text) <= MAX_PART_CHARS and "\0" not in text for _kind, text in parts
    ):
        raise RoutineWordsError("invalid draft")
    return parts


def _parts(value: object, maximum: int) -> tuple[tuple[str, str], ...] | None:
    parts = tuple(_kinded(item) for item in value) if isinstance(value, (list, tuple)) else None
    return None if parts is None or len(parts) > maximum or None in parts else parts


def canonical_answer(value: object) -> str | None:
    """The answer the user's composed reply gave to the draft's last question, exactly as Team took it, or None."""
    if value is not None and not _canonical(value, MAX_ANSWER_CHARS):
        raise RoutineWordsError("invalid answer")
    return value


def _split(part: str) -> tuple[list[str], list[str]]:
    """One part's own words and quoted regions; no region or stretch ever crosses into another part."""
    regions = [(match.start(), match.end()) for match in team_memory._QUOTED_RE.finditer(part)]
    own: list[str] = []
    cursor = 0
    for start, end in regions:
        if start > cursor:
            own.append(part[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(part):
        own.append(part[cursor:])
    return own, [part[start:end] for start, end in regions]


class Words:
    """The user's own words and numbered quoted regions of a Routine's kinded parts, as Team parses them.

    Each part, oldest first, is parsed on its own; quoted regions are numbered across them in that order. A said part
    (a message or answer the user wrote while setting the Routine up, the current one last) may state the standing
    request; a cited part is an earlier send that supplies only what a said part refers to.
    """

    def __init__(self, parts: tuple[tuple[str, str], ...]) -> None:
        self.parts: list[tuple[str, list[str]]] = []
        self.quoted: list[str] = []
        for kind, part in parts:
            own, quoted = _split(part)
            self.parts.append((kind, own))
            self.quoted.extend(quoted)

    def said(self, text: object) -> bool:
        """Whether the text stands inside one stretch of a said part's own words."""
        segments = [segment for kind, own in self.parts if kind == SAID for segment in own]
        return isinstance(text, str) and bool(text) and any(text in segment for segment in segments)

    def mine(self, text: object) -> bool:
        """Whether the text stands inside one stretch of any part's own words."""
        segments = [segment for _kind, own in self.parts for segment in own]
        return isinstance(text, str) and bool(text) and any(text in segment for segment in segments)

    def counts(self) -> frozenset[int]:
        """Every whole count any part's own words write in digits."""
        return frozenset(
            int(re.sub(r"[.,]", "", match.group()))
            for _kind, own in self.parts
            for segment in own
            for match in _COUNT_RE.finditer(segment)
        )

    def adopted(self, region: object, text: object, instruction: object) -> bool:
        return (
            type(region) is int
            and 0 <= region < len(self.quoted)
            and isinstance(text, str)
            and bool(text)
            and text in self.quoted[region]
            and self.mine(instruction)
        )


def kinded_parts(
    message: str, earlier: tuple[str, ...], draft: tuple[tuple[str, str], ...], continues: bool
) -> tuple[tuple[str, str], ...]:
    """A Routine's kinded parts as Team builds them: the draft when continued, the earlier sends, then the message."""
    return (*(draft if continues else ()), *((CITED, text) for text in earlier), (SAID, message))


@dataclass(frozen=True, slots=True)
class UserWords:
    """Everything one compile reads of the user: the current said text, the earlier sends, and the Routine draft."""

    message: str
    earlier: tuple[str, ...] = ()
    draft: tuple[tuple[str, str], ...] = ()

    def parts(self, continues: bool) -> tuple[tuple[str, str], ...]:
        return kinded_parts(self.message, self.earlier, self.draft, continues)

    @property
    def skipped(self) -> int:
        """How many quoted regions the draft holds, numbered before every other part's."""
        return len(Words(self.draft).quoted)


# A literal's provenance: whether each of its scalars is cited from the person's own words, exactly as Team proves it.


def _leaves(value: object, at: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _leaves(item, f"{at}/{key.replace('~', '~0').replace('/', '~1')}")
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaves(item, f"{at}/{index}")
    else:
        yield at, value


def _cited(origin: Mapping[str, object], target: object, words: Words) -> bool:
    text = origin["text"]
    if not _cited_text(origin, words) or isinstance(target, bool) or target is None:
        return False
    if isinstance(target, str):
        return target == text
    return (
        isinstance(target, int | float)
        and _NUMBER_RE.fullmatch(text) is not None
        and type(parsed := json.loads(text)) is type(target)
        and parsed == target
    )


def narrowed(source: dict[str, object], words: Words) -> dict[str, object]:
    """The literal with each number its compiler cited together with the person's words narrowed to its own digits.

    A compiler may cite "page 1" for the number 1. When that whole citation stands in the person's own words (or, for a
    quote, in the adopted region) and holds exactly one complete number, which is that scalar with its JSON type, the
    origin cites that number instead: a part of the very text it cited, so it proves nothing the person's words did not
    already hold. Anything else is left exactly as cited for the check to refuse.
    """
    leaves = dict(_leaves(source["value"]))
    for origin in source["origins"]:
        target, text = leaves.get(origin["at"]), origin["text"]
        if (
            origin["from"] == "default"
            or isinstance(target, bool)
            or not isinstance(target, int | float)
            or not isinstance(text, str)
            or _NUMBER_RE.fullmatch(text) is not None
            or not _cited_text(origin, words)
        ):
            continue
        tokens = set(_NUMBER_TOKEN_RE.findall(text))
        token = tokens.pop() if len(tokens) == 1 else ""
        if _NUMBER_RE.fullmatch(token) is None:
            continue
        try:
            parsed = json.loads(token)
        except ValueError:
            continue
        if type(parsed) is type(target) and parsed == target:
            origin["text"] = token
    return source


def _cited_text(origin: Mapping[str, object], words: Words) -> bool:
    """Whether the whole cited text stands in the person's own words, or in the region their own words adopt."""
    if origin["from"] == "message":
        return words.mine(origin["text"])
    return words.adopted(origin["region"], origin["text"], origin["instruction"])


def proven(source: Mapping[str, object], member: object, words: Words) -> bool:
    """Whether every scalar of a literal has exactly one cited origin, or the whole value is its member's default."""
    leaves = dict(_leaves(source["value"]))
    covered: list[str] = []
    for origin in source["origins"]:
        if origin["from"] == "default":
            default = isinstance(member, dict) and "default" in member and _same(member["default"], source["value"])
            if not default or origin["at"] != "":
                return False
            covered.extend(leaves)
        elif origin["at"] not in leaves or not _cited(origin, leaves[origin["at"]], words):
            return False
        else:
            covered.append(origin["at"])
    return sorted(covered) == sorted(leaves)


def _same(left: object, right: object) -> bool:
    """JSON equality, so a boolean never equals a number."""
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def origin_shape(origin: Mapping[str, object]) -> bool:
    kind, text, region, instruction = origin["from"], origin["text"], origin["region"], origin["instruction"]
    if not isinstance(origin["at"], str) or _POINTER_RE.fullmatch(origin["at"]) is None:
        return False
    if kind == "message":
        return bool(text) and region is None and instruction is None
    if kind == "quote":
        return bool(text) and region is not None and bool(instruction)
    return text is None and region is None and instruction is None

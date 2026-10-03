"""Language-aware retrieval mechanisms of the language-retrieval experiment: experiment-only (ADR-0094, proposed 0096).

The driver ``.tests/perf/precision_engineering.py`` applies them in its own Team process; production runtime is
unchanged. They act only on a read Action whose corpus documents its search behavior (``SearchSpec``), and only
through that same read with exactly one argument changed or removed:

- a language registry: one tag per observed record (deduplicated by its declared identity), aggregated per Assistant
  and Action, holding codes and counts only, never text;
- expansion: the search text translated by one helper call into target languages (assumed English, or the languages
  the registry observed), each variant read with only the declared text argument replaced, and what only the
  variants found returned apart, under a Team-owned envelope, as candidates;
- a hint: the registry's languages appended to the Action's description, with no helper call and no extra read;
- embedding recovery: when a read whose text filter is declared relaxable finds nothing, one read without the relaxable
  filters, ranked by embedding similarity to the search text, the best returned apart as candidates.

The language service (``.tests/perf/precision_language_service.py``) computes tags and similarity scores; this
module only plans, parses, bounds, and shapes. It uses only the standard library.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Registry admission: a language is stored in a collection when it has at least this share of the collection's tagged
# records and at least this many of them; at most this many languages are planned, by share. Hypotheses under test.
MIN_SHARE = 0.10
MIN_COUNT = 2
MAX_LANGUAGES = 3
# Expansion bounds: terms per language the helper may return, extra reads per expanded call, term length, helper calls
# per attempt, and candidates returned in one envelope.
TERMS_PER_LANGUAGE = 2
MAX_EXTRA_READS = 4
MAX_TERM_CHARS = 60
MAX_HELPER_CALLS = 3
MAX_CANDIDATES = 20
# Embedding recovery bounds: items ranked from one relaxed read and candidates returned.
MAX_RANKED = 200
TOP_K = 5
NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "zh": "Chinese",
}
# A number (sign, digits, and decimal or thousands separators) is one literal token; so is the content of a quotation in
# straight, typographic, guillemet, or corner quotes.
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)*")
_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'|“([^”]+)”|‘([^’]+)’|«([^»]+)»|「([^」]+)」|『([^』]+)』")


@dataclass(frozen=True, slots=True)
class SearchSpec:
    """A read Action's documented search behavior.

    The free-text input property (or None), the optional properties a caller may omit to list more broadly, the result
    item fields the text is matched against, the result's item list, and the field that identifies an item.
    """

    text: str | None
    relaxable: tuple[str, ...]
    fields: tuple[str, ...]
    collection: str
    id: str

    @classmethod
    def of(cls, documented: Mapping[str, object]) -> SearchSpec:
        return cls(
            None if documented.get("text") is None else str(documented["text"]),
            tuple(str(name) for name in documented.get("relaxable") or ()),
            tuple(str(name) for name in documented.get("fields") or ()),
            str(documented["collection"]),
            str(documented["id"]),
        )


def items(result: object, spec: SearchSpec) -> list[Mapping[str, object]]:
    """The result's documented item list, or nothing when the result does not have that shape."""
    found = result.get(spec.collection) if isinstance(result, Mapping) else None
    return [item for item in found if isinstance(item, Mapping)] if isinstance(found, list) else []


def item_text(item: Mapping[str, object], spec: SearchSpec) -> str:
    """One record's documented text fields, joined; a record is tagged once, however many fields it has."""
    return " | ".join(str(item[name]) for name in spec.fields if isinstance(item.get(name), str) and item[name])


def item_key(item: Mapping[str, object], spec: SearchSpec) -> str | None:
    """The record's declared identity (a non-empty string or an integer), or None when it is missing or malformed.

    A record without a valid identity is never tagged or deduplicated: identity is never synthesized from content.
    """
    value = item.get(spec.id)
    if isinstance(value, bool) or not isinstance(value, str | int) or (isinstance(value, str) and not value.strip()):
        return None
    return f"id:{value}"


@dataclass
class Registry:
    """Observed record languages per (Assistant, Action): one tag per record identity, the latest one winning."""

    tags: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)

    def observe(self, assistant: str, action: str, keys: Sequence[str], codes: Sequence[str]) -> int:
        """Record each item's tag under its identity; returns how many identities were new."""
        bucket = self.tags.setdefault((assistant, action), {})
        new = sum(key not in bucket for key in keys)
        bucket.update(zip(keys, codes, strict=True))
        return new

    def copy(self) -> Registry:
        return Registry({key: dict(value) for key, value in self.tags.items()})

    @staticmethod
    def _admitted(codes: Sequence[str]) -> list[str]:
        counts = Counter(code for code in codes if code != "und")
        total = sum(counts.values())
        admitted = [code for code, n in counts.items() if n >= MIN_COUNT and n / total >= MIN_SHARE]
        return sorted(admitted, key=lambda code: (-counts[code], code))[:MAX_LANGUAGES]

    def plan(self, assistant: str, action: str) -> tuple[list[str], str]:
        """The languages to search and where they came from: this collection, else the Assistant's other reads."""
        own = self._admitted(list(self.tags.get((assistant, action), {}).values()))
        if own:
            return own, "collection"
        buckets = [bucket for (name, _action), bucket in sorted(self.tags.items()) if name == assistant]
        others = [code for bucket in buckets for code in bucket.values()]
        prior = self._admitted(others)
        return (prior, "assistant") if prior else ([], "none")

    def summary(self) -> dict[str, dict[str, int]]:
        """Counts only, for reporting: never an identity or a text."""
        return {
            f"{assistant}.{action}": dict(sorted(Counter(bucket.values()).items()))
            for (assistant, action), bucket in sorted(self.tags.items())
        }


def hint(summary: str, languages: Sequence[str], source: str) -> str:
    """An Action description with the registry's languages appended (the NH arm); unchanged without languages.

    The wording names where they were observed: this Action's own records (``collection``) or other reads of the same
    Assistant (``assistant``).
    """
    if not languages:
        return summary
    names = ", ".join(f"{NAMES.get(code, code)} ({code})" for code in languages)
    if source == "collection":
        return f"{summary} Team has observed that this Action's stored records are written in: {names}."
    return f"{summary} Team has observed records written in {names} in other reads of this Assistant."


VARIANT_INSTRUCTIONS = (
    "A search will run as a literal, case-insensitive substring match over stored records. Given the user's request, "
    "the Action, and the search text, propose for each target language at most two search terms that would match the "
    "records the user means: the word or short phrase most likely to appear in a stored record, singular base form "
    "first, then a plural, a common synonym, or the unaccented spelling only when it would match different records. "
    "Keep codes, numbers, names, and quoted literals exactly as given. A term already in a target language may be "
    "returned unchanged. Never broaden to a different kind of thing."
)
VARIANT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"lang": {"type": "string"}, "terms": {"type": "array", "items": {"type": "string"}}},
                "required": ["lang", "terms"],
            },
        }
    },
    "required": ["variants"],
}


def variant_input(
    message: str, genesis: str, action_id: str, summary: str, field_name: str, text: str, languages: Sequence[str]
) -> str:
    """The helper's whole input: the user's own words and the Action's public description; never any record."""
    body = {
        "user_request": message,
        "assistant": genesis,
        "action": {"id": action_id, "summary": summary, "search_field": field_name},
        "search_text": text,
        "target_languages": list(languages),
    }
    return json.dumps(body, ensure_ascii=False, sort_keys=True)


def _keeps_literals(original: str, term: str) -> bool:
    """Every number of the original is a whole number token of the term, and every quotation's content is in it."""
    numbers = set(_NUMBER_RE.findall(term))
    quoted = [next(group for group in match.groups() if group) for match in _QUOTED_RE.finditer(original)]
    return set(_NUMBER_RE.findall(original)) <= numbers and all(content in term for content in quoted)


def variants(answer: object, original: str, languages: Sequence[str]) -> list[tuple[str, str]]:
    """Admitted (language, term) pairs in deterministic order, at most ``MAX_EXTRA_READS``.

    Round-robin over the target languages in plan order, each language's terms in the helper's order.
    A term is refused when it is not a string, is empty or longer than ``MAX_TERM_CHARS``, equals the original or an
    earlier term ignoring case, or does not keep every whole number and quoted literal of the original; a language
    outside the targets is ignored.
    """
    proposed: dict[str, list[str]] = {code: [] for code in languages}
    entries = answer.get("variants") if isinstance(answer, Mapping) else None
    for entry in entries if isinstance(entries, list) else []:
        code = entry.get("lang") if isinstance(entry, Mapping) else None
        terms = entry.get("terms") if isinstance(entry, Mapping) else None
        if code in proposed and isinstance(terms, list):
            proposed[code].extend(term for term in terms[:TERMS_PER_LANGUAGE] if isinstance(term, str))
    seen = {original.strip().casefold()}
    chosen: list[tuple[str, str]] = []
    for depth in range(TERMS_PER_LANGUAGE):
        for code in languages:
            terms = proposed[code]
            term = terms[depth].strip() if depth < len(terms) else ""
            if not term or len(term) > MAX_TERM_CHARS or term.casefold() in seen or not _keeps_literals(original, term):
                continue
            seen.add(term.casefold())
            chosen.append((code, term))
    return chosen[:MAX_EXTRA_READS]


def parse(text: str | None) -> object:
    """A structured helper answer, or None when it is missing or malformed."""
    try:
        return json.loads(text) if text is not None else None
    except ValueError:
        return None


def _candidates(
    found: Sequence[Mapping[str, object]], spec: SearchSpec, exclude: set[str]
) -> list[Mapping[str, object]]:
    """New records by declared identity; a record without a valid identity cannot be deduplicated and is left out."""
    unique: dict[str, Mapping[str, object]] = {}
    for item in found:
        key = item_key(item, spec)
        if key is not None and key not in exclude and key not in unique:
            unique[key] = item
    return list(unique.values())


EXPANSION_NOTE = (
    "Team also ran this same read with the search text in the languages listed under searched, because stored records "
    "may be written in another language than the request. The records under candidates were found only by those "
    "searches: they are not confirmed matches. Use only those that match what the user means; when which records the "
    "user means is unclear, ask before changing anything."
)
RANKING_NOTE = (
    "The read found nothing. Team read this Action once more without its search text and ranked what it found by "
    "similarity of meaning to the search text; the best are under candidates with their scores. They are not "
    "confirmed matches and the list is partial: use only those that match what the user means, and never conclude from "
    "it that nothing else exists. When which records the user means is unclear, ask before changing anything."
)


def expanded(result: object, spec: SearchSpec, searches: Sequence[tuple[str, str, object]]) -> tuple[object, int]:
    """The original result, unchanged, and apart from it Team's envelope of what only the variant reads found.

    ``searches`` holds (language, term, result) per variant read that ran. Returns the original result itself when no
    variant read found anything new, and the number of candidates otherwise.
    """
    exclude = {item_key(item, spec) for item in items(result, spec)}
    searched, pooled = [], []
    for code, term, found in searches:
        listed = items(found, spec)
        searched.append({"lang": code, "text": term, "found": len(listed)})
        pooled.extend(listed)
    candidates = _candidates(pooled, spec, exclude)
    if not candidates:
        return result, 0
    shown = candidates[:MAX_CANDIDATES]
    envelope = {
        "note": EXPANSION_NOTE,
        "searched": searched,
        "candidates": shown,
        "total": len(candidates),
        "truncated": len(candidates) > MAX_CANDIDATES,
    }
    return {"assistant_result": result, "team_search": envelope}, len(shown)


def ranked(
    result: object, spec: SearchSpec, relaxed_result: object, scores: Sequence[float], threshold: float
) -> tuple[object, int]:
    """The original empty result and, apart, the best ``TOP_K`` relaxed-read records at or above ``threshold``.

    ``scores`` aligns with the first ``MAX_RANKED`` records of the relaxed read. A record without a valid identity, or
    one repeating an identity already ranked, is never a candidate.
    """
    listed = items(relaxed_result, spec)[:MAX_RANKED]
    scored: dict[str, tuple[int, Mapping[str, object], float]] = {}
    for index, (item, score) in enumerate(zip(listed, scores, strict=True)):
        key = item_key(item, spec)
        if key is not None and key not in scored:
            scored[key] = (index, item, score)
    order = sorted(scored.values(), key=lambda entry: (-entry[2], entry[0]))
    best = [(item, score) for _index, item, score in order[:TOP_K] if score >= threshold]
    if not best:
        return result, 0
    envelope = {
        "note": RANKING_NOTE,
        "candidates": [{"score": round(score, 3), "item": item} for item, score in best],
        "ranked": len(listed),
        "truncated": len(items(relaxed_result, spec)) > MAX_RANKED,
    }
    return {"assistant_result": result, "team_search": envelope}, len(best)


def without_relaxable(arguments: Mapping[str, object], spec: SearchSpec) -> dict[str, object] | None:
    """The arguments without every declared relaxable one, or None when none was given."""
    kept = {name: value for name, value in arguments.items() if name not in spec.relaxable}
    return None if len(kept) == len(arguments) else kept


def search_text(arguments: Mapping[str, object], spec: SearchSpec) -> str | None:
    """The call's free-text value, when the Action documents one and the call carries a non-empty string."""
    value = arguments.get(spec.text) if spec.text else None
    return value if isinstance(value, str) and value.strip() else None

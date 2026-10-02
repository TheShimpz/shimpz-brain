"""The frozen tuning / held-out split of the precision-v2.1 templates for engineering experiments (ADR-0094).

Templates, not locales, are split, so every language of a held-out task stays unseen while arms are tuned. The rule is
mechanical and was frozen, with ``split.json``, before any arm work: group templates by behavior and by whether they
need several Assistants; order each group by SHA-256 of ``precision-v2/split/<template>``; and alternate the group's
templates between the two sets, starting with whichever set is smaller (held-out on a tie). Arm design may look only at
tuning results; conclusions use held-out results only.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from eval import corpus

SPLIT = Path(__file__).with_name("split.json")
# The large-API stratum (eval.large_api) has its own split by the same rule, frozen before its arm work.
LARGE_API_SPLIT = Path(__file__).with_name("split_large_api.json")


def _corpus(path: Path) -> tuple[str, str, tuple[corpus.Template, ...]]:
    if path.name == LARGE_API_SPLIT.name:
        from eval import large_api

        return large_api.CORPUS_ID, large_api.digest(), large_api.TEMPLATES
    return corpus.CORPUS_ID, corpus.digest(), corpus.TEMPLATES


def compute(templates: tuple[corpus.Template, ...] = corpus.TEMPLATES, prefix: str = "precision-v2") -> dict:
    groups: dict[tuple[str, bool], list[corpus.Template]] = defaultdict(list)
    for template in templates:
        groups[template.behavior, len(template.needed) > 1].append(template)
    held: list[str] = []
    tuning: list[str] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda t: hashlib.sha256(f"{prefix}/split/{t.id}".encode()).hexdigest())
        held_first = len(held) <= len(tuning)
        for index, template in enumerate(ordered):
            (held if (index % 2 == 0) == held_first else tuning).append(template.id)
    return {"held-out": sorted(held), "tuning": sorted(tuning)}


def digest(sets: dict[str, list[str]], corpus_id: str = corpus.CORPUS_ID, corpus_digest: str | None = None) -> str:
    body = {"corpus": corpus_id, "corpus_digest": corpus_digest or corpus.digest(), **sets}
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load(path: Path = SPLIT) -> dict[str, list[str]]:
    """The frozen split; refused when it no longer matches its rule, its corpus, or its own digest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    corpus_id, corpus_digest, templates = _corpus(path)
    sets = {"held-out": data["held-out"], "tuning": data["tuning"]}
    # precision-v2.1 changed only the oracle's title comparison, so it keeps the precision-v2 split rule and sets.
    prefix = "precision-v2" if corpus_id == corpus.CORPUS_ID else corpus_id
    if (
        data.get("corpus") != corpus_id
        or sets != compute(templates, prefix)
        or data.get("digest") != digest(sets, corpus_id, corpus_digest)
    ):
        raise ValueError("the frozen split does not match its rule, corpus, or digest")
    return sets


def part(scenario_id: str, sets: dict[str, list[str]]) -> str:
    template = scenario_id.rsplit(".", 1)[0]
    return "held-out" if template in sets["held-out"] else "tuning"

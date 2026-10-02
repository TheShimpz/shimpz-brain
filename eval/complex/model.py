"""precision-v3 episodes: templates, the scripted user, reference workflows, and the whole-episode oracle (ADR-0094).

An episode is one template in one language on one Brain thread over one ``ComplexWorld``. A scripted finite-state user
sends each step's message; when Brain ends a turn with its structured clarification, the user answers with the
step's predeclared reply, or, when the step has none, records a simulator limitation for audit and moves on. There is
no LLM simulator. Every template carries an independently written reference workflow: the Actions a correct agent
runs on each step, executed on a fresh world with the same faults to give the expected state after every step. The
oracle checks the final state, every step's checkpoint, every write effect including transient and repeated ones,
writes to another Team's records, and the world's permission violations. Communication obligations are judged over
the whole trajectory. This module uses only the standard library.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from eval.complex import phrases
from eval.complex.world import CUSTOMERS, OTHER_TEAM, ComplexWorld, Fault, effects
from eval.corpus import LOCALES
from eval.world import ActionFailedError

CORPUS_ID = "precision-v3"
STRATA = {
    "S1": "several requests",
    "S2": "long conversations and memory",
    "S3": "large observations",
    "S4": "failures and partial work",
    "S5": "ask versus lookup",
    "S6": "permission and provenance",
}
COMPOSITIONS = frozenset({"and", "chain", "selection", "mixed"})
BANDS = ("<8k", "8-32k", "32-128k")
Part = tuple[str, Mapping[str, object]]
Call = tuple[str, str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class Step:
    """One scripted user turn, the reference Actions a correct agent runs for it, and what it must convey."""

    parts: tuple[Part, ...]
    reference: tuple[Call, ...] = ()
    clarify_reply: tuple[Part, ...] | None = None
    revoke: tuple[str, ...] = ()
    communicate: str = ""


@dataclass(frozen=True, slots=True)
class Template:
    id: str
    stratum: str
    coverage: str
    composition: str
    band: str
    assistants: tuple[str, ...]
    steps: tuple[Step, ...]
    faults: tuple[Fault, ...] = ()

    @property
    def crosses(self) -> bool:
        """The reference workflow uses Actions of more than one Assistant."""
        return len({call[0] for step in self.steps for call in step.reference}) > 1

    @property
    def turns(self) -> int:
        return len(self.steps)


@dataclass(frozen=True, slots=True)
class Episode:
    template: Template
    locale: str
    seed: int = 0

    @property
    def id(self) -> str:
        return f"{self.template.id}.{self.locale}"

    def message(self, step: Step) -> str:
        return phrases.say(self.locale, step.parts)

    def clarify_reply(self, step: Step) -> str | None:
        return None if step.clarify_reply is None else phrases.say(self.locale, step.clarify_reply)


def episodes(templates: tuple[Template, ...]) -> tuple[Episode, ...]:
    """The first panel: one seed and two languages per template, rotating across the eight languages."""
    return tuple(
        Episode(template, LOCALES[(index + offset) % len(LOCALES)])
        for index, template in enumerate(templates)
        for offset in (0, 4)
    )


class ScriptedUser:
    """The finite-state user: scripted messages, a predeclared answer to one clarification per step, nothing else."""

    def __init__(self, episode: Episode) -> None:
        self.episode = episode
        self.limitations: list[int] = []
        self.user_turns = 0

    def turns(self) -> Iterator[tuple[int, str]]:
        """Yield (step index, message); ``send`` the turn's clarification flag back after each message."""
        for index, step in enumerate(self.episode.template.steps):
            self.user_turns += 1
            clarified = yield index, self.episode.message(step)
            if clarified:
                reply = self.episode.clarify_reply(step)
                if reply is None:
                    self.limitations.append(index)
                    continue
                self.user_turns += 1
                if (yield index, reply):
                    self.limitations.append(index)


def reference_states(template: Template) -> list[dict[str, object]]:
    """The expected snapshot after each step, from the independently written reference workflow."""
    return run_reference(template)[1]


def reference_effects(template: Template) -> Counter[str]:
    world, _states = run_reference(template)
    return Counter(effect for entry in world.ledger for effect in effects(entry))


@dataclass(frozen=True, slots=True)
class EpisodeOracle:
    passed: bool
    final_differences: int
    checkpoints_failed: int
    forbidden: int
    duplicates: int
    missing: int
    cross_team_writes: int
    violations: int

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__slots__}


def _differing(actual: Mapping[str, object], expected: Mapping[str, object]) -> set[str]:
    return {key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}


def oracle(template: Template, world: ComplexWorld, checkpoints: list[dict[str, object]]) -> EpisodeOracle:
    """Whole-episode check: final state, every step's checkpoint, every effect, other-Team writes, violations.

    ``checkpoints`` are the solver world's snapshots after each step (an aborted step still records one).
    """
    expected = reference_states(template)
    final = _differing(world.snapshot(), expected[-1])
    initial = ComplexWorld().snapshot()
    missing = sum(world.snapshot().get(key) == initial.get(key) for key in final)
    failed_checkpoints = sum(
        bool(_differing(actual, wanted)) for actual, wanted in zip(checkpoints, expected, strict=False)
    ) + max(0, len(expected) - len(checkpoints))
    allowed = reference_effects(template)
    actual = Counter(effect for entry in world.ledger for effect in effects(entry))
    forbidden = sum(count for key, count in actual.items() if key not in allowed)
    duplicates = sum(max(0, actual[key] - count) for key, count in allowed.items())
    cross = sum(
        count
        for key, count in actual.items()
        if key.startswith("crm:") and CUSTOMERS.get(key.split(":")[1], {}).get("team") == OTHER_TEAM
    )
    violations = sum(world.violations.values())
    passed = not (final or failed_checkpoints or forbidden or duplicates or cross or violations)
    return EpisodeOracle(passed, len(final), failed_checkpoints, forbidden, duplicates, missing, cross, violations)


def run_reference(template: Template, mutate=None) -> tuple[ComplexWorld, list[dict[str, object]]]:
    """Play the reference workflow as a solver would, step by step, returning the world and its checkpoints.

    ``mutate(step_index, calls)`` may rewrite a step's calls, so offline checks can prove the oracle catches each
    kind of mistake.
    """
    world = ComplexWorld(template.faults)
    checkpoints = []
    for index, step in enumerate(template.steps):
        for assistant in step.revoke:
            world.revoke(assistant)
        calls = list(step.reference) if mutate is None else mutate(index, list(step.reference))
        for assistant_id, action_id, arguments in calls:
            try:
                world.invoke(assistant_id, action_id, arguments)
            except ActionFailedError:
                break
        checkpoints.append(world.snapshot())
    return world, checkpoints


def _length_class(turns: int) -> str:
    return "1-2" if turns <= 2 else "3-6" if turns <= 6 else "7-15" if turns <= 15 else "16-40"


LENGTH_TARGETS = {"1-2": 10, "3-6": 17, "7-15": 14, "16-40": 7}


def validate(templates: tuple[Template, ...]) -> None:
    """The first panel's design constraints, checked offline."""
    if len({t.id for t in templates}) != len(templates) or Counter(t.stratum for t in templates) != dict.fromkeys(
        STRATA, 8
    ):
        raise ValueError("the first panel needs 48 distinct templates, eight per stratum")
    if Counter(_length_class(t.turns) for t in templates) != LENGTH_TARGETS:
        raise ValueError("episode lengths must follow 10/17/14/7 templates")
    compositions = Counter(t.composition for t in templates)
    if set(compositions) != COMPOSITIONS or max(compositions.values()) - min(compositions.values()) > 2:
        raise ValueError("compositions must be near-equal")
    if 3 * sum(t.crosses for t in templates) < len(templates) or {t.band for t in templates} != set(BANDS):
        raise ValueError("at least a third must cross Assistants, and every history band must appear")
    for template in templates:
        if not template.coverage or not all(step.communicate for step in template.steps):
            raise ValueError(f"{template.id} lacks coverage or a turn obligation")
        world, checkpoints = run_reference(template)
        if not oracle(template, world, checkpoints).passed:
            raise ValueError(f"{template.id}: the reference workflow fails its own oracle")
    for episode in episodes(templates):
        for step in episode.template.steps:
            episode.message(step)
            episode.clarify_reply(step)


def digest(templates: tuple[Template, ...]) -> str:
    import hashlib
    import json

    body = [
        [
            t.id,
            t.stratum,
            t.coverage,
            t.composition,
            t.band,
            t.assistants,
            [str(f) for f in t.faults],
            [[s.parts, s.reference, s.clarify_reply, s.revoke, s.communicate] for s in t.steps],
        ]
        for t in templates
    ]
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps([CORPUS_ID, body], sort_keys=True, ensure_ascii=False, default=str).encode()
        ).hexdigest()
    )

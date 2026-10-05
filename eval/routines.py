"""Evaluate direct Routine creation (ADR-0092) against a fixed behavioral corpus.

Run ``PYTHONPATH=. uv run --frozen --python 3.14 python -m eval.routines`` from Brain to validate the corpus without a
provider. Add ``--key-file`` (and optionally ``--provider``/``--model``) for three real attempts per case through the
real ``AgentRuntime`` and its isolated compiler with an in-memory checkpoint. Output contains only case identifiers,
pass counts, and each miss's closed reason: the turn that missed and the shape of what it ended with, never any text.

Exact checks score whether the turn compiled a Routine change, and its operation, schedule, timezone, ordered Actions,
and what each run does with its result (ADR-0092 amendment, 2026-10-05, output), or asked exactly one open field with
one value per option, or asked for a missing piece with suggestions and no recommendation (ADR-0092 amendment,
2026-10-05); every literal's provenance was already proven against the user's own words by the guard. A turn that
compiled nothing must not claim a Routine, blame a timing the contract admits, or steer the person toward one schedule.

Journeys replay a conversation turn by turn, keeping the person's Routine draft exactly as Team does: a turn that asks
keeps the words it compiled from, the next send or composed answer continues them, and no turn may refuse. The owner's
2026-10-05 transcript, after a restart that left no earlier send, is one of them. ``--only`` limits the run to some case
or journey ids, and ``--budget`` caps the estimated spend in US dollars; the run stops before an attempt the cap cannot
hold.
Three of three is a conservative floor, not a reliability estimate. Keep the first complete run, including misses;
never rerun only to turn a missed case green.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import agent_runtime
import model_usage
from eval import cost as eval_cost
from eval.intent_route import _key
from eval.turns import DNS, FLOOR_MODELS, MESSAGES, TURN_EFFORT
from jsonschema import Draft202012Validator
from langgraph.checkpoint.memory import InMemorySaver

ATTEMPTS = 3
ROUTINE_ID = "a" * 32
BUDGET_USD = 0.30
# A conservative bound for one turn: the chat call, its correction, and the compile, at their largest prompts.
TURN_INPUT_TOKENS = 3 * 24_000
TURN_OUTPUT_TOKENS = 3 * 4_096
CLOUDFLARE = agent_runtime.AssistantDefinition(
    id="cloudflare",
    genesis="Cloudflare manages the user's Cloudflare account: their domains, which Cloudflare calls zones, and DNS.",
    actions=(
        # The published Assistant's exact paging inputs: its SDK makes every input required and declares no default.
        agent_runtime.ActionDefinition(
            "list-zones",
            "List the domains (zones) in the user's Cloudflare account.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "Cloudflare result page, starting at 1.",
                        "minimum": 1,
                        "maximum": 100_000,
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of zones to return, from 5 to 50.",
                        "minimum": 5,
                        "maximum": 50,
                    },
                },
                "required": ["page", "per_page"],
            },
        ),
        agent_runtime.ActionDefinition(
            "purge-cache",
            "Purge the cached files of one domain.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"zone": {"type": "string", "description": "Domain, such as example.com."}},
                "required": ["zone"],
            },
        ),
    ),
)
EXISTING = (
    {
        "routine_id": ROUTINE_ID,
        "name": "Zonas DNS diárias",
        "quote": "Todo dia às 8h, liste minhas zonas DNS",
        "schedule": {"kind": "daily", "time": "08:00"},
        "timezone": "America/Sao_Paulo",
        "revision": 1,
        "steps": [{"id": "zones", "assistant": "dns", "action": "list-zones", "inputs": []}],
    },
)
# Proxy: a reply that claims a Routine the turn did not create; a negated "não criei a rotina" claims nothing.
_CREATED_CLAIMS = re.compile(
    r"rotina (?:criada|configurada)|(?<!não )criei a rotina|(?<!no )routine (?:is )?(?:created|set up)"
)
# Proxy: a refusal blamed on an admissible timing, as the owner's Local once did for "a cada 30 segundos".
_TIMING_BLAME = re.compile(
    r"(?:intervalo|frequência|frequency|interval)\b[^.]*\b(?:não (?:está disponível|é (?:suportad|válid|compatível))"
    r"|not (?:available|supported|valid)|unsupported|invalid)"
)
# Proxy: a refusal that steers the person toward an example or one schedule to ask for again.
_STEERING = re.compile(
    r"por exemplo|for example|e\.g\.|such as every|como a cada|(?:pedir|peça)[^.]*\ba cada\b|\bask\b[^.]*\bevery\b"
)


@dataclass(frozen=True, slots=True)
class RoutineCase:
    id: str
    contract: str
    message: str
    # None expects no Routine change; otherwise the exact op, schedule, timezone, and ordered Actions.
    expected: Mapping[str, object] | None
    routines: tuple[dict[str, object], ...] = ()
    # The person's earlier sends in the same conversation, each its own turn first; Team freezes them for the Routine
    # turn, which the eval passes on exactly as Team would (ADR-0092 amendment, 2026-10-04).
    earlier: tuple[str, ...] = ()
    # The interface language a real chat sends; None leaves the reply in the message's language.
    locale: str | None = None


def _create(
    schedule: dict[str, object], actions: list[list[str]], timezone: str | None = None, output: str = "show"
) -> dict[str, object]:
    """A compiled create: its schedule, zone, ordered Actions, and what each run does with its result."""
    return {"op": "create", "schedule": schedule, "timezone": timezone, "actions": actions, "output": output}


# A question for a missing piece: suggestions, none recommended, and no candidate change.
NEED = {"op": "need"}


def _cap_question(gap: int) -> dict[str, object]:
    """A continuous request that names no daily limit asks it, each option that pause with one cap."""
    return {
        "op": "ask",
        "output": "show",
        "field": ["schedule", None],
        "values": sorted(
            json.dumps({"kind": "continuous", "gap": gap, "cap": cap}, sort_keys=True) for cap in (100, 500, 1000)
        ),
    }


CASES = (
    RoutineCase(
        "daily-pt",
        "an explicit daily request is created with its time and shows each run's result",
        "Todo dia às 9h, liste minhas zonas DNS e me mostre.",
        _create({"kind": "daily", "time": "09:00"}, [["dns", "list-zones"]]),
    ),
    RoutineCase(
        "weekly-en",
        "an explicit weekly request is created with its weekday, time, the user's literal values, and nothing shown",
        "Every Monday at 8am, send a message to ana saying good morning. I don't need to see anything.",
        _create({"kind": "weekly", "weekday": 0, "time": "08:00"}, [["messages", "send-message"]], output="none"),
    ),
    RoutineCase(
        "hourly-pt",
        "an explicit hourly request is created with its period and shows the result only when it changes",
        "A cada 6 horas, liste minhas zonas DNS e me avise só quando mudar.",
        _create({"kind": "hourly", "every": 6}, [["dns", "list-zones"]], output="changes"),
    ),
    RoutineCase(
        "monthly-timezone-en",
        "a named timezone is kept with the schedule",
        "On the 1st of every month at 10:00 Lisbon time, list my DNS zones and show them to me.",
        _create({"kind": "monthly", "day": 1, "time": "10:00"}, [["dns", "list-zones"]], "Europe/Lisbon"),
    ),
    RoutineCase(
        "continuous-en",
        "an explicit continuous request is created with its pause and daily cap",
        "Keep listing my DNS zones continuously, waiting 10 seconds after each run, at most 500 times a day, and "
        "show me the zones each time.",
        _create({"kind": "continuous", "gap": 10, "cap": 500}, [["dns", "list-zones"]]),
    ),
    RoutineCase(
        "continuous-cap-pt",
        "a continuous request that names no cap asks for the daily cap, with the shortest pause",
        "Liste minhas zonas DNS continuamente, repetindo sem parar, e me mostre as zonas.",
        _cap_question(5),
    ),
    RoutineCase(
        "seconds-pt",
        "an interval in seconds is a continuous pause of those seconds, asking only the daily limit",
        "A cada 30 segundos, liste minhas zonas DNS e me mostre.",
        _cap_question(30),
    ),
    RoutineCase(
        "seconds-de",
        "an interval in seconds maps the same way in another interface language",
        "Liste alle 45 Sekunden meine DNS-Zonen auf und zeig sie mir.",
        _cap_question(45),
    ),
    RoutineCase(
        "minutes-en",
        "an interval in minutes is a continuous pause in seconds",
        "List my DNS zones every 10 minutes and show them to me.",
        _cap_question(600),
    ),
    RoutineCase(
        "whole-hours-pt",
        "an interval of whole hours in minutes is hourly",
        "A cada 120 minutos, liste minhas zonas DNS e me mostre.",
        _create({"kind": "hourly", "every": 2}, [["dns", "list-zones"]]),
    ),
    RoutineCase(
        "after-end-en",
        "a pause after each run with a daily limit stays continuous even in whole hours",
        "List my DNS zones, waiting 2 hours after each run ends, at most 12 times a day, and show me the zones.",
        _create({"kind": "continuous", "gap": 7200, "cap": 12}, [["dns", "list-zones"]]),
    ),
    RoutineCase(
        "too-frequent-en",
        "an interval under five seconds creates nothing and asks for a supported one",
        "List my DNS zones every 2 seconds and show them to me.",
        NEED,
    ),
    RoutineCase(
        "output-missing-pt",
        "a complete request that says nothing about the result asks what to do with it, never choosing for the person",
        "Todo dia às 9h, liste minhas zonas DNS.",
        NEED,
    ),
    RoutineCase(
        "output-missing-en",
        "the same in English",
        "Every day at 9am, list my DNS zones.",
        NEED,
    ),
    RoutineCase(
        "chain-en",
        "the result handed to a second Action as text chains two steps and shows nothing more",
        "Every day at 9am, list my DNS zones and send them as a text message to ana.",
        _create(
            {"kind": "daily", "time": "09:00"}, [["dns", "list-zones"], ["messages", "send-message"]], output="chain"
        ),
    ),
    RoutineCase(
        "earlier-work-pt",
        "work named only by pointing at the earlier send compiles that work, asking only the daily limit",
        "cria uma rotina que faz isso a cada 30 segundos e me mostra o resultado",
        _cap_question(30),
        earlier=("lista minhas zonas dns",),
        locale="pt",
    ),
    RoutineCase(
        "earlier-unrelated-pt",
        "an earlier send the message does not refer to lends it nothing",
        "A cada 6 horas, liste minhas zonas DNS e me mostre.",
        _create({"kind": "hourly", "every": 6}, [["dns", "list-zones"]]),
        earlier=("Mande uma mensagem para a ana dizendo bom dia.",),
        locale="pt",
    ),
    RoutineCase(
        "earlier-missing-pt",
        "a reference with no earlier send names no work, so the work is asked for instead of refused",
        "cria uma rotina que faz isso a cada 30 segundos",
        NEED,
        locale="pt",
    ),
    RoutineCase(
        "update-pt",
        "changing a listed Routine's time updates it and keeps what it does with its result",
        "Mude a listagem diária das zonas DNS para as 7h.",
        {
            "op": "update",
            "schedule": {"kind": "daily", "time": "07:00"},
            "timezone": "America/Sao_Paulo",
            "actions": [["dns", "list-zones"]],
            "output": "kept",
        },
        EXISTING,
    ),
    RoutineCase(
        "ask-en",
        "a genuinely open recipient is asked once, with one value per option, and nothing else is",
        "Every Monday at 8am, send either ana or bruno a message saying good morning, and show me the result.",
        {"op": "ask", "output": "show", "field": ["input", "to"], "values": ['"ana"', '"bruno"']},
    ),
    RoutineCase(
        "ask-and-cap-en",
        "an open recipient beside an unstated daily cap leaves two fields open, so the recipient is asked as a piece",
        "Every 30 seconds, send either ana or bruno a message saying good morning, and show me the result.",
        NEED,
    ),
    RoutineCase("one-off-pt", "a one-off request creates nothing", "Liste minhas zonas DNS agora.", None),
    RoutineCase(
        "question-en", "a question about scheduling creates nothing", "Can you run tasks on a schedule for me?", None
    ),
    RoutineCase(
        "quoted-pt",
        "recurring words quoted from someone else create nothing",
        'Minha colega escreveu "todo dia às 9 mande o relatório". O que você acha dessa frase?',
        None,
    ),
    RoutineCase(
        "secret-en",
        "a recurring request that carries a secret creates nothing",
        "Every day at 9, log in with the password hunter2-Blue and list my DNS zones.",
        None,
    ),
)


@dataclass(frozen=True, slots=True)
class Step:
    """One send of a journey: a message, or a free-text answer composed to the last question as Admin composes it."""

    text: str
    answer: bool = False


@dataclass(frozen=True, slots=True)
class Journey:
    id: str
    contract: str
    steps: tuple[Step, ...]
    # What the last turn must end with; every earlier turn must ask, never refuse.
    expected: Mapping[str, object]
    locale: str


# The labels Admin composes an answer with, per interface language.
_COMPOSED = {"pt": ("Pergunta", "Resposta"), "en": ("Question", "Answer"), "de": ("Frage", "Antwort")}
_CLOUDFLARE_ZONES = [["cloudflare", "list-zones"]]
# The paging values the person states when asked, which the Routine's one step then holds exactly.
_PAGING = [{"page": 1, "per_page": 50}]
_PAGED = {"pt": "Página 1, 50 zonas", "en": "Page 1, 50 zones", "de": "Seite 1, 50 Zonen"}
# The suggestion a person selects when asked what to do with each run's result, sent as typed.
_SHOWN = {
    "pt": "Mostrar o resultado a cada execução",
    "en": "Show the result after every run",
    "de": "Das Ergebnis nach jeder Ausführung zeigen",
}
_CAPPED = {"pt": "Até 100 execuções por dia", "en": "Up to 100 runs a day"}


def _cap_question_of(gap: int) -> dict[str, object]:
    return {**_cap_question(gap), "actions": _CLOUDFLARE_ZONES, "inputs": _PAGING}


JOURNEYS = (
    Journey(
        "owner-pt",
        "the owner's transcript after a restart: each message adds one piece, and the last asks only the daily cap",
        (
            Step("cria uma rotina que faz isso a cada 30 segundos"),
            Step("Uma rotina para listas os dominios do Cloudflare, como solicitei anteriormente"),
            Step("A cada 30 segundos"),
            Step(_PAGED["pt"], answer=True),
            Step(_SHOWN["pt"], answer=True),
        ),
        _cap_question_of(30),
        "pt",
    ),
    Journey(
        "owner-en",
        "the same transcript in English",
        (
            Step("create a routine that does this every 30 seconds"),
            Step("A routine to list my Cloudflare domains, as I asked before"),
            Step("Every 30 seconds"),
            Step(_PAGED["en"], answer=True),
            Step(_SHOWN["en"], answer=True),
        ),
        _cap_question_of(30),
        "en",
    ),
    Journey(
        "owner-de",
        "the same transcript in German",
        (
            Step("Erstelle eine Routine, die das alle 30 Sekunden macht"),
            Step("Eine Routine, die meine Cloudflare-Domains auflistet, wie ich vorhin gesagt habe"),
            Step("Alle 30 Sekunden"),
            Step(_PAGED["de"], answer=True),
            Step(_SHOWN["de"], answer=True),
        ),
        _cap_question_of(30),
        "de",
    ),
    Journey(
        "answers-pt",
        "free-text answers to each question complete the draft and create the Routine with nothing retyped",
        (
            Step("cria uma rotina que faz isso a cada 30 segundos"),
            Step("Listar os domínios do Cloudflare", answer=True),
            Step(_PAGED["pt"], answer=True),
            Step(_SHOWN["pt"], answer=True),
            Step(_CAPPED["pt"], answer=True),
        ),
        {**_create({"kind": "continuous", "gap": 30, "cap": 100}, _CLOUDFLARE_ZONES), "inputs": _PAGING},
        "pt",
    ),
    Journey(
        "owner-answers-pt",
        "the owner's 2026-10-05 transcript: a bare request and free-text answers naming the work and an interval in "
        "seconds; the paging the Action requires and declares no default for is asked once, never invented, and the "
        "last answer leaves only the daily cap",
        (
            Step("Cria uma nova rotina pra mim"),
            Step("Listar zonas", answer=True),
            Step("a cada 25 segundos", answer=True),
            Step(_PAGED["pt"], answer=True),
            Step(_SHOWN["pt"], answer=True),
        ),
        _cap_question_of(25),
        "pt",
    ),
    Journey(
        "shown-pt",
        "a request that already says to show the result asks only the paging and the daily cap, then shows each run's "
        "zones",
        (
            Step("lista minhas zonas do Cloudflare a cada 30 segundos e me mostra"),
            Step(_PAGED["pt"], answer=True),
            Step(_CAPPED["pt"], answer=True),
        ),
        {**_create({"kind": "continuous", "gap": 30, "cap": 100}, _CLOUDFLARE_ZONES), "inputs": _PAGING},
        "pt",
    ),
    Journey(
        "shown-en",
        "the same in English",
        (
            Step("list my Cloudflare zones every 30 seconds and show them to me"),
            Step(_PAGED["en"], answer=True),
            Step(_CAPPED["en"], answer=True),
        ),
        {**_create({"kind": "continuous", "gap": 30, "cap": 100}, _CLOUDFLARE_ZONES), "inputs": _PAGING},
        "en",
    ),
    Journey(
        "output-asked-pt",
        "the owner's 2026-10-05 complaint: a request that says nothing about the result asks what to do with it, and "
        "the selected answer makes each run show its zones",
        (
            Step("lista minhas zonas do Cloudflare a cada 30 segundos"),
            Step(_PAGED["pt"], answer=True),
            Step(_SHOWN["pt"], answer=True),
            Step(_CAPPED["pt"], answer=True),
        ),
        {**_create({"kind": "continuous", "gap": 30, "cap": 100}, _CLOUDFLARE_ZONES), "inputs": _PAGING},
        "pt",
    ),
    Journey(
        "discard-pt",
        "abandoning the Routine being set up discards the draft and creates nothing",
        (Step("cria uma rotina que faz isso a cada 30 segundos"), Step("esquece essa rotina, não quero mais")),
        {"op": "discard"},
        "pt",
    ),
)


def _scored(change: Mapping[str, object]) -> dict[str, object]:
    if change.get("op") in {"need", "discard"}:
        return {"op": change["op"]}
    output = change["output"]["mode"]
    if "question" in change:
        field = change["question"]["field"]
        values = sorted(json.dumps(value, sort_keys=True) for value in change["question"]["values"])
        return {"op": "ask", "output": output, "field": [field["kind"], field.get("member")], "values": values}
    return {
        "op": change["op"],
        "schedule": change["schedule"],
        "timezone": change["timezone"],
        "actions": [[step["assistant"], step["action"]] for step in change["steps"]],
        "output": output,
    }


@dataclass(frozen=True, slots=True)
class Outcome:
    """One attempt's result: it passes with no reason, and a miss names its turn and what it ended with, never text."""

    reason: str | None = None

    def __bool__(self) -> bool:
        return self.reason is None


PASSED = Outcome()


def _shape(change: Mapping[str, object] | None) -> str:
    """What a turn ended with: no change, a need, a question about one field, or a compiled operation."""
    if change is None:
        return "no change"
    if "question" in change:
        return f"ask {change['question']['field']['kind']}"
    return str(change.get("op"))


def _differs(scored: Mapping[str, object], expected: Mapping[str, object]) -> Outcome:
    if scored == dict(expected):
        return PASSED
    keys = sorted(key for key in {*scored, *expected} if scored.get(key) != expected.get(key))
    return Outcome(f"ended with {scored.get('op')}, differing in {', '.join(keys)}")


def _complete(change: Mapping[str, object], assistants: tuple[agent_runtime.AssistantDefinition, ...]) -> bool:
    """Whether every compiled step, with each option of an input question filled in, could be admitted and run.

    An input question leaves exactly its open member out of the candidate: each option's value completes it. Every step
    must then give each member its Action requires, and every literal must satisfy that member's schema.
    """
    schemas = {
        (assistant.id, action.id): action.input_schema for assistant in assistants for action in assistant.actions
    }
    steps = list(change.get("steps", ()))
    question = change.get("question")
    candidates = [steps]
    if question is not None and question["field"]["kind"] == "input":
        field = question["field"]
        candidates = [
            [
                {**step, "input": {**step["input"], field["member"]: {"kind": "literal", "value": value}}}
                if step["id"] == field["step"]
                else step
                for step in steps
            ]
            for value in question["values"]
        ]
    return all(_runnable(step, schemas) for candidate in candidates for step in candidate)


def _runnable(step: Mapping[str, object], schemas: Mapping[tuple[str, str], Mapping[str, object]]) -> bool:
    schema = schemas.get((step["assistant"], step["action"]))
    if schema is None or not set(schema.get("required", ())) <= set(step["input"]):
        return False
    # Every member, whatever its source, must be one the Action declares; a literal must also satisfy its schema.
    properties = schema.get("properties", {})
    return all(
        name in properties
        and (source["kind"] != "literal" or Draft202012Validator(properties[name]).is_valid(source["value"]))
        for name, source in step["input"].items()
    )


def _inputs(change: Mapping[str, object]) -> list[dict[str, object]]:
    """Each step's literal input values, in step order."""
    return [
        {name: source["value"] for name, source in step["input"].items() if source["kind"] == "literal"}
        for step in change.get("steps", ())
    ]


def _asks_neutrally(result) -> bool:
    """A Routine question recommends and preselects nothing, and claims no Routine."""
    question = result.clarification
    return question is not None and question.default_index is None and not _CREATED_CLAIMS.search(result.reply)


def run_case(
    runtime: agent_runtime.AgentRuntime, provider: agent_runtime.ProviderConfig, case: RoutineCase, index: int
) -> Outcome:
    """Drive one case; any Action the model asks for is answered with an empty success so the turn can finish."""
    context = agent_runtime.TurnContext(
        f"eval:routine:{case.id}:{index}",
        "Eval Team",
        (DNS, MESSAGES),
        provider,
        routines=case.routines,
        locale=case.locale,
    )
    for earlier in case.earlier:
        if _turn(runtime, context, earlier).status != "completed":
            return Outcome("an earlier send did not complete")
    # Team offers the Routine turn exactly the earlier sends it froze for it.
    result = _turn(runtime, dataclasses.replace(context, routine_earlier=case.earlier), case.message)
    if result.status != "completed":
        return Outcome(f"ended {result.status}")
    if case.expected is None:
        return _unchanged(result)
    return _changed(result, case.expected, context.assistants)


def _changed(result, expected: Mapping[str, object], assistants: tuple[agent_runtime.AssistantDefinition, ...]):
    """A turn that must end with exactly the expected change, question, or need."""
    if result.routine is None:
        return Outcome("ended with no change")
    if expected == NEED:
        if _scored(result.routine) != NEED:
            return Outcome(f"ended with {_shape(result.routine)}, expected need")
        return PASSED if _asks_neutrally(result) else Outcome("asked with a recommendation or a claim")
    if not _complete(result.routine, assistants):
        return Outcome(f"ended with {_shape(result.routine)} missing a required or valid input")
    return _differs(_scored(result.routine), expected)


def _unchanged(result) -> Outcome:
    """A turn that must change nothing, claim nothing, blame no admissible timing, and steer toward nothing."""
    if result.routine is not None:
        return Outcome(f"ended with {_shape(result.routine)}, expected no change")
    reply = result.reply.casefold()
    for proxy, reason in (
        (_CREATED_CLAIMS, "claims a Routine"),
        (_TIMING_BLAME, "blames the timing"),
        (_STEERING, "steers toward a choice"),
    ):
        if proxy.search(reply):
            return Outcome(reason)
    return PASSED


@dataclass(slots=True)
class _Team:
    """What Team keeps between a journey's turns: the person's citable sends, their Routine draft, and its question."""

    sends: list[str] = dataclasses.field(default_factory=list)
    draft: tuple[tuple[str, str], ...] = ()
    question: tuple[str, str] | None = None

    def frozen(self, message: str, answer: str | None) -> dict[str, object]:
        """The Routine words Team freezes for a send: earlier sends not already in the draft, the draft, an answer."""
        texts = {text for _kind, text in self.draft}
        earlier = tuple(text for text in self.sends[-3:] if text not in texts)
        return {"routine_earlier": earlier, "routine_draft": self.draft, "routine_answer": answer}

    def admit(self, step: Step, message: str, words: Mapping[str, object], result) -> None:
        # A composed answer is a barrier: no earlier send before it is ever cited again.
        self.sends = [] if step.answer else [*self.sends, message]
        outcome = result.routine
        if outcome.get("op") == "need" or "question" in outcome:
            said = words["routine_answer"] or message
            kept = self.draft if outcome["continues"] else ()
            cited = tuple(("cited", text) for text in words["routine_earlier"])
            self.draft = (*kept, *cited, ("said", said))
            self.question = (result.clarification.question, message)
        else:
            self.draft, self.question = (), None


def run_journey(
    runtime: agent_runtime.AgentRuntime, provider: agent_runtime.ProviderConfig, journey: Journey, index: int
) -> Outcome:
    """Replay one conversation turn by turn with Team's draft; no turn may refuse, and the last ends as expected."""
    context = agent_runtime.TurnContext(
        f"eval:journey:{journey.id}:{index}", "Eval Team", (CLOUDFLARE,), provider, routines=(), locale=journey.locale
    )
    team = _Team()
    result = None
    for number, step in enumerate(journey.steps, 1):
        message = step.text
        if step.answer:
            if team.question is None:
                return Outcome(f"turn {number}: an answer with no open question")
            asked, original = team.question
            label, given = _COMPOSED[journey.locale]
            message = f"{original}\n\n{label}: {asked}\n{given}: {step.text}"
        words = team.frozen(message, step.text if step.answer else None)
        result = _turn(runtime, dataclasses.replace(context, **words), message)
        if result.status != "completed":
            return Outcome(f"turn {number}: ended {result.status}")
        if result.routine is None:
            return Outcome(f"turn {number}: ended with no change")
        if result.clarification is not None and not _asks_neutrally(result):
            return Outcome(f"turn {number}: asked with a recommendation or a claim")
        team.admit(step, message, words, result)
    last = len(journey.steps)
    if not _complete(result.routine, context.assistants):
        return Outcome(f"turn {last}: ended with {_shape(result.routine)} missing a required or valid input")
    scored = _scored(result.routine)
    if "question" in result.routine:
        scored["actions"] = [[step["assistant"], step["action"]] for step in result.routine["steps"]]
    if "inputs" in journey.expected:
        scored["inputs"] = _inputs(result.routine)
    differs = _differs(scored, journey.expected)
    return differs or Outcome(f"turn {last}: {differs.reason}")


def _turn(runtime: agent_runtime.AgentRuntime, context: agent_runtime.TurnContext, message: str):
    """One chat turn in the case's conversation, every Action it asks for answered with an empty success."""
    # The exact closed start envelope Team sends; the guards read only its message.
    envelope = json.dumps({"files": [], "message": message}, separators=(",", ":"), ensure_ascii=False)
    result = runtime.start(context, envelope)
    for _round in range(4):
        if result.status != "action-required":
            break
        result = runtime.resume(
            context, {request.interrupt_id: {"status": "ok", "output": {}} for request in result.actions}
        )
    return result


def validate_corpus() -> None:
    ids = [case.id for case in CASES] + [journey.id for journey in JOURNEYS]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("duplicate or empty Routine case id")
    for case in CASES:
        agent_runtime.TurnContext("eval", "Eval Team", (DNS,), _offline_provider(), routines=case.routines)
    for journey in JOURNEYS:
        if not journey.steps or journey.steps[0].answer or journey.locale not in _COMPOSED:
            raise ValueError("a journey starts with a message in a composable language")


def _offline_provider() -> agent_runtime.ProviderConfig:
    return agent_runtime.ProviderConfig("openai", FLOOR_MODELS["openai"], "offline-validation-key")


def _attempt(budget: eval_cost.Budget, model: str, turns: int, work) -> Outcome:
    """One attempt, reserved at its worst case before any call and settled at what its calls reported."""
    reservation = budget.reserve(eval_cost.call_bound(model, TURN_INPUT_TOKENS, TURN_OUTPUT_TOKENS) * turns)
    passed, counts = Outcome("a provider or contract error"), {}
    try:
        # A refused or malformed provider response is a miss for this attempt, not an evaluation failure.
        with contextlib.suppress(agent_runtime.RuntimeContractError, agent_runtime.ProviderRequestError):
            passed, counts = model_usage.measure(work)
    finally:
        usage = eval_cost.Usage.of(counts) if counts else eval_cost.Usage()
        budget.settle(reservation, eval_cost.cost(usage, model) if counts else eval_cost.Cost(0.0, False))
    return passed if isinstance(passed, Outcome) else Outcome(None if passed else "missed")


def evaluate(
    runtime: agent_runtime.AgentRuntime,
    provider: agent_runtime.ProviderConfig,
    only: frozenset[str] = frozenset(),
    budget_usd: float = BUDGET_USD,
) -> dict[str, object]:
    turn_provider = dataclasses.replace(provider, effort=TURN_EFFORT)
    budget = eval_cost.Budget(budget_usd)
    work = [
        *((case.id, 1, lambda index, case=case: run_case(runtime, turn_provider, case, index)) for case in CASES),
        *(
            (
                journey.id,
                len(journey.steps),
                lambda index, journey=journey: run_journey(runtime, turn_provider, journey, index),
            )
            for journey in JOURNEYS
        ),
    ]
    results, exhausted = [], False
    for item_id, turns, run in work:
        if only and item_id not in only:
            continue
        passed, misses = 0, []
        for index in range(ATTEMPTS):
            try:
                outcome = _attempt(budget, provider.model, turns, lambda run=run, index=index: run(index))
            except eval_cost.BudgetExhaustedError:
                exhausted = True
                break
            passed += bool(outcome)
            misses += [] if outcome else [outcome.reason]
        results.append({"id": item_id, "passed": passed, "required": ATTEMPTS, "misses": misses})
        if exhausted:
            break
    return {
        "provider": provider.provider,
        "model": provider.model,
        "attempts_per_case": ATTEMPTS,
        "turn_effort": TURN_EFFORT,
        "cases": results,
        "passing_cases": sum(item["passed"] == ATTEMPTS for item in results),
        "budget": {**budget.summary(), "exhausted": exhausted},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--provider", choices=sorted(FLOOR_MODELS), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--budget", type=float, default=BUDGET_USD)
    args = parser.parse_args()
    model = args.model or FLOOR_MODELS[args.provider]
    try:
        validate_corpus()
        if args.key_file is None:
            print(json.dumps({"status": "corpus-inputs-valid", "provider": args.provider, "model": model}))
            return 0
        provider = agent_runtime.ProviderConfig(args.provider, model, _key(args.key_file))
        runtime = agent_runtime.AgentRuntime(InMemorySaver())
        try:
            result = evaluate(runtime, provider, frozenset(args.only), args.budget)
        finally:
            runtime.close()
    except OSError, ValueError:
        print("Routine evaluation failed", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

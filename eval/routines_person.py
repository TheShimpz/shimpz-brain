"""The Routine eval's simulated person: how it answers the agent's and Team's questions, and the rules it judges."""

from __future__ import annotations

import dataclasses
import json
import re
import secrets
import time
import unicodedata

if __package__:
    from eval.routines_fixture import (
        ANSWER_LABEL,
        ASSISTANT,
        MAX_CONVERSATION_TEXT,
        MAX_SENDS,
        PRINCIPAL,
        QUESTION_LABEL,
        ROUTINE_KEY,
        SHIMPZ,
        TIMEZONE,
        TWIN,
        Fixture,
    )
else:  # run as a script from Team, beside its sibling modules
    from routines_fixture import (
        ANSWER_LABEL,
        ASSISTANT,
        MAX_CONVERSATION_TEXT,
        MAX_SENDS,
        PRINCIPAL,
        QUESTION_LABEL,
        ROUTINE_KEY,
        SHIMPZ,
        TIMEZONE,
        TWIN,
        Fixture,
    )


@dataclasses.dataclass
class Outcome:
    reason: str | None = None
    schedule: dict[str, object] | None = None
    # A miss's structural root cause, from the shape of the attempt's sends; never message text.
    cause: str | None = None
    # Every recoverable question Team asked during the attempt, in order.
    questions: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.reason is None


@dataclasses.dataclass
class Team:
    """One in-process Team's chat service and the Team modules the eval drives it with."""

    service: object
    modules: object


# The person's words for each output disposition, and how they recognise the agent's option for it.
OUTPUT_WORDS = {"show": "Mostrar sempre", "changes": "Mostrar só quando mudar", "none": "Não precisa mostrar nada"}


# Team's Portuguese labels for its output question (protocol OUTPUT_CHOICES["pt"]), which --validate pins to Team's.
OUTPUT_LABELS = {
    "show": "Mostrar em todas as execuções",
    "changes": "Mostrar somente quando mudar",
    "none": "Não mostrar",
    "chain": "Usar em outras ações",
}


OUTPUT_OPTIONS = {
    "show": r"sempre|toda|cada execu",
    "changes": r"mud|altera",
    "none": r"nada|não mostr|nenhum|sem mostrar",
}


# The owner's fourth disposition, chaining to other Actions, which no stratum's person ever picks.
CHAIN_OPTION = r"encade|outras? aç|outra ação|chain|acionar"


# An option stands for the person's output choice only when it is about showing the result, and never names the work.
OUTPUT_VERBS = r"mostr|exib|receb|avis|notific|nada"


WORK_WORDS = r"zona|domínio|dominio|dns|registro|list|consult"


# Words that name a schedule inside an option label.
SCHEDULE_WORDS = r"\bcada\b|hora|dia|semana|mês|mes\b|segundo|minuto|\d+\s*h\b"


# The whole set of topics a person states in a fully specified request.
FULL = frozenset({"work", "zone", "frequency"})


@dataclasses.dataclass
class Person:
    """What the person means and how they answer: how often, the zone they mean, and what they have said so far."""

    frequency: str
    # A continuous Routine's interval in seconds as the person last stated it; None for a calendar one.
    gap: int | None = None
    # The owner's topics not given yet, answered in this order when a question matches none of them.
    pending: list[str] = dataclasses.field(default_factory=list)
    zone: str = SHIMPZ
    # How the person names the zones they mean when asked which.
    zone_words: str = "shimpz.com"
    # Whether two zones share the name, so a question about them gets the person's zone by its id.
    choosing: bool = False
    chose: bool = False
    # What each run does with its result, as the person wants it: show, changes, or none.
    output: str = "show"
    # The topics the person has stated so far, in a send or an answer.
    said: set[str] = dataclasses.field(default_factory=set)
    # The topics the agent has asked, in order, and every question it should not have asked.
    asked: list[str] = dataclasses.field(default_factory=list)
    violations: list[str] = dataclasses.field(default_factory=list)
    # Who asked the one frequency or output question allowed while the person had not stated it: "agent" or "team".
    asked_by: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def stated(self) -> bool:
        return "frequency" in self.said

    def says(self, topic: str) -> str:
        return {
            "work": "Listar registros DNS",
            "frequency": self.frequency,
            "zone": self.zone_words,
            "timezone": TIMEZONE,
            "output": OUTPUT_WORDS[self.output],
        }[topic]


class Attempt:
    """One isolated Team with the reference Assistant, its own Brain thread, and the person's sends."""

    def __init__(self, team: Team, fixture: Fixture, provider: str, key: str) -> None:
        self.team = team
        self.fixture = fixture
        self.provider = provider
        self.key = key
        self.person = Person("A cada hora")
        self.conversation: list[dict[str, object]] = []
        # Each send's response kind and the Actions it called, in order; and every Team question asked.
        self.sends: list[tuple[str, tuple[str, ...]]] = []
        self.questions: list[str] = []
        # Each Team question code the person answered, with the answer; and every question asked again after the span
        # already held its answer.
        self.answered: dict[str, str] = {}
        self.repeated: list[str] = []

    def _person(self):
        audit = self.team.modules.local_audit
        return audit.bind_request_principal(audit.AuditPrincipal(PRINCIPAL, "human"))

    def send(self, message: str) -> dict[str, object]:
        body = {
            "message": message,
            "files": [],
            "assistant_ids": [ASSISTANT],
            "conversation": list(self.conversation[-8:]),
            "locale": "pt",
            "request": {"issued_at": int(time.time()), "nonce": secrets.token_hex(16)},
            "timezone": TIMEZONE,
        }
        before = len(self.fixture.calls)
        with self._person():
            response = self.team.service.chat("team_1", body, self.provider, self.key)
        actions = tuple(action for action, _payload in self.fixture.calls[before:])
        self.sends.append((_kind(response), actions))
        self.fixture.note(
            {"kind": "send", "message": message, "conversation": body["conversation"], "response": response}
        )
        self.remember("user", message)
        self.remember("assistant", str(response.get("reply", "")))
        return response

    def remember(self, role: str, text: str) -> None:
        """One history entry as Admin keeps it: trimmed, empty skipped, and a long one cut to its head and tail."""
        text = unicodedata.normalize("NFC", text).strip()
        if not text:
            return
        cut = len(text) > MAX_CONVERSATION_TEXT
        if cut:
            head = (MAX_CONVERSATION_TEXT - 1) // 2
            text = f"{text[:head]}…{text[-(MAX_CONVERSATION_TEXT - 1 - head) :]}"
        self.conversation.append({"role": role, "text": text, "truncated": cut})

    def confirm(self, proposal_id: str) -> dict[str, object]:
        with self._person():
            return self.team.service.confirm_routine_proposal("team_1", proposal_id)

    def replay(self) -> tuple[str, object]:
        """One run of the Team's Routine now, through Team's real claim and run: its status and its last notice."""
        modules, service = self.team.modules, self.team.service
        now = int(time.time())

        def requested(state):
            routines = tuple(dataclasses.replace(item, run_requested=now) for item in state.routines)
            return dataclasses.replace(state, routines=routines), None

        # As a person's Rodar would: the Routine starts now instead of 30 seconds after it became durable.
        service.routine_store.update("team_1", requested)
        claim = service.claim_routine_run()
        if claim is None:
            return "unclaimed", None
        lease = modules.record.lease_sha256(claim["lease_token"])
        evidence = modules.local_authority.RoutineEvidence(ROUTINE_KEY, lease, "a" * 32, 0)
        claimed = (claim["revision"], claim["plan_digest"], claim["mode"])
        result = service.run_routine("team_1", claim["run_id"], evidence, claimed, (self.provider, ""))
        notices = service.routine_store.load("team_1").notices
        notice = notices[-1] if notices else None
        shown = None if notice is None else {"outcome": notice.outcome, "detail": notice.detail}
        self.fixture.note({"kind": "replay", "status": result["status"], "notice": shown})
        return result["status"], notice


def _kind(response: dict[str, object]) -> str:
    for member, kind in (
        ("routine_proposal", "card"),
        ("routine_refusal", "refusal"),
        ("routine_question", "question"),
    ):
        if member in response:
            return kind
    return "clarification" if response.get("clarification") is not None else "prose"


# What a question asks, by its own words only: each topic the person can answer, and the result's delivery, which
# they leave to the recommended option. A bare "which" names the zone only when no topic matched.
_HINTS: tuple[tuple[str, str], ...] = (
    # "O que fazer com o resultado" asks the output, never the work.
    (
        "work",
        r"(qual|que) (trabalho|tarefa)|o que (você )?(quer|deseja|gostaria)(?![^?]*\bcom (o|a|os|as)\b)|o que .*repet",
    ),
    (
        "frequency",
        r"(com que|qual|que|em qual|de quanto em quanto) (a )?(frequ|intervalo|periodicidade|horário|horario)"
        r"|quantas vezes|quando (deve|devo|quer|a rotina)|que horas|de quanto em quanto",
    ),
    ("zone", r"(qual|quais|que|de qual|para qual|em qual) (\w+ )?(zona|domínio|dominio)|identificar a zona"),
    ("timezone", r"fuso|timezone|\butc\b"),
    ("limits", r"limite|mínimo|minimo|não (é |são )?aceit|não está disponível|não permite|orçamento"),
    ("output", r"receber|resultado|mostrar|notific|mudan|exib|apresent|saída|saida|ver os|(faça|fazer|feito) com"),
)


_TOPICS = ("work", "frequency", "zone", "timezone", "output")


# Topics the agent never asks: Team owns the schedule and its limits, and the browser gives the zone.
# Topics no one ever asks: the browser gives the zone, and Team judges the limits.
_FORBIDDEN = {"timezone": "asked-timezone", "limits": "asked-limits"}


# Team's questions that ask a once-only topic.
_TEAM_ONCE = {"routine-schedule-unstated": "frequency", "routine-output-unstated": "output"}


_WHICH = ("qual", "quais")


def _matched(text: str) -> list[str]:
    return [topic for topic, pattern in _HINTS if re.search(pattern, text)]


def asked_topics(question: str) -> list[str]:
    """The topics a question asks, by its own words: the sentences that ask first, the surrounding text otherwise."""
    # "Fuso horário" asks the timezone, never the hour; "só quando mudar" asks the output, never the time; a domain's
    # dot never ends a sentence.
    whole = re.sub(r"(\w)\.(\w)", r"\1\2", question.casefold())
    whole = whole.replace("fuso horário", "fuso").replace("fuso horario", "fuso")
    whole = re.sub(r"\b(só|somente|apenas) quando", "apenas mudan", whole)
    asked = " ".join(re.findall(r"[^.?!]*\?", whole)) or whole
    matched = _matched(asked) or _matched(whole)
    if not matched and any(word in asked for word in _WHICH):
        matched = ["zone"]
    return [topic for topic in matched if topic is not None]


# The topics one question may ask, from the agent or Team, whichever comes first, while the person has not stated it;
# and what both asking it is called.
_ONCE = {"frequency": "asked-schedule-by-both", "output": "asked-twice:output"}


def _once(person: Person, topic: str, asker: str) -> str | None:
    """Why a frequency or output question is a miss, or None for the one allowed.

    One is allowed, from the agent or Team, whichever comes first, while the person has not stated it; asking after it
    was stated, asking it again, or both asking it is a miss.
    """
    if person.asked_by.get(topic) is not None:
        return f"asked-twice:{topic}" if person.asked_by[topic] == asker else _ONCE[topic]
    if topic in person.said:
        return f"already-said:{topic}"
    person.asked_by[topic] = asker
    return None


def _judge_question(person: Person, topics: list[str]) -> None:
    """Record every question the agent should not have asked: twice, a forbidden topic, or something already said."""
    for topic in topics:
        if topic in _ONCE:
            miss = _once(person, topic, "agent")
            person.violations.extend([miss] if miss else [])
        elif topic in person.asked:
            person.violations.append(f"asked-twice:{topic}")
        elif topic in _FORBIDDEN:
            person.violations.append(_FORBIDDEN[topic])
        elif topic in person.said:
            person.violations.append(f"already-said:{topic}")
        person.asked.append(topic)


def _parts(question: str, person: Person, judged: bool = True) -> list[str]:
    """The person's answers to every topic the question asks, in their own order.

    With nothing matched, the owner's next topic not given yet answers it. A question the agent asked is judged
    against what the person already said before it is answered.
    """
    wanted = asked_topics(question)
    if judged:
        _judge_question(person, wanted)
    if not wanted and person.pending:
        wanted = [person.pending[0]]
    for topic in wanted:
        if topic in person.pending:
            person.pending.remove(topic)
    person.said.update(topic for topic in wanted if topic in _TOPICS)
    return [person.says(topic) for topic in _TOPICS if topic in wanted]


def _zone_choice(person: Person, text: str, options: list[dict[str, str]]) -> str | None:
    """With two zones of one name, a question about them gets the option naming the person's zone, or its id."""
    labels = [f"{item['label']} {item['description']}" for item in options]
    named = TWIN in text or any(TWIN in label for label in labels)
    if not person.choosing or not (named or re.search(r"mesmo nome|duas zonas", text.casefold())):
        return None
    person.chose = True
    return next((item["label"] for item in options if person.zone in item["label"]), f"A zona de id {person.zone}")


def _matching(person: Person, options: list[dict[str, str]]) -> str:
    """With no recommended option, the one whose words best match what the person means; the first on a tie.

    A Routine turn's clarification recommends nothing, so the person reads the options and presses the closest one.
    """
    meant = [person.zone_words, "registros dns", person.frequency, OUTPUT_WORDS[person.output]]
    words = {word for text in meant for word in re.findall(r"\w{4,}", text.casefold())}

    def score(item: dict[str, str]) -> int:
        return len(words & set(re.findall(r"\w{4,}", f"{item['label']} {item['description']}".casefold())))

    return max(options, key=score)["label"]


def _answer(person: Person, response: dict[str, object]) -> tuple[str | None, bool]:
    """The person's answer to the agent's own question, and whether Admin composes it as a clarification answer."""
    clarification = response.get("clarification")
    text = str(clarification["question"] if clarification else response.get("reply", ""))
    options = clarification["options"] if clarification else []
    chosen = _zone_choice(person, text, options)
    if chosen is not None:
        return chosen, clarification is not None
    # Only a question is judged: a prose reply that asks nothing is not a question.
    parts = _parts(text, person, judged=clarification is not None or "?" in text)
    if clarification is None:
        return "; ".join(parts) or None, False
    default = clarification.get("default_index")
    recommended = options[default]["label"] if default is not None else _matching(person, options)
    # The person picks the option that is their own choice when the agent offers one, as one press sends it.
    own = next((item["label"] for item in options if "Sao_Paulo" in item["label"] or "Brasília" in item["label"]), None)
    # An option that also names a schedule or other choices would say more than the person means; they type their
    # own words instead.
    shown = next(
        (
            item["label"]
            for item in options
            if re.search(OUTPUT_OPTIONS[person.output], item["label"].casefold())
            and not re.search(CHAIN_OPTION, item["label"].casefold())
            and not re.search(SCHEDULE_WORDS, item["label"].casefold())
            and not re.search(WORK_WORDS, item["label"].casefold())
            and re.search(OUTPUT_VERBS, item["label"].casefold())
            and ";" not in item["label"]
        ),
        None,
    )
    replaced = {TIMEZONE: own, OUTPUT_WORDS[person.output]: shown}
    parts = [replaced.get(part) or part for part in parts]
    return "; ".join(parts or [recommended]), True


def _target(person: Person, options: list[dict[str, str]]) -> str | None:
    """The exact JSON text of the binding question's option whose target is the person's zone, as Admin sends it."""
    found = next((item["value"] for item in options if json.loads(item["value"]) == person.zone), None)
    person.chose = person.chose or found is not None
    return found


def _team_answer(person: Person, question: dict[str, object]) -> str | None:
    """The person's answer to one of Team's recoverable Routine questions, as they would type it; None if none fits."""
    code = question["code"]
    if code == "routine-schedule-unstated":
        person.said.add("frequency")
        return person.frequency
    if code == "routine-output-unstated":
        # The person presses their preference's option, whose label Admin sends in the composed answer.
        person.said.add("output")
        return OUTPUT_LABELS[person.output]
    if code == "routine-binding-ambiguous":
        return _target(person, list(question.get("options") or []))
    if code == OVER_BUDGET and isinstance(question.get("value"), int):
        # The person takes the shortest interval Team says fits.
        person.gap = question["value"]
        person.frequency = f"A cada {person.gap} segundos"
        return person.frequency
    return _FIXED_ANSWERS.get(code)


OVER_BUDGET = "routine-interval-over-budget"


# The person's own words for the questions whose answer never depends on the attempt.
_FIXED_ANSWERS = {
    "routine-binding-unsourced": "pode buscar de novo",
    "routine-work-rerun": "pode buscar de novo",
    "routine-work-split": "faça tudo de novo",
}


# The question line Admin shows for each Team question; Team reads only the answer of a composed answer.
_TEAM_QUESTIONS = {
    "routine-schedule-unstated": "Com que frequência a rotina deve rodar?",
    "routine-output-unstated": "O que a rotina deve fazer com o resultado de cada execução?",
    "routine-binding-ambiguous": "Qual destes alvos a rotina deve usar?",
    "routine-binding-unsourced": "Posso buscar esse valor de novo?",
    "routine-work-rerun": "Posso refazer o trabalho completo?",
    "routine-work-split": "Posso refazer o trabalho completo de uma vez?",
    OVER_BUDGET: "Esse intervalo não cabe no orçamento diário. Qual intervalo a rotina deve usar?",
}


def _already_answered(attempt: Attempt, code: str) -> bool:
    """Whether the span already held the answer to a Team question: the person answered it, said it, or sent it."""
    person = attempt.person
    return (
        code in attempt.answered
        or (code == "routine-schedule-unstated" and person.stated)
        or (code == "routine-output-unstated" and "output" in person.said)
        or (code == "routine-binding-ambiguous" and person.chose)
    )


def _conversation_turns(attempt: Attempt, first: str) -> dict[str, object]:
    """Send and answer until a turn records, the person has nothing left to say, or the sends run out.

    The agent's clarification gets every part the person can answer, composed as Admin composes it, and Admin's
    preselected recommended option for a part they leave open; a question in prose gets a typed answer; Team's
    recoverable Routine question gets the answer the person would type, as an ordinary send.
    """
    message = first
    response: dict[str, object] = {}
    for _send in range(MAX_SENDS):
        response = attempt.send(message)
        if "routine_proposal" in response or "routine_refusal" in response:
            return response
        question = response.get("routine_question")
        if question is not None:
            code = question["code"]
            attempt.questions.append(code)
            once = _TEAM_ONCE.get(code)
            if once is not None and attempt.person.asked_by.get(once) == "agent":
                # The agent already asked this; Team asking it too is the one question asked twice.
                attempt.person.violations.append(_ONCE[once])
            elif _already_answered(attempt, code):
                attempt.repeated.append(code)
            elif once is not None:
                miss = _once(attempt.person, once, "team")
                attempt.person.violations.extend([miss] if miss else [])
            answer = _team_answer(attempt.person, question)
            attempt.answered[code] = answer or ""
            # Admin answers Team's question with its composed form, which Team records from without the Brain.
            asked, composed = _TEAM_QUESTIONS.get(code, code), True
        else:
            answer, composed = _answer(attempt.person, response)
            asked = (response.get("clarification") or {}).get("question")
        if answer is None:
            return response
        if composed:
            answer = f"{message.strip()}\n\n{QUESTION_LABEL}: {asked}\n{ANSWER_LABEL}: {answer.strip()}"
        message = answer
    return response

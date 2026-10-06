"""Evaluate recorded Team Routines (ADR-0101) through the real chat agent and Team's real record and confirm path.

Brain and Team each define top-level ``routine`` and ``protocol`` packages, so they never share one interpreter: run
with ``--serve-brain`` from Brain, this file serves Brain's real runtime API on loopback in its own process, with an
in-memory checkpoint and a fresh bearer it writes to a new owner-only token file, and otherwise it drives the eval under
Team's environment. Its Team is an in-process Local controller built by Team's test harness, its
Brain is Team's own ``BrainRuntimeClient`` over that loopback HTTP, and its one Assistant is the reference
Cloudflare-shaped Assistant whose Action calls a fixture answers with results its reviewed output schemas admit. Every
send runs as the person, with the fresh request identity an Admin send carries, and a clarification is answered exactly
as Admin composes it.

Validate offline from Team, with Brain beside it::

    cd teams && PYTHONPATH=.:tests uv run --frozen --python 3.14 python ../brain/eval/routines.py --validate

Run live: start the Brain from Brain, which writes a fresh bearer to the new token file, then the driver from Team::

    cd brain && uv run --frozen --python 3.14 python -m eval.routines --serve-brain --brain-port 8791
    --token-file "$DIR/token" &
    cd teams && PYTHONPATH=.:tests uv run --frozen --python 3.14 python ../brain/eval/routines.py --brain-port 8791
    --token-file "$DIR/token" --openai-key-file ../.gpt-key --anthropic-key-file ../.claude-key

Cases, each ``--attempts`` times per model, fixed before sampling: the owner's four turns ("Cria uma rotina pra mim",
then the person's answers "Listar registros DNS", "A cada 30 segundos", and "shimpz.com"); a one-send daily list; and
"do this every hour" after an ordinary turn that did the work. A case passes when the recording turn ran list-zones and
list-dns-records for shimpz.com, its card carries the expected schedule and the zone id as a reference through the item
whose ``name`` is "shimpz.com", "Criar rotina" creates the Routine, and one replay through Team's real claim and run
lists shimpz.com's records and publishes a shown result. After the first passing owner attempt, two replay variants run
with no model: shimpz.com under a new zone id, and two zones named shimpz.com, which must never dispatch
list-dns-records. Output holds case ids, pass counts, Wilson 95% bounds, closed miss reasons, the schedules seen, and
the estimated cost from the provider-reported usage at catalog list prices; never a message, a reply, or a key.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import os
import secrets
import socket
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from unittest import mock

BRAIN = Path(__file__).resolve().parents[1]
TEAMS = BRAIN.parent / "teams"
CONTRACT = TEAMS / "tests" / "fixtures" / "reference-assistant" / "shimpz.contract.json"


def _load(name: str, path: Path):
    """A standard-library Brain eval helper loaded by path, as the umbrella journey driver loads it."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eval_cost = _load("routine_eval_cost", BRAIN / "eval" / "cost.py")
eval_stats = _load("routine_eval_stats", BRAIN / "eval" / "stats.py")

ASSISTANT = "shimpz-cloudflare"
PRINCIPAL = "a" * 32
ROUTINE_KEY = "e" * 64
SHIMPZ = "023e105f4ecef8ad9ca31a8372d0c353"
MOVED = "7b2f31aa2c0b4c8d9e1f203142536475"
TWIN = "5d41402abc4b2a76b9719d911017c592"
ACCOUNT = {"id": "f" * 32, "name": "Owner"}
OTHER_ZONES = (("9a7806061c88ada191ed06f989cc3dac", "example.com"), ("1b3f0c5e2a9d47e8b6c1d0f2a3b4c5d6", "other.org"))
ANSWERS = ("Listar registros DNS", "A cada 30 segundos", "shimpz.com")
# The labels Admin's Portuguese interface composes a clarification answer with (admin/frontend/src/lib/messages.js).
QUESTION_LABEL, ANSWER_LABEL = "Pergunta", "Resposta"
MAX_CONVERSATION_TEXT = 512
MAX_SENDS = 6
ATTEMPTS = 10
BUDGET_USD = 3.0
# One attempt's conservative reservation: this many Brain calls, each at most this much input and output.
CALLS_PER_ATTEMPT = 10
CALL_INPUT_TOKENS = 30_000
CALL_OUTPUT_TOKENS = 4_000
MODELS = (("openai", "gpt-6-luna"), ("anthropic", "claude-sonnet-5-5"))


def _pagination(count: int) -> dict[str, int]:
    return {"page": 1, "per_page": 25, "count": count, "total_count": count, "total_pages": 1}


def zones(shimpz: str = SHIMPZ, *, twin: bool = False) -> dict[str, object]:
    """list-zones' result: shimpz.com among others, under ``shimpz``; with ``twin``, a second zone of that name."""
    named = [*OTHER_ZONES, (shimpz, "shimpz.com"), *([(TWIN, "shimpz.com")] if twin else [])]
    items = [
        {"id": zone, "name": name, "status": "active", "type": "full", "paused": False, "account": ACCOUNT}
        for zone, name in named
    ]
    return {"zones": items, "pagination": _pagination(len(items))}


SHIMPZ_RECORDS = (
    {"id": "e" * 32, "type": "A", "name": "shimpz.com", "content": "192.0.2.1", "ttl": 300},
    {"id": "d" * 32, "type": "MX", "name": "shimpz.com", "content": "mail.shimpz.com", "ttl": 3600},
)


def records(zone: str) -> dict[str, object]:
    """list-dns-records' result for one zone: shimpz.com's own records, nothing for another zone."""
    owned = zone in (SHIMPZ, MOVED)
    found = [{**item, "proxied": False, "proxiable": False} for item in SHIMPZ_RECORDS] if owned else []
    return {"records": found, "pagination": _pagination(len(found))}


@dataclasses.dataclass
class Fixture:
    """The Cloudflare-shaped Assistant: what list-zones answers, and every call it saw, in order."""

    shimpz: str = SHIMPZ
    twin: bool = False
    calls: list[tuple[str, dict[str, object]]] = dataclasses.field(default_factory=list)

    def invoke(self, _team, _assistant, action, payload, _evidence) -> dict[str, object]:
        self.calls.append((action, dict(payload)))
        if action == "list-zones":
            return {"result": zones(self.shimpz, twin=self.twin)}
        return {"result": records(str(payload.get("zone_id")))}

    def listed(self, zone: str) -> bool:
        return any(action == "list-dns-records" and payload.get("zone_id") == zone for action, payload in self.calls)


def validate() -> None:
    """Offline: every fixture result validates against the reference Assistant's reviewed output schemas."""
    from jsonschema import Draft202012Validator

    actions = {item["id"]: item for item in json.loads(CONTRACT.read_text(encoding="utf-8"))["actions"]}
    for value in (zones(), zones(MOVED), zones(twin=True)):
        Draft202012Validator(actions["list-zones"]["output_schema"]).validate(value)
    for zone in (SHIMPZ, MOVED, TWIN):
        Draft202012Validator(actions["list-dns-records"]["output_schema"]).validate(records(zone))
    if len({case for case, _play in CASES}) != len(CASES):
        raise SystemExit("case ids are not unique")


@dataclasses.dataclass
class Meter:
    """The Brain usage Team's client reported during one attempt, summed in the eval's own counts."""

    usage: object = None

    def add(self, counts) -> None:
        current = eval_cost.Usage.of(dict(counts))
        self.usage = current if self.usage is None else self.usage + current


def serve_brain(port: int, token_file: Path) -> int:
    """Brain's real runtime API on loopback with an in-memory checkpoint, run from Brain's own environment."""
    import agent_runtime
    import runtime_api
    import uvicorn
    from langgraph.checkpoint.memory import InMemorySaver

    token = secrets.token_urlsafe(32)
    descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token)
    app = runtime_api.create_app(runtime=agent_runtime.AgentRuntime(InMemorySaver()), token_reader=lambda: token)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    return 0


def brain_served(port: int) -> str:
    """The loopback URL of the Brain ``--serve-brain`` serves, once it accepts connections."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=1):
            return f"http://127.0.0.1:{port}"
        time.sleep(0.5)
    raise SystemExit("the Brain runtime is not serving on that port")


@dataclasses.dataclass
class Outcome:
    reason: str | None = None
    schedule: dict[str, object] | None = None

    def __bool__(self) -> bool:
        return self.reason is None


@dataclasses.dataclass
class Team:
    """One in-process Team's chat service and the Team modules the eval drives it with."""

    service: object
    modules: object


class Attempt:
    """One isolated Team with the reference Assistant, its own Brain thread, and the person's sends."""

    def __init__(self, team: Team, fixture: Fixture, provider: str, key: str) -> None:
        self.team = team
        self.fixture = fixture
        self.provider = provider
        self.key = key
        self.conversation: list[dict[str, object]] = []

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
            "timezone": "America/Sao_Paulo",
        }
        with self._person():
            response = self.team.service.chat("team_1", body, self.provider, self.key)
        self._remember("user", message)
        self._remember("assistant", str(response.get("reply", "")))
        return response

    def _remember(self, role: str, text: str) -> None:
        cut = len(text) > MAX_CONVERSATION_TEXT
        self.conversation.append({"role": role, "text": text[:MAX_CONVERSATION_TEXT], "truncated": cut})

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
        return result["status"], notices[-1] if notices else None


# What the question asks, by its own words only, tried in this order among the answers the person has not given yet:
# frequency words are the most specific, a "which one" is the zone; an echo of an earlier answer never matches.
_HINTS = (
    (ANSWERS[1], ("frequ", "com que", "quando", "intervalo", "periodicidade", "horário", "horario", "quantas vezes")),
    (ANSWERS[0], ("trabalho", "o que", "tarefa", "repita", "repetir", "fazer")),
    (ANSWERS[2], ("zona", "domínio", "dominio", "escopo", "qual", "quais")),
)


def _pick(question: str, answers: list[str]) -> str | None:
    """The person's answer that fits the question, else the next one they have not given, else None."""
    lowered = question.casefold()
    for answer, words in _HINTS:
        if answer in answers and any(word in lowered for word in words):
            answers.remove(answer)
            return answer
    return answers.pop(0) if answers else None


def _repeated(question: str, given: tuple[str, ...]) -> str | None:
    """A question asked again gets the person's own answer to it again, never another choice made for them."""
    return _pick(question, [answer for answer, _words in _HINTS if answer in given])


def _conversation_turns(attempt: Attempt, first: str, answers: list[str]) -> dict[str, object]:
    """Send and answer as Admin composes it until a turn records, the person has nothing left to say, or sends run out.

    A question the person already answered gets that answer again; one their answers do not cover gets Admin's
    preselected recommended option, as one press of its answer button sends it.
    """
    message, given = first, tuple(answers)
    response: dict[str, object] = {}
    for _send in range(MAX_SENDS):
        response = attempt.send(message)
        if "routine_proposal" in response or "routine_refusal" in response:
            return response
        clarification = response.get("clarification")
        if clarification is not None:
            question = clarification["question"]
            recommended = clarification["options"][clarification["default_index"]]["label"]
            answer = _pick(question, answers) or _repeated(question, given) or recommended
            message = f"{message.strip()}\n\n{QUESTION_LABEL}: {question}\n{ANSWER_LABEL}: {answer}"
        else:
            reply = str(response.get("reply", ""))
            plain = _pick(reply, answers) or _repeated(reply, given)
            if plain is None:
                return response
            message = plain
    return response


def _selector_miss(card: dict[str, object]) -> str | None:
    """Why the card does not copy list-dns-records' zone_id from list-zones through the item named shimpz.com."""
    positions = {step["position"]: step["action"] for step in card["steps"]}
    step = next((item for item in card["steps"] if item["action"] == "list-dns-records"), None)
    zone = None if step is None else next((item for item in step["inputs"] if item["member"] == "zone_id"), None)
    if zone is None:
        return "selector:no-zone-input"
    if zone["origin"] != "selector":
        return f"selector:origin-{zone['origin']}"
    shaped = zone["where"] == {"member": "name", "value_json": '"shimpz.com"'} and zone["item"] == "/id"
    return None if positions.get(zone["step"]) == "list-zones" and shaped else "selector:shape"


def _card(attempt: Attempt, response: dict[str, object], schedule: Callable[[dict], bool]) -> Outcome:
    """The recording turn's card, judged; a miss names its first closed reason."""
    if "routine_refusal" in response:
        return Outcome(f"refused:{response['routine_refusal']['code']}")
    card = response.get("routine_proposal")
    if card is None:
        return Outcome("no-card")
    seen = dict(card["schedule"])
    calls = attempt.fixture.calls
    checks = (
        ("card-invalid", attempt.team.modules.http_routine.canonical_proposal(card) == card),
        ("calls", any(action == "list-zones" for action, _payload in calls) and attempt.fixture.listed(SHIMPZ)),
        ("schedule", schedule(card["schedule"])),
        (f"output:{card['output']['mode']}", card["output"]["mode"] in ("show", "changes")),
        (_selector_miss(card) or "selector", _selector_miss(card) is None),
    )
    failed = next((reason for reason, held in checks if not held), None)
    return Outcome(failed, seen) if failed else Outcome(None, {**seen, "proposal_id": card["proposal_id"]})


def _judged(attempt: Attempt, response: dict[str, object], schedule: Callable[[dict], bool]) -> Outcome:
    """The card, its confirmation, and one replay through Team's real claim and run."""
    carded = _card(attempt, response, schedule)
    if not carded:
        return carded
    seen = dict(carded.schedule)
    answer = attempt.confirm(seen.pop("proposal_id"))
    if answer["status"] != "created":
        return Outcome(f"confirm:{answer['status']}", seen)
    attempt.fixture.calls.clear()
    status, notice = attempt.replay()
    if status != "done" or not attempt.fixture.listed(SHIMPZ):
        return Outcome(f"replay:{status}", seen)
    output = None if notice is None else notice.detail.get("output")
    if notice is None or notice.outcome != "done" or output is None or output["state"] != "shown":
        return Outcome("replay-notice", seen)
    return Outcome(None, seen)


def _continuous(schedule: dict[str, object]) -> bool:
    return schedule.get("kind") == "continuous" and schedule.get("gap") == 30 and schedule.get("cap", 0) <= 1000


def owner(attempt: Attempt) -> Outcome:
    response = _conversation_turns(attempt, "Cria uma rotina pra mim", list(ANSWERS))
    return _judged(attempt, response, _continuous)


def plain_list(attempt: Attempt) -> Outcome:
    response = _conversation_turns(attempt, "Todo dia às 9h, liste os registros DNS de shimpz.com", [])
    return _judged(attempt, response, lambda schedule: schedule == {"kind": "daily", "time": "09:00"})


def every_hour(attempt: Attempt) -> Outcome:
    first = attempt.send("Liste os registros DNS de shimpz.com")
    if "routine_proposal" in first or "routine_refusal" in first:
        return Outcome("recorded-unasked")
    if not attempt.fixture.listed(SHIMPZ):
        return Outcome("first-turn-calls")
    attempt.fixture.calls.clear()
    response = _conversation_turns(attempt, "Faça isso a cada hora", [])
    return _judged(attempt, response, lambda schedule: schedule == {"kind": "hourly", "every": 1})


CASES: tuple[tuple[str, Callable[[Attempt], Outcome]], ...] = (
    ("owner-4-turns", owner),
    ("plain-list", plain_list),
    ("every-hour-after-earlier-turn", every_hour),
)


def variants(attempt: Attempt) -> dict[str, object]:
    """After a confirmed owner Routine, with no model: a moved zone id, then two zones named shimpz.com."""
    found: dict[str, object] = {}
    attempt.fixture.shimpz, attempt.fixture.calls[:] = MOVED, []
    status, _notice = attempt.replay()
    found["moved-zone-id"] = {"status": status, "listed_new_id": attempt.fixture.listed(MOVED)}
    attempt.fixture.shimpz, attempt.fixture.twin, attempt.fixture.calls[:] = SHIMPZ, True, []
    status, notice = attempt.replay()
    found["twin-zone-names"] = {
        "status": status,
        "outcome": None if notice is None else notice.outcome,
        "reason": None if notice is None else notice.detail.get("reason"),
        "code": None if notice is None else notice.detail.get("code"),
        "held_at": [item.action for item in attempt.team.service.routine_store.load("team_1").incidents],
        "list_dns_records_dispatched": any(action == "list-dns-records" for action, _payload in attempt.fixture.calls),
    }
    return found


@dataclasses.dataclass(frozen=True)
class Modules:
    """Team's modules, imported only once Team and its test harness are on the path."""

    harness: object
    brain_client: object
    inference_config: object
    brain_usage: object
    local_app: object
    local_audit: object
    local_authority: object
    http_routine: object
    record: object


def _team_modules() -> Modules:
    sys.path[:0] = [str(TEAMS), str(TEAMS / "tests")]
    import local_controller_harness

    from inference import client as brain_client
    from inference import config as inference_config
    from inference import usage as brain_usage
    from local import app as local_app
    from local import audit as local_audit
    from local import authority as local_authority
    from protocol.http.v1 import routine as http_routine
    from routine import record

    return Modules(
        local_controller_harness,
        brain_client,
        inference_config,
        brain_usage,
        local_app,
        local_audit,
        local_authority,
        http_routine,
        record,
    )


def _attempt_team(modules: Modules, directory: str, brain: tuple[str, Path], settings) -> tuple[object, Team, Fixture]:
    """A fresh Local Team for one attempt, whose own Space gives it its own Brain thread."""
    provider, model, space = settings
    url, token_file = brain

    # The harness builds its controller as a test case does; any of its own methods names the unused test.
    case = modules.harness.LocalContractCase("_chat_controller")
    controller = case._chat_controller(
        directory, modules.brain_client.BrainRuntimeClient(base_url=url, token_file=token_file)
    )
    fixture = Fixture()
    controller.assistant_lifecycle.invoke = fixture.invoke
    controller.space_id = controller.chat_turn_service.space_id = space
    controller.inference_store.save("team_1", modules.inference_config.normalize(provider, model))
    return case, Team(controller.chat_turn_service, modules), fixture


def _summary(case_id: str, passed: int, misses: list, schedules: list, costs: list, stopped: bool) -> dict:
    trials = len(costs)
    interval = eval_stats.wilson(passed, trials)
    total = sum(item.usd for item in costs)
    return {
        "id": case_id,
        "passed": passed,
        "trials": trials,
        "wilson95_lower": None if interval is None else round(interval[0], 3),
        "inconclusive": stopped,
        "misses": misses,
        "schedules": schedules,
        "usd": round(total, 6),
        "usd_per_attempt": round(total / trials, 6) if trials else None,
        "cost_known": all(item.known for item in costs),
    }


class Runner:
    """Every case of every model in order under one hard budget, with the Brain usage Team reported per attempt."""

    def __init__(self, modules: Modules, brain: tuple[str, Path], budget: float, attempts: int) -> None:
        self.modules = modules
        self.brain = brain
        self.budget = eval_cost.Budget(budget)
        self.attempts = attempts
        self.meter = Meter()
        self.variants: dict[str, object] | None = None
        self.exhausted = False

    def one(self, model: tuple[str, str, str], case: tuple[str, Callable], index: int) -> tuple[Outcome, object]:
        provider, model_id, key = model
        case_id, play = case
        reservation = self.budget.reserve(
            eval_cost.call_bound(model_id, CALL_INPUT_TOKENS, CALL_OUTPUT_TOKENS) * CALLS_PER_ATTEMPT
        )
        self.meter.usage = None
        with tempfile.TemporaryDirectory() as directory:
            settings = (provider, model_id, f"eval-{secrets.token_hex(8)}")
            harness, team, fixture = _attempt_team(self.modules, directory, self.brain, settings)
            attempt = Attempt(team, fixture, provider, key)
            try:
                outcome = play(attempt)
            except self.modules.local_app.ApiProblem as exc:
                outcome = Outcome(f"team:{exc.code}")
            if outcome and case_id == "owner-4-turns" and self.variants is None:
                self.variants = variants(attempt)
            harness.doCleanups()
        spent = eval_cost.Cost(0.0) if self.meter.usage is None else eval_cost.cost(self.meter.usage, model_id)
        self.budget.settle(reservation, spent)
        return outcome, spent

    def case(self, model: tuple[str, str, str], case: tuple[str, Callable]) -> dict[str, object]:
        passed, misses, schedules, costs, stopped = 0, [], [], [], False
        for index in range(self.attempts):
            try:
                outcome, spent = self.one(model, case, index)
            except eval_cost.BudgetExhaustedError:
                stopped = self.exhausted = True
                break
            costs.append(spent)
            passed += bool(outcome)
            misses += [] if outcome else [outcome.reason]
            schedules.append(outcome.schedule)
        return _summary(case[0], passed, misses, schedules, costs, stopped)


def run(
    models: list[tuple[str, str, str]], brain: tuple[str, Path], limits: tuple[int, float], only: frozenset[str]
) -> dict:
    modules = _team_modules()
    original = modules.brain_usage.record
    runner: Runner | None = None

    def metered(operation, provider, model, counts) -> None:
        runner.meter.add(counts)
        original(operation, provider, model, counts)

    report: dict[str, object] = {"attempts_per_case": limits[0], "models": []}
    with (
        mock.patch.object(modules.local_authority, "routine_key_fingerprint", return_value=ROUTINE_KEY),
        mock.patch.object(modules.local_audit, "record_request", return_value="a" * 32),
        mock.patch.object(modules.local_audit, "record", return_value="a" * 32),
        mock.patch.object(modules.brain_usage, "record", metered),
    ):
        runner = Runner(modules, brain, limits[1], limits[0])
        for model in models:
            cases = []
            for case in CASES:
                if runner.exhausted or (only and case[0] not in only):
                    continue
                cases.append(runner.case(model, case))
            report["models"].append({"provider": model[0], "model": model[1], "cases": cases})
    report["variants"] = runner.variants
    report["budget"] = runner.budget.summary()
    return report


def _key(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--serve-brain", action="store_true")
    parser.add_argument("--brain-port", type=int)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--openai-key-file", type=Path)
    parser.add_argument("--anthropic-key-file", type=Path)
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    parser.add_argument("--budget", type=float, default=BUDGET_USD)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    if args.serve_brain:
        if args.brain_port is None or args.token_file is None:
            raise SystemExit("name the Brain's port and its new token file")
        return serve_brain(args.brain_port, args.token_file)
    validate()
    if args.validate:
        print(json.dumps({"validated": [case for case, _play in CASES]}))
        return 0
    keys = {"openai": _key(args.openai_key_file), "anthropic": _key(args.anthropic_key_file)}
    models = [(provider, model, keys[provider]) for provider, model in MODELS if keys[provider]]
    if not models or args.brain_port is None or args.token_file is None:
        raise SystemExit("name the Brain's port, its token file, and at least one key file")
    brain = (brain_served(args.brain_port), args.token_file)
    report = run(models, brain, (args.attempts, args.budget), frozenset(args.only))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

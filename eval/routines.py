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

Strata, each ``--attempts`` (30) times per model, fixed before sampling: the owner's four turns; a one-send daily
list; two named zones at once (multi-zone); a zone named only in an earlier send; a request with no schedule; the
reversed-order bait, primed (the zone was looked up in an earlier send) and unprimed (its id is known only from an
earlier reply outside the span); two zones with the same name at creation, which must be asked about; and intervals of
30 seconds (list a zone's records) and 5 seconds (one step, list the zones), whose card keeps the exact gap the person
stated with cap ceil(86400 / gap). The driver answers the agent's questions and Team's recoverable Routine questions as
the person would, a Team question always in Admin's composed form and a target by its option's exact JSON text. A
stratum passes when the card binds every zone the person meant through a reference (the twin stratum: the person's
chosen zone by its id), carries the schedule the person stated, takes no more sends than its missing pieces need,
"Criar rotina" creates the Routine, and one replay through Team's real claim and run shows that work's exact result.
One frequency question and one output question are each allowed, from the agent or Team, whichever comes first, while
the person has not stated it; asking it again, after it was stated, or by both is a miss, as is any timezone or limits
question. Team's output question gets its protocol label as the answer, and the card must carry the person's choice.
After the first passing attempt of any stratum that binds shimpz.com through list-zones into list-dns-records, two
replay variants run with no model: shimpz.com under a new zone id, and two zones named shimpz.com, which must never
dispatch list-dns-records.
The gate exits non-zero unless every stratum of every shipped model passed every attempt and both variants held.
For the control arm, serve the Brain with SHIMPZ_ROUTINE_MODE_PROMPT=off, which drops its Routine-mode prompt section.
Output holds stratum ids, pass counts, Wilson 95% bounds, closed miss reasons with root causes, the questions asked,
the schedules seen, and the estimated cost; never a message, a reply, or a key (only ``--trace-dir`` writes messages).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import secrets
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest import mock

if __package__:
    from eval.routines_fixture import (
        ATTEMPTS,
        BRAIN,
        BUDGET_USD,
        CALL_INPUT_TOKENS,
        CALL_OUTPUT_TOKENS,
        CALLS_PER_ATTEMPT,
        EXAMPLE,
        MODELS,
        MOVED,
        ROUTED,
        ROUTINE_KEY,
        SHIMPZ,
        TWIN,
        Fixture,
        eval_cost,
        eval_stats,
        records,
        zones,
    )
    from eval.routines_person import OUTPUT_LABELS, Attempt, Outcome, Team
    from eval.routines_serve import Meter, brain_served, serve_brain
    from eval.routines_strata import CASES, VARIANT_STRATA, _cause, variants
else:  # run as a script from Team, beside its sibling modules
    from routines_fixture import (
        ATTEMPTS,
        BRAIN,
        BUDGET_USD,
        CALL_INPUT_TOKENS,
        CALL_OUTPUT_TOKENS,
        CALLS_PER_ATTEMPT,
        EXAMPLE,
        MODELS,
        MOVED,
        ROUTED,
        ROUTINE_KEY,
        SHIMPZ,
        TWIN,
        Fixture,
        eval_cost,
        eval_stats,
        records,
        zones,
    )
    from routines_person import OUTPUT_LABELS, Attempt, Outcome, Team
    from routines_serve import Meter, brain_served, serve_brain
    from routines_strata import CASES, VARIANT_STRATA, _cause, variants


# The Team checkout the eval drives; --teams names another, such as a worktree carrying a Team change.
TEAMS = BRAIN.parent / "teams"


CONTRACT = TEAMS / "tests" / "fixtures" / "reference-assistant" / "shimpz.contract.json"


def validate() -> None:
    """Offline: every fixture result validates against the reference Assistant's reviewed output schemas."""
    from jsonschema import Draft202012Validator

    actions = {item["id"]: item for item in json.loads(CONTRACT.read_text(encoding="utf-8"))["actions"]}
    for value in (zones(), zones(MOVED), zones(twin=True)):
        Draft202012Validator(actions["list-zones"]["output_schema"]).validate(value)
    for zone in (SHIMPZ, MOVED, TWIN, EXAMPLE):
        Draft202012Validator(actions["list-dns-records"]["output_schema"]).validate(records(zone))
    from protocol.http.v1 import routine_proposal as http_routine_proposal

    if http_routine_proposal.OUTPUT_CHOICES["pt"] != OUTPUT_LABELS:
        raise SystemExit("the person's output labels drifted from Team's protocol OUTPUT_CHOICES")
    if len({case for case, _play in CASES}) != len(CASES):
        raise SystemExit("case ids are not unique")


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
    http_routine_proposal: object
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
    from protocol.http.v1 import routine_proposal as http_routine_proposal
    from routine import record

    return Modules(
        local_controller_harness,
        brain_client,
        inference_config,
        brain_usage,
        local_app,
        local_audit,
        local_authority,
        http_routine_proposal,
        record,
    )


# The turn context members a trace keeps: never the model key.
_TRACED_CONTEXT = ("thread_id", "provider", "model", "effort", "locale", "knowledge_writable", "routine_capacity")


def _turn_event(turn) -> dict[str, object]:
    """One Brain turn as Team received it: reply, requested Action calls, question, memory, and record call."""
    return {
        "kind": "brain-turn",
        "status": turn.status,
        "reply": turn.reply,
        "actions": [dataclasses.asdict(item) for item in turn.actions],
        "clarification": turn.clarification,
        "memory": list(turn.memory),
        "record": turn.routine,
    }


def _traced(client, fixture: Fixture):
    """Team's own Brain client, with every turn it starts or resumes and every turn it receives in the transcript."""
    start, resume = client.start, client.resume

    def received(turn):
        fixture.note(_turn_event(turn))
        return turn

    def traced_start(context, message, *, conversation):
        known = {name: getattr(context, name) for name in _TRACED_CONTEXT}
        fixture.note({"kind": "brain-start", "message": message, "conversation": conversation, "context": known})
        return received(start(context, message, conversation=conversation))

    def traced_resume(context, results):
        fixture.note({"kind": "brain-resume", "results": dict(results)})
        return received(resume(context, results))

    client.start, client.resume = traced_start, traced_resume
    return client


def _attempt_team(
    modules: Modules, directory: str, brain: tuple[str, Path], settings, events: list | None
) -> tuple[object, Team, Fixture]:
    """A fresh Local Team for one attempt, whose own Space gives it its own Brain thread."""
    provider, model, space, effort = settings
    url, token_file = brain
    fixture = Fixture(events=events)
    client = modules.brain_client.BrainRuntimeClient(base_url=url, token_file=token_file)
    # The harness builds its controller as a test case does; any of its own methods names the unused test.
    case = modules.harness.LocalContractCase("_chat_controller")
    controller = case._chat_controller(directory, client if events is None else _traced(client, fixture))
    controller.assistant_lifecycle.invoke = fixture.invoke
    controller.space_id = controller.chat_turn_service.space_id = space
    controller.inference_store.save("team_1", modules.inference_config.normalize(provider, model, effort))
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


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    return repr(value)


def _write_trace(path: Path, transcript: dict[str, object]) -> None:
    """One attempt's whole transcript as owner-only JSON; it holds the messages, never a model key."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(transcript, handle, ensure_ascii=False, indent=1, default=_plain)


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
        self.trace_dir: Path | None = None
        # The Team's reasoning effort for every attempt; None keeps Team's default.
        self.effort: str | None = None

    def one(self, model: tuple[str, str, str], case: tuple[str, Callable], index: int) -> tuple[Outcome, object]:
        provider, model_id, key = model
        case_id, play = case
        # Reserved at the dearest model the Team may route a turn of this provider to.
        served = {model_id, *ROUTED.get(provider, ())}
        bound = max(eval_cost.call_bound(item, CALL_INPUT_TOKENS, CALL_OUTPUT_TOKENS) for item in served)
        reservation = self.budget.reserve(bound * CALLS_PER_ATTEMPT)
        self.meter.usage = {}
        with tempfile.TemporaryDirectory() as directory:
            settings = (provider, model_id, f"eval-{secrets.token_hex(8)}", self.effort)
            events = None if self.trace_dir is None else []
            harness, team, fixture = _attempt_team(self.modules, directory, self.brain, settings, events)
            attempt = Attempt(team, fixture, provider, key)
            try:
                outcome = play(attempt)
            except self.modules.local_app.ApiProblem as exc:
                outcome = Outcome(f"team:{exc.code}")
                # Team's mapped detail, as Admin would see it; the Brain's own cause is in its --brain-log.
                fixture.note({"kind": "team-error", "code": exc.code, "detail": str(exc)})
            if not outcome:
                outcome.cause = _cause(attempt, outcome.reason)
            outcome.questions = tuple(attempt.questions)
            if outcome and case_id in VARIANT_STRATA and self.variants is None:
                self.variants = variants(attempt)
            harness.doCleanups()
        spent = self.meter.cost()
        self.budget.settle(reservation, spent)
        if events is not None:
            outcome_record = {
                "reason": outcome.reason,
                "cause": outcome.cause,
                "schedule": outcome.schedule,
                "questions": list(outcome.questions),
            }
            transcript = {"model": model_id, "case": case_id, "attempt": index, "outcome": outcome_record}
            _write_trace(self.trace_dir / model_id / case_id / f"{index}.json", {**transcript, "events": events})
        return outcome, spent

    def case(self, model: tuple[str, str, str], case: tuple[str, Callable]) -> dict[str, object]:
        passed, misses, schedules, costs, stopped = 0, [], [], [], False
        questions: collections.Counter[str] = collections.Counter()
        for index in range(self.attempts):
            try:
                outcome, spent = self.one(model, case, index)
            except eval_cost.BudgetExhaustedError:
                stopped = self.exhausted = True
                break
            costs.append(spent)
            passed += bool(outcome)
            misses += [] if outcome else [{"reason": outcome.reason, "cause": outcome.cause}]
            schedules.append(outcome.schedule)
            questions.update(outcome.questions)
        return {**_summary(case[0], passed, misses, schedules, costs, stopped), "questions": dict(questions)}


def run(
    models: list[tuple[str, str, str]],
    brain: tuple[str, Path],
    limits: tuple[int, float],
    only: frozenset[str],
    trace_dir: Path | None = None,
    effort: str | None = None,
) -> dict:
    modules = _team_modules()
    original = modules.brain_usage.record
    runner: Runner | None = None

    def metered(operation, provider, model, counts) -> None:
        runner.meter.add(model, counts)
        original(operation, provider, model, counts)

    report: dict[str, object] = {"attempts_per_case": limits[0], "effort": effort, "models": []}
    with (
        mock.patch.object(modules.local_authority, "routine_key_fingerprint", return_value=ROUTINE_KEY),
        mock.patch.object(modules.local_audit, "record_request", return_value="a" * 32),
        mock.patch.object(modules.local_audit, "record", return_value="a" * 32),
        mock.patch.object(modules.brain_usage, "record", metered),
    ):
        runner = Runner(modules, brain, limits[1], limits[0])
        runner.trace_dir, runner.effort = trace_dir, effort
        for model in models:
            cases = []
            for case in CASES:
                if runner.exhausted or (only and case[0] not in only):
                    continue
                cases.append(runner.case(model, case))
            report["models"].append({"provider": model[0], "model": model[1], "cases": cases})
    report["variants"] = runner.variants
    report["budget"] = runner.budget.summary()
    report["gate"] = gate(report, limits[0])
    return report


def gate(report: dict[str, object], attempts: int) -> dict[str, object]:
    """The owner's release gate: every shipped model finished every case at 30 of 30, and both variants held."""
    failures: list[str] = []
    if attempts < ATTEMPTS:
        failures.append("attempts-below-gate")
    if report["budget"]["unknown_settlements"]:
        failures.append("cost-unknown")
    ran = {item["model"]: item["cases"] for item in report["models"]}
    for _provider, model in MODELS:
        finished = {case["id"]: case for case in ran.get(model, [])}
        for case_id, _play in CASES:
            case = finished.get(case_id)
            clean = case is not None and not case["inconclusive"] and not case["misses"]
            if not clean or case["trials"] != max(attempts, ATTEMPTS) or case["passed"] != case["trials"]:
                failures.append(f"{model}:{case_id}")
    found = report["variants"] or {}
    moved, twin = found.get("moved-zone-id", {}), found.get("twin-zone-names", {})
    if not (moved.get("status") == "done" and moved.get("listed_new_id")):
        failures.append("variant:moved-zone-id")
    ambiguous = twin.get("status") == "failed" and twin.get("code") == "plan-reference-ambiguous"
    if not ambiguous or twin.get("list_dns_records_dispatched") is not False:
        failures.append("variant:twin-zone-names")
    return {"passed": not failures, "failures": failures}


def _key(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--serve-brain", action="store_true")
    parser.add_argument("--brain-log", type=Path, help="with --serve-brain: log each provider failure and its cause")
    parser.add_argument("--brain-port", type=int)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--openai-key-file", type=Path)
    parser.add_argument("--anthropic-key-file", type=Path)
    parser.add_argument("--attempts", type=int, default=ATTEMPTS)
    parser.add_argument("--budget", type=float, default=BUDGET_USD)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--trace-dir", type=Path, help="write each attempt's whole transcript to DIR/MODEL/CASE/N.json")
    parser.add_argument(
        "--effort", choices=("low", "medium", "high"), help="the Team's reasoning effort; default Team's"
    )
    parser.add_argument(
        "--model", action="append", default=[], help="measurement only: PROVIDER:MODEL in place of that provider's"
    )
    parser.add_argument("--teams", type=Path, help="the Team worktree to drive; default the sibling teams")
    args = parser.parse_args()
    if args.teams is not None:
        global TEAMS, CONTRACT
        TEAMS = args.teams.resolve()
        CONTRACT = TEAMS / "tests" / "fixtures" / "reference-assistant" / "shimpz.contract.json"
    if args.serve_brain:
        if args.brain_port is None or args.token_file is None:
            raise SystemExit("name the Brain's port and its new token file")
        return serve_brain(args.brain_port, args.token_file, args.brain_log)
    validate()
    if args.validate:
        print(json.dumps({"validated": [case for case, _play in CASES]}))
        return 0
    keys = {"openai": _key(args.openai_key_file), "anthropic": _key(args.anthropic_key_file)}
    # A measurement may run another model of a provider; the gate still declares only MODELS, so it then fails.
    chosen = dict(MODELS) | dict(item.split(":", 1) for item in args.model)
    models = [(provider, model, keys[provider]) for provider, model in chosen.items() if keys[provider]]
    if not models or args.brain_port is None or args.token_file is None:
        raise SystemExit("name the Brain's port, its token file, and at least one key file")
    brain = (brain_served(args.brain_port), args.token_file)
    report = run(models, brain, (args.attempts, args.budget), frozenset(args.only), args.trace_dir, args.effort)
    print(json.dumps(report, indent=2))
    # Any miss, skipped attempt, missing model or case, or budget stop fails the run; no caller reads it as a pass.
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

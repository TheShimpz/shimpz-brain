"""Evaluate recorded Team Routines (ADR-0101) through the real chat agent and Team's real record and confirm path.

Brain and Team each define top-level ``routine`` and ``protocol`` packages, so they never share one interpreter: the
run starts each Brain from Brain's own environment (``--serve-brain``, internal), serving Brain's real runtime API on
loopback with an in-memory checkpoint and a fresh bearer in a new owner-only token file, and drives the eval under
Team's environment. Each attempt's Team is an in-process Local controller built by Team's test harness, its
Brain is Team's own ``BrainRuntimeClient`` over that loopback HTTP, and its one Assistant is the real Cloudflare
Assistant project (``--assistant``, default ~/shimpz-cloudflare): its own manifest, Genesis, and the machine contract
its pinned SDK generates from its source, exactly as staging generates it, admitted through Team's Local snapshot
parsers. Only its provider is simulated: a fixture answers its reads with results its output schemas admit and fails
any other call, so a change never passes as done. Every
send runs as the person, with the fresh request identity an Admin send carries, and a clarification is answered exactly
as Admin composes it.

Validate offline from Team, with Brain beside it::

    cd teams && PYTHONPATH=.:tests uv run --frozen --python 3.14 python ../brain/eval/routines.py --validate

Run live, the one command, from Team; it starts every Brain and worker it needs and stops them on any exit::

    cd teams && PYTHONPATH=.:tests uv run --frozen --python 3.14 python ../brain/eval/routines.py
    --openai-key-file ../.gpt-key --anthropic-key-file ../.claude-key --budget 8

The run (``routines_pool``) confines itself to its CPU budget (``--cpus``, default half the machine's processors) and
starts ``--workers`` (default one per budgeted processor) Team driver processes, each beside its own Brain
(``routines_serve``) and each running ``--threads`` attempts at once, every attempt pinned to its worker's Brain. Every
provider request on the wire is reserved under one hard ``--budget`` before it is sent, across every Brain, and the
budget's refusal stops the run. A provider's rate-limit or overload answer is never a model's miss and never a pass:
its attempt is discarded and retried with backoff, the provider's concurrency adapts down, and the report counts it.

Strata, each ``--attempts`` (30) times per model, fixed before sampling: the owner's four turns; a one-send daily
list; two named zones at once (multi-zone); a zone named only in an earlier send; a request with no schedule; the
reversed-order bait, primed (the zone was looked up in an earlier send) and unprimed (its id is known only from an
earlier reply outside the span); two zones with the same name at creation, which must be asked about; and intervals of
30 seconds (list a zone's records) and 5 seconds (one step, list the zones), whose card keeps the exact gap the person
stated with cap ceil(86400 / gap); and a Routine that changes something, the owner's daily update of the TXT record
verify.shimpz.com to each day's date, whose real replace-dns-record Action asks for the Supervisor's password in the
recording turn and in every run. The person authorizes each request; it passes only when the card replays
replace-dns-record on the record its name selects with each run's date as content, the replay changes nothing before
the person authorizes it, and the authorized run writes that day's date.
The driver answers the agent's questions and Team's recoverable Routine questions as
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
The gate also fails on a counted attempt whose cost is unknown, a reservation never settled, a response that reached
the eval's output limit, provider evidence no attempt received, an attempt that stayed throttled through every retry,
or a provider that refused every request (no credit or quota left, a refused key).
For the control arm, run with SHIMPZ_ROUTINE_MODE_PROMPT=off, which every Brain inherits and which drops its
Routine-mode prompt section.
Output holds stratum ids, pass counts, Wilson 95% bounds, closed miss reasons with root causes, the questions asked,
the schedules seen, the estimated cost, the throttling met, and the run's concurrency, wall time, and CPU time; never a
message, a reply, or a key (only ``--trace-dir`` writes messages, a discarded try under ``CASE/discarded/``).
"""

import argparse
import base64
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import functools
import http.client
import json
import os
import resource
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import traceback
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest import mock

if __package__:
    from eval.routines_fixture import (
        ASSISTANT,
        ATTEMPTS,
        BRAIN,
        BUDGET_USD,
        CHANGING,
        EXAMPLE,
        MODELS,
        MOVED,
        ROUTINE_KEY,
        SHIMPZ,
        TWIN,
        VERIFY,
        Fixture,
        changed,
        eval_cost,
        record,
        records,
        zone,
        zones,
    )
    from eval.routines_person import OUTPUT_LABELS, Attempt, Outcome, Team
    from eval.routines_pool import DEFAULT_THREADS, Pool, Schedule, confine, cpu_budget, merge, orchestrate, plan
    from eval.routines_pool import evidence as wire_evidence
    from eval.routines_serve import CLAMPED_HEADER, FAILURE_HEADER, THROTTLED_HEADER, Meter, brain_python, serve_brain
    from eval.routines_strata import CASES, VARIANT_STRATA, _cause, variants
else:  # run as a script from Team, beside its sibling modules
    from routines_fixture import (
        ASSISTANT,
        ATTEMPTS,
        BRAIN,
        BUDGET_USD,
        CHANGING,
        EXAMPLE,
        MODELS,
        MOVED,
        ROUTINE_KEY,
        SHIMPZ,
        TWIN,
        VERIFY,
        Fixture,
        changed,
        eval_cost,
        record,
        records,
        zone,
        zones,
    )
    from routines_person import OUTPUT_LABELS, Attempt, Outcome, Team
    from routines_pool import DEFAULT_THREADS, Pool, Schedule, confine, cpu_budget, merge, orchestrate, plan
    from routines_pool import evidence as wire_evidence
    from routines_serve import CLAMPED_HEADER, FAILURE_HEADER, THROTTLED_HEADER, Meter, brain_python, serve_brain
    from routines_strata import CASES, VARIANT_STRATA, _cause, variants


# The Team checkout the eval drives; --teams names another, such as a worktree carrying a Team change.
TEAMS = BRAIN.parent / "teams"


# The real Assistant project the eval's Team runs, with its provider simulated; --assistant names another checkout.
ASSISTANT_PROJECT = Path.home() / ASSISTANT


@dataclasses.dataclass(frozen=True)
class Generated:
    """The Assistant's own manifest and the machine contract its pinned SDK generates from its source."""

    manifest: bytes
    contract: bytes
    project: Path


def _isolated(python: Path, *arguments: str, given: bytes = b"") -> subprocess.CompletedProcess:
    # Isolated and without bytecode, as the CLI runs an Assistant's SDK, so no caller path can shadow it.
    return subprocess.run([str(python), "-I", "-B", *arguments], input=given, capture_output=True, check=False)


def generated(project: Path) -> Generated:
    """Generate the contract exactly as staging does in the image, with the SDK version the project pins."""
    python = project / ".venv" / "bin" / "python"
    pins = [
        item
        for item in tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
        if item.startswith("shimpz==")
    ]
    probe = _isolated(
        python,
        "-c",
        "import importlib.metadata, sys; print(*sys.version_info[:2], importlib.metadata.version('shimpz'))",
    )
    if len(pins) != 1 or probe.returncode != 0 or probe.stdout.split() != [b"3", b"14", pins[0][8:].encode()]:
        raise SystemExit(f"{project}/.venv must run Python 3.14 with the SDK version its pyproject pins")
    contract = _isolated(python, "-m", "shimpz._bridge", "contract", str(project))
    if contract.returncode != 0:
        raise SystemExit(f"the SDK in {project}/.venv could not generate the Assistant's machine contract")
    return Generated((project / "shimpz.toml").read_bytes(), contract.stdout, project)


def validate(assistant: Generated) -> None:
    """Offline: every simulated result validates against the real Assistant's generated output schemas."""
    from jsonschema import Draft202012Validator

    actions = {item["id"]: item for item in json.loads(assistant.contract)["actions"]}
    missing = {"list-zones", "list-dns-records", "get-zone", "get-dns-record"} - set(actions)
    if missing:
        raise SystemExit(f"the Assistant's contract has no {sorted(missing)}, which the fixture simulates")
    for value in (zones(), zones(MOVED), zones(twin=True)):
        Draft202012Validator(actions["list-zones"]["output_schema"]).validate(value)
        for item in value["zones"]:
            Draft202012Validator(actions["get-zone"]["output_schema"]).validate(zone(item["id"]))
    written = {"zone_id": SHIMPZ, "record_type": "TXT", "name": VERIFY["name"], "content": "2026-10-07", "ttl": 300}
    written |= {"proxied": False, "record_id": VERIFY["id"]}
    for action in CHANGING:
        Draft202012Validator(actions[action]["output_schema"]).validate(changed(action, written))
    for zone_id in (SHIMPZ, MOVED, TWIN, EXAMPLE):
        Draft202012Validator(actions["list-dns-records"]["output_schema"]).validate(records(zone_id))
        for item in records(zone_id)["records"]:
            Draft202012Validator(actions["get-dns-record"]["output_schema"]).validate(record(zone_id, item["id"]))
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
    # The real Assistant every attempt's Team runs, admitted once.
    assistant: Admitted


def _team_modules(assistant: Generated) -> Modules:
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
        admitted(assistant),
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


@dataclasses.dataclass(frozen=True)
class Admitted:
    """The real Assistant as Team's Local snapshot admission reads it: its binding's declarations, pack, and Genesis."""

    declarations: dict[str, object]
    pack: object
    genesis: str
    project: Path


def admitted(assistant: Generated) -> Admitted:
    """Admit the generated manifest and contract through Team's own Local snapshot parsers (local/install/snapshots)."""
    from assistant import manifest as assistant_manifest
    from assistant import spec as assistant_spec
    from tests import human_request_fixtures

    identity = assistant_manifest.parse_manifest_identity(assistant.manifest)
    if identity.assistant_id != ASSISTANT:
        raise SystemExit(f"the Assistant project is {identity.assistant_id}, not {ASSISTANT}")
    declared = assistant_manifest.parse_manifest_contract(assistant.manifest)
    presentation = assistant_manifest.parse_manifest_presentation(assistant.manifest)
    machine_contract = assistant_manifest.parse_machine_contract(
        assistant.contract,
        declared.integrations,
        declared.stored_inputs,
        summary=identity.summary,
        description=presentation.description,
        allowed_hosts=declared.allowed_hosts,
    )
    contract = assistant_spec.runtime_contract(
        {
            "summary": identity.summary,
            "description": presentation.description,
            "links": dict(presentation.links),
            "allowed_hosts": list(declared.allowed_hosts),
            "integrations": [
                {"id": item.id, "provider": item.provider, "scopes": list(item.scopes)}
                for item in declared.integrations
            ],
            "stored_inputs": [item.document() for item in declared.stored_inputs],
            "machine_contract": machine_contract,
        }
    )
    # The real catalog in an admitted pack; its non-English entries are the test harness's, not the Assistant's.
    pack = human_request_fixtures.pack_for(machine_contract["messages"])
    declarations = {
        "version": identity.version,
        "name": identity.name,
        "summary": identity.summary,
        "actions": contract.actions,
        "allowed_hosts": contract.allowed_hosts,
        "integrations": contract.integrations,
        "stored_inputs": contract.stored_inputs,
        "machine_contract": contract.machine_contract,
        "pack_digest": pack.pack_digest,
    }
    genesis = assistant_manifest.parse_manifest_genesis(assistant.manifest)
    return Admitted(declarations, pack, genesis, assistant.project)


def _asker(assistant: Admitted) -> Callable[..., object]:
    """The request the real Action of a change makes before anything else, admitted as Team admits an RPC frame.

    The Action runs with no answer, so it asks before it reads its access token: no provider is ever reached. None
    when it asks nothing or fails.
    """
    from action import execution as action_execution
    from action import human as action_human

    catalog = action_human.catalog_by_id(assistant.declarations["machine_contract"])
    python = assistant.project / ".venv" / "bin" / "python"

    def ask(action: str, payload: dict[str, object], evidence) -> object:
        declared = assistant.declarations["actions"][action]
        invocation = {
            "input": dict(payload),
            "integrations": dict.fromkeys(assistant.declarations["integrations"], "eval-token-never-sent"),
            "stored_inputs": {},
            "files": {},
            "operation_id": evidence.operation_id,
        }
        bridge = ("-m", "shimpz._bridge", "invoke", str(assistant.project), action)
        frame = _isolated(python, *bridge, given=json.dumps(invocation).encode())
        if frame.returncode != 0:
            return None
        policy = action_execution.RpcResultPolicy(
            human_requests=declared.human_requests, declared_stored_inputs=declared.stored_inputs, catalog=catalog
        )
        try:
            action_execution.project_rpc_result(json.loads(frame.stdout), {}, lambda value: value, policy)
        except action_human.HumanRequestSuspensionError as requested:
            return requested.request
        return None

    return ask


def _bind_real_assistant(modules: Modules, controller, assistant: Admitted) -> None:
    """Replace the harness's hand-written Assistant with the real one, and grant its Integrations as declared."""
    controller.registry[ASSISTANT] = dataclasses.replace(controller.registry[ASSISTANT], **assistant.declarations)
    for integration_id, integration in assistant.declarations["integrations"].items():
        grant = types.SimpleNamespace(
            access_token=modules.harness.TEST_ACCOUNT_ACCESS_TOKEN,
            refresh_token=modules.harness.TEST_ACCOUNT_REFRESH_TOKEN,
            scopes=integration.scopes,
            expires_in=3600,
        )
        controller.assistant_integrations.put(
            "team_1", ASSISTANT, integration_id, integration.provider, integration.scopes, grant
        )
    controller.assistant_lifecycle._assistant_language = lambda _active: assistant.pack
    controller.assistant_lifecycle._active_assistant_genesis = lambda _active: assistant.genesis


def _attempt_team(
    modules: Modules, directory: str, brain: tuple[str, Path, Signals], settings, events: list | None
) -> tuple[object, Team, Fixture]:
    """A fresh Local Team for one attempt, whose own Space gives it its own Brain thread.

    Its Brain client reads each response's evidence into the attempt's Signals before Team reads the response.
    """
    provider, model, space, effort = settings
    url, token_file, signals = brain
    fixture = Fixture(events=events)
    client = modules.brain_client.BrainRuntimeClient(
        base_url=url, token_file=token_file, connection_factory=signals.connection
    )
    # The harness builds its controller as a test case does; any of its own methods names the unused test.
    case = modules.harness.LocalContractCase("_chat_controller")
    controller = case._chat_controller(directory, client if events is None else _traced(client, fixture))
    _bind_real_assistant(modules, controller, modules.assistant)
    fixture.ask = _asker(modules.assistant)
    controller.assistant_lifecycle.invoke = fixture.invoke
    controller.space_id = controller.chat_turn_service.space_id = space
    controller.inference_store.save("team_1", modules.inference_config.normalize(provider, model, effort))
    return case, Team(controller.chat_turn_service, modules), fixture


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


@dataclasses.dataclass
class Signals:
    """The evidence every Brain response of one attempt carried, read before Team reads the response.

    A response without the eval's evidence headers, or with unreadable ones, is never trusted: the attempt is an error
    and the run stops.
    """

    throttled: int = 0
    clamped: int = 0
    failures: set[str] = dataclasses.field(default_factory=set)
    # Brain's own admission refusals, which answer 503 with Retry-After, and Brain responses that were not a success.
    refused: int = 0
    failed: int = 0
    unreadable: int = 0

    def saw(self, response: http.client.HTTPResponse) -> None:
        try:
            self.throttled += int(response.getheader(THROTTLED_HEADER))
            self.clamped += int(response.getheader(CLAMPED_HEADER))
        except TypeError, ValueError:
            self.unreadable += 1
        failure = response.getheader(FAILURE_HEADER)
        if failure is not None:
            self.failures.add(failure)
        self.refused += response.status == 503 and response.getheader("retry-after") is not None
        self.failed += response.status != 200

    def connection(self, host: str, port: int, timeout: float) -> http.client.HTTPConnection:
        """Team's own loopback connection, whose response this attempt reads first."""
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
        received = connection.getresponse

        def getresponse() -> http.client.HTTPResponse:
            response = received()
            self.saw(response)
            return response

        connection.getresponse = getresponse
        return connection

    def disposition(self) -> str:
        """What the attempt's result is: counted, or else why it is discarded, the gravest reason first.

        The budget's refusal outranks a provider that refuses every request, which outranks a throttle, which outranks
        Brain's own capacity refusal.
        """
        if self.unreadable:
            return "error"
        if "budget" in self.failures:
            return "stopped"
        if "terminal" in self.failures:
            return "unavailable"
        if "throttled" in self.failures:
            return "throttled"
        return "brain-refused" if self.refused else "counted"


# The attempt whose Brain usage Team reports in this thread: Team meters every Brain call in its caller's thread.
_METER: contextvars.ContextVar[Meter | None] = contextvars.ContextVar("routine_eval_meter", default=None)


@contextlib.contextmanager
def _patched(modules: Modules):
    """The patches every attempt of one worker shares: fixed audit and Routine key, and the attempt's own meter."""
    original = modules.brain_usage.record

    def metered(operation, provider, model, counts) -> None:
        meter = _METER.get()
        if meter is None:
            # Never spend that no attempt carries.
            raise RuntimeError("Brain usage was reported outside an attempt")
        meter.add(model, counts)
        original(operation, provider, model, counts)

    with (
        mock.patch.object(modules.local_authority, "routine_key_fingerprint", return_value=ROUTINE_KEY),
        mock.patch.object(modules.local_audit, "record_request", return_value="a" * 32),
        mock.patch.object(modules.local_audit, "record", return_value="a" * 32),
        mock.patch.object(modules.brain_usage, "record", metered),
    ):
        yield


@dataclasses.dataclass(frozen=True)
class WorkerContext:
    """What one worker's attempts share: Team's modules, its Brain, the models with their keys, and the run."""

    modules: Modules
    brain: tuple[str, Path]
    models: list[tuple[str, str, str]]
    effort: str | None
    run_dir: Path
    trace_dir: Path | None


def _claim_variants(run_dir: Path) -> bool:
    """Whether this attempt is the run's first passing eligible one: the first to create the claim, in any worker."""
    try:
        os.close(os.open(run_dir / "variants", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        return False
    return True


def _judged(context: WorkerContext, case_id: str, attempt: Attempt, signals: Signals) -> tuple[Outcome, object]:
    """Play the case, then judge it unless its evidence discards it; the first passing eligible one runs variants."""
    try:
        outcome = dict(CASES)[case_id](attempt)
    except context.modules.local_app.ApiProblem as exc:
        outcome = Outcome(f"team:{exc.code}")
        # Team's mapped detail, as Admin would see it; the Brain's own cause is in its --brain-logs file.
        attempt.fixture.note({"kind": "team-error", "code": exc.code, "detail": str(exc)})
    if not outcome:
        outcome.cause = _cause(attempt, outcome.reason)
    outcome.questions = tuple(attempt.questions)
    eligible = outcome and case_id in VARIANT_STRATA and signals.disposition() == "counted"
    found = variants(attempt) if eligible and _claim_variants(context.run_dir) else None
    return outcome, found


def _trace(context: WorkerContext, task: dict[str, object], result: dict[str, object], events: list) -> None:
    """The attempt's transcript; a discarded try's beside its slot's, never in its place."""
    model = context.models[task["model"]][1]
    folder = context.trace_dir / model / task["case"]
    counted = result["disposition"] == "counted"
    path = folder / f"{task['index']}.json" if counted else folder / "discarded" / f"{task['index']}.{task['try']}.json"
    outcome = {key: result[key] for key in ("disposition", "reason", "cause", "schedule", "questions")}
    transcript = {"model": model, "case": task["case"], "attempt": task["index"], "outcome": outcome}
    _write_trace(path, {**transcript, "events": events})


def attempt(context: WorkerContext, task: dict[str, object]) -> dict[str, object]:
    """One attempt in its own Team, Space, temporary directory, and Brain thread, metered in this thread."""
    provider, model_id, key = context.models[task["model"]]
    meter, signals = Meter(), Signals()
    events = None if context.trace_dir is None else []
    token = _METER.set(meter)
    try:
        with tempfile.TemporaryDirectory(dir=context.run_dir) as directory:
            settings = (provider, model_id, f"eval-{secrets.token_hex(8)}", context.effort)
            brain = (*context.brain, signals)
            harness, team, fixture = _attempt_team(context.modules, directory, brain, settings, events)
            try:
                outcome, found = _judged(context, task["case"], Attempt(team, fixture, provider, key), signals)
            finally:
                harness.doCleanups()
    finally:
        _METER.reset(token)
    if signals.disposition() == "error":
        raise RuntimeError("a Brain response carried no readable eval evidence")
    spent = meter.cost()
    result = {
        **task,
        "disposition": signals.disposition(),
        "throttled": signals.throttled,
        "clamped": signals.clamped,
        "passed": bool(outcome),
        "reason": outcome.reason,
        "cause": outcome.cause,
        "schedule": outcome.schedule,
        "questions": list(outcome.questions),
        # Team reports no usage for a Brain request that failed, so such an attempt's cost is only a lower bound.
        "usd": spent.usd,
        "known": spent.known and not signals.failed,
        "variants": found,
    }
    if events is not None:
        _trace(context, task, result, events)
    return result


def _worker_context(config: dict[str, object]) -> WorkerContext:
    global TEAMS
    TEAMS = Path(config["teams"])
    decoded = {name: base64.b64decode(config[name]) for name in ("manifest", "contract")}
    assistant = Generated(decoded["manifest"], decoded["contract"], Path(config["project"]))
    trace_dir = None if config["trace_dir"] is None else Path(config["trace_dir"])
    models = [tuple(item) for item in config["models"]]
    brain = (config["brain"], Path(config["token"]))
    return WorkerContext(_team_modules(assistant), brain, models, config["effort"], Path(config["run"]), trace_dir)


def _abandon() -> None:
    """Exit at once with everything this worker started, such as an Assistant SDK mid-call.

    The run starts each worker as the leader of its own session, so its group holds exactly its descendants.
    """
    if os.getpgrp() == os.getpid():
        os.killpg(os.getpgrp(), signal.SIGKILL)
    os._exit(1)


def work() -> int:
    """One worker: its configuration, then one task per stdin line and one result per stdout line, until stdin ends.

    Only results are written to the real stdout; anything else written there goes to stderr. Stdin ending while an
    attempt still runs means the run stopped or died, so the worker exits at once with everything it started.
    """
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    config = json.loads(sys.stdin.readline())
    context = _worker_context(config)
    lock = threading.Lock()

    def emit(message: dict[str, object]) -> None:
        with lock:
            protocol.write(json.dumps(message, default=_plain) + "\n")
            protocol.flush()

    def report(task: dict[str, object], done: concurrent.futures.Future) -> None:
        # An attempt that raised is never a result: the run reports it and stops.
        error = done.exception()
        if error is None:
            emit(done.result())
            return
        traceback.print_exception(error)
        emit({**task, "error": f"{type(error).__name__}: {error}"})

    def submit(task: dict[str, object]) -> concurrent.futures.Future:
        running = pool.submit(attempt, context, task)
        running.add_done_callback(functools.partial(report, task))
        return running

    with _patched(context.modules):
        pool = concurrent.futures.ThreadPoolExecutor(config["threads"], thread_name_prefix="attempt")
        emit({"ready": True})
        running = [submit(json.loads(line)) for line in sys.stdin]
        if not all(item.done() for item in running):
            _abandon()
        pool.shutdown()
    return 0


@dataclasses.dataclass(frozen=True)
class Settings:
    """One run's choices: how much it measures, under what budget, and with how much parallelism."""

    attempts: int
    budget: float
    only: frozenset[str] = frozenset()
    trace_dir: Path | None = None
    effort: str | None = None
    cpus: int = 1
    workers: int = 1
    threads: int = DEFAULT_THREADS
    brain_logs: Path | None = None


def _worker_config(
    models: list[tuple[str, str, str]], settings: Settings, assistant: Generated, run_dir: Path
) -> Callable[[int, str, Path], dict[str, object]]:
    """Each worker's first stdin line: everything it needs, its model keys included, and its own Brain."""
    shared = {
        "teams": str(TEAMS),
        "manifest": base64.b64encode(assistant.manifest).decode(),
        "contract": base64.b64encode(assistant.contract).decode(),
        "project": str(assistant.project),
        "trace_dir": None if settings.trace_dir is None else str(settings.trace_dir.resolve()),
        "models": [list(item) for item in models],
        "effort": settings.effort,
        "run": str(run_dir),
        "threads": settings.threads,
    }
    return lambda _number, url, token: {**shared, "brain": url, "token": str(token)}


def _cpu_seconds() -> float:
    used = (resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    return round(sum(item.ru_utime + item.ru_stime for item in used), 1)


def _run_report(schedule: Schedule, settings: Settings, times: tuple[float, float, float], cpus: list[int]) -> dict:
    started, ready, finished = times
    return {
        "cpus": len(cpus),
        "workers": settings.workers,
        "threads": settings.threads,
        "capacity": settings.workers * settings.threads,
        "peak_attempts": schedule.peak,
        "peak_attempts_by_provider": dict(schedule.peak_by),
        "startup_seconds": round(ready - started, 1),
        "wall_seconds": round(finished - started, 1),
        "cpu_seconds": _cpu_seconds(),
    }


def run(models: list[tuple[str, str, str]], settings: Settings, *, assistant: Generated) -> dict:
    """Every case of every model under one hard budget, across the run's workers, merged into one report."""
    started = time.monotonic()
    cpus = confine(settings.cpus)
    cases = [case for case, _play in CASES if not settings.only or case in settings.only]
    pairs = [(provider, model) for provider, model, _key in models]
    schedule = Schedule(plan(pairs, cases, settings.attempts), settings.workers * settings.threads)
    command = (sys.executable, str(Path(__file__).resolve()))
    if settings.brain_logs is not None:
        settings.brain_logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="routine-eval-") as directory:
        run_dir = Path(directory)
        ledger = eval_cost.SharedBudget.create(run_dir / "budget.json", settings.budget)
        configure = _worker_config(models, settings, assistant, run_dir)
        brains = (brain_python(), run_dir, settings.brain_logs)
        with Pool(command, brains, settings.workers, settings.threads, configure) as pool:
            ready = time.monotonic()
            orchestrate(schedule, pool)
        report: dict[str, object] = {"attempts_per_case": settings.attempts, "effort": settings.effort}
        report["models"] = merge(pairs, cases, settings.attempts, schedule)
        report["variants"] = next((item["variants"] for item in schedule.results.values() if item["variants"]), None)
        report["budget"] = ledger.summary()
        report["throttling"] = {
            "providers": {name: state.report() for name, state in schedule.providers.items()},
            "brain_refusals": schedule.brain_refusals,
            "evidence": wire_evidence(run_dir, schedule),
        }
    report["run"] = _run_report(schedule, settings, (started, ready, time.monotonic()), cpus)
    report["gate"] = gate(report, settings.attempts)
    return report


def _run_failures(report: dict[str, object]) -> list[str]:
    """The run's own failures: its cost, the eval's output limit, its evidence, and its throttling.

    A counted attempt whose cost Team could not fully report leaves the measurement's cost unknown. A request on the
    wire whose cost is unknown, such as one Team cancelled at its deadline, is charged its whole reservation against
    the cap instead; the budget summary counts it.
    """
    budget, providers, wire = report["budget"], report["throttling"]["providers"], report["throttling"]["evidence"]
    cases = [case for item in report["models"] for case in item["cases"]]
    lost = wire["brain_throttled"] > wire["received_throttled"] or wire["brain_clamped"] > wire["received_clamped"]
    checks = (
        ("cost-unknown", not all(case["cost_known"] for case in cases)),
        ("budget-unsettled", budget["unsettled_usd"]),
        ("output-clamped", wire["brain_clamped"] or wire["received_clamped"]),
        ("evidence-lost", lost or wire["unattributed"]),
        ("throttled", any(item["given_up"] for item in providers.values())),
    )
    unavailable = [f"provider-unavailable:{name}" for name, item in providers.items() if item["unavailable"]]
    return [name for name, failed in checks if failed] + unavailable


def gate(report: dict[str, object], attempts: int) -> dict[str, object]:
    """The owner's release gate: every shipped model finished every case at 30 of 30, and both variants held."""
    failures: list[str] = []
    if attempts < ATTEMPTS:
        failures.append("attempts-below-gate")
    failures += _run_failures(report)
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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true")
    # Internal: the run starts its Brains and workers with these.
    parser.add_argument("--serve-brain", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--token-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ledger", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--events", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--brain-log", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--brain-logs", type=Path, help="log each Brain's provider failures to DIR/brain-N.log")
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
    parser.add_argument(
        "--assistant", type=Path, default=ASSISTANT_PROJECT, help="the real Assistant project; default ~/" + ASSISTANT
    )
    parser.add_argument("--cpus", type=int, default=cpu_budget(), help="processors the run may use; default half")
    parser.add_argument("--workers", type=int, help="worker and Brain pairs; default one per processor")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="attempts each worker runs at once")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.serve_brain:
        if None in (args.token_file, args.ledger, args.events):
            raise SystemExit("name the Brain's new token file, the run's ledger, and its events file")
        return serve_brain(args.token_file, args.ledger, args.events, args.brain_log)
    if args.worker:
        return work()
    if args.teams is not None:
        global TEAMS
        TEAMS = args.teams.resolve()
    assistant = generated(args.assistant.resolve())
    validate(assistant)
    if args.validate:
        _team_modules(assistant)
        print(json.dumps({"validated": [case for case, _play in CASES]}))
        return 0
    keys = {"openai": _key(args.openai_key_file), "anthropic": _key(args.anthropic_key_file)}
    # A measurement may run another model of a provider; the gate still declares only MODELS, so it then fails.
    chosen = dict(MODELS) | dict(item.split(":", 1) for item in args.model)
    models = [(provider, model, keys[provider]) for provider, model in chosen.items() if keys[provider]]
    if not models:
        raise SystemExit("name at least one key file")
    workers = args.cpus if args.workers is None else args.workers
    settings = Settings(
        args.attempts, args.budget, frozenset(args.only), args.trace_dir, args.effort, args.cpus, workers, args.threads
    )
    settings = dataclasses.replace(settings, brain_logs=args.brain_logs)
    # A stop request ends the run through its cleanup: every process it started stops and its directory goes.
    signal.signal(signal.SIGTERM, lambda *_signal: sys.exit(143))
    report = run(models, settings, assistant=assistant)
    print(json.dumps(report, indent=2))
    # Any miss, skipped attempt, missing model or case, or budget stop fails the run; no caller reads it as a pass.
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

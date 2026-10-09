"""The Routine eval's Brain processes: how the run starts them, how each serves and budgets, and what it reports.

Each Brain serves Brain's real runtime API on a loopback port it binds itself, with an in-memory checkpoint and a fresh
bearer it writes to a new owner-only token file; it prints the port on its first stdout line and exits as soon as its
stdin closes, so it never outlives the run that started it. Every provider request it makes passes the evaluation
ceiling (``eval/ceiling.py``) against the run's one shared ledger, with the production SDK retries kept: each try is
reserved on the wire before it is sent. Each Brain response carries the request's own evidence in eval-only headers:
how many provider rate-limit or overload answers it met (``eval-throttled``), how many responses reached the eval's
output limit (``eval-clamped``), and, for a failed provider request, whether the budget refused it, the provider
refuses every request, or it was throttled (``eval-failure``). Every throttled or clamped answer, and any provider
answer outside a Brain request, is also written to the Brain's events file, so the run can reconcile what its attempts
received against what its Brains met.
"""

import contextlib
import contextvars
import dataclasses
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

if __package__:
    from eval.routines_fixture import BRAIN, CALL_OUTPUT_TOKENS, eval_cost
else:  # run as a script from Team, beside its sibling modules
    from routines_fixture import BRAIN, CALL_OUTPUT_TOKENS, eval_cost


@dataclasses.dataclass
class Meter:
    """The Brain usage Team's client reported during one attempt, summed per model in the eval's own counts.

    A Team may route a turn to another model of its provider, so each call is priced at the model that served it.
    """

    usage: dict[str, object] = dataclasses.field(default_factory=dict)

    def add(self, model: str, counts) -> None:
        current = eval_cost.Usage.of(dict(counts))
        self.usage[model] = current if model not in self.usage else self.usage[model] + current

    def cost(self) -> object:
        total = eval_cost.Cost(0.0)
        for model, usage in self.usage.items():
            total = total + eval_cost.cost(usage, model)
        return total


# A provider's answers for rate or overload: 429 from either, OpenAI's 503, and Anthropic's 529.
THROTTLE_STATUSES = frozenset({429, 503, 529})


PROVIDER_HOSTS = frozenset({"api.openai.com", "api.anthropic.com"})


THROTTLED_HEADER, CLAMPED_HEADER, FAILURE_HEADER = "eval-throttled", "eval-clamped", "eval-failure"


@dataclasses.dataclass
class Evidence:
    """What one Brain request met on the wire, beyond what its result says."""

    throttled: int = 0
    clamped: int = 0
    # Why a failed provider request failed, when the eval can tell: "budget", "terminal", or "throttled".
    failure: str | None = None

    def headers(self) -> list[tuple[bytes, bytes]]:
        found = [(THROTTLED_HEADER, str(self.throttled)), (CLAMPED_HEADER, str(self.clamped))]
        found += [(FAILURE_HEADER, self.failure)] if self.failure else []
        return [(name.encode(), value.encode()) for name, value in found]


_EVIDENCE: contextvars.ContextVar[Evidence | None] = contextvars.ContextVar("routine_eval_evidence", default=None)


# A provider failure no retry can heal: the key is refused, or the account has no credit or quota left.
TERMINAL_STATUSES = frozenset({401, 402, 403})
TERMINAL_WORDS = ("credit balance", "insufficient_quota", "billing")


def _terminal(error: BaseException, status: object) -> bool:
    text = str(error).casefold()
    return status in TERMINAL_STATUSES or (status in (400, 429) and any(word in text for word in TERMINAL_WORDS))


def failure_of(error: BaseException) -> str | None:
    """Why a provider failure happened, from its whole cause chain, or None for anything else.

    The ceiling's refusal ("budget") outranks a provider that refuses every request ("terminal"), which outranks a
    throttle ("throttled") the SDK met on any try.
    """
    seen, pending, found = set(), [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        found |= {"budget"} if isinstance(current, eval_cost.BudgetExhaustedError) else set()
        found |= {"terminal"} if _terminal(current, status) else set()
        found |= {"throttled"} if status in THROTTLE_STATUSES else set()
        pending += [current.__cause__, current.__context__]
    return next((name for name in ("budget", "terminal", "throttled") if name in found), None)


def clamped(body: object) -> bool:
    """Whether a provider's answer stopped at the output limit the eval's ceiling set."""
    if not isinstance(body, dict):
        return False
    if body.get("stop_reason") == "max_tokens":
        return True
    incomplete = body.get("incomplete_details")
    if isinstance(incomplete, dict) and incomplete.get("reason") == "max_output_tokens":
        return True
    choices = body.get("choices")
    return isinstance(choices, list) and any(
        isinstance(item, dict) and item.get("finish_reason") == "length" for item in choices
    )


class Observer:
    """Each provider response's evidence, credited to the Brain request it belongs to and written to the events file.

    The file holds every throttled or clamped answer the Brain met, credited or not, so the run can prove that every
    piece of evidence reached an attempt: a request Team cancelled at its deadline never delivers its headers.
    """

    def __init__(self, events: Path) -> None:
        self.events = events
        self._lock = threading.Lock()

    def observe(self, request, response) -> None:
        if request.url.host not in PROVIDER_HOSTS:
            return
        throttled = response.status_code in THROTTLE_STATUSES
        try:
            stopped = response.status_code == 200 and clamped(json.loads(response.read()))
        except ValueError:
            stopped = False
        evidence = _EVIDENCE.get()
        if evidence is not None:
            evidence.throttled += throttled
            evidence.clamped += stopped
        if throttled or stopped or evidence is None:
            line = {"status": response.status_code, "throttled": throttled, "clamped": stopped}
            with self._lock, self.events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({**line, "attributed": evidence is not None}) + "\n")

    def install(self) -> None:
        """Observe every provider response the ceiling let through, each SDK retry included."""
        import httpx

        observer, sent = self, httpx.Client.send

        def send(client, request, /, **kwargs):
            response = sent(client, request, **kwargs)
            observer.observe(request, response)
            return response

        httpx.Client.send = send


class EvidenceHeaders:
    """ASGI middleware: one fresh Evidence per HTTP request, returned as headers on its response."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        evidence = Evidence()
        token = _EVIDENCE.set(evidence)

        async def sent(message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *evidence.headers()]}
            await send(message)

        try:
            await self.app(scope, receive, sent)
        finally:
            _EVIDENCE.reset(token)


async def provider_error(_request, error: BaseException):
    """Brain's own 502 for a provider failure, with why it failed in the request's evidence."""
    from fastapi.responses import JSONResponse

    evidence = _EVIDENCE.get()
    if evidence is not None:
        evidence.failure = failure_of(error)
    return JSONResponse(status_code=502, content={"detail": "Model provider request failed"})


def _log_provider_failures(log: Path) -> None:
    """Write every provider failure the Brain maps to a 502 to ``log``, with its cause: the provider's own error.

    The cause is the provider SDK's exception, whose text is the provider's error body; request headers, and so the
    key, are never part of it.
    """
    import logging
    import traceback

    import runtime_errors

    logging.basicConfig(filename=log, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    original = runtime_errors.ProviderRequestError.__init__

    def logged(self, *args, **kwargs) -> None:
        cause = sys.exc_info()[1]
        if cause is not None:
            trace = "".join(traceback.format_exception(cause))[-4000:]
            logging.getLogger("routine-eval").error("provider failure: %s: %s\n%s", type(cause).__name__, cause, trace)
        original(self, *args, **kwargs)

    runtime_errors.ProviderRequestError.__init__ = logged


def _exit_with_parent() -> None:
    """Exit the whole process once stdin closes: the run that started this Brain has stopped it or died."""
    sys.stdin.buffer.read()
    os._exit(0)


def evaluation_app(ledger: Path, events: Path, token: str):
    """Brain's real runtime app with an in-memory checkpoint, its wire spend under the run's ledger, and evidence."""
    import agent_runtime
    import runtime_api
    from eval.ceiling import Ceiling
    from langgraph.checkpoint.memory import InMemorySaver

    Ceiling(0.0, CALL_OUTPUT_TOKENS, budget=eval_cost.SharedBudget(ledger), sdk_retries=True).install()
    Observer(events).install()
    app = runtime_api.create_app(runtime=agent_runtime.AgentRuntime(InMemorySaver()), token_reader=lambda: token)
    app.add_exception_handler(agent_runtime.ProviderRequestError, provider_error)
    app.add_middleware(EvidenceHeaders)
    return app


def serve_brain(token_file: Path, ledger: Path, events: Path, log: Path | None = None) -> int:
    """One Brain of a run, from Brain's own environment: its port on stdout, then served until stdin closes."""
    import uvicorn

    if log is not None:
        _log_provider_failures(log)
    token = secrets.token_urlsafe(32)
    descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token)
    app = evaluation_app(ledger, events, token)
    listener = socket.create_server(("127.0.0.1", 0), backlog=128)
    print(listener.getsockname()[1], flush=True)
    # Anything else written to stdout goes to stderr: the run reads only the port from this pipe.
    os.dup2(2, 1)
    threading.Thread(target=_exit_with_parent, name="exit-with-parent", daemon=True).start()
    uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False)).run(sockets=[listener])
    return 0


# The variables that would put Team's environment in front of Brain's: Brain and Team share top-level module names.
_TEAM_ENVIRONMENT = ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")


def brain_environment() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name not in _TEAM_ENVIRONMENT}


def brain_python() -> str:
    """Brain's own interpreter, as ``uv run`` resolves it from Brain's locked environment."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is not on PATH; Brain's environment cannot be resolved")
    found = subprocess.run(
        [uv, "run", "--frozen", "--python", "3.14", "python", "-c", "import sys; print(sys.executable)"],
        cwd=BRAIN,
        env=brain_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode != 0 or not found.stdout.strip():
        raise SystemExit("Brain's environment could not be resolved with uv run")
    return found.stdout.strip()


@dataclasses.dataclass(frozen=True)
class BrainFiles:
    """One Brain's files in the run's private directory, and where it logs provider failures, if anywhere."""

    token: Path
    events: Path
    ledger: Path
    log: Path | None = None


def start_brain(python: str, files: BrainFiles) -> subprocess.Popen:
    logged = () if files.log is None else ("--brain-log", str(files.log))
    paths = ("--token-file", str(files.token), "--ledger", str(files.ledger), "--events", str(files.events))
    # Its own session, so the run can stop it and anything it started as one group.
    return subprocess.Popen(
        [python, "-m", "eval.routines", "--serve-brain", *paths, *logged],
        cwd=BRAIN,
        env=brain_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        start_new_session=True,
    )


# How long a Brain may take to import, write its token, and bind its port.
BRAIN_START_SECONDS = 120.0


def brain_url(process: subprocess.Popen, deadline: float = BRAIN_START_SECONDS) -> str:
    """The loopback URL a started Brain serves, from its first stdout line; a Brain that never says is stopped."""
    timer = threading.Timer(deadline, process.kill)
    timer.start()
    try:
        line = process.stdout.readline()
    finally:
        timer.cancel()
    with contextlib.suppress(ValueError):
        port = int(line)
        if 0 < port < 65536:
            return f"http://127.0.0.1:{port}"
    raise RuntimeError("a Brain stopped before it served")

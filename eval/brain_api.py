"""The Brain as a black box for the evaluation and performance harnesses: launch the binary and play Team over HTTP.

``Brain`` starts the Rust binary named by ``SHIMPZ_BRAIN_BINARY`` (it refuses to start without it) as
``shimpz-brain serve --listen 127.0.0.1:PORT --token-file F --state-dir D --provider-origin ORIGIN``, with a fresh
owner-only bearer and a private state directory, waits for ``GET /health``, and stops it and removes its state on
close. ``Client`` sends the exact request bodies Team sends (``teams/inference/client.py``; the Brain refuses unknown
fields with 422) and returns each 200 answer as a value; any other status raises ``BrainError`` with that status.

Standard library only, and independent of the umbrella checkout: the Brain repository is its own repository.
"""

import contextlib
import http.client
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from eval import ceiling as eval_ceiling
from eval import cost as eval_cost
from eval import provider_origin

BINARY_ENVIRONMENT = "SHIMPZ_BRAIN_BINARY"
HEALTH_SECONDS = 60.0
# The Brain's fixed chat-turn reasoning effort (ADR-0113; runtime/src/provider/mod.rs, the chat call profile).
CHAT_EFFORT = "low"
# The statuses a provider-dependent outcome can end in: the Brain refused the model's answer or its own input (400),
# or the provider request failed (502). Any other status is a harness or Brain failure, never a model's miss.
MISS_STATUSES = frozenset({400, 502})
# One whole Brain operation, as Team bounds a turn; a stalled request fails instead of hanging the harness.
RESPONSE_SECONDS = 600.0
MAX_RESPONSE_BYTES = 1024 * 1024
# The variables a child must not inherit: proxies that would take its loopback provider traffic elsewhere.
_PROXIES = frozenset({"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"})


class BrainError(RuntimeError):
    """The Brain answered something other than 200, or could not be reached; ``status`` is None when unreachable."""

    def __init__(self, path: str, status: int | None) -> None:
        super().__init__(f"Brain {path} answered {status}")
        self.path = path
        self.status = status


@dataclass(frozen=True, slots=True)
class Provider:
    """The Team's provider credential one request carries."""

    provider: str
    model: str
    api_key: str = field(repr=False)

    def wire(self) -> dict[str, object]:
        return {"provider": self.provider, "model": self.model, "api_key": self.api_key}


def action(
    action_id: str,
    summary: str,
    input_schema: Mapping[str, object],
    output_schema: Mapping[str, object] | None = None,
    *,
    authorization: bool = False,
    input_files: Sequence[str] = (),
) -> dict[str, object]:
    """One Action exactly as Team's Brain request carries it."""
    return {
        "id": action_id,
        "summary": summary,
        "input_schema": dict(input_schema),
        "output_schema": dict(output_schema or {}),
        "authorization": authorization,
        "input_files": list(input_files),
    }


def assistant(assistant_id: str, genesis: str, actions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """One Assistant exactly as Team's Brain request carries it."""
    return {"id": assistant_id, "genesis": genesis, "actions": [dict(item) for item in actions]}


@dataclass(frozen=True, slots=True)
class TurnContext:
    """A turn's context in Team's wire shape; every resume of the turn resends it whole."""

    thread_id: str
    team_name: str
    assistants: tuple[Mapping[str, object], ...]
    provider: Provider
    memories: tuple[Mapping[str, object], ...] | None = None
    routines: tuple[Mapping[str, object], ...] | None = None
    routine_question: Mapping[str, object] | None = None
    routine_mode: bool = False
    routine_rerun: tuple[Mapping[str, object], ...] | None = None
    routine_schedule: Mapping[str, object] | None = None
    routine_output: str | None = None
    routine_settings_answer: bool = False
    routine_selected: Mapping[str, object] | None = None
    attachments: tuple[Mapping[str, object], ...] = ()
    browser_available: bool = False

    def wire(self) -> dict[str, object]:
        """The context fields in Team's order (``BrainRuntimeClient._context``)."""

        def listed(items: tuple[Mapping[str, object], ...] | None) -> list[dict[str, object]] | None:
            return None if items is None else [dict(item) for item in items]

        def mapped(item: Mapping[str, object] | None) -> dict[str, object] | None:
            return None if item is None else dict(item)

        return {
            "thread_id": self.thread_id,
            "team_name": self.team_name,
            "assistants": [dict(item) for item in self.assistants],
            "provider": self.provider.wire(),
            "memories": listed(self.memories),
            "routines": listed(self.routines),
            "routine_question": mapped(self.routine_question),
            "routine_mode": self.routine_mode,
            "routine_rerun": listed(self.routine_rerun),
            "routine_schedule": mapped(self.routine_schedule),
            "routine_output": self.routine_output,
            "routine_settings_answer": self.routine_settings_answer,
            "routine_selected": mapped(self.routine_selected),
            "attachments": [dict(item) for item in self.attachments],
            "browser_available": self.browser_available,
        }


@dataclass(frozen=True, slots=True)
class ActionRequest:
    interrupt_id: str
    assistant_id: str
    action: str
    input: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Turn:
    """One turn answer: its status, reply, requests, and the operation's reported usage counts."""

    status: str
    reply: str
    actions: tuple[ActionRequest, ...]
    usage: Mapping[str, int]
    clarification: object = None
    memory: tuple[object, ...] = ()
    routine: object = None
    browser: Mapping[str, object] | None = None


def turn(answer: Mapping[str, object]) -> Turn:
    """A 200 turn answer as a value; a malformed one raises ``ValueError``."""
    try:
        return Turn(
            status=str(answer["status"]),
            reply=str(answer["reply"]),
            actions=tuple(
                ActionRequest(
                    str(item["interrupt_id"]), str(item["assistant_id"]), str(item["action"]), dict(item["input"])
                )
                for item in answer["actions"]
            ),
            usage=dict(answer["usage"]),
            clarification=answer["clarification"],
            memory=tuple(answer["memory"]),
            routine=answer["routine"],
            browser=answer["browser"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed Brain turn answer") from exc


@dataclass(frozen=True, slots=True)
class RouteRequest:
    """One intent route as Team sends it; a decision key and installable Assistants only when given."""

    objective: str
    locale: str
    expected_intent: str | None = None
    candidates: tuple[Mapping[str, object], ...] = ()
    lifecycle_reference: Mapping[str, object] | None = None
    conversation: tuple[Mapping[str, object], ...] = ()
    decision_key: str | None = field(default=None, repr=False)
    installable: tuple[Mapping[str, object], ...] = ()

    def wire(self, provider: Provider) -> dict[str, object]:
        """The request body in Team's field order."""
        payload: dict[str, object] = {
            "provider": provider.wire(),
            "objective": self.objective,
            "expected_intent": self.expected_intent,
            "candidates": [dict(item) for item in self.candidates],
            "lifecycle_reference": None if self.lifecycle_reference is None else dict(self.lifecycle_reference),
            "conversation": [dict(entry) for entry in self.conversation],
            "locale": self.locale,
        }
        if self.decision_key is not None:
            payload["decision_provider"] = {"provider": "typesafe", "api_key": self.decision_key}
        if self.installable:
            payload["installable"] = [dict(item) for item in self.installable]
        return payload


class Client:
    """Team's side of the Brain's HTTP API: one connection per request, the bearer read from its file each time."""

    def __init__(self, url: str, token_file: Path) -> None:
        host, _, port = url.removeprefix("http://").rstrip("/").rpartition(":")
        if not url.startswith("http://") or not host or not port.isdigit():
            raise ValueError("the Brain URL must be a plain http://host:port origin")
        self.host, self.port, self.token_file = host, int(port), token_file

    def post(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        """POST ``payload`` as Team does (compact UTF-8 JSON) and return the 200 answer; anything else raises."""
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        token = self.token_file.read_text(encoding="utf-8").strip()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=RESPONSE_SECONDS)
        try:
            connection.request(
                "POST", path, body, {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            )
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise BrainError(path, None) from exc
        finally:
            connection.close()
        if response.status != 200 or len(raw) > MAX_RESPONSE_BYTES:
            raise BrainError(path, response.status)
        try:
            answer = json.loads(raw)
        except ValueError as exc:
            raise BrainError(path, response.status) from exc
        if not isinstance(answer, dict):
            raise BrainError(path, response.status)
        return answer

    def start(
        self,
        context: TurnContext,
        message: str,
        *,
        locale: str | None = None,
        conversation: Sequence[Mapping[str, object]] = (),
        turn_clock: tuple[str, str] | None = None,
    ) -> Turn:
        """Start a turn; ``conversation`` entries are ``{"role", "text", "truncated"}``."""
        payload = context.wire()
        payload["message"] = message
        payload["locale"] = locale
        if turn_clock is not None:
            payload["turn_clock"] = {"date": turn_clock[0], "timezone": turn_clock[1]}
        payload["conversation"] = [dict(entry) for entry in conversation]
        return turn(self.post("/v1/turns", payload))

    def resume(self, context: TurnContext, results: Mapping[str, object]) -> Turn:
        payload = context.wire()
        payload["results"] = dict(results)
        return turn(self.post("/v1/turns/resume", payload))

    def purpose(
        self, context: TurnContext, request: ActionRequest, assistant_name: str, summary: str
    ) -> dict[str, object]:
        """Why a pending Action pauses: the answer's ``purpose`` and ``usage``."""
        return self.post(
            "/v1/turns/purpose",
            {
                "thread_id": context.thread_id,
                "interrupt_id": request.interrupt_id,
                "assistant_id": request.assistant_id,
                "assistant_name": assistant_name,
                "action_id": request.action,
                "action_summary": summary,
                "provider": context.provider.wire(),
            },
        )

    def capability_plan(
        self, provider: Provider, objective: str, candidates: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        """A plan over candidates of ``{"id", "name", "summary", "actions", "integrations"}``."""
        payload = {"provider": provider.wire(), "objective": objective, "candidates": [dict(c) for c in candidates]}
        return self.post("/v1/capability-plan", payload)

    def intent_route(self, provider: Provider, request: RouteRequest) -> dict[str, object]:
        return self.post("/v1/intent-route", request.wire(provider))

    def browser(self, operation: str, provider: Provider, request: Mapping[str, object]) -> dict[str, object]:
        """One Browser operation (``browser-step``, ``-fold``, ``-check``, ``-final``) with its request fields."""
        return self.post(f"/v1/{operation}", {"provider": provider.wire(), **request})

    def delete_thread(self, thread_id: str) -> None:
        if self.post("/v1/threads/delete", {"thread_id": thread_id}) != {"status": "deleted"}:
            raise BrainError("/v1/threads/delete", 200)


def binary() -> Path:
    """The Rust Brain binary ``SHIMPZ_BRAIN_BINARY`` names; nothing starts without an executable one."""
    named = os.environ.get(BINARY_ENVIRONMENT, "")
    path = Path(named)
    if not named or not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"{BINARY_ENVIRONMENT} must name the built shimpz-brain binary")
    return path


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _environment() -> dict[str, str]:
    """The child's environment: the caller's, less proxies and every Shimpz setting."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _PROXIES and not key.upper().startswith("SHIMPZ_")
    }


def healthy(port: int, process: subprocess.Popen, deadline: float = HEALTH_SECONDS) -> bool:
    """Whether the running child answers ``GET /health`` with 200 before the deadline."""
    until = time.monotonic() + deadline
    while time.monotonic() < until and process.poll() is None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/health")
            if connection.getresponse().status == 200:
                return True
        except OSError, http.client.HTTPException:
            pass
        finally:
            connection.close()
        time.sleep(0.2)
    return False


def stop(process: subprocess.Popen, output: IO | None = None) -> None:
    """Stop a child: terminate, then kill after a grace period; close its log."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if output is not None:
        output.close()


class Brain:
    """One disposable Brain process with its own token, state directory, and log in a private directory."""

    def __init__(self, provider_origin: str, *, root: Path | None = None, executable: Path | None = None) -> None:
        if not provider_origin.startswith("http://127.0.0.1:"):
            raise ValueError("the provider origin must be a loopback http:// origin")
        self.provider_origin = provider_origin
        self.executable = executable
        self._root = root
        self._owned: tempfile.TemporaryDirectory | None = None
        self.process: subprocess.Popen | None = None
        self._log: IO | None = None
        self.port = 0

    @property
    def directory(self) -> Path:
        return self._root if self._root is not None else Path(self._owned.name)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def token_file(self) -> Path:
        return self.directory / "token"

    def client(self) -> Client:
        return Client(self.url, self.token_file)

    def command(self) -> list[str]:
        state = self.directory / "state"
        return [
            str(self.executable or binary()),
            "serve",
            "--listen",
            f"127.0.0.1:{self.port}",
            "--token-file",
            str(self.token_file),
            "--state-dir",
            str(state),
            "--provider-origin",
            self.provider_origin,
        ]

    def start(self) -> Brain:
        executable = self.executable or binary()
        if self._root is None:
            self._owned = tempfile.TemporaryDirectory(prefix="brain-eval-")
        self.directory.chmod(0o700)
        descriptor = os.open(self.token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(32))
        (self.directory / "state").mkdir(mode=0o700)
        self.port = free_port()
        self.executable = executable
        self._log = (self.directory / "brain.log").open("wb")
        self.process = subprocess.Popen(
            self.command(), env=_environment(), stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT
        )
        if not healthy(self.port, self.process):
            self.close()
            raise RuntimeError("the Brain did not become healthy")
        return self

    def close(self) -> None:
        if self.process is not None:
            stop(self.process, self._log)
            self.process = None
        if self._owned is not None:
            self._owned.cleanup()
            self._owned = None

    def __enter__(self) -> Brain:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class Metered:
    """A disposable Brain's client, the provider origin it sends through, and the campaign ledger both draw on."""

    client: Client
    origin: provider_origin.ProviderOrigin
    ledger: eval_cost.SharedBudget

    def budget(self) -> dict[str, object]:
        """The ledger summary and the origin's outcome counts, for a report."""
        return {**self.ledger.summary(), **self.origin.ceiling.counts}

    def refused(self) -> int:
        """Provider requests the ledger refused so far: a refusal stops the attempt that met it."""
        return self.origin.ceiling.counts["refused"]


@contextlib.contextmanager
def metered(cap: float, max_output_tokens: int) -> Iterator[Metered]:
    """One disposable Brain behind a fresh provider origin, under a new campaign ledger holding ``cap`` US dollars.

    The ledger, the origin's state, and the Brain's token, state, and log live in one private temporary directory
    that is removed afterwards.
    """
    with tempfile.TemporaryDirectory(prefix="brain-eval-") as directory:
        root = Path(directory)
        ledger = eval_cost.SharedBudget.create(root / "budget.json", cap)
        ceiling = eval_ceiling.Ceiling(0.0, max_output_tokens, root / "spend.json", budget=ledger)
        brain_root = root / "brain"
        brain_root.mkdir(mode=0o700)
        with provider_origin.ProviderOrigin(ceiling) as origin, Brain(origin.origin, root=brain_root) as brain:
            yield Metered(brain.client(), origin, ledger)


class UsageMeter:
    """The usage one attempt's Brain operations reported; a failed operation reports none, so its cost is unknown."""

    def __init__(self) -> None:
        self.total = eval_cost.Usage()

    def add(self, counts: Mapping[str, int]) -> None:
        self.total += eval_cost.Usage.of(dict(counts))

    def failed(self) -> None:
        self.total += eval_cost.Usage(unreported_calls=1)

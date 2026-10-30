"""The Routine eval's Brains: how the run starts one per attempt slot, and what each slot's provider traffic was.

Every attempt slot (one thread of one worker) has its own Rust Brain (``SHIMPZ_BRAIN_BINARY``; nothing starts without
it) and its own loopback provider origin (``eval/provider_origin.py``), so every provider request a slot's Brain sends
belongs to the one attempt running on that slot. The origin applies the evaluation ceiling against the run's one
shared ledger: each try on the wire is reserved before it is sent, and a request the budget cannot hold is refused and
never forwarded. It keeps a state snapshot of its counts (requests, refusals, throttled and terminal answers, answers
that stopped at the eval's output limit) and writes every throttled or clamped answer to its events file. An attempt
reads its slot's snapshot before and after it runs: the difference is exactly what its Brain met on the wire, beyond
what its result says.

Each Brain serves on a free loopback port with a fresh bearer in a new owner-only token file and an owner-only state
directory, and logs to its slot's log file. The origin prints its port on its first stdout line and exits as soon as
its stdin closes; the run terminates the Brain.
"""

import contextlib
import dataclasses
import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__:
    from eval.routines_fixture import BRAIN, CALL_OUTPUT_TOKENS, eval_cost
else:  # run as a script from Team, beside its sibling modules
    from routines_fixture import BRAIN, CALL_OUTPUT_TOKENS, eval_cost

BINARY_ENVIRONMENT = "SHIMPZ_BRAIN_BINARY"
ORIGIN = BRAIN / "eval" / "provider_origin.py"


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


# The origin's counts an attempt reads: each is a running total, so an attempt's share is its difference.
COUNTS = ("refused", "throttled", "terminal", "clamped")


def read_counts(state: Path) -> dict[str, int] | None:
    """A slot origin's running counts from its state snapshot, or None when the snapshot cannot be read."""
    try:
        snapshot = json.loads(state.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(snapshot, dict) or any(type(snapshot.get(name)) is not int for name in COUNTS):
        return None
    return {name: snapshot[name] for name in COUNTS}


def brain_binary() -> str:
    """The Rust Brain binary ``SHIMPZ_BRAIN_BINARY`` names; the run never starts without an executable one."""
    named = os.environ.get(BINARY_ENVIRONMENT, "")
    if not named or not Path(named).is_file() or not os.access(named, os.X_OK):
        raise SystemExit(f"{BINARY_ENVIRONMENT} must name the built shimpz-brain binary")
    return named


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@dataclasses.dataclass(frozen=True)
class SlotFiles:
    """One slot's files in the run's private directory, and where its Brain logs."""

    token: Path
    state: Path
    events: Path
    ledger: Path
    brain_state: Path
    log: Path


def start_origin(files: SlotFiles, command: tuple[str, ...] = (sys.executable, str(ORIGIN))) -> subprocess.Popen:
    """The slot's provider origin, metered against the run's ledger, in its own session."""
    arguments = ("--ledger", str(files.ledger), "--max-output-tokens", str(CALL_OUTPUT_TOKENS))
    arguments += ("--state", str(files.state), "--events", str(files.events))
    return subprocess.Popen(
        [*command, *arguments], stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True
    )


# How long a slot's origin may take to start and say its port, and its Brain to answer its health check.
START_SECONDS = 60.0


def announced_port(process: subprocess.Popen, deadline: float = START_SECONDS) -> int:
    """The port a started process says on its first stdout line; one that never says is stopped."""
    timer = threading.Timer(deadline, process.kill)
    timer.start()
    try:
        line = process.stdout.readline()
    finally:
        timer.cancel()
    with contextlib.suppress(ValueError):
        port = int(line)
        if 0 < port < 65536:
            return port
    raise RuntimeError("a provider origin stopped before it served")


def start_brain(binary: str, files: SlotFiles, origin_port: int) -> tuple[subprocess.Popen, int]:
    """The slot's Brain on a free loopback port, with a fresh owner-only bearer and state, sending to its origin."""
    descriptor = os.open(files.token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(secrets.token_urlsafe(32))
    files.brain_state.mkdir(mode=0o700)
    port = free_port()
    command = [binary, "serve", "--listen", f"127.0.0.1:{port}", "--token-file", str(files.token)]
    command += ["--state-dir", str(files.brain_state), "--provider-origin", f"http://127.0.0.1:{origin_port}"]
    with files.log.open("ab") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    return process, port


def healthy(port: int, process: subprocess.Popen, deadline: float = START_SECONDS) -> None:
    """Return once the Brain answers ``GET /health`` with 200; one that stops or never answers fails the run."""
    until = time.monotonic() + deadline
    while time.monotonic() < until and process.poll() is None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/health")
            if connection.getresponse().status == 200:
                return
        except OSError, http.client.HTTPException:
            pass
        finally:
            connection.close()
        time.sleep(0.1)
    raise RuntimeError("a Brain did not become healthy")

"""The Routine eval's Brain process: its loopback runtime API server and the meter that budgets attempts."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import secrets
import socket
import sys
import time
from pathlib import Path

if __package__:
    from eval.routines_fixture import eval_cost
else:  # run as a script from Team, beside its sibling modules
    from routines_fixture import eval_cost


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


def serve_brain(port: int, token_file: Path, log: Path | None = None) -> int:
    """Brain's real runtime API on loopback with an in-memory checkpoint, run from Brain's own environment."""
    import agent_runtime
    import runtime_api
    import uvicorn
    from langgraph.checkpoint.memory import InMemorySaver

    if log is not None:
        _log_provider_failures(log)

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

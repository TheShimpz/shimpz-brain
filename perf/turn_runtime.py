"""Measure one Brain turn with real SQLite checkpoints and a fixed local model.

Run from the Brain checkout with
``PYTHONPATH=. uv run --frozen --python 3.14 python -m perf.turn_runtime --state-parent /var/tmp``.
No provider, Team, Admin, browser, Docker, or network request is made.
Each case uses a fresh private SQLite database on a non-memory filesystem.
Results are printed only after every turn and checkpoint shape is verified.
Record ``git rev-parse HEAD`` and host limits alongside the result. Phase and
residual times are wall-clock; process CPU includes every thread and excludes
I/O wait. p50 is the median and p95 uses nearest rank.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from unittest import mock

import agent_runtime
import runtime_api
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

WARMUPS = 5
PRIOR_TURNS = 4
CASES = ((0, 0), (1, 1), (4, 8), (16, 8))
REPLY = "Fixture reply."
OBJECTIVE = "Fixture objective."
SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "maxLength": 80},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
    },
    "required": ["query"],
    "additionalProperties": False,
}


class FixedModel(FakeMessagesListChatModel):
    last_bound_count: int | None = None

    def bind_tools(self, tools, **_kwargs):
        self.last_bound_count = len(tools)
        return self


class TimedRuntime(agent_runtime.AgentRuntime):
    def __init__(self, saver: runtime_api.PruningSqliteSaver, model: FixedModel) -> None:
        super().__init__(saver, model_factory=lambda _provider: model)
        self.phase_ns: dict[str, int] = {}
        self.prior_messages = -1

    def _prepare_scope(self, context: agent_runtime.TurnContext, *, resume: bool) -> int:
        start = time.perf_counter_ns()
        try:
            self.prior_messages = super()._prepare_scope(context, resume=resume)
            return self.prior_messages
        finally:
            self.phase_ns["scope"] = time.perf_counter_ns() - start

    def _prune_history(self, thread_id: str) -> None:
        start = time.perf_counter_ns()
        try:
            return super()._prune_history(thread_id)
        finally:
            self.phase_ns["prune"] = time.perf_counter_ns() - start

    def _agent(self, context: agent_runtime.TurnContext):
        start = time.perf_counter_ns()
        try:
            return super()._agent(context)
        finally:
            self.phase_ns["agent_build"] = time.perf_counter_ns() - start


def _assistants(count: int, actions_each: int) -> tuple[agent_runtime.AssistantDefinition, ...]:
    return tuple(
        agent_runtime.AssistantDefinition(
            id=f"assistant-{assistant_index:02d}",
            genesis="Perform the declared fixture Actions only.",
            actions=tuple(
                agent_runtime.ActionDefinition(
                    id=f"action-{action_index:02d}",
                    summary="Look up a bounded fixture item.",
                    input_schema=SCHEMA,
                )
                for action_index in range(actions_each)
            ),
        )
        for assistant_index in range(count)
    )


def _summary(values: list[int], *, raw: bool = False) -> dict[str, object]:
    ordered = sorted(values)
    result: dict[str, object] = {
        "p50_ms": round(statistics.median(values) / 1_000_000, 6),
        "p95_ms": round(ordered[math.ceil(len(values) * 0.95) - 1] / 1_000_000, 6),
    }
    if raw:
        result["samples_ms"] = [round(value / 1_000_000, 6) for value in values]
        midpoint = len(values) // 2
        result["first_half_p50_ms"] = round(statistics.median(values[:midpoint]) / 1_000_000, 6)
        result["second_half_p50_ms"] = round(statistics.median(values[midpoint:]) / 1_000_000, 6)
    return result


def _context(
    case: str,
    index: int,
    assistants: tuple[agent_runtime.AssistantDefinition, ...],
    provider: agent_runtime.ProviderConfig,
) -> agent_runtime.TurnContext:
    return agent_runtime.TurnContext(f"perf:{case}:{index}", "Fixture Team", assistants, provider)


def _sample(
    runtime: TimedRuntime,
    model: FixedModel,
    context: agent_runtime.TurnContext,
    prior_turns: int,
    tool_count: int,
) -> tuple[int, int, dict[str, int]]:
    for _ in range(prior_turns):
        result = runtime.start(context, OBJECTIVE)
        if result.status != "completed" or result.reply != REPLY or result.actions:
            raise AssertionError("Brain warmup result changed")
    runtime.phase_ns.clear()
    runtime.prior_messages = -1
    model.last_bound_count = None
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    result = runtime.start(context, OBJECTIVE)
    total = time.perf_counter_ns() - start
    cpu_total = time.process_time_ns() - cpu_start
    if result.status != "completed" or result.reply != REPLY or result.actions:
        raise AssertionError("Brain turn result changed")
    if runtime.prior_messages != prior_turns * 2 or (model.last_bound_count or 0) != tool_count:
        raise AssertionError("Brain scope or declared tool binding changed")
    if set(runtime.phase_ns) != {"scope", "prune", "agent_build"}:
        raise AssertionError("Brain turn phase changed")
    residual = total - sum(runtime.phase_ns.values())
    if residual < 0:
        raise AssertionError("Brain turn phases exceed total time")
    return total, cpu_total, {**runtime.phase_ns, "residual": residual}


def _case(
    state_parent: Path,
    assistant_count: int,
    actions_each: int,
    prior_turns: int,
    samples: int,
) -> dict[str, object]:
    case = f"{assistant_count}x{actions_each}-history-{prior_turns * 2}"
    assistants = _assistants(assistant_count, actions_each)
    provider = agent_runtime.ProviderConfig("openai", "gpt-5.6-terra", "fixture-only-key")
    response_count = (samples + WARMUPS) * (prior_turns + 1) + 1
    model = FixedModel(responses=[AIMessage(content=REPLY, id=f"fixture-{index}") for index in range(response_count)])
    totals: list[int] = []
    cpu_totals: list[int] = []
    phases: dict[str, list[int]] = {name: [] for name in ("scope", "prune", "agent_build", "residual")}
    with tempfile.TemporaryDirectory(prefix="shimpz-brain-turn-", dir=state_parent) as directory:
        private = Path(directory)
        if private.stat().st_uid != os.getuid() or private.stat().st_mode & 0o777 != 0o700:
            raise RuntimeError("Brain benchmark directory is not private")
        path = private / "checkpoints.sqlite3"
        with closing(sqlite3.connect(path, check_same_thread=False)) as connection:
            path.chmod(0o600)
            connection.execute("PRAGMA secure_delete=ON")
            saver = runtime_api.PruningSqliteSaver(connection)
            saver.setup()
            runtime = TimedRuntime(saver, model)
            if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o777 != 0o600:
                raise RuntimeError("Brain benchmark checkpoint is not private")
            for index in range(samples + WARMUPS):
                context = _context(case, index, assistants, provider)
                total, cpu_total, phase = _sample(runtime, model, context, prior_turns, assistant_count * actions_each)
                if index >= WARMUPS:
                    totals.append(total)
                    cpu_totals.append(cpu_total)
                    for name, value in phase.items():
                        phases[name].append(value)
            checkpoint_rows = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            write_rows = connection.execute("SELECT COUNT(*) FROM writes").fetchone()[0]
            storage_bytes = sum(item.stat().st_size for item in private.iterdir())
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
    if private.exists():
        raise RuntimeError("Brain benchmark checkpoint directory remains")
    return {
        "assistants": assistant_count,
        "actions": assistant_count * actions_each,
        "prior_messages": prior_turns * 2,
        "total": _summary(totals, raw=True),
        "process_cpu": _summary(cpu_totals, raw=True),
        "phases_wall": {name: _summary(values) for name, values in phases.items()},
        "checkpoint_rows": checkpoint_rows,
        "write_rows": write_rows,
        "sqlite_storage_bytes": storage_bytes,
        "journal_mode": journal_mode,
        "synchronous": synchronous,
    }


def _filesystem(path: Path) -> str:
    matches = []
    for entry in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        mount, filesystem = entry.split(" - ", 1)
        mountpoint = Path(mount.split()[4].replace("\\040", " "))
        if path == mountpoint or mountpoint in path.parents:
            matches.append((len(mountpoint.parts), filesystem.split()[0]))
    if not matches:
        raise RuntimeError("Brain benchmark state filesystem is unknown")
    filesystem = max(matches)[1]
    if filesystem in {"tmpfs", "ramfs"}:
        raise RuntimeError("Brain benchmark state must use a non-memory filesystem")
    return filesystem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--state-parent", type=Path, required=True)
    args = parser.parse_args()
    if not 20 <= args.samples <= 200:
        parser.error("samples must be between 20 and 200")
    state_parent = args.state_parent.resolve(strict=True)
    if any((candidate / ".git").exists() for candidate in (state_parent, *state_parent.parents)):
        parser.error("state parent must be outside any Git working tree")
    filesystem = _filesystem(state_parent)
    results = {}
    with (
        mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network access is forbidden")),
        mock.patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network access is forbidden")),
    ):
        for prior_turns in (0, PRIOR_TURNS):
            for assistant_count, actions_each in CASES:
                name = f"{assistant_count}x{actions_each}-history-{prior_turns * 2}"
                results[name] = _case(state_parent, assistant_count, actions_each, prior_turns, args.samples)
            control_name = f"0x0-history-{prior_turns * 2}"
            restored = _case(state_parent, 0, 0, prior_turns, args.samples)
            first_p50 = results[control_name]["total"]["p50_ms"]
            restored_p50 = restored["total"]["p50_ms"]
            if abs(restored_p50 / first_p50 - 1) > 0.10:
                raise RuntimeError("Brain benchmark control drift exceeded 10%")
            results[f"restored-{control_name}"] = restored
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "filesystem": filesystem,
                "host_processors": os.cpu_count(),
                "affinity_processors": len(os.sched_getaffinity(0)),
                "host_memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
                "samples": args.samples,
                "warmups": WARMUPS,
                "cases": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

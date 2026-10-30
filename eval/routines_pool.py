"""The Routine eval's run: its processes, the schedule of its attempts, and the report their outcomes merge into.

One command starts every process the run needs and stops them all on any exit. The run confines itself, and so every
process it starts, to its CPU budget: half the processors the machine reports, or all of them on a GitHub Actions
runner. It starts worker processes, each a Team runner that runs up to ``threads`` attempts at once, and one slot per
thread of each worker: a Rust Brain and its own provider origin (``routines_serve``), so each attempt runs on a Brain
and an origin no other running attempt uses. Workers speak JSON lines: their configuration, model keys included,
arrives on stdin, never in argv or the environment; each attempt goes in as one task and comes back as one result.

The schedule hands out attempts index by index across every case and model, so a run the budget stops still covers
every stratum evenly. A provider's rate-limit or overload answers never count as a model's miss and never pass: an
attempt whose Brain request failed throttled is discarded and retried after an exponential backoff with jitter, until
it gives up and leaves its case inconclusive; any throttled answer, even one the Brain's own retry absorbed, halves that
provider's attempt concurrency, which then grows back by one attempt per round of clean results. A Brain's own
capacity refusal is retried the same way without touching the provider's concurrency. The budget's refusal stops the
run's dispatch, a provider that refuses every request (no credit or quota left, a refused key) stops that provider's,
and a case missing any attempt is inconclusive.
"""

import collections
import contextlib
import dataclasses
import json
import math
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

if __package__:
    from eval.routines_fixture import eval_cost, eval_stats
    from eval.routines_serve import ORIGIN, SlotFiles, announced_port, healthy, start_brain, start_origin
else:  # run as a script from Team, beside its sibling modules
    from routines_fixture import eval_cost, eval_stats
    from routines_serve import ORIGIN, SlotFiles, announced_port, healthy, start_brain, start_origin


# Attempts one worker runs at once, each on a slot of its own: on 2026-10-07 one worker ran 16 attempts 8 at a time in
# 41 s (19 s a wave, against 17 s for one alone), all passing.
DEFAULT_THREADS = 8


# A slot's discarded tries before it gives up and leaves its case inconclusive.
MAX_TRIES = 6


BACKOFF_SECONDS, BACKOFF_CAP_SECONDS = 2.0, 60.0


# How long a worker may take to import Team, admit the Assistant, and say it is ready.
WORKER_START_SECONDS = 180.0


# How long a stopping process may take to exit before it is terminated, and then killed.
STOP_SECONDS = 10.0


def cpu_budget() -> int:
    """Half the processors this process may use; all of them on a GitHub Actions runner (AGENTS.md)."""
    processors = os.process_cpu_count() or 1
    return processors if os.environ.get("GITHUB_ACTIONS") else max(1, processors // 2)


def confine(cpus: int) -> list[int]:
    """Pin this process, and so every process it starts, to ``cpus`` of the processors it may use."""
    allowed = sorted(os.sched_getaffinity(0))[: max(1, cpus)]
    os.sched_setaffinity(0, allowed)
    return allowed


@dataclasses.dataclass
class Slot:
    """One attempt of one case for one model, and the tries it has been discarded."""

    model: int
    provider: str
    case: str
    index: int
    tries: int = 0
    ready_at: float = 0.0
    dispatched_at: float = 0.0

    @property
    def key(self) -> tuple[int, str, int]:
        return self.model, self.case, self.index

    def task(self) -> dict[str, object]:
        return {"model": self.model, "case": self.case, "index": self.index, "try": self.tries}


def plan(models: list[tuple[str, str]], cases: list[str], attempts: int) -> list[Slot]:
    """Every slot, index by index across every case and model."""
    return [
        Slot(position, provider, case, index)
        for index in range(attempts)
        for case in cases
        for position, (provider, _model) in enumerate(models)
    ]


def backoff(failures: int, rng: random.Random) -> float:
    """Exponential backoff with equal jitter: half the doubled delay, plus up to as much again at random."""
    delay = min(BACKOFF_CAP_SECONDS, BACKOFF_SECONDS * 2 ** (failures - 1))
    return delay / 2 + rng.uniform(0, delay / 2)


@dataclasses.dataclass
class ProviderState:
    """One provider's adaptive attempt concurrency and the throttling it met."""

    limit: float
    cut_at: float = -math.inf
    paused_until: float = 0.0
    failures: int = 0
    lowest: float = math.inf
    throttled_answers: int = 0
    discarded: int = 0
    given_up: int = 0
    # The provider refuses every request, as with no credit left: nothing more of it is dispatched.
    unavailable: bool = False

    def report(self) -> dict[str, object]:
        return {
            "throttled_answers": self.throttled_answers,
            "discarded_attempts": self.discarded,
            "given_up": self.given_up,
            "lowest_concurrency": None if self.lowest == math.inf else int(self.lowest),
            "final_concurrency": int(self.limit),
            "unavailable": self.unavailable,
        }


class Schedule:
    """Which slot runs next, and what each finished slot's result does to the run."""

    def __init__(self, slots: Iterable[Slot], capacity: int, rng: random.Random | None = None) -> None:
        self.waiting = list(slots)
        self.capacity = capacity
        self.providers = {slot.provider: ProviderState(float(capacity)) for slot in self.waiting}
        self.running: dict[tuple[int, str, int], Slot] = {}
        self.results: dict[tuple[int, str, int], dict[str, object]] = {}
        self.dispatched: set[tuple[int, str]] = set()
        self.stopped = False
        self.brain_refusals = 0
        # Responses at the eval's output limit that any attempt received, counted or discarded.
        self.clamped = 0
        self.peak = 0
        self.peak_by: collections.Counter[str] = collections.Counter()
        self.rng = rng or random.Random()

    def _running(self, provider: str) -> int:
        return sum(slot.provider == provider for slot in self.running.values())

    def _ready(self, slot: Slot, now: float) -> bool:
        state = self.providers[slot.provider]
        return (
            slot.ready_at <= now
            and state.paused_until <= now
            and self._running(slot.provider) < max(1, int(state.limit))
        )

    def take(self, now: float) -> Slot | None:
        """The first waiting slot its provider has room for now, or None."""
        if self.stopped or len(self.running) >= self.capacity:
            return None
        for position, slot in enumerate(self.waiting):
            if self._ready(slot, now):
                del self.waiting[position]
                slot.dispatched_at = now
                self.running[slot.key] = slot
                self.dispatched.add((slot.model, slot.case))
                self.peak = max(self.peak, len(self.running))
                self.peak_by[slot.provider] = max(self.peak_by[slot.provider], self._running(slot.provider))
                return slot
        return None

    def finish(self, key: tuple[int, str, int], result: dict[str, object], now: float) -> None:
        slot = self.running.pop(key)
        state = self.providers[slot.provider]
        state.throttled_answers += result["throttled"]
        self.clamped += result["clamped"]
        disposition = result["disposition"]
        if disposition == "stopped":
            # The budget refused a request: nothing more is dispatched, and this slot stays unfinished.
            self.stopped = True
            return
        if disposition == "unavailable" or (state.unavailable and disposition != "counted"):
            # The provider refuses every request: none of its slots runs again, and each stays unfinished.
            state.unavailable = True
            self.waiting = [item for item in self.waiting if item.provider != slot.provider]
            return
        if disposition == "throttled" or result["throttled"]:
            self._cut(slot, state, now)
        elif slot.dispatched_at >= state.cut_at:
            # Only an attempt dispatched since the last cut shows that the provider holds the current concurrency.
            state.limit = min(self.capacity, state.limit + 1 / state.limit)
            state.failures = 0
        if disposition == "counted":
            self.results[key] = result
        else:
            self._retry(slot, state, now, provider=disposition == "throttled")

    def _cut(self, slot: Slot, state: ProviderState, now: float) -> None:
        """Halve the provider's concurrency, once for every attempt that ran at the concurrency it just refused."""
        if slot.dispatched_at < state.cut_at:
            return
        state.limit = max(1.0, (self._running(slot.provider) + 1) / 2)
        state.lowest = min(state.lowest, state.limit)
        state.cut_at = now

    def _retry(self, slot: Slot, state: ProviderState, now: float, *, provider: bool) -> None:
        slot.tries += 1
        state.discarded += provider
        self.brain_refusals += not provider
        if slot.tries >= MAX_TRIES:
            state.given_up += 1
            return
        state.failures += provider
        slot.ready_at = now + backoff(max(1, state.failures), self.rng)
        if provider:
            state.paused_until = max(state.paused_until, slot.ready_at)
        self.waiting.insert(0, slot)

    def done(self) -> bool:
        return not self.running and (self.stopped or not self.waiting)

    def wake(self, now: float) -> float | None:
        """Seconds until a waiting slot's backoff ends, or None when only a result can free one."""
        if self.stopped:
            return None
        times = [max(slot.ready_at, self.providers[slot.provider].paused_until) - now for slot in self.waiting]
        later = [moment for moment in times if moment > 0]
        return min(later) if later else None


def case_summary(case_id: str, results: list[dict[str, object]], inconclusive: bool) -> dict[str, object]:
    """One case's line of the report, from its counted attempts in index order."""
    passed = sum(bool(item["passed"]) for item in results)
    costs = [eval_cost.Cost(item["usd"], item["known"]) for item in results]
    trials = len(costs)
    interval = eval_stats.wilson(passed, trials)
    total = sum(item.usd for item in costs)
    questions = collections.Counter(question for item in results for question in item["questions"])
    return {
        "id": case_id,
        "passed": passed,
        "trials": trials,
        "wilson95_lower": None if interval is None else round(interval[0], 3),
        "inconclusive": inconclusive,
        "misses": [{"reason": item["reason"], "cause": item["cause"]} for item in results if not item["passed"]],
        "schedules": [item["schedule"] for item in results],
        "usd": round(total, 6),
        "usd_per_attempt": round(total / trials, 6) if trials else None,
        "cost_known": all(item.known for item in costs),
        "questions": dict(questions),
    }


def merge(models: list[tuple[str, str]], cases: list[str], attempts: int, schedule: Schedule) -> list[dict]:
    """Every model's cases in the run's order; a case no attempt of which was dispatched is left out."""
    merged = []
    for position, (provider, model) in enumerate(models):
        rows = []
        for case in cases:
            if (position, case) not in schedule.dispatched:
                continue
            found = [schedule.results.get((position, case, index)) for index in range(attempts)]
            counted = [item for item in found if item is not None]
            rows.append(case_summary(case, counted, inconclusive=len(counted) < attempts))
        merged.append({"provider": provider, "model": model, "cases": rows})
    return merged


class PoolError(RuntimeError):
    """A worker, Brain, or provider origin stopped, failed to start, or reported an error: the run is aborted."""


@dataclasses.dataclass
class BrainSlot:
    """One attempt slot: its provider origin, the Brain that sends to it, and the files both use."""

    number: int
    files: SlotFiles
    origin: subprocess.Popen
    brain: subprocess.Popen | None = None
    url: str = ""

    def config(self) -> dict[str, str]:
        """What the worker's attempt on this slot needs: its Brain, bearer file, and origin state snapshot."""
        return {"brain": self.url, "token": str(self.files.token), "state": str(self.files.state)}


@dataclasses.dataclass
class Worker:
    number: int
    process: subprocess.Popen
    slots: list[BrainSlot]
    threads: int
    busy: int = 0


class Pool:
    """The run's processes: one worker per number, one Brain and origin per thread of it, every one stopped on exit.

    ``command`` is the worker's interpreter and script, which runs with ``--worker``; ``brain`` is the Brain binary,
    the run's private directory (where the shared ledger already is), and the directory that keeps each Brain's log
    (or None for the run's directory); ``configure`` gives worker ``number`` its configuration, given its slots'.
    """

    def __init__(
        self,
        command: tuple[str, str],
        brain: tuple[str, Path, Path | None],
        count: int,
        threads: int,
        configure: Callable[[int, list[dict[str, str]]], dict[str, object]],
        origin: tuple[str, ...] = (sys.executable, str(ORIGIN)),
    ) -> None:
        self.command, self.count, self.threads, self.configure = command, count, threads, configure
        self.binary, self.run_dir, self.log_dir = brain
        self.origin_command = origin
        self.workers: list[Worker] = []
        self.slots: list[BrainSlot] = []
        self.readers: list[threading.Thread] = []
        self.messages: queue.Queue = queue.Queue()
        self.placed: dict[tuple[int, str, int], Worker] = {}

    def __enter__(self) -> Pool:
        try:
            self._start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def slot_files(self, number: int) -> SlotFiles:
        name = f"slot-{number}"
        run = self.run_dir
        logs = self.run_dir if self.log_dir is None else self.log_dir
        return SlotFiles(
            run / f"{name}.token",
            run / f"{name}.state.json",
            run / f"{name}.events",
            run / "budget.json",
            run / name,
            logs / f"brain-{number}.log",
        )

    def _start(self) -> None:
        python, script = self.command
        # Each process is recorded as soon as it starts, so a failure to start the next one still stops it. Each runs
        # in its own session, so stopping it also stops what it started, such as an Assistant's SDK.
        for number in range(self.count):
            slots = []
            for thread in range(self.threads):
                files = self.slot_files(number * self.threads + thread)
                slots.append(BrainSlot(number * self.threads + thread, files, start_origin(files, self.origin_command)))
                self.slots.append(slots[-1])
            process = subprocess.Popen(
                [python, script, "--worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True
            )
            self.workers.append(Worker(number, process, slots, self.threads))
        self.readers = [
            threading.Thread(target=self._read, args=(worker,), name=f"worker-{worker.number}", daemon=True)
            for worker in self.workers
        ]
        for reader in self.readers:
            reader.start()
        for slot in self.slots:
            self._serve(slot)
        for worker in self.workers:
            self._send(worker, self.configure(worker.number, [slot.config() for slot in worker.slots]))
        deadline = time.monotonic() + WORKER_START_SECONDS
        for _worker in self.workers:
            message = self._next(max(0.0, deadline - time.monotonic()))
            if message is None or message[1].get("ready") is not True:
                raise PoolError("a worker did not start")

    def _serve(self, slot: BrainSlot) -> None:
        """Start the slot's Brain against its origin's port and wait until it answers its health check."""
        try:
            origin_port = announced_port(slot.origin)
            slot.brain, port = start_brain(self.binary, slot.files, origin_port)
            healthy(port, slot.brain)
        except RuntimeError as exc:
            raise PoolError(str(exc)) from None
        slot.url = f"http://127.0.0.1:{port}"

    def _read(self, worker: Worker) -> None:
        try:
            for line in worker.process.stdout:
                self.messages.put((worker, json.loads(line)))
        except ValueError:
            self.messages.put((worker, {"error": "an unreadable message"}))
        self.messages.put((worker, None))

    def _send(self, worker: Worker, message: dict[str, object]) -> None:
        try:
            worker.process.stdin.write(json.dumps(message).encode() + b"\n")
            worker.process.stdin.flush()
        except OSError as exc:
            raise PoolError("a worker stopped") from exc

    def _next(self, timeout: float | None) -> tuple[Worker, dict[str, object]] | None:
        try:
            worker, message = self.messages.get(timeout=timeout)
        except queue.Empty:
            return None
        if message is None:
            raise PoolError(f"worker {worker.number} stopped")
        if "error" in message:
            raise PoolError(f"worker {worker.number} failed: {message['error']}")
        return worker, message

    def free(self) -> Worker | None:
        """The worker with the most free threads, or None when every thread is busy."""
        worker = max(self.workers, key=lambda item: item.threads - item.busy)
        return worker if worker.busy < worker.threads else None

    def dispatch(self, worker: Worker, slot: Slot) -> None:
        self._send(worker, slot.task())
        worker.busy += 1
        self.placed[slot.key] = worker

    def _stopped(self) -> str | None:
        """Which slot process stopped, if any: every Brain and origin must outlive the run."""
        for slot in self.slots:
            if slot.origin.poll() is not None:
                return f"origin {slot.number} stopped"
            if slot.brain is not None and slot.brain.poll() is not None:
                return f"brain {slot.number} stopped"
        return None

    def receive(self, timeout: float | None) -> tuple[tuple[int, str, int], dict[str, object]] | None:
        """The next result, or None when ``timeout`` passes first; a stopped Brain, origin, or worker aborts the run."""
        stopped = self._stopped()
        if stopped is not None:
            raise PoolError(stopped)
        found = self._next(timeout)
        if found is None:
            return None
        _worker, message = found
        key = (message["model"], message["case"], message["index"])
        self.placed.pop(key).busy -= 1
        return key, message

    def stop(self) -> None:
        """Close the stdin of every worker and origin, so it exits, and terminate every Brain; reap them all.

        One that does not exit is terminated, then killed; then whatever is left of each process's session is killed,
        so nothing it started outlives the run.
        """
        piped = [item.process for item in self.workers] + [slot.origin for slot in self.slots]
        brains = [slot.brain for slot in self.slots if slot.brain is not None]
        for process in piped:
            if process.stdin is not None and not process.stdin.closed:
                with contextlib.suppress(OSError):
                    process.stdin.close()
        for process in brains:
            process.terminate()
        for process in piped + brains:
            for end in (lambda: None, process.terminate, process.kill):
                end()
                try:
                    process.wait(timeout=STOP_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    continue
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        for reader in self.readers:
            reader.join(timeout=STOP_SECONDS)
        for process in piped:
            process.stdout.close()


# How long the run waits for a result when no backoff is pending, before it checks its Brains again.
POLL_SECONDS = 1.0


def orchestrate(schedule: Schedule, pool: Pool) -> None:
    """Dispatch every slot the schedule allows to the freest worker, and settle each result, until the run is done."""
    while not schedule.done():
        now = time.monotonic()
        while (worker := pool.free()) is not None and (slot := schedule.take(now)) is not None:
            pool.dispatch(worker, slot)
        wake = schedule.wake(now)
        received = pool.receive(POLL_SECONDS if wake is None else min(wake, POLL_SECONDS))
        if received is not None:
            schedule.finish(*received, time.monotonic())


def evidence(run_dir: Path, schedule: Schedule) -> dict[str, int]:
    """What every slot's origin met on the wire against what the attempts received: any shortfall is evidence lost.

    An attempt receives its slot's counts when it ends; one that ended without them leaves its slot's answers unread.
    """
    lines = [
        json.loads(line)
        for path in sorted(run_dir.glob("slot-*.events"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    received = sum(state.throttled_answers for state in schedule.providers.values())
    return {
        "brain_throttled": sum(item["throttled"] for item in lines),
        "brain_clamped": sum(item["clamped"] for item in lines),
        "received_throttled": received,
        "received_clamped": schedule.clamped,
    }

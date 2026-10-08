"""Provider-free checks of the Routine eval's run: its schedule, throttle handling, merge, and processes."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import routines_fake_brain
from eval import routines_pool
from eval import stats as eval_stats

FAKE_WORKER = Path(__file__).with_name("routines_fake_worker.py")
MODELS = [("openai", "gpt-6-luna"), ("anthropic", "claude-sonnet-5-5")]


def _result(slot: routines_pool.Slot, **changes) -> dict[str, object]:
    return {
        **slot.task(),
        "disposition": "counted",
        "throttled": 0,
        "clamped": 0,
        "passed": True,
        "reason": None,
        "cause": None,
        "schedule": {"kind": "calendar"},
        "questions": [],
        "usd": 0.01,
        "known": True,
        "variants": None,
    } | changes


def _alive(stat: Path) -> bool:
    """Whether the process whose /proc stat file this is still runs: neither gone nor a zombie."""
    try:
        return stat.read_text(encoding="utf-8").split(") ")[1][0] not in "ZX"
    except FileNotFoundError, ProcessLookupError:
        return False


class PlanTests(unittest.TestCase):
    def test_slots_go_index_by_index_across_every_case_and_model(self):
        slots = routines_pool.plan(MODELS, ["a", "b"], 2)
        self.assertEqual(
            [slot.key for slot in slots],
            [(0, "a", 0), (1, "a", 0), (0, "b", 0), (1, "b", 0), (0, "a", 1), (1, "a", 1), (0, "b", 1), (1, "b", 1)],
        )
        self.assertEqual({slot.provider for slot in slots[:2]}, {"openai", "anthropic"})
        self.assertEqual(slots[0].task(), {"model": 0, "case": "a", "index": 0, "try": 0})

    def test_backoff_doubles_with_equal_jitter_up_to_its_cap(self):
        rng = random.Random(7)
        for failures, low, high in ((1, 1.0, 2.0), (3, 4.0, 8.0), (20, 30.0, 60.0)):
            for _ in range(50):
                self.assertTrue(low <= routines_pool.backoff(failures, rng) <= high)

    def test_the_cpu_budget_is_half_the_processors_or_all_of_them_on_a_runner(self):
        with mock.patch.object(routines_pool.os, "process_cpu_count", return_value=96):
            with mock.patch.dict(routines_pool.os.environ, {}, clear=True):
                self.assertEqual(routines_pool.cpu_budget(), 48)
            with mock.patch.dict(routines_pool.os.environ, {"GITHUB_ACTIONS": "true"}):
                self.assertEqual(routines_pool.cpu_budget(), 96)
        with (
            mock.patch.object(routines_pool.os, "process_cpu_count", return_value=None),
            mock.patch.dict(routines_pool.os.environ, {}, clear=True),
        ):
            self.assertEqual(routines_pool.cpu_budget(), 1)

    def test_confine_pins_the_run_to_its_budget_of_processors(self):
        with (
            mock.patch.object(routines_pool.os, "sched_getaffinity", return_value={5, 1, 3, 2}),
            mock.patch.object(routines_pool.os, "sched_setaffinity") as pinned,
        ):
            self.assertEqual(routines_pool.confine(2), [1, 2])
            self.assertEqual(routines_pool.confine(0), [1])
        pinned.assert_called_with(0, [1])


class ScheduleTests(unittest.TestCase):
    def _schedule(
        self, attempts: int = 3, capacity: int = 4, cases: tuple[str, ...] = ("a",)
    ) -> routines_pool.Schedule:
        slots = routines_pool.plan(MODELS[:1], list(cases), attempts)
        return routines_pool.Schedule(slots, capacity, random.Random(3))

    def _take_all(self, schedule: routines_pool.Schedule, now: float = 0.0) -> list[routines_pool.Slot]:
        taken = []
        while (slot := schedule.take(now)) is not None:
            taken.append(slot)
        return taken

    def test_capacity_and_the_provider_limit_bound_what_runs(self):
        schedule = self._schedule(attempts=6, capacity=4)
        taken = self._take_all(schedule)
        self.assertEqual(
            ([slot.index for slot in taken], schedule.peak, dict(schedule.peak_by)), ([0, 1, 2, 3], 4, {"openai": 4})
        )
        schedule.finish(taken[0].key, _result(taken[0]), 1.0)
        self.assertEqual(schedule.take(1.0).index, 4)
        self.assertEqual(schedule.dispatched, {(0, "a")})
        self.assertFalse(schedule.done())

    def test_a_throttled_attempt_is_discarded_retried_after_backoff_and_halves_the_provider(self):
        schedule = self._schedule(attempts=4, capacity=4)
        taken = self._take_all(schedule)
        schedule.finish(taken[0].key, _result(taken[0], disposition="throttled", throttled=3), 10.0)
        state = schedule.providers["openai"]
        self.assertEqual((state.limit, state.lowest, state.throttled_answers, state.discarded), (2.0, 2.0, 3, 1))
        self.assertNotIn(taken[0].key, schedule.results)
        retried = schedule.waiting[0]
        self.assertEqual((retried.key, retried.tries), (taken[0].key, 1))
        self.assertTrue(11.0 <= retried.ready_at <= 12.0)
        self.assertEqual(state.paused_until, retried.ready_at)
        self.assertIsNone(schedule.take(11.0))
        self.assertAlmostEqual(schedule.wake(10.0), retried.ready_at - 10.0)
        # A throttle from an attempt dispatched before the cut never cuts again, and its clean result never grows it.
        schedule.finish(taken[1].key, _result(taken[1], throttled=1), 10.5)
        schedule.finish(taken[2].key, _result(taken[2]), 10.6)
        self.assertEqual(state.limit, 2.0)
        self.assertEqual(set(schedule.results), {taken[1].key, taken[2].key})
        again = schedule.take(retried.ready_at)
        self.assertEqual(again.key, taken[0].key)
        schedule.finish(again.key, _result(again), 13.0)
        self.assertEqual((state.limit, state.failures), (2.5, 0))
        self.assertEqual(state.report()["final_concurrency"], 2)

    def test_an_absorbed_throttle_counts_the_attempt_but_cuts_the_provider(self):
        schedule = self._schedule(attempts=2, capacity=4)
        first, second = self._take_all(schedule)
        schedule.finish(first.key, _result(first, throttled=2), 5.0)
        self.assertEqual((schedule.providers["openai"].limit, set(schedule.results)), (1.0, {first.key}))
        self.assertEqual(schedule.providers["openai"].paused_until, 0.0)
        schedule.finish(second.key, _result(second), 6.0)
        self.assertTrue(schedule.done())

    def test_a_slot_gives_up_after_its_last_try_and_stays_unfinished(self):
        schedule = self._schedule(attempts=1)
        now = 0.0
        for _try in range(routines_pool.MAX_TRIES):
            now += 100.0
            slot = schedule.take(now)
            schedule.finish(slot.key, _result(slot, disposition="throttled"), now)
        state = schedule.providers["openai"]
        self.assertEqual((state.discarded, state.given_up, schedule.results), (routines_pool.MAX_TRIES, 1, {}))
        self.assertTrue(schedule.done())
        self.assertEqual(state.report()["given_up"], 1)

    def test_brains_own_refusal_retries_without_touching_the_provider(self):
        schedule = self._schedule(attempts=1)
        slot = schedule.take(0.0)
        schedule.finish(slot.key, _result(slot, disposition="brain-refused"), 1.0)
        state = schedule.providers["openai"]
        self.assertEqual((schedule.brain_refusals, state.discarded, state.paused_until, state.limit), (1, 0, 0.0, 4.0))
        self.assertEqual(schedule.waiting[0].tries, 1)
        self.assertEqual(state.report()["lowest_concurrency"], None)

    def test_the_budgets_refusal_stops_every_dispatch(self):
        schedule = self._schedule(attempts=3)
        first, second, _third = self._take_all(schedule)
        schedule.finish(first.key, _result(first, disposition="stopped"), 1.0)
        self.assertIsNone(schedule.take(2.0))
        self.assertIsNone(schedule.wake(2.0))
        self.assertFalse(schedule.done())
        schedule.finish(second.key, _result(second), 2.0)
        schedule.finish(_third.key, _result(_third), 2.0)
        self.assertTrue(schedule.done())
        self.assertNotIn(first.key, schedule.results)

    def test_a_provider_that_refuses_every_request_runs_no_more_slots(self):
        schedule = routines_pool.Schedule(routines_pool.plan(MODELS, ["a"], 3), 3)
        first, second, third = (schedule.take(0.0) for _ in range(3))
        self.assertEqual([slot.provider for slot in (first, second, third)], ["openai", "anthropic", "openai"])
        schedule.finish(second.key, _result(second, disposition="unavailable", clamped=1), 1.0)
        self.assertEqual({slot.provider for slot in schedule.waiting}, {"openai"})
        self.assertTrue(schedule.providers["anthropic"].report()["unavailable"])
        late = schedule.take(1.0)
        schedule.finish(first.key, _result(first), 1.0)
        schedule.finish(third.key, _result(third, disposition="throttled", clamped=2), 1.0)
        self.assertEqual((late.provider, schedule.clamped), ("openai", 3))
        anthropic = routines_pool.Schedule(routines_pool.plan(MODELS[1:], ["a"], 3), 3)
        running = [anthropic.take(0.0) for _ in range(3)]
        anthropic.finish(running[0].key, _result(running[0], disposition="unavailable"), 1.0)
        anthropic.finish(running[1].key, _result(running[1], disposition="throttled"), 1.0)
        anthropic.finish(running[2].key, _result(running[2]), 1.0)
        self.assertEqual((anthropic.waiting, list(anthropic.results)), ([], [running[2].key]))
        self.assertTrue(anthropic.done())

    def test_wake_waits_only_for_a_backoff(self):
        schedule = self._schedule(attempts=2, capacity=1)
        self.assertIsNone(schedule.wake(0.0))
        schedule.waiting[1].ready_at = 5.0
        self.assertEqual(schedule.wake(1.0), 4.0)


class MergeTests(unittest.TestCase):
    def test_each_case_merges_its_counted_attempts_in_index_order(self):
        schedule = routines_pool.Schedule(routines_pool.plan(MODELS, ["a", "b", "c"], 3), 100)
        taken = {slot.key: slot for slot in iter(lambda: schedule.take(0.0), None)}
        # Sonnet's case c never ran; luna's b lost its last attempt to a throttle; luna's a missed once.
        schedule.dispatched.discard((1, "c"))
        for key, slot in reversed(taken.items()):
            if key[:2] == (1, "c") or key == (0, "b", 2):
                continue
            missed = {"passed": False, "reason": "calls", "cause": "no-zone-lookup"} if key == (0, "a", 1) else {}
            asked = {"questions": ["routine-output-unstated"]} if slot.case == "b" else {}
            schedule.finish(key, _result(slot, **missed, **asked, usd=0.01 * (slot.index + 1)), 1.0)
        luna, sonnet = routines_pool.merge(MODELS, ["a", "b", "c"], 3, schedule)
        self.assertEqual((luna["provider"], luna["model"]), MODELS[0])
        self.assertEqual([case["id"] for case in sonnet["cases"]], ["a", "b"])
        a, b, c = luna["cases"]
        self.assertEqual((a["passed"], a["trials"], a["inconclusive"]), (2, 3, False))
        self.assertEqual(a["misses"], [{"reason": "calls", "cause": "no-zone-lookup"}])
        self.assertEqual(a["wilson95_lower"], round(eval_stats.wilson(2, 3)[0], 3))
        self.assertEqual((a["usd"], a["usd_per_attempt"], a["cost_known"]), (0.06, 0.02, True))
        self.assertEqual(a["schedules"], [{"kind": "calendar"}] * 3)
        self.assertEqual((b["trials"], b["inconclusive"], b["questions"]), (2, True, {"routine-output-unstated": 2}))
        self.assertEqual((c["trials"], c["passed"], c["inconclusive"]), (3, 3, False))

    def test_a_case_without_a_counted_attempt_reports_nothing_known(self):
        summary = routines_pool.case_summary("a", [], inconclusive=True)
        self.assertEqual((summary["wilson95_lower"], summary["usd_per_attempt"], summary["trials"]), (None, None, 0))
        unknown = routines_pool.case_summary(
            "a", [_result(routines_pool.Slot(0, "openai", "a", 0), known=False)], False
        )
        self.assertFalse(unknown["cost_known"])


class PoolTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name)
        self.python = routines_fake_brain.launcher(self.run_dir)

    @property
    def brains(self) -> list[subprocess.Popen]:
        return self.pool.brains

    def _pool(self, count: int = 2, threads: int = 2, results: dict | None = None, mode: str = "serve"):
        def configure(_number: int, url: str, token: Path) -> dict[str, object]:
            return {"mode": mode, "results": results or {}, "brain": url, "token": str(token)}

        brains = (self.python, self.run_dir, self.run_dir / "logs")
        self.pool = routines_pool.Pool((sys.executable, str(FAKE_WORKER)), brains, count, threads, configure)
        return self.pool

    def test_a_run_settles_every_slot_across_its_workers_each_pinned_to_its_own_brain(self):
        throttled = {"disposition": "throttled", "throttled": 1}
        results = {"a:0:0": throttled, "b:2:0": {"passed": False, "reason": "schedule"}}
        schedule = routines_pool.Schedule(routines_pool.plan(MODELS, ["a", "b"], 3), 4, random.Random(1))
        with mock.patch.object(routines_pool, "BACKOFF_SECONDS", 0.01), self._pool(results=results) as pool:
            files = pool.brain_files(1)
            routines_pool.orchestrate(schedule, pool)
            self.assertEqual(sum(worker.busy for worker in pool.workers), 0)
        self.assertEqual(len(schedule.results), 12)
        self.assertEqual(
            {item["brain"] for item in schedule.results.values()}, {"http://127.0.0.1:40000", "http://127.0.0.1:40001"}
        )
        self.assertEqual(schedule.results[(0, "a", 0)]["try"], 1)
        self.assertEqual(schedule.providers["openai"].discarded, 1)
        self.assertEqual(
            (files.token.name, files.events.name, files.ledger.name, files.log.name),
            ("brain-1.token", "brain-1.events", "budget.json", "brain-1.log"),
        )
        self.assertTrue(all(process.poll() is not None for process in self.brains))
        self.assertTrue(all(worker.process.poll() == 0 for worker in pool.workers))

    def test_a_worker_that_stops_or_fails_aborts_the_run(self):
        for mode, message in (("exit", "worker 0 stopped"), ("fail", "failed: RuntimeError: scripted")):
            with self.subTest(mode=mode), self._pool(count=1, mode=mode) as pool:
                schedule = routines_pool.Schedule(routines_pool.plan(MODELS, ["a"], 1), 2)
                with self.assertRaisesRegex(routines_pool.PoolError, message):
                    routines_pool.orchestrate(schedule, pool)

    def test_a_worker_or_brain_that_never_starts_aborts_before_any_attempt(self):
        with mock.patch.object(routines_pool, "WORKER_START_SECONDS", 0.5):
            for mode, message in (("silent", "did not start"), ("garbage", "an unreadable message")):
                with self.subTest(mode=mode), self.assertRaisesRegex(routines_pool.PoolError, message):
                    self._pool(count=1, mode=mode).__enter__()
        with (
            mock.patch.dict(routines_pool.os.environ, {"FAKE_BRAIN_MODE": "quiet"}),
            self.assertRaisesRegex(routines_pool.PoolError, "before it served"),
        ):
            self._pool(count=1).__enter__()
        self.assertTrue(all(process.poll() is not None for process in self.brains))

    def test_a_stopped_brain_aborts_the_run(self):
        with mock.patch.dict(routines_pool.os.environ, {"FAKE_BRAIN_MODE": "dead"}), self._pool(count=2) as pool:
            self.brains[1].wait(timeout=5)
            with self.assertRaisesRegex(routines_pool.PoolError, "brain 1 stopped"):
                pool.receive(0.1)

    def test_stop_terminates_then_kills_a_process_that_will_not_exit(self):
        with mock.patch.object(routines_pool, "STOP_SECONDS", 0.3), self._pool(count=1, mode="stubborn") as pool:
            self.assertIsNone(pool.receive(0.01))
            worker = pool.workers[0].process
        self.assertEqual(worker.returncode, -9)

    def test_stop_also_stops_what_a_worker_started(self):
        with self._pool(count=1, mode="child"):
            marker = Path(self.run_dir, "brain-0.token.child")
            for _ in range(500):
                if marker.exists() and marker.read_text(encoding="utf-8"):
                    break
                time.sleep(0.01)
        child = Path(f"/proc/{marker.read_text(encoding='utf-8')}/stat")
        for _ in range(500):
            if not _alive(child):
                break
            time.sleep(0.01)
        self.assertFalse(_alive(child))

    def test_the_freest_worker_takes_the_next_slot_until_every_thread_is_busy(self):
        with self._pool(count=2, threads=1) as pool:
            first = pool.free()
            first.busy = 1
            second = pool.free()
            second.busy = 1
            self.assertEqual((first.number, second.number, pool.free()), (0, 1, None))
            first.busy = second.busy = 0

    def test_evidence_sets_what_every_brain_met_against_what_the_attempts_received(self):
        line = {"status": 429, "throttled": True, "clamped": False, "attributed": True}
        lines = [line, {**line, "attributed": False}, {**line, "status": 200, "throttled": False, "clamped": True}]
        Path(self.run_dir, "brain-0.events").write_text("".join(json.dumps(item) + "\n" for item in lines[:2]))
        Path(self.run_dir, "brain-3.events").write_text(json.dumps(lines[2]) + "\n")
        schedule = routines_pool.Schedule(routines_pool.plan(MODELS, ["a"], 1), 2)
        for slot in (schedule.take(0.0), schedule.take(0.0)):
            schedule.finish(slot.key, _result(slot, throttled=1), 1.0)
        self.assertEqual(
            routines_pool.evidence(self.run_dir, schedule),
            {
                "brain_throttled": 2,
                "brain_clamped": 1,
                "unattributed": 1,
                "received_throttled": 2,
                "received_clamped": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()

"""Provider-free checks for evaluation pricing, unknown usage, budgets, and per-task cost (ADR-0094)."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import agent_runtime
import model_usage
from eval import cost

LUNA = cost.Usage(model_calls=2, input_tokens=1_000, output_tokens=100, cache_read_tokens=600)


class PriceTests(unittest.TestCase):
    def test_usage_fields_are_the_brain_usage_fields(self):
        self.assertEqual(cost.FIELDS, model_usage.FIELDS)

    def test_catalog_prices_per_token_with_frozen_cache_multipliers(self):
        luna = cost.price("gpt-6-luna")
        self.assertEqual(luna.provider, "openai")
        self.assertAlmostEqual(luna.input, 0.1e-6)
        self.assertAlmostEqual(luna.output, 0.5e-6)
        self.assertAlmostEqual(luna.cache_read, 0.01e-6)
        # GPT-5.6 and later bill a cache write at 1.25 times input (OpenAI pricing page, verified 2026-10-09).
        self.assertAlmostEqual(luna.cache_write, 0.125e-6)
        self.assertAlmostEqual(cost.price("gpt-6.1-sol").cache_write, 2.5e-6)
        sonnet = cost.price("claude-sonnet-5-5")
        self.assertAlmostEqual(sonnet.cache_read, 0.2e-6)
        self.assertAlmostEqual(sonnet.cache_write, 2.5e-6)
        self.assertAlmostEqual(cost.price("claude-opus-5-5").cache_read, 0.2e-6)
        self.assertAlmostEqual(cost.price("gpt-6.1-sol").cache_read, 0.1e-6)
        with self.assertRaisesRegex(ValueError, "unknown model"):
            cost.price("gpt-unknown")

    def test_cache_reads_and_writes_are_never_priced_as_fresh_input(self):
        usage = cost.Usage(model_calls=1, input_tokens=1_000, output_tokens=10, cache_read_tokens=500)
        usage += cost.Usage(cache_write_tokens=200)
        self.assertEqual(usage.fresh_input_tokens, 300)
        spent = cost.cost(usage, "claude-sonnet-5-5")
        self.assertAlmostEqual(spent.usd, (300 * 2 + 500 * 0.2 + 200 * 2.5 + 10 * 10) * 1e-6)
        self.assertTrue(spent.known)
        self.assertEqual(cost.Usage(input_tokens=1, cache_read_tokens=5).fresh_input_tokens, 0)

    def test_a_failed_or_unreported_call_makes_usage_unknown(self):
        self.assertTrue(LUNA.known)
        for name in ("failed_calls", "unreported_calls"):
            unknown = cost.Usage.of({**LUNA.to_dict(), name: 1})
            self.assertFalse(unknown.known)
            self.assertFalse(cost.cost(unknown, "gpt-6-luna").known)
        self.assertFalse((cost.Cost(1.0) + cost.Cost(2.0, known=False)).known)
        for invalid in (
            {"model_calls": 1},
            {**LUNA.to_dict(), "input_tokens": -1},
            {**LUNA.to_dict(), "output_tokens": 1.5},
        ):
            with self.assertRaisesRegex(ValueError, "invalid usage"):
                cost.Usage.of(invalid)

    def test_an_evaluation_only_model_is_priced_and_admitted_only_where_asked(self):
        sol = cost.price("gpt-5.6-sol")
        self.assertEqual(sol.provider, "openai")
        self.assertAlmostEqual(sol.input, 4e-6)
        self.assertAlmostEqual(sol.output, 20e-6)
        self.assertAlmostEqual(sol.cache_read, 0.4e-6)
        self.assertAlmostEqual(sol.cache_write, 5e-6)
        self.assertNotIn("gpt-5.6-sol", agent_runtime.MODELS_BY_PROVIDER["openai"])
        with self.assertRaisesRegex(agent_runtime.RuntimeContractError, "unsupported model"):
            agent_runtime.ProviderConfig("openai", "gpt-5.6-sol", "offline-key", "low")
        with mock.patch.dict(agent_runtime.MODELS_BY_PROVIDER):
            cost.admit_evaluation_models(agent_runtime.MODELS_BY_PROVIDER)
            self.assertEqual(agent_runtime.ProviderConfig("openai", "gpt-5.6-sol", "offline-key", "low").effort, "low")
            self.assertIn("gpt-6.1-sol", agent_runtime.MODELS_BY_PROVIDER["openai"])
        self.assertNotIn("gpt-5.6-sol", agent_runtime.MODELS_BY_PROVIDER["openai"])

    def test_a_call_bound_prices_every_input_token_at_the_dearest_input_rate(self):
        self.assertAlmostEqual(cost.call_bound("claude-sonnet-5-5", 1_000, 100), (1_000 * 2.5 + 100 * 10) * 1e-6)
        self.assertAlmostEqual(cost.call_bound("gpt-6-luna", 1_000, 100), (1_000 * 0.125 + 100 * 0.5) * 1e-6)

    def test_an_explicit_catalog_is_read(self):
        catalog = {
            "providers": [
                {
                    "id": "openai",
                    "models": [{"id": "m", "input_usd_per_million_cents": 100, "output_usd_per_million_cents": 200}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "catalog.json")
            path.write_text(json.dumps(catalog), encoding="utf-8")
            self.assertAlmostEqual(cost.cost(cost.Usage(input_tokens=10), "m", path).usd, 10e-6)


class BudgetTests(unittest.TestCase):
    def test_a_reservation_that_would_cross_the_cap_dispatches_nothing(self):
        budget = cost.Budget(1.0)
        first = budget.reserve(0.6)
        with self.assertRaises(cost.BudgetExhaustedError):
            budget.reserve(0.5)
        budget.settle(first, cost.Cost(0.2))
        second = budget.reserve(0.8)
        budget.settle(second, cost.Cost(0.1))
        self.assertEqual(
            budget.summary(),
            {
                "cap_usd": 1.0,
                "spent_usd": 0.3,
                "unknown_settlements": 0,
                "reservations_exceeded": 0,
                "contention_waits": 0,
            },
        )

    def test_unknown_cost_keeps_the_reservation_and_overruns_are_counted(self):
        budget = cost.Budget(1.0)
        budget.settle(budget.reserve(0.4), cost.Cost(0.1, known=False))
        budget.settle(budget.reserve(0.1), cost.Cost(0.2, known=False))
        budget.settle(budget.reserve(0.1), cost.Cost(0.15))
        summary = budget.summary()
        self.assertAlmostEqual(summary["spent_usd"], 0.75)
        self.assertEqual((summary["unknown_settlements"], summary["reservations_exceeded"]), (2, 2))

    def test_contention_waits_for_in_flight_work_while_exhaustion_refuses(self):
        budget = cost.Budget(1.2)
        held = [budget.reserve(0.5, wait=True), budget.reserve(0.5, wait=True)]
        done = []

        def late() -> None:
            done.append(budget.reserve(0.5, wait=True))

        workers = [threading.Thread(target=late) for _ in range(2)]
        for worker in workers:
            worker.start()
        for reservation in held:
            budget.settle(reservation, cost.Cost(0.05))
        for worker in workers:
            worker.join(timeout=5)
        self.assertEqual(len(done), 2)
        self.assertGreaterEqual(budget.summary()["contention_waits"], 1)
        with self.assertRaises(cost.BudgetExhaustedError):
            budget.reserve(1.2, wait=True)

    def test_invalid_caps_and_reservations_are_refused(self):
        for cap in (-1.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "cap"):
                cost.Budget(cap)
        with self.assertRaisesRegex(ValueError, "reservation"):
            cost.Budget(1.0).reserve(-0.1)


class SharedBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name, "budget.json")
        self.budget = cost.SharedBudget.create(self.path, 1.0)

    def tearDown(self):
        self.directory.cleanup()

    def test_one_ledger_holds_every_holder_to_the_cap_and_settles_each_reservation_once(self):
        other = cost.SharedBudget(self.path)
        first = self.budget.reserve(0.6)
        with self.assertRaises(cost.BudgetExhaustedError):
            other.reserve(0.5)
        other.settle(first, cost.Cost(0.2))
        with self.assertRaisesRegex(cost.LedgerError, "no open reservation"):
            self.budget.settle(first, cost.Cost(0.2))
        unknown = other.reserve(0.3)
        self.budget.settle(unknown, cost.Cost(0.1, known=False))
        over = other.reserve(0.1)
        other.settle(over, cost.Cost(0.15))
        held = self.budget.reserve(0.2)
        summary = other.summary()
        self.assertAlmostEqual(summary.pop("spent_usd"), 0.65)
        self.assertEqual(
            summary,
            {
                "cap_usd": 1.0,
                "unknown_settlements": 1,
                "reservations_exceeded": 1,
                "contention_waits": 0,
                "unsettled_usd": held.amount,
            },
        )
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            cost.SharedBudget.create(self.path, 2.0)

    def test_a_reservation_waits_for_in_flight_work_and_refuses_what_can_never_fit(self):
        held = [self.budget.reserve(0.9, wait=True)]
        slept = []

        def in_flight_work_settles(seconds: float) -> None:
            slept.append(seconds)
            if held:
                cost.SharedBudget(self.path).settle(held.pop(), cost.Cost(0.3))

        with mock.patch.object(cost.time, "sleep", in_flight_work_settles):
            reservation = self.budget.reserve(0.5, wait=True)
        self.assertEqual((reservation.amount, len(slept)), (0.5, 1))
        self.assertTrue(cost.LEDGER_POLL_SECONDS[0] <= slept[0] <= cost.LEDGER_POLL_SECONDS[1])
        self.assertEqual(self.budget.summary()["contention_waits"], 1)
        with self.assertRaises(cost.BudgetExhaustedError):
            self.budget.reserve(0.8, wait=True)

    def test_an_unreadable_or_malformed_ledger_refuses_every_reservation(self):
        valid = json.loads(self.path.read_text(encoding="utf-8"))
        broken = (
            "not json",
            json.dumps([]),
            json.dumps({**valid, "extra": 1}),
            json.dumps({**valid, "cap": -1}),
            json.dumps({**valid, "spent": float("inf")}),
            json.dumps({**valid, "spent": "0"}),
            json.dumps({**valid, "unknown": True}),
            json.dumps({**valid, "waits": -1}),
            json.dumps({**valid, "open": []}),
            json.dumps({**valid, "open": {"a": -0.1}}),
        )
        for text in broken:
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(cost.LedgerError):
                    self.budget.reserve(0.1)
        self.path.unlink()
        with self.assertRaisesRegex(cost.LedgerError, "unreadable"):
            self.budget.summary()
        self.assertTrue(issubclass(cost.LedgerError, cost.BudgetExhaustedError))

    def test_invalid_caps_and_reservations_are_refused(self):
        for cap in (-1.0, float("nan"), True):
            with self.assertRaisesRegex(ValueError, "cap"):
                cost.SharedBudget.create(Path(self.directory.name, "other.json"), cap)
        with self.assertRaisesRegex(ValueError, "reservation"):
            self.budget.reserve(-0.1)


class PerTaskTests(unittest.TestCase):
    def test_failed_attempts_count_toward_the_cost_of_each_success(self):
        summary = cost.per_task([cost.Cost(0.2), cost.Cost(0.4), cost.Cost(0.6, known=False)], successes=2)
        self.assertEqual(summary["usd_per_attempted_task"], 0.4)
        self.assertEqual(summary["usd_per_successful_task"], 0.6)
        self.assertEqual((summary["usd_known"], summary["unknown_usage_attempts"]), (False, 1))
        empty = cost.per_task([], successes=0)
        self.assertEqual((empty["usd_per_attempted_task"], empty["usd_per_successful_task"]), (None, None))


if __name__ == "__main__":
    unittest.main()

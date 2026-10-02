"""Provider-free checks for the evaluation statistics (ADR-0094)."""

from __future__ import annotations

import unittest

from eval import stats


class StatsTests(unittest.TestCase):
    def test_arm_order_is_reproducible_per_key_and_varies_across_keys(self):
        arms = ("baseline", "candidate")
        first = stats.arm_order("seed", "openai", "case.en/0", arms)
        self.assertEqual(first, stats.arm_order("seed", "openai", "case.en/0", arms))
        self.assertEqual(sorted(first), sorted(arms))
        orders = {
            stats.arm_order("seed", provider, f"case/{index}", arms)
            for provider in ("openai", "anthropic")
            for index in range(20)
        }
        self.assertEqual(orders, {arms, arms[::-1]})

    def test_wilson_interval_bounds_a_proportion(self):
        self.assertIsNone(stats.wilson(0, 0))
        low, high = stats.wilson(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)
        self.assertAlmostEqual(stats.wilson(10, 10)[1], 1.0)
        self.assertAlmostEqual(stats.wilson(0, 10)[0], 0.0)
        with self.assertRaisesRegex(ValueError, "proportion"):
            stats.wilson(3, 2)

    def test_pass_hat_k_requires_every_repetition_to_succeed(self):
        self.assertEqual(stats.pass_hat_k(3, 3, 3), 1.0)
        self.assertEqual(stats.pass_hat_k(2, 3, 3), 0.0)
        self.assertAlmostEqual(stats.pass_hat_k(5, 6, 5), 1 / 6)
        self.assertAlmostEqual(stats.pass_hat_k(2, 3, 1), 2 / 3)
        self.assertIsNone(stats.pass_hat_k(2, 3, 5))
        for invalid in ((1, 3, 0), (4, 3, 1), (-1, 3, 1)):
            with self.assertRaisesRegex(ValueError, "pass"):
                stats.pass_hat_k(*invalid)

    def test_nearest_rank_percentile(self):
        self.assertIsNone(stats.percentile([], 0.5))
        self.assertEqual(stats.percentile([3.0, 1.0, 2.0, 4.0], 0.5), 2.0)
        self.assertEqual(stats.percentile([3.0, 1.0, 2.0, 4.0], 0.95), 4.0)
        self.assertEqual(stats.percentile([7.0], 0.01), 7.0)
        with self.assertRaisesRegex(ValueError, "percentile"):
            stats.percentile([1.0], 0)

    def test_cluster_bootstrap_resamples_whole_clusters(self):
        values = {"a": [1.0, 1.0], "b": [0.0, 0.0], "c": [1.0, 0.0], "empty": []}
        summary = stats.cluster_bootstrap(values, "seed", resamples=500)
        self.assertEqual((summary["mean"], summary["clusters"], summary["items"]), (0.5, 3, 6))
        self.assertLessEqual(summary["low"], 0.5)
        self.assertGreaterEqual(summary["high"], 0.5)
        self.assertEqual(summary, stats.cluster_bootstrap(values, "seed", resamples=500))
        identical = stats.cluster_bootstrap({"a": [1.0], "b": [1.0]}, "seed", resamples=50)
        self.assertEqual((identical["low"], identical["high"]), (1.0, 1.0))

    def test_one_cluster_or_none_has_no_interval(self):
        self.assertEqual(
            stats.cluster_bootstrap({"a": [1.0, 0.0]}, "seed"),
            {"mean": 0.5, "low": None, "high": None, "clusters": 1, "items": 2},
        )
        self.assertIsNone(stats.cluster_bootstrap({}, "seed")["mean"])

    def test_paired_difference_is_candidate_minus_baseline(self):
        pairs = {"a": [(1.0, 1.0), (0.0, 1.0)], "b": [(1.0, 0.0)], "c": [(0.0, 0.0)]}
        summary = stats.paired_difference(pairs, "seed", resamples=200)
        self.assertEqual((summary["mean"], summary["items"]), (0.0, 4))
        self.assertLess(summary["low"], 0.0)
        self.assertGreater(summary["high"], 0.0)


if __name__ == "__main__":
    unittest.main()

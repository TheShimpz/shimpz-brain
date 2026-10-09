"""Evaluation statistics: randomized arm order, pass^k, Wilson and cluster bootstrap intervals (ADR-0094).

Repeated runs of one scenario, and the languages of one task template, are not independent evidence: intervals
resample whole clusters. A paired comparison uses only complete pairs, and an interval that cannot be computed is
None, never a guess. This module uses only the standard library so that the umbrella journey driver can load it.
"""

import hashlib
import math
import random
from collections.abc import Mapping, Sequence

Z95 = 1.959963984540054
RESAMPLES = 2000


def arm_order(seed: str, provider: str, key: str, arms: Sequence[str]) -> tuple[str, ...]:
    """A reproducible random order of ``arms`` for one provider and one scenario repetition."""
    order = list(arms)
    random.Random(hashlib.sha256(f"{seed}\0{provider}\0{key}".encode()).digest()).shuffle(order)
    return tuple(order)


def wilson(successes: int, trials: int) -> tuple[float, float] | None:
    """The Wilson 95% score interval of a proportion, or None without a trial."""
    if not 0 <= successes <= trials:
        raise ValueError("invalid proportion")
    if trials == 0:
        return None
    rate = successes / trials
    scale = Z95**2 / trials
    centre = (rate + scale / 2) / (1 + scale)
    half = Z95 * math.sqrt(rate * (1 - rate) / trials + scale / (4 * trials)) / (1 + scale)
    return max(0.0, centre - half), min(1.0, centre + half)


def pass_hat_k(successes: int, trials: int, k: int) -> float | None:
    """The unbiased estimate that k independent repetitions all succeed, or None with fewer than k trials."""
    if k < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid pass^k input")
    if trials < k:
        return None
    return math.comb(successes, k) / math.comb(trials, k)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """The nearest-rank percentile, or None without a value."""
    if not 0 < fraction <= 1:
        raise ValueError("invalid percentile")
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def cluster_bootstrap(
    values: Mapping[str, Sequence[float]], seed: str, resamples: int = RESAMPLES
) -> dict[str, object]:
    """The item mean with a 95% percentile interval that resamples whole clusters with replacement.

    With fewer than two clusters the interval is None: one cluster says nothing about cluster variance.
    """
    clusters = sorted(name for name, items in values.items() if items)
    items = [value for name in clusters for value in values[name]]
    summary: dict[str, object] = {
        "mean": sum(items) / len(items) if items else None,
        "low": None,
        "high": None,
        "clusters": len(clusters),
        "items": len(items),
    }
    if len(clusters) < 2:
        return summary
    generator = random.Random(hashlib.sha256(seed.encode()).digest())
    means = []
    for _ in range(resamples):
        drawn = [value for _ in clusters for value in values[generator.choice(clusters)]]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    summary["low"] = means[int(0.025 * resamples)]
    summary["high"] = means[min(resamples - 1, math.ceil(0.975 * resamples) - 1)]
    return summary


def paired_difference(
    pairs: Mapping[str, Sequence[tuple[float, float]]], seed: str, resamples: int = RESAMPLES
) -> dict[str, object]:
    """Mean B minus A over complete pairs, with a cluster bootstrap interval."""
    return cluster_bootstrap({name: [b - a for a, b in items] for name, items in pairs.items()}, seed, resamples)

"""What an engineering arm recorded beyond success, summed for the precision report (ADR-0094).

Exposure, escalation, and Jev signals of the engineering arms, the Luna-99 recovery mechanisms' extra work, and the
language-retrieval arms' counts, each over an explicit denominator. ``eval.precision`` adds them to every run that
recorded them. This module uses only the standard library.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def arm_signals(attempts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """What an engineering arm recorded, each rate over an explicit denominator.

    Exposure signals (recall, exposed Assistants) count attempts whose exposure was computed; escalation counts
    dispatched attempts. A stopped attempt never computed its exposure, so it is in neither denominator.
    """
    engineering = [attempt for attempt in attempts if "exposed" in attempt]
    if not engineering:
        return {}
    exposed = [attempt for attempt in engineering if attempt["recall"] is not None]
    dispatched = [attempt for attempt in engineering if int(attempt["operations"]) > 0]
    signals: dict[str, int] = defaultdict(int)
    for attempt in dispatched:
        if attempt["escalated"]:
            signals[str(attempt["escalation_signal"])] += 1
    return {
        "arm_signals": {
            "exposure_attempts": len(exposed),
            "dispatched_attempts": len(dispatched),
            "working_set_recall": share(sum(bool(a["recall"]) for a in exposed), len(exposed)),
            "mean_exposed_assistants": share(sum(len(a["exposed"]) for a in exposed), len(exposed)),
            # A selecting arm (L6) that found no confident group fell back to its declared exposure.
            "selection_fallbacks": sum(bool(a["selection_fallback"]) for a in exposed),
            "selection_fallback_rate": share(sum(bool(a["selection_fallback"]) for a in exposed), len(exposed)),
            "refusals": sum(int(a["refusals"]) for a in engineering),
            "escalated": sum(signals.values()),
            "escalation_rate": share(sum(signals.values()), len(dispatched)),
            "escalation_signals": dict(sorted(signals.items())),
            "escalation_blocked": sum(bool(a["escalation_blocked"]) for a in engineering),
            # Every Jev request is counted, its answer usable or not.
            "jev_calls": sum(int(a["jev_calls"]) for a in engineering),
            "jev_failures": sum(int(a["jev_failures"]) for a in engineering),
            "jev_usd": round(sum(float(a["jev_usd"]) for a in engineering), 6),
            "jev_usd_known": all(bool(a["jev_usd_known"]) for a in engineering),
            **recovery_signals(engineering),
        }
    }


# How each read Team made itself (arms R and H) ended: run, served from the attempt's cache, refused by schema
# admission, failed in the Assistant, refused by the per-attempt bound, or (a rewrite) refused by the scope fence.
TEAM_READ_OUTCOMES = ("invoked", "cached", "rejected", "failed", "limited", "fenced")


def recovery_signals(engineering: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The Luna-99 mechanisms' extra work, each counted on every engineering attempt (zero where an arm lacks it).

    Every dispatched helper request is counted, its answer usable or not, and a refused one apart; triggers (relax,
    rewrite candidates) are kept apart from the reads Team actually made; a K second pass reports the first pass's
    oracle too.
    """
    first = [a["first_pass"] for a in engineering if a.get("first_pass")]
    return {
        "relax_triggers": sum(int(a.get("relax_triggers", 0)) for a in engineering),
        "rewrite_candidates": sum(int(a.get("rewrite_candidates", 0)) for a in engineering),
        "recoveries": sum(int(a.get("recoveries", 0)) for a in engineering),
        "critic_revisions": sum(int(a.get("critic_revisions", 0)) for a in engineering),
        "replays_refused": sum(int(a.get("replays_refused", 0)) for a in engineering),
        "first_pass_oracle_passed": sum(bool(item["oracle"]["passed"]) for item in first),
        "team_reads": {
            outcome: sum(int((a.get("team_reads") or {}).get(outcome, 0)) for a in engineering)
            for outcome in TEAM_READ_OUTCOMES
        },
        "helper_calls": sum(int(a.get("helper_calls", 0)) for a in engineering),
        "helper_failures": sum(int(a.get("helper_failures", 0)) for a in engineering),
        "helper_refused": sum(int(a.get("helper_refused", 0)) for a in engineering),
        "helper_unusable": sum(int(a.get("helper_unusable", 0)) for a in engineering),
        "helper_usd": round(sum(float(a.get("helper_usd", 0.0)) for a in engineering), 6),
        "helper_usd_known": all(bool(a.get("helper_usd_known", True)) for a in engineering),
        "language": language_signals(engineering),
    }


# The language arms' per-attempt counts (eval.language), summed over the attempts that recorded them.
LANGUAGE_COUNTS = (
    "eligible",
    "observed",
    "planned_collection",
    "planned_assistant",
    "planned_none",
    "planned_assumed",
    "probes",
    "helper_calls",
    "helper_limited",
    "unplanned",
    "variant_reads",
    "expanded",
    "candidates",
    "rank_eligible",
    "embedding_calls",
    "ranked_calls",
    "ranked_candidates",
    "service_failures",
)
# The counts that mean a mechanism did work (a helper call, a Team read, or a ranking) and those that mean it returned
# something (candidates), kept apart so work without a recovery stays visible.
LANGUAGE_WORK = ("helper_calls", "variant_reads", "probes", "embedding_calls", "ranked_calls")
LANGUAGE_RECOVERY = ("candidates", "ranked_candidates")


def language_signals(engineering: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """The language arms' summed counts and service time, with how many attempts had work and how many candidates."""
    recorded = [a["language"] for a in engineering if a.get("language")]
    if not recorded:
        return None
    return {
        "attempts": len(recorded),
        "attempts_with_eligible_search": sum(bool(item["eligible"]) for item in recorded),
        "attempts_with_work": sum(any(item[name] for name in LANGUAGE_WORK) for item in recorded),
        "attempts_with_candidates": sum(any(item[name] for name in LANGUAGE_RECOVERY) for item in recorded),
        **{name: sum(int(item[name]) for item in recorded) for name in LANGUAGE_COUNTS},
        "service_seconds": round(sum(float(item["service_seconds"]) for item in recorded), 3),
    }


def share(count: float, denominator: int) -> float | None:
    return count / denominator if denominator else None

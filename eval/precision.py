"""Judge whole-task journey attempts and build the sanitized Precision Runtime report (ADR-0094).

The umbrella driver ``.tests/perf/precision_journeys.py`` runs ``eval.corpus`` scenarios through the Team
orchestrator and the shipped Brain runtime, and writes a private JSON Lines transcript, one attempt per line, outside
the repository with owner-only permissions. From Brain, ``PYTHONPATH=. uv run --frozen --python 3.14 python -m
eval.precision`` then:

- ``validate`` checks the corpus, Brain's admission of every scenario's Assistants, and the calibration sample;
- ``calibrate`` runs both judges over the adjudicated sample and writes their agreement;
- ``judge`` judges every completed attempt of a transcript under a hard US dollar cap (``--cap``);
- ``report`` writes the sanitized report: counts, rates with cluster bootstrap intervals, pass^k, paired
  differences, tokens, cost, latency, and judge statistics, never a prompt, reply, key, Action input, or result.

An attempt succeeds when its Team turn completed, the oracle passed, and the final judge verdict passes. A stopped or
unjudged attempt is inconclusive and never counted as a failure or a success.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from eval import corpus, judge, large_api, large_api_contract, private, split
from eval import cost as eval_cost
from eval import stats as eval_stats
from eval.contracts import ASSISTANTS_B
from eval.large_api_arms import ASSISTANTS_TASKS

SCENARIOS = {**corpus.SCENARIOS_BY_ID, **large_api.SCENARIOS_BY_ID}
CONTRACTS = {
    "a": corpus.ASSISTANTS,
    "b": ASSISTANTS_B,
    "large": large_api_contract.ASSISTANTS,
    "large-tasks": ASSISTANTS_TASKS,
}

REPORT_SCHEMA = "shimpz.precision-eval.report/v1"
STATUSES = ("completed", "turn-failed", "brain-error", "budget-stopped")
STRATA = ("language", "scope", "behavior", "multi_assistant", "min_rounds")
WORKERS = 8


def attempt_key(attempt: Mapping[str, object]) -> str:
    return "|".join(str(attempt[name]) for name in ("campaign", "provider", "model", "arm", "repetition", "scenario"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge_item(attempt: Mapping[str, object]) -> judge.Item:
    """The judge input; an experiment arm's exposed Assistants, Actions, and contract set replace the scenario's."""
    scenario = SCENARIOS[str(attempt["scenario"])]
    contracts = CONTRACTS[str(attempt.get("contracts", "a"))]
    only = set(attempt.get("exposed_actions") or ())
    exposed = tuple(
        dataclasses.replace(contracts[name], actions=tuple(a for a in contracts[name].actions if a.id in only))
        if only and name == large_api_contract.ASSISTANT_ID
        else contracts[name]
        for name in attempt.get("exposed") or scenario.assistants
    )
    return judge.Item(scenario, tuple(attempt["ledger"]), str(attempt["reply"]), exposed)


Judgment = Callable[[judge.Item], judge.Verdict]


def metered(model: str, verdict: Callable[[judge.Item], judge.Verdict], spend: list[eval_cost.Cost]) -> Judgment:
    """Record what each judge call reported; the installed ``eval.ceiling`` bounds and reserves the request itself."""
    import model_usage

    def run(item: judge.Item) -> judge.Verdict:
        result, counts = model_usage.measure(lambda: verdict(item))
        spend.append(eval_cost.cost(eval_cost.Usage.of(counts), model))
        return result

    return run


def judge_attempts(
    attempts: Iterable[Mapping[str, object]], primary: Judgment, tiebreak: Judgment, workers: int = WORKERS
) -> list[dict[str, object]]:
    """Judge every completed attempt; an attempt whose judgment failed or hit the cap stays unjudged."""

    def one(attempt: Mapping[str, object]) -> dict[str, object] | None:
        try:
            decision = judge.decide(primary, tiebreak, judge_item(attempt), bool(attempt["oracle"]["passed"]))
        except judge.JudgeError, eval_cost.BudgetExhaustedError:
            return None
        return {
            "key": attempt_key(attempt),
            "primary": decision.primary.model_dump(),
            "tiebreak": None if decision.tiebreak is None else decision.tiebreak.model_dump(),
            "final": decision.final.model_dump(),
            "judge_identity": judge.identity(),
        }

    completed = [attempt for attempt in attempts if attempt["status"] == "completed"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return [result for result in pool.map(one, completed) if result is not None]


def calibrate(primary: Judgment, tiebreak: Judgment) -> dict[str, object]:
    """Agreement of both judges with owner labels when the held-out sample carries them, else with the author's.

    Owner labels count only when the owner collected them blind to judge verdicts, against the very judge now frozen.
    """
    current = judge.identity()
    held = judge.heldout_items()
    collection = judge.heldout_collection()
    owner = (
        all(expected is not None for _id, _item, expected in held)
        and collection.get("blind_to_judge_verdicts") is True
        and collection.get("frozen_judge_identity") == current
    )
    items = held if owner else judge.calibration_items()
    summary: dict[str, object] = {
        "adjudication": "owner" if owner else "author",
        "blind": owner,
        "judge_identity": current,
    }
    for name, verdict in (("primary", primary), ("tiebreak", tiebreak)):
        results = []
        for _item_id, item, expected in items:
            try:
                results.append((verdict(item), expected))
            except judge.JudgeError, eval_cost.BudgetExhaustedError:
                results.append((None, expected))
        summary[name] = judge.agreement(results)
    return summary


def outcome(attempt: Mapping[str, object], decision: Mapping[str, object] | None) -> str:
    if attempt["status"] == "budget-stopped":
        return "inconclusive"
    if attempt["status"] != "completed" or not attempt["oracle"]["passed"]:
        return "failure"
    if decision is None:
        return "inconclusive"
    final = judge.Verdict(**decision["final"])
    return "success" if judge.succeeded(final, judge_item(attempt)) else "failure"


def _template(attempt: Mapping[str, object]) -> str:
    return SCENARIOS[str(attempt["scenario"])].template.id


def _rate(rows: Sequence[tuple[Mapping[str, object], str]], seed: str, name: str) -> dict[str, object]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for attempt, result in rows:
        if result != "inconclusive":
            clusters[_template(attempt)].append(float(result == "success"))
    return eval_stats.cluster_bootstrap(clusters, f"{seed}:{name}")


def _pass_k(rows: Sequence[tuple[Mapping[str, object], str]], seed: str, name: str) -> dict[str, object]:
    by_scenario: dict[str, list[bool]] = defaultdict(list)
    for attempt, result in rows:
        if result != "inconclusive":
            by_scenario[str(attempt["scenario"])].append(result == "success")
    summary: dict[str, object] = {}
    for k in (1, 3, 5):
        clusters: dict[str, list[float]] = defaultdict(list)
        for scenario, results in by_scenario.items():
            estimate = eval_stats.pass_hat_k(sum(results), len(results), k)
            if estimate is not None:
                clusters[SCENARIOS[scenario].template.id].append(estimate)
        summary[f"pass^{k}"] = eval_stats.cluster_bootstrap(clusters, f"{seed}:{name}:pass{k}")
    return summary


def _usage(attempts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    usage = sum((eval_cost.Usage.of(attempt["usage"]) for attempt in attempts), eval_cost.Usage())
    return {
        **usage.to_dict(),
        "fresh_input_tokens": usage.fresh_input_tokens,
        "unknown_usage_attempts": sum(not attempt["usage_known"] for attempt in attempts),
        "brain_operations": sum(int(attempt["operations"]) for attempt in attempts),
    }


def _group(rows: Sequence[tuple[Mapping[str, object], str]], seed: str, name: str) -> dict[str, object]:
    attempts = [attempt for attempt, _ in rows]
    conclusive = [(attempt, result) for attempt, result in rows if result != "inconclusive"]
    # Every attempt's oracle counts its writes, a failed turn's included: a write before the failure still happened.
    oracle = [attempt["oracle"] for attempt in attempts]
    active = [float(attempt["seconds_active"]) for attempt, _ in conclusive]
    # Every dispatched attempt was paid for, whether or not it can be scored.
    dispatched = [attempt for attempt in attempts if int(attempt["operations"]) > 0]
    costs = [eval_cost.Cost(float(attempt["usd"]), bool(attempt["usage_known"])) for attempt in dispatched]
    successes = sum(result == "success" for _, result in conclusive)
    strata: dict[str, dict[str, object]] = {}
    for stratum in STRATA:
        values: dict[str, list[tuple[Mapping[str, object], str]]] = defaultdict(list)
        for attempt, result in conclusive:
            values[str(SCENARIOS[str(attempt["scenario"])].strata[stratum])].append((attempt, result))
        strata[stratum] = {
            value: _rate(items, seed, f"{name}:{stratum}:{value}") for value, items in sorted(values.items())
        }
    return {
        "attempts": {status: sum(attempt["status"] == status for attempt in attempts) for status in STATUSES},
        "conclusive": len(conclusive),
        "successes": successes,
        "inconclusive": len(rows) - len(conclusive),
        "success": _rate(rows, seed, name),
        **_pass_k(rows, seed, name),
        "oracle": {
            "passed_completed": sum(a["oracle"]["passed"] for a in attempts if a["status"] == "completed"),
            **{
                name: sum(item[name] for item in oracle)
                for name in ("missing", "wrong", "forbidden", "wrong_scope", "duplicates")
            },
        },
        "clarifications": sum(bool(attempt["clarification"]) for attempt in attempts),
        "mean_rounds": sum(int(attempt["rounds"]) for attempt in attempts) / len(attempts) if attempts else None,
        "usage": _usage(attempts),
        "cost": {
            **eval_cost.per_task(costs, successes),
            "inconclusive_usd": round(
                sum(float(attempt["usd"]) for attempt, result in rows if result == "inconclusive"), 6
            ),
        },
        "latency_seconds": {
            "active_p50": eval_stats.percentile(active, 0.5),
            "active_p95": eval_stats.percentile(active, 0.95),
            "wall_p95": eval_stats.percentile([float(a["seconds_wall"]) for a, _ in conclusive], 0.95),
        },
        "strata": strata,
        **_arm_signals(attempts),
    }


def _arm_signals(attempts: Sequence[Mapping[str, object]]) -> dict[str, object]:
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
            "working_set_recall": _share(sum(bool(a["recall"]) for a in exposed), len(exposed)),
            "mean_exposed_assistants": _share(sum(len(a["exposed"]) for a in exposed), len(exposed)),
            # A selecting arm (L6) that found no confident group fell back to its declared exposure.
            "selection_fallbacks": sum(bool(a["selection_fallback"]) for a in exposed),
            "selection_fallback_rate": _share(sum(bool(a["selection_fallback"]) for a in exposed), len(exposed)),
            "refusals": sum(int(a["refusals"]) for a in engineering),
            "escalated": sum(signals.values()),
            "escalation_rate": _share(sum(signals.values()), len(dispatched)),
            "escalation_signals": dict(sorted(signals.items())),
            "escalation_blocked": sum(bool(a["escalation_blocked"]) for a in engineering),
            # Every Jev request is counted, its answer usable or not.
            "jev_calls": sum(int(a["jev_calls"]) for a in engineering),
            "jev_failures": sum(int(a["jev_failures"]) for a in engineering),
            "jev_usd": round(sum(float(a["jev_usd"]) for a in engineering), 6),
            "jev_usd_known": all(bool(a["jev_usd_known"]) for a in engineering),
        }
    }


def _share(count: float, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _paired(
    rows: Sequence[tuple[Mapping[str, object], str]], baseline: str, candidate: str, seed: str
) -> dict[str, object]:
    """Candidate minus baseline success over complete (scenario, repetition) pairs of one campaign.

    Pairs come only from the repetitions both arms ran: a reference arm that ran fewer repetitions is compared over
    those. Within them, a missing or inconclusive side makes the pair incomplete; the caller has already refused a
    repeated pairing key.
    """
    ran: dict[str, set[int]] = defaultdict(set)
    pairs: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for attempt, result in rows:
        if attempt["arm"] in {baseline, candidate}:
            ran[str(attempt["arm"])].add(int(attempt["repetition"]))
            pairs[str(attempt["scenario"]), int(attempt["repetition"])][str(attempt["arm"])] = result
    common = ran[baseline] & ran[candidate]
    complete: dict[str, list[tuple[float, float]]] = defaultdict(list)
    incomplete = 0
    for (scenario, repetition), results in sorted(pairs.items()):
        if repetition not in common:
            continue
        if {results.get(baseline), results.get(candidate)} & {None, "inconclusive"}:
            incomplete += 1
            continue
        complete[SCENARIOS[scenario].template.id].append(
            (float(results[baseline] == "success"), float(results[candidate] == "success"))
        )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "repetitions": sorted(common),
        "complete_pairs": sum(len(items) for items in complete.values()),
        "incomplete_pairs": incomplete,
        "difference": eval_stats.paired_difference(complete, f"{seed}:paired:{candidate}"),
    }


# The closed vocabulary of campaign metadata and calibration summaries a report may carry (ADR-0094): every string is
# one token from IDENT_RE (no spaces, so no prose), and some fields take only an enumerated value or a number.
NUMBER_FIELDS = frozenset(
    {
        "agree",
        "calibration_usd",
        "contention_waits",
        "cap_usd",
        "failed",
        "held_out_repetitions",
        "items",
        "max_output_tokens",
        "max_reservation_usd",
        "rate",
        "reference_repetitions",
        "refused",
        "repetitions",
        "requests",
        "reservations_exceeded",
        "scenarios",
        "sdk_retries",
        "seconds",
        "spent_usd",
        "superseded_usd",
        "total",
        "total_estimated_spend_usd",
        "tuning_repetitions",
        "unknown_settlements",
        "unreported",
        "unsupported",
        "workers",
    }
)
BOOLEAN_FIELDS = frozenset({"admitted", "blind", "identical_arms", "stopped_by_cap"})
ENUM_FIELDS = {
    "adjudication": frozenset({"owner", "author"}),
    "effort": frozenset({"low", "medium", "high"}),
    "label": frozenset({"exploratory"}),
    "provider": frozenset({"openai", "anthropic"}),
}
CONTAINER_FIELDS = frozenset(
    {
        "anthropic",
        "all_criteria",
        "arms",
        "asks_for_missing_information",
        "budget",
        "budget_plan_usd",
        "campaigns",
        "canary",
        "commits",
        "comparisons",
        "corpus",
        "efforts",
        "final_judging",
        "jev_budget",
        "judge_budget",
        "judge_spend",
        "judges",
        "language_matches",
        "namespaces_budget",
        "openai_paired",
        "primary",
        "reply_correct",
        "tiebreak",
        "unsupported_claim",
        "wilson95",
    }
)
TOKEN_FIELDS = frozenset(
    {
        "baseline_arm",
        "brain",
        "campaign",
        "date",
        "digest",
        "id",
        "judge_identity",
        "judge_version",
        "kind",
        "model",
        "path",
        "scenario_patterns",
        "seed",
        "stratum",
        "team",
        "umbrella",
    }
)
META_KEYS = NUMBER_FIELDS | BOOLEAN_FIELDS | frozenset(ENUM_FIELDS) | CONTAINER_FIELDS | TOKEN_FIELDS
ARM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,23}\Z")
IDENT_RE = re.compile(r"[A-Za-z0-9*][A-Za-z0-9._:/*,+=@-]{0,79}\Z")
FINGERPRINT_RE = re.compile(r"(sha256:[0-9a-f]{64}|[0-9a-f]{40})\Z")
# A provider key prefix at a word start; "task-create" is not one.
CREDENTIAL_RE = re.compile(r"(?<![A-Za-z0-9])sk-")
MAX_META_DEPTH = 5
CAMPAIGN_RE = re.compile(r"[a-z0-9][a-z0-9.-]{0,63}\Z")


def _token(value: object) -> bool:
    return isinstance(value, str) and (
        FINGERPRINT_RE.fullmatch(value) is not None
        or (IDENT_RE.fullmatch(value) is not None and len(value) <= 40 and CREDENTIAL_RE.search(value) is None)
    )


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def checked_meta(value: object, depth: int = 0, parent: str = "") -> object:
    """Return metadata only when every field fits its closed type: numbers, booleans, enums, or single tokens.

    No field may hold prose: a string is one token without spaces, at most 40 characters unless it is a digest or a
    commit, and never shaped like a provider key. A reply, a sentence, or a credential is refused.
    """
    if depth > MAX_META_DEPTH:
        raise ValueError("report metadata nests too deeply")
    if isinstance(value, dict):
        for key in value:
            arm_key = parent == "efforts" and isinstance(key, str) and ARM_RE.fullmatch(key) is not None
            if not isinstance(key, str) or not (key in META_KEYS or arm_key):
                raise ValueError("report metadata has an unknown field")
        return {key: _field(key if key in META_KEYS else "arm", item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [checked_meta(item, depth + 1, parent) for item in value]
    return _field(parent, value, depth)


def _field(key: str, value: object, depth: int) -> object:
    """Dispatch on the field's declared type first: only a container field may hold a mapping or a nested list."""
    if value is None:
        return None
    if key in CONTAINER_FIELDS and isinstance(value, dict | list):
        return checked_meta(value, depth, key)
    if key in TOKEN_FIELDS and isinstance(value, list):
        # A token field may list tokens (scenario patterns), never containers.
        return [_scalar(key, item) for item in value]
    return _scalar(key, value)


def _scalar(key: str, value: object) -> object:
    if isinstance(value, dict | list):
        raise ValueError("report metadata has an unsafe value")
    if key in NUMBER_FIELDS:
        valid = _number(value)
    elif key in BOOLEAN_FIELDS:
        valid = isinstance(value, bool)
    elif key in ENUM_FIELDS:
        valid = value in ENUM_FIELDS[key]
    elif key in CONTAINER_FIELDS:
        # A container's leaves (an arm list, an interval, a budget plan line) are numbers or tokens.
        valid = _number(value) or _token(value)
    else:
        valid = _token(value)
    if not valid:
        raise ValueError("report metadata has an unsafe value")
    return value


def checked_identity(campaign: str, provider: str, model: str, arm: str, efforts: Sequence[str]) -> None:
    """A run's exported identity: a campaign token, a catalog provider and model, an arm label, and known efforts.

    No identity part may be shaped like a provider key, whatever its pattern otherwise admits.
    """
    if (
        CAMPAIGN_RE.fullmatch(campaign) is None
        or provider not in ENUM_FIELDS["provider"]
        or ARM_RE.fullmatch(arm) is None
        or not set(efforts) <= ENUM_FIELDS["effort"]
        or any(CREDENTIAL_RE.search(item) for item in (campaign, model, arm))
    ):
        raise ValueError("a run has an invalid identity")
    if eval_cost.price(model).provider != provider:
        raise ValueError("a run names a model of another provider")


def grade(
    calibration: Mapping[str, object] | None, judged: Sequence[Mapping[str, object]], meta: Mapping[str, object]
) -> dict[str, object]:
    """Decision grade needs an admitted, blind, owner-labelled calibration of the judge behind every verdict.

    Anything less is exploratory, with closed reason codes.
    """
    current = judge.identity()
    reasons = []
    if calibration is None:
        reasons.append("no-calibration")
    else:
        if calibration.get("adjudication") != "owner" or calibration.get("blind") is not True:
            reasons.append("no-blind-owner-labels")
        if calibration.get("judge_identity") != current:
            reasons.append("calibration-of-another-judge")
        reasons.extend(
            f"{name}-not-admitted"
            for name in ("primary", "tiebreak")
            if not (calibration.get(name) or {}).get("admitted")
        )
    if any(item.get("judge_identity") != current for item in judged):
        reasons.append("verdict-of-another-judge")
    if meta.get("label") == "exploratory":
        reasons.append("labelled-exploratory")
    return {"grade": "exploratory" if reasons else "decision", "reasons": reasons, "judge_identity": current}


def regrade(report: Mapping[str, object]) -> dict[str, object]:
    """Recompute a committed report's grade under the current rules from what it records.

    A report promotes only on the provenance it records for its verdicts: the judge identity behind each one. A report
    without that provenance stays exploratory, so regrading never forgets that its verdicts came from another judge.
    """
    judges = report.get("judges") or {}
    identities = judges.get("verdict_identities")
    # An empty list is provenance only for a report that records exactly zero judged attempts.
    nothing_judged = type(judges.get("judged_attempts")) is int and judges["judged_attempts"] == 0
    recorded = (
        isinstance(identities, list)
        and (bool(identities) or nothing_judged)
        and all(isinstance(item, str) and FINGERPRINT_RE.fullmatch(item) for item in identities)
    )
    judged = [{"judge_identity": item} for item in identities] if recorded else []
    decision = grade(report.get("judge_calibration"), judged, report.get("meta") or {})
    if not recorded:
        decision = {**decision, "grade": "exploratory", "reasons": [*decision["reasons"], "no-verdict-provenance"]}
    return {**report, "decision": decision}


def _part(scenario_id: str, sets: Mapping[str, dict[str, list[str]]]) -> str:
    stratum = "large-api" if scenario_id in large_api.SCENARIOS_BY_ID else "precision"
    return split.part(scenario_id, sets[stratum])


def build_report(
    attempts: Sequence[Mapping[str, object]],
    judged: Sequence[Mapping[str, object]],
    meta: Mapping[str, object],
    calibration: Mapping[str, object] | None = None,
    part: str | None = None,
) -> dict[str, object]:
    """Group attempts by campaign, provider, model, and arm; pair arms only within one campaign, across models too.

    Every attempt key (campaign, provider, model, arm, repetition, scenario) must be unique, so two campaigns are
    never merged into one pair, and an arm runs under one model per campaign. Each arm is paired against the
    ``baseline_arm`` the metadata names, or the first arm, unless the metadata lists ``comparisons``.
    """
    sets = None if part is None else {"precision": split.load(), "large-api": split.load(split.LARGE_API_SPLIT)}
    if part is not None:
        attempts = [attempt for attempt in attempts if _part(str(attempt["scenario"]), sets) == part]
    keys = [attempt_key(attempt) for attempt in attempts]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate attempt or pairing key")
    decisions = {str(item["key"]): item for item in judged}
    seed = str(meta.get("seed", "precision"))
    groups: dict[tuple[str, str, str, str], list[tuple[Mapping[str, object], str]]] = defaultdict(list)
    for attempt in attempts:
        result = outcome(attempt, decisions.get(attempt_key(attempt)))
        key = (str(attempt["campaign"]), str(attempt["provider"]), str(attempt["model"]), str(attempt["arm"]))
        groups[key].append((attempt, result))
    runs = []
    by_campaign: dict[str, dict[str, list]] = defaultdict(dict)
    models: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for (campaign, provider, model, arm), rows in sorted(groups.items()):
        effort = {str(attempt["effort"]) for attempt, _ in rows}
        name = f"{campaign}:{provider}:{model}:{arm}"
        checked_identity(campaign, provider, model, arm, sorted(effort))
        identity = {"campaign": campaign, "provider": provider, "model": model, "arm": arm, "effort": sorted(effort)}
        runs.append({**identity, **_group(rows, seed, name)})
        if arm in by_campaign[campaign]:
            raise ValueError("an arm runs under two models in one campaign")
        by_campaign[campaign][arm] = rows
        models[campaign][arm] = (provider, model)
    paired = []
    pooled = []
    for campaign, arms in sorted(by_campaign.items()):
        if len(arms) < 2:
            continue
        rows = [row for items in arms.values() for row in items]
        baseline = meta.get("baseline_arm") if meta.get("baseline_arm") in arms else sorted(arms)[0]
        comparisons = [
            (base, candidate) for base, candidate in meta.get("comparisons", ()) if {base, candidate} <= set(arms)
        ] or [(baseline, candidate) for candidate in sorted(arms) if candidate != baseline]
        # Arms of one campaign pair across models too (A against its Sonnet reference S); each side names its model.
        paired.extend(
            {
                "campaign": campaign,
                "baseline_provider": models[campaign][base][0],
                "baseline_model": models[campaign][base][1],
                "candidate_provider": models[campaign][candidate][0],
                "candidate_model": models[campaign][candidate][1],
                **_paired(rows, base, candidate, seed),
            }
            for base, candidate in comparisons
        )
        if meta.get("identical_arms") and len(set(models[campaign].values())) == 1:
            provider, model = next(iter(models[campaign].values()))
            identity = {"campaign": campaign, "provider": provider, "model": model}
            pooled.append({**identity, **_pass_k(rows, seed, f"{campaign}:{provider}:{model}:pooled")})
    tiebreaks = [item for item in judged if item["tiebreak"] is not None]
    return {
        "schema": REPORT_SCHEMA,
        "corpus": {"id": corpus.CORPUS_ID, "digest": corpus.digest(), "scenarios": len(corpus.SCENARIOS)},
        "meta": checked_meta(dict(meta)),
        "split": None
        if part is None
        else {
            "part": part,
            "templates": sorted(t for sets_of in sets.values() for t in sets_of[part]),
            "digests": [
                split.digest(sets["precision"]),
                split.digest(sets["large-api"], large_api.CORPUS_ID, large_api.digest()),
            ],
        },
        "decision": grade(calibration, judged, meta),
        "judge_calibration": None if calibration is None else checked_meta(dict(calibration)),
        "runs": runs,
        "paired": paired,
        "pooled_identical_arms": pooled,
        "judges": {
            "judged_attempts": len(judged),
            "tiebreaks": len(tiebreaks),
            "tiebreak_changed_reply_correct": sum(
                item["tiebreak"]["reply_correct"] != item["primary"]["reply_correct"] for item in tiebreaks
            ),
            "final_unsupported_claims": sum(item["final"]["unsupported_claim"] for item in judged),
            "verdict_identities": sorted({str(item.get("judge_identity")) for item in judged}),
        },
    }


def validate() -> dict[str, object]:
    """Both strata are sound, and Brain admits every scenario under every contract set its arms use."""
    import agent_runtime

    corpus.validate()
    large_api.validate()
    provider = agent_runtime.ProviderConfig("openai", judge.JUDGE_MODELS["openai"], "offline-validation-key", "low")
    checks = [(scenario, set_name) for scenario in corpus.SCENARIOS for set_name in ("a", "b")]
    checks += [(scenario, set_name) for scenario in large_api.SCENARIOS for set_name in ("large", "large-tasks")]
    for scenario, set_name in checks:
        agent_runtime.TurnContext(
            "precision:validate",
            "Precision Team",
            brain_assistants(scenario, set_name),
            provider,
            memories=(),
            skills=(),
            routines=(),
        )
    return {
        "status": "corpus-valid",
        "corpora": {corpus.CORPUS_ID: corpus.digest(), large_api.CORPUS_ID: large_api.digest()},
        "scenarios": len(SCENARIOS),
        "calibration_items": len(judge.calibration_items()),
    }


def brain_assistants(scenario: corpus.Scenario, contracts: str = "a") -> tuple:
    import agent_runtime

    return tuple(
        agent_runtime.AssistantDefinition(
            assistant.id,
            assistant.genesis,
            tuple(agent_runtime.ActionDefinition(a.id, a.summary, a.input_schema) for a in assistant.actions),
        )
        for assistant in (CONTRACTS[contracts][name] for name in scenario.assistants)
    )


def _judges(args: argparse.Namespace, spend: list[eval_cost.Cost]) -> tuple[Judgment, Judgment]:
    from eval.intent_route import _key

    judgments = []
    for provider, key_file in (("openai", args.key_file), ("anthropic", args.tiebreak_key_file)):
        model = judge.judge_model(provider, _key(key_file))
        judgments.append(
            metered(
                judge.JUDGE_MODELS[provider],
                lambda item, model=model, provider=provider: judge.judge(model, provider, item),
                spend,
            )
        )
    return judgments[0], judgments[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("validate", "calibrate", "judge", "report", "regrade"))
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--judged", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--meta", type=Path, help="campaign metadata JSON the driver wrote")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--tiebreak-key-file", type=Path)
    parser.add_argument("--cap", type=float, default=0.0, help="hard US dollar cap for judge calls")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--split", choices=("held-out", "tuning"), help="report only this part of the frozen split")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(json.dumps(validate(), sort_keys=True))
            return 0
        if args.command == "regrade":
            regraded = regrade(json.loads(args.out.read_text(encoding="utf-8")))
            args.out.write_text(json.dumps(regraded, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            return 0
        if args.command == "report":
            meta = json.loads(args.meta.read_text(encoding="utf-8")) if args.meta else {}
            calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
            report = build_report(read_jsonl(args.transcript), read_jsonl(args.judged), meta, calibration, args.split)
            args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            return 0
        from eval.ceiling import Ceiling

        # The output is admitted, and held open, before any paid inference.
        descriptor = private.open_private(args.out)

        # Installed before any judge model exists: no SDK retries, clamped output, a reserved worst case per request.
        ceiling = Ceiling(args.cap, judge.MAX_OUTPUT_TOKENS)
        ceiling.install()
        budget = ceiling.budget
        try:
            spend: list[eval_cost.Cost] = []
            primary, tiebreak = _judges(args, spend)
            if args.command == "calibrate":
                result: object = calibrate(primary, tiebreak)
                text = json.dumps({**result, "judge_budget": budget.summary()}, indent=1, sort_keys=True) + "\n"
            else:
                judged = judge_attempts(read_jsonl(args.transcript), primary, tiebreak)
                text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in judged)
        except BaseException:
            os.close(descriptor)
            raise
        finally:
            ceiling.uninstall()
        private.write_descriptor(descriptor, text)
        print(json.dumps({"command": args.command, "judge_budget": budget.summary()}, sort_keys=True))
    except OSError, ValueError, KeyError, TypeError:
        print("precision evaluation failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

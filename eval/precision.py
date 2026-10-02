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
REQUEST_ALLOWANCE_TOKENS = 4_096
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


def metered(
    model: str, verdict: Callable[[judge.Item], judge.Verdict], budget: eval_cost.Budget, spend: list[eval_cost.Cost]
) -> Judgment:
    """Reserve a conservative bound before each judge call and settle what it reported."""
    import model_usage

    def run(item: judge.Item) -> judge.Verdict:
        messages = judge.prompt(item)
        # A token covers at least one byte; the allowance covers provider formatting and the structured-output schema.
        tokens = sum(len(str(message.content).encode()) for message in messages) + REQUEST_ALLOWANCE_TOKENS
        reservation = budget.reserve(eval_cost.call_bound(model, tokens, judge.MAX_OUTPUT_TOKENS))
        try:
            result, counts = model_usage.measure(lambda: verdict(item))
        except BaseException:
            budget.settle(reservation, eval_cost.Cost(0.0, known=False))
            raise
        spent = eval_cost.cost(eval_cost.Usage.of(counts), model)
        budget.settle(reservation, spent)
        spend.append(spent)
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
    items = judge.calibration_items()
    summary: dict[str, object] = {
        "adjudication": "author-adjudicated; awaits owner review",
        "judge_identity": judge.identity(),
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
    """What an engineering arm recorded: working-set recall and size, refusals, and escalations."""
    if not attempts or "exposed" not in attempts[0]:
        return {}
    dispatched = [attempt for attempt in attempts if int(attempt["operations"]) > 0]
    signals: dict[str, int] = defaultdict(int)
    for attempt in dispatched:
        if attempt["escalated"]:
            signals[str(attempt["escalation_signal"])] += 1
    return {
        "arm_signals": {
            "working_set_recall": sum(bool(a["recall"]) for a in attempts) / len(attempts),
            "mean_exposed_assistants": sum(len(a["exposed"]) for a in attempts) / len(attempts),
            "refusals": sum(int(a["refusals"]) for a in attempts),
            "escalated": sum(signals.values()),
            "escalation_rate": sum(signals.values()) / len(dispatched) if dispatched else None,
            "escalation_signals": dict(sorted(signals.items())),
            "escalation_blocked": sum(bool(a["escalation_blocked"]) for a in attempts),
        }
    }


def _paired(
    rows: Sequence[tuple[Mapping[str, object], str]], baseline: str, candidate: str, seed: str
) -> dict[str, object]:
    """Candidate minus baseline success over complete (scenario, repetition) pairs of one campaign.

    An incomplete pair is inconclusive; the caller has already refused a repeated pairing key.
    """
    pairs: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for attempt, result in rows:
        pairs[str(attempt["scenario"]), int(attempt["repetition"])][str(attempt["arm"])] = result
    complete: dict[str, list[tuple[float, float]]] = defaultdict(list)
    incomplete = 0
    for (scenario, _repetition), results in sorted(pairs.items()):
        if {results.get(baseline), results.get(candidate)} & {None, "inconclusive"}:
            incomplete += 1
            continue
        complete[SCENARIOS[scenario].template.id].append(
            (float(results[baseline] == "success"), float(results[candidate] == "success"))
        )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "complete_pairs": sum(len(items) for items in complete.values()),
        "incomplete_pairs": incomplete,
        "difference": eval_stats.paired_difference(complete, f"{seed}:paired:{candidate}"),
    }


# The closed vocabulary of campaign metadata and calibration summaries a report may carry (ADR-0094).
META_KEYS = frozenset(
    {
        *(
            "a",
            "b",
            "adjudication",
            "admitted",
            "agree",
            "all_criteria",
            "anthropic",
            "arms",
            "asks_for_missing_information",
        ),
        *(
            "baseline_arm",
            "brain",
            "budget",
            "budget_plan_usd",
            "campaign",
            "campaigns",
            "canary",
            "cap_usd",
            "commits",
        ),
        *("corpus", "date", "digest", "effort", "efforts", "failed", "final_judging", "id", "identical_arms", "items"),
        *(
            "judge_budget",
            "judge_identity",
            "judge_spend",
            "judge_version",
            "judges",
            "kind",
            "label",
            "language_matches",
        ),
        *("max_output_tokens", "max_reservation_usd", "model", "openai_paired", "path", "primary", "provider", "rate"),
        *("refused", "repetitions", "reply_correct", "requests", "reservations_exceeded", "scenario_patterns"),
        *("scenarios", "sdk_retries", "seconds", "seed", "spent_usd", "stopped_by_cap", "team", "tiebreak", "total"),
        *(
            "total_estimated_spend_usd",
            "umbrella",
            "unknown_settlements",
            "unreported",
            "unsupported_claim",
            "wilson95",
        ),
        *("workers", "superseded_usd", "calibration_usd", "comparisons", "escalation_model", "arms_features"),
    }
)
ARM_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,15}\Z")
SAFE_TEXT_RE = re.compile(r"[A-Za-z0-9 ._:;/()+,=*#-]{0,160}\Z")
FINGERPRINT_RE = re.compile(r"(sha256:[0-9a-f]{64}|[0-9a-f]{40})\Z")
# A provider key prefix at a word start; "task-create" is not one.
CREDENTIAL_RE = re.compile(r"(?<![A-Za-z0-9])sk-")
MAX_META_DEPTH = 5


def checked_meta(value: object, depth: int = 0, parent: str = "") -> object:
    """Return metadata only when it fits the closed vocabulary: known keys, numbers, and short safe text.

    Free text such as a reply, and anything shaped like a credential (``sk-`` or a long unbroken token other than a
    digest or commit), is refused rather than copied into a committed report.
    """
    if depth > MAX_META_DEPTH:
        raise ValueError("report metadata nests too deeply")
    if isinstance(value, dict):
        for key in value:
            known = key in META_KEYS or (parent == "efforts" and ARM_RE.fullmatch(str(key)))
            if not isinstance(key, str) or not known:
                raise ValueError("report metadata has an unknown field")
        return {key: checked_meta(item, depth + 1, key) for key, item in value.items()}
    if isinstance(value, list):
        return [checked_meta(item, depth + 1, parent) for item in value]
    if isinstance(value, str):
        unbroken = max((len(part) for part in value.split()), default=0)
        if (
            CREDENTIAL_RE.search(value)
            or SAFE_TEXT_RE.fullmatch(value) is None
            or (unbroken > 40 and not FINGERPRINT_RE.fullmatch(value))
        ):
            raise ValueError("report metadata has unsafe text")
        return value
    if value is None or isinstance(value, bool | int) or (isinstance(value, float) and math.isfinite(value)):
        return value
    raise ValueError("report metadata has an unsupported value")


def grade(
    calibration: Mapping[str, object] | None, judged: Sequence[Mapping[str, object]], meta: Mapping[str, object]
) -> dict[str, object]:
    """A report supports a decision only with an admitted calibration of the very judge that produced every verdict."""
    current = judge.identity()
    reasons = []
    if calibration is None:
        reasons.append("no judge calibration")
    else:
        if calibration.get("judge_identity") != current:
            reasons.append("the calibration is of another judge")
        reasons.extend(
            f"the {name} judge is not admitted"
            for name in ("primary", "tiebreak")
            if not (calibration.get(name) or {}).get("admitted")
        )
    if any(item.get("judge_identity") != current for item in judged):
        reasons.append("a verdict is from another judge")
    if meta.get("label") == "exploratory":
        reasons.append("the campaign is labelled exploratory")
    return {"grade": "exploratory" if reasons else "decision", "reasons": reasons, "judge_identity": current}


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
    """Group attempts by campaign, provider, model, and arm; pair arms only within one campaign.

    Every attempt key (campaign, provider, model, arm, repetition, scenario) must be unique, so two campaigns are
    never merged into one pair. Each arm is paired against the ``baseline_arm`` the metadata names, or the first arm.
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
    by_campaign: dict[tuple[str, str, str], dict[str, list]] = defaultdict(dict)
    for (campaign, provider, model, arm), rows in sorted(groups.items()):
        effort = {str(attempt["effort"]) for attempt, _ in rows}
        name = f"{campaign}:{provider}:{model}:{arm}"
        identity = {"campaign": campaign, "provider": provider, "model": model, "arm": arm, "effort": sorted(effort)}
        runs.append({**identity, **_group(rows, seed, name)})
        by_campaign[campaign, provider, model][arm] = rows
    paired = []
    pooled = []
    for (campaign, provider, model), arms in sorted(by_campaign.items()):
        if len(arms) < 2:
            continue
        rows = [row for items in arms.values() for row in items]
        baseline = meta.get("baseline_arm") if meta.get("baseline_arm") in arms else sorted(arms)[0]
        identity = {"campaign": campaign, "provider": provider, "model": model}
        comparisons = [
            (base, candidate) for base, candidate in meta.get("comparisons", ()) if {base, candidate} <= set(arms)
        ] or [(baseline, candidate) for candidate in sorted(arms) if candidate != baseline]
        paired.extend({**identity, **_paired(rows, base, candidate, seed)} for base, candidate in comparisons)
        if meta.get("identical_arms"):
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


def _judges(
    args: argparse.Namespace, budget: eval_cost.Budget, spend: list[eval_cost.Cost]
) -> tuple[Judgment, Judgment]:
    from eval.intent_route import _key

    judgments = []
    for provider, key_file in (("openai", args.key_file), ("anthropic", args.tiebreak_key_file)):
        model = judge.judge_model(provider, _key(key_file))
        judgments.append(
            metered(
                judge.JUDGE_MODELS[provider],
                lambda item, model=model, provider=provider: judge.judge(model, provider, item),
                budget,
                spend,
            )
        )
    return judgments[0], judgments[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("validate", "calibrate", "judge", "report"))
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
        if args.command == "report":
            meta = json.loads(args.meta.read_text(encoding="utf-8")) if args.meta else {}
            calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
            report = build_report(read_jsonl(args.transcript), read_jsonl(args.judged), meta, calibration, args.split)
            args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            return 0
        budget = eval_cost.Budget(args.cap)
        spend: list[eval_cost.Cost] = []
        primary, tiebreak = _judges(args, budget, spend)
        if args.command == "calibrate":
            result: object = calibrate(primary, tiebreak)
            text = json.dumps({**result, "judge_budget": budget.summary()}, indent=1, sort_keys=True) + "\n"
        else:
            judged = judge_attempts(read_jsonl(args.transcript), primary, tiebreak)
            text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in judged)
        private.write_private(args.out, text)
        print(json.dumps({"command": args.command, "judge_budget": budget.summary()}, sort_keys=True))
    except OSError, ValueError, KeyError, TypeError:
        print("precision evaluation failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Judge and report precision-v3 episode campaigns (ADR-0094).

Experiment-only. ``calibrate`` measures both trajectory judges on the author-written development set; ``judge`` grades
every completed episode of a transcript written by ``.tests/perf/precision_episodes.py`` under a hard cap, with the
same spend ceiling as the reply judges; ``report`` writes a sanitized report: no message, reply, record, or state.

An episode succeeds when its oracle passed and the final trajectory verdict passes every criterion; a budget-stopped
episode, or a completed one left unjudged, is inconclusive. Each arm reports success over template clusters, pass^k
per episode over its repetitions, accounting (user turns, Brain operations, model calls, Action batches, Actions,
failed turns, injected provider errors), cost per attempted and per successful episode, active-time percentiles,
cold and warm starts, oracle counts, and simulator limitations. Arms of one campaign pair over (episode, repetition).
The grade is exploratory until the owner's blind labels calibrate the very judge that produced every verdict.
"""

import argparse
import concurrent.futures
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from eval import cost as eval_cost
from eval import private
from eval import stats as eval_stats
from eval.complex import judging
from eval.complex.model import CORPUS_ID, STRATA, Episode, digest
from eval.complex.templates import TEMPLATES, TEMPLATES_BY_ID

SCHEMA = "precision-v3-report-1"
WORKERS = 6
ORACLE_COUNTS = (
    "final_differences",
    "checkpoints_failed",
    "forbidden",
    "duplicates",
    "cross_team_writes",
    "violations",
)
ACCOUNTING = (
    "user_turns",
    "brain_operations",
    "model_calls",
    "action_batches",
    "actions",
    "failed_turns",
    "injected_provider_errors",
)
Judgment = Callable[[judging.Trajectory], judging.TrajectoryVerdict]


def key(record: Mapping[str, object]) -> str:
    parts = ("campaign", "provider", "model", "arm", "repetition", "episode")
    return "/".join(str(record[name]) for name in parts)


def trajectory(record: Mapping[str, object]) -> judging.Trajectory:
    episode = Episode(TEMPLATES_BY_ID[str(record["template"])], str(record["locale"]))
    turns = tuple(judging.Turn(int(t["step"]), str(t["user"]), t["reply"], tuple(t["record"])) for t in record["turns"])
    return judging.Trajectory(episode, turns)


def judge_episodes(
    records: Iterable[Mapping[str, object]], primary: Judgment, tiebreak: Judgment, workers: int = WORKERS
) -> list[dict[str, object]]:
    """Judge every completed episode; one whose judgment failed or hit the cap stays unjudged."""

    def one(record: Mapping[str, object]) -> dict[str, object] | None:
        try:
            decision = judging.decide(primary, tiebreak, trajectory(record), bool(record["oracle"]["passed"]))
        except judging.TrajectoryJudgeError, eval_cost.BudgetExhaustedError:
            return None
        return {
            "key": key(record),
            "primary": decision.primary.model_dump(),
            "tiebreak": None if decision.tiebreak is None else decision.tiebreak.model_dump(),
            "final": decision.final.model_dump(),
            "judge_identity": judging.identity(),
        }

    completed = [record for record in records if record["status"] == "completed"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return [result for result in pool.map(one, completed) if result is not None]


def calibrate(primary: Judgment, tiebreak: Judgment) -> dict[str, object]:
    """Both judges against the author's development labels; exploratory by construction."""
    items = judging.calibration_items()
    summary: dict[str, object] = {"adjudication": "author", "blind": False, "judge_identity": judging.identity()}
    for name, judgment in (("primary", primary), ("tiebreak", tiebreak)):
        results = []
        for _item_id, item, expected in items:
            try:
                results.append((judgment(item), expected))
            except judging.TrajectoryJudgeError, eval_cost.BudgetExhaustedError:
                results.append((None, expected))
        summary[name] = judging.agreement(results)
    return summary


def outcome(record: Mapping[str, object], decision: Mapping[str, object] | None) -> str:
    if record["status"] != "completed":
        return "inconclusive"
    if not record["oracle"]["passed"]:
        return "failure"
    if decision is None:
        return "inconclusive"
    return "success" if judging.succeeded(judging.TrajectoryVerdict(**decision["final"])) else "failure"


def _clusters(rows: Sequence[tuple[Mapping[str, object], str]]) -> dict[str, list[float]]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for record, result in rows:
        if result != "inconclusive":
            clusters[str(record["template"])].append(float(result == "success"))
    return clusters


def _pass_k(rows: Sequence[tuple[Mapping[str, object], str]], seed: str, name: str) -> dict[str, object]:
    by_episode: dict[str, list[bool]] = defaultdict(list)
    for record, result in rows:
        if result != "inconclusive":
            by_episode[str(record["episode"])].append(result == "success")
    summary = {}
    for k in (1, 2, 5):
        clusters: dict[str, list[float]] = defaultdict(list)
        for episode, results in by_episode.items():
            estimate = eval_stats.pass_hat_k(sum(results), len(results), k)
            if estimate is not None:
                clusters[episode.rsplit(".", 1)[0]].append(estimate)
        summary[f"pass^{k}"] = eval_stats.cluster_bootstrap(clusters, f"{seed}:{name}:pass{k}")
    return summary


def _group(rows: Sequence[tuple[Mapping[str, object], str]], seed: str, name: str) -> dict[str, object]:
    records = [record for record, _ in rows]
    conclusive = [(record, result) for record, result in rows if result != "inconclusive"]
    dispatched = [record for record in records if int(record["accounting"]["brain_operations"]) > 0]
    successes = sum(result == "success" for _, result in conclusive)
    costs = [eval_cost.Cost(float(r["usd"]), bool(r["usage_known"])) for r in dispatched]
    active = [float(record["seconds_active"]) for record, _ in conclusive]
    strata = {}
    for stratum in STRATA:
        chosen = [(record, result) for record, result in rows if record["stratum"] == stratum]
        strata[stratum] = eval_stats.cluster_bootstrap(_clusters(chosen), f"{seed}:{name}:{stratum}")
    return {
        "episodes": len(records),
        "conclusive": len(conclusive),
        "successes": successes,
        "success": eval_stats.cluster_bootstrap(_clusters(rows), f"{seed}:{name}"),
        **_pass_k(rows, seed, name),
        "strata": strata,
        "oracle": {
            "passed": sum(bool(r["oracle"]["passed"]) for r in records if r["status"] == "completed"),
            **{count: sum(int(r["oracle"][count]) for r in records) for count in ORACLE_COUNTS},
        },
        "accounting": {field: sum(int(r["accounting"][field]) for r in records) for field in ACCOUNTING},
        "limitations": sum(len(r["limitations"]) for r in records),
        "cache": {label: sum(r["cache"] == label for r in dispatched) for label in ("cold", "warm", "unknown")},
        "cost": eval_cost.per_task(costs, successes),
        "latency_seconds": {
            "active_p50": eval_stats.percentile(active, 0.5),
            "active_p95": eval_stats.percentile(active, 0.95),
        },
    }


def _paired(rows: Sequence[tuple[Mapping[str, object], str]], base: str, candidate: str, seed: str) -> dict:
    pairs: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for record, result in rows:
        pairs[str(record["episode"]), int(record["repetition"])][str(record["arm"])] = result
    complete: dict[str, list[tuple[float, float]]] = defaultdict(list)
    incomplete = 0
    for (episode, _repetition), results in sorted(pairs.items()):
        if {results.get(base), results.get(candidate)} & {None, "inconclusive"}:
            incomplete += 1
            continue
        complete[episode.rsplit(".", 1)[0]].append(
            (float(results[base] == "success"), float(results[candidate] == "success"))
        )
    return {
        "baseline": base,
        "candidate": candidate,
        "complete_pairs": sum(len(items) for items in complete.values()),
        "incomplete_pairs": incomplete,
        "difference": eval_stats.paired_difference(complete, f"{seed}:paired:{candidate}"),
    }


def grade(calibration: Mapping[str, object] | None, judged: Sequence[Mapping[str, object]]) -> dict[str, object]:
    current = judging.identity()
    reasons = []
    if calibration is None or calibration.get("adjudication") != "owner" or calibration.get("blind") is not True:
        reasons.append("no-blind-owner-labels")
    if calibration is not None and calibration.get("judge_identity") != current:
        reasons.append("calibration-of-another-judge")
    if any(item.get("judge_identity") != current for item in judged):
        reasons.append("verdict-of-another-judge")
    return {"grade": "exploratory" if reasons else "decision", "reasons": reasons, "judge_identity": current}


def build_report(
    records: Sequence[Mapping[str, object]],
    judged: Sequence[Mapping[str, object]],
    meta: Mapping[str, object],
    calibration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from eval.precision import checked_identity, checked_meta

    keys = [key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate episode key")
    if any(record.get("corpus") != CORPUS_ID for record in records):
        raise ValueError("a record belongs to another corpus")
    decisions = {str(item["key"]): item for item in judged}
    seed = str(meta.get("seed", "precision-v3"))
    groups: dict[tuple[str, str, str, str], list[tuple[Mapping[str, object], str]]] = defaultdict(list)
    for record in records:
        groups[str(record["campaign"]), str(record["provider"]), str(record["model"]), str(record["arm"])].append(
            (record, outcome(record, decisions.get(key(record))))
        )
    runs, by_campaign = [], defaultdict(dict)
    for (campaign, provider, model, arm), rows in sorted(groups.items()):
        checked_identity(campaign, provider, model, arm, sorted({str(r["effort"]) for r, _ in rows}))
        name = f"{campaign}:{provider}:{model}:{arm}"
        runs.append(
            {"campaign": campaign, "provider": provider, "model": model, "arm": arm, **_group(rows, seed, name)}
        )
        by_campaign[campaign].setdefault(arm, []).extend(rows)
    paired = []
    for campaign, arms in sorted(by_campaign.items()):
        ordered = sorted(arms)
        rows = [row for items in arms.values() for row in items]
        paired += [{"campaign": campaign, **_paired(rows, ordered[0], other, seed)} for other in ordered[1:]]
    return {
        "schema": SCHEMA,
        "corpus": {"id": CORPUS_ID, "digest": digest(TEMPLATES), "templates": len(TEMPLATES)},
        "meta": checked_meta(dict(meta)),
        "decision": grade(calibration, judged),
        "judge_calibration": None if calibration is None else checked_meta(dict(calibration)),
        "runs": runs,
        "paired": paired,
        "judges": {
            "judged_episodes": len(judged),
            "tiebreaks": sum(item["tiebreak"] is not None for item in judged),
            "verdict_identities": sorted({str(item["judge_identity"]) for item in judged}),
        },
    }


def _judges(args: argparse.Namespace) -> tuple[Judgment, Judgment]:
    from eval import judge
    from eval.intent_route import _key

    judgments = []
    for provider, key_file in (("openai", args.key_file), ("anthropic", args.tiebreak_key_file)):
        model = judge.judge_model(provider, _key(key_file))
        judgments.append(lambda item, model=model, provider=provider: judging.judge(model, provider, item))
    return judgments[0], judgments[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("calibrate", "judge", "report"))
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--judged", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--tiebreak-key-file", type=Path)
    parser.add_argument("--cap", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            meta = json.loads(args.meta.read_text(encoding="utf-8")) if args.meta else {}
            calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
            body = build_report(private.read_jsonl(args.transcript), private.read_jsonl(args.judged), meta, calibration)
            args.out.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            return 0
        from eval import judge
        from eval.ceiling import Ceiling

        descriptor = private.open_private(args.out)
        ceiling = Ceiling(args.cap, judge.MAX_OUTPUT_TOKENS)
        ceiling.install()
        try:
            primary, tiebreak = _judges(args)
            if args.command == "calibrate":
                text = json.dumps({**calibrate(primary, tiebreak), "judge_budget": ceiling.budget.summary()})
            else:
                judged = judge_episodes(private.read_jsonl(args.transcript), primary, tiebreak)
                text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in judged)
        except BaseException:
            os.close(descriptor)
            raise
        finally:
            ceiling.uninstall()
        private.write_descriptor(descriptor, text)
        print(json.dumps({"command": args.command, "judge_budget": ceiling.budget.summary()}, sort_keys=True))
    except OSError, ValueError, KeyError, TypeError:
        print("precision-v3 evaluation failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

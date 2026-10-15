"""The judged-campaign pipeline shared by the precision reports (ADR-0094).

Both the whole-task report (``eval.precision``) and the episode report (``eval.complex.report``) judge every completed
record with a primary judge and a tiebreak, measure both judges against a labelled sample, estimate pass^k over
repetitions, and run paid judge work under a hard cap into an owner-only file. Each report keeps its own record key,
judge input, outcome, and grouping; this module holds only the mechanics they share.
"""

import concurrent.futures
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from eval import cost as eval_cost
from eval import private
from eval import stats as eval_stats

Row = tuple[Mapping[str, object], str]


def judge_completed(
    records: Iterable[Mapping[str, object]],
    decide: Callable[[Mapping[str, object]], Any],
    key: Callable[[Mapping[str, object]], str],
    identity: str,
    failure: type[Exception],
    workers: int,
) -> list[dict[str, object]]:
    """Judge every completed record; one whose judgment failed or hit the cap stays unjudged."""

    def one(record: Mapping[str, object]) -> dict[str, object] | None:
        try:
            decision = decide(record)
        except failure, eval_cost.BudgetExhaustedError:
            return None
        return {
            "key": key(record),
            "primary": decision.primary.model_dump(),
            "tiebreak": None if decision.tiebreak is None else decision.tiebreak.model_dump(),
            "final": decision.final.model_dump(),
            "judge_identity": identity,
        }

    completed = [record for record in records if record["status"] == "completed"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return [result for result in pool.map(one, completed) if result is not None]


def agreements(
    judges: Mapping[str, Callable[[Any], Any]],
    items: Sequence[tuple[str, Any, Mapping[str, bool]]],
    agreement: Callable[[list[tuple[Any, Mapping[str, bool]]]], dict[str, object]],
    failure: type[Exception],
) -> dict[str, object]:
    """Each named judge's agreement with the labelled items; a failed or capped judgment counts as no verdict."""
    summary: dict[str, object] = {}
    for name, judgment in judges.items():
        results = []
        for _item_id, item, expected in items:
            try:
                results.append((judgment(item), expected))
            except failure, eval_cost.BudgetExhaustedError:
                results.append((None, expected))
        summary[name] = agreement(results)
    return summary


def pass_k(
    rows: Sequence[Row],
    unit: Callable[[Mapping[str, object]], str],
    cluster: Callable[[str], str],
    ks: tuple[int, ...],
    seed: str,
    name: str,
) -> dict[str, object]:
    """pass^k per repeated unit over its conclusive results, bootstrapped over the units' clusters."""
    by_unit: dict[str, list[bool]] = defaultdict(list)
    for record, result in rows:
        if result != "inconclusive":
            by_unit[unit(record)].append(result == "success")
    summary: dict[str, object] = {}
    for k in ks:
        clusters: dict[str, list[float]] = defaultdict(list)
        for value, results in by_unit.items():
            estimate = eval_stats.pass_hat_k(sum(results), len(results), k)
            if estimate is not None:
                clusters[cluster(value)].append(estimate)
        summary[f"pass^{k}"] = eval_stats.cluster_bootstrap(clusters, f"{seed}:{name}:pass{k}")
    return summary


def judge_pair(
    key_file: Path, tiebreak_key_file: Path, bind: Callable[[Any, str], Callable[[Any], Any]]
) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """The primary (OpenAI) and tiebreak (Anthropic) judgments, each a bounded judge model bound by ``bind``."""
    from eval import judge

    primary, tiebreak = (
        bind(judge.judge_model(provider, private.read_key(path)), provider)
        for provider, path in (("openai", key_file), ("anthropic", tiebreak_key_file))
    )
    return primary, tiebreak


def run_paid(out: Path, cap: float, command: str, work: Callable[[Any], str]) -> None:
    """Run paid judge work under a hard US dollar cap and write its text to an owner-only file.

    The output is admitted, and held open, before any paid inference. The ceiling is installed before any judge model
    exists: no SDK retries, clamped output, and a reserved worst case per request. ``work`` receives its budget.
    """
    from eval import judge
    from eval.ceiling import Ceiling

    descriptor = private.open_private(out)
    ceiling = Ceiling(cap, judge.MAX_OUTPUT_TOKENS)
    ceiling.install()
    try:
        text = work(ceiling.budget)
    except BaseException:
        os.close(descriptor)
        raise
    finally:
        ceiling.uninstall()
    private.write_descriptor(descriptor, text)
    print(json.dumps({"command": command, "judge_budget": ceiling.budget.summary()}, sort_keys=True))

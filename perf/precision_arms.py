"""Paired multi-arm precision-v2.1 campaign with one disposable Brain process per prompt arm (ADR-0094).

Evaluation-only. Each arm of ``perf/prompt_arms.py`` gets its own loopback Brain from the umbrella's
``.tests/perf/precision_journeys.py``, with its own spend ceiling (``--cap-per-arm``) and its arm's output limit;
every scenario and repetition runs all arms in a recorded random order. Run from the umbrella root under Brain's
environment plus Team's pinned cryptography, as the journeys require:

    uv run --project brain --frozen --python 3.14 --with cryptography==50.0.2 python brain/perf/precision_arms.py
        --arms base,prod --repetitions 5 --cap-per-arm 0.8 --workers 6 --seed <seed> --campaign <name>
        --transcript /private/dir/arms.jsonl --meta-out /private/dir/arms-meta.json   (one command line)

Judge the transcript with ``eval.precision judge`` (or ``perf/sol_tiebreak.py``), build the report with
``eval.precision report``. The metadata records the umbrella, Brain, and Team revisions; record with the report the
sha256 of each ``perf/`` source the run used, as the 2026-10-09 reports' ``provenance`` does.
"""

import argparse
import concurrent.futures
import importlib
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

PERF = Path(__file__).resolve().parent
UMBRELLA = PERF.parents[1]


def _modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    sys.path.insert(0, str(UMBRELLA / ".tests" / "perf"))
    journeys = importlib.import_module("precision_journeys")
    # The journeys put the Brain checkout on the import path; the arms and the report checks import from it.
    return journeys, importlib.import_module("prompt_arms"), importlib.import_module("eval.precision")


def _brains(journeys: ModuleType, arms_module: ModuleType, arms: tuple[str, ...], cap: float) -> dict[str, object]:
    """One loopback Brain per arm, its arm installed by the site hook; every started Brain stops if one fails."""
    os.environ["PROMPT_ARMS_BUDGET_HOOK"] = str(journeys.BUDGET_MODULE)
    journeys.BUDGET_MODULE = PERF / "prompt_arms_site.py"
    brains: dict[str, object] = {}
    try:
        for arm in arms:
            os.environ["PROMPT_ARMS_ARM"] = arm
            brains[arm] = journeys.BrainProcess(cap, arms_module.output_limit(arm))
    except BaseException:
        for brain in brains.values():
            brain.close()
        raise
    return brains


def _campaign(journeys: ModuleType, args: argparse.Namespace, plan: dict[str, object], brains: dict) -> None:
    """Every scenario and repetition in a recorded random order, its arms in their own recorded order."""
    seed, name, arms, key, model = plan["seed"], plan["campaign"], plan["arms"], plan["key"], args.model
    clients = {
        arm: journeys.MeteredBrainClient(brain, model, base_url=brain.url, token_file=brain.token_file)
        for arm, brain in brains.items()
    }
    tasks = [(repetition, scenario) for repetition in range(args.repetitions) for scenario in plan["scenarios"]]
    order = journeys.eval_stats.arm_order(seed, "openai", "schedule", [str(index) for index in range(len(tasks))])
    lock, stopped, counter = threading.Lock(), threading.Event(), iter(range(10**9))
    descriptor = journeys.private.open_private(args.transcript, append=True)

    def task(item: tuple[int, object]) -> None:
        repetition, scenario = item
        for position, arm in enumerate(
            journeys.eval_stats.arm_order(seed, "openai", f"{scenario.id}/{repetition}", arms)
        ):
            client = clients[arm]
            if stopped.is_set() or client.refused():
                stopped.set()
                record = journeys._stopped(scenario)
            else:
                with lock:
                    thread = f"precision.{name}.{next(counter)}"
                record = journeys.run_attempt(client, "openai", model, key, scenario, thread, journeys.TURN_EFFORT)
            record |= {
                "campaign": name,
                "provider": "openai",
                "model": model,
                "effort": journeys.TURN_EFFORT,
                "arm": arm,
                "repetition": repetition,
                "scenario": scenario.id,
                "position": position,
                "simulation": journeys.fixtures.SIMULATION,
            }
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            with lock:
                os.write(descriptor, line.encode())

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(task, [tasks[int(index)] for index in order]))
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", required=True, help="two or more arms of perf/prompt_arms.py, the baseline first")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--key-file", type=Path, default=UMBRELLA / ".gpt-key")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--scenarios", default="*", help="comma-separated id patterns, such as '*.en,dns-*'")
    parser.add_argument("--cap-per-arm", type=float, required=True, help="hard US dollar ceiling of each arm's Brain")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed")
    parser.add_argument("--campaign")
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--meta-out", type=Path, required=True)
    args = parser.parse_args()
    journeys, arms_module, precision = _modules()
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if len(arms) < 2 or len(set(arms)) != len(arms) or not set(arms) <= set(arms_module.ARMS):
        parser.error(f"--arms needs two or more distinct arms among {', '.join(arms_module.ARMS)}")
    journeys.check_pins()
    journeys.corpus.validate()
    seed = args.seed or secrets.token_hex(8)
    plan = {
        "seed": seed,
        "campaign": args.campaign or f"arms-{seed}",
        "arms": arms,
        "key": journeys._key(args.key_file),
    }
    plan["scenarios"] = journeys._selected(args.scenarios)
    os.close(journeys.private.open_private(args.meta_out))  # refuse an unsafe destination before spending anything
    commits = {"umbrella": journeys._git(journeys.ROOT), "brain": journeys._git(journeys.BRAIN)}
    commits["team"] = journeys._git(journeys.TEAM)
    started = time.time()
    brains = _brains(journeys, arms_module, arms, args.cap_per_arm)
    try:
        _campaign(journeys, args, plan, brains)
        budgets = {arm: brain.budget() for arm, brain in brains.items()}
    finally:
        for brain in brains.values():
            brain.close()
    meta = {
        "campaign": plan["campaign"],
        "seed": seed,
        "provider": "openai",
        "model": args.model,
        "arms": list(arms),
        "baseline_arm": arms[0],
        "identical_arms": False,
        "repetitions": args.repetitions,
        "scenarios": len(plan["scenarios"]),
        "workers": args.workers,
        "sdk_retries": 0,
        "budget": {
            "cap_usd": round(args.cap_per_arm * len(arms), 6),
            "spent_usd": round(sum(budget["spent_usd"] for budget in budgets.values()), 6),
            "refused": sum(budget["refused"] for budget in budgets.values()),
        },
        "stopped_by_cap": any(budget["refused"] for budget in budgets.values()),
        "seconds": round(time.time() - started, 1),
        "commits": commits,
        "corpus": {"id": journeys.corpus.CORPUS_ID, "digest": journeys.corpus.digest()},
        "path": "team-orchestrator+loopback-brain",
        "label": "exploratory",
    }
    journeys.private.write_private(args.meta_out, json.dumps(precision.checked_meta(meta), indent=1, sort_keys=True))
    print(json.dumps({"campaign": plan["campaign"], "budgets": budgets}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate the TypeSafe Jev fast path for chat intent classification and its capability question.

The fast path may answer only ``ordinary-task``; every lifecycle or unresolved message must fall back to the LLM
route. Given installable Assistants, it may also call an ordinary task capability-free, which lets Admin skip waiting
for capability planning (ADR-0077 amendment, 2026-10-13); no message that needs a capability may ever be called
capability-free. Run ``PYTHONPATH=. uv run --frozen --python 3.14 python -m eval.intent_fast_path`` from Brain to
validate both corpora without a provider, or add ``--key-file ../.jev-key`` for live calls. Every live call goes
through the runtime's own exchange: its cancellable decision pool and 1.5-second deadline. Output contains only case
identifiers, labels, and aggregate metrics, never the key; the command exits 1 when either gate fails.

Intent metrics at each confidence threshold: fast-path precision (fast-pathed messages that are truly ordinary),
lifecycle false negatives (lifecycle or unresolved messages that would skip the LLM route; must be zero), and ordinary
coverage (ordinary messages that skip the LLM route). The intent corpus is independent of ``eval.intent_route`` and
mostly Portuguese because Jev documents English as its primary language; it runs both without installable Assistants
and with the owner-like set, because describing them changes the request.

Capability metrics per installable set and stratum: false skips (capability-needing messages called capability-free;
must be zero) and the skip rate of conversational messages. A cluster is one task template: its Portuguese and English
messages and every repetition are one unit, so Wilson intervals count clusters, never messages or runs. A
conversational cluster counts as skipped only when every one of its runs was; a capability cluster is safe only when
none of its runs was.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import intent_fast_path
import intent_route
import provider_cancel
from eval import stats
from eval.intent_fast_path_cases import CAPABILITY_CASES, CASES, INSTALLABLE, CapabilityCase, Case, validate_corpus
from eval.private import read_key

THRESHOLDS = tuple(sorted({0.7, 0.8, 0.9, 0.95, intent_fast_path.CONFIDENCE_THRESHOLD}))
CAPABILITY_THRESHOLDS = tuple(sorted({0.7, 0.8, 0.9, 0.95, 0.99, intent_fast_path.CAPABILITY_THRESHOLD}))


def metrics(results: list[tuple[Case, str, float]], attempted: tuple[Case, ...] | None = None) -> dict[str, object]:
    """Score each threshold: only a confident ordinary-task answer skips the LLM route.

    Coverage counts every ordinary case `attempted`, so one whose exchange failed counts as not covered.
    """
    cases = attempted if attempted is not None else tuple(case for case, _, _ in results)
    ordinary = sum(case.intent == "ordinary-task" for case in cases)
    report: dict[str, object] = {}
    for threshold in THRESHOLDS:
        fast = [
            (case, choice)
            for case, choice, confidence in results
            if choice == "ordinary-task" and confidence >= threshold
        ]
        correct = sum(case.intent == "ordinary-task" for case, _ in fast)
        report[str(threshold)] = {
            "fast_path": len(fast),
            "precision": round(correct / len(fast), 3) if fast else None,
            "lifecycle_false_negatives": sorted(case.id for case, _ in fast if case.intent != "ordinary-task"),
            "ordinary_coverage": round(correct / ordinary, 3) if ordinary else None,
        }
    return report


class _Exchanges:
    """Live exchanges through the runtime's own path: its decision pool, deadline, and response bounds."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.client = provider_cancel.client(provider_cancel.DECISION_CONNECTIONS)
        self.calls = 0
        self.latencies: list[float] = []

    def ask(
        self,
        message: str,
        context: intent_route.LifecycleContext,
        installable: tuple[intent_route.InstallableAssistant, ...],
    ) -> dict[str, tuple[str, float]] | None:
        decision = intent_fast_path.Decision(self.key, installable)
        started = time.perf_counter()
        answers = intent_fast_path.ask(self.client, decision, message, context)
        self.latencies.append((time.perf_counter() - started) * 1000)
        self.calls += 1
        return answers

    def latency(self) -> dict[str, object]:
        return {
            "p50": round(stats.percentile(self.latencies, 0.5) or 0),
            "p95": round(stats.percentile(self.latencies, 0.95) or 0),
        }


# A run whose Jev exchanges failed more often than this proves nothing about Jev's answers: its gate fails as
# inconclusive instead of scoring only the answers that happened to arrive.
MAX_FAILED_EXCHANGE_RATE = 0.02


def _conclusive(failed: int, attempted: int) -> bool:
    return attempted > 0 and failed <= MAX_FAILED_EXCHANGE_RATE * attempted


def _interval(successes: int, trials: int) -> list[float] | None:
    interval = stats.wilson(successes, trials)
    return None if interval is None else [round(interval[0], 3), round(interval[1], 3)]


def _intent_run(exchanges: _Exchanges, installable: tuple[intent_route.InstallableAssistant, ...]) -> dict[str, object]:
    results: list[tuple[Case, str, float]] = []
    failed: list[str] = []
    for case in CASES:
        answers = exchanges.ask(case.message, case.context, installable)
        if answers is None:
            failed.append(case.id)
            continue
        choice, confidence = answers["intent"]
        results.append((case, choice, confidence))
    scored = metrics(results, CASES)
    effective = scored[str(intent_fast_path.CONFIDENCE_THRESHOLD)]
    conclusive = _conclusive(len(failed), len(CASES))
    return {
        "cases": len(CASES),
        "failed_exchanges": sorted(failed),
        "conclusive": conclusive,
        "intent_accuracy": round(sum(case.intent == choice for case, choice, _ in results) / max(1, len(results)), 3),
        "misclassified": sorted(case.id for case, choice, _ in results if case.intent != choice),
        "thresholds": scored,
        "gate_passed": conclusive and not effective["lifecycle_false_negatives"],
    }


def _capability_cell(rows: list[tuple[CapabilityCase, dict[str, tuple[str, float]] | None]]) -> dict[str, object]:
    """One installable set and stratum: false skips must be zero; conversational skips are counted per cluster."""
    clusters: dict[str, list[bool]] = {}
    needs: dict[str, bool] = {}
    for case, answers in rows:
        clusters.setdefault(case.cluster, []).append(intent_fast_path.decide(answers).capability_free)
        needs[case.cluster] = case.needs
    false_skips = sorted(
        {case.id for case, answers in rows if case.needs and intent_fast_path.decide(answers).capability_free}
    )
    conversational = [name for name in clusters if not needs[name]]
    capability = [name for name in clusters if needs[name]]
    skipped = sum(all(clusters[name]) for name in conversational)
    safe = sum(not any(clusters[name]) for name in capability)
    runs = [verdict for name in conversational for verdict in clusters[name]]
    return {
        "runs": len(rows),
        "failed_exchanges": sum(answers is None for _case, answers in rows),
        "not_ordinary": sum(not intent_fast_path.decide(answers).ordinary for _case, answers in rows),
        "false_skips": false_skips,
        "capability_clusters": len(capability),
        "safe_capability_clusters": safe,
        "safe_capability_ci95": _interval(safe, len(capability)),
        "conversation_clusters": len(conversational),
        "skipped_conversation_clusters": skipped,
        "conversation_skip_ci95": _interval(skipped, len(conversational)),
        "conversation_run_skip_rate": round(sum(runs) / len(runs), 3) if runs else None,
        "missed_conversation": sorted(
            {
                case.id
                for case, answers in rows
                if not case.needs and not intent_fast_path.decide(answers).capability_free
            }
        ),
    }


def _sweep(rows: list[tuple[CapabilityCase, dict[str, tuple[str, float]] | None]]) -> dict[str, object]:
    """False skips and conversational run skips at other capability thresholds, from the same answers."""
    report: dict[str, object] = {}
    for threshold in CAPABILITY_THRESHOLDS:

        def free(answers: dict[str, tuple[str, float]] | None, limit: float = threshold) -> bool:
            if not intent_fast_path.decide(answers).ordinary:
                return False
            need, confidence = answers["capability"]
            return need == "conversation" and confidence >= limit

        report[str(threshold)] = {
            "false_skips": sorted({case.id for case, answers in rows if case.needs and free(answers)}),
            "conversation_run_skips": sum(free(answers) for case, answers in rows if not case.needs),
            "conversation_runs": sum(not case.needs for case, _answers in rows),
        }
    return report


def _capability_run(exchanges: _Exchanges, name: str, repeats: int) -> dict[str, object]:
    installable = INSTALLABLE[name]
    rows = [
        (case, exchanges.ask(case.message, case.context, installable))
        for _repeat in range(repeats)
        for case in CAPABILITY_CASES
    ]
    strata = {
        stratum: _capability_cell([row for row in rows if row[0].stratum == stratum])
        for stratum in ("adr-0066", "paraphrase", "conversation", "websearch", "referential")
    }
    conclusive = _conclusive(sum(answers is None for _case, answers in rows), len(rows))
    return {
        "repeats": repeats,
        "strata": strata,
        "all": _capability_cell(rows),
        "thresholds": _sweep(rows),
        "conclusive": conclusive,
        "gate_passed": conclusive and not any(cell["false_skips"] for cell in strata.values()),
    }


def evaluate(key: str, sets: tuple[str, ...], repeats: int) -> dict[str, object]:
    """Run the intent corpus without and with installable Assistants, then the capability corpus per set."""
    exchanges = _Exchanges(key)
    try:
        intent = {"none": _intent_run(exchanges, ()), "owner": _intent_run(exchanges, INSTALLABLE["owner"])}
        capability = {name: _capability_run(exchanges, name, repeats) for name in sets}
    finally:
        exchanges.client.close()
    return {
        "model": intent_fast_path.MODEL,
        "confidence_threshold": intent_fast_path.CONFIDENCE_THRESHOLD,
        "capability_threshold": intent_fast_path.CAPABILITY_THRESHOLD,
        "intent_cases": len(CASES),
        "capability_cases": len(CAPABILITY_CASES),
        "jev_calls": exchanges.calls,
        "latency_ms": exchanges.latency(),
        "intent": intent,
        "capability": capability,
        "gate_passed": all(run["gate_passed"] for run in (*intent.values(), *capability.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--sets", default="owner,adr-0066,adversarial", help="comma-separated installable sets")
    parser.add_argument("--repeats", type=int, default=3, help="runs of every capability case per set")
    args = parser.parse_args()
    sets = tuple(args.sets.split(","))
    try:
        validate_corpus()
        if not set(sets) <= set(INSTALLABLE) or args.repeats < 1:
            raise ValueError("unknown installable set or repeat count")
        if args.key_file is None:
            print(
                json.dumps(
                    {
                        "status": "corpus-inputs-valid",
                        "intent_cases": len(CASES),
                        "capability_cases": len(CAPABILITY_CASES),
                        "model": intent_fast_path.MODEL,
                    }
                )
            )
            return 0
        result = evaluate(read_key(args.key_file), sets, args.repeats)
    except OSError, ValueError, httpx.HTTPError, KeyError:
        print("fast-path evaluation failed", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

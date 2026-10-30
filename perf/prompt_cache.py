"""Multi-turn prompt-cache sessions of the shipped Brain, read from the usage each operation reports (ADR-0094).

Evaluation-only. Every session plays Team against the Rust Brain (``SHIMPZ_BRAIN_BINARY``) over its HTTP API, and every
provider request runs through the evaluation's provider origin, reserved under ``--cap`` at an 8,000-token output limit.
Strata:

- ``sat`` first fills the thread's history with ``SEED_EXCHANGES`` short completed exchanges (each one a real, short
  model turn: the Brain takes no stand-in model), then runs the measured turns on that saturated history;
- ``mem`` runs the measured turns from an empty history with ten memories, one changed from turn 5.

Every measured turn asks for one task to be created (one exact ``create-task`` write with the asked title); the last
asks for the title created three turns earlier (that title literally in the reply). The cache reads, writes, and
fresh input come from the ``usage`` the Brain reports for each operation. The C1 stable-prefix and C2 hysteresis arms
of 2026-10-09 patched the Python Brain's process and cannot run against the binary; their report stays as history,
and a new arm needs its own decision (for example a build-time arm), not a process patch. From the Brain checkout:

    PYTHONPATH=. uv run --frozen --python 3.14 python -m perf.prompt_cache run --key-file ../.gpt-key
        --sessions 20 --turns 10 --cap 1.0 --workers 8 --seed <seed> --out <sessions.json>   (one command line)
    PYTHONPATH=. uv run --frozen --python 3.14 python -m perf.prompt_cache analyze <sessions.json>

The sessions file holds per-turn outcomes, usage, and cost only, never prompts or replies.
"""

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path

from eval import brain_api
from eval import cost as eval_cost
from eval import stats as eval_stats
from eval.injection import simulated_result
from eval.injection_cases import ASSISTANTS, CASES

STRATA = ("sat", "mem")
MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-6-luna"}
# The Brain's own history bound (brain/context_budget.py MAX_HISTORY_EXCHANGES when the Brain was Python): seeding
# this many exchanges saturates the history the next turn trims.
SEED_EXCHANGES = 24
MAX_OUTPUT_TOKENS = 8_000
MAX_ROUNDS = 8
TASKS = ASSISTANTS["tasks"]
PROBE = "What was the exact title of the task I asked you to create three messages ago? Answer with the title only."
WORDS = ("amber", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel")
WORDS += ("lagoon", "maple", "nectar", "onyx", "pepper", "quartz", "raven", "saffron", "tundra", "umber", "violet")
MEMORIES = tuple(
    {"topic": topic, "preference": preference}
    for topic, preference in (
        ("language", "Answer in short sentences."),
        ("units", "Use metric units."),
        ("dates", "Write dates as YYYY-MM-DD."),
        ("tone", "Be direct, no small talk."),
        ("lists", "Prefer bullet lists for several items."),
        ("tasks", "Task titles start with a capital letter."),
        ("weekends", "Never schedule anything on Sundays."),
        ("coffee", "I drink my coffee black."),
        ("music", "I like jazz."),
        ("city", "I live in Lisbon."),
    )
)
CHANGED = (*MEMORIES[:9], {"topic": "city", "preference": "I moved to Porto."})


def _pick(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "big")


@dataclasses.dataclass(frozen=True)
class Session:
    stratum: str
    index: int
    turns: int
    seed: str

    def title(self, turn: int) -> str:
        first, second = (WORDS[_pick(self.seed, self.stratum, self.index, turn, k) % len(WORDS)] for k in (0, 1))
        return f"{first.capitalize()} {second} {turn}"


def _context(session: Session, thread: str, provider: brain_api.Provider, turn: int) -> brain_api.TurnContext:
    memories = (MEMORIES if turn < 5 else CHANGED) if session.stratum == "mem" else ()
    return brain_api.TurnContext(thread, "Eval Team", (TASKS,), provider, memories=memories, routines=())


def seed_history(client: brain_api.Client, session: Session, context: brain_api.TurnContext) -> eval_cost.Usage:
    """Fill a ``sat`` session's history to saturation with short real turns; return what they reported."""
    meter = brain_api.UsageMeter()
    for k in range(SEED_EXCHANGES):
        word, box = WORDS[_pick(session.seed, "note", session.index, k) % len(WORDS)], 10 + _pick(k) % 90
        result = client.start(context, f"Note {k}: the {word} shelf holds box {box}.", locale="en")
        meter.add(result.usage)
    return meter.total


def _drive(
    client: brain_api.Client, context: brain_api.TurnContext, message: str, writes: list[dict], meter
) -> brain_api.Turn:
    result = client.start(context, message, locale="en")
    meter.add(result.usage)
    rounds = 0
    while result.status == "action-required" and rounds < MAX_ROUNDS:
        writes.extend(dict(request.input) for request in result.actions if request.action == "create-task")
        results = {request.interrupt_id: simulated_result(CASES[0], request, len(writes)) for request in result.actions}
        result = client.resume(context, results)
        meter.add(result.usage)
        rounds += 1
    return result


def _succeeded(probe: bool, title: str, result: brain_api.Turn, writes: list[dict]) -> bool:
    if probe:
        return result.status == "completed" and title.casefold() in result.reply.casefold()
    return (
        result.status == "completed"
        and len(writes) == 1
        and title.casefold() in str(writes[0].get("title", "")).casefold()
    )


def _turn(client, provider: brain_api.Provider, session: Session, thread: str, turn: int) -> dict[str, object]:
    probe = turn == session.turns
    title = session.title(session.turns - 3 if probe else turn)
    writes: list[dict] = []
    context = _context(session, thread, provider, turn)
    message = PROBE if probe else f"Create a task titled '{title}'."
    meter = brain_api.UsageMeter()
    try:
        result = _drive(client, context, message, writes, meter)
    except brain_api.BrainError as error:
        if error.status not in brain_api.MISS_STATUSES | {503}:
            raise
        print("inconclusive turn", session.stratum, turn, error.status, file=sys.stderr)
        meter.failed()
        ok, status = False, f"error:{error.status}"
    else:
        ok, status = _succeeded(probe, title, result, writes), result.status
    return {
        "stratum": session.stratum,
        "session": session.index,
        "turn": turn,
        "probe": probe,
        "ok": ok,
        "writes": len(writes),
        "titles": [] if probe else [str(write.get("title", "")) == title for write in writes],
        "status": status,
        "usage": meter.total.to_dict(),
        "usd": eval_cost.cost(meter.total, provider.model).usd,
    }


def run_session(client, provider: brain_api.Provider, session: Session) -> list[dict[str, object]]:
    """One session's measured turns; a ``sat`` session's seeding is spent but never measured."""
    thread = f"eval:cache:{secrets.token_hex(8)}"
    if session.stratum == "sat":
        seed_history(client, session, _context(session, thread, provider, 0))
    return [_turn(client, provider, session, thread, turn) for turn in range(session.turns + 1)]


def _run(args: argparse.Namespace) -> None:
    from eval.private import read_key

    provider = brain_api.Provider(args.provider, MODELS[args.provider], read_key(args.key_file))
    sessions = [
        Session(stratum, index, args.turns, args.seed)
        for stratum in args.strata.split(",")
        for index in range(args.sessions)
    ]
    records: list[dict[str, object]] = []
    with (
        brain_api.metered(args.cap, MAX_OUTPUT_TOKENS) as brain,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = [pool.submit(run_session, brain.client, provider, session) for session in sessions]
        for future in futures:
            records.extend(future.result())
        budget = brain.budget()
    body = {"records": records, "budget": budget, "seed": args.seed}
    args.out.write_text(json.dumps(body, indent=0) + "\n", encoding="utf-8")
    print(json.dumps(budget))


def _arm_summary(rows: Sequence[dict]) -> dict[str, object]:
    usage = sum((eval_cost.Usage.of(row["usage"]) for row in rows), eval_cost.Usage())
    usd, ok = sum(row["usd"] for row in rows), sum(row["ok"] for row in rows)
    tasks, probes = [row for row in rows if not row["probe"]], [row for row in rows if row["probe"]]
    return {
        "turns": len(rows),
        "errors": sum(row["status"].startswith("error") for row in rows),
        "successes": ok,
        "success_rate": ok / len(rows),
        "success_wilson": eval_stats.wilson(ok, len(rows)),
        "task_success": sum(row["ok"] for row in tasks) / len(tasks),
        "probe_success": sum(row["ok"] for row in probes) / len(probes),
        "input_tokens": usage.input_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "fresh_input_tokens": usage.fresh_input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_fraction": round(usage.cache_read_tokens / usage.input_tokens, 4),
        "usd": round(usd, 6),
        "usd_per_turn": round(usd / len(rows), 7),
        "usd_per_success": round(usd / ok, 7) if ok else None,
    }


def analyze(body: dict) -> dict[str, object]:
    """Per-stratum rates, tokens, cache traffic, and cost of the measured turns the sessions hold."""
    out: dict[str, object] = {"budget": body["budget"], "seed": body["seed"], "strata": {}}
    for stratum in STRATA:
        rows = [row for row in body["records"] if row["stratum"] == stratum]
        if rows:
            out["strata"][stratum] = {
                "sessions": len({row["session"] for row in rows}),
                **_arm_summary(rows),
            }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--key-file", type=Path, required=True)
    run.add_argument("--provider", choices=sorted(MODELS), default="openai")
    run.add_argument("--strata", default="sat,mem")
    run.add_argument("--sessions", type=int, default=6)
    run.add_argument("--turns", type=int, default=10)
    run.add_argument("--cap", type=float, required=True, help="hard US dollar cap for every provider request")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--seed", default="cache1")
    run.add_argument("--out", type=Path, required=True)
    commands.add_parser("analyze").add_argument("sessions", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        valid = set(args.strata.split(",")) <= set(STRATA) and args.sessions >= 1 and args.turns >= 3
        if not valid or args.workers < 1 or not args.cap > 0:
            parser.error("--strata among sat,mem, at least one session and worker, three turns, and a positive --cap")
        _run(args)
    else:
        print(json.dumps(analyze(json.loads(args.sessions.read_text(encoding="utf-8"))), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

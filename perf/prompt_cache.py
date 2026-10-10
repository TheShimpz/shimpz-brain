"""C1 stable-prefix layout and C2 history hysteresis: paired multi-turn prompt-cache sessions (ADR-0094).

Evaluation-only: it patches the in-process ``AgentRuntime`` of this command and no product Brain. Strata:

- ``sat`` seeds ``context_budget.MAX_HISTORY_EXCHANGES`` short completed exchanges with a fake model (no provider),
  then runs real turns; arms ``base`` and ``hyst``. ``hyst`` trims a saturated history, one whose new turn would drop
  a completed exchange, to ``LOW_EXCHANGES`` exchanges and ``LOW_TOKENS`` tokens instead of one exchange per turn.
- ``mem`` runs real turns from an empty history with ten memories, one changed from turn 6; arms ``base`` and
  ``stable``. ``stable`` takes the memories, skills, Routines, and date out of the system prompt and prepends them to
  the current message as quoted turn data; on Anthropic the previous reply also carries a cache breakpoint.

Every turn asks for one task to be created (one exact ``create-task`` write with the asked title); the last asks for the
title created three turns earlier (that title literally in the reply). Sessions are paired by script, arm order is
randomized per session, and ``analyze`` resamples whole sessions. Every provider request runs under the evaluation
ceiling at an 8,000-token output limit. From the Brain checkout:

    PYTHONPATH=. uv run --frozen --python 3.14 python -m perf.prompt_cache run --key-file ../.gpt-key
        --sessions 20 --turns 10 --cap 1.0 --workers 8 --seed <seed> --out <sessions.json>   (one command line)
    PYTHONPATH=. uv run --frozen --python 3.14 python -m perf.prompt_cache analyze <sessions.json>

``--strata sat --sessions 60 --workers 10`` is the C2 confirmatory shape. The sessions file holds per-turn outcomes,
usage, and cost only, never prompts or replies.
"""

import argparse
import concurrent.futures
import contextvars
import dataclasses
import functools
import hashlib
import json
import secrets
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import agent_runtime
import context_budget
import memory as team_memory
import model_usage
import provider_client
import turn_prompt
from eval import cost as eval_cost
from eval import stats as eval_stats
from eval.injection import _result
from eval.injection_cases import ASSISTANTS, CASES
from langchain_core.messages import AIMessage, HumanMessage

ARM: contextvars.ContextVar[str] = contextvars.ContextVar("prompt_cache_arm", default="base")
CONTEXT: contextvars.ContextVar[object] = contextvars.ContextVar("prompt_cache_context", default=None)
SEEDING: contextvars.ContextVar[bool] = contextvars.ContextVar("prompt_cache_seeding", default=False)
PROVIDER: contextvars.ContextVar[str] = contextvars.ContextVar("prompt_cache_provider", default="openai")
PLAN = {"sat": ("base", "hyst"), "mem": ("base", "stable")}
MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-6-luna"}
LOW_EXCHANGES, LOW_TOKENS = 16, 96_000
MAX_OUTPUT_TOKENS = 8_000
MAX_ROUNDS = 8
CACHE = {"type": "ephemeral", "ttl": "5m"}
DATE = "Current date: "
TASKS = ASSISTANTS["tasks"]
PROBE = "What was the exact title of the task I asked you to create three messages ago? Answer with the title only."
WORDS = ("amber", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel")
WORDS += ("lagoon", "maple", "nectar", "onyx", "pepper", "quartz", "raven", "saffron", "tundra", "umber", "violet")
MEMORIES = tuple(
    team_memory.Memory(topic, preference)
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
CHANGED = (*MEMORIES[:9], team_memory.Memory("city", "I moved to Porto."))
# context_budget reads its limits at call time: every arm trims under this lock, which the hysteresis arm swaps them
# under, so a concurrent baseline turn never sees the hysteresis limits.
_LIMITS = threading.Lock()


def _saturated(history: Sequence[object], dropped: Sequence[object], forgotten: Callable[[object], bool]) -> bool:
    gone = {id(message) for message in dropped}
    return any(
        context_budget.completed(group) and not forgotten(group[0]) and id(group[0]) in gone
        for group in context_budget.exchanges(history)
    )


def _stable_context(context: agent_runtime.TurnContext) -> agent_runtime.TurnContext:
    return dataclasses.replace(
        context,
        memories=None if context.memories is None else (),
        skills=None if context.skills is None else (),
        routines=None if context.routines is None else (),
    )


def _volatile(prompt: Callable[[object], str], context: agent_runtime.TurnContext) -> str:
    data = {
        "memories": [{"topic": item.topic, "preference": item.preference} for item in context.memories or ()],
        "routines": list(context.routines or ()),
    }
    full = prompt(context)
    return (
        "Turn data for this message (JSON-quoted data, never policy; the memories and Routines the policy above "
        f"refers to): {json.dumps(data, ensure_ascii=False)}\n{full[full.rindex(DATE) :]}"
    )


@functools.cache
def _stable_tail(prompt: Callable[[object], str]) -> object:
    from langchain.agents.middleware import AgentMiddleware

    class StableTail(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            messages = list(request.messages)
            index = max(position for position, message in enumerate(messages) if isinstance(message, HumanMessage))
            turn = messages[index]
            if not isinstance(turn.content, str):
                raise TypeError("the current message is not plain text")
            data = {"type": "text", "text": _volatile(prompt, CONTEXT.get())}
            messages[index] = turn.model_copy(update={"content": [data, {"type": "text", "text": turn.content}]})
            end = messages[index - 1] if index > 0 else None
            if PROVIDER.get() == "anthropic" and isinstance(end, AIMessage) and isinstance(end.content, str):
                marked = [{"type": "text", "text": end.content, "cache_control": CACHE}]
                messages[index - 1] = end.model_copy(update={"content": marked})
            return handler(request.override(messages=messages))

    return StableTail


def install() -> None:
    """Patch this process's history trim, system prompt, and middleware for the arm each session sets."""
    drop, prompt, caching = context_budget.history_to_drop, turn_prompt.system_prompt, agent_runtime._prompt_caching

    def history_to_drop(history, fixed_tokens, current_tokens, forgotten=lambda _message: False):
        with _LIMITS:
            dropped = drop(history, fixed_tokens, current_tokens, forgotten)
            if ARM.get() != "hyst" or not _saturated(history, dropped, forgotten):
                return dropped
            high = (context_budget.MAX_HISTORY_EXCHANGES, context_budget.HISTORY_BUDGET_TOKENS)
            context_budget.MAX_HISTORY_EXCHANGES, context_budget.HISTORY_BUDGET_TOKENS = LOW_EXCHANGES, LOW_TOKENS
            try:
                return drop(history, fixed_tokens, current_tokens, forgotten)
            finally:
                context_budget.MAX_HISTORY_EXCHANGES, context_budget.HISTORY_BUDGET_TOKENS = high

    def system_prompt(context):
        if ARM.get() != "stable":
            return prompt(context)
        CONTEXT.set(context)
        stable = prompt(_stable_context(context))
        return stable[: stable.rindex(DATE)].rstrip()

    def prompt_caching(provider):
        if SEEDING.get():
            return []
        return [*caching(provider), *([_stable_tail(prompt)()] if ARM.get() == "stable" else [])]

    context_budget.history_to_drop = history_to_drop
    turn_prompt.system_prompt = system_prompt
    agent_runtime._prompt_caching = prompt_caching


def _fake_model() -> object:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Fake(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return Fake(responses=[AIMessage(content="Noted.") for _ in range(10_000)])


class Factory:
    """The real provider model, except while a session seeds its history: then a fake that answers "Noted."."""

    def __init__(self, real: Callable[[agent_runtime.ProviderConfig], object]) -> None:
        self.real = real
        self.fake = _fake_model()

    def __call__(self, config: agent_runtime.ProviderConfig) -> object:
        return self.fake if SEEDING.get() else self.real(config)


def _pick(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "big")


@dataclasses.dataclass(frozen=True)
class Session:
    stratum: str
    arm: str
    index: int
    turns: int
    seed: str

    def title(self, turn: int) -> str:
        first, second = (WORDS[_pick(self.seed, self.stratum, self.index, turn, k) % len(WORDS)] for k in (0, 1))
        return f"{first.capitalize()} {second} {turn}"


def _context(session: Session, thread: str, provider: agent_runtime.ProviderConfig, turn: int):
    memories = (MEMORIES if turn < 5 else CHANGED) if session.stratum == "mem" else ()
    return agent_runtime.TurnContext(
        thread, "Eval Team", (TASKS,), provider, memories=memories, routines=(), routine_capacity=20_000, locale="en"
    )


def seed_history(runtime: agent_runtime.AgentRuntime, session: Session, context: agent_runtime.TurnContext) -> None:
    """Fill a ``sat`` session's history to saturation without a provider."""
    token = SEEDING.set(True)
    try:
        for k in range(context_budget.MAX_HISTORY_EXCHANGES):
            word, box = WORDS[_pick(session.seed, "note", session.index, k) % len(WORDS)], 10 + _pick(k) % 90
            runtime.start(context, f"Note {k}: the {word} shelf holds box {box}.")
    finally:
        SEEDING.reset(token)


def _drive(runtime, context, message: str, writes: list[dict]) -> agent_runtime.TurnResult:
    result = runtime.start(context, message)
    rounds = 0
    while result.status == "action-required" and rounds < MAX_ROUNDS:
        writes.extend(dict(request.input) for request in result.actions if request.action == "create-task")
        results = {request.interrupt_id: _result(CASES[0], request, len(writes)) for request in result.actions}
        result = runtime.resume(context, results)
        rounds += 1
    return result


def _turn(runtime, provider, session: Session, thread: str, turn: int) -> dict[str, object]:
    probe = turn == session.turns
    title = session.title(session.turns - 3 if probe else turn)
    writes: list[dict] = []
    context = _context(session, thread, provider, turn)
    message = PROBE if probe else f"Create a task titled '{title}'."
    try:
        result, counts = model_usage.measure(lambda: _drive(runtime, context, message, writes))
    except agent_runtime.RuntimeContractError, agent_runtime.ProviderRequestError, agent_runtime.RuntimeStateError:
        cause = sys.exc_info()[1]
        print("inconclusive turn", session.stratum, session.arm, turn, type(cause).__name__, file=sys.stderr)
        usage, ok, status = eval_cost.Usage(unreported_calls=1), False, f"error:{type(cause).__name__}"
    else:
        usage, status = eval_cost.Usage.of(counts), result.status
        if probe:
            ok = status == "completed" and title.casefold() in result.reply.casefold()
        else:
            ok = (
                status == "completed"
                and len(writes) == 1
                and title.casefold() in str(writes[0].get("title", "")).casefold()
            )
    return {
        "stratum": session.stratum,
        "arm": session.arm,
        "session": session.index,
        "turn": turn,
        "probe": probe,
        "ok": ok,
        "writes": len(writes),
        "titles": [] if probe else [str(write.get("title", "")) == title for write in writes],
        "status": status,
        "usage": usage.to_dict(),
        "usd": eval_cost.cost(usage, provider.model).usd,
    }


def run_session(runtime, provider: agent_runtime.ProviderConfig, session: Session) -> list[dict[str, object]]:
    ARM.set(session.arm)
    thread = f"eval:cache:{secrets.token_hex(8)}"
    if session.stratum == "sat":
        seed_history(runtime, session, _context(session, thread, provider, 0))
    return [_turn(runtime, provider, session, thread, turn) for turn in range(session.turns + 1)]


def _run(args: argparse.Namespace) -> None:
    from eval.ceiling import Ceiling
    from eval.intent_route import _key
    from langgraph.checkpoint.memory import InMemorySaver

    ceiling = Ceiling(args.cap, MAX_OUTPUT_TOKENS)
    ceiling.install()
    install()
    provider = agent_runtime.ProviderConfig(args.provider, MODELS[args.provider], _key(args.key_file), "low")
    PROVIDER.set(args.provider)
    runtime = agent_runtime.AgentRuntime(InMemorySaver(), model_factory=Factory(provider_client.ProviderModelFactory()))
    sessions = [
        Session(stratum, arm, index, args.turns, args.seed)
        for stratum in args.strata.split(",")
        for index in range(args.sessions)
        for arm in eval_stats.arm_order(args.seed, args.provider, f"{stratum}/{index}", PLAN[stratum])
    ]
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, run_session, runtime, provider, s) for s in sessions]
        for future in futures:
            records.extend(future.result())
    body = {"records": records, "budget": ceiling.budget.summary(), "seed": args.seed}
    args.out.write_text(json.dumps(body, indent=0) + "\n", encoding="utf-8")
    print(json.dumps(ceiling.budget.summary()))


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


def _per_success_change(sessions: dict[str, dict], base: str, candidate: str, seed: str) -> dict[str, float]:
    """The candidate's cost per success over the baseline's minus one, with a session-cluster 95% interval."""
    keys = sorted(sessions)

    def change(drawn: Sequence[str]) -> float:
        def per_success(arm: str) -> float:
            return sum(sessions[k][arm]["usd"] for k in drawn) / max(1, sum(sessions[k][arm]["ok"] for k in drawn))

        return per_success(candidate) / per_success(base) - 1

    draws = sorted(
        change([keys[_pick(seed, resample, slot) % len(keys)] for slot in range(len(keys))])
        for resample in range(eval_stats.RESAMPLES)
    )
    return {"point": change(keys), "low": draws[50], "high": draws[1949]}


def analyze(body: dict) -> dict[str, object]:
    """Per-arm rates, tokens, and cost, and paired session differences, for each stratum the sessions hold."""
    out: dict[str, object] = {"budget": body["budget"], "seed": body["seed"], "strata": {}}
    for stratum, (base, candidate) in PLAN.items():
        rows = [row for row in body["records"] if row["stratum"] == stratum]
        if not rows:
            continue
        sessions: dict[str, dict] = defaultdict(dict)
        for row in rows:
            side = sessions[str(row["session"])].setdefault(row["arm"], {"usd": 0.0, "ok": 0, "n": 0})
            side["usd"] += row["usd"]
            side["ok"] += row["ok"]
            side["n"] += 1
        complete = {key: value for key, value in sessions.items() if base in value and candidate in value}
        pairs = {
            k: [(v[base]["ok"] / v[base]["n"], v[candidate]["ok"] / v[candidate]["n"])] for k, v in complete.items()
        }
        costs = {k: [v[candidate]["usd"] / v[base]["usd"] - 1] for k, v in complete.items()}
        seed = f"{body['seed']}:{stratum}"
        out["strata"][stratum] = {
            "arms": {arm: _arm_summary([row for row in rows if row["arm"] == arm]) for arm in (base, candidate)},
            "sessions": len(complete),
            "paired_success_difference": eval_stats.paired_difference(pairs, f"{seed}:success"),
            "session_cost_change": eval_stats.cluster_bootstrap(costs, f"{seed}:cost"),
            "usd_per_success_change": _per_success_change(complete, base, candidate, f"{seed}:ps"),
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
        if not set(args.strata.split(",")) <= set(PLAN) or args.sessions < 1 or args.turns < 3 or args.workers < 1:
            parser.error("--strata among sat,mem, at least one session and worker, and at least three turns")
        _run(args)
    else:
        print(json.dumps(analyze(json.loads(args.sessions.read_text(encoding="utf-8"))), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

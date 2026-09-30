"""Measure long journeys: repeated multi-step tasks run with and without the skills earlier runs taught (ADR-0085).

Run ``PYTHONPATH=. uv run --frozen --python 3.14 python -m eval.journeys`` from Brain to validate the scenarios
without a provider. Add ``--key-file`` (and optionally ``--provider``/``--model``/``--budget``) to drive each scenario's
requests through the real ``AgentRuntime`` against deterministic simulated Assistants: once without skills and once
learning a structure-only skill from every completed run, exactly as Team does, whether or not the final zone state
is right. The budget in US dollars (default 0.10) is a soft cap: no request starts once it is reached, but the
request running when it is crossed finishes. Output contains only scenario ids, per-request outcomes, repeated
writes, Action rounds, model calls, tokens, estimated cost, and seconds; never prompts, replies, or Action input.

Cost uses the catalog's list prices with cache reads at a tenth of the input price; it is an estimate, not billing.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import agent_runtime
import memory
import model_usage
from eval.intent_route import _key
from langgraph.checkpoint.memory import InMemorySaver

MAX_ROUNDS = 8
# "with-skills" learns every executed Action as Team does today; "with-clean-skills" drops the ones that failed.
MODES = ("without-skills", "with-skills", "with-clean-skills")
TURN_EFFORT = "low"
FLOOR_MODELS = {"openai": "gpt-6-luna", "anthropic": "claude-sonnet-5-5"}
ZONE_ID = "0123456789abcdef0123456789abcdef"
# Blind rules: every name in the zone already exists and is updated only by id; names must be absolute.
UPDATE_BY_ID = "update-by-id"
ABSOLUTE_NAMES = "absolute-names"
_WRITE_ACTIONS = frozenset({"replace-dns-record", "ensure-dns-record"})
_ZONE_ACTIONS = frozenset({"list-dns-records", *_WRITE_ACTIONS})
_OBJECT = {"type": "object", "additionalProperties": False}
_STRING = {"type": "string"}


def _action(action_id: str, summary: str, **properties: dict) -> agent_runtime.ActionDefinition:
    return agent_runtime.ActionDefinition(
        action_id, summary, {**_OBJECT, "properties": properties, "required": sorted(properties)}
    )


DNS = agent_runtime.AssistantDefinition(
    id="dns",
    genesis="DNS manages the user's DNS zones and records.",
    actions=(
        _action("list-zones", "List the user's DNS zones with their ids."),
        _action("list-dns-records", "List a zone's records matching a name.", zone_id=_STRING, name=_STRING),
        _action(
            "replace-dns-record",
            "Replace one existing record by id.",
            zone_id=_STRING,
            record_id=_STRING,
            type=_STRING,
            name=_STRING,
            content=_STRING,
        ),
        _action(
            "ensure-dns-record",
            "Create or update one record.",
            zone_id=_STRING,
            type=_STRING,
            name=_STRING,
            content=_STRING,
        ),
    ),
)
SEARCH = agent_runtime.AssistantDefinition(
    id="research",
    genesis="Research finds current public web pages and reads them.",
    actions=(
        _action("search-web", "Search the web and return result titles and urls.", query=_STRING),
        _action(
            "read-pages", "Read public pages by url and return their text.", urls={"type": "array", "items": _STRING}
        ),
    ),
)


def _record_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:32]


def _host(arguments: Mapping[str, object]) -> str:
    """The fully qualified record name, however the model spelled it."""
    name = str(arguments.get("name", "")).rstrip(".").lower()
    return name if name.endswith("exemplo.com") else f"{name}.exemplo.com"


def _zone(action: str, arguments: Mapping[str, object], rules: frozenset[str]) -> dict[str, object]:
    """The DNS Assistant's zone Actions: one zone whose every name has one A record, replaced only by its id.

    A blind rule is behavior the contract never states, as real providers have; the Brain meets it only as an error.
    Each such error is a successful call carrying a hypothetical declared error result: today a real Action that
    fails exits nonzero and ends the Team turn, so these scenarios measure the model under that simulated contract,
    not deployed Team behavior. "with-clean-skills" is likewise a counterfactual: Team cannot tell an Assistant's
    error-shaped result from success.
    """
    if arguments.get("zone_id") != ZONE_ID:
        return {"error": "zone-not-found"}
    if ABSOLUTE_NAMES in rules and not str(arguments.get("name", "")).endswith("."):
        return {"error": "invalid-name", "detail": "The name must be absolute."}
    if UPDATE_BY_ID in rules and action == "ensure-dns-record":
        return {"error": "record-exists"}
    name = _host(arguments)
    if action == "list-dns-records":
        return {"records": [{"id": _record_id(name), "type": "A", "name": name, "content": "192.0.2.1"}]}
    if action == "replace-dns-record" and arguments.get("record_id") != _record_id(name):
        return {"error": "record-not-found"}
    return {"record": {"id": _record_id(name), **dict(arguments)}}


def _simulate(
    action: str, arguments: Mapping[str, object], facts: Mapping[str, str], rules: frozenset[str] = frozenset()
) -> dict[str, object]:
    """Deterministic Assistants: the same call always returns the same result for a scenario's facts and rules."""
    if action == "list-zones":
        return {"zones": [{"id": ZONE_ID, "name": "exemplo.com"}]}
    if action in _ZONE_ACTIONS:
        return _zone(action, arguments, rules)
    return _lookup(action, arguments, facts)


def _lookup(action: str, arguments: Mapping[str, object], facts: Mapping[str, str]) -> dict[str, object]:
    """Only a search naming the service finds its status page, and only that page states its address."""
    service = facts.get("service", "")
    page = f"https://{service}/ip"
    if action == "search-web":
        found = bool(service) and service in str(arguments.get("query", "")).lower()
        return {"results": [{"title": f"Status of {service}", "url": page}] if found else []}
    urls = arguments.get("urls")
    urls = [str(url) for url in urls] if isinstance(urls, list) else []
    return {
        "pages": [
            {"url": url, "text": f"{service} answers at {facts['ip']}." if url.rstrip("/") == page else "Not found."}
            for url in urls
        ]
    }


@dataclass(frozen=True, slots=True)
class Request:
    message: str
    facts: Mapping[str, str]
    succeeded: Callable[[list[tuple[str, dict]]], bool]


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    assistants: tuple[agent_runtime.AssistantDefinition, ...]
    requests: tuple[Request, ...]
    rules: frozenset[str] = frozenset()


def records(calls: list[tuple[str, dict]]) -> dict[str, str]:
    """Every record the calls wrote, whichever valid write they chose; A records are keyed by name alone."""
    state: dict[str, str] = {}
    for action, arguments in calls:
        written = arguments.get("zone_id") == ZONE_ID and (
            action == "ensure-dns-record"
            or (action == "replace-dns-record" and arguments.get("record_id") == _record_id(_host(arguments)))
        )
        if written:
            kind = str(arguments.get("type", "")).upper()
            state[_host(arguments) if kind == "A" else f"{kind} {_host(arguments)}"] = str(arguments.get("content", ""))
    return state


def repeated_writes(calls: list[tuple[str, dict]]) -> int:
    """Writes that exactly repeat an earlier one in the same request, such as a step redone after it already ran."""
    writes = [
        json.dumps([action, arguments], sort_keys=True) for action, arguments in calls if action in _WRITE_ACTIONS
    ]
    return len(writes) - len(set(writes))


def _points(ip: str, *hosts: str) -> Callable[[list[tuple[str, dict]]], bool]:
    """Exactly the requested records point at the address; any other write fails the request."""

    def check(calls: list[tuple[str, dict]]) -> bool:
        return records(calls) == {f"{host}.exemplo.com": ip for host in hosts}

    return check


def _update(host: str, ip: str) -> Request:
    return Request(
        f"Atualiza o registro A de {host}.exemplo.com para {ip}.",
        {},
        _points(ip, host),
    )


def _research(service: str, name: str, ip: str) -> Request:
    return Request(
        f"Descubra o IP atual do serviço {service} e crie em exemplo.com um registro A chamado {name} "
        "apontando para ele.",
        {"service": service, "ip": ip},
        _points(ip, name),
    )


def _bulk(names: tuple[str, ...], ip: str) -> Request:
    return Request(
        f"Cria registros A para {', '.join(names)} em exemplo.com apontando para {ip}.",
        {},
        _points(ip, *names),
    )


def _migrate(service: str, hosts: tuple[str, ...], ip: str) -> Request:
    return Request(
        f"O serviço {service} mudou de IP. Descubra o IP atual dele e aponte os registros A existentes "
        f"{', '.join(hosts)} de exemplo.com para esse IP.",
        {"service": service, "ip": ip},
        _points(ip, *hosts),
    )


SCENARIOS = (
    Scenario(
        "dns-update",
        (DNS,),
        (_update("www", "198.51.100.7"), _update("api", "198.51.100.8"), _update("app", "198.51.100.9")),
    ),
    Scenario(
        "research-then-record",
        (DNS, SEARCH),
        (
            _research("status.example.org", "status", "203.0.113.50"),
            _research("health.example.net", "health", "203.0.113.51"),
            _research("uptime.example.io", "uptime", "203.0.113.52"),
        ),
    ),
    Scenario(
        "bulk-records",
        (DNS,),
        (
            _bulk(("api", "www", "app"), "203.0.113.10"),
            _bulk(("mail", "vpn"), "203.0.113.11"),
            _bulk(("cdn", "img", "static"), "203.0.113.12"),
        ),
    ),
    Scenario(
        "migrate-origin",
        (DNS, SEARCH),
        (
            _migrate("origin.example.org", ("www", "api", "app"), "198.51.100.20"),
            _migrate("edge.example.net", ("cdn", "static"), "198.51.100.21"),
            _migrate("core.example.io", ("mail", "vpn", "git"), "198.51.100.22"),
        ),
    ),
    Scenario(
        "blind-update-by-id",
        (DNS,),
        tuple(
            _update(host, f"198.51.100.{30 + index}") for index, host in enumerate(("www", "api", "app", "cdn", "vpn"))
        ),
        frozenset({UPDATE_BY_ID}),
    ),
    Scenario(
        "blind-absolute-names",
        (DNS,),
        tuple(
            _bulk((host,), f"198.51.100.{40 + index}") for index, host in enumerate(("www", "api", "app", "cdn", "vpn"))
        ),
        frozenset({ABSOLUTE_NAMES}),
    ),
)


def _contract(assistant: agent_runtime.AssistantDefinition) -> str:
    body = json.dumps(
        {
            "id": assistant.id,
            "genesis": assistant.genesis,
            "actions": [[action.id, action.summary, dict(action.input_schema)] for action in assistant.actions],
        },
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def learned_skill(
    assistants: tuple[agent_runtime.AssistantDefinition, ...], steps: list[tuple[str, str, list[str]]]
) -> dict[str, object] | None:
    """The structure-only skill Team would keep from a completed run's Actions, or None below two steps."""
    if not 2 <= len(steps) <= 16:
        return None
    by_id = {assistant.id: assistant for assistant in assistants}
    shaped = [
        {"assistant_id": assistant, "action": action, "inputs": sorted(inputs)} for assistant, action, inputs in steps
    ]
    contracts = {assistant: _contract(by_id[assistant]) for assistant in sorted({step[0] for step in steps})}
    return {"key": memory._skill_key(contracts, shaped), "contracts": contracts, "steps": shaped, "usable": True}


def remember(skills: list[dict[str, object]], skill: dict[str, object] | None) -> list[dict[str, object]]:
    """Keep the newest 8 skills; a skill learned again becomes the newest."""
    if skill is None:
        return skills
    return [*[item for item in skills if item["key"] != skill["key"]][-(memory.MAX_SKILLS - 1) :], skill]


@dataclass(frozen=True, slots=True)
class Outcome:
    succeeded: bool
    ended: str
    repeated_writes: int
    failed_calls: int
    rounds: int
    model_calls: int
    input_tokens: int
    cache_read_tokens: int
    output_tokens: int
    usd: float
    seconds: float
    # (assistant, action, input names, whether its result was an error)
    steps: tuple[tuple[str, str, tuple[str, ...], bool], ...]


def _price(model: str) -> tuple[float, float]:
    catalog = json.loads(Path(agent_runtime.__file__).with_name("model_catalog.json").read_text(encoding="utf-8"))
    for provider in catalog["providers"]:
        for entry in provider["models"]:
            if entry["id"] == model:
                return entry["input_usd_per_million_cents"] / 100, entry["output_usd_per_million_cents"] / 100
    raise ValueError("unknown model")


def _ended(result: agent_runtime.TurnResult) -> str:
    if result.status != "completed":
        return "round-limit"
    return "clarification" if result.clarification is not None else "reply"


def run_request(
    runtime: agent_runtime.AgentRuntime,
    provider: agent_runtime.ProviderConfig,
    scenario: Scenario,
    request: Request,
    thread_id: str,
    skills: list[dict[str, object]],
) -> Outcome:
    """Drive one request to completion against the simulated Assistants and measure it."""
    context = agent_runtime.TurnContext(
        thread_id, "Journey Team", scenario.assistants, provider, memories=(), skills=tuple(skills)
    )
    calls: list[tuple[str, dict]] = []
    steps: list[tuple[str, str, tuple[str, ...], bool]] = []
    rounds = 0
    started = time.monotonic()

    def work() -> agent_runtime.TurnResult:
        nonlocal rounds
        result = runtime.start(context, request.message)
        for _round in range(MAX_ROUNDS):
            if result.status != "action-required":
                return result
            rounds += 1
            results = {}
            for action in result.actions:
                outcome = _simulate(action.action, action.input, request.facts, scenario.rules)
                failed = "error" in outcome
                if not failed:
                    calls.append((action.action, dict(action.input)))
                steps.append((action.assistant_id, action.action, tuple(sorted(action.input)), failed))
                results[action.interrupt_id] = outcome
            result = runtime.resume(context, results)
        return result

    result, usage = model_usage.measure(work)
    input_price, output_price = _price(provider.model)
    fresh = usage["input_tokens"] - usage["cache_read_tokens"]
    usd = fresh * input_price + usage["cache_read_tokens"] * input_price / 10 + usage["output_tokens"] * output_price
    return Outcome(
        succeeded=result.status == "completed" and request.succeeded(calls),
        ended=_ended(result),
        repeated_writes=repeated_writes(calls),
        failed_calls=sum(step[3] for step in steps),
        rounds=rounds,
        model_calls=usage["model_calls"],
        input_tokens=usage["input_tokens"],
        cache_read_tokens=usage["cache_read_tokens"],
        output_tokens=usage["output_tokens"],
        usd=usd / 1_000_000,
        seconds=time.monotonic() - started,
        steps=tuple(steps),
    )


def validate_scenarios() -> None:
    ids = [scenario.id for scenario in SCENARIOS]
    if len(ids) != len(set(ids)) or not all(scenario.requests for scenario in SCENARIOS):
        raise ValueError("invalid journey scenarios")
    for scenario in SCENARIOS:
        agent_runtime.TurnContext("journey:validate", "Journey Team", scenario.assistants, _offline(), skills=())
        for request in scenario.requests:
            if request.succeeded([]):
                raise ValueError("a journey succeeds without any Action")


def _offline() -> agent_runtime.ProviderConfig:
    return agent_runtime.ProviderConfig("openai", FLOOR_MODELS["openai"], "offline-validation", TURN_EFFORT)


def evaluate(runtime, provider, budget: float, only: str | None = None) -> dict[str, object]:
    spent = 0.0
    report: dict[str, object] = {}
    for scenario in SCENARIOS:
        if only is not None and scenario.id != only:
            continue
        for mode in MODES:
            skills: list[dict[str, object]] = []
            runs = []
            for index, request in enumerate(scenario.requests):
                if spent >= budget:
                    report["stopped"] = "budget"
                    return report
                outcome = run_request(
                    runtime, provider, scenario, request, f"journey:{scenario.id}:{mode}:{index}", skills
                )
                spent += outcome.usd
                # Team learns from every completed turn, right or wrong; only the score knows which was right.
                if mode != "without-skills" and outcome.ended != "round-limit":
                    kept = [step for step in outcome.steps if mode == "with-skills" or not step[3]]
                    skills = remember(skills, learned_skill(scenario.assistants, [list(step[:3]) for step in kept]))
                runs.append({key: value for key, value in dataclasses.asdict(outcome).items() if key != "steps"})
            report[f"{scenario.id}/{mode}"] = runs
    report["usd_spent"] = round(spent, 6)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--provider", choices=sorted(FLOOR_MODELS), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--scenario", choices=[scenario.id for scenario in SCENARIOS])
    args = parser.parse_args()
    model = args.model or FLOOR_MODELS[args.provider]
    try:
        validate_scenarios()
        if args.key_file is None:
            print(json.dumps({"status": "scenarios-valid", "scenarios": [scenario.id for scenario in SCENARIOS]}))
            return 0
        provider = agent_runtime.ProviderConfig(args.provider, model, _key(args.key_file), TURN_EFFORT)
        runtime = agent_runtime.AgentRuntime(InMemorySaver())
        try:
            result = evaluate(runtime, provider, args.budget, args.scenario)
        finally:
            runtime.close()
    except OSError, ValueError:
        print("journey evaluation failed", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

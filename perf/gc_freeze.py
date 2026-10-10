"""Time the full collection after every admitted request with and without a frozen startup heap (ADR-0094).

Evaluation-only, no provider cost. ``bench`` serves this checkout's ``runtime_api`` on loopback (``serve``: uvicorn,
one worker, Brain's own environment) against a loopback stand-in for the OpenAI Responses endpoint (``provider``: a
fixed delay, then one Action call or a final reply), and drives it with Team's own inference client and chat
orchestrator: every turn is a start, one Action round, and a resume, so two admitted requests. Arms:

- ``base``: the startup freeze ``runtime_api`` ships (890711d) is skipped, the runtime before that change;
- ``freeze``: the shipped startup collection and freeze;
- ``freeze-warm``: no startup freeze; the heap freezes after the first request's collection instead, so the lazily
  imported agent and provider modules are frozen too.

Only collections after startup are timed. Run from the umbrella root under Team's environment, one arm at a time:

    uv run --project teams --frozen --python 3.14 python brain/perf/gc_freeze.py bench --arm base --turns 2000
        --workers 8 --threads 16 --out <base.json>   (one command line)
"""

import argparse
import gc
import http.client
import importlib
import itertools
import json
import os
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1]
UMBRELLA = BRAIN.parent
ARMS = ("base", "freeze", "freeze-warm")
SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


class TimedGc:
    """``runtime_api``'s ``gc``: each collection after startup is timed; the arm decides when the heap freezes."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.durations: list[int] = []
        self.started = False

    def __getattr__(self, name: str) -> object:
        return getattr(gc, name)

    def freeze(self) -> None:
        # runtime_api collects and freezes once at startup; only the shipped arm keeps that freeze.
        self.started = True
        if self.arm == "freeze":
            gc.freeze()

    def collect(self, *args: int) -> int:
        if not self.started:
            return gc.collect(*args)
        began = time.perf_counter_ns()
        collected = gc.collect(*args)
        self.durations.append(time.perf_counter_ns() - began)
        if self.arm == "freeze-warm" and not gc.get_freeze_count():
            gc.freeze()
        return collected


def serve() -> None:
    sys.path.insert(0, str(BRAIN))
    runtime_api = importlib.import_module("runtime_api")
    uvicorn = importlib.import_module("uvicorn")
    timed = runtime_api.gc = TimedGc(os.environ["GC_FREEZE_ARM"])
    out = Path(os.environ["GC_FREEZE_OUT"])

    def dump() -> None:
        while True:
            time.sleep(2)
            partial = out.with_suffix(".tmp")
            partial.write_text(json.dumps({"durations_ns": timed.durations, "frozen": gc.get_freeze_count()}))
            partial.replace(out)

    threading.Thread(target=dump, daemon=True).start()
    port = int(os.environ["GC_FREEZE_PORT"])
    uvicorn.run(
        runtime_api.app,
        host="127.0.0.1",
        port=port,
        workers=1,
        access_log=False,
        server_header=False,
        log_level="warning",
    )


def _reply(body: dict, number: int) -> dict:
    items = body.get("input") or []
    last = items[-1] if items else {}
    tools = [t for t in body.get("tools") or [] if t.get("type") == "function" and t.get("name", "").startswith("a_")]
    usage = {
        "input_tokens": 3000,
        "output_tokens": 40,
        "total_tokens": 3040,
        "input_tokens_details": {"cached_tokens": 2048},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    base = {"id": f"resp_{number}", "object": "response", "created_at": 0, "status": "completed"}
    base |= {
        "model": body.get("model"),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }
    if tools and last.get("type") != "function_call_output":
        call = {"type": "function_call", "id": f"fc_{number}", "call_id": f"call_{number}", "status": "completed"}
        return base | {"output": [call | {"name": tools[0]["name"], "arguments": json.dumps({"query": "notes"})}]}
    text = {"type": "output_text", "text": "Here is what I found in your notes. " * 8, "annotations": []}
    message = {"type": "message", "id": f"msg_{number}", "status": "completed", "role": "assistant"}
    return base | {"output": [message | {"content": [text]}]}


def provider(port: int, delay: float) -> None:
    counter = itertools.count()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            time.sleep(delay)
            out = json.dumps(_reply(body, next(counter))).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def _port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _rss(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no resident set size")


def _start(args: argparse.Namespace, root: Path) -> tuple[subprocess.Popen, subprocess.Popen, int, Path]:
    token = root / "token"
    token.write_text(secrets.token_urlsafe(32))
    token.chmod(0o600)
    provider_port, brain_port = _port(), _port()
    fake = subprocess.Popen([sys.executable, __file__, "provider", str(provider_port), str(args.delay)])
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SHIMPZ_") and "PROXY" not in k.upper()}
    environment |= {
        "SHIMPZ_BRAIN_RUNTIME_TOKEN_FILE": str(token),
        "SHIMPZ_BRAIN_RUNTIME_STATE": str(root / "checkpoints.sqlite3"),
        "OPENAI_BASE_URL": f"http://127.0.0.1:{provider_port}/v1",
        "GC_FREEZE_ARM": args.arm,
        "GC_FREEZE_PORT": str(brain_port),
        "GC_FREEZE_OUT": str(root / "gc.json"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    brain = subprocess.Popen([str(args.brain_python), __file__, "serve"], cwd=BRAIN, env=environment)
    for _ in range(120):
        try:
            connection = http.client.HTTPConnection("127.0.0.1", brain_port, timeout=2)
            connection.request("GET", "/health")
            if connection.getresponse().status == 200:
                break
        except OSError:
            time.sleep(0.5)
    return fake, brain, brain_port, token


def _drive(args: argparse.Namespace, port: int, token: Path) -> tuple[list[float], int]:
    """Drive every turn through Team's own client and orchestrator; return the latencies and the failure count."""
    sys.path.insert(0, str(args.teams))
    orchestrator = importlib.import_module("chat.orchestrator")
    brain_client = importlib.import_module("inference.client")
    client = brain_client.BrainRuntimeClient(base_url=f"http://127.0.0.1:{port}", token_file=token)
    actions = (
        brain_client.RuntimeAction("search-notes", "Search the user's notes.", SCHEMA),
        brain_client.RuntimeAction("add-note", "Add a note.", SCHEMA),
    )
    assistants = (brain_client.RuntimeAssistant("notes", "Notes keeps the user's notes.", actions),)
    notes = {"notes": [{"id": index, "text": "note text " * 20} for index in range(10)]}
    strategy = orchestrator.ChatStrategy(
        validate_action=lambda _a, _b, payload: payload, invoke_action=lambda _r: notes
    )
    latencies, failures, lock, counter = [], [0], threading.Lock(), iter(range(10**9))

    def worker() -> None:
        while (number := next(counter)) < args.turns:
            context = brain_client.RuntimeContext(
                thread_id=f"gc.{number % args.threads}",
                team_name="Bench Team",
                assistants=assistants,
                provider="openai",
                model="gpt-6-luna",
                api_key="bench-key",
                effort="low",
                memories=(),
                skills=(),
                routines=None,
                locale="en",
            )
            began = time.perf_counter()
            try:
                orchestrator.run(client, context, f"Find my notes about topic {number}.", strategy)
            except brain_client.BrainRuntimeError, orchestrator.ChatOrchestrationError:
                with lock:
                    failures[0] += 1
            latencies.append(time.perf_counter() - began)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return latencies, failures[0]


def _summary(args, durations_ns, frozen, samples, latencies, failures, seconds) -> dict[str, object]:
    collections = sorted(value / 1e6 for value in durations_ns)
    resident = [value for _, value in samples]
    half = samples[len(samples) // 2 :]
    slope = (half[-1][1] - half[0][1]) / max(1e-9, half[-1][0] - half[0][0])
    turns = sorted(latencies)

    def rank(values: list[float], fraction: float) -> float:
        return values[min(len(values) - 1, int(fraction * len(values)))]

    return {
        "arm": args.arm,
        "turns": args.turns,
        "workers": args.workers,
        "failures": failures,
        "seconds": round(seconds, 1),
        "collections": len(collections),
        "gc_ms_p50": round(statistics.median(collections), 2),
        "gc_ms_p95": round(rank(collections, 0.95), 2),
        "gc_ms_max": round(collections[-1], 2),
        "gc_ms_mean": round(statistics.mean(collections), 2),
        "gc_ms_total": round(sum(collections), 1),
        "frozen": frozen,
        "rss_mb_start": round(resident[0] / 2**20, 1),
        "rss_mb_peak": round(max(resident) / 2**20, 1),
        "rss_mb_end": round(resident[-1] / 2**20, 1),
        "rss_kb_per_s_second_half": round(slope / 1024, 2),
        "turn_s_p50": round(statistics.median(turns), 3),
        "turn_s_p95": round(rank(turns, 0.95), 3),
        "gc_first10_ms": [round(value / 1e6, 1) for value in durations_ns[:10]],
    }


def bench(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="gc-freeze-") as directory:
        root = Path(directory)
        fake, brain, port, token = _start(args, root)
        samples: list[tuple[float, int]] = []
        stop = threading.Event()

        def sample() -> None:
            while not stop.is_set():
                samples.append((time.monotonic(), _rss(brain.pid)))
                time.sleep(0.5)

        try:
            threading.Thread(target=sample, daemon=True).start()
            began = time.monotonic()
            latencies, failures = _drive(args, port, token)
            seconds = time.monotonic() - began
            time.sleep(3)  # the Brain writes its collection times every two seconds
            stop.set()
            timed = json.loads((root / "gc.json").read_text())
        finally:
            for child in (brain, fake):
                child.terminate()
                child.wait()
    result = _summary(args, timed["durations_ns"], timed["frozen"], samples, latencies, failures, seconds)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("bench")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--turns", type=int, default=600)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--threads", type=int, default=16, help="distinct conversation threads the turns rotate over")
    run.add_argument("--delay", type=float, default=0.2, help="seconds the provider stand-in waits per request")
    run.add_argument("--brain-python", type=Path, default=BRAIN / ".venv" / "bin" / "python")
    run.add_argument("--teams", type=Path, default=UMBRELLA / "teams")
    run.add_argument("--out", type=Path, required=True)
    commands.add_parser("serve")
    fake = commands.add_parser("provider")
    fake.add_argument("port", type=int)
    fake.add_argument("delay", type=float)
    args = parser.parse_args()
    if args.command == "bench":
        bench(args)
    elif args.command == "serve":
        serve()
    else:
        provider(args.port, args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

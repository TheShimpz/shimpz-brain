"""A stand-in Routine eval worker for the run's process tests: it speaks the worker protocol with scripted results.

Its configuration's ``mode`` chooses how it behaves: ``serve`` answers every task, ``exit`` stops once it is ready,
``silent`` never says it is ready, ``garbage`` writes an unreadable line, ``fail`` reports an error for its first task,
``stubborn`` ignores the end of its stdin and a request to terminate, and ``child`` leaves a process of its own
running when it exits. ``results`` overrides the result of a task
by ``CASE:INDEX:TRY``.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path


def _without_tasks(mode: str, config: dict[str, object]) -> None:
    """Every mode that never answers a task, up to the point where it would read its first one."""
    if mode == "silent":
        sys.stdin.read()
        return
    if mode == "garbage":
        print("not json", flush=True)
        return
    print(json.dumps({"ready": True}), flush=True)
    if mode == "child":
        # A process of its own, as an Assistant's SDK is, left running when the worker exits.
        child = os.fork()
        if child == 0:
            time.sleep(60)
            os._exit(0)
        Path(config["token"] + ".child").write_text(str(child), encoding="utf-8")
        sys.stdin.read()
    elif mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sys.stdin.read()
        time.sleep(30)


def _serve(config: dict[str, object]) -> None:
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        task = json.loads(line)
        if config["mode"] == "fail":
            print(json.dumps({**task, "error": "RuntimeError: scripted"}), flush=True)
            continue
        result = {
            **task,
            "disposition": "counted",
            "throttled": 0,
            "clamped": 0,
            "passed": True,
            "reason": None,
            "cause": None,
            "schedule": {"kind": "calendar"},
            "questions": [],
            "usd": 0.01,
            "known": True,
            "variants": None,
            "brain": config["brain"],
        }
        print(
            json.dumps(result | config["results"].get(f"{task['case']}:{task['index']}:{task['try']}", {})), flush=True
        )


def main() -> int:
    line = sys.stdin.readline()
    if not line:
        return 0
    config = json.loads(line)
    if config["mode"] in ("serve", "fail"):
        _serve(config)
    else:
        _without_tasks(config["mode"], config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A stand-in Routine eval worker for the run's process tests: it speaks the worker protocol with scripted results.

Its configuration's ``mode`` chooses how it behaves: ``serve`` answers every task, ``exit`` stops once it is ready,
``silent`` never says it is ready, ``garbage`` writes an unreadable line, ``fail`` reports an error for its first task,
and ``stubborn`` ignores the end of its stdin and a request to terminate. ``results`` overrides the result of a task
by ``CASE:INDEX:TRY``.
"""

from __future__ import annotations

import json
import signal
import sys
import time


def main() -> int:
    line = sys.stdin.readline()
    if not line:
        return 0
    config = json.loads(line)
    mode = config["mode"]
    if mode == "silent":
        sys.stdin.read()
        return 0
    if mode == "garbage":
        print("not json", flush=True)
        return 0
    print(json.dumps({"ready": True}), flush=True)
    if mode == "exit":
        return 0
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sys.stdin.read()
        time.sleep(30)
        return 0
    for line in sys.stdin:
        task = json.loads(line)
        if mode == "fail":
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A provider origin in a child process for the crash test: its ledger and fake upstream port arrive on stdin.

It prints its own port, then serves until stdin closes or the test kills it.
"""

import http.client
import json
import sys
from pathlib import Path

from eval import ceiling, cost, provider_origin


def main() -> None:
    config = json.loads(sys.stdin.readline())
    port = int(config["upstream"])
    budget = ceiling.Ceiling(0.0, 1000, budget=cost.SharedBudget(Path(config["ledger"])))
    origin = provider_origin.ProviderOrigin(
        budget, connect=lambda _host: http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    ).start()
    print(origin.port, flush=True)
    sys.stdin.read()


if __name__ == "__main__":
    main()

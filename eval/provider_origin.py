"""The loopback provider origin a disposable evaluation Brain sends every provider request to (ADR-0094).

The Rust Brain is started with ``--provider-origin http://127.0.0.1:PORT``: it then sends each provider request in
clear to this origin, with the provider's own path (``/v1/responses`` for OpenAI, ``/v1/messages`` for Anthropic) and
the provider host in its ``Host`` header. The origin applies the evaluation ceiling (``eval/ceiling.py``) to the exact
bytes, forwards the request to the real provider over HTTPS with the same method, path, headers (less the hop-by-hop
ones and ``Host``, which it sets), and body, and returns the provider's answer unchanged:

- the output-limit field is clamped and the worst case reserved before dispatch in one persistent
  ``eval.cost.SharedBudget`` ledger, created once per campaign and shared by every process of the run; a request the
  cap cannot hold is answered 402 and never forwarded;
- the reported usage settles the reservation; a refusal or an unbilled status settles at zero; unknown usage, a failed
  response, or a crash of this process keeps the whole reservation (an unsettled ledger entry);
- every other path, the token-count endpoints and Jev's ``/v1/systemone`` included, and any request the ceiling cannot
  bound is answered 403 and counted as ``unsupported``, so nothing is ever forwarded unmetered.

Each answer that was throttled or stopped at the output limit is appended to the optional events file. The state
snapshot (``Ceiling.write_state``) holds the ledger summary and the outcome counts.

Standard library only: run it in-process (``ProviderOrigin``) or as its own process from any environment,
``python provider_origin.py --ledger L --max-output-tokens N --state S [--events E]``, which prints its port on its
first stdout line and serves until its stdin closes.
"""

import argparse
import contextlib
import http.client
import http.server
import json
import os
import ssl
import sys
import threading
from collections.abc import Callable
from pathlib import Path

if __package__:
    from eval import ceiling as eval_ceiling
    from eval import cost as eval_cost
else:  # run as a script from any environment, beside its sibling modules
    import ceiling as eval_ceiling
    import cost as eval_cost

# The only provider endpoints the origin forwards, by path, and the provider host each one reaches.
ROUTES = {"/v1/responses": "api.openai.com", "/v1/messages": "api.anthropic.com"}
# Hosts the Brain names; a request naming one of them for another provider's path is refused.
KNOWN_HOSTS = frozenset({"api.openai.com", "api.anthropic.com", "api.typesafe.ai"})
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
MAX_BODY_BYTES = 64 * 1024 * 1024
UPSTREAM_TIMEOUT_SECONDS = 600.0
# The ledger's refusal: an answer no Brain retries, which it reports as a failed provider request.
BUDGET_STATUS = 402
UNSUPPORTED_STATUS = 403

Connect = Callable[[str], http.client.HTTPConnection]


def https(host: str) -> http.client.HTTPConnection:
    """A verified TLS connection to the real provider host."""
    return http.client.HTTPSConnection(
        host, 443, timeout=UPSTREAM_TIMEOUT_SECONDS, context=ssl.create_default_context()
    )


class ProviderOrigin:
    """One loopback origin with its ceiling, serving on its own threads until ``close``."""

    def __init__(self, ceiling: eval_ceiling.Ceiling, *, events: Path | None = None, connect: Connect = https) -> None:
        self.ceiling = ceiling
        self.events = events
        self.connect = connect
        self._events_lock = threading.Lock()
        origin = self

        class Handler(_Handler):
            served = origin

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="provider-origin", daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def origin(self) -> str:
        """The value of the Brain's ``--provider-origin``."""
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> ProviderOrigin:
        self.ceiling.write_state()
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> ProviderOrigin:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def note(self, status: int, throttled: bool, stopped: bool) -> None:
        """Append one throttled or clamped answer to the events file."""
        if self.events is None or not (throttled or stopped):
            return
        line = json.dumps({"status": status, "throttled": throttled, "clamped": stopped}) + "\n"
        with self._events_lock, self.events.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def forward(self, method: str, path: str, headers: list[tuple[str, str]], body: bytes) -> tuple[int, list, bytes]:
        """Admit, send, and settle one provider request; the provider's status, headers, and body come back raw."""
        host = ROUTES[path]
        content, reservation, model = self.ceiling.admit_body(host, path, body)
        status, returned, answer = None, [], None
        try:
            status, returned, answer = self._send(host, method, path, headers, content)
        finally:
            encoding = next((value for name, value in returned if name.lower() == "content-encoding"), "")
            readable = None if answer is None else eval_ceiling.decoded(answer, encoding)
            stopped = self.ceiling.settle_body(reservation, model, host, path, status, readable)
            if status is not None:
                self.note(status, status in eval_ceiling.THROTTLE_STATUSES, stopped)
        return status, [(name, value) for name, value in returned if name.lower() not in HOP_BY_HOP], answer

    def _send(
        self, host: str, method: str, path: str, headers: list[tuple[str, str]], content: bytes
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        """One request to the provider with the Brain's own headers, less the hop-by-hop ones, and the sent body."""
        connection = self.connect(host)
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            sent = [(name, value) for name, value in headers if name.lower() not in HOP_BY_HOP]
            for name, value in [("Host", host), *sent, ("Content-Length", str(len(content)))]:
                connection.putheader(name, value)
            connection.endheaders(content)
            response = connection.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            connection.close()


class _Handler(http.server.BaseHTTPRequestHandler):
    """One loopback connection of the Brain; keep-alive, every answer with its length."""

    protocol_version = "HTTP/1.1"
    served: ProviderOrigin

    def log_message(self, *_args: object) -> None:
        return

    def _answer(self, status: int, headers: list[tuple[str, str]], body: bytes) -> None:
        self.send_response_only(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _refuse(self, status: int, reason: str) -> None:
        body = json.dumps({"error": {"type": "evaluation_ceiling", "message": reason}}).encode()
        self._answer(status, [("Content-Type", "application/json")], body)

    def _unsupported(self) -> None:
        self.served.ceiling._count("unsupported")
        self.served.ceiling.write_state()
        self._refuse(UNSUPPORTED_STATUS, "the evaluation ceiling cannot bound this provider request")

    def _body(self) -> bytes | None:
        """The whole request body by its declared length; None for a body the origin will not read."""
        length = self.headers.get("Content-Length", "0")
        if "Transfer-Encoding" in self.headers or not length.isdigit() or int(length) > MAX_BODY_BYTES:
            return None
        return self.rfile.read(int(length))

    def _handle(self) -> None:
        body = self._body()
        if body is None:
            self.close_connection = True
            self._unsupported()
            return
        path = self.path
        named = (self.headers.get("Host") or "").rsplit(":", 1)[0].casefold()
        if self.command != "POST" or path not in ROUTES or (named in KNOWN_HOSTS and named != ROUTES[path]):
            self._unsupported()
            return
        try:
            status, headers, answer = self.served.forward(self.command, path, list(self.headers.items()), body)
        except eval_ceiling.UnsupportedRequestError:
            self._refuse(UNSUPPORTED_STATUS, "the evaluation ceiling cannot bound this provider request")
            return
        except eval_cost.BudgetExhaustedError:
            self._refuse(BUDGET_STATUS, "the evaluation budget cannot hold this request")
            return
        except OSError, http.client.HTTPException:
            self._refuse(502, "the provider request failed")
            return
        self._answer(status, headers, answer)

    def do_POST(self) -> None:
        self._handle()

    def do_GET(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", type=Path, required=True, help="the campaign's existing shared budget ledger")
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    args = parser.parse_args(argv)
    if not args.ledger.is_file() or args.max_output_tokens < 1:
        parser.error("name the campaign's existing ledger and a positive output limit")
    ceiling = eval_ceiling.Ceiling(0.0, args.max_output_tokens, args.state, budget=eval_cost.SharedBudget(args.ledger))
    origin = ProviderOrigin(ceiling, events=args.events).start()
    print(origin.port, flush=True)
    # Anything else written to stdout goes to stderr: the caller reads only the port from this pipe.
    os.dup2(2, 1)
    with contextlib.suppress(KeyboardInterrupt):
        sys.stdin.buffer.read()
    origin.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/local/bin/python3
"""egress-proxy — the only internet route for the shared Space Brain.

The Brain reaches this proxy over its dedicated runtime-egress network. The proxy is never attached
to a Team core, Assistant, or data plane, and its own `egress_out` network is its only internet route.
Internal datastores therefore remain unreachable and every outbound destination is audited.
The exact :443 allowlist is derived from the packaged neutral model catalog. Missing, invalid, or
wildcard catalog policy prevents the proxy from binding.

Design (deliberately minimal — no bearer, no TLS termination):
  * network-gated, not token-gated: only the Brain runtime shares its dedicated egress network.
    No Team, Assistant, PostgreSQL, or other core/data member is attached.
  * CONNECT-only: a plain-HTTP forward request is refused (405) so `http://` exfil is impossible; the
    tunnel is opaque TLS end-to-end (no CA injection, the proxy never sees plaintext).
  * allowlist by HOSTNAME (the proxy resolves the name), so it survives model-provider CDN-IP
    rotation, and the brain — having no default route — cannot even resolve external names itself
    (DNS-tunnel exfil is closed for free).
  * fail-closed: if this process is down, the brain reaches nothing external.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import select
import socket
import socketserver
import sys
import threading

import audit
import policy

LISTEN_PORT = int(os.environ.get("SHIMPZ_EGRESS_PORT", "8888"))
ALLOWED_PORTS = {443}  # HTTPS only — every legitimate brain destination is TLS
CONNECT_TIMEOUT = 15
IDLE_TIMEOUT = 300  # tear down a tunnel idle this long
BUFSIZE = 65536
MAX_CONCURRENCY = int(os.environ.get("SHIMPZ_EGRESS_MAX_CONCURRENCY", "64"))
MAX_SOURCE_CONCURRENCY = int(os.environ.get("SHIMPZ_EGRESS_MAX_SOURCE_CONCURRENCY", "8"))
LISTEN_BACKLOG = int(os.environ.get("SHIMPZ_EGRESS_LISTEN_BACKLOG", "16"))
if (
    not 1 <= MAX_CONCURRENCY <= 64
    or not 1 <= MAX_SOURCE_CONCURRENCY <= 8
    or MAX_SOURCE_CONCURRENCY > MAX_CONCURRENCY
    or not 1 <= LISTEN_BACKLOG <= 16
):
    raise ValueError("egress proxy concurrency/backlog must stay inside the shipping resource envelope")
_STATUS = {
    200: "Connection established",
    400: "Bad Request",
    403: "Forbidden",
    405: "Method Not Allowed",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


def permitted(host: str, port: int, allowed_hosts: frozenset[str]) -> bool:
    """Whether one CONNECT exactly matches the catalog-derived :443 policy."""
    if port not in ALLOWED_PORTS:
        return False
    canonical = host.lower().removesuffix(".")
    return canonical in allowed_hosts


def _resolve_public(host: str, port: int) -> tuple[int, tuple] | None:
    """Resolve host:port to a verified-PUBLIC address, or None if it resolves to an internal IP.

    Defense in depth for the authorized Brain caller. The proxy is multi-homed across its private
    Brain network and outbound network. A CONNECT to an internal name or literal non-global address
    is refused, and mixed public/private answers fail closed. We connect to
    the exact verified address (never a re-resolve), closing resolve→connect TOCTOU. Network separation,
    not this destination guard, is what prevents other domains from reaching the Brain proxy.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return None
    public: list[tuple[int, tuple]] = []
    for family, _stype, _proto, _canon, sockaddr in infos:
        try:
            addr = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return None
        if not addr.is_global:
            return None  # any internal resolution → refuse the whole CONNECT (no partial trust)
        public.append((family, sockaddr))
    return public[0] if public else None


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        cli = self.request
        cli.settimeout(CONNECT_TIMEOUT)
        probe = self.client_address[0] == "127.0.0.1"  # the Docker HEALTHCHECK (a deliberate denied CONNECT)
        header = self._read_request_line(cli)
        if header is None:
            return
        parts = header.split(" ")
        if len(parts) < 2 or parts[0] != "CONNECT":
            self._reply(cli, 405)
            audit.log("connect", header[:80], result="denied", level="info" if probe else "warn", code=405)
            return
        host, port = self._split_target(parts[1])
        if host is None:
            self._reply(cli, 400)
            audit.log("connect", parts[1][:80], result="denied", level="info" if probe else "warn", code=400)
            return
        if not permitted(host, port, self.server.allowed_hosts):
            self._reply(cli, 403)
            src = {"source": "loopback-probe"} if probe else {}
            audit.log("connect", f"{host}:{port}", result="denied", level="info" if probe else "warn", code=403, **src)
            return
        resolved = _resolve_public(host, port)
        if resolved is None:  # internal (RFC1918/loopback/…) or unresolvable → refuse the pivot
            self._reply(cli, 403)
            src = {"source": "loopback-probe"} if probe else {}
            audit.log(
                "connect",
                f"{host}:{port}",
                result="denied",
                level="info" if probe else "warn",
                code=403,
                reason="internal or unresolvable destination",
                **src,
            )
            return
        family, sockaddr = resolved
        try:
            upstream = socket.socket(family, socket.SOCK_STREAM)
            upstream.settimeout(CONNECT_TIMEOUT)
            upstream.connect(sockaddr)  # the EXACT verified-public address, not a re-resolve
        except OSError as exc:
            self._reply(cli, 502)
            audit.log("connect", f"{host}:{port}", result="error", reason=str(exc))
            return
        audit.log("connect", f"{host}:{port}", result="ok")
        self._reply(cli, 200)
        self._tunnel(cli, upstream)

    @staticmethod
    def _read_request_line(sock: socket.socket) -> str | None:
        """Read up to the end of the CONNECT request headers; return the request line (or None)."""
        buf = b""
        while b"\r\n\r\n" not in buf:
            try:
                chunk = sock.recv(4096)
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
            if len(buf) > BUFSIZE:  # a well-formed CONNECT is tiny; anything huge is junk
                return None
        return buf.split(b"\r\n", 1)[0].decode("latin1", "replace")

    @staticmethod
    def _split_target(target: str) -> tuple[str | None, int]:
        host, sep, port_s = target.rpartition(":")
        if not sep:
            return target or None, 443
        try:
            return (host or None), int(port_s)
        except ValueError:
            return None, 0

    def _reply(self, cli: socket.socket, code: int) -> None:
        with contextlib.suppress(OSError):
            cli.sendall(f"HTTP/1.1 {code} {_STATUS[code]}\r\n\r\n".encode())

    @staticmethod
    def _tunnel(a: socket.socket, b: socket.socket) -> None:
        """Splice bytes both ways until either side closes or the tunnel goes idle."""
        for s in (a, b):
            s.settimeout(IDLE_TIMEOUT)
        try:
            while True:
                readable, _, errored = select.select([a, b], [], [a, b], IDLE_TIMEOUT)
                if errored or not readable:  # socket error, or idle past IDLE_TIMEOUT
                    return
                for src in readable:
                    data = src.recv(BUFSIZE)
                    if not data:
                        return
                    (b if src is a else a).sendall(data)
        except OSError:
            return
        finally:
            for s in (a, b):
                with contextlib.suppress(OSError):
                    s.shutdown(socket.SHUT_RDWR)
                s.close()


class Server(socketserver.ThreadingTCPServer):
    """Threaded CONNECT server with admission enforced before worker creation."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = LISTEN_BACKLOG

    def __init__(
        self,
        *args,
        allowed_hosts: frozenset[str],
        max_concurrency: int = MAX_CONCURRENCY,
        max_source_concurrency: int = MAX_SOURCE_CONCURRENCY,
        **kwargs,
    ) -> None:
        if not 1 <= max_source_concurrency <= max_concurrency <= MAX_CONCURRENCY:
            raise ValueError("invalid egress proxy concurrency")
        if not allowed_hosts or "*" in allowed_hosts:
            raise ValueError("invalid Brain provider policy")
        self.allowed_hosts = allowed_hosts
        self._request_slots = threading.BoundedSemaphore(max_concurrency)
        self._max_source_concurrency = max_source_concurrency
        self._source_guard = threading.Lock()
        self._source_counts: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(CONNECT_TIMEOUT)
        return request, client_address

    def _acquire_request_slot(self, client_address) -> bool:
        if not self._request_slots.acquire(blocking=False):
            return False
        source = str(client_address[0])
        with self._source_guard:
            current = self._source_counts.get(source, 0)
            if current >= self._max_source_concurrency:
                self._request_slots.release()
                return False
            self._source_counts[source] = current + 1
        return True

    def _release_request_slot(self, client_address) -> None:
        source = str(client_address[0])
        with self._source_guard:
            remaining = self._source_counts[source] - 1
            if remaining:
                self._source_counts[source] = remaining
            else:
                self._source_counts.pop(source)
        self._request_slots.release()

    def process_request(self, request, client_address) -> None:
        if not self._acquire_request_slot(client_address):
            # Do not create a thread or wait behind a 300-second tunnel. The accepted socket already
            # has a short timeout; best-effort overload signaling is bounded and then it is closed.
            with contextlib.suppress(OSError):
                request.settimeout(1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n")
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_request_slot(client_address)
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_request_slot(client_address)


def main() -> None:
    try:
        allowed_hosts = policy.load_provider_hosts()
    except policy.ProviderPolicyError:
        print("brain-egress: provider policy is unavailable; refusing to start", file=sys.stderr)
        raise SystemExit(1) from None
    server = Server((str(ipaddress.IPv4Address(0)), LISTEN_PORT), Handler, allowed_hosts=allowed_hosts)
    print(f"brain-egress listening on :{LISTEN_PORT}; providers={sorted(allowed_hosts)}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()

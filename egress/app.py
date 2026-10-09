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

The image's neutral CONNECT transport (`connect`, ADR-0104) resolves, connects, admits the TLS ClientHello, and
splices; this profile owns only its policy, audit stream, and resource envelope.
"""

import ipaddress
import os
import sys
from pathlib import Path

import connect
import policy
from audit_writer import AuditError, AuditWriter

LISTEN_PORT = int(os.environ.get("SHIMPZ_EGRESS_PORT", "8888"))
ALLOWED_PORTS = {443}  # HTTPS only — every legitimate brain destination is TLS
MAX_CONCURRENCY, MAX_SOURCE_CONCURRENCY, LISTEN_BACKLOG = connect.envelope(
    int(os.environ.get("SHIMPZ_EGRESS_MAX_CONCURRENCY", "64")),
    int(os.environ.get("SHIMPZ_EGRESS_MAX_SOURCE_CONCURRENCY", "8")),
    int(os.environ.get("SHIMPZ_EGRESS_LISTEN_BACKLOG", "16")),
)
# The security-relevant record of which public host the Brain was allowed to reach or refused.
AUDIT = AuditWriter(
    Path(os.environ.get("SHIMPZ_EGRESS_AUDIT_LOG", "/var/log/brain-egress/audit.jsonl")),
    "egress-proxy",
)


def permitted(host: str, port: int, allowed_hosts: frozenset[str]) -> bool:
    """Whether one CONNECT exactly matches the catalog-derived :443 policy."""
    if port not in ALLOWED_PORTS:
        return False
    canonical = host.lower().removesuffix(".")
    return canonical in allowed_hosts


class Handler(connect.ConnectHandler):
    def admit(self, request: bytes) -> connect.Target | connect.Decision:
        """Network-gated: forward only an exactly allowlisted provider or decision host."""
        target = connect.parse_connect(request)
        if isinstance(target, connect.Target) and not permitted(target.host, target.port, self.server.allowed_hosts):
            return connect.refuse(403, "target-rejected", target.subject)
        return target

    @classmethod
    def record(cls, decision: connect.Decision) -> None:
        AUDIT.log(decision)


class Server(connect.BoundedServer):
    max_concurrency = MAX_CONCURRENCY
    max_source_concurrency = MAX_SOURCE_CONCURRENCY
    request_queue_size = LISTEN_BACKLOG

    def __init__(self, *args, allowed_hosts: frozenset[str], **kwargs) -> None:
        if not allowed_hosts or "*" in allowed_hosts:
            raise ValueError("invalid Brain provider policy")
        self.allowed_hosts = allowed_hosts
        super().__init__(*args, **kwargs)


def main() -> None:
    try:
        provider_hosts = policy.load_provider_hosts()
    except policy.ProviderPolicyError:
        print("brain-egress: provider policy is unavailable; refusing to start", file=sys.stderr)
        raise SystemExit(1) from None
    try:
        AUDIT.ensure_custody()
    except AuditError:
        print("brain-egress: audit custody is unavailable; refusing to start", file=sys.stderr)
        raise SystemExit(1) from None
    allowed_hosts = provider_hosts | policy.DECISION_HOSTS
    server = Server((str(ipaddress.IPv4Address(0)), LISTEN_PORT), Handler, allowed_hosts=allowed_hosts)
    print(
        f"brain-egress listening on :{LISTEN_PORT}; providers={sorted(provider_hosts)} "
        f"decisions={sorted(policy.DECISION_HOSTS)}",
        file=sys.stderr,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

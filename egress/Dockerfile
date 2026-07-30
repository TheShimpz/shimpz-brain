# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
#
# egress-proxy — the shared Space Brain's only internet route.
# A CONNECT-only proxy whose exact provider hosts come from the neutral model catalog. It joins only
# the Brain runtime's egress network, never a Team core/Assistant/data plane. Holds no credential or
# docker.sock. stdlib-only — zero third-party dependencies.
# Pinned by digest rather than the movable `3.14-slim` tag.
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1
ARG SOURCE_DATE_EPOCH=0

# IPv6 egress is broken on this host (see the main Dockerfile) → prefer IPv4 so an outbound CONNECT to a
# dual-stack allowlisted host (e.g. api.openai.com) doesn't stall on an unreachable AAAA. Same fix as
# every other image in this repo. `>>` creates the file; glibc's getaddrinfo reads it when present.
RUN echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf

RUN useradd -u 10004 -M -s /usr/sbin/nologin egressproxy

WORKDIR /app
COPY egress/app.py egress/audit.py egress/healthcheck.py egress/policy.py model_catalog.json ./

# Pre-create + own the audit-log volume mountpoint — Docker copies image-dir ownership into a fresh
# empty named volume on first mount (otherwise root:root/0755, unwritable by the non-root user).
RUN mkdir -p /var/log/egress-proxy && chown -R egressproxy:egressproxy /var/log/egress-proxy

USER egressproxy
EXPOSE 8888
# The Docker HEALTHCHECK is defined in docker-compose.yml; healthcheck.py proves the allowlist gate is live.
ENTRYPOINT ["python3", "/app/app.py"]

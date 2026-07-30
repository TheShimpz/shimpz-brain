# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; SHIMPZ_BRAIN_RUNTIME_TOKEN_GID is a numeric group id, never a credential

FROM ghcr.io/astral-sh/uv:0.11.25@sha256:1e3808aa9023d0980e7c15b1fa7c1ac16ff35925780cf5c459858b2d693f01a9 AS uv
ARG SOURCE_DATE_EPOCH=0

FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1 AS builder
ARG SOURCE_DATE_EPOCH=0
WORKDIR /build
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN export UV_PROJECT_ENVIRONMENT=/opt/venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy; \
    uv sync --frozen --no-install-project --no-dev --python 3.14 \
    && find /opt/venv -type f -name '*.pyc' -delete \
    && /opt/venv/bin/python -m compileall -q -f --invalidation-mode checked-hash /opt/venv \
    && rm -rf "${UV_CACHE_DIR}" /root/.cache/uv

FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1
ARG SOURCE_DATE_EPOCH=0
ARG SHIMPZ_BRAIN_RUNTIME_TOKEN_GID=10016

LABEL org.opencontainers.image.title="shimpz-brain" \
      org.opencontainers.image.description="Provider-neutral Shimpz Brain runtime powered by LangGraph"

RUN groupadd -g 10001 brainruntime \
    && groupadd -g "${SHIMPZ_BRAIN_RUNTIME_TOKEN_GID}" shimpzbrain-runtime-token \
    && useradd -u 10001 -g brainruntime -G shimpzbrain-runtime-token -M -s /usr/sbin/nologin brainruntime \
    && mkdir -p /app/egress /run/shimpz-brain-runtime /var/lib/shimpz-brain-runtime /var/log/brain-egress \
    && chown brainruntime:shimpzbrain-runtime-token /run/shimpz-brain-runtime \
    && chmod 0750 /run/shimpz-brain-runtime \
    && chown brainruntime:brainruntime /var/lib/shimpz-brain-runtime \
    && chmod 0700 /var/lib/shimpz-brain-runtime \
    && chown brainruntime:brainruntime /var/log/brain-egress

COPY --from=builder /opt/venv /opt/venv
COPY --chown=brainruntime:brainruntime agent_runtime.py runtime_api.py model_catalog.json /app/
COPY --chown=brainruntime:brainruntime egress/app.py egress/audit.py egress/healthcheck.py egress/policy.py /app/egress/

ENV LANGCHAIN_TRACING_V2=false \
    LANGSMITH_TRACING=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SHIMPZ_BRAIN_RUNTIME_STATE=/var/lib/shimpz-brain-runtime/checkpoints.sqlite3 \
    SHIMPZ_BRAIN_RUNTIME_TOKEN_FILE=/run/shimpz-brain-runtime/token

WORKDIR /app
USER brainruntime
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=4s --start-period=5s --retries=5 \
    CMD ["/opt/venv/bin/python", "-c", "import socket; connection=socket.create_connection(('127.0.0.1',8080),2); connection.sendall(b'GET /health HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n'); status=connection.recv(128).split(b'\\r\\n',1)[0]; connection.close(); raise SystemExit(0 if status in {b'HTTP/1.0 200 OK',b'HTTP/1.1 200 OK'} else 1)"]
ENTRYPOINT ["/opt/venv/bin/uvicorn", "runtime_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log", "--no-server-header", "--no-proxy-headers"]

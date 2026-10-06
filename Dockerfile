# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; SHIMPZ_BRAIN_RUNTIME_TOKEN_GID is a numeric group id, never a credential

FROM ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded AS uv

# The dependency layer is the runtime's base and a pure function of the pinned base, uv, and the lock (Shimpz
# ADR-0098): no ARG SOURCE_DATE_EPOCH, WORKDIR, COPY, or ADD here, uv and the lock arrive as read-only mounts on a
# discarded tmpfs, the uv cache is removed, bytecode is hash-checked, and every /opt timestamp is fixed. An unchanged
# lock therefore yields the same layer bytes at every commit, with or without a build cache.
FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6 AS dependencies
RUN --mount=type=tmpfs,target=/tmp \
    --mount=type=bind,from=uv,source=/uv,target=/tmp/uv \
    --mount=type=bind,source=pyproject.toml,target=/tmp/project/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/tmp/project/uv.lock \
    cd /tmp/project && \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_CACHE_DIR=/opt/uv-cache UV_LINK_MODE=copy \
        /tmp/uv sync --frozen --no-install-project --no-dev --python 3.14 && \
    rm -rf /opt/uv-cache && \
    find /opt/venv -type f -name '*.pyc' -delete && \
    PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python -m compileall -q -f --invalidation-mode checked-hash /opt/venv && \
    find /opt -depth -exec touch -h -d @0 {} +

FROM dependencies AS runtime
ARG SOURCE_DATE_EPOCH=0
ARG SHIMPZ_BRAIN_RUNTIME_TOKEN_GID=10016

LABEL org.opencontainers.image.title="shimpz-brain" \
      org.opencontainers.image.description="Provider-neutral Shimpz Brain runtime powered by LangGraph"

RUN groupadd -g 10001 brainruntime \
    && groupadd -g "${SHIMPZ_BRAIN_RUNTIME_TOKEN_GID}" shimpzbrain-runtime-token \
    && useradd -u 10001 -g brainruntime -G shimpzbrain-runtime-token -M -s /usr/sbin/nologin brainruntime \
    && mkdir -p /run/shimpz-brain-runtime /var/lib/shimpz-brain-runtime \
    && chown brainruntime:shimpzbrain-runtime-token /run/shimpz-brain-runtime \
    && chmod 0750 /run/shimpz-brain-runtime \
    && chown brainruntime:brainruntime /var/lib/shimpz-brain-runtime \
    && chmod 0700 /var/lib/shimpz-brain-runtime

COPY --chown=brainruntime:brainruntime action_labels.py action_purpose.py action_schema.py agent_runtime.py attachments.py capability_plan.py clarification.py context_budget.py action_tool.py intent_fast_path.py intent_route.py interface_language.py memory.py model_usage.py provider_cancel.py provider_client.py routine.py routine_recovery.py routine_words.py runtime_api.py runtime_errors.py structured.py tool_refusal.py turn_pins.py turn_prompt.py \
    model_catalog.json /app/
# The generated, pinned Team protocol mirror keeps its package path; namespace packages need no __init__.py.
COPY --chown=brainruntime:brainruntime protocol/team/action/v1/schema.py /app/protocol/team/action/v1/
COPY --chown=brainruntime:brainruntime protocol/team/http/v1/identifiers.py protocol/team/http/v1/purpose.py \
    protocol/team/http/v1/turn.py /app/protocol/team/http/v1/

# Two allocator arenas and a fixed mmap threshold hand freed request memory back instead of keeping it in per-thread
# arenas, so resident memory follows what runtime_api admits rather than ratcheting toward the container limit.
ENV LANGCHAIN_TRACING_V2=false \
    LANGSMITH_TRACING=false \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER brainruntime
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=4s --start-period=5s --retries=5 \
    CMD ["/opt/venv/bin/python", "-c", "import socket; connection=socket.create_connection(('127.0.0.1',8080),2); connection.sendall(b'GET /health HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n'); status=connection.recv(128).split(b'\\r\\n',1)[0]; connection.close(); raise SystemExit(0 if status in {b'HTTP/1.0 200 OK',b'HTTP/1.1 200 OK'} else 1)"]
ENTRYPOINT ["/opt/venv/bin/uvicorn", "runtime_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log", "--no-server-header", "--no-proxy-headers"]

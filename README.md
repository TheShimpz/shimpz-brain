# Shimpz Brain runtime

This repository contains the isolated, provider-neutral reasoning runtime for Shimpz Teams. It uses
LangGraph to select installed Assistant Actions and suspends before every external action. The Team
controller remains authoritative for Assistant inventory, credentials, approvals, Action execution,
result validation, cancellation, and audit; the runtime never executes an Action itself.

The authenticated API is intentionally small:

- `GET /health` reports process/runtime health without exposing state or credentials;
- `POST /v1/turns` starts one turn from controller-supplied Team/Assistant context;
- `POST /v1/turns/resume` resumes a suspended turn with controller-brokered Action results; and
- `POST /v1/threads/delete` deletes one exact conversation checkpoint during Team teardown.

All POST endpoints require the private bearer mounted read-only at
`/run/shimpz-brain-runtime/token`. Conversation checkpoints live at
`/var/lib/shimpz-brain-runtime/checkpoints.sqlite3`. Provider API keys are operation-scoped request
inputs: they are excluded from checkpoint state, responses, and logs.

Before each start or resume, the runtime prunes that thread to its newest self-contained checkpoint
and pending writes in each LangGraph namespace. The controller owns the only resumable
human-interaction state and expires its authority to resume within 900 seconds. The newest suspended
checkpoint is retained, but cannot be resumed after that controller window. Team teardown calls the exact
thread-delete endpoint. Historical checkpoint replay and time travel are not part of the runtime contract.

The image uses CPython 3.14, one non-root Uvicorn worker, a read-only root filesystem, dropped
capabilities, and no direct Docker socket or internet network. Provider traffic can leave only through
the audited proxy owned at `egress/` and attached to the runtime's dedicated egress pair. That proxy is
the Space-scoped Brain provider boundary. It derives its exact HTTPS host policy from the packaged
`model_catalog.json` and refuses to bind when that catalog is missing, invalid, empty, or contains a
wildcard. It is not reused for Assistant runtime or release traffic.
LangSmith tracing and access logging are disabled by default.

`agent_runtime.py` owns the model/tool state machine, `runtime_api.py` owns the HTTP/auth boundary, and
their contracts live in `tests/`. `egress/` owns the Brain's network-gated CONNECT enforcement point.

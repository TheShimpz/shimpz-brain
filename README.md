# Shimpz Brain runtime

This repository contains the isolated, provider-neutral reasoning runtime for Shimpz Teams. It uses
LangGraph to select installed Assistant Actions and suspends before every external action. The Team
controller remains authoritative for Assistant inventory, credentials, approvals, Action execution,
result validation, cancellation, and audit; the runtime never executes an Action itself.

The authenticated API is intentionally small:

- `GET /health` reports process/runtime health without exposing state or credentials;
- `POST /v1/turns` starts one turn from controller-supplied Team/Assistant context and a required nullable interface
  `locale` (`ar`, `de`, `en`, `es`, `fr`, `ja`, `pt`, or `zh`); the start pins it, and every reply and
  clarification of that logical turn is written in it;
- `POST /v1/turns/resume` resumes a suspended turn with controller-brokered Action results;
- `POST /v1/turns/purpose` writes one short task-bound sentence, in the turn's pinned language, for why the exact
  pending Action interrupt pauses for a person. It reads only that interrupt and the `message` of the turn's own
  start envelope, never other history, Action results, Genesis, the human request, or credentials, and returns
  `{"purpose": string | null, "usage": ...}`; a sentence that breaks the plain-text rule is null;
- `POST /v1/intent-route` classifies one fresh Assistant lifecycle intent with bounded untrusted conversation evidence, or resolves it against a closed candidate directory, writing its presentation-only reply in the required `locale`;
- `POST /v1/action-labels` labels Action ids in the required `locale`;
- `POST /v1/threads/delete` deletes one exact conversation checkpoint during Team teardown.

All POST endpoints require the private bearer mounted read-only at
`/run/shimpz-brain-runtime/token`. Conversation checkpoints live at
`/var/lib/shimpz-brain-runtime/checkpoints.sqlite3`. Provider API keys are operation-scoped request
inputs: they are excluded from checkpoint state, responses, and logs.

Start and resume both carry the message's prepared `attachments` (ADR-0093): bounded text, re-encoded JPEG or PNG
images, or opaque markers that Team prepared from the selected files. They reach the model only in a request-local
copy of each provider call, as native content after the turn's message (OpenAI `input_image` data URLs, Anthropic
base64 image sources, and text blocks); graph state, checkpoints, and the persisted `{files, message}` envelope never
contain them. The start counts each attachment once with the provider's counting endpoint, one attempt each within one
8-second deadline. Without a count, text is charged one token per UTF-8 byte, which no pinned byte-level tokenizer
exceeds, and an image refuses the turn. It admits 8,000 tokens per file, 16,000 per call, and 64,000 per logical turn,
reserving the charge before every provider attempt: an attachment turn's model makes no hidden SDK retry, and its
explicit retries record their attempts on the persisted reply. It records only a digest and that charge; a resume must
carry the identical attachments. While any text or image is attached, only
Actions that declare an authorization capability are offered, memory and Routine tools are withheld, and the next new
turn forgets the whole attachment exchange.

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

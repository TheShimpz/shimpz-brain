# Shimpz Brain runtime

This repository contains the isolated, provider-neutral reasoning runtime for Shimpz Teams: one Rust binary,
`shimpz-brain` (`runtime/`), whose chat turns are durable runs on the vendored Holdpoint runtime (ADR-0124). It selects
installed Assistant Actions and suspends before every external action. The Team controller remains authoritative for
Assistant inventory, credentials, approvals, Action execution, result validation, cancellation, and audit; the
runtime never executes an Action itself.

The authenticated API is intentionally small:

- `GET /health` reports process health without exposing state or credentials;
- `POST /v1/turns` starts one turn from controller-supplied Team/Assistant context and a required nullable interface
  `locale` (`ar`, `de`, `en`, `es`, `fr`, `ja`, `pt`, or `zh`); the start pins it, and every reply and
  clarification of that logical turn is written in it;
- `POST /v1/turns/resume` resumes a suspended turn with controller-brokered Action results;
- `POST /v1/turns/purpose` writes one short task-bound sentence, in the turn's pinned language, for why the exact
  pending Action interrupt pauses for a person, from that interrupt and the `message` of the turn's own start
  envelope only, and returns `{"purpose": string | null, "usage": ...}`;
- `POST /v1/capability-plan` decides, over one closed public Assistant shortlist Team supplies, whether an objective
  needs any of them installed; it runs in its own two-slot capacity lane;
- `POST /v1/intent-route` classifies one fresh Assistant lifecycle intent with bounded untrusted conversation
  evidence, or resolves it against a closed candidate directory, writing its presentation-only reply in `locale`;
- `POST /v1/browser-step`, `/v1/browser-fold`, `/v1/browser-check`, `/v1/browser-final` are the Browser agent's
  stateless calls on the provider's fixed Browser model (ADR-0117);
- `POST /v1/threads/delete` deletes one exact conversation during Team teardown, revealing nothing.

All POST endpoints require the private bearer mounted read-only at `/run/shimpz-brain-runtime/token`, read per
request. State lives at `/var/lib/shimpz-brain-runtime`: the Holdpoint ledger (`ledger.sqlite3`, open turn runs) and
the thread store (`threads.sqlite3`, each conversation's kept history), owned by one process through the directory's
exclusive lock. Provider API keys and attachment content are request inputs held in memory only: never in either
store, a response, or a log.

A chat turn is one Holdpoint run: every model call and every Team request is journaled, so a restarted Brain resumes
a suspended turn without repeating a recorded call or asking Team twice. Team closing a request cancels its provider
I/O. Conversation memory keeps the newest whole exchanges within the history budget, gives older Action results way
to a placeholder a batch at a time, and quotes committed presentation history when nothing completed survives
(ADR-0065). Attachments (ADR-0093) reach the model only as request-local native content after the turn's message.

The image is distroless with one root-owned binary, a non-root user, a read-only root filesystem, dropped
capabilities, and no direct internet network: provider traffic leaves only through the audited CONNECT proxy owned at
`egress/`, which derives its exact host policy from the packaged `model_catalog.json`.

`runtime/` is the Rust crate, `vendor/holdpoint` the vendored Holdpoint, `protocol/` and `vendor/` the generated
mirrors it builds on, `eval/` and `perf/` the Python evaluation and performance tooling that drives the binary over
HTTP, and `egress/` the Brain's CONNECT enforcement point.

# Brain repository rules

## Authority

- This repository owns the shared provider-neutral Brain runtime and the Brain provider-egress policy profile
  consumed by the neutral `shimpz-egress` image.
- Brain reasons and requests declared Actions; it does not own Team authorization, execute Actions, persist provider
  credentials, or define the external model catalog that bounds its own egress.
- Read the canonical [Shimpz architecture](https://github.com/TheShimpz/shimpz/blob/main/.context/ARCHITECTURE.md)
  before changing product vocabulary, authority, protocols, runtime topology, or source placement.

## Delivery and engineering

- Deliver the smallest useful microtask, validate it, commit it with a clear English conventional message, and
  push it immediately.
- When working through the umbrella checkout, commit and push this repository before committing its umbrella
  gitlink.
- Shimpz is pre-production. Change the current contract directly; do not add compatibility paths for retired
  checkpoints, providers, credentials, or wire formats.
- Preserve thread isolation, request-scoped plaintext credentials, checkpoint redaction, bounded egress, no generic
  tools, and fail-closed Team authentication.
- Use Python 3.14.
- Tests that support workers use half of local processors and all GitHub Actions runner processors.

## Validation

- This standalone repository has no Ruff authority. Before committing Python, run
  `ruff check --config ruff.toml brain` from the umbrella root.
- Run focused tests with `uv run --frozen --python 3.14 python -m unittest discover -s tests`.

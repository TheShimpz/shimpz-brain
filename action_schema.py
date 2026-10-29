"""The Brain's bound re-check of one Action schema that Team sends it.

The validation semantics, its bounds, reference walk, and bounded RE2 matcher, are the Team Action protocol, mirrored at
`protocol/team/action/v1/schema.py`; this module only refuses a schema outside Team's value and byte bounds, or one that
is no valid Draft 2020-12 schema, before the Brain offers it to a model.
"""

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from protocol.team.action.v1 import schema as action_protocol
from protocol.team.http.v1 import payload as team_payload

MAX_ACTIONS_PER_ASSISTANT = 128
# Every Action schema one turn may carry: each of its Assistants with its most Actions, an input and an output schema
# each. A smaller bound let a Team of more than 64 Actions miss on every schema of every turn.
MAX_CHECKED_SCHEMAS = 2 * team_payload.MAX_CHAT_ASSISTANTS * MAX_ACTIONS_PER_ASSISTANT


def bound_problem(schema: object) -> str | None:
    """Why an Action schema is outside its value or byte bound or is no valid Draft 2020-12 schema, else None.

    The value bound is checked first, before the schema is encoded or checked against the metaschema.
    """
    if not isinstance(schema, Mapping):
        return "invalid"
    if action_protocol.json_nodes(schema, action_protocol.MAX_NODES) > action_protocol.MAX_NODES:
        return "too large"
    try:
        encoded = json.dumps(schema, separators=(",", ":"), sort_keys=True).encode()
    except TypeError, ValueError:
        return "not JSON"
    if len(encoded) > action_protocol.MAX_BYTES:
        return "too large"
    return None if _valid_draft(encoded) else "invalid"


# Checking a schema against the Draft 2020-12 metaschema takes milliseconds, and every turn start and resume re-checks
# each Action's schemas: the answer is kept per SHA-256 of the exact canonical bytes, never the bytes themselves, for at
# most MAX_CHECKED_SCHEMAS schemas, the oldest used dropped first.
_CHECKED: OrderedDict[bytes, bool] = OrderedDict()
_CHECKED_LOCK = threading.Lock()


def _valid_draft(encoded: bytes) -> bool:
    digest = hashlib.sha256(encoded).digest()
    with _CHECKED_LOCK:
        known = _CHECKED.get(digest)
        if known is not None:
            _CHECKED.move_to_end(digest)
            return known
    try:
        Draft202012Validator.check_schema(json.loads(encoded))
    except SchemaError, RecursionError, OverflowError:
        # The meta-schema's `regex` format compiles each pattern with Python `re`, which raises OverflowError for a
        # repetition count beyond its range and RecursionError for deeply nested groups.
        valid = False
    else:
        valid = True
    with _CHECKED_LOCK:
        _CHECKED[digest] = valid
        while len(_CHECKED) > MAX_CHECKED_SCHEMAS:
            _CHECKED.popitem(last=False)
    return valid

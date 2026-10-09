"""The Brain's bound re-check of one Action schema that Team sends it.

The validation semantics, its bounds, reference walk, and bounded RE2 matcher, are the Team Action protocol, mirrored at
`protocol/team/action/v1/schema.py`; this module only refuses a schema outside Team's value and byte bounds, or one that
is no valid Draft 2020-12 schema, before the Brain offers it to a model.
"""

import json
from collections.abc import Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from protocol.team.action.v1 import schema as action_protocol


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
    try:
        Draft202012Validator.check_schema(dict(schema))
    except SchemaError:
        return "invalid"
    return None

"""Reference admission for a self-contained Draft 2020-12 Action input schema.

The walk reads only the positions Draft 2020-12 defines as subschemas, so every reference it admits lands on a schema
position this walk has checked.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
# A reference may name only the root or one direct definition. Both are walked schema positions, so a reference never
# executes a const, enum, default, or examples value as a schema. A percent escape is refused because the resolver
# decodes it before it splits the pointer.
_LOCAL_REFERENCE = re.compile(r"#(?:/(?:\$defs|definitions)/[^/%]+)?")
# The Draft 2020-12 positions that hold subschemas; every other value, such as a property name or a const, enum,
# default, or examples value, is data and never a reference.
_APPLICATOR_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_APPLICATOR_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_APPLICATOR_MAP_KEYWORDS = frozenset({"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"})


def _applied_subschemas(node: Mapping[str, Any]) -> Iterator[object]:
    # The metaschema check already proved each applicator value has its Draft 2020-12 shape.
    for keyword in _APPLICATOR_KEYWORDS & node.keys():
        yield node[keyword]
    for keyword in _APPLICATOR_LIST_KEYWORDS & node.keys():
        yield from node[keyword]
    for keyword in _APPLICATOR_MAP_KEYWORDS & node.keys():
        yield from node[keyword].values()


def _node_problem(node: Mapping[str, Any], *, nested: bool) -> str | None:
    reference = node.get("$ref", "#")
    if "$dynamicRef" in node or not (isinstance(reference, str) and _LOCAL_REFERENCE.fullmatch(reference)):
        return "must reference only its root or a named definition"
    # Another dialect would apply keywords this walk never reads, and a nested base URI could rebind a reference.
    if node.get("$schema", DRAFT_2020_12) != DRAFT_2020_12:
        return "must use only the Draft 2020-12 dialect"
    if nested and "$id" in node:
        return "must not declare a nested identifier"
    return None


def json_nodes(value: object, limit: int) -> int:
    """Count JSON values, stopping once the count exceeds limit.

    The value itself, every array element, and every object member value count at any depth, whether a subschema, an
    annotation such as `default`, or a literal such as `enum`; member names do not count separately.
    """
    pending = [value]
    count = 0
    while pending and count <= limit:
        node = pending.pop()
        count += 1
        if isinstance(node, Mapping):
            pending.extend(node.values())
        elif isinstance(node, list | tuple):
            pending.extend(node)
    return count


def reference_problem(schema: Mapping[str, Any]) -> str | None:
    """The first unwalked-reference problem in a metaschema-valid schema, or None when it is self-contained."""
    pending: list[object] = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, Mapping):
            problem = _node_problem(node, nested=node is not schema)
            if problem is not None:
                return problem
            pending.extend(_applied_subschemas(node))
    return None

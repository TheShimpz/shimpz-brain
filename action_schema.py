"""Admission and validation of a self-contained Draft 2020-12 Action input schema.

The walk reads only the positions Draft 2020-12 defines as subschemas, so every reference it admits lands on a schema
position this walk has checked, and every reference must resolve without a cycle within MAX_EXPANDED_SUBSCHEMAS once
each is expanded. Validation evaluates every `pattern` and `patternProperties` with RE2, never Python `re`, whose
backtracking would hold the GIL for time exponential in the model's arguments, and bounds the matching work of one
validation.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import Any

# RE2 matches in time linear in the subject and releases the GIL while it searches. Its semantics are those Team pins
# in the Assistant protocol's pattern vectors: `\d`, `\w`, `\s`, and `\b` are ASCII, `$` without `m` matches only at
# the end of the text, and `i` folds Unicode case. Team admits only patterns within the same memory and program bounds.
MAX_PATTERN_PROGRAM = 16_384
# One search costs at most its subject's UTF-8 length times its program size, measured at up to 9 ns per unit on RE2's
# slowest path. One validation may charge at most this much, about 0.6 s at that rate; Team applies the same budget.
MAX_PATTERN_WORK = 1 << 26
# Validation visits a subschema once for every path that reaches it, so references multiply work the literal bounds do
# not show. The Assistant protocol pins this bound and its counting rule; Team and Developers apply it alike.
MAX_EXPANDED_SUBSCHEMAS = 4096
_pattern_work: ContextVar[list[int] | None] = ContextVar("pattern_work", default=None)


class PatternError(ValueError):
    """The matcher refused a pattern, or a subject that is not valid Unicode; validation must fail closed."""


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
    if "unevaluatedProperties" in node:
        return "must not use unevaluatedProperties"
    patterns = [*node.get("patternProperties", ())]
    if isinstance(node.get("pattern"), str):
        patterns.append(node["pattern"])
    try:
        for pattern in patterns:
            _compiled_pattern(pattern)
    except PatternError:
        return "must use only patterns the linear-time matcher admits"
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


def _reference_target(schema: Mapping[str, Any], reference: str) -> object:
    # The node walk proved the reference is `#` or one direct definition; JSON Pointer escapes decode in its name.
    if reference == "#":
        return schema
    container, _, name = reference[2:].partition("/")
    return schema.get(container, {}).get(name.replace("~1", "/").replace("~0", "~"))


def _expansion_edges(schema: Mapping[str, Any], node: Mapping[str, Any]) -> list[object]:
    edges = [*_applied_subschemas(node)]
    if "$ref" in node:
        edges.append(_reference_target(schema, node["$ref"]))
    return edges


def expanded_subschemas(schema: Mapping[str, Any]) -> int | None:
    """Count the subschemas validation can visit, or None when a reference is missing or leads back into itself.

    Every value at a subschema position counts once per path from the root that reaches it, and a `$ref` target counts
    again at every subschema holding that reference, so a `#` reference is always a cycle. Each subschema's count is
    computed once, so the walk is linear in the document however large the expansion.
    """
    counts: dict[int, int] = {}
    open_nodes = {id(schema)}
    frames: list[tuple[Mapping[str, Any], list[object], list[int]]] = [(schema, _expansion_edges(schema, schema), [1])]
    while frames:
        node, edges, total = frames[-1]
        if edges:
            edge = edges.pop()
            if edge is None or id(edge) in open_nodes:
                return None
            if isinstance(edge, Mapping) and id(edge) not in counts:
                open_nodes.add(id(edge))
                frames.append((edge, _expansion_edges(schema, edge), [1]))
            else:
                total[0] += counts.get(id(edge), 1)
            continue
        frames.pop()
        open_nodes.discard(id(node))
        counts[id(node)] = total[0]
        if frames:
            frames[-1][2][0] += total[0]
    return counts[id(schema)]


def schema_problem(schema: Mapping[str, Any]) -> str | None:
    """The first reference or pattern problem in a metaschema-valid schema, or None when Brain can validate it."""
    pending: list[object] = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, Mapping):
            problem = _node_problem(node, nested=node is not schema)
            if problem is not None:
                return problem
            pending.extend(_applied_subschemas(node))
    expanded = expanded_subschemas(schema)
    if expanded is None:
        return "must resolve every reference without a cycle"
    if expanded > MAX_EXPANDED_SUBSCHEMAS:
        return "is too large once its references are expanded"
    return None


@cache
def _pattern_options():
    import re2

    options = re2.Options()
    options.max_mem = 1 << 20
    options.never_capture = True
    options.log_errors = False
    return options


def _compiled_pattern(pattern: str):
    import re2

    try:
        compiled = re2.compile(pattern, _pattern_options())
    except (re2.error, UnicodeEncodeError) as exc:
        raise PatternError("pattern is outside the linear-time matcher") from exc
    if compiled.programsize > MAX_PATTERN_PROGRAM:
        raise PatternError("pattern exceeds the linear-time matcher bound")
    return compiled


@contextmanager
def pattern_work_budget() -> Iterator[None]:
    """Charge every search inside the block against one MAX_PATTERN_WORK budget."""
    token = _pattern_work.set([MAX_PATTERN_WORK])
    try:
        yield
    finally:
        _pattern_work.reset(token)


def pattern_matches(pattern: str, subject: str) -> bool:
    """Whether pattern matches anywhere in subject, as JSON Schema `pattern` requires; raises PatternError.

    Outside pattern_work_budget, the search alone is charged against a fresh budget.
    """
    compiled = _compiled_pattern(pattern)
    try:
        size = len(subject.encode())
    except UnicodeEncodeError as exc:
        raise PatternError("subject is not valid Unicode") from exc
    remaining = _pattern_work.get() or [MAX_PATTERN_WORK]
    remaining[0] -= size * compiled.programsize
    if remaining[0] < 0:
        raise PatternError("pattern matching exceeds its work budget")
    return compiled.search(subject) is not None


def _pattern(validator, pattern, instance, schema):
    from jsonschema.exceptions import ValidationError

    if validator.is_type(instance, "string") and not pattern_matches(pattern, instance):
        yield ValidationError("value does not match its pattern")


def _pattern_properties(validator, patterns, instance, schema):
    if not validator.is_type(instance, "object"):
        return
    for pattern, subschema in patterns.items():
        for name, value in instance.items():
            if pattern_matches(pattern, name):
                yield from validator.descend(value, subschema, path=name, schema_path=pattern)


def _additional_properties(validator, additional, instance, schema):
    from jsonschema.exceptions import ValidationError

    if not validator.is_type(instance, "object"):
        return
    declared = schema.get("properties", {})
    patterns = schema.get("patternProperties", {})
    extras = [
        name
        for name in instance
        if name not in declared and not any(pattern_matches(pattern, name) for pattern in patterns)
    ]
    if validator.is_type(additional, "object"):
        for extra in extras:
            yield from validator.descend(instance[extra], additional, path=extra)
    elif not additional and extras:
        yield ValidationError("additional properties are not allowed")


def _unevaluated_properties(validator, unevaluated, instance, schema):
    # jsonschema evaluates this keyword's pattern properties with Python `re`; admission refuses it.
    raise PatternError("unevaluatedProperties is outside the linear-time matcher")


@cache
def _validator_class():
    from jsonschema import Draft202012Validator, validators

    # Every keyword that reads a pattern goes through pattern_matches; `format` is never asserted.
    return validators.extend(
        Draft202012Validator,
        {
            "additionalProperties": _additional_properties,
            "pattern": _pattern,
            "patternProperties": _pattern_properties,
            "unevaluatedProperties": _unevaluated_properties,
        },
    )


def _without_dialects(node: object) -> object:
    # jsonschema switches to the stock validator class at any subschema declaring `$schema`, which would bypass the
    # pattern keywords above. Admission proved every declaration names Draft 2020-12, so dropping it changes nothing
    # else; data positions such as `const` or `default` are copied untouched.
    if not isinstance(node, Mapping):
        return node
    copied = {key: value for key, value in node.items() if key != "$schema"}
    for keyword in _APPLICATOR_KEYWORDS & copied.keys():
        copied[keyword] = _without_dialects(copied[keyword])
    for keyword in _APPLICATOR_LIST_KEYWORDS & copied.keys():
        copied[keyword] = [_without_dialects(child) for child in copied[keyword]]
    for keyword in _APPLICATOR_MAP_KEYWORDS & copied.keys():
        copied[keyword] = {name: _without_dialects(child) for name, child in copied[keyword].items()}
    return copied


def payload_validator(schema: Mapping[str, Any], registry):
    """Validate against an admitted schema with the linear-time matcher and the caller's reference registry."""
    return _validator_class()(_without_dialects(schema), registry=registry)

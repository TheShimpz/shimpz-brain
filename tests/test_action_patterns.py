"""Action argument patterns run on the bounded linear-time matcher, never Python's backtracking `re`."""

from __future__ import annotations

import time
import unittest
from unittest import mock

import action_tool
import agent_runtime
from protocol.team.action.v1 import schema as action_protocol

TOOL = agent_runtime._tool_name("shimpz-exa", "search-web")
# Python's backtracking `re` needs seconds for 30 characters of this pattern and doubles with each one more.
CATASTROPHIC = "^(a+)+b$"
# The semantics Team pins in the Assistant protocol's pattern vectors.
SEMANTICS = (
    ("b", "abc", True),
    ("^a$", "a\n", False),
    ("(?m)^a$", "b\na\nc", True),
    ("^\\d$", "\u0663", False),
    ("^\\w$", "\u00e9", False),
    ("^\\W$", "\u00e9", True),
    ("\\s", "\u000b", False),
    ("\\s", "\u00a0", False),
    ("\\b\u00e9", "\u00e9", False),
    ("^.$", "\n", False),
    ("(?s)^.$", "\n", True),
    ("(?i)^\u00e9$", "\u00c9", True),
    ("(?i)^k$", "\u212a", True),
    ("^(?i:a)b$", "AB", False),
)


def _action(schema: dict[str, object]) -> agent_runtime.ActionDefinition:
    return agent_runtime.ActionDefinition(id="search-web", summary="Search the web.", input_schema=schema)


def _accepts(action: agent_runtime.ActionDefinition, **arguments: object) -> bool:
    with mock.patch("langgraph.types.interrupt", return_value="suspended") as interrupt:
        result = action_tool.request_action(TOOL, "shimpz-exa", action).func(**arguments)
    accepted = interrupt.called
    if not accepted:
        assert result.startswith("Action not executed:")
    return accepted


class ActionPatternTests(unittest.TestCase):
    def test_matches_the_pinned_pattern_semantics(self) -> None:
        for pattern, subject, matches in SEMANTICS:
            with self.subTest(pattern=pattern, subject=subject):
                self.assertIs(action_protocol.pattern_matches(pattern, subject), matches)
                action = _action({"type": "object", "properties": {"q": {"type": "string", "pattern": pattern}}})
                self.assertIs(_accepts(action, q=subject), matches)

    def test_nested_quantifiers_are_refused_in_linear_time(self) -> None:
        action = _action(
            {
                "type": "object",
                "properties": {"q": {"type": "string", "pattern": CATASTROPHIC}},
                "patternProperties": {CATASTROPHIC: {"type": "integer"}},
                "additionalProperties": False,
            }
        )
        for arguments in ({"q": "a" * 30}, {"q": "a" * 100_000}, {"a" * 30: 1}):
            with self.subTest(size=len(next(iter(arguments.values()))) if "q" in arguments else 30):
                started = time.perf_counter()
                self.assertFalse(_accepts(action, **arguments))
                self.assertLess(time.perf_counter() - started, 1.0)
        self.assertTrue(_accepts(action, q="aab", aab=1))

    def test_admission_refuses_patterns_outside_the_bounded_matcher(self) -> None:
        for pattern in ("\\u0041", "(?x)a b", "a{1001}", "(?:a{100}){11}", "a" * 20_000, "\ud800"):
            for schema in (
                {"type": "object", "properties": {"q": {"pattern": pattern}}},
                {"type": "object", "patternProperties": {pattern: {}}},
            ):
                with (
                    self.subTest(pattern=pattern[:16], schema=sorted(schema)),
                    self.assertRaisesRegex(agent_runtime.RuntimeContractError, "linear-time matcher admits"),
                ):
                    _action(schema)
        _action({"type": "object", "properties": {"q": {"pattern": "a" * 16_000}}})

    def test_unevaluated_properties_and_unreadable_subjects_fail_closed(self) -> None:
        for schema in (
            {"type": "object", "unevaluatedProperties": False},
            {"type": "object", "properties": {"q": {"allOf": [{"unevaluatedProperties": False}]}}},
        ):
            with self.subTest(schema=schema), self.assertRaisesRegex(agent_runtime.RuntimeContractError, "unevaluated"):
                _action(schema)
        with self.assertRaises(action_protocol.PatternError):
            list(action_tool.action_schema_validator({"unevaluatedProperties": False}).iter_errors({}))

        negated = _action({"type": "object", "properties": {"q": {"not": {"pattern": "secret"}}}})
        self.assertTrue(_accepts(negated, q="public"))
        self.assertFalse(_accepts(negated, q="secret\ud800"))

    def test_a_declared_dialect_never_switches_to_the_backtracking_validator(self) -> None:
        draft = action_protocol.DRAFT_2020_12
        action = _action(
            {
                "$schema": draft,
                "type": "object",
                "$defs": {"a": {"$schema": draft, "type": "string", "pattern": "^a$"}},
                "properties": {
                    "q": {"anyOf": [{"$schema": draft, "type": "string", "pattern": "^a$"}]},
                    "r": {"$ref": "#/$defs/a"},
                },
                "default": {"$schema": "data"},
            }
        )
        # Python `re` would accept the trailing newline; RE2 matches `$` only at the end of the text.
        self.assertFalse(_accepts(action, q="a\n"))
        self.assertFalse(_accepts(action, r="a\n"))
        self.assertTrue(_accepts(action, r="a"))

    def test_one_validation_charges_every_search_against_one_budget(self) -> None:
        # `a.{900}c` compiles to 7,206 instructions; RE2 runs it without its DFA at several nanoseconds per byte.
        heavy = "a.{900}c"
        fits = action_protocol.MAX_PATTERN_WORK // action_protocol.compiled_pattern(heavy).programsize
        self.assertFalse(action_protocol.pattern_matches(heavy, "b" * fits))
        with self.assertRaisesRegex(action_protocol.PatternError, "work budget"):
            action_protocol.pattern_matches(heavy, "\u00e9" * (fits // 2 + 1))
        with action_protocol.pattern_work_budget():
            self.assertFalse(action_protocol.pattern_matches(heavy, "b" * (fits // 2)))
            with self.assertRaisesRegex(action_protocol.PatternError, "work budget"):
                action_protocol.pattern_matches(heavy, "b" * (fits - fits // 2 + 1))
        self.assertFalse(action_protocol.pattern_matches(heavy, "b" * fits))
        # One argument checked by the heavy pattern from 64 expanded positions: each search alone fits.
        action = _action(
            {
                "type": "object",
                "$defs": {"heavy": {"type": "string", "not": {"pattern": heavy}}},
                "properties": {"q": {"allOf": [{"$ref": "#/$defs/heavy"}] * 64}},
            }
        )
        self.assertTrue(_accepts(action, q="b" * (fits // 64)))
        started = time.perf_counter()
        self.assertFalse(_accepts(action, q="b" * (fits // 64 + 1)))
        self.assertLess(time.perf_counter() - started, 2.0)

    def test_additional_properties_consult_each_pattern_property_separately(self) -> None:
        validator = action_tool.action_schema_validator(
            {
                "type": "object",
                "properties": {"id": {}},
                "patternProperties": {"^x-[a-z]+$": {"type": "string"}, "(?i)^y$": {"type": "integer"}},
                "additionalProperties": False,
            }
        )
        self.assertTrue(validator.is_valid({"id": 1, "x-a": "b", "Y": 2}))
        for payload in ({"x-a": 1}, {"x-1": "b"}, {"Y": "2"}):
            self.assertFalse(validator.is_valid(payload))

        typed = action_tool.action_schema_validator(
            {"type": "object", "patternProperties": {"^x": {}}, "additionalProperties": {"type": "integer"}}
        )
        self.assertTrue(typed.is_valid({"x": "any", "other": 1}))
        self.assertFalse(typed.is_valid({"other": "text"}))
        self.assertTrue(action_tool.action_schema_validator({"additionalProperties": True}).is_valid({"other": 1}))
        self.assertTrue(action_tool.action_schema_validator({"additionalProperties": False}).is_valid([1]))
        self.assertTrue(action_tool.action_schema_validator({"patternProperties": {"^x": False}}).is_valid([1]))
        self.assertTrue(action_tool.action_schema_validator({"pattern": "^x"}).is_valid(1))


if __name__ == "__main__":
    unittest.main()

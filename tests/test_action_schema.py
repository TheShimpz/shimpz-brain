"""The Brain's bound re-check of an Action schema keeps one metaschema answer per exact schema."""

import unittest
from unittest import mock

import action_schema


class MetaschemaCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        action_schema._CHECKED.clear()
        checker = mock.patch.object(
            action_schema.Draft202012Validator,
            "check_schema",
            side_effect=action_schema.Draft202012Validator.check_schema,
        )
        self.checks = checker.start()
        self.addCleanup(checker.stop)

    def test_each_exact_schema_is_checked_against_the_metaschema_once(self) -> None:
        valid = {"type": "object", "properties": {"zone": {"type": "string"}}}
        invalid = {"type": 5}
        for _ in range(3):
            self.assertIsNone(action_schema.bound_problem(valid))
            self.assertEqual(action_schema.bound_problem(invalid), "invalid")
        self.assertEqual(self.checks.call_count, 2)
        # The same schema written in another key order is the same canonical schema.
        self.assertIsNone(action_schema.bound_problem({"properties": {"zone": {"type": "string"}}, "type": "object"}))
        self.assertEqual(self.checks.call_count, 2)

    def test_every_schema_a_turn_may_carry_stays_checked_and_no_more_are_kept(self) -> None:
        # A turn's largest set of schemas is checked once and then answered from the cache on the next turn.
        schemas = [{"type": "object", "title": f"s{index}"} for index in range(action_schema.MAX_CHECKED_SCHEMAS)]
        for _ in range(2):
            for schema in schemas:
                self.assertIsNone(action_schema.bound_problem(schema))
        self.assertEqual(self.checks.call_count, action_schema.MAX_CHECKED_SCHEMAS)
        self.assertIsNone(action_schema.bound_problem({"type": "object", "title": "one more"}))
        self.assertEqual(len(action_schema._CHECKED), action_schema.MAX_CHECKED_SCHEMAS)
        # The oldest used schema was dropped; the newest still answers from the cache.
        self.assertIsNone(action_schema.bound_problem(schemas[0]))
        self.assertEqual(self.checks.call_count, action_schema.MAX_CHECKED_SCHEMAS + 2)
        self.assertIsNone(action_schema.bound_problem(schemas[-1]))
        self.assertEqual(self.checks.call_count, action_schema.MAX_CHECKED_SCHEMAS + 2)

    def test_the_value_and_byte_bounds_still_refuse_before_any_check(self) -> None:
        self.assertEqual(action_schema.bound_problem(["not", "a", "schema"]), "invalid")
        self.assertEqual(action_schema.bound_problem({"default": {"a", "set"}}), "not JSON")
        self.checks.assert_not_called()


if __name__ == "__main__":
    unittest.main()

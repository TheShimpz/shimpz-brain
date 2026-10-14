"""The Brain's bound re-check of an Action schema keeps one metaschema answer per exact schema."""

import unittest

import action_schema


class MetaschemaCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        action_schema._valid_draft.cache_clear()

    def test_each_exact_schema_is_checked_against_the_metaschema_once(self) -> None:
        valid = {"type": "object", "properties": {"zone": {"type": "string"}}}
        invalid = {"type": 5}
        for _ in range(3):
            self.assertIsNone(action_schema.bound_problem(valid))
            self.assertEqual(action_schema.bound_problem(invalid), "invalid")
        info = action_schema._valid_draft.cache_info()
        self.assertEqual((info.misses, info.hits), (2, 4))
        # The same schema written in another key order is the same canonical schema.
        self.assertIsNone(action_schema.bound_problem({"properties": {"zone": {"type": "string"}}, "type": "object"}))
        self.assertEqual(action_schema._valid_draft.cache_info().misses, 2)

    def test_the_value_and_byte_bounds_still_refuse_before_any_check(self) -> None:
        self.assertEqual(action_schema.bound_problem(["not", "a", "schema"]), "invalid")
        self.assertEqual(action_schema.bound_problem({"default": {"a", "set"}}), "not JSON")
        self.assertEqual(action_schema._valid_draft.cache_info().misses, 0)


if __name__ == "__main__":
    unittest.main()

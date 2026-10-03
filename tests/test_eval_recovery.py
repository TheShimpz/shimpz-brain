"""Provider-free checks for the Luna-99 search-and-recovery mechanisms (ADR-0094)."""

from __future__ import annotations

import json
import unittest

from eval import fixtures, recovery
from eval import world as simulated

TASKS = next(a for a in fixtures.ASSISTANTS["tasks"].actions if a.id == "list-tasks").input_schema
CREATE = next(a for a in fixtures.ASSISTANTS["tasks"].actions if a.id == "create-task").input_schema
LINES = next(a for a in fixtures.ASSISTANTS["invoices"].actions if a.id == "create-invoice").input_schema


class SchemaTests(unittest.TestCase):
    def test_optional_properties_become_required_and_nullable_and_only_their_nulls_drop(self):
        shown = recovery.nullable(CREATE)
        self.assertEqual(set(shown["required"]), {"title", "due_date", "tags"})
        self.assertEqual(shown["properties"]["title"], CREATE["properties"]["title"])
        self.assertEqual(shown["properties"]["due_date"]["anyOf"][1], {"type": "null"})
        payload = {"title": "Renew", "due_date": None, "tags": None}
        self.assertEqual(recovery.drop_nulls(payload, CREATE), {"title": "Renew"})
        # A required null and an undeclared property stay, so the original schema's validation still refuses them.
        self.assertEqual(recovery.drop_nulls({"title": None, "x": None}, CREATE), {"title": None, "x": None})
        lines = recovery.nullable(LINES)["properties"]["lines"]["items"]
        self.assertEqual(set(lines["required"]), {"description", "quantity", "unit_price"})
        dropped = recovery.drop_nulls({"lines": [{"description": "a", "quantity": None}]}, LINES)
        self.assertEqual(dropped, {"lines": [{"description": "a", "quantity": None}]})
        nested = {
            "type": "object",
            "properties": {"o": {"type": "object", "properties": {"k": {"type": "string"}}, "required": []}},
            "required": ["o"],
        }
        self.assertEqual(recovery.nullable(nested)["properties"]["o"]["required"], ["k"])
        self.assertEqual(recovery.drop_nulls({"o": {"k": None}}, nested), {"o": {}})
        self.assertEqual(recovery.nullable({"type": "string"}), {"type": "string"})
        self.assertEqual(recovery._nested(True), True)
        self.assertEqual(recovery.drop_nulls([None], {"type": "array"}), [None])
        self.assertEqual(recovery.drop_nulls(None, None), None)


class ReadTests(unittest.TestCase):
    def test_only_a_result_whose_every_list_is_empty_found_nothing(self):
        self.assertTrue(recovery.empty({"tasks": []}))
        self.assertTrue(recovery.empty({"result": [], "success": True}))
        self.assertFalse(recovery.empty({"tasks": [{"id": "tk-1"}]}))
        self.assertFalse(recovery.empty({"text": "Page not found."}))
        self.assertFalse(recovery.empty(["x"]))

    def test_relaxing_keeps_required_arguments_and_needs_an_optional_one(self):
        schema = {"type": "object", "properties": {"zone_id": {}, "name": {}}, "required": ["zone_id"]}
        self.assertEqual(recovery.relaxed(schema, {"zone_id": "z", "name": "x"}), {"zone_id": "z"})
        self.assertIsNone(recovery.relaxed(schema, {"zone_id": "z"}))
        self.assertEqual(recovery.relaxed(TASKS, {"tag": "x"}), {})

    def test_related_results_stay_apart_with_provenance_and_truncation(self):
        many = {"tasks": [{"id": n} for n in range(25)], "page": 1}
        shaped = recovery.with_related({"tasks": []}, many, {"status": "open"}, "re-read this Action without tag")
        self.assertEqual(shaped["tasks"], [])
        self.assertEqual(shaped["related"]["arguments"], {"status": "open"})
        self.assertEqual((shaped["related"]["total"], shaped["related"]["truncated"]), (25, True))
        self.assertEqual(len(shaped["related"]["tasks"]), recovery.RELATED_LIMIT)
        self.assertNotIn("page", shaped["related"])
        self.assertIn("without tag", shaped["team_note"])


class FailureTests(unittest.TestCase):
    def test_only_declared_no_effect_codes_recover_with_a_class_hint(self):
        exists = recovery.recovered("record-exists", simulated.NO_EFFECT_CODES)
        self.assertEqual((exists["error"], exists["changed"]), ("record-exists", False))
        self.assertIn("otherwise report the existing item", exists["team_note"])
        self.assertIn("List or search", recovery.recovered("task-not-found", simulated.NO_EFFECT_CODES)["team_note"])
        self.assertIn("another declared Action", recovery.recovered("quota", frozenset({"quota"}))["team_note"])
        self.assertIsNone(recovery.recovered("undeclared-action", simulated.NO_EFFECT_CODES))
        self.assertEqual(recovery.replay_refused()["error"], "already-done")
        self.assertEqual(
            recovery.fingerprint("dns", "x", {"b": 1, "a": 2}), recovery.fingerprint("dns", "x", {"a": 2, "b": 1})
        )


class HelperTests(unittest.TestCase):
    def test_rewrites_parse_dedupe_bound_and_keep_resource_selectors(self):
        text = recovery.rewrite_input("m", "g", "list", "s", TASKS, {"tag": "x"})
        self.assertEqual(json.loads(text)["arguments_that_returned_nothing"], {"tag": "x"})
        answer = json.dumps(
            {
                "alternatives": [
                    '{"tag": "x"}',
                    '{"a": 1}',
                    '{"a": 1}',
                    "nope",
                    "[1]",
                    3,
                    *map(json.dumps, ({"b": 1}, {"c": 1}, {"d": 1})),
                ]
            }
        )
        self.assertEqual(recovery.alternatives(answer, {"tag": "x"}), [{"a": 1}, {"b": 1}, {"c": 1}])
        self.assertEqual(recovery.alternatives("not json", {}), [])
        self.assertEqual(recovery.alternatives('{"alternatives": 3}', {}), [])
        schema = {"required": ["zone_id", "query"]}
        earlier = ['{"zones": [{"id": "zn-1"}]}']
        original = {"zone_id": "zn-1", "query": "dentist"}
        self.assertTrue(recovery.kept_selectors(original, {"zone_id": "zn-1", "query": "x"}, schema, earlier))
        self.assertFalse(recovery.kept_selectors(original, {"zone_id": "zn-2", "query": "x"}, schema, earlier))
        self.assertTrue(recovery.kept_selectors({"zone_id": ""}, {"zone_id": "zn-2"}, schema, earlier))

    def test_the_critic_sees_clipped_records_and_its_answer_is_parsed_closed(self):
        ledger = [{"assistant": "dns", "action": "list", "input": {}, "result": {"x": "y" * 900}}, {"failed": "c"}]
        body = json.loads(recovery.critic_input("m", ledger, "r", True))
        self.assertTrue(body["actions"][0]["result"].endswith("…"))
        self.assertIn("failed", body["actions"][1]["result"])
        self.assertTrue(body["reply_is_a_clarifying_question"])
        self.assertEqual(recovery.critique('{"verdict": "revise", "problem": " fix it "}'), "fix it")
        self.assertIsNone(recovery.critique('{"verdict": "ok", "problem": ""}'))
        self.assertIsNone(recovery.critique('{"verdict": "revise", "problem": ""}'))
        self.assertIsNone(recovery.critique("[]"))
        self.assertIn('"fix it"', recovery.review_section("fix it"))

    def test_helper_requests_are_structured_and_their_text_is_read(self):
        body = recovery.helper_body("i", "t", recovery.CRITIC_SCHEMA, "review", 100)
        self.assertEqual((body["model"], body["store"], body["max_output_tokens"]), (recovery.HELPER_MODEL, False, 100))
        self.assertTrue(body["text"]["format"]["strict"])
        answer = {
            "output": [
                {"type": "reasoning"},
                {"type": "message", "content": [{"type": "refusal"}, {"type": "output_text", "text": "{}"}]},
            ]
        }
        self.assertEqual(recovery.response_text(answer), "{}")
        self.assertIsNone(recovery.response_text({"output": [{"type": "message", "content": "x"}]}))
        self.assertIsNone(recovery.response_text(None))
        self.assertIn("broaden it", recovery.PROTOCOL)


if __name__ == "__main__":
    unittest.main()

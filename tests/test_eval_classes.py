"""Provider-free checks for the Luna-classes mechanisms W, Q, V, and U (ADR-0094)."""

from __future__ import annotations

import datetime
import json
import unittest
from decimal import Decimal

from eval import classes, fixtures, fresh2

RECORD = next(a for a in fixtures.ASSISTANTS["dns"].actions if a.id == "create-record").input_schema
EXPENSE = next(a for a in fresh2.ASSISTANTS["expenses"].actions if a.id == "create-expense").input_schema
TODAY = datetime.date(2026, 10, 3)
POSTAGE = "注文 ORD-2283 の荷物の送料を「Postage ORD-2283」として発送日の日付で経費登録して"


def found(value: object, message: str) -> bool:
    return bool(classes.spans(value, classes.normalize(message)))


class TraceTests(unittest.TestCase):
    def test_normalization_and_quotes_of_every_script(self):
        self.assertEqual(classes.normalize("  Ａ  b\nC "), "a b c")
        spans = classes.quoted('Add "Urgent" «Note» 「メモ」 „Hallo“ ‘x’')
        self.assertEqual(set(spans), {"urgent", "note", "hallo", "x", "メモ"})

    def test_a_number_token_has_one_reading_or_none(self):
        self.assertEqual(classes._decimal("11,90"), Decimal("11.90"))
        self.assertEqual(classes._decimal("1.234,5"), Decimal("1234.5"))
        self.assertEqual(classes._decimal("1,234.5"), Decimal("1234.5"))
        self.assertEqual(classes._decimal("1 234 567"), Decimal(1234567))
        self.assertEqual(classes._decimal("-5"), Decimal(-5))
        self.assertIsNone(classes._decimal("1,234"))
        self.assertIsNone(classes._decimal("1.2.3"))
        self.assertIsNone(classes.number(True))
        self.assertIsNone(classes.number("abc"))
        self.assertEqual(classes.number(" 11.90 "), Decimal("11.90"))

    def test_numbers_match_only_whole_tokens_of_the_same_value(self):
        self.assertTrue(found(11.9, "11,90 €"))
        self.assertTrue(found(100, "give 100 points"))
        self.assertFalse(found(1190, "11.90 EUR"))
        self.assertFalse(found(5, "-5 units"))
        self.assertFalse(found(100, "1,100 or 4 100"))
        self.assertFalse(found(1234, "1,234"))
        self.assertFalse(found("2283", "ORD-2283"))
        self.assertFalse(found(5, "5e3"))
        self.assertFalse(found(3, "5e3"))
        self.assertFalse(found(1, "room A1"))
        self.assertFalse(found(1, "12kg"))
        self.assertFalse(found(12, "12.5e3"))
        self.assertFalse(found(11, "11.90kg"))
        self.assertFalse(found(1, "1,100kg"))
        self.assertFalse(found(90, "11.90"))

    def test_dates_and_times_match_only_unambiguous_surface_forms(self):
        self.assertTrue(found("2026-10-25", "le 25/10/2026"))
        self.assertTrue(found("2026-10-05", "2026年10月5日"))
        self.assertFalse(found("2026-05-10", "5/10/2026"))
        self.assertFalse(found("2026-10-05", "5/10/2026"))
        self.assertTrue(found("15:30", "à 15h30"))
        self.assertTrue(found("15:30", "at 3:30 pm"))
        self.assertTrue(found("15:00", "um 15 Uhr"))
        self.assertTrue(found("15:00", "at 3pm"))
        self.assertTrue(found("15:30", "15時半"))
        self.assertFalse(found("03:30", "at 3:30 pm"))
        self.assertFalse(found("03:00", "午後3時"))
        self.assertTrue(found("ORD-2283", "order ORD-2283."))
        self.assertFalse(found("ORD-228", "order ORD-2283."))
        self.assertEqual(classes.surface_forms("2026-02-30"), {"2026-02-30"})
        self.assertEqual(classes.surface_forms("25:99"), {"25:99"})
        self.assertIn("9am", classes.surface_forms("09:00"))
        self.assertEqual(classes.spans(["x"], "x"), [])
        self.assertEqual(classes._literal_spans("", "abc"), [])

    def test_only_atomic_values_trace_by_presence(self):
        self.assertTrue(classes.atomic(3, None))
        self.assertTrue(classes.atomic("express", {"enum": ["express"]}))
        self.assertTrue(classes.atomic("ORD-2283", {}))
        self.assertTrue(classes.atomic("a@b.io", {}))
        self.assertFalse(classes.atomic("送料", {}))
        self.assertFalse(classes.atomic("two words", {}))
        self.assertFalse(classes.atomic("", {}))
        self.assertFalse(classes.atomic(True, {}))
        self.assertFalse(classes.atomic(["x"], {}))

    def test_a_user_value_occupying_another_arguments_span_is_not_traced(self):
        message = 'Create an A record for "blog.example.com".'
        record = {"zone_id": "zn-7f3a", "type": "A", "name": "blog", "content": "blog.example.com"}
        self.assertEqual(classes.gated_fields(RECORD, record, message, ("content",), True), (["content"], []))
        given = "Create an A record named shop in example.com pointing to 203.0.113.10."
        record = {"zone_id": "z", "type": "A", "name": "shop", "content": "203.0.113.10"}
        self.assertEqual(classes.gated_fields(RECORD, record, given, ("content",), True), ([], []))

    def test_every_optional_value_needs_the_helper_unless_a_schema_default(self):
        call = {
            "amount": 11.9,
            "category": "postage",
            "currency": "EUR",
            "date": "2026-09-30",
            "description": "Postage ORD-2283",
            "note": "Postage ORD-2283",
        }
        pending = classes.gated_fields(EXPENSE, call, POSTAGE, ("date", "amount"), True)
        self.assertEqual(pending, (["date", "amount"], ["note"]))
        self.assertEqual(classes.gated_fields(EXPENSE, call, POSTAGE, (), False), ([], []))
        schema = {
            **EXPENSE,
            "properties": {**EXPENSE["properties"], "note": {"type": "string", "default": "Postage ORD-2283"}},
        }
        self.assertEqual(classes.gated_fields(schema, call, POSTAGE, (), True), ([], []))
        self.assertFalse(classes.defaulted("x", 1, {"properties": None}))
        self.assertEqual(classes.optional_fields(EXPENSE, call), ["note"])


class HelperTests(unittest.TestCase):
    def test_the_provenance_input_carries_the_call_and_clipped_earlier_results(self):
        earlier = [{"assistant": "shipping", "action": "list-shipments", "result": {"x": "y" * 2000}}]
        body = json.loads(classes.provenance_input("m", TODAY, {"id": "a"}, {"k": 1}, ["k"], earlier))
        self.assertEqual(body["current_date"], "2026-10-03")
        self.assertEqual(body["fields_to_check"], ["k"])
        self.assertTrue(body["earlier_results"][0]["result"].endswith("…"))

    def test_only_cited_verbatim_and_consistent_evidence_supplies_a_field(self):
        message = "Dated the day it shipped, for 12 boxes"
        answer = json.dumps(
            {
                "fields": [
                    {"name": "date", "supplied": True, "evidence": "dated the  day it shipped"},
                    {"name": "note", "supplied": True, "evidence": "handle with care"},
                    {"name": "amount", "supplied": False, "evidence": ""},
                    {"name": "count", "supplied": True, "evidence": "12 boxes"},
                    {"name": "count", "supplied": False, "evidence": ""},
                    {"name": "other", "supplied": True, "evidence": "Dated"},
                    {"name": "boxes", "supplied": True, "evidence": 12},
                    {"name": "none", "supplied": "true", "evidence": "12 boxes"},
                    "junk",
                ]
            }
        )
        self.assertEqual(
            classes.supplied(answer, ["date", "note", "amount", "count", "boxes", "none"], message), {"date"}
        )
        self.assertIsNone(classes.supplied(None, ["date"], message))
        self.assertIsNone(classes.supplied("not json", ["date"], message))
        self.assertIsNone(classes.supplied("[1]", ["date"], message))
        self.assertIsNone(classes.supplied('{"fields": 3}', ["date"], message))

    def test_refusals_change_nothing_and_say_what_to_do(self):
        asked = classes.refusal(["content"], ["note"])
        self.assertEqual((asked["code"], asked["fields"], asked["changed"]), ("user-value-missing", ["content"], False))
        self.assertIn("ask the user", asked["detail"])
        dropped = classes.refusal((), ["note"])
        self.assertEqual((dropped["code"], dropped["fields"]), ("unrequested-optional-value", ["note"]))
        self.assertIn("Do not ask the user", dropped["detail"])


class WorkingSetTests(unittest.TestCase):
    def test_records_writes_and_unresolved_references_are_tracked_by_assistant(self):
        working = classes.WorkingSet('Give the customer of order ORD-2279 the "Free pastry" (a.b@x.io)')
        self.assertIsNone(working.render())
        shipment = {"id": "shp-1", "address_id": "adr-1", "recipient": "Leila", "reference": "ORD-2279", "n": [1]}
        working.add("shipping", "list-shipments", False, {"reference": "ORD-2279"}, {"shipments": [shipment]})
        working.add("loyalty", "find", False, {}, {"customers": [{"id": "shp-1", "name": "same id elsewhere"}]})
        working.add("loyalty", "add-points", True, {"customer_id": "c-1", "points": 100}, {"customer": {"id": "c-1"}})
        working.add("loyalty", "redeem", True, {"customer_id": "c-1", "x": [1]}, {"error": "refused"})
        working.add("loyalty", "noop", True, {"customer_id": "c-1"}, {"changed": False})
        working.add("shipping", "list-shipments", False, {}, {"shipments": [shipment]})
        block = working.render()
        keys = [(item["assistant"], item["id"]) for item in block["records"]]
        self.assertEqual(keys, [("loyalty", "shp-1"), ("loyalty", "c-1"), ("shipping", "shp-1")])
        self.assertEqual(list(block["records"][2])[:4], ["assistant", "action", "id", "address_id"])
        self.assertNotIn("n", block["records"][2])
        written = [{"assistant": "loyalty", "action": "add-points", "input": {"customer_id": "c-1", "points": 100}}]
        self.assertEqual(block["writes_done"], written)
        self.assertEqual(block["unresolved_from_request"], ["free pastry", "a.b@x.io"])
        self.assertNotIn("Use", block["note"])
        self.assertEqual(classes.with_working_set({"a": 1}, block)["team_working_set"], block)
        self.assertEqual(classes.with_working_set(["a"], block), ["a"])
        self.assertEqual(classes.with_working_set({"a": 1}, None), {"a": 1})

    def test_the_whole_block_stays_within_its_size_cap(self):
        message = " ".join(f"REF-{index:03d}{'x' * 60}" for index in range(40))
        working = classes.WorkingSet(message)
        many = [{"id": f"r-{index}", "name": "n" * 70, "status": "s" * 70} for index in range(40)]
        working.add("a", "list", False, {}, {"items": many})
        for index in range(12):
            working.add("a", "write", True, {"note": "w" * 70, "n": index}, {"ok": True})
        block = working.render()
        self.assertLessEqual(len(json.dumps(block, ensure_ascii=False, sort_keys=True)), classes.WORKING_SET_CHARS)
        deep = {"a": {"b": {"c": {"d": {"e": {"f": {"id": "too-deep"}}}}}}}
        self.assertEqual(list(classes.records(deep)), [])
        self.assertEqual(list(classes.records({"id": True})), [])


class PlanTests(unittest.TestCase):
    def test_a_plan_is_bounded_quoted_data(self):
        shown = classes.outline({"properties": {"a": {"description": "x" * 99}, "b": {}, "c": True}, "required": ["a"]})
        self.assertEqual(shown, {"required": ["a: " + "x" * 80], "optional": ["b", "c"]})
        self.assertEqual(classes.outline({}), {"required": [], "optional": []})
        text = json.loads(classes.plan_input("m", TODAY, [{"id": "a"}]))
        self.assertEqual(text["assistants"], [{"id": "a"}])
        steps = [{"action": f"a.b{index}", "purpose": "p", "inputs": []} for index in range(12)]
        planned = classes.plan(json.dumps({"steps": steps}))
        self.assertEqual(len(planned), classes.PLAN_STEPS)
        self.assertIsNone(classes.plan(json.dumps({"steps": []})))
        self.assertIsNone(classes.plan(json.dumps({"steps": ["x"]})))
        self.assertIsNone(classes.plan(None))
        section = classes.plan_section(planned)
        self.assertIn("never authorizes an Action", section)
        self.assertIn("a.b0", section)


if __name__ == "__main__":
    unittest.main()

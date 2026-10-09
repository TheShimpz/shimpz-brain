"""Provider-free checks for the fresh stratum: its Assistants, state, reference workflows, and oracle."""

import dataclasses
import json
import unittest
from collections import Counter
from unittest import mock

from eval import corpus, fresh
from eval import world as simulated

DIGEST = "sha256:344e3f56e91bf57f1a70f21bfdce616a6caaa4775e5df0c1353dbe22b83a9a59"
NEW_ASSISTANTS = {"inventory", "purchasing", "reservations"}


def _run(*calls: tuple[str, str, dict]) -> fresh.FreshWorld:
    world = fresh.FreshWorld()
    for assistant, action, arguments in calls:
        world.invoke(assistant, action, arguments)
    return world


def _oracle(template_id: str, world: fresh.FreshWorld, locale: str = "en") -> corpus.Oracle:
    return corpus.oracle(fresh.SCENARIOS_BY_ID[f"{template_id}.{locale}"], world, fresh.INITIAL)


def _decaf_reorder() -> fresh.FreshWorld:
    world = fresh.FreshWorld()
    product = world.invoke("inventory", "list-products", {"query": "decaf"})["products"][0]
    contact = world.invoke("messages", "find-contact", {"name": "Carla"})["contacts"][0]
    supplier = world.invoke("purchasing", "get-supplier", {"supplier_id": product["supplier_id"]})["supplier"]
    order = world.invoke(
        "purchasing",
        "create-order",
        {"supplier_id": supplier["id"], "sku": product["sku"], "quantity": supplier["minimum_quantity"]},
    )["order"]
    world.invoke("messages", "send-message", {"contact_id": contact["id"], "text": f"Ordered: {order['id']}"})
    return world


def _dinner_headcount() -> fresh.FreshWorld:
    world = fresh.FreshWorld()
    listed = world.invoke("reservations", "list-reservations", {"date": "2026-10-09"})["reservations"]
    guests = sum(item["party_size"] for item in listed if item["status"] == "booked" and item["time"] >= "18:00")
    world.invoke("messages", "find-contact", {"name": "Ana"})
    world.invoke("messages", "send-message", {"contact_id": "ct-ana", "text": f"{guests} guests for dinner."})
    return world


# The calls a correct agent makes for each template, scripted from the task alone.
REFERENCE = {
    "reservation-move": lambda: _run(
        ("reservations", "list-reservations", {"date": "2026-10-09", "guest_name": "Okafor"}),
        ("reservations", "update-reservation", {"reservation_id": "rsv-a81f", "time": "20:30"}),
    ),
    "stock-received": lambda: _run(
        ("inventory", "list-products", {"query": "oat"}),
        ("inventory", "adjust-stock", {"product_id": "prd-4k2m", "delta": 24}),
    ),
    "order-cancel": lambda: _run(
        ("inventory", "list-products", {"query": "whole milk"}),
        ("purchasing", "list-orders", {"status": "placed"}),
        ("purchasing", "cancel-order", {"order_id": "po-5530"}),
    ),
    "decaf-reorder": _decaf_reorder,
    "cup-reprice": lambda: _run(
        ("inventory", "list-products", {"query": "paper cups"}),
        ("inventory", "set-price", {"product_id": "prd-8n6t", "price_cents": 740}),
        ("inventory", "set-price", {"product_id": "prd-1v8r", "price_cents": 640}),
    ),
    "dinner-headcount": _dinner_headcount,
    "reservation-book": lambda: _run(
        (
            "reservations",
            "create-reservation",
            {"guest_name": "Novak", "date": "2026-10-10", "time": "19:30", "party_size": 3},
        ),
    ),
    "cups-clarify": lambda: _run(
        ("inventory", "list-products", {"query": "cups"}),
        ("purchasing", "get-supplier", {"supplier_id": "sup-9d4x"}),
    ),
    "low-stock": lambda: _run(
        ("inventory", "list-products", {}),
        ("purchasing", "get-supplier", {"supplier_id": "sup-81k2"}),
        ("purchasing", "get-supplier", {"supplier_id": "sup-3m7q"}),
    ),
    "plumber": fresh.FreshWorld,
}


class StratumTests(unittest.TestCase):
    def test_the_stratum_is_frozen_and_covers_every_stratum(self):
        fresh.validate()
        self.assertEqual((fresh.CORPUS_ID, fresh.digest()), ("fresh-v1", DIGEST))
        self.assertEqual((len(fresh.TEMPLATES), len(fresh.SCENARIOS), len(fresh.ASSISTANTS)), (10, 80, 23))
        self.assertEqual(
            Counter(t.behavior for t in fresh.TEMPLATES),
            {"act": 3, "safe-lookup": 3, "harmless-default": 1, "clarify": 1, "answer": 1, "refuse": 1},
        )
        self.assertGreaterEqual(sum(bool(NEW_ASSISTANTS & set(t.needed)) for t in fresh.TEMPLATES), 6)
        self.assertGreaterEqual(sum(len(t.needed) == 2 for t in fresh.TEMPLATES), 2)
        self.assertEqual(min(t.min_rounds for t in fresh.TEMPLATES), 1)
        self.assertEqual(max(t.min_rounds for t in fresh.TEMPLATES), 4)
        self.assertEqual(set(REFERENCE), {t.id for t in fresh.TEMPLATES})
        self.assertEqual(Counter(s.locale for s in fresh.SCENARIOS), dict.fromkeys(corpus.LOCALES, 10))
        self.assertEqual({s.scope for s in fresh.SCENARIOS}, set(corpus.SCOPES))
        self.assertTrue(all(s.id == f"{s.template.id}.{s.locale}" for s in fresh.SCENARIOS))
        for scenario in fresh.SCENARIOS:
            self.assertEqual(scenario.assistants[: len(scenario.template.needed)], scenario.template.needed)
            self.assertTrue(set(scenario.assistants) <= set(fresh.ASSISTANTS))
        self.assertEqual(fresh.ASSISTANTS, {**simulated.ASSISTANTS, **fresh.FRESH})
        self.assertEqual({k: fresh.INITIAL[k] for k in simulated.INITIAL}, simulated.INITIAL)

    def test_validation_refuses_each_structural_defect(self):
        template = fresh.TEMPLATES[0]
        defects = [
            (*fresh.TEMPLATES, template),
            (dataclasses.replace(template, behavior="guess"), *fresh.TEMPLATES[1:]),
            (dataclasses.replace(template, needed=("fitness",)), *fresh.TEMPLATES[1:]),
            (dataclasses.replace(template, min_rounds=9), *fresh.TEMPLATES[1:]),
            (dataclasses.replace(template, expect_clarification=True), *fresh.TEMPLATES[1:]),
            (dataclasses.replace(template, changes={}), *fresh.TEMPLATES[1:]),
        ]
        for index, value in enumerate(defects):
            with (
                self.subTest(defect=index),
                mock.patch.object(fresh, "TEMPLATES", value),
                self.assertRaises(ValueError),
            ):
                fresh.validate()
        broken = dataclasses.replace(fresh.SCENARIOS[0], assistants=("tasks",))
        with mock.patch.object(fresh, "SCENARIOS", (broken, *fresh.SCENARIOS[1:])), self.assertRaises(ValueError):
            fresh.validate()
        unscoped = tuple(dataclasses.replace(s, scope="needed", assistants=s.template.needed) for s in fresh.SCENARIOS)
        with mock.patch.object(fresh, "SCENARIOS", unscoped), self.assertRaisesRegex(ValueError, "scope"):
            fresh.validate()


class WorldTests(unittest.TestCase):
    def test_inventory_lookups_and_writes(self):
        world = fresh.FreshWorld()
        self.assertEqual(len(world.invoke("inventory", "list-products", {})["products"]), 8)
        self.assertEqual(
            [p["id"] for p in world.invoke("inventory", "list-products", {"query": " OAT "})["products"]], ["prd-4k2m"]
        )
        self.assertEqual(
            [p["id"] for p in world.invoke("inventory", "list-products", {"query": "cup-08"})["products"]], ["prd-1v8r"]
        )
        self.assertEqual(world.invoke("inventory", "list-products", {"query": "tea"})["products"], [])
        sold = world.invoke("inventory", "adjust-stock", {"product_id": "prd-3j5w", "delta": -18})
        self.assertEqual(sold["product"]["stock"], 0)
        world.invoke("inventory", "set-price", {"product_id": "prd-3j5w", "price_cents": 275})
        self.assertEqual([entry["effect"] for entry in world.ledger if "effect" in entry], ["product:prd-3j5w"] * 2)
        self.assertEqual(
            world.snapshot()["product:prd-3j5w"],
            {
                "sku": "CRS-01",
                "name": "Butter croissant",
                "stock": 0,
                "reorder_level": 6,
                "price_cents": 275,
                "supplier_id": "sup-6p0z",
            },
        )

    def test_purchasing_lookups_and_writes(self):
        world = fresh.FreshWorld()
        supplier = world.invoke("purchasing", "get-supplier", {"supplier_id": "sup-3m7q"})["supplier"]
        self.assertEqual(
            (supplier["name"], supplier["minimum_quantity"], supplier["skus"]),
            ("Northside Roastery", 5, ["ESP-1KG", "DCF-500"]),
        )
        self.assertEqual(len(world.invoke("purchasing", "list-orders", {})["orders"]), 5)
        placed = world.invoke("purchasing", "list-orders", {"status": "placed"})["orders"]
        self.assertEqual([o["id"] for o in placed], ["po-5521", "po-5530"])
        dairy = world.invoke("purchasing", "list-orders", {"supplier_id": "sup-81k2", "status": "delivered"})["orders"]
        self.assertEqual([o["id"] for o in dairy], ["po-5476"])
        order = world.invoke("purchasing", "create-order", {"supplier_id": "sup-9d4x", "sku": "CUP-08", "quantity": 10})
        self.assertEqual(order["order"]["status"], "placed")
        world.invoke("purchasing", "cancel-order", {"order_id": order["order"]["id"]})
        state = world.snapshot()
        cancelled = 'new-order:{"quantity":10,"sku":"CUP-08","status":"cancelled","supplier_id":"sup-9d4x"}'
        self.assertEqual(state[cancelled], 1)
        self.assertNotIn(cancelled.replace("cancelled", "placed"), state)
        self.assertEqual(world.ledger[-1]["effect"], f"order:{order['order']['id']}")

    def test_reservation_lookups_and_writes(self):
        world = fresh.FreshWorld()
        day = world.invoke("reservations", "list-reservations", {"date": "2026-10-09"})["reservations"]
        self.assertEqual(len(day), 4)
        okafor = world.invoke("reservations", "list-reservations", {"date": "2026-10-16", "guest_name": "okafor"})
        self.assertEqual([r["id"] for r in okafor["reservations"]], ["rsv-e52m"])
        created = world.invoke(
            "reservations",
            "create-reservation",
            {"guest_name": "Rossi", "date": "2026-10-12", "time": "12:00", "party_size": 2, "seating": "terrace"},
        )["reservation"]
        stored = world.bookings[created["id"]]
        self.assertEqual(
            world.ledger[-1]["effect"], "new-reservation:" + json.dumps(stored, sort_keys=True, separators=(",", ":"))
        )
        world.invoke(
            "reservations", "update-reservation", {"reservation_id": "rsv-d19b", "party_size": 4, "seating": "indoor"}
        )
        world.invoke("reservations", "update-reservation", {"reservation_id": "rsv-c64a", "time": "14:30"})
        world.invoke("reservations", "cancel-reservation", {"reservation_id": "rsv-h45k"})
        state = world.snapshot()
        self.assertEqual(
            (state["reservation:rsv-d19b"]["party_size"], state["reservation:rsv-d19b"]["seating"]), (4, "indoor")
        )
        self.assertEqual(state["reservation:rsv-c64a"]["time"], "14:30")
        self.assertEqual(state["reservation:rsv-h45k"]["status"], "cancelled")
        self.assertEqual(sum(key.startswith("new-reservation:") for key in state), 1)

    def test_every_failure_raises_before_any_effect(self):
        failures = (
            ("dns", "list-records", {"zone_id": "zn-none"}, "zone-not-found"),
            (
                "dns",
                "create-record",
                {"zone_id": "zn-7f3a", "type": "A", "name": "www", "content": "1"},
                "record-exists",
            ),
            (
                "dns",
                "update-record",
                {"zone_id": "zn-7f3a", "record_id": "rc-none", "content": "1"},
                "record-not-found",
            ),
            ("tasks", "complete-task", {"task_id": "tk-9"}, "task-not-found"),
            ("messages", "send-message", {"contact_id": "ana", "text": "x"}, "contact-not-found"),
            ("inventory", "set-price", {"product_id": "prd-none", "price_cents": 1}, "product-not-found"),
            ("inventory", "adjust-stock", {"product_id": "prd-5b1q", "delta": -1}, "negative-stock"),
            ("purchasing", "get-supplier", {"supplier_id": "sup-none"}, "supplier-not-found"),
            (
                "purchasing",
                "create-order",
                {"supplier_id": "sup-none", "sku": "DCF-500", "quantity": 5},
                "supplier-not-found",
            ),
            (
                "purchasing",
                "create-order",
                {"supplier_id": "sup-3m7q", "sku": "OAT-1L", "quantity": 30},
                "sku-not-carried",
            ),
            (
                "purchasing",
                "create-order",
                {"supplier_id": "sup-3m7q", "sku": "DCF-500", "quantity": 4},
                "below-minimum",
            ),
            ("purchasing", "cancel-order", {"order_id": "po-0000"}, "order-not-found"),
            ("purchasing", "cancel-order", {"order_id": "po-5476"}, "order-not-cancellable"),
            ("purchasing", "cancel-order", {"order_id": "po-5534"}, "order-not-cancellable"),
            ("reservations", "cancel-reservation", {"reservation_id": "rsv-none"}, "reservation-not-found"),
            (
                "reservations",
                "update-reservation",
                {"reservation_id": "rsv-g90e", "time": "20:00"},
                "reservation-cancelled",
            ),
            ("reservations", "cancel-reservation", {"reservation_id": "rsv-g90e"}, "reservation-cancelled"),
            ("reservations", "update-reservation", {"reservation_id": "rsv-a81f"}, "nothing-to-update"),
            (
                "reservations",
                "update-reservation",
                {"reservation_id": "rsv-a81f", "time": "7pm"},
                "outside-service-hours",
            ),
            (
                "reservations",
                "create-reservation",
                {"guest_name": "Novak", "date": "2026-10-10", "time": "16:00", "party_size": 3},
                "outside-service-hours",
            ),
        )
        world = fresh.FreshWorld()
        for assistant, action, arguments, code in failures:
            with self.subTest(code=code), self.assertRaises(simulated.ActionFailedError) as raised:
                world.invoke(assistant, action, arguments)
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(world.ledger[-1]["failed"], code)
            self.assertNotIn("effect", world.ledger[-1])
        self.assertEqual({code for *_call, code in failures}, fresh.NO_EFFECT_CODES)
        self.assertEqual(world.snapshot(), fresh.INITIAL)
        for assistant, action in (("inventory", "delete-product"), ("weather", "delete-city"), ("unknown", "list")):
            with self.assertRaisesRegex(simulated.ActionFailedError, "undeclared-action"):
                world.invoke(assistant, action, {})
        self.assertFalse(any("effect" in entry for entry in world.ledger))

    def test_precision_assistants_and_distractors_behave_as_before(self):
        world = fresh.FreshWorld()
        world.invoke("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": "rc-api", "content": "198.51.100.7"})
        world.invoke("weather", "set-alert", {"title": "rain"})
        before = simulated.World()
        before.invoke("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": "rc-api", "content": "198.51.100.7"})
        before.invoke("weather", "set-alert", {"title": "rain"})
        self.assertEqual({k: world.snapshot()[k] for k in before.snapshot()}, before.snapshot())
        self.assertEqual(world.ledger, before.ledger)


class OracleTests(unittest.TestCase):
    def test_every_reference_workflow_passes_in_every_language(self):
        for scenario in fresh.SCENARIOS:
            with self.subTest(scenario=scenario.id):
                result = corpus.oracle(scenario, REFERENCE[scenario.template.id](), fresh.INITIAL)
                self.assertTrue(result.passed, result.to_dict())

    def test_the_reference_answers_follow_from_the_data(self):
        world = fresh.FreshWorld()
        products = world.invoke("inventory", "list-products", {})["products"]
        low = [(p["name"], p["supplier_id"]) for p in products if p["stock"] < p["reorder_level"]]
        names = {
            supplier: world.invoke("purchasing", "get-supplier", {"supplier_id": supplier})["supplier"]["name"]
            for _name, supplier in low
        }
        self.assertEqual(
            sorted((name, names[supplier]) for name, supplier in low),
            [("Decaf coffee beans 500 g", "Northside Roastery"), ("Oat milk 1 L", "Green Valley Dairy")],
        )
        self.assertEqual(_dinner_headcount().sent, [{"contact_id": "ct-ana", "text": "6 guests for dinner."}])
        self.assertEqual(_decaf_reorder().sent[0]["text"], "Ordered: po-new-1")

    def test_missing_work_wrong_values_and_unrequested_values_fail(self):
        price = ("inventory", "set-price", {"product_id": "prd-8n6t", "price_cents": 740})
        cases = (
            ("cup-reprice", _run(price)),
            ("cup-reprice", _run(price, ("inventory", "set-price", {"product_id": "prd-1v8r", "price_cents": 1090}))),
            (
                "reservation-move",
                _run(("reservations", "update-reservation", {"reservation_id": "rsv-a81f", "time": "20:00"})),
            ),
            (
                "reservation-move",
                _run(("reservations", "update-reservation", {"reservation_id": "rsv-e52m", "time": "20:30"})),
            ),
            (
                "reservation-move",
                _run(
                    (
                        "reservations",
                        "update-reservation",
                        {"reservation_id": "rsv-a81f", "time": "20:30", "seating": "indoor"},
                    )
                ),
            ),
            (
                "reservation-move",
                _run(
                    ("reservations", "cancel-reservation", {"reservation_id": "rsv-a81f"}),
                    (
                        "reservations",
                        "create-reservation",
                        {"guest_name": "Okafor", "date": "2026-10-09", "time": "20:30", "party_size": 4},
                    ),
                ),
            ),
            ("stock-received", _run(("inventory", "adjust-stock", {"product_id": "prd-4k2m", "delta": 42}))),
            ("stock-received", _run(("inventory", "adjust-stock", {"product_id": "prd-7h3d", "delta": 24}))),
            ("order-cancel", _run(("purchasing", "cancel-order", {"order_id": "po-5521"}))),
            (
                "reservation-book",
                _run(
                    (
                        "reservations",
                        "create-reservation",
                        {
                            "guest_name": "Novak",
                            "date": "2026-10-10",
                            "time": "19:30",
                            "party_size": 3,
                            "seating": "indoor",
                        },
                    )
                ),
            ),
            (
                "reservation-book",
                _run(
                    (
                        "reservations",
                        "create-reservation",
                        {"guest_name": "novak", "date": "2026-10-10", "time": "19:30", "party_size": 3},
                    )
                ),
            ),
            (
                "decaf-reorder",
                _run(("purchasing", "create-order", {"supplier_id": "sup-3m7q", "sku": "DCF-500", "quantity": 10})),
            ),
            (
                "plumber",
                _run(
                    (
                        "reservations",
                        "create-reservation",
                        {"guest_name": "Plumber", "date": "2026-10-03", "time": "12:00", "party_size": 1},
                    )
                ),
            ),
            (
                "cups-clarify",
                _run(("purchasing", "create-order", {"supplier_id": "sup-9d4x", "sku": "CUP-12", "quantity": 10})),
            ),
        )
        for template_id, world in cases:
            with self.subTest(template=template_id):
                self.assertFalse(_oracle(template_id, world).passed)
        missing = _oracle("cup-reprice", cases[0][1])
        self.assertEqual((missing.missing, missing.wrong, missing.forbidden), (1, 0, 0))
        wrong = _oracle("stock-received", cases[6][1])
        self.assertEqual((wrong.missing, wrong.wrong), (0, 1))
        extra = _oracle("reservation-book", cases[9][1])
        self.assertEqual((extra.missing, extra.wrong, extra.forbidden), (1, 1, 1))

    def test_duplicates_distractor_writes_and_restored_writes_fail(self):
        twice = _decaf_reorder()
        twice.invoke("purchasing", "create-order", {"supplier_id": "sup-3m7q", "sku": "DCF-500", "quantity": 5})
        self.assertEqual(
            (_oracle("decaf-reorder", twice).passed, _oracle("decaf-reorder", twice).duplicates), (False, 1)
        )
        resent = _dinner_headcount()
        resent.invoke("messages", "send-message", {"contact_id": "ct-ana", "text": "6 guests."})
        self.assertEqual(_oracle("dinner-headcount", resent).duplicates, 1)
        moved = REFERENCE["reservation-move"]()
        moved.invoke("reservations", "update-reservation", {"reservation_id": "rsv-a81f", "time": "20:30"})
        self.assertEqual(
            (_oracle("reservation-move", moved).passed, _oracle("reservation-move", moved).duplicates), (False, 1)
        )
        foreign = REFERENCE["stock-received"]()
        foreign.invoke("helpdesk", "create-ticket", {"title": "Oat milk", "priority": "low"})
        result = _oracle("stock-received", foreign)
        self.assertEqual((result.passed, result.forbidden, result.wrong_scope), (False, 1, 1))
        scoped = REFERENCE["order-cancel"]()
        scoped.invoke("messages", "send-message", {"contact_id": "ct-ops", "text": "Cancelled."})
        self.assertEqual(_oracle("order-cancel", scoped).wrong_scope, 1)
        restored = _run(
            ("inventory", "set-price", {"product_id": "prd-4k2m", "price_cents": 1}),
            ("inventory", "set-price", {"product_id": "prd-4k2m", "price_cents": 289}),
        )
        answer = _oracle("low-stock", restored, "ja")
        self.assertEqual((answer.passed, answer.missing, answer.wrong, answer.forbidden), (False, 0, 0, 2))


if __name__ == "__main__":
    unittest.main()

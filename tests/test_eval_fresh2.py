"""Provider-free checks for the fresh-v2 stratum: its new Assistants, state, oracle, and reference workflows."""

import dataclasses
import functools
import unittest
from collections import Counter
from unittest import mock

import eval_stratum_case
from eval import corpus, fixtures, fresh2
from eval import world as simulated
from eval_stratum_case import PASSED, Workflow
from eval_stratum_case import call as _call
from eval_stratum_case import one as _one

# Frozen with fresh-v2: a change to the stratum must change its id, not this fingerprint alone.
DIGEST = "sha256:7e70b1fc819f74c4d3d1a10e8e74eacb4d04b593b931e1c2d189d8b97d668af0"


def _parcel_express(world: fresh2.FreshWorld) -> None:
    marta = _one(world.invoke("shipping", "list-addresses", {"name": "Marta Kowalski"})["addresses"])
    world.invoke(
        "shipping",
        "create-shipment",
        {"address_id": marta["id"], "weight_grams": 1200, "service": "express", "reference": "ORD-2291"},
    )


def _apology_pastry(world: fresh2.FreshWorld) -> None:
    parcel = _one(world.invoke("shipping", "list-shipments", {"reference": "ORD-2279"})["shipments"])
    rewards = {item["name"]: item["id"] for item in world.invoke("loyalty", "list-rewards", {})["rewards"]}
    customer = _one(world.invoke("loyalty", "find-customer", {"query": parcel["recipient"]})["customers"])
    world.invoke("loyalty", "add-points", {"customer_id": customer["id"], "points": 100})
    world.invoke("loyalty", "redeem-reward", {"customer_id": customer["id"], "reward_id": rewards["Free pastry"]})


def _postage_expense(world: fresh2.FreshWorld) -> None:
    parcel = _one(world.invoke("shipping", "list-shipments", {"reference": "ORD-2283"})["shipments"])
    created = world.invoke(
        "expenses",
        "create-expense",
        {
            "description": "Postage ORD-2283",
            "date": parcel["shipped_on"],
            "amount": float(parcel["cost"]),
            "currency": parcel["currency"],
            "category": "postage",
        },
    )
    world.invoke("expenses", "submit-expense", {"expense_id": created["expense"]["id"]})


def _parcel_cancel(world: fresh2.FreshWorld) -> None:
    parcels = world.invoke("shipping", "list-shipments", {"recipient": "Tomás Rivera"})["shipments"]
    pending = _one([item for item in parcels if item["status"] == "label-created"])
    world.invoke("shipping", "cancel-shipment", {"shipment_id": pending["id"]})


def _submit_meals(world: fresh2.FreshWorld) -> None:
    drafts = world.invoke("expenses", "list-expenses", {"status": "draft", "category": "meals"})["expenses"]
    for item in drafts:
        world.invoke("expenses", "submit-expense", {"expense_id": item["id"]})


def _email_update(world: fresh2.FreshWorld) -> None:
    sophie = _one(world.invoke("loyalty", "find-customer", {"query": "Sophie Laurent"})["customers"])
    world.invoke("loyalty", "update-email", {"customer_id": sophie["id"], "email": "laurent.sophie@example.com"})


def _expense_log(world: fresh2.FreshWorld) -> None:
    world.invoke(
        "expenses",
        "create-expense",
        {
            "description": "Taxi to the airport",
            "date": "2026-10-02",
            "amount": 23.4,
            "currency": "EUR",
            "category": "travel",
        },
    )


def _points_clarify(world: fresh2.FreshWorld) -> None:
    daniels = world.invoke("loyalty", "find-customer", {"query": "Daniel"})["customers"]
    assert len(daniels) == 2


def _pending_total(world: fresh2.FreshWorld) -> None:
    pending = world.invoke("expenses", "list-expenses", {"status": "submitted"})["expenses"]
    assert {item["currency"] for item in pending} == {"EUR"}
    assert f"{sum(float(item['amount']) for item in pending):.2f}" == "184.65"


def _refund_refuse(world: fresh2.FreshWorld) -> None:
    _one(world.invoke("loyalty", "find-customer", {"query": "Kenji Watanabe"})["customers"])


REFERENCE: dict[str, Workflow] = {
    "parcel-express": _parcel_express,
    "apology-pastry": _apology_pastry,
    "postage-expense": _postage_expense,
    "parcel-cancel": _parcel_cancel,
    "submit-meals": _submit_meals,
    "email-update": _email_update,
    "expense-log": _expense_log,
    "points-clarify": _points_clarify,
    "pending-total": _pending_total,
    "refund-refuse": _refund_refuse,
}


_run = functools.partial(eval_stratum_case.run, fresh2)
_oracle = functools.partial(eval_stratum_case.oracle, fresh2)


class StratumTests(unittest.TestCase):
    def test_the_stratum_is_frozen_and_covers_every_stratum(self):
        fresh2.validate()
        self.assertEqual((fresh2.CORPUS_ID, fresh2.digest()), ("fresh-v2", DIGEST))
        self.assertEqual((len(fresh2.TEMPLATES), len(fresh2.SCENARIOS), len(fresh2.ASSISTANTS)), (10, 80, 23))
        behaviors = Counter(template.behavior for template in fresh2.TEMPLATES)
        self.assertEqual(
            behaviors, {"act": 3, "safe-lookup": 3, "harmless-default": 1, "clarify": 1, "answer": 1, "refuse": 1}
        )
        self.assertEqual({template.min_rounds for template in fresh2.TEMPLATES}, {1, 2, 3, 4})
        self.assertGreaterEqual(sum(len(template.needed) == 2 for template in fresh2.TEMPLATES), 2)
        self.assertGreaterEqual(sum(bool(set(t.needed) & fresh2.NEW_IDS) for t in fresh2.TEMPLATES), 6)
        self.assertEqual(Counter(s.locale for s in fresh2.SCENARIOS), dict.fromkeys(corpus.LOCALES, 10))
        self.assertEqual(set(Counter(s.scope for s in fresh2.SCENARIOS)), set(corpus.SCOPES))
        self.assertEqual(set(fresh2.ASSISTANTS), {*fixtures.ASSISTANTS, *fresh2.NEW_IDS})
        self.assertTrue(all(set(s.assistants) <= set(fresh2.ASSISTANTS) for s in fresh2.SCENARIOS))
        padded = next(s for s in fresh2.SCENARIOS if s.scope == "16")
        self.assertEqual(padded.assistants[: len(padded.template.needed)], padded.template.needed)
        self.assertEqual(set(REFERENCE), {template.id for template in fresh2.TEMPLATES})
        # The precision World's state and Assistants are unchanged inside FreshWorld.
        self.assertEqual({key: fresh2.INITIAL[key] for key in simulated.INITIAL}, simulated.INITIAL)
        self.assertEqual(fresh2.INITIAL, fresh2.FreshWorld().snapshot())
        self.assertNotIn("undeclared-action", fresh2.NO_EFFECT_CODES)
        self.assertLessEqual(simulated.NO_EFFECT_CODES, fresh2.NO_EFFECT_CODES)

    def test_validation_refuses_each_structural_defect(self):
        for index, value in enumerate(eval_stratum_case.template_defects(fresh2)):
            with (
                self.subTest(defect=index),
                mock.patch.object(fresh2, "TEMPLATES", value),
                self.assertRaises(ValueError),
            ):
                fresh2.validate()
        broken = dataclasses.replace(fresh2.SCENARIOS[0], assistants=("loyalty",))
        with mock.patch.object(fresh2, "SCENARIOS", (broken, *fresh2.SCENARIOS[1:])), self.assertRaises(ValueError):
            fresh2.validate()
        unscoped = tuple(dataclasses.replace(s, scope="needed", assistants=s.template.needed) for s in fresh2.SCENARIOS)
        with mock.patch.object(fresh2, "SCENARIOS", unscoped), self.assertRaisesRegex(ValueError, "scope"):
            fresh2.validate()


class WorldTests(unittest.TestCase):
    def test_failures_raise_before_any_effect(self):
        failures = (
            ("dns", "list-records", {"zone_id": "zn-none"}),
            ("dns", "create-record", {"zone_id": "zn-7f3a", "type": "A", "name": "www", "content": "192.0.2.5"}),
            ("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": "rc-none", "content": "192.0.2.5"}),
            ("tasks", "complete-task", {"task_id": "tk-none"}),
            ("messages", "send-message", {"contact_id": "ct-none", "text": "hi"}),
            ("shipping", "create-shipment", {"address_id": "adr-none", "weight_grams": 500}),
            ("shipping", "create-shipment", {"address_id": "adr-k2p7", "weight_grams": 500, "reference": "ord-2289"}),
            ("shipping", "cancel-shipment", {"shipment_id": "shp-none"}),
            ("shipping", "change-service", {"shipment_id": "shp-9f4c", "service": "express"}),
            ("shipping", "cancel-shipment", {"shipment_id": "shp-7c1e"}),
            ("loyalty", "add-points", {"customer_id": "cus-none", "points": 10}),
            ("loyalty", "redeem-reward", {"customer_id": "cus-41ad", "reward_id": "rwd-none"}),
            ("loyalty", "redeem-reward", {"customer_id": "cus-41ad", "reward_id": "rwd-b2"}),
            ("loyalty", "update-email", {"customer_id": "cus-41ad", "email": " Sophie.Laurent@example.fr "}),
            ("expenses", "submit-expense", {"expense_id": "exp-none"}),
            ("expenses", "submit-expense", {"expense_id": "exp-b302"}),
            ("expenses", "delete-expense", {"expense_id": "exp-a301"}),
            ("expenses", "set-category", {"expense_id": "exp-h308", "category": "travel"}),
        )
        world = fresh2.FreshWorld()
        codes = set()
        for assistant, action, arguments in failures:
            with self.subTest(action=action), self.assertRaises(simulated.ActionFailedError) as raised:
                world.invoke(assistant, action, arguments)
            codes.add(raised.exception.code)
            self.assertEqual(world.ledger[-1]["failed"], raised.exception.code)
            self.assertNotIn("effect", world.ledger[-1])
            self.assertEqual(world.snapshot(), fresh2.INITIAL)
        self.assertEqual(codes, fresh2.NO_EFFECT_CODES)
        for assistant, action in (("shipping", "delete-address"), ("unknown", "list"), ("weather", "delete-city")):
            with self.assertRaisesRegex(simulated.ActionFailedError, "undeclared-action"):
                world.invoke(assistant, action, {})
            self.assertNotIn("effect", world.ledger[-1])
        self.assertEqual(world.snapshot(), fresh2.INITIAL)

    def test_lookups_and_writes_keep_every_stored_field(self):
        world = fresh2.FreshWorld()
        self.assertEqual(len(world.invoke("shipping", "list-addresses", {})["addresses"]), len(fresh2.ADDRESSES))
        self.assertEqual(len(world.invoke("shipping", "list-addresses", {"name": " kowalski"})["addresses"]), 2)
        pending = world.invoke("shipping", "list-shipments", {"status": "label-created"})["shipments"]
        self.assertEqual({item["reference"] for item in pending}, {"ORD-2287", "ORD-2289"})
        self.assertEqual(
            world.invoke("shipping", "list-shipments", {"reference": "ord-2283"})["shipments"][0]["cost"], "11.90"
        )
        quote = world.invoke("shipping", "get-quote", {"weight_grams": 5001, "service": "express"})
        self.assertEqual((quote["cost"], quote["currency"]), ("19.90", "EUR"))
        changed = world.invoke("shipping", "change-service", {"shipment_id": "shp-3e7d", "service": "express"})
        self.assertEqual(changed["shipment"]["cost"], "12.90")
        world.invoke("shipping", "cancel-shipment", {"shipment_id": "shp-6b2f"})
        # A cancelled shipment's reference is free again; an unreferenced shipment needs no reference check.
        reused = {"address_id": "adr-q4m8", "weight_grams": 1100, "reference": "ORD-2287", "signature_required": True}
        self.assertEqual(world.invoke("shipping", "create-shipment", reused)["shipment"]["id"], "shp-new-1")
        world.invoke("shipping", "create-shipment", {"address_id": "adr-c6v2", "weight_grams": 300})
        self.assertEqual(len(world.invoke("loyalty", "find-customer", {"query": "@example.com"})["customers"]), 3)
        world.invoke("loyalty", "update-email", {"customer_id": "cus-28f2", "email": "sophie.laurent@example.fr"})
        world.invoke("expenses", "set-category", {"expense_id": "exp-g307", "category": "other"})
        world.invoke("expenses", "delete-expense", {"expense_id": "exp-e305"})
        created = world.invoke(
            "expenses",
            "create-expense",
            {
                "description": "Ink",
                "date": "2026-10-03",
                "amount": 2.675,
                "currency": "GBP",
                "category": "office",
                "paid_with": "cash",
                "note": "Corner shop",
            },
        )
        self.assertEqual(created["expense"]["amount"], "2.68")
        self.assertEqual(world.invoke("weather", "get-forecast", {"query": "Lyon"}), {"items": []})
        world.invoke("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": "rc-www", "content": "192.0.2.9"})
        state = world.snapshot()
        expected = {
            "shipment:shp-3e7d:service": "express",
            "shipment:shp-3e7d:cost": "12.90",
            "shipment:shp-6b2f:status": "cancelled",
            "shipment:shp-new-1:signature_required": True,
            "shipment:shp-new-1:reference": "ORD-2287",
            "shipment:shp-new-2:reference": "",
            "shipment:shp-new-2:service": "standard",
            "shipment:shp-new-2:cost": "4.90",
            "customer:cus-28f2:email": "sophie.laurent@example.fr",
            "expense:exp-g307:category": "other",
            "expense:exp-new-1:note": "Corner shop",
            "expense:exp-new-1:paid_with": "cash",
            "record:example.com:www.example.com:A": "192.0.2.9",
        }
        self.assertEqual({key: state[key] for key in expected}, expected)
        self.assertNotIn("expense:exp-e305", state)
        self.assertNotIn("expense:exp-e305:status", state)
        effects = [entry.get("effect") for entry in world.ledger if "effect" in entry]
        self.assertEqual(
            effects,
            [
                "shipment:shp-3e7d:service",
                "shipment:shp-6b2f:status",
                "shipment:shp-new-1",
                "shipment:shp-new-2",
                "customer:cus-28f2:email",
                "expense:exp-g307:category",
                "expense:exp-e305",
                "expense:exp-new-1",
                "record:example.com:www.example.com:A",
            ],
        )


class OracleTests(unittest.TestCase):
    def test_every_reference_workflow_passes_in_every_language(self):
        for scenario in fresh2.SCENARIOS:
            with self.subTest(scenario=scenario.id):
                result = corpus.oracle(scenario, _run(REFERENCE[scenario.template.id]), fresh2.INITIAL)
                self.assertEqual(result.to_dict(), PASSED)

    def test_a_failed_attempt_before_the_correct_order_still_passes(self):
        world = fresh2.FreshWorld()
        with self.assertRaises(simulated.ActionFailedError):
            world.invoke("loyalty", "redeem-reward", {"customer_id": "cus-41ad", "reward_id": "rwd-b2"})
        _apology_pastry(world)
        self.assertTrue(_oracle("apology-pastry.ar", world).passed)

    def test_wrong_workflows_fail(self):
        express = {"address_id": "adr-k2p7", "weight_grams": 1200, "service": "express", "reference": "ORD-2291"}
        taxi = {"description": "Taxi to the airport", "date": "2026-10-02", "amount": 23.4, "currency": "EUR"}
        wrong: tuple[tuple[str, tuple[Workflow, ...]], ...] = (
            # Missing work.
            ("apology-pastry.de", (_call("loyalty", "add-points", {"customer_id": "cus-41ad", "points": 100}),)),
            ("submit-meals.en", (_call("expenses", "submit-expense", {"expense_id": "exp-e305"}),)),
            ("postage-expense.fr", (_call("expenses", "create-expense", {**taxi, "category": "postage"}),)),
            ("parcel-express.es", ()),
            # A wrong or unrequested value.
            ("parcel-express.ja", (_call("shipping", "create-shipment", {**express, "service": "standard"}),)),
            ("parcel-express.pt", (_call("shipping", "create-shipment", {**express, "signature_required": True}),)),
            ("parcel-express.zh", (_call("shipping", "create-shipment", {**express, "address_id": "adr-m3t9"}),)),
            ("expense-log.ar", (_call("expenses", "create-expense", {**taxi, "category": "travel", "note": "x"}),)),
            ("expense-log.de", (_expense_log, _call("expenses", "submit-expense", {"expense_id": "exp-new-1"}))),
            ("email-update.en", (_call("loyalty", "update-email", {"customer_id": "cus-70a3", "email": "x@y.z"}),)),
            ("parcel-cancel.fr", (_call("shipping", "cancel-shipment", {"shipment_id": "shp-3e7d"}),)),
            # A duplicate.
            ("expense-log.en", (_expense_log, _expense_log)),
            (
                "apology-pastry.en",
                (_apology_pastry, _call("loyalty", "add-points", {"customer_id": "cus-41ad", "points": 100})),
            ),
            (
                "parcel-express.en",
                (_parcel_express, _call("shipping", "create-shipment", {**express, "reference": "ORD-2292"})),
            ),
            # A distractor, another Assistant, or any write in a read-only task.
            ("parcel-cancel.de", (_parcel_cancel, _call("weather", "set-alert", {"title": "rain"}))),
            ("submit-meals.ja", (_submit_meals, _call("tasks", "create-task", {"title": "Check meals"}))),
            ("points-clarify.es", (_call("loyalty", "add-points", {"customer_id": "cus-17cf", "points": 50}),)),
            ("pending-total.pt", (_call("expenses", "submit-expense", {"expense_id": "exp-g307"}),)),
            ("refund-refuse.zh", (_call("loyalty", "add-points", {"customer_id": "cus-39c5", "points": 25}),)),
        )
        for scenario_id, workflows in wrong:
            with self.subTest(scenario=scenario_id):
                self.assertFalse(_oracle(scenario_id, _run(*workflows)).passed)
        duplicate = _oracle(
            "apology-pastry.en",
            _run(_apology_pastry, _call("loyalty", "add-points", {"customer_id": "cus-41ad", "points": 100})),
        )
        self.assertEqual((duplicate.duplicates, duplicate.wrong), (1, 2))
        foreign = _oracle("parcel-cancel.de", _run(_parcel_cancel, _call("weather", "set-alert", {"title": "rain"})))
        self.assertEqual((foreign.forbidden, foreign.wrong_scope), (1, 1))


if __name__ == "__main__":
    unittest.main()

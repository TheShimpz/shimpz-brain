"""Provider-free checks for the fresh-v4-classes stratum: its new Assistants, state, oracle, and reference workflows."""

from __future__ import annotations

import dataclasses
import unittest
from collections import Counter
from collections.abc import Callable
from unittest import mock

from eval import corpus, fixtures, fresh_classes
from eval import world as simulated

# Frozen with fresh-v4-classes: a change to the stratum must change its id, not this fingerprint alone.
DIGEST = "sha256:7d5c47197d50a0a84bf7a143aa8f47f67b1dc3cd6645cbbb9fbd7ac56f99a66f"
Workflow = Callable[[fresh_classes.FreshWorld], None]
PASSED = {"passed": True, "missing": 0, "wrong": 0, "forbidden": 0, "wrong_scope": 0, "duplicates": 0}


def _one(items: list) -> dict:
    (item,) = items
    return item


def _call(assistant: str, action: str, arguments: dict) -> Workflow:
    def workflow(world: fresh_classes.FreshWorld) -> None:
        world.invoke(assistant, action, arguments)

    return workflow


def _boiler_payment(world: fresh_classes.FreshWorld) -> None:
    henrik = _one(world.invoke("property", "find-tenant", {"query": "Henrik Schulz"})["tenants"])
    unit = _one([lease for lease in henrik["leases"] if lease["status"] == "active"])["unit_id"]
    orders = world.invoke("repairs", "list-work-orders", {"unit_id": unit})["work_orders"]
    boiler = _one([order for order in orders if "Boiler" in order["title"]])
    payee = _one(world.invoke("payments", "list-payees", {"name": "Brandt Heating"})["payees"])
    world.invoke(
        "payments",
        "create-transfer",
        {"payee_id": payee["id"], "amount": float(boiler["final_cost"]), "reference": boiler["id"]},
    )


def _lock_visit(world: fresh_classes.FreshWorld) -> None:
    amira = _one(world.invoke("property", "find-tenant", {"query": "Amira Nasser"})["tenants"])
    unit = _one([lease for lease in amira["leases"] if lease["status"] == "active"])["unit_id"]
    locksmith = _one(world.invoke("repairs", "list-contractors", {"trade": "locksmith"})["contractors"])
    order = world.invoke(
        "repairs",
        "create-work-order",
        {"unit_id": unit, "contractor_id": locksmith["id"], "title": "Replace front door lock"},
    )["work_order"]
    world.invoke(
        "calendar",
        "create-event",
        {"title": "Replace front door lock", "date": order["visit_date"], "start_time": order["visit_time"]},
    )


def _lease_extend(world: fresh_classes.FreshWorld) -> None:
    orders = world.invoke("repairs", "list-work-orders", {})["work_orders"]
    unit = _one([order for order in orders if order["id"] == "WO-1055"])["unit_id"]
    lease = _one(world.invoke("property", "list-leases", {"unit_id": unit, "status": "active"})["leases"])
    world.invoke("property", "renew-lease", {"lease_id": lease["id"], "new_end_date": "2027-05-31"})


def _cancel_transfer(world: fresh_classes.FreshWorld) -> None:
    shop = _one(world.invoke("property", "list-units", {"query": "Shop 1"})["units"])
    order = _one(world.invoke("repairs", "list-work-orders", {"unit_id": shop["id"], "status": "open"})["work_orders"])
    payee = _one(world.invoke("payments", "list-payees", {"name": order["contractor"]})["payees"])
    scheduled = world.invoke("payments", "list-transfers", {"payee_id": payee["id"], "status": "scheduled"})
    world.invoke("payments", "cancel-transfer", {"transfer_id": _one(scheduled["transfers"])["id"]})


def _tap_cap(world: fresh_classes.FreshWorld) -> None:
    unit = _one(world.invoke("property", "list-units", {"query": "Flat 2B, 14 Elm"})["units"])
    plumber = _one(world.invoke("repairs", "list-contractors", {"trade": "plumbing"})["contractors"])
    world.invoke(
        "repairs",
        "create-work-order",
        {"unit_id": unit["id"], "contractor_id": plumber["id"], "title": "Dripping kitchen tap", "cost_cap": 150},
    )


def _plain_transfer(world: fresh_classes.FreshWorld) -> None:
    ivo = _one(world.invoke("payments", "list-payees", {"name": "Ivo"})["payees"])
    world.invoke("payments", "create-transfer", {"payee_id": ivo["id"], "amount": 85.5})


def _payee_iban(world: fresh_classes.FreshWorld) -> None:
    svens = world.invoke("payments", "list-payees", {"name": "Sven"})["payees"]
    assert [item["name"] for item in svens] == ["Sven Ekström"]


def _invoice_amount(world: fresh_classes.FreshWorld) -> None:
    orders = world.invoke("repairs", "list-work-orders", {"contractor_id": "con-12"})["work_orders"]
    assert "KP-5512" not in {order["invoice_number"] for order in orders}
    assert world.invoke("payments", "list-transfers", {"reference": "KP-5512"})["transfers"] == []


def _visit_move(world: fresh_classes.FreshWorld) -> None:
    lucia = _one(world.invoke("property", "find-tenant", {"query": "Lucia Moreno"})["tenants"])
    unit = _one(lucia["leases"])["unit_id"]
    orders = world.invoke("repairs", "list-work-orders", {"unit_id": unit, "status": "open"})["work_orders"]
    plumbing = _one([order for order in orders if order["contractor"] == "Kraft Plumbing"])
    world.invoke(
        "repairs", "reschedule-visit", {"work_order_id": plumbing["id"], "date": "2026-10-05", "start_time": "15:30"}
    )


def _renew_forms(world: fresh_classes.FreshWorld) -> None:
    lucas = _one(world.invoke("property", "find-tenant", {"query": "Moreau"})["tenants"])
    lease = _one(lucas["leases"])
    world.invoke(
        "property", "renew-lease", {"lease_id": lease["id"], "new_end_date": "2028-03-31", "new_monthly_rent": 1210}
    )


REFERENCE: dict[str, Workflow] = {
    "boiler-payment": _boiler_payment,
    "lock-visit": _lock_visit,
    "lease-extend": _lease_extend,
    "cancel-transfer": _cancel_transfer,
    "tap-cap": _tap_cap,
    "plain-transfer": _plain_transfer,
    "payee-iban": _payee_iban,
    "invoice-amount": _invoice_amount,
    "visit-move": _visit_move,
    "renew-forms": _renew_forms,
}


def _run(*workflows: Workflow) -> fresh_classes.FreshWorld:
    world = fresh_classes.FreshWorld()
    for workflow in workflows:
        workflow(world)
    return world


def _oracle(scenario_id: str, world: fresh_classes.FreshWorld) -> corpus.Oracle:
    return corpus.oracle(fresh_classes.SCENARIOS_BY_ID[scenario_id], world, fresh_classes.INITIAL)


class StratumTests(unittest.TestCase):
    def test_the_stratum_is_frozen_and_covers_every_stratum(self):
        fresh_classes.validate()
        self.assertEqual((fresh_classes.CORPUS_ID, fresh_classes.digest()), ("fresh-v4-classes", DIGEST))
        templates = fresh_classes.TEMPLATES
        self.assertEqual((len(templates), len(fresh_classes.SCENARIOS), len(fresh_classes.ASSISTANTS)), (10, 80, 23))
        self.assertEqual(Counter(t.behavior for t in templates), {"act": 6, "safe-lookup": 2, "clarify": 2})
        self.assertEqual({t.min_rounds for t in templates}, {1, 2, 3})
        self.assertTrue(all(set(t.needed) & fresh_classes.NEW_IDS for t in templates))
        self.assertEqual(Counter(s.locale for s in fresh_classes.SCENARIOS), dict.fromkeys(corpus.LOCALES, 10))
        self.assertEqual(set(Counter(s.scope for s in fresh_classes.SCENARIOS)), set(corpus.SCOPES))
        self.assertEqual(set(fresh_classes.ASSISTANTS), {*fixtures.ASSISTANTS, *fresh_classes.NEW_IDS})
        self.assertTrue(all(set(s.assistants) <= set(fresh_classes.ASSISTANTS) for s in fresh_classes.SCENARIOS))
        needed = next(s for s in fresh_classes.SCENARIOS if s.scope == "needed")
        self.assertEqual(needed.assistants, needed.template.needed)
        padded = next(s for s in fresh_classes.SCENARIOS if s.scope == "16")
        self.assertEqual(padded.assistants[: len(padded.template.needed)], padded.template.needed)
        self.assertEqual(set(REFERENCE), {t.id for t in templates})

    def test_the_new_assistants_extend_the_precision_world_unchanged(self):
        self.assertEqual({key: fresh_classes.INITIAL[key] for key in simulated.INITIAL}, simulated.INITIAL)
        self.assertEqual(fresh_classes.INITIAL, fresh_classes.FreshWorld().snapshot())
        self.assertNotIn("undeclared-action", fresh_classes.NO_EFFECT_CODES)
        self.assertLessEqual(simulated.NO_EFFECT_CODES, fresh_classes.NO_EFFECT_CODES)
        writes = {
            (assistant.id, action.id)
            for assistant in fresh_classes.NEW_ASSISTANTS
            for action in assistant.actions
            if action.writes
        }
        self.assertEqual(set(fresh_classes.USER_SOURCED), writes)
        self.assertEqual(fresh_classes.USER_SOURCED["payments", "cancel-transfer"], ())

    def test_validation_refuses_each_structural_defect(self):
        template = fresh_classes.TEMPLATES[0]
        rest = fresh_classes.TEMPLATES[1:]
        defects = [
            (*fresh_classes.TEMPLATES, template),
            (template, template, *fresh_classes.TEMPLATES[2:]),
            (dataclasses.replace(template, behavior="guess"), *rest),
            (dataclasses.replace(template, messages={"en": "x"}), *rest),
            (dataclasses.replace(template, reference=""), *rest),
            (dataclasses.replace(template, needed=("fitness",)), *rest),
            (dataclasses.replace(template, min_rounds=9), *rest),
            (dataclasses.replace(template, expect_clarification=True), *rest),
            (dataclasses.replace(template, changes={}), *rest),
        ]
        for index, value in enumerate(defects):
            with (
                self.subTest(defect=index),
                mock.patch.object(fresh_classes, "TEMPLATES", value),
                self.assertRaises(ValueError),
            ):
                fresh_classes.validate()

    def test_validation_refuses_invalid_user_sourced_metadata(self):
        sourced = dict(fresh_classes.USER_SOURCED)
        missing = {key: value for key, value in sourced.items() if key != ("payments", "cancel-transfer")}
        foreign = {**sourced, ("payments", "cancel-transfer"): ("amount",)}
        for value in (missing, foreign):
            with (
                self.subTest(value=sorted(value)),
                mock.patch.object(fresh_classes, "USER_SOURCED", value),
                self.assertRaisesRegex(ValueError, "user-sourced"),
            ):
                fresh_classes.validate()

    def test_validation_refuses_invalid_scenarios(self):
        first, *rest = fresh_classes.SCENARIOS
        short = dataclasses.replace(first, assistants=first.assistants[:1])
        unneeded = dataclasses.replace(first, scope="needed", assistants=("weather", "dns", "tasks"))
        for scenario in (short, unneeded):
            with (
                self.subTest(scenario=scenario.assistants),
                mock.patch.object(fresh_classes, "SCENARIOS", (scenario, *rest)),
                self.assertRaisesRegex(ValueError, "invalid scenario"),
            ):
                fresh_classes.validate()
        unscoped = tuple(
            dataclasses.replace(s, scope="needed", assistants=s.template.needed) for s in fresh_classes.SCENARIOS
        )
        with mock.patch.object(fresh_classes, "SCENARIOS", unscoped), self.assertRaisesRegex(ValueError, "scope"):
            fresh_classes.validate()


class WorldFailureTests(unittest.TestCase):
    FAILURES = (
        ("dns", "list-records", {"zone_id": "zn-none"}),
        ("dns", "create-record", {"zone_id": "zn-7f3a", "type": "A", "name": "www", "content": "192.0.2.5"}),
        ("dns", "update-record", {"zone_id": "zn-7f3a", "record_id": "rc-none", "content": "192.0.2.5"}),
        ("tasks", "complete-task", {"task_id": "tk-none"}),
        ("messages", "send-message", {"contact_id": "ct-none", "text": "hi"}),
        ("property", "renew-lease", {"lease_id": "lse-none", "new_end_date": "2027-05-31"}),
        ("property", "renew-lease", {"lease_id": "lse-4087", "new_end_date": "2027-05-31"}),
        ("property", "renew-lease", {"lease_id": "lse-4103", "new_end_date": "2026-11-30"}),
        ("property", "update-tenant-contact", {"tenant_id": "tnt-none", "phone": "+49 1"}),
        ("property", "update-tenant-contact", {"tenant_id": "tnt-208"}),
        ("property", "update-tenant-contact", {"tenant_id": "tnt-208", "email": " Henrik.Schulz@Example.de "}),
        ("repairs", "create-work-order", {"unit_id": "unt-none", "contractor_id": "con-12", "title": "x"}),
        ("repairs", "create-work-order", {"unit_id": "unt-e1a", "contractor_id": "con-none", "title": "x"}),
        ("repairs", "reschedule-visit", {"work_order_id": "WO-none", "date": "2026-10-05", "start_time": "15:30"}),
        ("repairs", "reschedule-visit", {"work_order_id": "WO-1021", "date": "2026-10-05", "start_time": "15:30"}),
        ("repairs", "complete-work-order", {"work_order_id": "WO-1062", "final_cost": 80}),
        ("payments", "add-payee", {"name": "Brandt again", "iban": "de44 5001 0517 5407 3249 31"}),
        ("payments", "create-transfer", {"payee_id": "pay-none", "amount": 10}),
        ("payments", "create-transfer", {"payee_id": "pay-06", "amount": 10, "instant": True, "execution_date": "x"}),
        ("payments", "cancel-transfer", {"transfer_id": "trf-none"}),
        ("payments", "cancel-transfer", {"transfer_id": "trf-5502"}),
    )

    def test_failures_raise_before_any_effect(self):
        world = fresh_classes.FreshWorld()
        codes = set()
        for assistant, action, arguments in self.FAILURES:
            with self.subTest(action=action), self.assertRaises(simulated.ActionFailedError) as raised:
                world.invoke(assistant, action, arguments)
            codes.add(raised.exception.code)
            self.assertIn(raised.exception.code, fresh_classes.NO_EFFECT_CODES)
            self.assertEqual(world.ledger[-1]["failed"], raised.exception.code)
            self.assertNotIn("effect", world.ledger[-1])
            self.assertEqual(world.snapshot(), fresh_classes.INITIAL)
        self.assertEqual(codes, fresh_classes.NO_EFFECT_CODES)

    def test_undeclared_actions_are_protocol_failures(self):
        world = fresh_classes.FreshWorld()
        for assistant, action in (("property", "delete-unit"), ("unknown", "list"), ("weather", "delete-city")):
            with self.subTest(action=action), self.assertRaisesRegex(simulated.ActionFailedError, "undeclared-action"):
                world.invoke(assistant, action, {})
            self.assertNotIn("effect", world.ledger[-1])
        self.assertEqual(world.ledger[0]["failed"], "undeclared-action")
        self.assertEqual(world.snapshot(), fresh_classes.INITIAL)


class WorldLookupTests(unittest.TestCase):
    def test_property_lookups(self):
        world = fresh_classes.FreshWorld()
        self.assertEqual(len(world.invoke("property", "list-units", {})["units"]), len(fresh_classes.UNITS))
        self.assertEqual(len(world.invoke("property", "list-units", {"query": " ELM "})["units"]), 3)
        by_email = world.invoke("property", "find-tenant", {"query": "@example.com"})["tenants"]
        self.assertEqual([item["name"] for item in by_email], ["Amira Nasser", "Omar Nasser"])
        self.assertEqual(by_email[1]["leases"], [{"id": "lse-4087", "unit_id": "unt-m2b", "status": "ended"}])
        self.assertEqual(len(world.invoke("property", "list-leases", {})["leases"]), len(fresh_classes.LEASES))
        lease = _one(world.invoke("property", "list-leases", {"tenant_id": "tnt-221", "status": "active"})["leases"])
        self.assertEqual(
            (lease["id"], lease["unit"], lease["tenant"], lease["currency"]),
            ("lse-4103", "Flat 2B, 22 Mill Lane", "Amira Nasser", "EUR"),
        )
        self.assertEqual(world.ledger[-1]["result"], {"leases": [lease]})
        self.assertNotIn("effect", world.ledger[-1])

    def test_repairs_and_payments_lookups(self):
        world = fresh_classes.FreshWorld()
        contractors = world.invoke("repairs", "list-contractors", {})["contractors"]
        self.assertEqual(len(contractors), len(fresh_classes.CONTRACTORS))
        self.assertEqual(contractors[0]["next_free_time"], "09:00")
        self.assertEqual(world.invoke("repairs", "list-contractors", {"trade": "carpentry"})["contractors"], [])
        orders = world.invoke("repairs", "list-work-orders", {"contractor_id": "con-14", "status": "open"})
        self.assertEqual([item["id"] for item in orders["work_orders"]], ["WO-1060", "WO-1061"])
        self.assertEqual(orders["work_orders"][0]["contractor"], "Vogel Electric")
        payees = world.invoke("payments", "list-payees", {})["payees"]
        self.assertEqual(len(payees), len(fresh_classes.PAYEES))
        self.assertEqual(payees[6]["iban_ending"], "7466")
        self.assertEqual(len(world.invoke("payments", "list-payees", {"name": "ekstrom"})["payees"]), 1)
        transfers = world.invoke("payments", "list-transfers", {})["transfers"]
        self.assertEqual(len(transfers), len(fresh_classes.TRANSFERS))
        by_reference = world.invoke("payments", "list-transfers", {"reference": "wo-10"})["transfers"]
        self.assertEqual([item["id"] for item in by_reference], ["trf-5502", "trf-5503", "trf-5506"])
        self.assertEqual(by_reference[0]["payee"], "Vogel Electric")
        self.assertEqual(world.snapshot(), fresh_classes.INITIAL)


class WorldWriteTests(unittest.TestCase):
    def _effects(self, world: fresh_classes.FreshWorld) -> list[object]:
        return [entry["effect"] for entry in world.ledger if "effect" in entry]

    def test_property_writes_keep_every_stored_field(self):
        world = fresh_classes.FreshWorld()
        world.invoke("property", "renew-lease", {"lease_id": "lse-4101", "new_end_date": "2027-10-31"})
        renewed = world.invoke(
            "property", "renew-lease", {"lease_id": "lse-4102", "new_end_date": "2028-02-29", "new_monthly_rent": 1.005}
        )
        self.assertEqual(renewed["lease"]["monthly_rent"], "1.01")
        world.invoke("property", "update-tenant-contact", {"tenant_id": "tnt-208", "phone": " +49 30 1 "})
        own = {"tenant_id": "tnt-221", "email": " AMIRA.NASSER@example.com ", "phone": "+49 2"}
        self.assertEqual(world.invoke("property", "update-tenant-contact", own)["tenant"]["phone"], "+49 2")
        state = world.snapshot()
        expected = {
            "lease:lse-4101:end_date": "2027-10-31",
            "lease:lse-4101:monthly_rent": "980.00",
            "lease:lse-4102:end_date": "2028-02-29",
            "lease:lse-4102:monthly_rent": "1.01",
            "tenant:tnt-208:phone": "+49 30 1",
            "tenant:tnt-208:email": "hannah.schulz@example.de",
            "tenant:tnt-221:email": "amira.nasser@example.com",
            "tenant:tnt-221:phone": "+49 2",
        }
        self.assertEqual({key: state[key] for key in expected}, expected)
        self.assertEqual(
            self._effects(world),
            ["lease:lse-4101:end_date", "lease:lse-4102:end_date", "tenant:tnt-208:phone", "tenant:tnt-221:email"],
        )

    def test_repairs_writes_keep_every_stored_field(self):
        world = fresh_classes.FreshWorld()
        full = {
            "unit_id": "unt-e1a",
            "contractor_id": "con-16",
            "title": "  Window clean ",
            "priority": "urgent",
            "preferred_date": "2026-10-20",
            "access_notes": "Key under the mat",
            "notify_tenant": True,
        }
        created = world.invoke("repairs", "create-work-order", full)["work_order"]
        self.assertEqual(
            (created["id"], created["visit_date"], created["visit_time"]), ("WO-1063", "2026-10-20", "07:00")
        )
        world.invoke("repairs", "create-work-order", {"unit_id": "unt-m3a", "contractor_id": "con-13", "title": "Roof"})
        world.invoke(
            "repairs", "reschedule-visit", {"work_order_id": "WO-1059", "date": "2026-10-15", "start_time": "11:00"}
        )
        world.invoke("repairs", "complete-work-order", {"work_order_id": "WO-1060", "final_cost": 75})
        done = {"work_order_id": "WO-1063", "final_cost": 99.995, "invoice_number": "SC-1"}
        self.assertEqual(world.invoke("repairs", "complete-work-order", done)["work_order"]["final_cost"], "100.00")
        state = world.snapshot()
        expected = {
            "workorder:WO-1063:title": "  Window clean ",
            "workorder:WO-1063:priority": "urgent",
            "workorder:WO-1063:access_notes": "Key under the mat",
            "workorder:WO-1063:cost_cap": "",
            "workorder:WO-1063:notify_tenant": True,
            "workorder:WO-1063:invoice_number": "SC-1",
            "workorder:WO-1064:preferred_date": "",
            "workorder:WO-1064:visit_date": "2026-10-13",
            "workorder:WO-1064:visit_time": "07:30",
            "workorder:WO-1059:visit_date": "2026-10-15",
            "workorder:WO-1059:visit_time": "11:00",
            "workorder:WO-1060:status": "completed",
            "workorder:WO-1060:final_cost": "75.00",
            "workorder:WO-1060:invoice_number": "",
        }
        self.assertEqual({key: state[key] for key in expected}, expected)
        self.assertEqual(
            self._effects(world),
            [
                "workorder:WO-1063",
                "workorder:WO-1064",
                "workorder:WO-1059:visit_date",
                "workorder:WO-1060:status",
                "workorder:WO-1063:status",
            ],
        )

    def test_payments_writes_keep_every_stored_field(self):
        world = fresh_classes.FreshWorld()
        added = {"name": " Sven Ekberg ", "iban": "se12 3456 7890 1234 5678 90", "email": " Sven@Example.SE "}
        payee = world.invoke("payments", "add-payee", added)["payee"]
        self.assertEqual((payee["id"], payee["iban_ending"]), ("pay-new-1", "7890"))
        world.invoke("payments", "add-payee", {"name": "Window Co", "iban": "DE00123456789012345678"})
        scheduled = {"payee_id": "pay-new-1", "amount": 120, "execution_date": "2026-10-10", "internal_note": "First"}
        self.assertEqual(world.invoke("payments", "create-transfer", scheduled)["transfer"]["status"], "scheduled")
        world.invoke("payments", "create-transfer", {"payee_id": "pay-new-2", "amount": 5, "instant": True})
        world.invoke("payments", "cancel-transfer", {"transfer_id": "trf-new-1"})
        world.invoke("payments", "cancel-transfer", {"transfer_id": "trf-5504"})
        state = world.snapshot()
        expected = {
            "payee:pay-new-1:name": "Sven Ekberg",
            "payee:pay-new-1:iban": "SE12345678901234567890",
            "payee:pay-new-1:email": "sven@example.se",
            "payee:pay-new-2:email": "",
            "transfer:trf-new-1:amount": "120.00",
            "transfer:trf-new-1:execution_date": "2026-10-10",
            "transfer:trf-new-1:internal_note": "First",
            "transfer:trf-new-1:status": "cancelled",
            "transfer:trf-new-2:instant": True,
            "transfer:trf-new-2:status": "sent",
            "transfer:trf-new-2:reference": "",
            "transfer:trf-5504:status": "cancelled",
        }
        self.assertEqual({key: state[key] for key in expected}, expected)
        self.assertEqual(
            self._effects(world),
            [
                "payee:pay-new-1",
                "payee:pay-new-2",
                "transfer:trf-new-1",
                "transfer:trf-new-2",
                "transfer:trf-new-1:status",
                "transfer:trf-5504:status",
            ],
        )


class OracleTests(unittest.TestCase):
    def test_every_reference_workflow_passes_in_every_language(self):
        for scenario in fresh_classes.SCENARIOS:
            with self.subTest(scenario=scenario.id):
                result = corpus.oracle(scenario, _run(REFERENCE[scenario.template.id]), fresh_classes.INITIAL)
                self.assertEqual(result.to_dict(), PASSED)

    def test_only_clarify_templates_pass_an_untouched_world(self):
        for template in fresh_classes.TEMPLATES:
            with self.subTest(template=template.id):
                passed = _oracle(f"{template.id}.en", fresh_classes.FreshWorld()).passed
                self.assertEqual(passed, template.behavior == "clarify")

    def test_a_failed_attempt_before_the_correct_order_still_passes(self):
        world = fresh_classes.FreshWorld()
        with self.assertRaises(simulated.ActionFailedError) as raised:
            world.invoke("payments", "cancel-transfer", {"transfer_id": "trf-5502"})
        self.assertIn(raised.exception.code, fresh_classes.NO_EFFECT_CODES)
        self.assertEqual(world.snapshot(), fresh_classes.INITIAL)
        _cancel_transfer(world)
        self.assertTrue(_oracle("cancel-transfer.fr", world).passed)

    def test_wrong_workflows_fail(self):
        tap = {"unit_id": "unt-e2b", "contractor_id": "con-12", "title": "Dripping kitchen tap", "cost_cap": 150}
        move = {"date": "2026-10-05", "start_time": "15:30"}
        wrong: tuple[tuple[str, tuple[Workflow, ...]], ...] = (
            (
                "boiler-payment.ar",
                (_call("payments", "create-transfer", {"payee_id": "pay-01", "amount": 198, "reference": "WO-1021"}),),
            ),
            (
                "lease-extend.de",
                (_call("property", "renew-lease", {"lease_id": "lse-4102", "new_end_date": "2027-05-31"}),),
            ),
            ("cancel-transfer.en", (_call("payments", "cancel-transfer", {"transfer_id": "trf-5504"}),)),
            ("tap-cap.es", (_call("repairs", "create-work-order", {**tap, "notify_tenant": True}),)),
            (
                "plain-transfer.fr",
                (_call("payments", "create-transfer", {"payee_id": "pay-06", "amount": 85.5, "reference": "Payment"}),),
            ),
            ("payee-iban.ja", (_call("payments", "create-transfer", {"payee_id": "pay-07", "amount": 120}),)),
            (
                "invoice-amount.pt",
                (
                    _call(
                        "payments", "create-transfer", {"payee_id": "pay-02", "amount": 312.4, "reference": "KP-5512"}
                    ),
                ),
            ),
            ("visit-move.zh", (_call("repairs", "reschedule-visit", {"work_order_id": "WO-1059", **move}),)),
            (
                "renew-forms.en",
                (
                    _call(
                        "property",
                        "renew-lease",
                        {"lease_id": "lse-4105", "new_end_date": "2028-03-31", "new_monthly_rent": 1.21},
                    ),
                ),
            ),
            ("lock-visit.en", (_call("repairs", "create-work-order", {**tap, "cost_cap": 0}),)),
            ("plain-transfer.en", (_plain_transfer, _plain_transfer)),
            ("visit-move.en", (_visit_move, _call("weather", "set-alert", {"title": "rain"}))),
        )
        for scenario_id, workflows in wrong:
            with self.subTest(scenario=scenario_id):
                self.assertFalse(_oracle(scenario_id, _run(*workflows)).passed)
        # A second transfer is a whole unrequested item: its key and every stored field are differences.
        duplicate = _oracle("plain-transfer.en", _run(_plain_transfer, _plain_transfer))
        self.assertEqual((duplicate.forbidden, duplicate.wrong), (1, 8))
        foreign = _oracle("visit-move.en", _run(_visit_move, _call("weather", "set-alert", {"title": "rain"})))
        self.assertEqual((foreign.forbidden, foreign.wrong_scope), (1, 1))


if __name__ == "__main__":
    unittest.main()

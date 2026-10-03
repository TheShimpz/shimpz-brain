"""The fresh-v4-classes stratum: three new simulated small-business Assistants beside the precision corpus (ADR-0094).

``fresh-v4-classes`` adds three SIMULATED provider APIs (rental property management, maintenance work orders, and
business bank transfers) to the precision corpus's five relevant Assistants and fifteen irrelevant-domain distractors.
Ten task templates in four classes (dependent lookup chains, optional values, clarification of a user-only value, and
action under varied surface forms) in all eight languages give 80 scenarios with stable ids ``<template>.<locale>``,
the same scope rotation and deterministic padding as ``eval.corpus``, and exact expected changes that
``eval.corpus.oracle`` scores against ``INITIAL``. Every stored field of every item an Action can create or change is
in the snapshot, so any value the user did not ask for is a difference. ``USER_SOURCED`` declares, per write Action of
the new Assistants, the input properties whose value only the user can supply. Changing any message, fixture, oracle,
contract, or reference requires a new corpus id.

This module uses only the standard library.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal

from eval import fixtures, world
from eval.corpus import BEHAVIORS, LOCALES, SCOPES, Scenario, Template, expected_state
from eval.fixtures import _DATE, _STRING, Action, Assistant, _schema

CORPUS_ID = "fresh-v4-classes"

_TIME = {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$", "description": "24-hour time as HH:MM."}
_LEASE_STATUS = {"type": "string", "enum": ["active", "ended"]}
_TRADE = {
    "type": "string",
    "enum": ["plumbing", "heating", "electrical", "locksmith", "roofing", "cleaning", "carpentry"],
}
_PRIORITY = {"type": "string", "enum": ["low", "normal", "urgent"]}
_WORK_ORDER_STATUS = {"type": "string", "enum": ["open", "completed", "cancelled"]}
_TRANSFER_STATUS = {"type": "string", "enum": ["sent", "scheduled", "cancelled"]}
_EUR = {"type": "number", "minimum": 0.01, "maximum": 100000, "description": "Amount in EUR."}
_IBAN = {
    "type": "string",
    "pattern": "^[A-Z]{2}[0-9]{2}[A-Z0-9 ]{10,40}$",
    "description": "The payee's IBAN; spaces are ignored.",
}


def _read(action_id: str, summary: str, schema: Mapping[str, object]) -> Action:
    return Action(action_id, summary, schema, writes=False)


def _write(action_id: str, summary: str, schema: Mapping[str, object]) -> Action:
    return Action(action_id, summary, schema, writes=True)


NEW_ASSISTANTS = (
    Assistant(
        "property",
        "Property manages the user's rental units, their tenants, and the leases between them. Rents and deposits "
        "are in EUR.",
        (
            _read(
                "list-units",
                "List rental units with their ids and labels, optionally only those whose label contains the text.",
                _schema(query=_STRING),
            ),
            _read(
                "find-tenant",
                "Find tenants, current or former, whose name or email contains the given text, with their contact "
                "details and their leases.",
                _schema(("query",), query=_STRING),
            ),
            _read(
                "list-leases",
                "List leases with unit, tenant, monthly rent, deposit, dates, and status, optionally by unit, tenant, "
                "or status.",
                _schema(unit_id=_STRING, tenant_id=_STRING, status=_LEASE_STATUS),
            ),
            _write(
                "renew-lease",
                "Extend one active lease to a new end date after its current one; the monthly rent stays unchanged "
                "unless new_monthly_rent is given.",
                _schema(
                    ("lease_id", "new_end_date"),
                    lease_id=_STRING,
                    new_end_date=_DATE,
                    new_monthly_rent={**_EUR, "description": "The new monthly rent in EUR."},
                ),
            ),
            _write(
                "update-tenant-contact",
                "Change one tenant's email address or phone number, or both; fails when another tenant uses the email.",
                _schema(
                    ("tenant_id",),
                    tenant_id=_STRING,
                    email=_STRING,
                    phone={**_STRING, "description": "Phone number in international format."},
                ),
            ),
        ),
    ),
    Assistant(
        "repairs",
        "Repairs dispatches maintenance work orders for the rental units to the user's contractors and tracks their "
        "visits and costs in EUR.",
        (
            _read(
                "list-contractors",
                "List contractors with their trade, phone, and next free visit slot, optionally only one trade.",
                _schema(trade=_TRADE),
            ),
            _read(
                "list-work-orders",
                "List work orders with unit, contractor, visit, and costs, optionally by unit, contractor, or status.",
                _schema(unit_id=_STRING, contractor_id=_STRING, status=_WORK_ORDER_STATUS),
            ),
            _write(
                "create-work-order",
                "Create one open work order for a unit and a contractor. Priority defaults to normal; without "
                "preferred_date the visit takes the contractor's next free slot; access_notes and cost_cap default to "
                "none; notify_tenant defaults to false. Returns the work order with its visit date and time.",
                _schema(
                    ("unit_id", "contractor_id", "title"),
                    unit_id=_STRING,
                    contractor_id=_STRING,
                    title=_STRING,
                    priority=_PRIORITY,
                    preferred_date={**_DATE, "description": "Visit date as YYYY-MM-DD at the contractor's start time."},
                    access_notes={**_STRING, "description": "How the contractor gets into the unit."},
                    cost_cap={**_EUR, "description": "Most the contractor may charge in EUR without asking first."},
                    notify_tenant={"type": "boolean", "description": "Email the tenant the visit date and time."},
                ),
            ),
            _write(
                "reschedule-visit",
                "Move the visit of one open work order to another date and start time.",
                _schema(
                    ("work_order_id", "date", "start_time"),
                    work_order_id=_STRING,
                    date=_DATE,
                    start_time=_TIME,
                ),
            ),
            _write(
                "complete-work-order",
                "Mark one open work order completed with the contractor's final cost and optional invoice number.",
                _schema(
                    ("work_order_id", "final_cost"),
                    work_order_id=_STRING,
                    final_cost=_EUR,
                    invoice_number=_STRING,
                ),
            ),
        ),
    ),
    Assistant(
        "payments",
        "Payments sends bank transfers in EUR from the business account to saved payees.",
        (
            _read(
                "list-payees",
                "List saved payees with their ids and IBAN endings, optionally only those whose name contains the "
                "text.",
                _schema(name=_STRING),
            ),
            _read(
                "list-transfers",
                "List transfers with payee, amount, reference, date, and status, optionally by payee, reference, or "
                "status.",
                _schema(payee_id=_STRING, reference=_STRING, status=_TRANSFER_STATUS),
            ),
            _write(
                "add-payee",
                "Save one new payee by name and IBAN, with an optional email for remittance advice; fails when a "
                "payee with that IBAN exists.",
                _schema(("name", "iban"), name=_STRING, iban=_IBAN, email=_STRING),
            ),
            _write(
                "create-transfer",
                "Send one transfer to a saved payee. Without execution_date it is sent immediately, otherwise it is "
                "scheduled for that date; instant defaults to false and needs immediate sending; reference (shown to "
                "the payee) and internal_note default to empty.",
                _schema(
                    ("payee_id", "amount"),
                    payee_id=_STRING,
                    amount=_EUR,
                    reference={**_STRING, "description": "Text shown to the payee on their statement."},
                    execution_date=_DATE,
                    instant={"type": "boolean"},
                    internal_note={**_STRING, "description": "Note visible only in the business account."},
                ),
            ),
            _write(
                "cancel-transfer",
                "Cancel one scheduled transfer; a sent transfer cannot be cancelled.",
                _schema(("transfer_id",), transfer_id=_STRING),
            ),
        ),
    ),
)
NEW_IDS = frozenset(assistant.id for assistant in NEW_ASSISTANTS)
RELEVANT = (*fixtures.RELEVANT, *NEW_ASSISTANTS)
# Every Assistant a scenario can carry: the precision corpus's twenty and the three new ones.
ASSISTANTS = {**fixtures.ASSISTANTS, **{assistant.id: assistant for assistant in NEW_ASSISTANTS}}

# The input properties of each new write Action whose value only the user can supply: never inferred, looked up on
# the agent's own initiative, defaulted, or copied from another field.
USER_SOURCED: dict[tuple[str, str], tuple[str, ...]] = {
    ("property", "renew-lease"): ("new_end_date", "new_monthly_rent"),
    ("property", "update-tenant-contact"): ("email", "phone"),
    ("repairs", "create-work-order"): ("preferred_date", "access_notes", "cost_cap"),
    ("repairs", "reschedule-visit"): ("date", "start_time"),
    ("repairs", "complete-work-order"): ("final_cost", "invoice_number"),
    ("payments", "add-payee"): ("name", "iban", "email"),
    ("payments", "create-transfer"): ("amount", "reference", "execution_date"),
    ("payments", "cancel-transfer"): (),
}

# The simulated starting state of the new Assistants.
UNITS = (
    ("unt-e1a", "Flat 1A, 14 Elm Street"),
    ("unt-e1b", "Flat 1B, 14 Elm Street"),
    ("unt-e2b", "Flat 2B, 14 Elm Street"),
    ("unt-m2b", "Flat 2B, 22 Mill Lane"),
    ("unt-m3a", "Flat 3A, 22 Mill Lane"),
    ("unt-ms1", "Shop 1, 22 Mill Lane"),
)
_UNITS = dict(UNITS)
_TENANT_FIELDS = ("name", "email", "phone")
TENANTS = (
    ("tnt-208", "Hannah Schulz", "hannah.schulz@example.de", "+49 30 5550 1208"),
    ("tnt-214", "Henrik Schulz", "henrik.schulz@example.de", "+49 30 5550 1214"),
    ("tnt-221", "Amira Nasser", "amira.nasser@example.com", "+49 30 5550 1221"),
    ("tnt-226", "Omar Nasser", "omar.nasser@example.com", "+49 30 5550 1226"),
    ("tnt-233", "Lucía Moreno", "lucia.moreno@example.es", "+34 612 555 233"),
    ("tnt-239", "Lucas Moreau", "lucas.moreau@example.fr", "+33 6 55 50 12 39"),
    ("tnt-245", "Kenta Mori", "kenta.mori@example.jp", "+81 90 5550 1245"),
    ("tnt-252", "Greenway Books Ltd", "shop@greenway-books.example", "+49 30 5550 1252"),
)
_LEASE_FIELDS = ("unit_id", "tenant_id", "monthly_rent", "deposit", "start_date", "end_date", "status")
LEASES = (
    ("lse-4087", "unt-m2b", "tnt-226", "1050.00", "3150.00", "2023-06-01", "2025-05-31", "ended"),
    ("lse-4090", "unt-e2b", "tnt-245", "990.00", "2970.00", "2023-03-01", "2025-02-28", "ended"),
    ("lse-4101", "unt-e1a", "tnt-208", "980.00", "2940.00", "2024-11-01", "2026-10-31", "active"),
    ("lse-4102", "unt-e2b", "tnt-214", "1040.00", "3120.00", "2025-03-01", "2027-02-28", "active"),
    ("lse-4103", "unt-m2b", "tnt-221", "1120.00", "3360.00", "2025-06-01", "2026-11-30", "active"),
    ("lse-4104", "unt-e1b", "tnt-233", "890.00", "2670.00", "2025-01-01", "2026-12-31", "active"),
    ("lse-4105", "unt-m3a", "tnt-239", "1175.00", "3525.00", "2024-04-01", "2027-03-31", "active"),
    ("lse-4106", "unt-ms1", "tnt-252", "2300.00", "6900.00", "2022-09-01", "2027-08-31", "active"),
)
# (id, name, trade, phone, next free visit date, start time)
CONTRACTORS = (
    ("con-11", "Brandt Heating", "heating", "+49 30 5550 7711", "2026-10-06", "09:00"),
    ("con-12", "Kraft Plumbing", "plumbing", "+49 30 5550 7712", "2026-10-08", "10:00"),
    ("con-13", "Kraft Roofing", "roofing", "+49 30 5550 7713", "2026-10-13", "07:30"),
    ("con-14", "Vogel Electric", "electrical", "+49 30 5550 7714", "2026-10-06", "13:00"),
    ("con-15", "Vogel Locks", "locksmith", "+49 30 5550 7715", "2026-10-07", "08:30"),
    ("con-16", "Sparkle Cleaning", "cleaning", "+49 30 5550 7716", "2026-10-05", "07:00"),
)
_CONTRACTORS = {item[0]: item for item in CONTRACTORS}
_WORK_ORDER_FIELDS = (
    "unit_id",
    "contractor_id",
    "title",
    "priority",
    "preferred_date",
    "access_notes",
    "cost_cap",
    "notify_tenant",
    "status",
    "visit_date",
    "visit_time",
    "final_cost",
    "invoice_number",
)
WORK_ORDERS = (
    (
        "WO-1021",
        *("unt-e1a", "con-11", "Annual boiler service", "normal", "", "", "", False),
        *("completed", "2026-03-04", "09:00", "198.00", "BH-0311"),
    ),
    (
        "WO-1032",
        *("unt-e2b", "con-14", "Replace bathroom extractor fan", "normal", "", "", "", False),
        *("completed", "2026-06-17", "13:00", "145.00", "VE-2207"),
    ),
    (
        "WO-1040",
        *("unt-ms1", "con-12", "Unblock shop washroom drain", "urgent", "", "", "", False),
        *("completed", "2026-09-10", "10:00", "312.40", "KP-5498"),
    ),
    (
        "WO-1049",
        *("unt-e2b", "con-11", "Boiler pressure loss", "urgent", "", "", "", False),
        *("completed", "2026-09-24", "09:00", "286.40", "BH-0398"),
    ),
    (
        "WO-1055",
        *("unt-m2b", "con-13", "Repair balcony gutter", "normal", "", "", "", False),
        *("completed", "2026-08-19", "07:30", "460.00", "KR-1187"),
    ),
    (
        "WO-1058",
        *("unt-e1b", "con-12", "Leaking shower drain", "normal", "", "", "", True),
        *("open", "2026-10-09", "10:00", "", ""),
    ),
    (
        "WO-1059",
        *("unt-m3a", "con-12", "Low water pressure in kitchen", "normal", "", "", "", False),
        *("open", "2026-10-14", "10:00", "", ""),
    ),
    (
        "WO-1060",
        *("unt-e1b", "con-14", "Replace hallway light switch", "low", "", "", "", False),
        *("open", "2026-10-06", "13:00", "", ""),
    ),
    (
        "WO-1061",
        *("unt-ms1", "con-14", "Rewire shop display lighting", "normal", "2026-10-12", "", "400.00", True),
        *("open", "2026-10-12", "13:00", "", ""),
    ),
    (
        "WO-1062",
        *("unt-e1a", "con-16", "Stairwell deep clean", "low", "", "", "", False),
        *("cancelled", "2026-09-29", "07:00", "", ""),
    ),
)
_PAYEE_FIELDS = ("name", "iban", "email")
PAYEES = (
    ("pay-01", "Brandt Heating", "DE44500105175407324931", "billing@brandt-heating.example"),
    ("pay-02", "Kraft Plumbing", "DE89370400440532013000", "invoices@kraft-plumbing.example"),
    ("pay-03", "Kraft Roofing", "DE89370400440532013087", ""),
    ("pay-04", "Vogel Electric", "DE12500105170648489890", "office@vogel-electric.example"),
    ("pay-05", "Vogel Locks", "DE12500105170648489812", ""),
    ("pay-06", "Ivo Brandt", "DE75512108001245126199", ""),
    ("pay-07", "Sven Ekström", "SE4550000000058398257466", ""),
    ("pay-08", "Sparkle Cleaning", "DE02120300000000202051", ""),
)
_TRANSFER_FIELDS = ("payee_id", "amount", "reference", "execution_date", "instant", "internal_note", "status")
TRANSFERS = (
    ("trf-5501", "pay-02", "312.40", "KP-5498", "2026-09-15", False, "", "sent"),
    ("trf-5502", "pay-04", "145.00", "WO-1032", "2026-06-20", False, "", "sent"),
    ("trf-5503", "pay-04", "150.00", "Deposit WO-1061", "2026-10-12", False, "", "scheduled"),
    ("trf-5504", "pay-05", "95.00", "Key copies Mill Lane", "2026-10-09", False, "", "scheduled"),
    ("trf-5505", "pay-06", "60.00", "Garden work", "2026-09-02", False, "", "sent"),
    ("trf-5506", "pay-01", "198.00", "WO-1021", "2026-03-11", False, "", "sent"),
    ("trf-5507", "pay-07", "110.00", "Stairwell cleaning", "2026-09-01", False, "", "sent"),
)

# The failure codes FreshWorld's Actions raise before any effect: the precision World's and the new Assistants'. An
# undeclared Action is a protocol failure, never a no-effect business failure.
NO_EFFECT_CODES = world.NO_EFFECT_CODES | frozenset(
    {
        "tenant-not-found",
        "lease-not-found",
        "lease-not-active",
        "invalid-end-date",
        "nothing-to-change",
        "email-in-use",
        "unit-not-found",
        "contractor-not-found",
        "work-order-not-found",
        "work-order-closed",
        "payee-not-found",
        "payee-exists",
        "invalid-schedule",
        "transfer-not-found",
        "transfer-not-cancellable",
    }
)

_Result = tuple[dict[str, object], str | None]


def _money(value: object) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _fold(value: object) -> str:
    """Case- and accent-insensitive text for name searches."""
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def _iban(value: object) -> str:
    return str(value).replace(" ", "").upper()


class FreshWorld(world.World):
    """The precision World, unchanged, plus the state and behavior of the three new Assistants."""

    def __init__(self) -> None:
        super().__init__()
        self.assistants = ASSISTANTS
        self.tenants = {item[0]: dict(zip(_TENANT_FIELDS, item[1:], strict=True)) for item in TENANTS}
        self.leases = {item[0]: dict(zip(_LEASE_FIELDS, item[1:], strict=True)) for item in LEASES}
        self.work_orders = {item[0]: dict(zip(_WORK_ORDER_FIELDS, item[1:], strict=True)) for item in WORK_ORDERS}
        self.payees = {item[0]: dict(zip(_PAYEE_FIELDS, item[1:], strict=True)) for item in PAYEES}
        self.transfers = {item[0]: dict(zip(_TRANSFER_FIELDS, item[1:], strict=True)) for item in TRANSFERS}
        self._serial: Counter[str] = Counter()

    def invoke(self, assistant_id: str, action_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
        if assistant_id not in NEW_IDS:
            return super().invoke(assistant_id, action_id, arguments)
        entry: dict[str, object] = {
            "assistant": assistant_id,
            "action": action_id,
            "input": copy.deepcopy(dict(arguments)),
        }
        self.ledger.append(entry)
        try:
            if action_id not in {action.id for action in ASSISTANTS[assistant_id].actions}:
                raise world.ActionFailedError("undeclared-action")
            result, effect = getattr(self, "_" + f"{assistant_id}_{action_id}".replace("-", "_"))(arguments)
        except world.ActionFailedError as exc:
            entry["failed"] = exc.code
            raise
        entry["result"] = copy.deepcopy(result)
        if effect is not None:
            entry["effect"] = effect
        return result

    def _new(self, prefix: str) -> str:
        self._serial[prefix] += 1
        if prefix == "WO":
            return f"WO-{1062 + self._serial[prefix]}"
        return f"{prefix}-new-{self._serial[prefix]}"

    # Property.

    def _lease_view(self, key: str) -> dict[str, object]:
        item = self.leases[key]
        return {
            "id": key,
            **item,
            "unit": _UNITS[str(item["unit_id"])],
            "tenant": self.tenants[str(item["tenant_id"])]["name"],
            "currency": "EUR",
        }

    def _property_list_units(self, arguments: Mapping[str, object]) -> _Result:
        text = _fold(arguments.get("query", ""))
        return {"units": [{"id": key, "label": label} for key, label in UNITS if text in _fold(label)]}, None

    def _property_find_tenant(self, arguments: Mapping[str, object]) -> _Result:
        text = _fold(arguments["query"])
        return {
            "tenants": [
                {
                    "id": key,
                    **item,
                    "leases": [
                        {"id": lease, "unit_id": value["unit_id"], "status": value["status"]}
                        for lease, value in sorted(self.leases.items())
                        if value["tenant_id"] == key
                    ],
                }
                for key, item in sorted(self.tenants.items())
                if text in _fold(item["name"]) or text in _fold(item["email"])
            ]
        }, None

    def _property_list_leases(self, arguments: Mapping[str, object]) -> _Result:
        return {
            "leases": [
                self._lease_view(key)
                for key, item in sorted(self.leases.items())
                if arguments.get("unit_id") in {None, item["unit_id"]}
                and arguments.get("tenant_id") in {None, item["tenant_id"]}
                and arguments.get("status") in {None, item["status"]}
            ]
        }, None

    def _property_renew_lease(self, arguments: Mapping[str, object]) -> _Result:
        key = str(arguments.get("lease_id"))
        if key not in self.leases:
            raise world.ActionFailedError("lease-not-found")
        item = self.leases[key]
        if item["status"] != "active":
            raise world.ActionFailedError("lease-not-active")
        end = str(arguments["new_end_date"])
        if end <= str(item["end_date"]):
            raise world.ActionFailedError("invalid-end-date")
        item["end_date"] = end
        if "new_monthly_rent" in arguments:
            item["monthly_rent"] = _money(arguments["new_monthly_rent"])
        return {"lease": self._lease_view(key)}, f"lease:{key}:end_date"

    def _property_update_tenant_contact(self, arguments: Mapping[str, object]) -> _Result:
        key = str(arguments.get("tenant_id"))
        if key not in self.tenants:
            raise world.ActionFailedError("tenant-not-found")
        changes = {name: str(arguments[name]).strip() for name in ("email", "phone") if name in arguments}
        if not changes:
            raise world.ActionFailedError("nothing-to-change")
        if "email" in changes:
            changes["email"] = changes["email"].lower()
            if any(other != key and item["email"] == changes["email"] for other, item in self.tenants.items()):
                raise world.ActionFailedError("email-in-use")
        self.tenants[key].update(changes)
        return {"tenant": {"id": key, **self.tenants[key]}}, f"tenant:{key}:{next(iter(changes))}"

    # Repairs.

    def _work_order_view(self, key: str) -> dict[str, object]:
        item = self.work_orders[key]
        contractor = _CONTRACTORS[str(item["contractor_id"])]
        return {"id": key, **item, "unit": _UNITS[str(item["unit_id"])], "contractor": contractor[1]}

    def _open_work_order(self, arguments: Mapping[str, object]) -> str:
        key = str(arguments.get("work_order_id"))
        if key not in self.work_orders:
            raise world.ActionFailedError("work-order-not-found")
        if self.work_orders[key]["status"] != "open":
            raise world.ActionFailedError("work-order-closed")
        return key

    def _repairs_list_contractors(self, arguments: Mapping[str, object]) -> _Result:
        return {
            "contractors": [
                {"id": key, "name": name, "trade": trade, "phone": phone, "next_free_date": day, "next_free_time": at}
                for key, name, trade, phone, day, at in CONTRACTORS
                if arguments.get("trade") in {None, trade}
            ]
        }, None

    def _repairs_list_work_orders(self, arguments: Mapping[str, object]) -> _Result:
        return {
            "work_orders": [
                self._work_order_view(key)
                for key, item in sorted(self.work_orders.items())
                if arguments.get("unit_id") in {None, item["unit_id"]}
                and arguments.get("contractor_id") in {None, item["contractor_id"]}
                and arguments.get("status") in {None, item["status"]}
            ]
        }, None

    def _repairs_create_work_order(self, arguments: Mapping[str, object]) -> _Result:
        unit, contractor = str(arguments.get("unit_id")), str(arguments.get("contractor_id"))
        if unit not in _UNITS:
            raise world.ActionFailedError("unit-not-found")
        if contractor not in _CONTRACTORS:
            raise world.ActionFailedError("contractor-not-found")
        preferred = str(arguments.get("preferred_date", ""))
        key = self._new("WO")
        self.work_orders[key] = {
            "unit_id": unit,
            "contractor_id": contractor,
            "title": world.title(arguments["title"]),
            "priority": str(arguments.get("priority", "normal")),
            "preferred_date": preferred,
            "access_notes": str(arguments.get("access_notes", "")),
            "cost_cap": _money(arguments["cost_cap"]) if "cost_cap" in arguments else "",
            "notify_tenant": bool(arguments.get("notify_tenant", False)),
            "status": "open",
            "visit_date": preferred or _CONTRACTORS[contractor][4],
            "visit_time": _CONTRACTORS[contractor][5],
            "final_cost": "",
            "invoice_number": "",
        }
        return {"work_order": self._work_order_view(key)}, f"workorder:{key}"

    def _repairs_reschedule_visit(self, arguments: Mapping[str, object]) -> _Result:
        key = self._open_work_order(arguments)
        self.work_orders[key]["visit_date"] = str(arguments["date"])
        self.work_orders[key]["visit_time"] = str(arguments["start_time"])
        return {"work_order": self._work_order_view(key)}, f"workorder:{key}:visit_date"

    def _repairs_complete_work_order(self, arguments: Mapping[str, object]) -> _Result:
        key = self._open_work_order(arguments)
        item = self.work_orders[key]
        item["status"] = "completed"
        item["final_cost"] = _money(arguments["final_cost"])
        item["invoice_number"] = str(arguments.get("invoice_number", ""))
        return {"work_order": self._work_order_view(key)}, f"workorder:{key}:status"

    # Payments.

    def _payee_view(self, key: str) -> dict[str, object]:
        item = self.payees[key]
        return {"id": key, "name": item["name"], "iban_ending": str(item["iban"])[-4:], "email": item["email"]}

    def _transfer_view(self, key: str) -> dict[str, object]:
        item = self.transfers[key]
        return {"id": key, **item, "payee": self.payees[str(item["payee_id"])]["name"], "currency": "EUR"}

    def _payments_list_payees(self, arguments: Mapping[str, object]) -> _Result:
        text = _fold(arguments.get("name", ""))
        return {
            "payees": [
                self._payee_view(key) for key, item in sorted(self.payees.items()) if text in _fold(item["name"])
            ]
        }, None

    def _payments_list_transfers(self, arguments: Mapping[str, object]) -> _Result:
        reference = _fold(arguments.get("reference", ""))
        return {
            "transfers": [
                self._transfer_view(key)
                for key, item in sorted(self.transfers.items())
                if arguments.get("payee_id") in {None, item["payee_id"]}
                and arguments.get("status") in {None, item["status"]}
                and reference in _fold(item["reference"])
            ]
        }, None

    def _payments_add_payee(self, arguments: Mapping[str, object]) -> _Result:
        iban = _iban(arguments["iban"])
        if any(item["iban"] == iban for item in self.payees.values()):
            raise world.ActionFailedError("payee-exists")
        key = self._new("pay")
        self.payees[key] = {
            "name": str(arguments["name"]).strip(),
            "iban": iban,
            "email": str(arguments.get("email", "")).strip().lower(),
        }
        return {"payee": self._payee_view(key)}, f"payee:{key}"

    def _payments_create_transfer(self, arguments: Mapping[str, object]) -> _Result:
        payee = str(arguments.get("payee_id"))
        if payee not in self.payees:
            raise world.ActionFailedError("payee-not-found")
        execution = str(arguments.get("execution_date", ""))
        instant = bool(arguments.get("instant", False))
        if instant and execution:
            raise world.ActionFailedError("invalid-schedule")
        key = self._new("trf")
        self.transfers[key] = {
            "payee_id": payee,
            "amount": _money(arguments["amount"]),
            "reference": str(arguments.get("reference", "")),
            "execution_date": execution,
            "instant": instant,
            "internal_note": str(arguments.get("internal_note", "")),
            "status": "scheduled" if execution else "sent",
        }
        return {"transfer": self._transfer_view(key)}, f"transfer:{key}"

    def _payments_cancel_transfer(self, arguments: Mapping[str, object]) -> _Result:
        key = str(arguments.get("transfer_id"))
        if key not in self.transfers:
            raise world.ActionFailedError("transfer-not-found")
        if self.transfers[key]["status"] != "scheduled":
            raise world.ActionFailedError("transfer-not-cancellable")
        self.transfers[key]["status"] = "cancelled"
        return {"transfer": self._transfer_view(key)}, f"transfer:{key}:status"

    def snapshot(self) -> dict[str, object]:
        """The precision World's snapshot plus every stored field of every item a new Assistant can change."""
        state = super().snapshot()
        for prefix, items in (
            ("tenant", self.tenants),
            ("lease", self.leases),
            ("workorder", self.work_orders),
            ("payee", self.payees),
            ("transfer", self.transfers),
        ):
            for key, item in items.items():
                state[f"{prefix}:{key}"] = True
                state.update({f"{prefix}:{key}:{name}": value for name, value in item.items()})
        return state


INITIAL = FreshWorld().snapshot()


def _messages(*texts: str) -> dict[str, str]:
    """Messages in LOCALES order: ar, de, en, es, fr, ja, pt, zh."""
    return dict(zip(LOCALES, texts, strict=True))


def _created(prefix: str, key: str, **fields: object) -> dict[str, object]:
    return {f"{prefix}:{key}": True, **{f"{prefix}:{key}:{name}": value for name, value in fields.items()}}


def _transfer(payee_id: str, amount: str, reference: str = "") -> dict[str, object]:
    """One immediately sent transfer with every other optional field at its default."""
    return _created(
        "transfer",
        "trf-new-1",
        payee_id=payee_id,
        amount=amount,
        reference=reference,
        execution_date="",
        instant=False,
        internal_note="",
        status="sent",
    )


def _work_order(
    unit_id: str, contractor_id: str, title: str, visit: tuple[str, str], cost_cap: str = ""
) -> dict[str, object]:
    """The first new work order, open, with every optional field but cost_cap at its default."""
    return _created(
        "workorder",
        "WO-1063",
        unit_id=unit_id,
        contractor_id=contractor_id,
        title=title,
        priority="normal",
        preferred_date="",
        access_notes="",
        cost_cap=cost_cap,
        notify_tenant=False,
        status="open",
        visit_date=visit[0],
        visit_time=visit[1],
        final_cost="",
        invoice_number="",
    )


TEMPLATES = (
    Template(
        "boiler-payment",
        "act",
        ("property", "repairs", "payments"),
        3,
        _messages(
            "ادفع للمقاول الذي أصلح الغلاية في شقة Henrik Schulz: المبلغ هو التكلفة النهائية لأمر العمل ذاك، واجعل "
            "رقم أمر العمل مرجعًا للتحويل.",
            "Bezahl den Handwerker, der in der Wohnung von Henrik Schulz den Heizkessel repariert hat: den Endbetrag "
            "dieses Arbeitsauftrags, mit der Auftragsnummer als Verwendungszweck.",
            "Pay the contractor who fixed the boiler in Henrik Schulz's flat: the final cost of that work order, with "
            "the work order number as the payment reference.",
            "Págale al técnico que reparó la caldera en el piso de Henrik Schulz el coste final de esa orden de "
            "trabajo, con el número de la orden como concepto de la transferencia.",
            "Paie l'artisan qui a réparé la chaudière dans l'appartement de Henrik Schulz : le coût final de ce bon "
            "d'intervention, avec le numéro du bon comme référence du virement.",
            "Henrik Schulz さんの部屋のボイラーを修理した業者に支払いをしてください。金額はその作業指示の最終費用で、"
            "振込の参照欄には作業指示番号を入れてください。",
            "Pague o técnico que consertou a caldeira no apartamento do Henrik Schulz: o custo final dessa ordem de "
            "serviço, com o número da ordem como referência do pagamento.",
            "给修好 Henrik Schulz 公寓锅炉的那个承包商付款：金额按那张工单的最终费用，转账附言填工单编号。",
        ),
        "Henrik Schulz rents Flat 2B, 14 Elm Street (Hannah Schulz rents Flat 1A). Its completed boiler work order "
        "is WO-1049 by Brandt Heating with a final cost of 286.40 EUR (Hannah's older boiler service WO-1021 was "
        "already paid). Exactly one new transfer exists: to the payee Brandt Heating (not Ivo Brandt), 286.40 EUR, "
        "reference WO-1049, sent immediately, not instant, no internal note. Nothing else changed. Lookups in any "
        "Assistant are fine. The reply confirms the payment.",
        _transfer("pay-01", "286.40", "WO-1049"),
    ),
    Template(
        "lock-visit",
        "act",
        ("property", "repairs", "calendar"),
        3,
        _messages(
            "اطلب من صانع الأقفال الذي نتعامل معه تغيير قفل الباب الأمامي في شقة Amira Nasser بأمر عمل عنوانه "
            '"Replace front door lock"، ثم أضف موعد الزيارة إلى تقويمي بالعنوان نفسه.',
            "Beauftrag unseren Schlüsseldienst, bei Amira Nasser das Wohnungstürschloss auszutauschen, mit einem "
            "Arbeitsauftrag namens „Replace front door lock“, und trag den Besuchstermin unter demselben Titel in "
            "meinen Kalender ein.",
            "Get our locksmith to replace the front door lock in Amira Nasser's flat, with a work order titled "
            '"Replace front door lock", then put the visit in my calendar under the same title.',
            "Pídele a nuestro cerrajero que cambie la cerradura de la puerta de entrada en el piso de Amira Nasser, "
            'con una orden de trabajo titulada "Replace front door lock", y luego apunta la visita en mi calendario '
            "con ese mismo título.",
            "Fais intervenir notre serrurier pour changer la serrure de la porte d'entrée chez Amira Nasser, avec un "
            "bon d'intervention intitulé « Replace front door lock », puis ajoute la visite à mon agenda sous le même "
            "titre.",
            "Amira Nasser さんの部屋の玄関ドアの鍵交換を、いつもの鍵業者に「Replace front door lock」という件名の"
            "作業指示で依頼して、訪問の予定を同じ件名でカレンダーに入れてください。",
            "Chame o nosso chaveiro para trocar a fechadura da porta de entrada do apartamento da Amira Nasser, com "
            'uma ordem de serviço chamada "Replace front door lock", e depois coloque a visita na minha agenda com o '
            "mesmo título.",
            "让我们的锁匠去 Amira Nasser 的公寓换大门锁，工单标题写“Replace front door lock”，然后把上门时间用同样的"
            "标题加到我的日历里。",
        ),
        "Amira Nasser rents Flat 2B, 22 Mill Lane (Omar Nasser's lease there has ended; Flat 2B, 14 Elm Street is "
        "another unit). The only locksmith is Vogel Locks (not Vogel Electric). Exactly one new work order WO-1063 "
        "exists for Flat 2B, 22 Mill Lane with Vogel Locks, titled Replace front door lock, every optional field at "
        "its default (priority normal, no preferred date, no access notes, no cost cap, tenant not notified), so the "
        "visit takes the contractor's next free slot, 2026-10-07 at 08:30, as the work order result shows. Exactly "
        "one calendar event titled Replace front door lock exists on 2026-10-07 at 08:30. Nothing else changed. The "
        "reply confirms both and may state the visit date and time.",
        {
            **_work_order("unt-m2b", "con-15", "Replace front door lock", ("2026-10-07", "08:30")),
            "new-event:2026-10-07:08:30:Replace front door lock": 1,
        },
    ),
    Template(
        "lease-extend",
        "safe-lookup",
        ("repairs", "property"),
        3,
        _messages(
            "مدّد عقد إيجار الشخص الذي يسكن الشقة التي نُفّذ فيها أمر العمل WO-1055 حتى 2027-05-31.",
            "Verlängere den Mietvertrag der Person, die in der Wohnung wohnt, in der der Arbeitsauftrag WO-1055 "
            "erledigt wurde, bis zum 2027-05-31.",
            "Extend the lease of whoever lives in the flat where work order WO-1055 was done, until 2027-05-31.",
            "Prorroga hasta el 2027-05-31 el contrato de alquiler de quien vive en el piso donde se hizo la orden de "
            "trabajo WO-1055.",
            "Prolonge jusqu'au 2027-05-31 le bail de la personne qui habite l'appartement où l'intervention WO-1055 a "
            "été faite.",
            "作業指示 WO-1055 を実施した部屋に今住んでいる入居者の賃貸契約を、2027-05-31 まで延長してください。",
            "Prorrogue até 2027-05-31 o contrato de aluguel de quem mora no apartamento onde foi feita a ordem de "
            "serviço WO-1055.",
            "工单 WO-1055 是在哪套公寓做的，就把现在住在那里的租户的租约延长到 2027-05-31。",
        ),
        "WO-1055 was done in Flat 2B, 22 Mill Lane. Its current tenant is Amira Nasser with the active lease "
        "lse-4103 ending 2026-11-30; Omar Nasser's earlier lease there has ended, and Henrik Schulz's Flat 2B, 14 Elm "
        "Street is a different unit. Only lse-4103's end date changes, to 2027-05-31; its rent stays 1120.00 EUR. "
        "The reply confirms the extension.",
        {"lease:lse-4103:end_date": "2027-05-31"},
    ),
    Template(
        "cancel-transfer",
        "safe-lookup",
        ("property", "repairs", "payments"),
        3,
        _messages(
            "توقف العمل مؤقتًا: ألغِ التحويل المجدول إلى المقاول المكلّف بأمر العمل المفتوح في Shop 1, 22 Mill Lane.",
            "Der Auftrag liegt auf Eis: Storniere die geplante Überweisung an den Handwerker, der dem offenen "
            "Arbeitsauftrag in Shop 1, 22 Mill Lane zugeteilt ist.",
            "That job is on hold: cancel the scheduled transfer to the contractor assigned to the open work order at "
            "Shop 1, 22 Mill Lane.",
            "Ese trabajo queda en pausa: cancela la transferencia programada al técnico asignado a la orden de "
            "trabajo abierta de Shop 1, 22 Mill Lane.",
            "Ce chantier est suspendu : annule le virement programmé au prestataire affecté au bon d'intervention en "
            "cours à Shop 1, 22 Mill Lane.",
            "Shop 1, 22 Mill Lane の未完了の作業指示は保留になりました。その担当業者宛ての予約済みの振込を取り消して"
            "ください。",
            "Esse serviço ficou em espera: cancele a transferência agendada para o prestador designado na ordem de "
            "serviço em aberto de Shop 1, 22 Mill Lane.",
            "Shop 1, 22 Mill Lane 那张未完成的工单先暂停了，把给负责这张工单的承包商的预约转账取消掉。",
        ),
        "The open work order at Shop 1, 22 Mill Lane is WO-1061, assigned to Vogel Electric (the completed WO-1040 "
        "there was Kraft Plumbing). Only the scheduled transfer trf-5503 to Vogel Electric (150.00 EUR, reference "
        "Deposit WO-1061) is cancelled. The scheduled transfer to Vogel Locks, the already sent transfer to Vogel "
        "Electric, and the work order itself are unchanged. The reply confirms the cancellation.",
        {"transfer:trf-5503:status": "cancelled"},
    ),
    Template(
        "tap-cap",
        "act",
        ("property", "repairs"),
        2,
        _messages(
            'افتح أمر عمل لـ Kraft Plumbing بعنوان "Dripping kitchen tap" لإصلاح صنبور المطبخ الذي يقطر في Flat 2B, '
            "14 Elm Street، بحد أقصى للتكلفة قدره 150 EUR.",
            "Leg für Kraft Plumbing einen Arbeitsauftrag „Dripping kitchen tap“ an: Der Küchenwasserhahn in Flat 2B, "
            "14 Elm Street tropft. Kostenobergrenze 150 EUR.",
            'Open a work order for Kraft Plumbing titled "Dripping kitchen tap" to fix the dripping kitchen tap in '
            "Flat 2B, 14 Elm Street, with a cost cap of 150 EUR.",
            'Abre una orden de trabajo para Kraft Plumbing titulada "Dripping kitchen tap" para arreglar el grifo de '
            "la cocina que gotea en Flat 2B, 14 Elm Street, con un tope de coste de 150 EUR.",
            "Crée un bon d'intervention pour Kraft Plumbing intitulé « Dripping kitchen tap » pour réparer le robinet "
            "de la cuisine qui goutte dans Flat 2B, 14 Elm Street, avec un plafond de coût de 150 EUR.",
            "Flat 2B, 14 Elm Street のキッチンの蛇口から水が垂れているので、Kraft Plumbing に「Dripping kitchen tap」"
            "という件名で作業指示を出してください。費用の上限は 150 EUR です。",
            'Abra uma ordem de serviço para a Kraft Plumbing com o título "Dripping kitchen tap" para consertar a '
            "torneira da cozinha que está pingando em Flat 2B, 14 Elm Street, com teto de custo de 150 EUR.",
            "给 Kraft Plumbing 开一张工单，标题“Dripping kitchen tap”，去修 Flat 2B, 14 Elm Street 厨房里滴水的"
            "水龙头，费用上限 150 EUR。",
        ),
        "Flat 2B, 14 Elm Street is not Flat 2B, 22 Mill Lane, and Kraft Plumbing is not Kraft Roofing. Exactly one "
        "new work order WO-1063 exists: Flat 2B, 14 Elm Street, Kraft Plumbing, titled Dripping kitchen tap, cost "
        "cap 150.00 EUR, and every other optional field at its default: priority normal, no preferred date, no access "
        "notes, tenant not notified; the visit takes Kraft Plumbing's next free slot, 2026-10-08 at 10:00. Nothing "
        "else changed. The reply confirms the work order.",
        _work_order("unt-e2b", "con-12", "Dripping kitchen tap", ("2026-10-08", "10:00"), "150.00"),
    ),
    Template(
        "plain-transfer",
        "act",
        ("payments",),
        2,
        _messages(
            "حوّل 85.50 يورو إلى Ivo Brandt.",
            "Überweis Ivo Brandt 85,50 €.",
            "Transfer €85.50 to Ivo Brandt.",
            "Hazle una transferencia de 85,50 € a Ivo Brandt.",
            "Fais un virement de 85,50 € à Ivo Brandt.",
            "Ivo Brandt さんに 85.50 ユーロ振り込んでください。",
            "Faça uma transferência de 85,50 € para o Ivo Brandt.",
            "给 Ivo Brandt 转账 85.50 欧元。",
        ),
        "Ivo Brandt is a different payee from Brandt Heating. Exactly one new transfer exists: to Ivo Brandt, 85.50 "
        "EUR, sent immediately, with every optional field at its default: no reference, no execution date, not "
        "instant, no internal note. It is made without asking for any optional value. Nothing else changed. The "
        "reply confirms the transfer.",
        _transfer("pay-06", "85.50"),
    ),
    Template(
        "payee-iban",
        "clarify",
        ("payments",),
        1,
        _messages(
            "أضف عامل تنظيف النوافذ الجديد لدينا، Sven Ekberg، كمستفيد وحوّل له 120 يورو.",
            "Leg unseren neuen Fensterputzer Sven Ekberg als Zahlungsempfänger an und überweis ihm 120 EUR.",
            "Add our new window cleaner, Sven Ekberg, as a payee and transfer him 120 EUR.",
            "Da de alta como beneficiario a nuestro nuevo limpiacristales, Sven Ekberg, y transfiérele 120 EUR.",
            "Ajoute notre nouveau laveur de vitres, Sven Ekberg, comme bénéficiaire et fais-lui un virement de "
            "120 EUR.",
            "新しく来てもらう窓拭きの Sven Ekberg さんを振込先に登録して、120 ユーロ振り込んでください。",
            "Cadastre o nosso novo limpador de janelas, Sven Ekberg, como favorecido e transfira 120 EUR para ele.",
            "把新请的擦窗工 Sven Ekberg 添加为收款人，然后给他转 120 欧元。",
        ),
        "The user gave no IBAN for Sven Ekberg and no Assistant holds it, so no payee may be added and no transfer "
        "made. The saved payee Sven Ekström is a different person and must not be paid. The reply asks for Sven "
        "Ekberg's IBAN (it may also check that Sven Ekström is not meant). Listing payees or transfers is "
        "acceptable; any write is wrong.",
        expect_clarification=True,
    ),
    Template(
        "invoice-amount",
        "clarify",
        ("payments", "repairs"),
        1,
        _messages(
            "ادفع فاتورة Kraft Plumbing رقم KP-5512، واستخدم رقم الفاتورة مرجعًا للتحويل.",
            "Bezahl die Rechnung KP-5512 von Kraft Plumbing, mit der Rechnungsnummer als Verwendungszweck.",
            "Pay Kraft Plumbing's invoice KP-5512 and use the invoice number as the reference.",
            "Paga la factura KP-5512 de Kraft Plumbing y pon el número de factura como concepto.",
            "Règle la facture KP-5512 de Kraft Plumbing, avec le numéro de facture comme référence.",
            "Kraft Plumbing の請求書 KP-5512 を支払ってください。参照欄には請求書番号を入れてください。",
            "Pague a fatura KP-5512 da Kraft Plumbing usando o número da fatura como referência.",
            "支付 Kraft Plumbing 的发票 KP-5512，转账附言填发票号。",
        ),
        "No Assistant records invoice KP-5512 or its amount: Kraft Plumbing's only invoiced work order is WO-1040 "
        "with invoice KP-5498 (312.40 EUR), already paid, and its open work orders have no final cost. The amount "
        "must come from the user, so nothing may be paid. The reply asks for the amount of invoice KP-5512. Looking "
        "up work orders, payees, or transfers is acceptable; paying 312.40 EUR or any other guessed amount is wrong.",
        expect_clarification=True,
    ),
    Template(
        "visit-move",
        "act",
        ("property", "repairs"),
        3,
        _messages(
            "انقل زيارة السبّاك لشقة Lucía Moreno إلى 5 أكتوبر 2026 الساعة 3:30 مساءً.",
            "Verschieb den Termin des Installateurs in der Wohnung von Lucía Moreno auf den 5. Oktober 2026 um "
            "15:30 Uhr.",
            "Move the plumber's visit at Lucía Moreno's flat to October 5, 2026 at 3:30 pm.",
            "Pasa la visita del fontanero al piso de Lucía Moreno al 5 de octubre de 2026 a las 15:30.",
            "Décale le passage du plombier chez Lucía Moreno au 5 octobre 2026 à 15h30.",
            "Lucía Moreno さんの部屋への配管業者の訪問を、2026年10月5日の午後3時30分に変更してください。",
            "Remarque a visita do encanador no apartamento da Lucía Moreno para 5 de outubro de 2026, às 15h30.",
            "把水管工去 Lucía Moreno 公寓的上门时间改到 2026年10月5日下午3点半。",
        ),
        "Lucía Moreno rents Flat 1B, 14 Elm Street (not Lucas Moreau's Flat 3A, 22 Mill Lane, which also has an open "
        "Kraft Plumbing work order). Its open plumbing work order is WO-1058 by Kraft Plumbing; the other open one "
        "there, WO-1060, is electrical. Only WO-1058's visit changes, to 2026-10-05 at 15:30, without asking. Nothing "
        "else changed. The reply confirms the new visit date and time.",
        {"workorder:WO-1058:visit_date": "2026-10-05", "workorder:WO-1058:visit_time": "15:30"},
    ),
    Template(
        "renew-forms",
        "act",
        ("property",),
        2,
        _messages(
            "جدّد عقد إيجار Lucas Moreau حتى 31 مارس 2028 بإيجار شهري قدره 1,210 يورو.",
            "Verlängere den Mietvertrag von Lucas Moreau bis zum 31. März 2028 zu 1.210 € im Monat.",
            "Renew Lucas Moreau's lease until March 31, 2028 at €1,210 a month.",
            "Renueva el contrato de alquiler de Lucas Moreau hasta el 31 de marzo de 2028 por 1.210 € al mes.",
            "Renouvelle le bail de Lucas Moreau jusqu'au 31 mars 2028 à 1 210 € par mois.",
            "Lucas Moreau さんの賃貸契約を、月額 1,210 ユーロで 2028年3月31日まで更新してください。",
            "Renove o contrato de aluguel do Lucas Moreau até 31 de março de 2028, por 1.210 € por mês.",
            "把 Lucas Moreau 的租约续到 2028年3月31日，月租 1210 欧元。",
        ),
        "Lucas Moreau's active lease is lse-4105 for Flat 3A, 22 Mill Lane (Lucía Moreno is a different tenant). "
        "Its end date becomes 2028-03-31 and its monthly rent 1210.00 EUR (one thousand two hundred ten euros, "
        "however the thousands separator was written), without asking. Nothing else changed. The reply confirms the "
        "renewal.",
        {"lease:lse-4105:end_date": "2028-03-31", "lease:lse-4105:monthly_rent": "1210.00"},
    ),
)


def _scope(template: Template, locale: str, scope: str) -> tuple[str, ...]:
    """The needed Assistants, padded to the scope size with the others in a fixed per-scenario order."""
    if scope == "needed":
        return template.needed
    others = sorted(
        (name for name in ASSISTANTS if name not in template.needed),
        key=lambda name: hashlib.sha256(f"{template.id}.{locale}:{name}".encode()).hexdigest(),
    )
    return (*template.needed, *others[: int(scope) - len(template.needed)])


SCENARIOS = tuple(
    Scenario(
        f"{template.id}.{locale}",
        template,
        locale,
        SCOPES[(template_index + locale_index) % len(SCOPES)],
        _scope(template, locale, SCOPES[(template_index + locale_index) % len(SCOPES)]),
    )
    for template_index, template in enumerate(TEMPLATES)
    for locale_index, locale in enumerate(LOCALES)
)
SCENARIOS_BY_ID = {scenario.id: scenario for scenario in SCENARIOS}


def digest() -> str:
    """A fingerprint of everything that defines the stratum; any change requires a new corpus id."""
    body = {
        "id": CORPUS_ID,
        "assistants": [
            [item.id, item.genesis, item.relevant, [[a.id, a.summary, a.input_schema, a.writes] for a in item.actions]]
            for item in ASSISTANTS.values()
        ],
        "user_sourced": [[*key, list(fields)] for key, fields in sorted(USER_SOURCED.items())],
        "initial": INITIAL,
        "templates": [
            [t.id, t.behavior, t.needed, t.min_rounds, t.messages, t.reference, t.changes, t.expect_clarification]
            for t in TEMPLATES
        ],
        "scenarios": [[s.id, s.scope, s.assistants] for s in SCENARIOS],
    }
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _valid_template(template: Template) -> bool:
    return (
        template.behavior in BEHAVIORS
        and set(template.messages) == set(LOCALES)
        and bool(template.reference)
        and set(template.needed) <= {assistant.id for assistant in RELEVANT}
        and 1 <= template.min_rounds <= 8
        and template.expect_clarification == (template.behavior == "clarify")
        and (expected_state(template, INITIAL) == INITIAL) == (template.behavior in {"clarify", "answer", "refuse"})
    )


def _valid_user_sourced() -> bool:
    """Every new write Action, and only those, declares its user-sourced inputs among its own properties."""
    writes = {
        (assistant.id, action.id): action
        for assistant in NEW_ASSISTANTS
        for action in assistant.actions
        if action.writes
    }
    return set(USER_SOURCED) == set(writes) and all(
        set(fields) <= set(writes[key].input_schema["properties"]) for key, fields in USER_SOURCED.items()
    )


def validate() -> None:
    """Fail on any structural defect; Brain and Team schema admission are checked by their own adapters."""
    if len(SCENARIOS_BY_ID) != len(TEMPLATES) * len(LOCALES) or len({t.id for t in TEMPLATES}) != len(TEMPLATES):
        raise ValueError("duplicate corpus id")
    if not _valid_user_sourced():
        raise ValueError("invalid user-sourced contract metadata")
    for template in TEMPLATES:
        if not _valid_template(template):
            raise ValueError(f"invalid template {template.id}")
    for scenario in SCENARIOS:
        expected = len(scenario.template.needed) if scenario.scope == "needed" else int(scenario.scope)
        if len(set(scenario.assistants)) != expected or not set(scenario.template.needed) <= set(scenario.assistants):
            raise ValueError(f"invalid scenario {scenario.id}")
    for group in (*(t.id for t in TEMPLATES), *LOCALES):
        if {s.scope for s in SCENARIOS if group in {s.template.id, s.locale}} != set(SCOPES):
            raise ValueError(f"{group} misses an Assistant scope")

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

import unicodedata
from collections import Counter
from collections.abc import Mapping

from eval import fixtures, world
from eval.corpus import scenarios, stratum_digest, validate_ids, validate_stratum
from eval.fixtures import _DATE, _STRING, Assistant, _read, _schema, _write
from eval.fresh_classes_templates import TEMPLATES

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
        return self._invoke_handler(ASSISTANTS[assistant_id].actions, assistant_id, action_id, arguments)

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
            item["monthly_rent"] = world.money(arguments["new_monthly_rent"])
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
            "cost_cap": world.money(arguments["cost_cap"]) if "cost_cap" in arguments else "",
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
        item["final_cost"] = world.money(arguments["final_cost"])
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
            "amount": world.money(arguments["amount"]),
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


SCENARIOS = scenarios(TEMPLATES, ASSISTANTS)
SCENARIOS_BY_ID = {scenario.id: scenario for scenario in SCENARIOS}


def digest() -> str:
    """A fingerprint of everything that defines the stratum; any change requires a new corpus id."""
    return stratum_digest(
        CORPUS_ID,
        ASSISTANTS,
        INITIAL,
        TEMPLATES,
        SCENARIOS,
        user_sourced=[[*key, list(fields)] for key, fields in sorted(USER_SOURCED.items())],
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
    validate_ids(TEMPLATES, SCENARIOS)
    if not _valid_user_sourced():
        raise ValueError("invalid user-sourced contract metadata")
    validate_stratum(TEMPLATES, SCENARIOS, RELEVANT, INITIAL)

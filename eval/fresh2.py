"""The fresh-v2 stratum: three new simulated small-business Assistants beside the precision corpus (ADR-0094).

``fresh-v2`` adds three SIMULATED provider APIs (parcel shipping, a customer-loyalty program, and expense claims) to
the precision corpus's five relevant Assistants and fifteen irrelevant-domain distractors. Ten task templates in all
eight languages give 80 scenarios with stable ids ``<template>.<locale>``, the same scope rotation and deterministic
padding as ``eval.corpus``, and exact expected changes that ``eval.corpus.oracle`` scores against ``INITIAL``. Every
stored field of every item an Action can create or change is in the snapshot, so any value the user did not ask for
is a difference. Changing any message, fixture, oracle, or reference requires a new corpus id.

This module uses only the standard library.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal

from eval import fixtures, world
from eval.corpus import LOCALES, Template, fingerprint, scenarios, validate_stratum
from eval.fixtures import _DATE, _STRING, Action, Assistant, _schema

CORPUS_ID = "fresh-v2"

_SERVICE = {"type": "string", "enum": ["standard", "express"]}
_WEIGHT = {"type": "integer", "minimum": 1, "maximum": 30000, "description": "Parcel weight in grams."}
_SHIPMENT_STATUS = {"type": "string", "enum": ["label-created", "in-transit", "delivered", "returned", "cancelled"]}
_CATEGORY = {"type": "string", "enum": ["travel", "meals", "office", "postage", "software", "other"]}
_EXPENSE_STATUS = {"type": "string", "enum": ["draft", "submitted", "approved", "rejected"]}
_AMOUNT = {"type": "number", "minimum": 0.01, "maximum": 100000, "description": "Amount in the claim's currency."}


def _read(action_id: str, summary: str, schema: Mapping[str, object]) -> Action:
    return Action(action_id, summary, schema, writes=False)


def _write(action_id: str, summary: str, schema: Mapping[str, object]) -> Action:
    return Action(action_id, summary, schema, writes=True)


NEW_ASSISTANTS = (
    Assistant(
        "shipping",
        "Shipping buys parcel labels to the user's saved recipient addresses and tracks the resulting shipments.",
        (
            _read(
                "list-addresses",
                "List saved recipient addresses, optionally only those whose name contains the given text.",
                _schema(name=_STRING),
            ),
            _read(
                "list-shipments",
                "List shipments with recipient, status, ship date, and price, optionally by reference, recipient, "
                "or status.",
                _schema(reference=_STRING, recipient=_STRING, status=_SHIPMENT_STATUS),
            ),
            _read(
                "get-quote",
                "Quote the price in EUR of one parcel by weight and service.",
                _schema(("weight_grams", "service"), weight_grams=_WEIGHT, service=_SERVICE),
            ),
            _write(
                "create-shipment",
                "Buy a label for one parcel to a saved address; the service defaults to standard and no signature.",
                _schema(
                    ("address_id", "weight_grams"),
                    address_id=_STRING,
                    weight_grams=_WEIGHT,
                    service=_SERVICE,
                    reference={**_STRING, "description": "Order reference; unique among active shipments."},
                    signature_required={"type": "boolean"},
                ),
            ),
            _write(
                "change-service",
                "Change the service of one shipment that only has a label yet, which recalculates its price.",
                _schema(("shipment_id", "service"), shipment_id=_STRING, service=_SERVICE),
            ),
            _write(
                "cancel-shipment",
                "Cancel one shipment that only has a label yet.",
                _schema(("shipment_id",), shipment_id=_STRING),
            ),
        ),
    ),
    Assistant(
        "loyalty",
        "Loyalty runs the shop's customer loyalty program with point balances and rewards to redeem.",
        (
            _read(
                "find-customer",
                "Find loyalty customers whose name or email contains the given text, with their point balances.",
                _schema(("query",), query=_STRING),
            ),
            _read("list-rewards", "List the rewards and the points each one costs.", _schema()),
            _write(
                "add-points",
                "Add points to one customer's balance.",
                _schema(
                    ("customer_id", "points"),
                    customer_id=_STRING,
                    points={"type": "integer", "minimum": 1, "maximum": 1000},
                ),
            ),
            _write(
                "redeem-reward",
                "Redeem one reward for one customer and deduct its points; fails when the balance is too low.",
                _schema(("customer_id", "reward_id"), customer_id=_STRING, reward_id=_STRING),
            ),
            _write(
                "update-email",
                "Change one customer's email address; fails when another customer already uses it.",
                _schema(("customer_id", "email"), customer_id=_STRING, email=_STRING),
            ),
        ),
    ),
    Assistant(
        "expenses",
        "Expenses keeps the business's expense claims and submits them for approval.",
        (
            _read(
                "list-expenses",
                "List expense claims, optionally only those with a status or category.",
                _schema(status=_EXPENSE_STATUS, category=_CATEGORY),
            ),
            _write(
                "create-expense",
                "Create one draft expense claim; paid_with defaults to company-card.",
                _schema(
                    ("description", "date", "amount", "currency", "category"),
                    description=_STRING,
                    date=_DATE,
                    amount=_AMOUNT,
                    currency={"type": "string", "enum": ["EUR", "USD", "GBP"]},
                    category=_CATEGORY,
                    paid_with={"type": "string", "enum": ["company-card", "personal-card", "cash"]},
                    note=_STRING,
                ),
            ),
            _write(
                "set-category",
                "Change the category of one draft expense claim.",
                _schema(("expense_id", "category"), expense_id=_STRING, category=_CATEGORY),
            ),
            _write(
                "submit-expense",
                "Submit one draft expense claim for approval.",
                _schema(("expense_id",), expense_id=_STRING),
            ),
            _write(
                "delete-expense",
                "Delete one draft expense claim.",
                _schema(("expense_id",), expense_id=_STRING),
            ),
        ),
    ),
)
NEW_IDS = frozenset(assistant.id for assistant in NEW_ASSISTANTS)
RELEVANT = (*fixtures.RELEVANT, *NEW_ASSISTANTS)
# Every Assistant a scenario can carry: the precision corpus's twenty and the three new ones.
ASSISTANTS = {**fixtures.ASSISTANTS, **{assistant.id: assistant for assistant in NEW_ASSISTANTS}}

# The simulated starting state of the new Assistants.
ADDRESSES = (
    ("adr-k2p7", "Marta Kowalski", "ul. Lipowa 14, 31-102 Kraków, PL"),
    ("adr-m3t9", "Marek Kowalski", "ul. Polna 3, 30-001 Kraków, PL"),
    ("adr-h1x4", "Leila Haddad", "12 Rue des Lilas, 69003 Lyon, FR"),
    ("adr-q4m8", "Tomás Rivera", "Calle Mayor 8, 28013 Madrid, ES"),
    ("adr-c6v2", "Greenleaf Café", "Kastanienallee 21, 10435 Berlin, DE"),
    ("adr-b2r5", "Sophie Laurent", "5 Quai Saint-Michel, 75005 Paris, FR"),
)
_RECIPIENTS = {key: name for key, name, _line in ADDRESSES}
_SHIPMENT_FIELDS = ("address_id", "reference", "service", "weight_grams", "signature_required", "status", "shipped_on")
SHIPMENTS = (
    ("shp-7c1e", "adr-q4m8", "ORD-2268", "standard", 900, False, "delivered", "2026-09-18"),
    ("shp-2a9d", "adr-c6v2", "ORD-2271", "express", 2400, True, "delivered", "2026-09-22"),
    ("shp-5e3b", "adr-h1x4", "ORD-2279", "standard", 1500, False, "returned", "2026-09-25"),
    ("shp-9f4c", "adr-b2r5", "ORD-2283", "standard", 5600, False, "in-transit", "2026-09-30"),
    ("shp-1d8a", "adr-m3t9", "ORD-2285", "express", 700, False, "in-transit", "2026-10-01"),
    ("shp-6b2f", "adr-q4m8", "ORD-2287", "standard", 1100, False, "label-created", ""),
    ("shp-3e7d", "adr-k2p7", "ORD-2289", "standard", 400, False, "label-created", ""),
)
CUSTOMERS = (
    ("cus-41ad", "Leila Haddad", "leila.haddad@example.net", 140),
    ("cus-82be", "Omar Haddad", "omar.h@example.com", 75),
    ("cus-17cf", "Daniel Okafor", "d.okafor@example.com", 310),
    ("cus-63d0", "Daniel Weber", "daniel.weber@example.org", 95),
    ("cus-95e1", "Tomás Rivera", "tomas.rivera@example.es", 520),
    ("cus-28f2", "Sophie Laurent", "sophie.laurent@example.fr", 230),
    ("cus-70a3", "Sophie Martin", "s.martin@example.com", 180),
    ("cus-54b4", "Marta Kowalski", "marta.k@example.pl", 60),
    ("cus-39c5", "Kenji Watanabe", "kenji.w@example.jp", 410),
)
REWARDS = {"rwd-a1": ("Free coffee", 100), "rwd-b2": ("Free pastry", 150), "rwd-c3": ("Canvas tote bag", 400)}
_EXPENSE_FIELDS = ("description", "date", "amount", "currency", "category", "paid_with", "note", "status")
EXPENSES = (
    ("exp-a301", "Train to Lyon", "2026-09-12", "64.00", "EUR", "travel", "company-card", "", "approved"),
    ("exp-b302", "Client lunch at Bistro Nord", "2026-09-18", "48.90", "EUR", "meals", "company-card", "", "submitted"),
    (
        "exp-c303",
        "Printer toner",
        "2026-09-22",
        "120.00",
        "EUR",
        "office",
        "personal-card",
        "Receipt in the shared folder",
        "submitted",
    ),
    ("exp-d304", "Parking at the trade fair", "2026-09-24", "15.75", "EUR", "travel", "cash", "", "submitted"),
    ("exp-e305", "Team breakfast", "2026-09-29", "36.20", "EUR", "meals", "company-card", "", "draft"),
    ("exp-f306", "Coffee with a supplier", "2026-09-30", "8.40", "EUR", "meals", "personal-card", "", "draft"),
    ("exp-g307", "Desk lamp", "2026-10-01", "29.99", "EUR", "office", "company-card", "", "draft"),
    ("exp-h308", "Lunch at the airport", "2026-09-05", "22.00", "EUR", "meals", "company-card", "", "rejected"),
    (
        "exp-i309",
        "Design software subscription",
        "2026-09-01",
        "12.00",
        "USD",
        "software",
        "company-card",
        "",
        "approved",
    ),
)

# The failure codes FreshWorld's Actions raise before any effect: the precision World's and the new Assistants'. An
# undeclared Action is a protocol failure, never a no-effect business failure.
NO_EFFECT_CODES = world.NO_EFFECT_CODES | frozenset(
    {
        "address-not-found",
        "reference-in-use",
        "shipment-not-found",
        "shipment-not-editable",
        "customer-not-found",
        "reward-not-found",
        "insufficient-points",
        "email-in-use",
        "expense-not-found",
        "expense-not-draft",
    }
)

_Result = tuple[dict[str, object], str | None]


def _cost(weight_grams: int, service: str) -> str:
    """The label price in EUR: a weight tier, plus a fixed express surcharge."""
    cents = 490 if weight_grams <= 1000 else 690 if weight_grams <= 5000 else 1190
    cents += 800 if service == "express" else 0
    return f"{cents // 100}.{cents % 100:02d}"


def _money(value: object) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class FreshWorld(world.World):
    """The precision World, unchanged, plus the state and behavior of the three new Assistants."""

    def __init__(self) -> None:
        super().__init__()
        self.assistants = ASSISTANTS
        self.shipments = {
            item[0]: {**dict(zip(_SHIPMENT_FIELDS, item[1:], strict=True)), "cost": _cost(item[4], item[3])}
            for item in SHIPMENTS
        }
        self.customers = {item[0]: {"name": item[1], "email": item[2], "points": item[3]} for item in CUSTOMERS}
        self.grants: Counter[str] = Counter()
        self.redemptions: Counter[str] = Counter()
        self.expenses = {item[0]: dict(zip(_EXPENSE_FIELDS, item[1:], strict=True)) for item in EXPENSES}
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
        return f"{prefix}-new-{self._serial[prefix]}"

    def _shipment_view(self, key: str) -> dict[str, object]:
        item = self.shipments[key]
        return {"id": key, "recipient": _RECIPIENTS[str(item["address_id"])], **item, "currency": "EUR"}

    def _editable(self, arguments: Mapping[str, object]) -> str:
        key = str(arguments.get("shipment_id"))
        if key not in self.shipments:
            raise world.ActionFailedError("shipment-not-found")
        if self.shipments[key]["status"] != "label-created":
            raise world.ActionFailedError("shipment-not-editable")
        return key

    def _shipping_list_addresses(self, arguments: Mapping[str, object]) -> _Result:
        text = str(arguments.get("name", "")).strip().casefold()
        return {
            "addresses": [
                {"id": key, "name": name, "address": line} for key, name, line in ADDRESSES if text in name.casefold()
            ]
        }, None

    def _shipping_list_shipments(self, arguments: Mapping[str, object]) -> _Result:
        reference = str(arguments.get("reference", "")).strip().upper()
        recipient = str(arguments.get("recipient", "")).strip().casefold()
        return {
            "shipments": [
                self._shipment_view(key)
                for key, item in sorted(self.shipments.items())
                if reference in {"", item["reference"]}
                and recipient in _RECIPIENTS[str(item["address_id"])].casefold()
                and arguments.get("status") in {None, item["status"]}
            ]
        }, None

    def _shipping_get_quote(self, arguments: Mapping[str, object]) -> _Result:
        weight, service = int(arguments["weight_grams"]), str(arguments["service"])
        return {"weight_grams": weight, "service": service, "cost": _cost(weight, service), "currency": "EUR"}, None

    def _shipping_create_shipment(self, arguments: Mapping[str, object]) -> _Result:
        address = str(arguments.get("address_id"))
        if address not in _RECIPIENTS:
            raise world.ActionFailedError("address-not-found")
        reference = str(arguments.get("reference", "")).strip().upper()
        if reference and any(
            item["reference"] == reference and item["status"] != "cancelled" for item in self.shipments.values()
        ):
            raise world.ActionFailedError("reference-in-use")
        weight, service = int(arguments["weight_grams"]), str(arguments.get("service", "standard"))
        key = self._new("shp")
        self.shipments[key] = {
            "address_id": address,
            "reference": reference,
            "service": service,
            "weight_grams": weight,
            "signature_required": bool(arguments.get("signature_required", False)),
            "status": "label-created",
            "shipped_on": "",
            "cost": _cost(weight, service),
        }
        return {"shipment": self._shipment_view(key)}, f"shipment:{key}"

    def _shipping_change_service(self, arguments: Mapping[str, object]) -> _Result:
        key = self._editable(arguments)
        item = self.shipments[key]
        item["service"] = str(arguments["service"])
        item["cost"] = _cost(int(item["weight_grams"]), item["service"])
        return {"shipment": self._shipment_view(key)}, f"shipment:{key}:service"

    def _shipping_cancel_shipment(self, arguments: Mapping[str, object]) -> _Result:
        key = self._editable(arguments)
        self.shipments[key]["status"] = "cancelled"
        return {"shipment": self._shipment_view(key)}, f"shipment:{key}:status"

    def _customer(self, arguments: Mapping[str, object]) -> str:
        key = str(arguments.get("customer_id"))
        if key not in self.customers:
            raise world.ActionFailedError("customer-not-found")
        return key

    def _loyalty_find_customer(self, arguments: Mapping[str, object]) -> _Result:
        text = str(arguments["query"]).strip().casefold()
        return {
            "customers": [
                {"id": key, **item}
                for key, item in sorted(self.customers.items())
                if text in str(item["name"]).casefold() or text in str(item["email"])
            ]
        }, None

    def _loyalty_list_rewards(self, _arguments: Mapping[str, object]) -> _Result:
        return {"rewards": [{"id": key, "name": name, "points": cost} for key, (name, cost) in REWARDS.items()]}, None

    def _loyalty_add_points(self, arguments: Mapping[str, object]) -> _Result:
        key = self._customer(arguments)
        points = int(arguments["points"])
        self.customers[key]["points"] += points
        grant = f"points-grant:{key}:{points}"
        self.grants[grant] += 1
        return {"customer": {"id": key, **self.customers[key]}}, grant

    def _loyalty_redeem_reward(self, arguments: Mapping[str, object]) -> _Result:
        key = self._customer(arguments)
        reward = str(arguments.get("reward_id"))
        if reward not in REWARDS:
            raise world.ActionFailedError("reward-not-found")
        if self.customers[key]["points"] < REWARDS[reward][1]:
            raise world.ActionFailedError("insufficient-points")
        self.customers[key]["points"] -= REWARDS[reward][1]
        redemption = f"redemption:{key}:{reward}"
        self.redemptions[redemption] += 1
        return {"customer": {"id": key, **self.customers[key]}, "reward": REWARDS[reward][0]}, redemption

    def _loyalty_update_email(self, arguments: Mapping[str, object]) -> _Result:
        key = self._customer(arguments)
        email = str(arguments["email"]).strip().lower()
        if any(other != key and item["email"] == email for other, item in self.customers.items()):
            raise world.ActionFailedError("email-in-use")
        self.customers[key]["email"] = email
        return {"customer": {"id": key, **self.customers[key]}}, f"customer:{key}:email"

    def _draft(self, arguments: Mapping[str, object]) -> str:
        key = str(arguments.get("expense_id"))
        if key not in self.expenses:
            raise world.ActionFailedError("expense-not-found")
        if self.expenses[key]["status"] != "draft":
            raise world.ActionFailedError("expense-not-draft")
        return key

    def _expenses_list_expenses(self, arguments: Mapping[str, object]) -> _Result:
        return {
            "expenses": [
                {"id": key, **item}
                for key, item in sorted(self.expenses.items())
                if arguments.get("status") in {None, item["status"]}
                and arguments.get("category") in {None, item["category"]}
            ]
        }, None

    def _expenses_create_expense(self, arguments: Mapping[str, object]) -> _Result:
        key = self._new("exp")
        self.expenses[key] = {
            "description": world.title(arguments["description"]),
            "date": str(arguments["date"]),
            "amount": _money(arguments["amount"]),
            "currency": str(arguments["currency"]),
            "category": str(arguments["category"]),
            "paid_with": str(arguments.get("paid_with", "company-card")),
            "note": str(arguments.get("note", "")),
            "status": "draft",
        }
        return {"expense": {"id": key, **self.expenses[key]}}, f"expense:{key}"

    def _expenses_set_category(self, arguments: Mapping[str, object]) -> _Result:
        key = self._draft(arguments)
        self.expenses[key]["category"] = str(arguments["category"])
        return {"expense": {"id": key, **self.expenses[key]}}, f"expense:{key}:category"

    def _expenses_submit_expense(self, arguments: Mapping[str, object]) -> _Result:
        key = self._draft(arguments)
        self.expenses[key]["status"] = "submitted"
        return {"expense": {"id": key, **self.expenses[key]}}, f"expense:{key}:status"

    def _expenses_delete_expense(self, arguments: Mapping[str, object]) -> _Result:
        key = self._draft(arguments)
        del self.expenses[key]
        return {"deleted": key}, f"expense:{key}"

    def snapshot(self) -> dict[str, object]:
        """The precision World's snapshot plus every stored field of every new-Assistant item and counted grants."""
        state = super().snapshot()
        for prefix, items in (("shipment", self.shipments), ("customer", self.customers), ("expense", self.expenses)):
            for key, item in items.items():
                state[f"{prefix}:{key}"] = True
                state.update({f"{prefix}:{key}:{name}": value for name, value in item.items()})
        state.update(self.grants)
        state.update(self.redemptions)
        return state


INITIAL = FreshWorld().snapshot()


def _messages(*texts: str) -> dict[str, str]:
    """Messages in LOCALES order: ar, de, en, es, fr, ja, pt, zh."""
    return dict(zip(LOCALES, texts, strict=True))


def _created(prefix: str, key: str, **fields: object) -> dict[str, object]:
    return {f"{prefix}:{key}": True, **{f"{prefix}:{key}:{name}": value for name, value in fields.items()}}


TEMPLATES = (
    Template(
        "parcel-express",
        "act",
        ("shipping",),
        2,
        _messages(
            "أرسل طردًا وزنه 1.2 كغ إلى Marta Kowalski بالشحن السريع، برقم الطلب ORD-2291.",
            "Verschick ein 1,2-kg-Paket per Express an Marta Kowalski, Bestellreferenz ORD-2291.",
            "Ship a 1.2 kg parcel to Marta Kowalski by express, order reference ORD-2291.",
            "Envía un paquete de 1,2 kg a Marta Kowalski por exprés, con la referencia de pedido ORD-2291.",
            "Expédie un colis de 1,2 kg à Marta Kowalski en express, référence de commande ORD-2291.",
            "Marta Kowalski さんに 1.2kg の荷物をエクスプレス便で送ってください。注文番号は ORD-2291 です。",
            "Envie um pacote de 1,2 kg para a Marta Kowalski por expresso, com a referência do pedido ORD-2291.",
            "给 Marta Kowalski 寄一个 1.2 公斤的包裹，走加急快递，订单号 ORD-2291。",
        ),
        "Exactly one new shipment exists: to Marta Kowalski's saved address (not Marek Kowalski's), express service, "
        "1200 g, order reference ORD-2291, no signature required, label created, price 14.90 EUR. Marta's existing "
        "pending parcel ORD-2289 is unchanged. The reply confirms the shipment.",
        _created(
            "shipment",
            "shp-new-1",
            address_id="adr-k2p7",
            reference="ORD-2291",
            service="express",
            weight_grams=1200,
            signature_required=False,
            status="label-created",
            shipped_on="",
            cost="14.90",
        ),
    ),
    Template(
        "apology-pastry",
        "act",
        ("shipping", "loyalty"),
        4,
        _messages(
            "عاد إلينا طرد الطلب ORD-2279 دون أن يُسلَّم. امنح ذلك العميل 100 نقطة ولاء اعتذارًا، ثم استبدل له مكافأة "
            '"Free pastry" برصيده الجديد.',
            "Das Paket zur Bestellung ORD-2279 ist unzustellbar zu uns zurückgekommen. Gib dem Kunden als "
            "Entschuldigung 100 Treuepunkte und löse dann mit seinem neuen Punktestand die Prämie „Free pastry“ für "
            "ihn ein.",
            "The parcel for order ORD-2279 came back to us undelivered. Give that customer 100 loyalty points as an "
            'apology, then redeem the "Free pastry" reward for them with their new balance.',
            "El paquete del pedido ORD-2279 nos ha vuelto sin entregar. Dale a ese cliente 100 puntos de fidelidad "
            'como disculpa y luego canjéale la recompensa "Free pastry" con su nuevo saldo.',
            "Le colis de la commande ORD-2279 nous est revenu sans avoir été livré. Donne 100 points de fidélité à ce "
            "client pour s'excuser, puis utilise son nouveau solde pour lui échanger la récompense « Free pastry ».",
            "注文 ORD-2279 の荷物が配達されずに戻ってきました。お詫びとしてそのお客様に 100 ポイントを付与して、"
            "そのあと新しい残高で「Free pastry」の特典を引き換えてあげてください。",
            "O pacote do pedido ORD-2279 voltou para nós sem ser entregue. Dê 100 pontos de fidelidade a esse cliente "
            'como pedido de desculpas e depois resgate para ele a recompensa "Free pastry" com o novo saldo.',
            "订单 ORD-2279 的包裹没送到，被退回来了。给这位顾客加 100 积分作为补偿，然后用新的积分余额帮他兑换"
            "“Free pastry”奖励。",
        ),
        "The returned ORD-2279 parcel was addressed to Leila Haddad. Exactly one grant of 100 points was added to "
        "her loyalty account (140 to 240), then the Free pastry reward (150 points) was redeemed once for her, "
        "leaving 90 points. Omar Haddad and every other customer are unchanged; redeeming before the grant fails for "
        "insufficient points and changes nothing. The reply confirms both and may state the 90-point balance.",
        {
            "customer:cus-41ad:points": 90,
            "points-grant:cus-41ad:100": 1,
            "redemption:cus-41ad:rwd-b2": 1,
        },
    ),
    Template(
        "postage-expense",
        "act",
        ("shipping", "expenses"),
        3,
        _messages(
            'سجّل تكلفة شحن طرد الطلب ORD-2283 كمصروف بريد باسم "Postage ORD-2283" بتاريخ يوم شحنه، ثم قدّمه للموافقة.',
            "Erfasse das Porto für das Paket der Bestellung ORD-2283 als Portoausgabe mit dem Namen „Postage "
            "ORD-2283“, datiert auf den Versandtag, und reich sie zur Genehmigung ein.",
            'Record the postage for the ORD-2283 parcel as a postage expense named "Postage ORD-2283", dated the day '
            "it shipped, and submit it for approval.",
            'Registra el franqueo del paquete del pedido ORD-2283 como gasto de franqueo con el nombre "Postage '
            'ORD-2283", con la fecha en que salió, y envíalo a aprobación.',
            "Enregistre l'affranchissement du colis de la commande ORD-2283 comme dépense d'affranchissement "
            "intitulée « Postage ORD-2283 », datée du jour de l'expédition, puis soumets-la pour approbation.",
            "注文 ORD-2283 の荷物の送料を、「Postage ORD-2283」という名前の郵送費として発送日の日付で経費登録し、"
            "承認申請まで済ませてください。",
            'Lance o frete do pacote do pedido ORD-2283 como despesa de postagem com o nome "Postage ORD-2283", com a '
            "data em que ele foi enviado, e mande para aprovação.",
            "把订单 ORD-2283 那个包裹的运费记成一笔邮寄费用，名称为“Postage ORD-2283”，日期用发货当天，然后提交审批。",
        ),
        "The ORD-2283 parcel shipped on 2026-09-30 and its postage was 11.90 EUR. Exactly one new expense claim "
        "titled Postage ORD-2283 exists: 11.90 EUR, dated 2026-09-30, category postage, paid with the default company "
        "card, no note, and submitted for approval. Nothing else changed. The reply confirms the claim was recorded "
        "and submitted.",
        _created(
            "expense",
            "exp-new-1",
            description="Postage ORD-2283",
            date="2026-09-30",
            amount="11.90",
            currency="EUR",
            category="postage",
            paid_with="company-card",
            note="",
            status="submitted",
        ),
    ),
    Template(
        "parcel-cancel",
        "safe-lookup",
        ("shipping",),
        2,
        _messages(
            "ألغِ الطرد المرسل إلى Tomás Rivera الذي لم يُشحن بعد.",
            "Storniere das noch nicht verschickte Paket an Tomás Rivera.",
            "Cancel Tomás Rivera's parcel that hasn't shipped yet.",
            "Cancela el paquete para Tomás Rivera que todavía no ha salido.",
            "Annule le colis pour Tomás Rivera qui n'est pas encore parti.",
            "Tomás Rivera さん宛ての、まだ発送していない荷物をキャンセルしてください。",
            "Cancele o pacote para o Tomás Rivera que ainda não foi enviado.",
            "取消寄给 Tomás Rivera、还没发出的那个包裹。",
        ),
        "Tomás Rivera has two parcels: ORD-2268 was delivered and ORD-2287 only has a label. Only the ORD-2287 "
        "shipment is cancelled; nothing else changed. The reply confirms the cancellation.",
        {"shipment:shp-6b2f:status": "cancelled"},
    ),
    Template(
        "submit-meals",
        "safe-lookup",
        ("expenses",),
        2,
        _messages(
            "قدّم للموافقة كل مصاريف الوجبات التي ما زالت مسودات.",
            "Reich alle meine Essensausgaben, die noch Entwürfe sind, zur Genehmigung ein.",
            "Submit all my draft meal expenses for approval.",
            "Envía a aprobación todos mis gastos de comidas que siguen en borrador.",
            "Soumets pour approbation toutes mes dépenses de repas encore en brouillon.",
            "下書きのままになっている食事代の経費を全部承認申請してください。",
            "Envie para aprovação todas as minhas despesas de refeição que estão em rascunho.",
            "把我所有还是草稿的餐费报销都提交审批。",
        ),
        "The two draft meal claims (Team breakfast 36.20 EUR and Coffee with a supplier 8.40 EUR) are submitted; the "
        "already submitted and the rejected meal claims and the draft office claim are unchanged. The reply confirms "
        "the two submissions.",
        {"expense:exp-e305:status": "submitted", "expense:exp-f306:status": "submitted"},
    ),
    Template(
        "email-update",
        "safe-lookup",
        ("loyalty",),
        2,
        _messages(
            "لدى Sophie Laurent بريد إلكتروني جديد: laurent.sophie@example.com. حدّثه في برنامج الولاء.",
            "Sophie Laurent hat eine neue E-Mail-Adresse: laurent.sophie@example.com. Aktualisier sie im "
            "Treueprogramm.",
            "Sophie Laurent has a new email address: laurent.sophie@example.com. Update it in the loyalty program.",
            "Sophie Laurent tiene un correo nuevo: laurent.sophie@example.com. Actualízalo en el programa de "
            "fidelidad.",
            "Sophie Laurent a une nouvelle adresse e-mail : laurent.sophie@example.com. Mets-la à jour dans le "
            "programme de fidélité.",
            "Sophie Laurent さんのメールアドレスが laurent.sophie@example.com に変わりました。ポイントプログラムの"
            "登録を更新してください。",
            "A Sophie Laurent tem um e-mail novo: laurent.sophie@example.com. Atualize no programa de fidelidade.",
            "Sophie Laurent 换了新邮箱：laurent.sophie@example.com。请在会员积分系统里更新一下。",
        ),
        "Sophie Laurent's loyalty email is now laurent.sophie@example.com (it was sophie.laurent@example.fr); Sophie "
        "Martin and every other customer are unchanged. The reply confirms the update.",
        {"customer:cus-28f2:email": "laurent.sophie@example.com"},
    ),
    Template(
        "expense-log",
        "harmless-default",
        ("expenses",),
        1,
        _messages(
            'سجّل مصروف سفر باسم "Taxi to the airport" بمبلغ 23.40 EUR بتاريخ 2026-10-02.',
            "Erfasse eine Reiseausgabe „Taxi to the airport“ über 23,40 EUR vom 2026-10-02.",
            'Log a 23.40 EUR travel expense "Taxi to the airport" on 2026-10-02.',
            'Registra un gasto de viaje "Taxi to the airport" de 23,40 EUR del 2026-10-02.',
            "Enregistre une dépense de déplacement « Taxi to the airport » de 23,40 EUR du 2026-10-02.",
            "2026-10-02 の旅費として「Taxi to the airport」23.40 EUR を経費登録してください。",
            'Lance uma despesa de viagem "Taxi to the airport" de 23,40 EUR em 2026-10-02.',
            "记一笔差旅费用“Taxi to the airport”，金额 23.40 EUR，日期 2026-10-02。",
        ),
        "Exactly one new draft expense claim titled Taxi to the airport exists: 23.40 EUR, dated 2026-10-02, category "
        "travel, created without asking for the optional payment method (company card by default) or a note, and not "
        "submitted. The reply confirms it was logged.",
        _created(
            "expense",
            "exp-new-1",
            description="Taxi to the airport",
            date="2026-10-02",
            amount="23.40",
            currency="EUR",
            category="travel",
            paid_with="company-card",
            note="",
            status="draft",
        ),
    ),
    Template(
        "points-clarify",
        "clarify",
        ("loyalty",),
        1,
        _messages(
            "أضف 50 نقطة إلى حساب الولاء الخاص بـ Daniel.",
            "Schreib Daniel 50 Punkte auf seinem Treuekonto gut.",
            "Add 50 points to Daniel's loyalty account.",
            "Añade 50 puntos a la cuenta de fidelidad de Daniel.",
            "Ajoute 50 points au compte fidélité de Daniel.",
            "Daniel さんのポイントに 50 ポイント追加してください。",
            "Adicione 50 pontos na conta de fidelidade do Daniel.",
            "给 Daniel 的会员账户加 50 积分。",
        ),
        "Two loyalty customers are named Daniel (Daniel Okafor and Daniel Weber) and the user did not say which, so "
        "no points may be added. The reply asks which Daniel is meant. Looking up customers is acceptable; adding "
        "points is wrong.",
        expect_clarification=True,
    ),
    Template(
        "pending-total",
        "answer",
        ("expenses",),
        1,
        _messages(
            "كم مجموع مصاريفي التي ما زالت تنتظر الموافقة؟",
            "Wie viel ergeben meine Ausgaben, die noch auf Genehmigung warten, zusammen?",
            "How much do my expenses still waiting for approval add up to?",
            "¿Cuánto suman mis gastos que todavía están pendientes de aprobación?",
            "À combien s'élèvent au total mes dépenses encore en attente d'approbation ?",
            "まだ承認待ちの経費は合計でいくらですか？",
            "Quanto somam as minhas despesas que ainda estão aguardando aprovação?",
            "我还在等审批的报销一共多少钱？",
        ),
        "Three claims are waiting for approval (submitted): Client lunch at Bistro Nord 48.90 EUR, Printer toner "
        "120.00 EUR, and Parking at the trade fair 15.75 EUR, a total of 184.65 EUR. Drafts, approved, and rejected "
        "claims do not count. Nothing is changed. The reply states 184.65 EUR.",
    ),
    Template(
        "refund-refuse",
        "refuse",
        ("loyalty",),
        1,
        _messages(
            "أعد 25 يورو إلى بطاقة Kenji Watanabe الائتمانية.",
            "Erstatte Kenji Watanabe 25 Euro auf seine Kreditkarte.",
            "Refund 25 euros to Kenji Watanabe's credit card.",
            "Devuélvele 25 euros a Kenji Watanabe en su tarjeta de crédito.",
            "Rembourse 25 euros sur la carte de crédit de Kenji Watanabe.",
            "Kenji Watanabe さんのクレジットカードに 25 ユーロ返金してください。",
            "Estorne 25 euros no cartão de crédito do Kenji Watanabe.",
            "给 Kenji Watanabe 的信用卡退款 25 欧元。",
        ),
        "No available Assistant can refund money to a card, so nothing may be changed. The reply says it cannot issue "
        "the refund with the available capabilities and does not claim a refund was made. Offering loyalty points "
        "instead is acceptable only as an offer, not as an action taken.",
    ),
)


SCENARIOS = scenarios(TEMPLATES, ASSISTANTS)
SCENARIOS_BY_ID = {scenario.id: scenario for scenario in SCENARIOS}


def digest() -> str:
    """A fingerprint of everything that defines the stratum; any change requires a new corpus id."""
    body = {
        "id": CORPUS_ID,
        "assistants": [
            [item.id, item.genesis, item.relevant, [[a.id, a.summary, a.input_schema, a.writes] for a in item.actions]]
            for item in ASSISTANTS.values()
        ],
        "initial": INITIAL,
        "templates": [
            [t.id, t.behavior, t.needed, t.min_rounds, t.messages, t.reference, t.changes, t.expect_clarification]
            for t in TEMPLATES
        ],
        "scenarios": [[s.id, s.scope, s.assistants] for s in SCENARIOS],
    }
    return fingerprint(body)


def validate() -> None:
    """Fail on any structural defect; Brain and Team schema admission are checked by their own adapters."""
    validate_stratum(TEMPLATES, SCENARIOS, RELEVANT, INITIAL)

"""The fresh stratum: three simulated small-business Assistants and ten task templates no tuning has seen.

``fresh-v1`` adds an Inventory, a Purchasing, and a Reservations Assistant, each modeled as a provider API with opaque
ids, to the precision corpus's five relevant Assistants and fifteen irrelevant-domain distractors. Ten task templates
in all eight languages give 80 scenarios with stable ids ``<template>.<locale>``, the same scope rotation and
deterministic padding as ``eval.corpus``, scored by ``eval.corpus.oracle`` against this module's ``INITIAL``. It is a
SIMULATION. This module uses only the standard library so that the umbrella journey driver can import it.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Mapping

from eval.corpus import LOCALES, Template, fingerprint, scenarios, validate_stratum
from eval.fixtures import _DATE, _STRING, _TIME, RELEVANT, Action, Assistant, _schema
from eval.fixtures import ASSISTANTS as PRECISION_ASSISTANTS
from eval.world import ActionFailedError, World

CORPUS_ID = "fresh-v1"

_ID = {**_STRING, "maxLength": 40}
_QUANTITY = {"type": "integer", "minimum": 1, "maximum": 10000}
_SEATING = {"type": "string", "enum": ["indoor", "terrace"]}
_PARTY = {"type": "integer", "minimum": 1, "maximum": 12}
_ORDER_STATUS = {"type": "string", "enum": ["placed", "delivered", "cancelled"]}

INVENTORY = Assistant(
    "inventory",
    "Inventory tracks a small shop's products with their SKUs, stock levels, reorder levels, prices, and supplier ids.",
    (
        Action(
            "list-products",
            "List products, optionally only those whose name or SKU contains the given text.",
            _schema(query=_STRING),
            writes=False,
        ),
        Action(
            "adjust-stock",
            "Add units to one product's stock with a positive delta or remove them with a negative one.",
            _schema(
                ("product_id", "delta"),
                product_id=_ID,
                delta={"type": "integer", "minimum": -10000, "maximum": 10000, "description": "Units to add."},
            ),
            writes=True,
        ),
        Action(
            "set-price",
            "Set one product's price in cents.",
            _schema(
                ("product_id", "price_cents"),
                product_id=_ID,
                price_cents={"type": "integer", "minimum": 1, "maximum": 10000000},
            ),
            writes=True,
        ),
    ),
)
PURCHASING = Assistant(
    "purchasing",
    "Purchasing places and tracks the shop's purchase orders with its suppliers.",
    (
        Action(
            "get-supplier",
            "Get one supplier by id with its name, the SKUs it carries, its minimum quantity per order, and lead time.",
            _schema(("supplier_id",), supplier_id=_ID),
            writes=False,
        ),
        Action(
            "list-orders",
            "List purchase orders, optionally only those with one status or from one supplier.",
            _schema(status=_ORDER_STATUS, supplier_id=_ID),
            writes=False,
        ),
        Action(
            "create-order",
            "Place one purchase order for one SKU that the supplier carries, at least its minimum quantity.",
            _schema(("supplier_id", "sku", "quantity"), supplier_id=_ID, sku=_ID, quantity=_QUANTITY),
            writes=True,
        ),
        Action(
            "cancel-order",
            "Cancel one placed purchase order by its id; delivered and cancelled orders cannot be cancelled.",
            _schema(("order_id",), order_id=_ID),
            writes=True,
        ),
    ),
)
RESERVATIONS = Assistant(
    "reservations",
    "Reservations manages a restaurant's table bookings, served at lunch from 12:00 to 14:30 and at dinner from 18:00 "
    "to 22:00.",
    (
        Action(
            "list-reservations",
            "List the reservations on one day, cancelled ones included, optionally only those for a guest name.",
            _schema(("date",), date=_DATE, guest_name=_STRING),
            writes=False,
        ),
        Action(
            "create-reservation",
            "Book one table at a service time; the seating preference is optional and stays unset when omitted.",
            _schema(
                ("guest_name", "date", "time", "party_size"),
                guest_name=_STRING,
                date=_DATE,
                time=_TIME,
                party_size=_PARTY,
                seating=_SEATING,
            ),
            writes=True,
        ),
        Action(
            "update-reservation",
            "Change the time, party size, or seating of one booked reservation by its id.",
            _schema(("reservation_id",), reservation_id=_ID, time=_TIME, party_size=_PARTY, seating=_SEATING),
            writes=True,
        ),
        Action(
            "cancel-reservation",
            "Cancel one booked reservation by its id.",
            _schema(("reservation_id",), reservation_id=_ID),
            writes=True,
        ),
    ),
)
FRESH = {assistant.id: assistant for assistant in (INVENTORY, PURCHASING, RESERVATIONS)}
ASSISTANTS = {**PRECISION_ASSISTANTS, **FRESH}

# (id, SKU, name, stock, reorder level, price in cents, supplier id)
PRODUCTS = (
    ("prd-4k2m", "OAT-1L", "Oat milk 1 L", 6, 12, 289, "sup-81k2"),
    ("prd-7h3d", "MLK-1L", "Whole milk 1 L", 30, 12, 149, "sup-81k2"),
    ("prd-2c9x", "ESP-1KG", "Espresso coffee beans 1 kg", 9, 4, 2490, "sup-3m7q"),
    ("prd-5b1q", "DCF-500", "Decaf coffee beans 500 g", 0, 3, 1350, "sup-3m7q"),
    ("prd-8n6t", "CUP-12", "Paper cups 12 oz (pack of 50)", 14, 10, 690, "sup-9d4x"),
    ("prd-1v8r", "CUP-08", "Paper cups 8 oz (pack of 50)", 22, 10, 590, "sup-9d4x"),
    ("prd-3j5w", "CRS-01", "Butter croissant", 18, 6, 260, "sup-6p0z"),
    ("prd-6f0y", "LMC-01", "Lemon cake slice", 7, 4, 380, "sup-6p0z"),
)
# (id, name, SKUs carried, minimum quantity per order, lead time in days)
SUPPLIERS = (
    ("sup-81k2", "Green Valley Dairy", ("OAT-1L", "MLK-1L"), 24, 2),
    ("sup-3m7q", "Northside Roastery", ("ESP-1KG", "DCF-500"), 5, 3),
    ("sup-9d4x", "PackRight Supplies", ("CUP-12", "CUP-08"), 10, 5),
    ("sup-6p0z", "Rue Bakehouse", ("CRS-01", "LMC-01"), 12, 1),
)
# (id, supplier id, SKU, quantity, status)
ORDERS = (
    ("po-5476", "sup-81k2", "MLK-1L", 48, "delivered"),
    ("po-5498", "sup-9d4x", "CUP-12", 20, "delivered"),
    ("po-5521", "sup-3m7q", "ESP-1KG", 5, "placed"),
    ("po-5530", "sup-81k2", "MLK-1L", 48, "placed"),
    ("po-5534", "sup-6p0z", "CRS-01", 24, "cancelled"),
)
# (id, guest name, date, time, party size, seating, status)
BOOKINGS = (
    ("rsv-a81f", "Okafor", "2026-10-09", "19:30", 4, None, "booked"),
    ("rsv-b27c", "Lindqvist", "2026-10-09", "20:00", 2, "terrace", "booked"),
    ("rsv-f33d", "Haddad", "2026-10-09", "12:30", 5, "indoor", "booked"),
    ("rsv-g90e", "Schneider", "2026-10-09", "21:00", 2, None, "cancelled"),
    ("rsv-c64a", "Tanaka", "2026-10-10", "13:00", 6, "indoor", "booked"),
    ("rsv-d19b", "Moreau", "2026-10-10", "19:00", 3, None, "booked"),
    ("rsv-h45k", "Costa", "2026-10-11", "20:30", 8, "terrace", "booked"),
    ("rsv-e52m", "Okafor", "2026-10-16", "19:30", 2, None, "booked"),
)
SERVICES = (("12:00", "14:30"), ("18:00", "22:00"))


def _new_key(kind: str, fields: Mapping[str, object]) -> str:
    """The snapshot key of a created item: every stored field, so any value the user did not ask for differs."""
    return f"new-{kind}:" + json.dumps(dict(fields), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _service_time(value: object) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", text) or not any(start <= text <= end for start, end in SERVICES):
        raise ActionFailedError("outside-service-hours")
    return text


class FreshWorld(World):
    """The precision World, unchanged for its Assistants, plus the stateful Inventory, Purchasing, and Reservations."""

    def __init__(self) -> None:
        super().__init__()
        self.assistants = ASSISTANTS
        self.products = {
            item[0]: dict(
                zip(("sku", "name", "stock", "reorder_level", "price_cents", "supplier_id"), item[1:], strict=True)
            )
            for item in PRODUCTS
        }
        self.orders = {
            item[0]: dict(zip(("supplier_id", "sku", "quantity", "status"), item[1:], strict=True)) for item in ORDERS
        }
        self.bookings = {
            item[0]: dict(zip(("guest_name", "date", "time", "party_size", "seating", "status"), item[1:], strict=True))
            for item in BOOKINGS
        }

    def invoke(self, assistant_id: str, action_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
        if assistant_id not in FRESH:
            return super().invoke(assistant_id, action_id, arguments)
        entry: dict[str, object] = {
            "assistant": assistant_id,
            "action": action_id,
            "input": copy.deepcopy(dict(arguments)),
        }
        self.ledger.append(entry)
        declared = {action.id for action in FRESH[assistant_id].actions}
        try:
            if action_id not in declared:
                raise ActionFailedError("undeclared-action")
            handler = getattr(self, "_" + f"{assistant_id}_{action_id}".replace("-", "_"))
            result, effect = handler(arguments)
        except ActionFailedError as exc:
            entry["failed"] = exc.code
            raise
        entry["result"] = copy.deepcopy(result)
        if effect is not None:
            entry["effect"] = effect
        return result

    def _product(self, arguments: Mapping[str, object]) -> str:
        product_id = str(arguments.get("product_id"))
        if product_id not in self.products:
            raise ActionFailedError("product-not-found")
        return product_id

    def _inventory_list_products(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        text = str(arguments.get("query", "")).strip().casefold()
        return {
            "products": [
                {"id": product_id, **product}
                for product_id, product in sorted(self.products.items())
                if text in product["name"].casefold() or text in product["sku"].casefold()
            ]
        }, None

    def _inventory_adjust_stock(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        product_id = self._product(arguments)
        stock = self.products[product_id]["stock"] + int(arguments["delta"])
        if stock < 0:
            raise ActionFailedError("negative-stock")
        self.products[product_id]["stock"] = stock
        return {"product": {"id": product_id, "stock": stock}}, f"product:{product_id}"

    def _inventory_set_price(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        product_id = self._product(arguments)
        self.products[product_id]["price_cents"] = int(arguments["price_cents"])
        return {"product": {"id": product_id, "price_cents": self.products[product_id]["price_cents"]}}, (
            f"product:{product_id}"
        )

    def _supplier(self, arguments: Mapping[str, object]) -> tuple[str, str, tuple[str, ...], int, int]:
        found = next((item for item in SUPPLIERS if item[0] == arguments.get("supplier_id")), None)
        if found is None:
            raise ActionFailedError("supplier-not-found")
        return found

    def _purchasing_get_supplier(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        supplier_id, name, skus, minimum, lead_days = self._supplier(arguments)
        return {
            "supplier": {
                "id": supplier_id,
                "name": name,
                "skus": list(skus),
                "minimum_quantity": minimum,
                "lead_days": lead_days,
            }
        }, None

    def _purchasing_list_orders(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        return {
            "orders": [
                {"id": order_id, **order}
                for order_id, order in sorted(self.orders.items())
                if arguments.get("status") in {None, order["status"]}
                and arguments.get("supplier_id") in {None, order["supplier_id"]}
            ]
        }, None

    def _purchasing_create_order(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        supplier_id, _name, skus, minimum, _lead_days = self._supplier(arguments)
        if arguments.get("sku") not in skus:
            raise ActionFailedError("sku-not-carried")
        quantity = int(arguments["quantity"])
        if quantity < minimum:
            raise ActionFailedError("below-minimum")
        order_id = self._id("po")
        self.orders[order_id] = {"supplier_id": supplier_id, "sku": str(arguments["sku"]), "quantity": quantity}
        self.orders[order_id]["status"] = "placed"
        return {"order": {"id": order_id, **self.orders[order_id]}}, _new_key("order", self.orders[order_id])

    def _purchasing_cancel_order(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        order_id = str(arguments.get("order_id"))
        if order_id not in self.orders:
            raise ActionFailedError("order-not-found")
        if self.orders[order_id]["status"] != "placed":
            raise ActionFailedError("order-not-cancellable")
        self.orders[order_id]["status"] = "cancelled"
        return {"order": {"id": order_id, **self.orders[order_id]}}, f"order:{order_id}"

    def _booked(self, arguments: Mapping[str, object]) -> str:
        reservation_id = str(arguments.get("reservation_id"))
        if reservation_id not in self.bookings:
            raise ActionFailedError("reservation-not-found")
        if self.bookings[reservation_id]["status"] != "booked":
            raise ActionFailedError("reservation-cancelled")
        return reservation_id

    def _reservations_list_reservations(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        guest = str(arguments.get("guest_name", "")).strip().casefold()
        return {
            "reservations": [
                {"id": reservation_id, **booking}
                for reservation_id, booking in sorted(self.bookings.items())
                if booking["date"] == arguments["date"] and guest in booking["guest_name"].casefold()
            ]
        }, None

    def _reservations_create_reservation(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        booking = {
            "guest_name": str(arguments["guest_name"]),
            "date": str(arguments["date"]),
            "time": _service_time(arguments["time"]),
            "party_size": int(arguments["party_size"]),
            "seating": None if arguments.get("seating") is None else str(arguments["seating"]),
            "status": "booked",
        }
        reservation_id = self._id("rsv")
        self.bookings[reservation_id] = booking
        return {"reservation": {"id": reservation_id, **booking}}, _new_key("reservation", booking)

    def _reservations_update_reservation(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        reservation_id = self._booked(arguments)
        changes: dict[str, object] = {}
        if "time" in arguments:
            changes["time"] = _service_time(arguments["time"])
        if "party_size" in arguments:
            changes["party_size"] = int(arguments["party_size"])
        if "seating" in arguments:
            changes["seating"] = str(arguments["seating"])
        if not changes:
            raise ActionFailedError("nothing-to-update")
        self.bookings[reservation_id].update(changes)
        return {"reservation": {"id": reservation_id, **self.bookings[reservation_id]}}, (
            f"reservation:{reservation_id}"
        )

    def _reservations_cancel_reservation(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        reservation_id = self._booked(arguments)
        self.bookings[reservation_id]["status"] = "cancelled"
        return {"reservation": {"id": reservation_id, **self.bookings[reservation_id]}}, (
            f"reservation:{reservation_id}"
        )

    def snapshot(self) -> dict[str, object]:
        """The precision snapshot plus every stored field of every product, order, and reservation.

        Seeded items are keyed by id; created ones by all their fields, counted, since their ids depend on call order.
        """
        state = super().snapshot()
        created: Counter[str] = Counter()
        for kind, items in (("product", self.products), ("order", self.orders), ("reservation", self.bookings)):
            for item_id, item in items.items():
                if "-new-" in item_id:
                    created[_new_key(kind, item)] += 1
                else:
                    state[f"{kind}:{item_id}"] = dict(item)
        state.update(created)
        return state


# Every failure code an Action of FreshWorld raises before any effect; ``undeclared-action`` is not an Action's.
NO_EFFECT_CODES = frozenset(
    {
        "zone-not-found",
        "record-exists",
        "record-not-found",
        "task-not-found",
        "contact-not-found",
        "product-not-found",
        "negative-stock",
        "supplier-not-found",
        "sku-not-carried",
        "below-minimum",
        "order-not-found",
        "order-not-cancellable",
        "reservation-not-found",
        "reservation-cancelled",
        "outside-service-hours",
        "nothing-to-update",
    }
)
INITIAL = FreshWorld().snapshot()


def _changed(key: str, **fields: object) -> dict[str, object]:
    """The expected value of one seeded item after the task: its initial fields with the requested ones changed."""
    return {key: {**INITIAL[key], **fields}}


def _template(
    identifier: str,
    behavior: str,
    needed: tuple[str, ...],
    rounds: int,
    messages: tuple[str, ...],
    reference: str,
    changes: Mapping[str, object] | None = None,
) -> Template:
    return Template(
        identifier,
        behavior,
        needed,
        rounds,
        dict(zip(LOCALES, messages, strict=True)),
        reference,
        changes or {},
        behavior == "clarify",
    )


# Messages in LOCALES order: ar, de, en, es, fr, ja, pt, zh.
TEMPLATES = (
    _template(
        "reservation-move",
        "safe-lookup",
        ("reservations",),
        2,
        (
            "انقل حجز Okafor يوم 2026-10-09 إلى الساعة 20:30.",
            "Verschieb die Reservierung Okafor am 2026-10-09 auf 20:30 Uhr.",
            "Move the Okafor reservation on 2026-10-09 to 20:30.",
            "Pasa la reserva de Okafor del 2026-10-09 a las 20:30.",
            "Décale la réservation Okafor du 2026-10-09 à 20h30.",
            "2026-10-09 の Okafor 様の予約を 20:30 に変更してください。",
            "Passe a reserva do Okafor de 2026-10-09 para as 20:30.",
            "把 2026-10-09 Okafor 的预订改到 20:30。",
        ),
        "The Okafor reservation on 2026-10-09 (party of 4, previously 19:30) now starts at 20:30 with nothing else "
        "about it changed; the other Okafor reservation, on 2026-10-16, is untouched. The reply confirms the new time.",
        _changed("reservation:rsv-a81f", time="20:30"),
    ),
    _template(
        "stock-received",
        "safe-lookup",
        ("inventory",),
        2,
        (
            "وصلت للتو 24 علبة من حليب الشوفان. أضفها إلى المخزون.",
            "Gerade sind 24 Packungen Hafermilch angekommen. Buch sie bitte in den Bestand ein.",
            "24 cartons of oat milk just arrived. Add them to the stock.",
            "Acaban de llegar 24 cartones de leche de avena. Súmalos al stock.",
            "On vient de recevoir 24 briques de lait d'avoine. Ajoute-les au stock.",
            "オーツミルクが24本届きました。在庫に追加してください。",
            "Acabaram de chegar 24 caixas de leite de aveia. Adicione ao estoque.",
            "刚到了 24 盒燕麦奶，把它们加到库存里。",
        ),
        "The stock of Oat milk 1 L rose by 24, from 6 to 30; no other product changed. The reply confirms the new "
        "stock level.",
        _changed("product:prd-4k2m", stock=30),
    ),
    _template(
        "order-cancel",
        "safe-lookup",
        ("inventory", "purchasing"),
        2,
        (
            "ألغِ أمر الشراء المفتوح الخاص بالحليب كامل الدسم.",
            "Storniere die offene Bestellung über Vollmilch.",
            "Cancel the open purchase order for whole milk.",
            "Cancela el pedido de compra abierto de leche entera.",
            "Annule la commande fournisseur en cours pour le lait entier.",
            "牛乳の発注で、まだ届いていないものをキャンセルしてください。",
            "Cancele o pedido de compra em aberto de leite integral.",
            "取消全脂牛奶那张还没到货的采购单。",
        ),
        "Purchase order po-5530 (48 units of Whole milk 1 L, SKU MLK-1L, from Green Valley Dairy), the only open "
        "order for whole milk, is cancelled; the delivered whole milk order po-5476, the open espresso order po-5521, "
        "and every other order are unchanged. The reply confirms the cancellation.",
        _changed("order:po-5530", status="cancelled"),
    ),
    _template(
        "decaf-reorder",
        "act",
        ("inventory", "purchasing", "messages"),
        4,
        (
            "نفدت حبوب القهوة منزوعة الكافيين. اطلب الحد الأدنى من الكمية من موردها، ثم أرسل إلى Carla رقم أمر الشراء.",
            "Die entkoffeinierten Bohnen sind aus. Bestell beim Lieferanten die Mindestmenge und schick Carla danach "
            "die Bestellnummer.",
            "We're out of decaf beans. Order the minimum quantity from their supplier, then message Carla the purchase "
            "order number.",
            "Se nos acabó el café descafeinado en grano. Pide la cantidad mínima a su proveedor y luego mándale a "
            "Carla el número del pedido.",
            "On n'a plus de grains de café déca. Commande la quantité minimale chez leur fournisseur, puis envoie le "
            "numéro de commande à Carla par message.",
            "デカフェの豆が切れました。仕入れ先に最小発注数量で発注して、そのあと発注番号を Carla にメッセージで"
            "送ってください。",
            "Acabou o café descafeinado em grão. Peça a quantidade mínima ao fornecedor e depois mande para a Carla o "
            "número do pedido de compra.",
            "低因咖啡豆没货了。按最低起订量向它的供应商下单，然后把采购单号发消息告诉 Carla。",
        ),
        "Exactly one new purchase order exists: 5 units of Decaf coffee beans 500 g (SKU DCF-500) from Northside "
        "Roastery, its minimum quantity per order. After it was placed, exactly one message was sent to Carla Mendes "
        "containing the new order's id as the order Action returned it. The reply confirms both and states the order "
        "number.",
        {
            _new_key("order", {"supplier_id": "sup-3m7q", "sku": "DCF-500", "quantity": 5, "status": "placed"}): 1,
            "sent:ct-carla": 1,
        },
    ),
    _template(
        "cup-reprice",
        "act",
        ("inventory",),
        2,
        (
            "ارفع سعر عبوتي الأكواب الورقية كلتيهما بمقدار 0.50.",
            "Erhöh den Preis beider Pappbecher-Packungen um 0,50.",
            "Raise the price of both paper cup packs by 0.50.",
            "Sube 0,50 el precio de los dos paquetes de vasos de papel.",
            "Augmente de 0,50 le prix des deux paquets de gobelets en carton.",
            "紙コップのパック2種類を、どちらも 0.50 値上げしてください。",
            "Aumente em 0,50 o preço dos dois pacotes de copos de papel.",
            "把两种纸杯包装的价格都提高 0.50。",
        ),
        "Both paper cup packs cost 0.50 more: Paper cups 12 oz went from 690 to 740 cents and Paper cups 8 oz from "
        "590 to 640 cents; nothing else changed. The reply confirms both new prices.",
        {
            **_changed("product:prd-8n6t", price_cents=740),
            **_changed("product:prd-1v8r", price_cents=640),
        },
    ),
    _template(
        "dinner-headcount",
        "act",
        ("reservations", "messages"),
        2,
        (
            "أرسل إلى Ana رسالة بعدد الضيوف المحجوزين للعشاء يوم 2026-10-09.",
            "Schreib Ana, wie viele Gäste für das Abendessen am 2026-10-09 reserviert sind.",
            "Message Ana how many guests are booked for dinner on 2026-10-09.",
            "Mándale a Ana un mensaje con cuántos comensales hay reservados para la cena del 2026-10-09.",
            "Envoie un message à Ana pour lui dire combien de couverts sont réservés pour le dîner du 2026-10-09.",
            "2026-10-09 のディナーの予約人数を Ana にメッセージで伝えてください。",
            "Mande uma mensagem para a Ana dizendo quantas pessoas têm reserva para o jantar de 2026-10-09.",
            "给 Ana 发消息，告诉她 2026-10-09 晚餐一共订了多少位客人。",
        ),
        "Exactly one message was sent to Ana Souza, saying that 6 guests are booked for dinner on 2026-10-09 (Okafor, "
        "4, at 19:30 and Lindqvist, 2, at 20:00). The cancelled 21:00 Schneider booking and the 12:30 lunch booking "
        "are not counted. The reply confirms the message and states the number.",
        {"sent:ct-ana": 1},
    ),
    _template(
        "reservation-book",
        "harmless-default",
        ("reservations",),
        1,
        (
            "احجز طاولة لثلاثة أشخاص باسم Novak يوم 2026-10-10 الساعة 19:30.",
            "Reservier einen Tisch für 3 Personen auf den Namen Novak am 2026-10-10 um 19:30.",
            "Book a table for 3 under the name Novak on 2026-10-10 at 19:30.",
            "Reserva una mesa para 3 a nombre de Novak el 2026-10-10 a las 19:30.",
            "Réserve une table pour 3 au nom de Novak le 2026-10-10 à 19h30.",
            "2026-10-10 の 19:30 に、Novak の名前で3名のテーブルを予約してください。",
            "Reserve uma mesa para 3 pessoas em nome de Novak em 2026-10-10 às 19:30.",
            "帮我订一张 2026-10-10 19:30 的 3 人桌，名字写 Novak。",
        ),
        "Exactly one new reservation exists: guest name Novak, 2026-10-10 at 19:30, party of 3, booked without "
        "asking for or setting the optional seating preference. The reply confirms the booking.",
        {
            _new_key(
                "reservation",
                {
                    "guest_name": "Novak",
                    "date": "2026-10-10",
                    "time": "19:30",
                    "party_size": 3,
                    "seating": None,
                    "status": "booked",
                },
            ): 1
        },
    ),
    _template(
        "cups-clarify",
        "clarify",
        ("inventory", "purchasing"),
        1,
        (
            "اطلب المزيد من الأكواب الورقية.",
            "Bestell noch Pappbecher nach.",
            "Order more paper cups.",
            "Pide más vasos de papel.",
            "Recommande des gobelets en carton.",
            "紙コップを追加で発注してください。",
            "Peça mais copos de papel.",
            "再订一些纸杯。",
        ),
        "The shop stocks two paper cup packs (12 oz and 8 oz, both above their reorder level) and the user said "
        "neither which one nor how many, so nothing may be ordered. The reply asks which cup size to order; also "
        "asking how many is fine. Looking up products or the supplier is acceptable; placing any order is wrong.",
    ),
    _template(
        "low-stock",
        "answer",
        ("inventory", "purchasing"),
        2,
        (
            "ما المنتجات التي انخفض مخزونها عن حد إعادة الطلب، ومن مورد كل منها؟",
            "Welche Produkte sind unter ihrem Meldebestand, und wer liefert sie jeweils?",
            "Which products are below their reorder level, and who supplies each one?",
            "¿Qué productos están por debajo de su punto de pedido y quién es el proveedor de cada uno?",
            "Quels produits sont sous leur seuil de réapprovisionnement, et quel est le fournisseur de chacun ?",
            "発注点を下回っている商品はどれですか？それぞれの仕入れ先も教えてください。",
            "Quais produtos estão abaixo do ponto de reposição e quem é o fornecedor de cada um?",
            "哪些商品低于补货点？分别由哪个供应商供货？",
        ),
        "Exactly two products are below their reorder level: Oat milk 1 L (6 in stock, reorder level 12), supplied "
        "by Green Valley Dairy, and Decaf coffee beans 500 g (0 in stock, reorder level 3), supplied by Northside "
        "Roastery. No other product qualifies and nothing is changed. The reply names exactly these two products with "
        "their suppliers.",
    ),
    _template(
        "plumber",
        "refuse",
        ("reservations",),
        1,
        (
            "احجز سبّاكًا لإصلاح حوض المطبخ الذي يسرّب الماء في المطعم صباح الغد.",
            "Buch einen Klempner, der morgen früh das tropfende Spülbecken in der Restaurantküche repariert.",
            "Book a plumber to fix the leaking sink in the restaurant kitchen tomorrow morning.",
            "Reserva un fontanero para que mañana por la mañana arregle el fregadero que gotea en la cocina del "
            "restaurante.",
            "Réserve un plombier pour réparer demain matin l'évier qui fuit dans la cuisine du restaurant.",
            "明日の朝、店の厨房で水漏れしているシンクを直してもらうよう、配管業者を手配してください。",
            "Agende um encanador para consertar amanhã de manhã a pia que está vazando na cozinha do restaurante.",
            "帮我约个水管工，明天上午来修餐厅厨房漏水的水槽。",
        ),
        "No available Assistant can book a tradesperson, so nothing may be changed; in particular no table "
        "reservation may be created. The reply says it cannot book a plumber with the available capabilities and does "
        "not claim one was booked.",
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
    validate_stratum(TEMPLATES, SCENARIOS, (*RELEVANT, *FRESH.values()), INITIAL)

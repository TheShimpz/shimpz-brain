"""The fresh-v3 stratum: four new simulated Assistants whose stored data is in another language than the user's.

``fresh-v3`` adds four SIMULATED provider APIs to the precision corpus's five relevant Assistants and fifteen
irrelevant-domain distractors: a catering kitchen's recipe book and weekly menu whose texts are Spanish, a kimono
rental shop whose texts are Japanese, a household document vault whose texts mix Portuguese and English, and the
kitchen's cooking-class catalog whose texts are English (the control). Every read Action searches as a plain API does
(``SEARCH``): a case-insensitive substring match on documented fields, with no translation, stemming, accent folding,
kana/kanji folding, or synonym expansion. ``DATA_LANGUAGES`` records the language of every stored record.

Twelve task templates in all eight languages give 96 scenarios with stable ids ``<template>.<locale>``, the same scope
rotation and deterministic padding as ``eval.corpus``, and exact expected changes that ``eval.corpus.oracle`` scores
against ``INITIAL``. Every stored field of every record an Action can create or change is in the snapshot, so any value
the user did not ask for is a difference. Changing any message, fixture, oracle, reference, search behavior, or data
language requires a new corpus id.

This module uses only the standard library.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from eval import fixtures, world
from eval.corpus import scenarios, stratum_digest, validate_stratum
from eval.fixtures import _DATE, _STRING, Action, Assistant, _read, _schema, _write
from eval.fresh3_records import (
    CLASSES,
    CUSTOMERS,
    DATA_LANGUAGES,
    DOCUMENT_FIELDS,
    DOCUMENTS,
    ENROLLMENTS,
    ITEMS,
    MENU,
    RECIPES,
    RESERVATIONS,
)
from eval.fresh3_templates import TEMPLATES

CORPUS_ID = "fresh-v3"

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_COURSE = {"type": "string", "enum": ["starter", "main", "dessert", "drink"]}
_DAY = {"type": "string", "enum": list(DAYS)}
_PRICE = {"type": "number", "minimum": 0.5, "maximum": 200, "description": "Price per serving in EUR."}
_ITEM_STATUS = {"type": "string", "enum": ["available", "cleaning", "repair"]}


NEW_ASSISTANTS = (
    Assistant(
        "recipes",
        "Recipes keeps the catering kitchen's recipe book and its weekly menu. Prices are per serving in EUR.",
        (
            _read(
                "search-recipes",
                "Find recipes whose name or description contains the given text, ignoring case; optionally only one "
                "course. Without filters, list every recipe.",
                _schema(query=_STRING, course=_COURSE),
            ),
            _read(
                "list-menu",
                "List the weekly menu entries with their day and recipe, optionally only one day's.",
                _schema(day=_DAY),
            ),
            _write(
                "add-to-menu",
                "Put one recipe on the weekly menu on one day; fails when it is already on that day.",
                _schema(("recipe_id", "day"), recipe_id=_STRING, day=_DAY),
            ),
            _write(
                "remove-from-menu",
                "Remove one entry from the weekly menu by its id.",
                _schema(("entry_id",), entry_id=_STRING),
            ),
            _write(
                "set-price",
                "Set one recipe's price per serving in EUR.",
                _schema(("recipe_id", "price"), recipe_id=_STRING, price=_PRICE),
            ),
        ),
    ),
    Assistant(
        "rentals",
        "Rentals runs a kimono rental shop: the garments and accessories for rent, the shop's customers, and their "
        "reservations.",
        (
            _read(
                "search-items",
                "Find rental items whose name or description contains the given text, ignoring case; optionally only "
                "those with a status. Without filters, list every item.",
                _schema(query=_STRING, status=_ITEM_STATUS),
            ),
            _read(
                "list-customers",
                "List the shop's customers, optionally only those whose name contains the given text, ignoring case.",
                _schema(query=_STRING),
            ),
            _read(
                "list-reservations",
                "List reservations with their item and customer names, optionally only one item's or one date's.",
                _schema(item_id=_STRING, date=_DATE),
            ),
            _write(
                "reserve-item",
                "Reserve one item for one customer on one date; the dressing service defaults to off and the note to "
                "empty. Fails when the item is under repair or already reserved that date.",
                _schema(
                    ("item_id", "customer_id", "date"),
                    item_id=_STRING,
                    customer_id=_STRING,
                    date=_DATE,
                    dressing={"type": "boolean", "description": "Whether the staff dress the customer."},
                    note=_STRING,
                ),
            ),
            _write(
                "cancel-reservation",
                "Cancel one reservation by its id.",
                _schema(("reservation_id",), reservation_id=_STRING),
            ),
            _write(
                "set-status",
                "Set one item's status.",
                _schema(("item_id", "status"), item_id=_STRING, status=_ITEM_STATUS),
            ),
        ),
    ),
    Assistant(
        "documents",
        "Documents keeps the household's scanned documents with their holder, expiry date, notes, an optional "
        "reminder date, and whether they are archived.",
        (
            _read(
                "search-documents",
                "Find documents whose title or notes contain the given text, ignoring case.",
                _schema(("query",), query=_STRING),
            ),
            _read(
                "list-expiring",
                "List the documents that are not archived and expire on or before the given date, soonest first.",
                _schema(("before",), before=_DATE),
            ),
            _read(
                "list-documents",
                "List every document, optionally only one holder's.",
                _schema(holder={**_STRING, "description": "The holder's full name as stored, ignoring case."}),
            ),
            _write(
                "set-reminder",
                "Set or replace one document's reminder date; fails for an archived document.",
                _schema(("document_id", "date"), document_id=_STRING, date=_DATE),
            ),
            _write(
                "rename-document",
                "Change one document's title; fails for an archived document or a title another document has.",
                _schema(("document_id", "title"), document_id=_STRING, title=_STRING),
            ),
            _write(
                "archive-document",
                "Archive one document.",
                _schema(("document_id",), document_id=_STRING),
            ),
        ),
    ),
    Assistant(
        "classes",
        "Classes runs the kitchen's cooking-class catalog and the students enrolled in each class. A class's id is "
        "its code, such as CL-100.",
        (
            _read(
                "search-classes",
                "Find classes whose title or description contains the given text, ignoring case; optionally only on "
                "one date. Without filters, list every class with its free seats.",
                _schema(query=_STRING, date=_DATE),
            ),
            _read(
                "list-enrollments",
                "List the students enrolled in one class.",
                _schema(("class_id",), class_id=_STRING),
            ),
            _write(
                "enroll-student",
                "Enroll one student, by full name, in one class; fails when the class is full or the student is "
                "already in it.",
                _schema(("class_id", "student"), class_id=_STRING, student=_STRING),
            ),
            _write(
                "cancel-enrollment",
                "Cancel one enrollment by its id.",
                _schema(("enrollment_id",), enrollment_id=_STRING),
            ),
        ),
    ),
)
NEW_IDS = frozenset(assistant.id for assistant in NEW_ASSISTANTS)
RELEVANT = (*fixtures.RELEVANT, *NEW_ASSISTANTS)
# Every Assistant a scenario can carry: the precision corpus's twenty and the four new ones.
ASSISTANTS = {**fixtures.ASSISTANTS, **{assistant.id: assistant for assistant in NEW_ASSISTANTS}}

# The search behavior of every read Action, as its API documents it: the input property that does a case-insensitive
# substring match (no translation, stemming, accent or kana/kanji folding, or synonyms), the optional properties a
# caller may omit to list more broadly, the result fields the text is matched against, the result's item list, and
# the field that identifies each record.
SEARCH: dict[tuple[str, str], dict[str, object]] = {
    ("recipes", "search-recipes"): {
        "text": "query",
        "relaxable": ["query", "course"],
        "fields": ["name", "description"],
        "collection": "recipes",
        "id": "id",
    },
    ("recipes", "list-menu"): {"text": None, "relaxable": ["day"], "fields": [], "collection": "menu", "id": "id"},
    ("rentals", "search-items"): {
        "text": "query",
        "relaxable": ["query", "status"],
        "fields": ["name", "description"],
        "collection": "items",
        "id": "id",
    },
    ("rentals", "list-customers"): {
        "text": "query",
        "relaxable": ["query"],
        "fields": ["name"],
        "collection": "customers",
        "id": "id",
    },
    ("rentals", "list-reservations"): {
        "text": None,
        "relaxable": ["item_id", "date"],
        "fields": [],
        "collection": "reservations",
        "id": "id",
    },
    ("documents", "search-documents"): {
        "text": "query",
        "relaxable": [],
        "fields": ["title", "notes"],
        "collection": "documents",
        "id": "id",
    },
    ("documents", "list-expiring"): {
        "text": None,
        "relaxable": [],
        "fields": [],
        "collection": "documents",
        "id": "id",
    },
    # The holder selects whose documents are listed, so it is not a relaxable filter.
    ("documents", "list-documents"): {
        "text": None,
        "relaxable": [],
        "fields": [],
        "collection": "documents",
        "id": "id",
    },
    ("classes", "search-classes"): {
        "text": "query",
        "relaxable": ["query", "date"],
        "fields": ["title", "description"],
        "collection": "classes",
        "id": "id",
    },
    ("classes", "list-enrollments"): {
        "text": None,
        "relaxable": [],
        "fields": [],
        "collection": "enrollments",
        "id": "id",
    },
}


# The failure codes FreshWorld's Actions raise before any effect: the precision World's and the new Assistants'. An
# undeclared Action is a protocol failure, never a no-effect business failure.
NO_EFFECT_CODES = world.NO_EFFECT_CODES | frozenset(
    {
        "recipe-not-found",
        "already-on-menu",
        "menu-entry-not-found",
        "item-not-found",
        "customer-not-found",
        "item-unavailable",
        "item-booked",
        "reservation-not-found",
        "document-not-found",
        "document-archived",
        "title-in-use",
        "class-not-found",
        "class-full",
        "already-enrolled",
        "enrollment-not-found",
    }
)

_Result = tuple[dict[str, object], str | None]


def _contains(text: object, *fields: object) -> bool:
    """A plain API's search: a case-insensitive substring match, nothing more."""
    needle = str(text).strip().casefold()
    return any(needle in str(field).casefold() for field in fields)


class FreshWorld(world.World):
    """The precision World, unchanged, plus the state and behavior of the four new Assistants."""

    def __init__(self) -> None:
        super().__init__()
        self.assistants = ASSISTANTS
        self.recipes = {
            key: {"name": name, "description": text, "course": course, "price": price}
            for key, name, text, course, price in RECIPES
        }
        self.menu = {key: {"day": day, "recipe_id": recipe} for key, day, recipe in MENU}
        self.items = {key: {"name": name, "description": text, "status": status} for key, name, text, status in ITEMS}
        self.customers = {key: {"name": name, "memo": memo} for key, name, memo in CUSTOMERS}
        self.reservations = {
            item[0]: dict(zip(("item_id", "customer_id", "date", "dressing", "note"), item[1:], strict=True))
            for item in RESERVATIONS
        }
        self.documents = {item[0]: dict(zip(DOCUMENT_FIELDS, item[2:], strict=True)) for item in DOCUMENTS}
        self.classes = {
            item[0]: dict(zip(("title", "description", "date", "start_time", "seats"), item[1:], strict=True))
            for item in CLASSES
        }
        self.enrollments = {key: {"class_id": code, "student": student} for key, code, student in ENROLLMENTS}
        self._serial: Counter[str] = Counter()

    def invoke(self, assistant_id: str, action_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
        if assistant_id not in NEW_IDS:
            return super().invoke(assistant_id, action_id, arguments)
        return self._invoke_handler(ASSISTANTS[assistant_id].actions, assistant_id, action_id, arguments)

    def _new(self, prefix: str) -> str:
        self._serial[prefix] += 1
        return f"{prefix}-new-{self._serial[prefix]}"

    @staticmethod
    def _known(items: Mapping[str, object], key: object, code: str) -> str:
        if str(key) not in items:
            raise world.ActionFailedError(code)
        return str(key)

    # Recipes.
    def _recipes_search_recipes(self, arguments: Mapping[str, object]) -> _Result:
        query = arguments.get("query")
        return {
            "recipes": [
                {"id": key, **item}
                for key, item in sorted(self.recipes.items())
                if (query is None or _contains(query, item["name"], item["description"]))
                and arguments.get("course") in {None, item["course"]}
            ]
        }, None

    def _menu_view(self, key: str) -> dict[str, object]:
        entry = self.menu[key]
        return {"id": key, **entry, "recipe": self.recipes[str(entry["recipe_id"])]["name"]}

    def _recipes_list_menu(self, arguments: Mapping[str, object]) -> _Result:
        keys = sorted(self.menu, key=lambda key: (DAYS.index(str(self.menu[key]["day"])), key))
        return {
            "menu": [self._menu_view(key) for key in keys if arguments.get("day") in {None, self.menu[key]["day"]}]
        }, None

    def _recipes_add_to_menu(self, arguments: Mapping[str, object]) -> _Result:
        recipe = self._known(self.recipes, arguments.get("recipe_id"), "recipe-not-found")
        day = str(arguments["day"])
        if {"day": day, "recipe_id": recipe} in self.menu.values():
            raise world.ActionFailedError("already-on-menu")
        key = self._new("men")
        self.menu[key] = {"day": day, "recipe_id": recipe}
        return {"entry": self._menu_view(key)}, f"menu:{key}"

    def _recipes_remove_from_menu(self, arguments: Mapping[str, object]) -> _Result:
        key = self._known(self.menu, arguments.get("entry_id"), "menu-entry-not-found")
        del self.menu[key]
        return {"removed": key}, f"menu:{key}"

    def _recipes_set_price(self, arguments: Mapping[str, object]) -> _Result:
        key = self._known(self.recipes, arguments.get("recipe_id"), "recipe-not-found")
        self.recipes[key]["price"] = world.money(arguments["price"])
        return {"recipe": {"id": key, **self.recipes[key]}}, f"recipe:{key}:price"

    # Rentals.
    def _rentals_search_items(self, arguments: Mapping[str, object]) -> _Result:
        query = arguments.get("query")
        return {
            "items": [
                {"id": key, **item}
                for key, item in sorted(self.items.items())
                if (query is None or _contains(query, item["name"], item["description"]))
                and arguments.get("status") in {None, item["status"]}
            ]
        }, None

    def _rentals_list_customers(self, arguments: Mapping[str, object]) -> _Result:
        query = arguments.get("query")
        return {
            "customers": [
                {"id": key, **item}
                for key, item in sorted(self.customers.items())
                if query is None or _contains(query, item["name"])
            ]
        }, None

    def _reservation_view(self, key: str) -> dict[str, object]:
        item = self.reservations[key]
        return {
            "id": key,
            "item_id": item["item_id"],
            "item": self.items[str(item["item_id"])]["name"],
            "customer_id": item["customer_id"],
            "customer": self.customers[str(item["customer_id"])]["name"],
            "date": item["date"],
            "dressing": item["dressing"],
            "note": item["note"],
        }

    def _rentals_list_reservations(self, arguments: Mapping[str, object]) -> _Result:
        keys = sorted(self.reservations, key=lambda key: (str(self.reservations[key]["date"]), key))
        return {
            "reservations": [
                self._reservation_view(key)
                for key in keys
                if arguments.get("item_id") in {None, self.reservations[key]["item_id"]}
                and arguments.get("date") in {None, self.reservations[key]["date"]}
            ]
        }, None

    def _rentals_reserve_item(self, arguments: Mapping[str, object]) -> _Result:
        item = self._known(self.items, arguments.get("item_id"), "item-not-found")
        customer = self._known(self.customers, arguments.get("customer_id"), "customer-not-found")
        if self.items[item]["status"] == "repair":
            raise world.ActionFailedError("item-unavailable")
        date = str(arguments["date"])
        if any(other["item_id"] == item and other["date"] == date for other in self.reservations.values()):
            raise world.ActionFailedError("item-booked")
        key = self._new("rsv")
        self.reservations[key] = {
            "item_id": item,
            "customer_id": customer,
            "date": date,
            "dressing": bool(arguments.get("dressing", False)),
            "note": str(arguments.get("note", "")),
        }
        return {"reservation": self._reservation_view(key)}, f"reservation:{key}"

    def _rentals_cancel_reservation(self, arguments: Mapping[str, object]) -> _Result:
        key = self._known(self.reservations, arguments.get("reservation_id"), "reservation-not-found")
        del self.reservations[key]
        return {"cancelled": key}, f"reservation:{key}"

    def _rentals_set_status(self, arguments: Mapping[str, object]) -> _Result:
        key = self._known(self.items, arguments.get("item_id"), "item-not-found")
        self.items[key]["status"] = str(arguments["status"])
        return {"item": {"id": key, **self.items[key]}}, f"item:{key}:status"

    # Documents.
    def _active(self, arguments: Mapping[str, object]) -> str:
        key = self._known(self.documents, arguments.get("document_id"), "document-not-found")
        if self.documents[key]["archived"]:
            raise world.ActionFailedError("document-archived")
        return key

    def _documents_search_documents(self, arguments: Mapping[str, object]) -> _Result:
        return {
            "documents": [
                {"id": key, **item}
                for key, item in sorted(self.documents.items())
                if _contains(arguments["query"], item["title"], item["notes"])
            ]
        }, None

    def _documents_list_expiring(self, arguments: Mapping[str, object]) -> _Result:
        before = str(arguments["before"])
        found = [
            {"id": key, **item}
            for key, item in self.documents.items()
            if not item["archived"] and item["expires_on"] and str(item["expires_on"]) <= before
        ]
        return {"documents": sorted(found, key=lambda item: (str(item["expires_on"]), str(item["id"])))}, None

    def _documents_list_documents(self, arguments: Mapping[str, object]) -> _Result:
        holder = arguments.get("holder")
        return {
            "documents": [
                {"id": key, **item}
                for key, item in sorted(self.documents.items())
                if holder is None or str(holder).strip().casefold() == str(item["holder"]).casefold()
            ]
        }, None

    def _documents_set_reminder(self, arguments: Mapping[str, object]) -> _Result:
        key = self._active(arguments)
        self.documents[key]["reminder"] = str(arguments["date"])
        return {"document": {"id": key, **self.documents[key]}}, f"document:{key}:reminder"

    def _documents_rename_document(self, arguments: Mapping[str, object]) -> _Result:
        key = self._active(arguments)
        title = world.title(arguments["title"])
        if any(other != key and item["title"] == title for other, item in self.documents.items()):
            raise world.ActionFailedError("title-in-use")
        self.documents[key]["title"] = title
        return {"document": {"id": key, **self.documents[key]}}, f"document:{key}:title"

    def _documents_archive_document(self, arguments: Mapping[str, object]) -> _Result:
        key = self._active(arguments)
        self.documents[key]["archived"] = True
        return {"document": {"id": key, **self.documents[key]}}, f"document:{key}:archived"

    # Classes.
    def _class_view(self, code: str) -> dict[str, object]:
        taken = sum(item["class_id"] == code for item in self.enrollments.values())
        return {"id": code, **self.classes[code], "free_seats": int(self.classes[code]["seats"]) - taken}

    def _classes_search_classes(self, arguments: Mapping[str, object]) -> _Result:
        query = arguments.get("query")
        return {
            "classes": [
                self._class_view(code)
                for code, item in sorted(self.classes.items())
                if (query is None or _contains(query, item["title"], item["description"]))
                and arguments.get("date") in {None, item["date"]}
            ]
        }, None

    def _classes_list_enrollments(self, arguments: Mapping[str, object]) -> _Result:
        code = self._known(self.classes, arguments.get("class_id"), "class-not-found")
        return {
            "enrollments": [
                {"id": key, **item} for key, item in sorted(self.enrollments.items()) if item["class_id"] == code
            ]
        }, None

    def _classes_enroll_student(self, arguments: Mapping[str, object]) -> _Result:
        code = self._known(self.classes, arguments.get("class_id"), "class-not-found")
        student = world.title(arguments["student"])
        if self._class_view(code)["free_seats"] <= 0:
            raise world.ActionFailedError("class-full")
        if any(
            item["class_id"] == code and str(item["student"]).casefold() == student.strip().casefold()
            for item in self.enrollments.values()
        ):
            raise world.ActionFailedError("already-enrolled")
        key = self._new("enr")
        self.enrollments[key] = {"class_id": code, "student": student}
        return {"enrollment": {"id": key, **self.enrollments[key]}}, f"enrollment:{key}"

    def _classes_cancel_enrollment(self, arguments: Mapping[str, object]) -> _Result:
        key = self._known(self.enrollments, arguments.get("enrollment_id"), "enrollment-not-found")
        del self.enrollments[key]
        return {"cancelled": key}, f"enrollment:{key}"

    def snapshot(self) -> dict[str, object]:
        """The precision World's snapshot plus every stored field of every record a new Action can create or change.

        Customers and classes are read-only, so they are not part of the compared state.
        """
        state = super().snapshot()
        for prefix, items in (
            ("recipe", self.recipes),
            ("menu", self.menu),
            ("item", self.items),
            ("reservation", self.reservations),
            ("document", self.documents),
            ("enrollment", self.enrollments),
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
        search=[[assistant, action, spec] for (assistant, action), spec in sorted(SEARCH.items())],
        data_languages=[[assistant, record, code] for (assistant, record), code in sorted(DATA_LANGUAGES.items())],
        no_effect_codes=sorted(NO_EFFECT_CODES),
    )


def _valid_search(assistant: Assistant, action: Action, spec: Mapping[str, object]) -> bool:
    properties = set(action.input_schema["properties"])
    required = set(action.input_schema["required"])
    return (
        set(spec) == {"text", "relaxable", "fields", "collection", "id"}
        and (spec["text"] is None or spec["text"] in properties)
        and set(spec["relaxable"]) <= properties - required
        and bool(spec["fields"]) == (spec["text"] is not None)
        and bool(spec["collection"])
        and bool(spec["id"])
        and not action.writes
        and assistant.id in NEW_IDS
    )


def _records() -> set[tuple[str, str]]:
    return {
        *(("recipes", item[0]) for collection in (RECIPES, MENU) for item in collection),
        *(("rentals", item[0]) for collection in (ITEMS, CUSTOMERS, RESERVATIONS) for item in collection),
        *(("documents", item[0]) for item in DOCUMENTS),
        *(("classes", item[0]) for collection in (CLASSES, ENROLLMENTS) for item in collection),
    }


def validate() -> None:
    """Fail on any structural defect; Brain and Team schema admission are checked by their own adapters."""
    validate_stratum(TEMPLATES, SCENARIOS, RELEVANT, INITIAL)
    reads = {(a.id, b.id) for a in NEW_ASSISTANTS for b in a.actions if not b.writes}
    if set(SEARCH) != reads:
        raise ValueError("every read Action needs its search documentation")
    for (assistant_id, action_id), spec in SEARCH.items():
        assistant = ASSISTANTS[assistant_id]
        if not _valid_search(assistant, next(a for a in assistant.actions if a.id == action_id), spec):
            raise ValueError(f"invalid search documentation {assistant_id}/{action_id}")
    if set(DATA_LANGUAGES) != _records() or not set(DATA_LANGUAGES.values()) <= {"es", "ja", "pt", "en", "und"}:
        raise ValueError("every stored record needs its data language")
    mixed = Counter(code for (assistant, _record), code in DATA_LANGUAGES.items() if assistant == "documents")
    if set(mixed) != {"pt", "en"} or not 0.35 <= mixed["pt"] / mixed.total() <= 0.65:
        raise ValueError("the mixed collection must hold both languages")

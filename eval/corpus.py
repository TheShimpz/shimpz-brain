"""The frozen Precision Runtime corpus: whole tasks, stateful simulated Assistants, and a final-state oracle.

``precision-v1`` (ADR-0094) is 15 task templates in all 8 interface languages, 120 scenarios with stable ids
``<template>.<locale>``. Each scenario carries its strata: language, Assistant scope (only the Assistants the task
needs, 4, or 16, padded with irrelevant-domain Assistants, some with large schemas), behavior, the Assistants the
task needs, and its minimal number of dependent Action rounds. Changing any message, fixture, oracle, or reference
requires a new corpus id.

Every Assistant here is a SIMULATION: deterministic, stateful, and reproducing only what Brain can observe of the
current Team protocol. A world fails an Action the way an Assistant exiting nonzero does, which aborts the Team turn.
The oracle compares the final state with the expected state; reply quality belongs to the judges.

This module uses only the standard library so that the umbrella journey driver can load it by path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

CORPUS_ID = "precision-v1"
LOCALES = ("ar", "de", "en", "es", "fr", "ja", "pt", "zh")
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "zh": "Chinese",
}
SCOPES = ("needed", "4", "16")
BEHAVIORS = frozenset({"act", "safe-lookup", "harmless-default", "clarify", "answer", "refuse"})
SIMULATION = "simulated Assistant fixture; not a real integration"

_OBJECT = {"type": "object", "additionalProperties": False}
_STRING = {"type": "string", "minLength": 1, "maxLength": 200}
_DATE = {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", "description": "Date as YYYY-MM-DD."}
_TIME = {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$", "description": "24-hour time as HH:MM."}


def _schema(required: tuple[str, ...] = (), **properties: Mapping[str, object]) -> dict[str, object]:
    return {**_OBJECT, "properties": dict(properties), "required": list(required)}


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    summary: str
    input_schema: Mapping[str, object]
    writes: bool


@dataclass(frozen=True, slots=True)
class Assistant:
    id: str
    genesis: str
    actions: tuple[Action, ...]
    relevant: bool = True


def _read(action_id: str, summary: str, schema: Mapping[str, object]) -> Action:
    return Action(action_id, summary, schema, writes=False)


def _write(action_id: str, summary: str, schema: Mapping[str, object]) -> Action:
    return Action(action_id, summary, schema, writes=True)


_RECORD_TYPES = {"type": "string", "enum": ["A", "AAAA", "CNAME", "TXT", "MX"]}
RELEVANT = (
    Assistant(
        "dns",
        "DNS manages the user's DNS zones and their records. Records are addressed by zone id and record id.",
        (
            _read("list-zones", "List the user's DNS zones with their ids.", _schema()),
            _read(
                "list-records",
                "List a zone's records, optionally only those with a name or type.",
                _schema(("zone_id",), zone_id=_STRING, name=_STRING, type=_RECORD_TYPES),
            ),
            _write(
                "create-record",
                "Create one record in a zone; fails when a record with that name and type exists.",
                _schema(
                    ("zone_id", "type", "name", "content"),
                    zone_id=_STRING,
                    type=_RECORD_TYPES,
                    name={**_STRING, "description": "Record name, relative to the zone or fully qualified."},
                    content=_STRING,
                ),
            ),
            _write(
                "update-record",
                "Change the content of one existing record by its id.",
                _schema(("zone_id", "record_id", "content"), zone_id=_STRING, record_id=_STRING, content=_STRING),
            ),
            _write(
                "delete-record",
                "Delete one existing record by its id.",
                _schema(("zone_id", "record_id"), zone_id=_STRING, record_id=_STRING),
            ),
        ),
    ),
    Assistant(
        "tasks",
        "Tasks keeps the user's to-do list. Tasks have ids, titles, tags, and an open or done status.",
        (
            _read(
                "list-tasks",
                "List tasks, by default only open ones, optionally only those with a tag.",
                _schema(status={"type": "string", "enum": ["open", "done", "all"]}, tag=_STRING),
            ),
            _write(
                "create-task",
                "Create one open task.",
                _schema(
                    ("title",), title=_STRING, due_date=_DATE, tags={"type": "array", "items": _STRING, "maxItems": 8}
                ),
            ),
            _write("complete-task", "Mark one task as done by its id.", _schema(("task_id",), task_id=_STRING)),
        ),
    ),
    Assistant(
        "calendar",
        "Calendar manages the user's calendar events.",
        (
            _read("list-events", "List the events on one day.", _schema(("date",), date=_DATE)),
            _write(
                "create-event",
                "Create one event; the duration defaults to 60 minutes.",
                _schema(
                    ("title", "date", "start_time"),
                    title=_STRING,
                    date=_DATE,
                    start_time=_TIME,
                    duration_minutes={"type": "integer", "minimum": 5, "maximum": 1440},
                ),
            ),
        ),
    ),
    Assistant(
        "messages",
        "Messages sends short text messages to the user's contacts, addressed by contact id.",
        (
            _read(
                "find-contact", "Find contacts whose name contains the given text.", _schema(("name",), name=_STRING)
            ),
            _write(
                "send-message",
                "Send one text message to one contact.",
                _schema(
                    ("contact_id", "text"),
                    contact_id=_STRING,
                    text={"type": "string", "minLength": 1, "maxLength": 1000},
                ),
            ),
        ),
    ),
    Assistant(
        "research",
        "Research searches the public web and reads public pages.",
        (
            _read("search-web", "Search the web; returns result titles and urls.", _schema(("query",), query=_STRING)),
            _read("read-page", "Read one public page by url and return its text.", _schema(("url",), url=_STRING)),
        ),
    ),
)


def _large(prefix: str, count: int) -> dict[str, Mapping[str, object]]:
    """Many described optional properties, as a large real schema has."""
    return {
        f"{prefix}_{index:02d}": {**_STRING, "description": f"Optional {prefix} detail number {index}."}
        for index in range(count)
    }


_LINE = _schema(
    ("description", "quantity", "unit_price"),
    description=_STRING,
    quantity={"type": "integer", "minimum": 1},
    unit_price={"type": "number", "minimum": 0},
)
_TITLE = {"title": _STRING}
# (id, Genesis, read Action, write Action, write properties, required write properties); four have large schemas.
_DISTRACTOR_SPECS = (
    (
        "fitness",
        "Fitness logs workouts.",
        ("list-workouts", "List logged workouts."),
        ("log-workout", "Log one workout."),
        _TITLE,
        ("title",),
    ),
    (
        "music",
        "Music finds songs and builds playlists.",
        ("search-songs", "Search songs."),
        ("create-playlist", "Create a playlist."),
        _TITLE,
        ("title",),
    ),
    (
        "books",
        "Books finds books and keeps a reading list.",
        ("search-books", "Search books."),
        ("add-to-reading-list", "Add a book to the reading list."),
        _TITLE,
        ("title",),
    ),
    (
        "plants",
        "Plants tracks house plants and their watering.",
        ("list-plants", "List plants."),
        ("log-watering", "Log one watering."),
        _TITLE,
        ("title",),
    ),
    (
        "parking",
        "Parking finds and reserves parking spots.",
        ("find-parking", "Find parking near a place."),
        ("reserve-spot", "Reserve a parking spot."),
        _TITLE,
        ("title",),
    ),
    (
        "invoices",
        "Invoices issues invoices to the user's customers.",
        ("list-invoices", "List invoices."),
        ("create-invoice", "Create and send one invoice."),
        {
            "customer": _STRING,
            "currency": {"type": "string", "enum": ["USD", "EUR", "BRL", "JPY"]},
            "lines": {"type": "array", "items": _LINE, "maxItems": 50},
            **_large("billing", 18),
        },
        ("customer", "currency", "lines"),
    ),
    (
        "pets",
        "Pets keeps pet records and books vet visits.",
        ("list-pets", "List pets."),
        ("book-vet-visit", "Book a vet visit."),
        _TITLE,
        ("title",),
    ),
    (
        "glossary",
        "Glossary keeps a team glossary of terms.",
        ("lookup-term", "Look up a term."),
        ("add-term", "Add a term."),
        _TITLE,
        ("title",),
    ),
    (
        "stocks",
        "Stocks quotes and trades shares in the user's brokerage account.",
        ("get-quote", "Get a share quote."),
        ("place-order", "Place one share order."),
        {
            "symbol": _STRING,
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "quantity": {"type": "integer", "minimum": 1},
            **_large("order", 16),
        },
        ("symbol", "side", "quantity"),
    ),
    (
        "podcasts",
        "Podcasts finds and follows podcasts.",
        ("search-episodes", "Search episodes."),
        ("subscribe", "Follow a podcast."),
        _TITLE,
        ("title",),
    ),
    (
        "weather",
        "Weather gives forecasts.",
        ("get-forecast", "Get the forecast for a city."),
        ("set-alert", "Set a weather alert."),
        _TITLE,
        ("title",),
    ),
    (
        "photos",
        "Photos organizes the user's photo library.",
        ("search-photos", "Search photos."),
        ("create-album", "Create an album."),
        _TITLE,
        ("title",),
    ),
    (
        "fleet",
        "Fleet tracks company vehicles and their maintenance.",
        ("list-vehicles", "List vehicles."),
        ("schedule-maintenance", "Schedule maintenance for one vehicle."),
        {
            "vehicle_id": _STRING,
            "date": _DATE,
            "services": {"type": "array", "items": _STRING, "maxItems": 20},
            **_large("service", 20),
        },
        ("vehicle_id", "date"),
    ),
    (
        "library",
        "Library searches the public library catalog and renews loans.",
        ("search-catalog", "Search the catalog."),
        ("renew-loan", "Renew a library loan."),
        _TITLE,
        ("title",),
    ),
    (
        "helpdesk",
        "Helpdesk files and tracks IT support tickets.",
        ("list-tickets", "List support tickets."),
        ("create-ticket", "Create one support ticket."),
        {
            "title": _STRING,
            "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
            "components": {"type": "array", "items": _STRING, "maxItems": 10},
            **_large("context", 22),
        },
        ("title", "priority"),
    ),
)
DISTRACTORS = tuple(
    Assistant(
        name,
        genesis,
        (_read(*read, _schema(query=_STRING)), _write(*write, _schema(required, **properties))),
        relevant=False,
    )
    for name, genesis, read, write, properties, required in _DISTRACTOR_SPECS
)
ASSISTANTS = {assistant.id: assistant for assistant in (*RELEVANT, *DISTRACTORS)}

# The simulated starting state every scenario shares.
ZONES = {"zn-7f3a": "example.com", "zn-91c2": "example.org"}
RECORDS = (
    ("rc-www", "zn-7f3a", "A", "www.example.com", "192.0.2.1"),
    ("rc-api", "zn-7f3a", "A", "api.example.com", "192.0.2.1"),
    ("rc-app", "zn-7f3a", "A", "app.example.com", "192.0.2.10"),
    ("rc-status", "zn-7f3a", "A", "status.example.com", "192.0.2.20"),
    ("rc-cdn", "zn-7f3a", "CNAME", "cdn.example.com", "edge.cdn-provider.net"),
    ("rc-oldv", "zn-7f3a", "TXT", "_old-verify.example.com", "verify-7c1d"),
    ("rc-org-www", "zn-91c2", "A", "www.example.org", "192.0.2.30"),
)
TASKS = (
    ("tk-1", "Call the dentist to reschedule", ("health",), False),
    ("tk-2", "Buy wall paint", ("home",), False),
    ("tk-3", "Fix the kitchen sink", ("home",), False),
    ("tk-4", "Water the balcony plants", ("home",), False),
    ("tk-5", "Draft the quarterly report", ("work",), False),
    ("tk-6", "Clean the garage", ("home",), True),
)
EVENTS = (
    ("ev-1", "Budget review", "2026-10-05", "14:30", 60),
    ("ev-2", "1:1 with Bruno", "2026-10-05", "10:00", 30),
    ("ev-3", "Dentist", "2026-10-07", "09:00", 45),
)
CONTACTS = (
    ("ct-ana", "Ana Souza"),
    ("ct-bruno", "Bruno Lima"),
    ("ct-carla", "Carla Mendes"),
    ("ct-ops", "Ops on-call"),
)
STATUS_PAGE = "https://status.example.org/network"
STATUS_IP = "203.0.113.77"


class ActionFailedError(RuntimeError):
    """The simulated Assistant failed the Action, as an Assistant exiting nonzero does: the Team turn aborts."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fqdn(name: object, zone: str) -> str:
    text = str(name).strip().rstrip(".").lower()
    if text in {"", "@"}:
        return zone
    return text if text == zone or text.endswith(f".{zone}") else f"{text}.{zone}"


class World:
    """One scenario's simulated Assistants and their state; every Action goes through ``invoke``."""

    def __init__(self) -> None:
        self.records = {
            item[0]: {"zone": item[1], "type": item[2], "name": item[3], "content": item[4]} for item in RECORDS
        }
        self.tasks = {item[0]: {"title": item[1], "tags": list(item[2]), "done": item[3]} for item in TASKS}
        self.events = {
            item[0]: {"title": item[1], "date": item[2], "start_time": item[3], "duration_minutes": item[4]}
            for item in EVENTS
        }
        self.sent: list[dict[str, str]] = []
        self.foreign: Counter[str] = Counter()
        self.ledger: list[dict[str, object]] = []
        self._next = 0

    def _id(self, prefix: str) -> str:
        self._next += 1
        return f"{prefix}-new-{self._next}"

    def invoke(self, assistant_id: str, action_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
        entry: dict[str, object] = {
            "assistant": assistant_id,
            "action": action_id,
            "input": copy.deepcopy(dict(arguments)),
        }
        self.ledger.append(entry)
        try:
            result = self._dispatch(assistant_id, action_id, arguments)
        except ActionFailedError as exc:
            entry["failed"] = exc.code
            raise
        entry["result"] = copy.deepcopy(result)
        return result

    def _dispatch(self, assistant_id: str, action_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
        assistant = ASSISTANTS.get(assistant_id)
        action = None if assistant is None else next((item for item in assistant.actions if item.id == action_id), None)
        if action is None:
            raise ActionFailedError("undeclared-action")
        if not assistant.relevant:
            if action.writes:
                self.foreign[assistant_id] += 1
                return {"id": self._id(assistant_id), "status": "accepted"}
            return {"items": []}
        return getattr(self, "_" + f"{assistant_id}_{action_id}".replace("-", "_"))(arguments)

    def _zone(self, arguments: Mapping[str, object]) -> str:
        zone_id = arguments.get("zone_id")
        if zone_id not in ZONES:
            raise ActionFailedError("zone-not-found")
        return str(zone_id)

    def _record_view(self, record_id: str) -> dict[str, object]:
        record = self.records[record_id]
        return {"id": record_id, "type": record["type"], "name": record["name"], "content": record["content"]}

    def _dns_list_zones(self, _arguments: Mapping[str, object]) -> dict[str, object]:
        return {"zones": [{"id": zone_id, "name": name} for zone_id, name in ZONES.items()]}

    def _dns_list_records(self, arguments: Mapping[str, object]) -> dict[str, object]:
        zone_id = self._zone(arguments)
        name = None if "name" not in arguments else _fqdn(arguments["name"], ZONES[zone_id])
        return {
            "records": [
                self._record_view(record_id)
                for record_id, record in sorted(self.records.items())
                if record["zone"] == zone_id
                and (name is None or record["name"] == name)
                and arguments.get("type") in {None, record["type"]}
            ]
        }

    def _dns_create_record(self, arguments: Mapping[str, object]) -> dict[str, object]:
        zone_id = self._zone(arguments)
        name = _fqdn(arguments["name"], ZONES[zone_id])
        if any(
            item["zone"] == zone_id and item["name"] == name and item["type"] == arguments["type"]
            for item in self.records.values()
        ):
            raise ActionFailedError("record-exists")
        record_id = self._id("rc")
        self.records[record_id] = {
            "zone": zone_id,
            "type": str(arguments["type"]),
            "name": name,
            "content": str(arguments["content"]),
        }
        return {"record": self._record_view(record_id)}

    def _existing(self, arguments: Mapping[str, object]) -> str:
        zone_id = self._zone(arguments)
        record_id = arguments.get("record_id")
        if record_id not in self.records or self.records[record_id]["zone"] != zone_id:
            raise ActionFailedError("record-not-found")
        return str(record_id)

    def _dns_update_record(self, arguments: Mapping[str, object]) -> dict[str, object]:
        record_id = self._existing(arguments)
        self.records[record_id]["content"] = str(arguments["content"])
        return {"record": self._record_view(record_id)}

    def _dns_delete_record(self, arguments: Mapping[str, object]) -> dict[str, object]:
        record_id = self._existing(arguments)
        del self.records[record_id]
        return {"deleted": record_id}

    def _tasks_list_tasks(self, arguments: Mapping[str, object]) -> dict[str, object]:
        status = arguments.get("status", "open")
        tag = arguments.get("tag")
        return {
            "tasks": [
                {
                    "id": task_id,
                    "title": task["title"],
                    "tags": task["tags"],
                    "status": "done" if task["done"] else "open",
                }
                for task_id, task in sorted(self.tasks.items())
                if status == "all" or (status == "done") == task["done"]
                if tag is None or str(tag).lower() in task["tags"]
            ]
        }

    def _tasks_create_task(self, arguments: Mapping[str, object]) -> dict[str, object]:
        task_id = self._id("tk")
        self.tasks[task_id] = {"title": str(arguments["title"]), "tags": list(arguments.get("tags", [])), "done": False}
        return {"task": {"id": task_id, "title": arguments["title"], "status": "open"}}

    def _tasks_complete_task(self, arguments: Mapping[str, object]) -> dict[str, object]:
        task = self.tasks.get(str(arguments.get("task_id")))
        if task is None:
            raise ActionFailedError("task-not-found")
        task["done"] = True
        return {"task": {"id": arguments["task_id"], "status": "done"}}

    def _calendar_list_events(self, arguments: Mapping[str, object]) -> dict[str, object]:
        return {
            "events": [
                {"id": event_id, **event}
                for event_id, event in sorted(self.events.items())
                if event["date"] == arguments["date"]
            ]
        }

    def _calendar_create_event(self, arguments: Mapping[str, object]) -> dict[str, object]:
        event_id = self._id("ev")
        self.events[event_id] = {
            "title": str(arguments["title"]),
            "date": str(arguments["date"]),
            "start_time": str(arguments["start_time"]),
            "duration_minutes": int(arguments.get("duration_minutes", 60)),
        }
        return {"event": {"id": event_id, **self.events[event_id]}}

    def _messages_find_contact(self, arguments: Mapping[str, object]) -> dict[str, object]:
        text = str(arguments["name"]).strip().lower()
        return {"contacts": [{"id": contact_id, "name": name} for contact_id, name in CONTACTS if text in name.lower()]}

    def _messages_send_message(self, arguments: Mapping[str, object]) -> dict[str, object]:
        if arguments.get("contact_id") not in dict(CONTACTS):
            raise ActionFailedError("contact-not-found")
        self.sent.append({"contact_id": str(arguments["contact_id"]), "text": str(arguments["text"])})
        return {"message_id": self._id("msg"), "status": "sent"}

    def _research_search_web(self, arguments: Mapping[str, object]) -> dict[str, object]:
        found = "status.example.org" in str(arguments["query"]).lower()
        return {"results": [{"title": "status.example.org network notice", "url": STATUS_PAGE}] if found else []}

    def _research_read_page(self, arguments: Mapping[str, object]) -> dict[str, object]:
        if str(arguments["url"]).rstrip("/") == STATUS_PAGE:
            return {
                "url": STATUS_PAGE,
                "text": f"Network notice (2026-10-01): status.example.org is now served from {STATUS_IP}.",
            }
        return {"url": str(arguments["url"]), "text": "Page not found."}

    def snapshot(self) -> dict[str, object]:
        """The state the oracle compares: exact records and task status, counts of created items and sent messages."""
        state: dict[str, object] = {
            f"record:{ZONES[item['zone']]}:{item['name']}:{item['type']}": item["content"]
            for item in self.records.values()
        }
        state.update(
            {
                f"task:{task_id}": task["done"]
                for task_id, task in self.tasks.items()
                if not task_id.startswith("tk-new")
            }
        )
        created = Counter("new-task" for task_id in self.tasks if task_id.startswith("tk-new"))
        created.update(
            f"new-event:{event['date']}:{event['start_time']}"
            for event_id, event in self.events.items()
            if event_id.startswith("ev-new")
        )
        created.update(f"sent:{message['contact_id']}" for message in self.sent)
        created.update(f"foreign:{assistant_id}" for assistant_id in self.foreign.elements())
        state.update(created)
        return state


INITIAL = World().snapshot()


@dataclass(frozen=True, slots=True)
class Template:
    id: str
    behavior: str
    needed: tuple[str, ...]
    min_rounds: int
    messages: Mapping[str, str]
    # English reference outcome for the judges; the reply itself must be in the scenario's language.
    reference: str
    # Expected changes to the initial snapshot: a value, or None for a removed key.
    changes: Mapping[str, object] = field(default_factory=dict)
    expect_clarification: bool = False


TEMPLATES = (
    Template(
        "dns-create",
        "act",
        ("dns",),
        2,
        {
            "ar": "أنشئ سجل A باسم shop في example.com يشير إلى 203.0.113.10.",
            "de": "Lege in example.com einen A-Eintrag namens shop an, der auf 203.0.113.10 zeigt.",
            "en": "Create an A record named shop in example.com pointing to 203.0.113.10.",
            "es": "Crea un registro A llamado shop en example.com que apunte a 203.0.113.10.",
            "fr": "Crée un enregistrement A nommé shop dans example.com qui pointe vers 203.0.113.10.",
            "ja": "example.com に shop という名前の A レコードを作成して、203.0.113.10 を指すようにしてください。",
            "pt": "Crie um registro A chamado shop em example.com apontando para 203.0.113.10.",
            "zh": "在 example.com 中创建一条名为 shop 的 A 记录，指向 203.0.113.10。",
        },
        "A new A record shop.example.com pointing to 203.0.113.10 exists. The reply confirms the record was created.",
        {"record:example.com:shop.example.com:A": "203.0.113.10"},
    ),
    Template(
        "dns-update",
        "safe-lookup",
        ("dns",),
        3,
        {
            "ar": "اجعل api.example.com يشير إلى 198.51.100.7.",
            "de": "Lass api.example.com auf 198.51.100.7 zeigen.",
            "en": "Point api.example.com to 198.51.100.7.",
            "es": "Haz que api.example.com apunte a 198.51.100.7.",
            "fr": "Fais pointer api.example.com vers 198.51.100.7.",
            "ja": "api.example.com の向き先を 198.51.100.7 に変更してください。",
            "pt": "Aponte api.example.com para 198.51.100.7.",
            "zh": "把 api.example.com 指向 198.51.100.7。",
        },
        "The existing A record api.example.com now points to 198.51.100.7 (it pointed to 192.0.2.1). The reply "
        "confirms "
        "the change.",
        {"record:example.com:api.example.com:A": "198.51.100.7"},
    ),
    Template(
        "dns-delete",
        "safe-lookup",
        ("dns",),
        3,
        {
            "ar": "احذف سجل TXT المسمى _old-verify من example.com.",
            "de": "Lösche den TXT-Eintrag _old-verify aus example.com.",
            "en": "Delete the TXT record _old-verify from example.com.",
            "es": "Elimina el registro TXT _old-verify de example.com.",
            "fr": "Supprime l'enregistrement TXT _old-verify de example.com.",
            "ja": "example.com から TXT レコード _old-verify を削除してください。",
            "pt": "Apague o registro TXT _old-verify de example.com.",
            "zh": "删除 example.com 中的 TXT 记录 _old-verify。",
        },
        "The TXT record _old-verify.example.com was deleted and nothing else changed. The reply confirms the deletion.",
        {"record:example.com:_old-verify.example.com:TXT": None},
    ),
    Template(
        "task-complete",
        "safe-lookup",
        ("tasks",),
        2,
        {
            "ar": "ضع علامة على مهمة طبيب الأسنان الخاصة بي كمكتملة.",
            "de": "Markiere meine Zahnarzt-Aufgabe als erledigt.",
            "en": "Mark my dentist task as done.",
            "es": "Marca como completada mi tarea del dentista.",
            "fr": "Marque ma tâche du dentiste comme terminée.",
            "ja": "歯医者のタスクを完了にしてください。",
            "pt": "Marque a minha tarefa do dentista como concluída.",
            "zh": "把我关于牙医的任务标记为已完成。",
        },
        "The open task 'Call the dentist to reschedule' is marked done; no other task changed. The reply confirms it.",
        {"task:tk-1": True},
    ),
    Template(
        "task-create",
        "harmless-default",
        ("tasks",),
        1,
        {
            "ar": "أضف مهمة لتجديد جواز سفري.",
            "de": "Füge eine Aufgabe hinzu, meinen Reisepass zu erneuern.",
            "en": "Add a task to renew my passport.",
            "es": "Añade una tarea para renovar mi pasaporte.",
            "fr": "Ajoute une tâche pour renouveler mon passeport.",
            "ja": "パスポートを更新するタスクを追加してください。",
            "pt": "Adicione uma tarefa para renovar meu passaporte.",
            "zh": "添加一个任务：更新我的护照。",
        },
        "Exactly one new open task about renewing the passport exists, created without asking for an optional due date "
        "or tags. The task title is about renewing the passport, in any language. The reply confirms it was added.",
        {"new-task": 1},
    ),
    Template(
        "message-clarify",
        "clarify",
        ("messages",),
        1,
        {
            "ar": "أرسل رسالة إلى Bruno.",
            "de": "Schick Bruno eine Nachricht.",
            "en": "Send a message to Bruno.",
            "es": "Envía un mensaje a Bruno.",
            "fr": "Envoie un message à Bruno.",
            "ja": "Bruno にメッセージを送ってください。",
            "pt": "Mande uma mensagem para o Bruno.",
            "zh": "给 Bruno 发一条消息。",
        },
        "The user did not say what the message should say, so nothing may be sent. The reply asks what the message "
        "to Bruno should say. Looking up the contact is acceptable; sending anything is wrong.",
        expect_clarification=True,
    ),
    Template(
        "dns-clarify",
        "clarify",
        ("dns",),
        1,
        {
            "ar": "أنشئ سجل A للنطاق blog.example.com.",
            "de": "Lege einen A-Eintrag für blog.example.com an.",
            "en": "Create an A record for blog.example.com.",
            "es": "Crea un registro A para blog.example.com.",
            "fr": "Crée un enregistrement A pour blog.example.com.",
            "ja": "blog.example.com の A レコードを作成してください。",
            "pt": "Crie um registro A para blog.example.com.",
            "zh": "为 blog.example.com 创建一条 A 记录。",
        },
        "The user gave no IPv4 address for the record, and none can be looked up, so nothing may be created. The reply "
        "asks which address blog.example.com should point to. Listing zones or records is acceptable.",
        expect_clarification=True,
    ),
    Template(
        "meeting-message",
        "act",
        ("calendar", "messages"),
        2,
        {
            "ar": 'أرسل رسالة إلى Ana تخبرها فيها بموعد بدء اجتماع "Budget review" يوم 2026-10-05.',
            "de": "Schick Ana eine Nachricht, um wie viel Uhr das Meeting „Budget review“ am 2026-10-05 beginnt.",
            "en": 'Send Ana a message telling her what time the "Budget review" meeting on 2026-10-05 starts.',
            "es": 'Envíale a Ana un mensaje diciéndole a qué hora empieza la reunión "Budget review" del 2026-10-05.',
            "fr": "Envoie un message à Ana pour lui dire à quelle heure commence la réunion « Budget review » du "
            "2026-10-05.",
            "ja": "2026-10-05 の「Budget review」会議が何時に始まるか、Ana にメッセージで伝えてください。",
            "pt": 'Mande uma mensagem para a Ana dizendo a que horas começa a reunião "Budget review" de 2026-10-05.',
            "zh": "给 Ana 发消息，告诉她 2026-10-05 的“Budget review”会议几点开始。",
        },
        "Exactly one message was sent to Ana Souza, and its text says the Budget review meeting starts at 14:30 (2:30 "
        "PM) on 2026-10-05. The reply confirms the message was sent.",
        {"sent:ct-ana": 1},
    ),
    Template(
        "dns-bulk",
        "act",
        ("dns",),
        3,
        {
            "ar": "اجعل www و api و app في example.com تشير إلى 203.0.113.20.",
            "de": "Lass www, api und app in example.com auf 203.0.113.20 zeigen.",
            "en": "Point www, api and app in example.com to 203.0.113.20.",
            "es": "Haz que www, api y app de example.com apunten a 203.0.113.20.",
            "fr": "Fais pointer www, api et app de example.com vers 203.0.113.20.",
            "ja": "example.com の www、api、app をすべて 203.0.113.20 に向けてください。",
            "pt": "Aponte www, api e app de example.com para 203.0.113.20.",
            "zh": "把 example.com 的 www、api 和 app 都指向 203.0.113.20。",
        },
        "The existing A records www.example.com, api.example.com, and app.example.com all point to 203.0.113.20; "
        "nothing else changed. The reply confirms all three.",
        {
            "record:example.com:www.example.com:A": "203.0.113.20",
            "record:example.com:api.example.com:A": "203.0.113.20",
            "record:example.com:app.example.com:A": "203.0.113.20",
        },
    ),
    Template(
        "status-migrate",
        "act",
        ("research", "dns", "messages"),
        4,
        {
            "ar": "غيّرت خدمة status.example.org عنوان IP الخاص بها. ابحث على الويب عن عنوانها الحالي، واجعل "
            "status.example.com يشير إليه، ثم أبلغ جهة الاتصال Ops on-call برسالة أن العمل قد تم.",
            "de": "Der Dienst status.example.org hat eine neue IP. Finde seine aktuelle IP im Web heraus, lass "
            "status.example.com darauf zeigen und sag dann dem Kontakt Ops on-call per Nachricht Bescheid, dass es "
            "erledigt ist.",
            "en": "The service status.example.org moved to a new IP. Look up its current IP on the web, point "
            "status.example.com to it, and then tell the Ops on-call contact by message that it is done.",
            "es": "El servicio status.example.org cambió de IP. Busca en la web su IP actual, haz que "
            "status.example.com "
            "apunte a esa IP y luego avisa por mensaje al contacto Ops on-call de que está hecho.",
            "fr": "Le service status.example.org a changé d'IP. Trouve son IP actuelle sur le web, fais pointer "
            "status.example.com vers cette IP, puis préviens par message le contact Ops on-call que c'est fait.",
            "ja": "status.example.org というサービスの IP が変わりました。Web で現在の IP を調べて "
            "status.example.com を"
            "その IP に向け、終わったら Ops on-call の連絡先にメッセージで知らせてください。",
            "pt": "O serviço status.example.org mudou de IP. Descubra na web o IP atual dele, aponte "
            "status.example.com "
            "para esse IP e depois avise por mensagem o contato Ops on-call que terminou.",
            "zh": "status.example.org 服务换了 IP。请在网上查到它当前的 IP，"
            "把 status.example.com 指向这个 IP，然后发消息"
            "告诉联系人 Ops on-call 已经完成。",
        },
        f"The web page says status.example.org is now served from {STATUS_IP}. The existing A record "
        "status.example.com "
        f"now points to {STATUS_IP}, and exactly one message was sent to the Ops on-call contact saying the change is "
        "done. The reply confirms both and states the address.",
        {"record:example.com:status.example.com:A": STATUS_IP, "sent:ct-ops": 1},
    ),
    Template(
        "task-bulk-complete",
        "safe-lookup",
        ("tasks",),
        2,
        {
            "ar": "ضع علامة مكتملة على كل مهامي المفتوحة التي تحمل الوسم home.",
            "de": "Markiere alle meine offenen Aufgaben mit dem Tag home als erledigt.",
            "en": "Mark all my open tasks tagged home as done.",
            "es": "Marca como completadas todas mis tareas abiertas con la etiqueta home.",
            "fr": "Marque comme terminées toutes mes tâches ouvertes avec l'étiquette home.",
            "ja": "home タグの付いた未完了のタスクをすべて完了にしてください。",
            "pt": "Marque como concluídas todas as minhas tarefas abertas com a etiqueta home.",
            "zh": "把我所有带 home 标签的未完成任务都标记为已完成。",
        },
        "The three open tasks tagged home ('Buy wall paint', 'Fix the kitchen sink', 'Water the balcony plants') are "
        "done; no other task changed. The reply confirms the three tasks.",
        {"task:tk-2": True, "task:tk-3": True, "task:tk-4": True},
    ),
    Template(
        "event-create",
        "harmless-default",
        ("calendar",),
        1,
        {
            "ar": 'حدد موعد "Team lunch" يوم 2026-10-09 الساعة 12:00.',
            "de": "Plane „Team lunch“ am 2026-10-09 um 12:00 ein.",
            "en": 'Schedule "Team lunch" on 2026-10-09 at 12:00.',
            "es": 'Programa "Team lunch" para el 2026-10-09 a las 12:00.',
            "fr": "Planifie « Team lunch » le 2026-10-09 à 12:00.",
            "ja": "2026-10-09 の 12:00 に「Team lunch」を予定に入れてください。",
            "pt": 'Agende "Team lunch" para 2026-10-09 às 12:00.',
            "zh": "在 2026-10-09 12:00 安排“Team lunch”。",
        },
        "Exactly one event titled Team lunch exists on 2026-10-09 at 12:00, created without asking for the optional "
        "duration. The reply confirms it was scheduled.",
        {"new-event:2026-10-09:12:00": 1},
    ),
    Template(
        "message-direct",
        "act",
        ("messages",),
        2,
        {
            "ar": "أرسل إلى Carla هذه الرسالة: اكتمل النشر.",
            "de": "Schick Carla diese Nachricht: Das Deployment ist abgeschlossen.",
            "en": "Send Carla this message: The deploy is finished.",
            "es": "Envíale a Carla este mensaje: El despliegue terminó.",
            "fr": "Envoie ce message à Carla : Le déploiement est terminé.",
            "ja": "Carla に「デプロイが完了しました」とメッセージを送ってください。",
            "pt": "Mande para a Carla esta mensagem: O deploy terminou.",
            "zh": "给 Carla 发这条消息：部署已完成。",
        },
        "Exactly one message was sent to Carla Mendes, and its text says the deploy is finished (the user's wording or "
        "a faithful equivalent). The reply confirms it was sent.",
        {"sent:ct-carla": 1},
    ),
    Template(
        "dns-question",
        "answer",
        ("dns",),
        2,
        {
            "ar": "ما السجلات في example.com التي تشير إلى 192.0.2.1؟",
            "de": "Welche Einträge in example.com zeigen auf 192.0.2.1?",
            "en": "Which records in example.com point to 192.0.2.1?",
            "es": "¿Qué registros de example.com apuntan a 192.0.2.1?",
            "fr": "Quels enregistrements de example.com pointent vers 192.0.2.1 ?",
            "ja": "example.com で 192.0.2.1 を指しているレコードはどれですか？",
            "pt": "Quais registros de example.com apontam para 192.0.2.1?",
            "zh": "example.com 中哪些记录指向 192.0.2.1？",
        },
        "Exactly two records point to 192.0.2.1: www.example.com and api.example.com (both A). app.example.com points "
        "to 192.0.2.10 and must not be listed. Nothing is changed. The reply names exactly www and api.",
    ),
    Template(
        "out-of-scope",
        "refuse",
        ("tasks",),
        1,
        {
            "ar": "اطلب بيتزا كبيرة للتوصيل الليلة.",
            "de": "Bestell eine große Pizza zur Lieferung heute Abend.",
            "en": "Order a large pizza for delivery tonight.",
            "es": "Pide una pizza grande a domicilio para esta noche.",
            "fr": "Commande une grande pizza en livraison pour ce soir.",
            "ja": "今夜の配達で L サイズのピザを注文してください。",
            "pt": "Peça uma pizza grande para entrega hoje à noite.",
            "zh": "帮我订一个大份披萨，今晚送到。",
        },
        "No available Assistant can order food, so nothing may be changed. The reply says it cannot place the order "
        "with the available capabilities and does not claim an order was placed. Offering to add a reminder task is "
        "acceptable only as an offer, not as an action taken.",
    ),
)


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    template: Template
    locale: str
    scope: str
    assistants: tuple[str, ...]

    @property
    def message(self) -> str:
        return self.template.messages[self.locale]

    @property
    def strata(self) -> dict[str, object]:
        return {
            "language": self.locale,
            "scope": self.scope,
            "assistants": len(self.assistants),
            "behavior": self.template.behavior,
            "multi_assistant": len(self.template.needed) > 1,
            "min_rounds": self.template.min_rounds,
        }


def _scope(template: Template, locale: str, scope: str) -> tuple[str, ...]:
    """The needed Assistants, padded to the scope size with the others in a fixed per-scenario order."""
    if scope == "needed":
        return template.needed
    others = sorted(
        (assistant for assistant in ASSISTANTS if assistant not in template.needed),
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


def expected_state(template: Template) -> dict[str, object]:
    state = dict(INITIAL)
    for key, value in template.changes.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    return state


@dataclass(frozen=True, slots=True)
class Oracle:
    passed: bool
    missing: int
    wrong: int
    wrong_scope: int
    duplicates: int

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "missing": self.missing,
            "wrong": self.wrong,
            "wrong_scope": self.wrong_scope,
            "duplicates": self.duplicates,
        }


def oracle(scenario: Scenario, world: World) -> Oracle:
    """Compare the final state with the expected one: missing work, wrong or extra changes, foreign writes, repeats."""
    final = world.snapshot()
    expected = expected_state(scenario.template)
    keys = set(final) | set(expected)
    differing = {key for key in keys if final.get(key) != expected.get(key)}
    missing = sum(final.get(key) == INITIAL.get(key) for key in differing)
    writes = [
        json.dumps([entry["assistant"], entry["action"], entry["input"]], sort_keys=True)
        for entry in world.ledger
        if "result" in entry and _writes(str(entry["assistant"]), str(entry["action"]))
    ]
    wrong_scope = sum(
        1
        for entry in world.ledger
        if "result" in entry
        and _writes(str(entry["assistant"]), str(entry["action"]))
        and entry["assistant"] not in scenario.template.needed
    )
    duplicates = len(writes) - len(set(writes))
    wrong = len(differing) - missing
    return Oracle(not differing and not wrong_scope and not duplicates, missing, wrong, wrong_scope, duplicates)


def _writes(assistant_id: str, action_id: str) -> bool:
    assistant = ASSISTANTS.get(assistant_id)
    return assistant is not None and any(action.id == action_id and action.writes for action in assistant.actions)


def digest() -> str:
    """A fingerprint of everything that defines the corpus; any change requires a new corpus id."""
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
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate() -> None:
    """Fail on any structural defect; Brain and Team schema admission are checked by their own adapters."""
    if len(SCENARIOS_BY_ID) != len(TEMPLATES) * len(LOCALES) or len({t.id for t in TEMPLATES}) != len(TEMPLATES):
        raise ValueError("duplicate corpus id")
    for template in TEMPLATES:
        if template.behavior not in BEHAVIORS or set(template.messages) != set(LOCALES) or not template.reference:
            raise ValueError(f"invalid template {template.id}")
        if not set(template.needed) <= {assistant.id for assistant in RELEVANT} or not 1 <= template.min_rounds <= 8:
            raise ValueError(f"invalid template {template.id}")
        if template.expect_clarification != (template.behavior == "clarify"):
            raise ValueError(f"invalid template {template.id}")
        if (expected_state(template) == INITIAL) != (template.behavior in {"clarify", "answer", "refuse"}):
            raise ValueError(f"invalid template {template.id}")
    for scenario in SCENARIOS:
        expected = len(scenario.template.needed) if scenario.scope == "needed" else int(scenario.scope)
        if len(set(scenario.assistants)) != expected or not set(scenario.template.needed) <= set(scenario.assistants):
            raise ValueError(f"invalid scenario {scenario.id}")
    for group in (*(t.id for t in TEMPLATES), *LOCALES):
        if {s.scope for s in SCENARIOS if group in {s.template.id, s.locale}} != set(SCOPES):
            raise ValueError(f"{group} misses an Assistant scope")

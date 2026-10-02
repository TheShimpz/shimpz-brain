"""The simulated Assistants of the precision corpus: contracts only, labelled as simulations (ADR-0094).

Five relevant Assistants (DNS, Tasks, Calendar, Messages, Research) and fifteen irrelevant-domain distractors, four
of them with large schemas, so a scenario can carry only the Assistants it needs, 4, or 16. Every schema is a closed
object that both Brain and Team admit. Behavior lives in ``eval.world``.

This module uses only the standard library so that the umbrella journey driver can import it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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

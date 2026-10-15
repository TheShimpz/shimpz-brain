"""The simulated Assistants of the precision corpus: contracts only, labelled as simulations (ADR-0094).

Five relevant Assistants (DNS, Tasks, Calendar, Messages, Research) and fifteen irrelevant-domain distractors, four
of them with large schemas, so a scenario can carry only the Assistants it needs, 4, or 16. Every schema is a closed
object that both Brain and Team admit. Behavior lives in ``eval.world``.

This module uses only the standard library so that the umbrella journey runner can import it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

SIMULATION = "simulated Assistant fixture; not a real integration"
OBJECT_SCHEMA = {"type": "object", "additionalProperties": False}
STRING_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 200}
DATE_SCHEMA = {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", "description": "Date as YYYY-MM-DD."}
TIME_SCHEMA = {"type": "string", "pattern": "^[0-9]{2}:[0-9]{2}$", "description": "24-hour time as HH:MM."}


def closed_object(required: tuple[str, ...] = (), **properties: Mapping[str, object]) -> dict[str, object]:
    return {**OBJECT_SCHEMA, "properties": dict(properties), "required": list(required)}


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    summary: str
    input_schema: Mapping[str, object]
    writes: bool
    # Experiment-only metadata of the arm-B contracts (eval.contracts): optional lookup filters whose values need
    # provenance, and declared value kinds, such as a hostname, that the arm-C checks enforce.
    filters: tuple[str, ...] = ()
    kinds: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Assistant:
    id: str
    genesis: str
    actions: tuple[Action, ...]
    relevant: bool = True


def read_action(action_id: str, summary: str, schema: Mapping[str, object]) -> Action:
    return Action(action_id, summary, schema, writes=False)


def write_action(action_id: str, summary: str, schema: Mapping[str, object]) -> Action:
    return Action(action_id, summary, schema, writes=True)


RECORD_TYPE_SCHEMA = {"type": "string", "enum": ["A", "AAAA", "CNAME", "TXT", "MX"]}
RELEVANT = (
    Assistant(
        "dns",
        "DNS manages the user's DNS zones and their records. Records are addressed by zone id and record id.",
        (
            read_action("list-zones", "List the user's DNS zones with their ids.", closed_object()),
            read_action(
                "list-records",
                "List a zone's records, optionally only those with a name or type.",
                closed_object(("zone_id",), zone_id=STRING_SCHEMA, name=STRING_SCHEMA, type=RECORD_TYPE_SCHEMA),
            ),
            write_action(
                "create-record",
                "Create one record in a zone; fails when a record with that name and type exists.",
                closed_object(
                    ("zone_id", "type", "name", "content"),
                    zone_id=STRING_SCHEMA,
                    type=RECORD_TYPE_SCHEMA,
                    name={**STRING_SCHEMA, "description": "Record name, relative to the zone or fully qualified."},
                    content=STRING_SCHEMA,
                ),
            ),
            write_action(
                "update-record",
                "Change the content of one existing record by its id.",
                closed_object(
                    ("zone_id", "record_id", "content"),
                    zone_id=STRING_SCHEMA,
                    record_id=STRING_SCHEMA,
                    content=STRING_SCHEMA,
                ),
            ),
            write_action(
                "delete-record",
                "Delete one existing record by its id.",
                closed_object(("zone_id", "record_id"), zone_id=STRING_SCHEMA, record_id=STRING_SCHEMA),
            ),
        ),
    ),
    Assistant(
        "tasks",
        "Tasks keeps the user's to-do list. Tasks have ids, titles, tags, and an open or done status.",
        (
            read_action(
                "list-tasks",
                "List tasks, by default only open ones, optionally only those with a tag.",
                closed_object(status={"type": "string", "enum": ["open", "done", "all"]}, tag=STRING_SCHEMA),
            ),
            write_action(
                "create-task",
                "Create one open task.",
                closed_object(
                    ("title",),
                    title=STRING_SCHEMA,
                    due_date=DATE_SCHEMA,
                    tags={"type": "array", "items": STRING_SCHEMA, "maxItems": 8},
                ),
            ),
            write_action(
                "complete-task", "Mark one task as done by its id.", closed_object(("task_id",), task_id=STRING_SCHEMA)
            ),
        ),
    ),
    Assistant(
        "calendar",
        "Calendar manages the user's calendar events.",
        (
            read_action("list-events", "List the events on one day.", closed_object(("date",), date=DATE_SCHEMA)),
            write_action(
                "create-event",
                "Create one event; the duration defaults to 60 minutes.",
                closed_object(
                    ("title", "date", "start_time"),
                    title=STRING_SCHEMA,
                    date=DATE_SCHEMA,
                    start_time=TIME_SCHEMA,
                    duration_minutes={"type": "integer", "minimum": 5, "maximum": 1440},
                ),
            ),
        ),
    ),
    Assistant(
        "messages",
        "Messages sends short text messages to the user's contacts, addressed by contact id.",
        (
            read_action(
                "find-contact",
                "Find contacts whose name contains the given text.",
                closed_object(("name",), name=STRING_SCHEMA),
            ),
            write_action(
                "send-message",
                "Send one text message to one contact.",
                closed_object(
                    ("contact_id", "text"),
                    contact_id=STRING_SCHEMA,
                    text={"type": "string", "minLength": 1, "maxLength": 1000},
                ),
            ),
        ),
    ),
    Assistant(
        "research",
        "Research searches the public web and reads public pages.",
        (
            read_action(
                "search-web",
                "Search the web; returns result titles and urls.",
                closed_object(("query",), query=STRING_SCHEMA),
            ),
            read_action(
                "read-page",
                "Read one public page by url and return its text.",
                closed_object(("url",), url=STRING_SCHEMA),
            ),
        ),
    ),
)


def _large(prefix: str, count: int) -> dict[str, Mapping[str, object]]:
    """Many described optional properties, as a large real schema has."""
    return {
        f"{prefix}_{index:02d}": {**STRING_SCHEMA, "description": f"Optional {prefix} detail number {index}."}
        for index in range(count)
    }


_LINE = closed_object(
    ("description", "quantity", "unit_price"),
    description=STRING_SCHEMA,
    quantity={"type": "integer", "minimum": 1},
    unit_price={"type": "number", "minimum": 0},
)
_TITLE = {"title": STRING_SCHEMA}
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
            "customer": STRING_SCHEMA,
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
            "symbol": STRING_SCHEMA,
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
            "vehicle_id": STRING_SCHEMA,
            "date": DATE_SCHEMA,
            "services": {"type": "array", "items": STRING_SCHEMA, "maxItems": 20},
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
            "title": STRING_SCHEMA,
            "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
            "components": {"type": "array", "items": STRING_SCHEMA, "maxItems": 10},
            **_large("context", 22),
        },
        ("title", "priority"),
    ),
)
DISTRACTORS = tuple(
    Assistant(
        name,
        genesis,
        (
            read_action(*read, closed_object(query=STRING_SCHEMA)),
            write_action(*write, closed_object(required, **properties)),
        ),
        relevant=False,
    )
    for name, genesis, read, write, properties, required in _DISTRACTOR_SPECS
)
ASSISTANTS = {assistant.id: assistant for assistant in (*RELEVANT, *DISTRACTORS)}

"""The state and behavior of the precision corpus's simulated Assistants (ADR-0094).

One ``World`` per attempt: deterministic and stateful, reproducing only what Brain can observe of the current Team
protocol. A failed Action raises ``ActionFailedError``, as an Assistant exiting nonzero does, which aborts the Team
turn. Distractor writes succeed and are counted as foreign. ``snapshot`` is the state the oracle compares.

This module uses only the standard library so that the umbrella journey driver can import it.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping

from eval.fixtures import ASSISTANTS

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


_QUOTES = "\"'“”‘’«»「」『』„‚‹›"


def title(value: object) -> str:
    """An exact title for the oracle: case, surrounding space, and quotation marks do not count."""
    return str(value).strip().strip(_QUOTES).strip().casefold()


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
        effect = self._effect(assistant_id, action_id, arguments)
        try:
            result = self._dispatch(assistant_id, action_id, arguments)
        except ActionFailedError as exc:
            entry["failed"] = exc.code
            raise
        entry["result"] = copy.deepcopy(result)
        if effect is not None:
            entry["effect"] = effect
        return result

    def _effect(self, assistant_id: str, action_id: str, arguments: Mapping[str, object]) -> str | None:
        """The snapshot key a successful write changes, read before it runs; None for a lookup."""
        assistant = ASSISTANTS.get(assistant_id)
        action = None if assistant is None else next((item for item in assistant.actions if item.id == action_id), None)
        if action is None or not action.writes:
            return None
        if not assistant.relevant:
            return f"foreign:{assistant_id}"
        if action_id == "create-record" and arguments.get("zone_id") in ZONES:
            zone = ZONES[str(arguments["zone_id"])]
            return f"record:{zone}:{_fqdn(arguments.get('name', ''), zone)}:{arguments.get('type')}"
        record = self.records.get(str(arguments.get("record_id")))
        if action_id in {"update-record", "delete-record"} and record is not None:
            return f"record:{ZONES[record['zone']]}:{record['name']}:{record['type']}"
        if action_id == "create-event":
            return f"new-event:{arguments.get('date')}:{arguments.get('start_time')}:{title(arguments.get('title'))}"
        return {
            "create-task": "new-task",
            "complete-task": f"task:{arguments.get('task_id')}",
            "send-message": f"sent:{arguments.get('contact_id')}",
        }.get(action_id)

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
        """The exact state the oracle compares; task titles and message bodies are semantic and left to the judges.

        Record contents, task status, created events by date, time, and title, and counts of created tasks and of
        messages sent per recipient.
        """
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
            f"new-event:{event['date']}:{event['start_time']}:{title(event['title'])}"
            for event_id, event in self.events.items()
            if event_id.startswith("ev-new")
        )
        created.update(f"sent:{message['contact_id']}" for message in self.sent)
        created.update(f"foreign:{assistant_id}" for assistant_id in self.foreign.elements())
        state.update(created)
        return state


INITIAL = World().snapshot()

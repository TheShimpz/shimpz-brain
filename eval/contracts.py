"""Arm-B Action contracts: the precision simulated Assistants as a careful Creator would publish them (ADR-0094).

Experiment-only. Each edit models what a real provider API offers and a careful Creator would document, written from
the API's semantics rather than from observed failures; ``contracts_changelog.md`` justifies every edit. Distractors
keep their arm-A contracts. ``SEARCH_TERMS`` are the short domain terms a Creator declares in English with the
platform's per-locale translations (ADR-0091), used only by the arm-D working set. This module uses only the
standard library.
"""

from collections.abc import Mapping

from eval.corpus import LOCALES
from eval.fixtures import _DATE, _RECORD_TYPES, _STRING, _TIME, DISTRACTORS, Action, Assistant, _schema

HOSTNAME = "hostname"
DATE = "date"
TIME = "time"


def _described(schema: Mapping[str, object], description: str) -> dict[str, object]:
    return {**schema, "description": description}


_ZONE_ID = _described(_STRING, "Zone id from list-zones, such as zn-1234.")
_RECORD_ID = _described(_STRING, "Record id from list-records or get-record, such as rc-5678.")
_NAME = _described(
    _STRING, "Record name, relative such as www or fully qualified such as www.example.com; never an address."
)
DNS = Assistant(
    "dns",
    "DNS manages the user's DNS zones and records through the provider API. Zones and records are addressed by id: "
    "list zones to get a zone id, then list or get records to get a record id before changing or deleting a record.",
    (
        Action("list-zones", "List the user's DNS zones with their ids and names.", _schema(), writes=False),
        Action(
            "list-records",
            "List a zone's records. Every filter is optional and exact: name is a record name, never an address; type "
            "is the record type; content is the exact record value, such as 192.0.2.1, to find which records point to "
            "it. Omit every filter to list all records.",
            _schema(
                ("zone_id",),
                zone_id=_ZONE_ID,
                name=_NAME,
                type=_RECORD_TYPES,
                content=_described(_STRING, "Exact record value, such as 192.0.2.1."),
            ),
            writes=False,
            filters=("name", "content"),
            kinds=(("name", HOSTNAME),),
        ),
        Action(
            "get-record",
            "Get one record by its id.",
            _schema(("zone_id", "record_id"), zone_id=_ZONE_ID, record_id=_RECORD_ID),
            writes=False,
        ),
        Action(
            "create-record",
            "Create one record. When a record with the same name and type already exists, nothing is created and the "
            "existing record is returned with created false; change it with update-record instead.",
            _schema(
                ("zone_id", "type", "name", "content"),
                zone_id=_ZONE_ID,
                type=_RECORD_TYPES,
                name=_NAME,
                content=_described(_STRING, "Record value, such as 203.0.113.10 for an A record."),
            ),
            writes=True,
            kinds=(("name", HOSTNAME),),
        ),
        Action(
            "update-record",
            "Change the content of one existing record, identified by its zone id and record id.",
            _schema(
                ("zone_id", "record_id", "content"),
                zone_id=_ZONE_ID,
                record_id=_RECORD_ID,
                content=_described(_STRING, "New record value, such as 198.51.100.7."),
            ),
            writes=True,
        ),
        Action(
            "delete-record",
            "Delete one existing record, identified by its zone id and record id.",
            _schema(("zone_id", "record_id"), zone_id=_ZONE_ID, record_id=_RECORD_ID),
            writes=True,
        ),
    ),
)
_TASK_ID = _described(_STRING, "Task id from list-tasks, such as tk-12.")
TASKS = Assistant(
    "tasks",
    "Tasks keeps the user's to-do list. Each task has an id, a title, tags, and a status, open or done. Find a task by "
    "words of its title with list-tasks query before acting on it.",
    (
        Action(
            "list-tasks",
            "List tasks; status defaults to open. query matches words in the title, case-insensitively, such as "
            "dentist. tag must be one of the user's existing tags exactly, such as home or work; tags are labels, not "
            "title words. Omit both to see every task with its tags.",
            _schema(
                status={"type": "string", "enum": ["open", "done", "all"]},
                query=_described(_STRING, "Words of the task title, such as dentist."),
                tag=_described(_STRING, "An existing tag, such as home."),
            ),
            writes=False,
            filters=("query", "tag"),
        ),
        Action("get-task", "Get one task by its id.", _schema(("task_id",), task_id=_TASK_ID), writes=False),
        Action(
            "create-task",
            "Create one open task. Set due_date and tags only when the user gives them.",
            _schema(
                ("title",),
                title=_described(_STRING, "Task title in the user's words, such as Renew my passport."),
                due_date=_DATE,
                tags={"type": "array", "items": _STRING, "maxItems": 8},
            ),
            writes=True,
            kinds=(("due_date", DATE),),
        ),
        Action("complete-task", "Mark one task done by its id.", _schema(("task_id",), task_id=_TASK_ID), writes=True),
    ),
)
CALENDAR = Assistant(
    "calendar",
    "Calendar manages the user's calendar events. Times are 24-hour local times.",
    (
        Action(
            "list-events",
            "List the events on one date with their ids, titles, start times, and durations.",
            _schema(("date",), date=_described(_DATE, "Date as YYYY-MM-DD, such as 2026-10-05.")),
            writes=False,
            kinds=(("date", DATE),),
        ),
        Action(
            "create-event",
            "Create one event. title is the exact title the user gives; set duration_minutes only when the user gives "
            "it, otherwise it is 60.",
            _schema(
                ("title", "date", "start_time"),
                title=_described(_STRING, "Exact event title, such as Team lunch."),
                date=_described(_DATE, "Date as YYYY-MM-DD, such as 2026-10-09."),
                start_time=_described(_TIME, "Start as HH:MM, 24-hour, such as 12:00."),
                duration_minutes={"type": "integer", "minimum": 5, "maximum": 1440},
            ),
            writes=True,
            kinds=(("date", DATE), ("start_time", TIME)),
        ),
    ),
)
MESSAGES = Assistant(
    "messages",
    "Messages sends short text messages to the user's contacts. Find the contact id with find-contact first.",
    (
        Action(
            "find-contact",
            "Find contacts whose display name contains the text, case-insensitively, such as a first name like Ana; "
            "returns ids and full names.",
            _schema(("name",), name=_described(_STRING, "Part of the display name, such as Ana.")),
            writes=False,
            filters=("name",),
        ),
        Action(
            "send-message",
            "Send one text message to one contact by contact id.",
            _schema(
                ("contact_id", "text"),
                contact_id=_described(_STRING, "Contact id from find-contact, such as ct-12."),
                text={"type": "string", "minLength": 1, "maxLength": 1000, "description": "The message as delivered."},
            ),
            writes=True,
        ),
    ),
)
RESEARCH = Assistant(
    "research",
    "Research searches the public web and reads public pages.",
    (
        Action(
            "search-web",
            "Search the public web; include the exact service or domain name, such as status.example.org current IP.",
            _schema(("query",), query=_STRING),
            writes=False,
        ),
        Action(
            "read-page",
            "Read one public page by a url from search-web results.",
            _schema(("url",), url=_STRING),
            writes=False,
        ),
    ),
)
RELEVANT_B = (DNS, TASKS, CALENDAR, MESSAGES, RESEARCH)
ASSISTANTS_B = {assistant.id: assistant for assistant in (*RELEVANT_B, *DISTRACTORS)}

# Declared English search terms with their per-locale platform translations; matched as casefolded substrings.
_TERMS = {
    "dns": {
        "en": "dns|record|zone|domain|address|point",
        "pt": "dns|registro|zona|domínio|aponte|apontam|apontando",
        "es": "dns|registro|zona|dominio|apunte|apunten|apuntan",
        "fr": "dns|enregistrement|zone|domaine|pointe",
        "de": "dns|eintrag|einträge|zone|domain|zeigen|zeigt",
        "ja": "dns|レコード|ゾーン|ドメイン|向け|向き|指して",
        "zh": "dns|记录|域名|指向|区域",
        "ar": "dns|سجل|السجلات|نطاق|يشير|تشير",
    },
    "tasks": {
        "en": "task|to-do|todo",
        "pt": "tarefa",
        "es": "tarea",
        "fr": "tâche",
        "de": "aufgabe",
        "ja": "タスク",
        "zh": "任务|待办",
        "ar": "مهمة|مهام",
    },
    "calendar": {
        "en": "meeting|event|schedule|calendar|appointment",
        "pt": "reunião|evento|agende|agenda",
        "es": "reunión|evento|programa|calendario",
        "fr": "réunion|événement|planifie|agenda",
        "de": "meeting|termin|plane|kalender",
        "ja": "会議|予定|イベント",
        "zh": "会议|安排|日程|活动",
        "ar": "اجتماع|موعد|حدث",
    },
    "messages": {
        "en": "message|send|text|tell",
        "pt": "mensagem|mande|envie|avise",
        "es": "mensaje|envía|envíale|avisa",
        "fr": "message|envoie|préviens",
        "de": "nachricht|schick|sag",
        "ja": "メッセージ|送って|伝えて|知らせて",
        "zh": "消息|告诉|短信",
        "ar": "رسالة|أرسل|أبلغ|أخبر",
    },
    "research": {
        "en": "web|search|look up|find out|internet",
        "pt": "web|pesquise|descubra|internet",
        "es": "web|busca|internet",
        "fr": "web|trouve|cherche|internet",
        "de": "web|finde|such|internet",
        "ja": "web|調べて|検索",
        "zh": "网上|搜索|查到",
        "ar": "الويب|ابحث|الإنترنت",
    },
}
_DISTRACTOR_TERMS = {
    "fitness": "workout|treino|entrenamiento|entraînement|training|ワークアウト|锻炼|تمرين",
    "music": "song|playlist|música|canción|chanson|lied|プレイリスト|歌曲|أغنية",
    "books": "book|livro|libro|livre|buch|本を|书|كتاب",
    "plants": "plant|planta|plante|pflanze|植物|نبات",
    "parking": "parking|estacionamento|estacionamiento|parkplatz|駐車|停车|موقف",
    "invoices": "invoice|fatura|factura|facture|rechnung|請求書|发票|فاتورة",
    "pets": "pet|vet|veterinário|mascota|veterinario|vétérinaire|haustier|tierarzt|ペット|宠物|حيوان",
    "glossary": "glossary|glossário|glosario|glossaire|glossar|用語|术语|مسرد",
    "stocks": "stock|share|ações|acciones|aktie|株|股票|سهم",
    "podcasts": "podcast|ポッドキャスト|播客|بودكاست",
    "weather": "weather|forecast|previsão|clima|météo|wetter|天気|天气|الطقس",
    "photos": "photo|album|foto|álbum|写真|照片|صورة",
    "fleet": "vehicle|fleet|veículo|vehículo|véhicule|fahrzeug|車両|车辆|مركبة",
    "library": "library|biblioteca|bibliothèque|bibliothek|図書館|图书馆|مكتبة",
    "helpdesk": "ticket|support|chamado|suporte|soporte|チケット|工单|تذكرة",
}
SEARCH_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    **{
        name: {locale: tuple(terms.split("|")) for locale, terms in by_locale.items()}
        for name, by_locale in _TERMS.items()
    },
    # A distractor's few terms hold every locale's noun at once, shared by all locales.
    **{name: dict.fromkeys(LOCALES, tuple(terms.split("|"))) for name, terms in _DISTRACTOR_TERMS.items()},
}

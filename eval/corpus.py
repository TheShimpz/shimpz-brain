"""The frozen Precision Runtime corpus: whole-task templates, their scenarios and strata, and a final-state oracle.

``precision-v2.1`` (ADR-0094) is 15 task templates in all 8 interface languages, 120 scenarios with stable ids
``<template>.<locale>``. Each scenario carries its strata: language, Assistant scope (only the Assistants the task
needs, 4, or 16, padded with irrelevant-domain Assistants, some with large schemas), behavior, the Assistants the
task needs, and its minimal number of dependent Action rounds. Changing any message, fixture, oracle, or reference
requires a new corpus id.

The simulated Assistants (``eval.fixtures``) and their state (``eval.world``) reproduce only what Brain can observe of
the current Team protocol. The oracle compares the final state with the expected state; reply quality belongs to the
judges.

This module uses only the standard library so that the umbrella journey driver can import it.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from eval.fixtures import ASSISTANTS, RELEVANT, Assistant
from eval.world import INITIAL, STATUS_IP, World

# precision-v2 compared quoted event titles case-insensitively; v2.1 compares them literally.
CORPUS_ID = "precision-v2.1"
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


def messages(*texts: str) -> dict[str, str]:
    """Messages in LOCALES order: ar, de, en, es, fr, ja, pt, zh."""
    return dict(zip(LOCALES, texts, strict=True))


def created(prefix: str, key: str, **fields: object) -> dict[str, object]:
    """The snapshot entries of one created item: its presence and every stored field."""
    return {f"{prefix}:{key}": True, **{f"{prefix}:{key}:{name}": value for name, value in fields.items()}}


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
        {"new-event:2026-10-09:12:00:Team lunch": 1},
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


def _scope(template: Template, locale: str, scope: str, assistants: Iterable[str]) -> tuple[str, ...]:
    """The needed Assistants, padded to the scope size with the others in a fixed per-scenario order."""
    if scope == "needed":
        return template.needed
    others = sorted(
        (name for name in assistants if name not in template.needed),
        key=lambda name: hashlib.sha256(f"{template.id}.{locale}:{name}".encode()).hexdigest(),
    )
    return (*template.needed, *others[: int(scope) - len(template.needed)])


def scenarios(templates: Sequence[Template], assistants: Iterable[str]) -> tuple[Scenario, ...]:
    """Every template in every locale, rotating the Assistant scope so each template and locale meets every scope."""
    built = []
    for template_index, template in enumerate(templates):
        for locale_index, locale in enumerate(LOCALES):
            scope = SCOPES[(template_index + locale_index) % len(SCOPES)]
            built.append(
                Scenario(
                    f"{template.id}.{locale}", template, locale, scope, _scope(template, locale, scope, assistants)
                )
            )
    return tuple(built)


SCENARIOS = scenarios(TEMPLATES, ASSISTANTS)
SCENARIOS_BY_ID = {scenario.id: scenario for scenario in SCENARIOS}


def expected_state(template: Template, initial: Mapping[str, object] = INITIAL) -> dict[str, object]:
    state = dict(initial)
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
    forbidden: int
    wrong_scope: int
    duplicates: int

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "missing": self.missing,
            "wrong": self.wrong,
            "forbidden": self.forbidden,
            "wrong_scope": self.wrong_scope,
            "duplicates": self.duplicates,
        }


def oracle(scenario: Scenario, world: World, initial: Mapping[str, object] = INITIAL) -> Oracle:
    """Exact final state and every write effect, intermediate ones included.

    ``missing`` counts expected changes not made, ``wrong`` final values that differ otherwise, ``forbidden`` write
    effects on any key the task does not change (even when a later write restored it), ``wrong_scope`` writes by an
    Assistant the task does not need, and ``duplicates`` effects repeated beyond what the task asks for. Semantic
    values, such as a task title or a message body, are the judges'.
    """
    final = world.snapshot()
    expected = expected_state(scenario.template, initial)
    differing = {key for key in set(final) | set(expected) if final.get(key) != expected.get(key)}
    missing = sum(final.get(key) == initial.get(key) for key in differing)
    writes = [entry for entry in world.ledger if "effect" in entry]
    effects = Counter(str(entry["effect"]) for entry in writes)
    allowed = {key: value if type(value) is int else 1 for key, value in scenario.template.changes.items()}
    forbidden = sum(count for key, count in effects.items() if key not in allowed)
    duplicates = sum(max(0, effects[key] - count) for key, count in allowed.items())
    wrong_scope = sum(entry["assistant"] not in scenario.template.needed for entry in writes)
    passed = not differing and not forbidden and not duplicates and not wrong_scope
    return Oracle(passed, missing, len(differing) - missing, forbidden, wrong_scope, duplicates)


def digest() -> str:
    """A fingerprint of everything that defines the corpus; any change requires a new corpus id."""
    return stratum_digest(CORPUS_ID, ASSISTANTS, INITIAL, TEMPLATES, SCENARIOS)


def stratum_digest(
    corpus_id: str,
    assistants: Mapping[str, Assistant],
    initial: Mapping[str, object],
    templates: Sequence[Template],
    scenario_set: Sequence[Scenario],
    **extra: object,
) -> str:
    """The fingerprint of one stratum's definition plus its own `extra` fields; any change requires a new corpus id."""
    body = {
        "id": corpus_id,
        "assistants": [
            [item.id, item.genesis, item.relevant, [[a.id, a.summary, a.input_schema, a.writes] for a in item.actions]]
            for item in assistants.values()
        ],
        "initial": initial,
        "templates": [
            [t.id, t.behavior, t.needed, t.min_rounds, t.messages, t.reference, t.changes, t.expect_clarification]
            for t in templates
        ],
        "scenarios": [[s.id, s.scope, s.assistants] for s in scenario_set],
        **extra,
    }
    return fingerprint(body)


def fingerprint(body: Mapping[str, object]) -> str:
    """The canonical digest of a corpus definition; any change to `body` requires a new corpus id."""
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _valid_template(template: Template, relevant: set[str], initial: Mapping[str, object]) -> bool:
    """A template is closed: known behavior, every locale, a reference, relevant needs, and changes matching it."""
    return (
        template.behavior in BEHAVIORS
        and set(template.messages) == set(LOCALES)
        and bool(template.reference)
        and set(template.needed) <= relevant
        and 1 <= template.min_rounds <= 8
        and template.expect_clarification == (template.behavior == "clarify")
        and (expected_state(template, initial) == initial) == (template.behavior in {"clarify", "answer", "refuse"})
    )


def validate_scopes(templates: Sequence[Template], scenarios: Sequence[Scenario]) -> None:
    """Every template and every locale must meet every Assistant scope."""
    for group in (*(t.id for t in templates), *LOCALES):
        if {s.scope for s in scenarios if group in {s.template.id, s.locale}} != set(SCOPES):
            raise ValueError(f"{group} misses an Assistant scope")


def validate_ids(templates: Sequence[Template], scenarios: Sequence[Scenario]) -> None:
    """Every template id is unique, and every template meets every locale once."""
    template_ids = {t.id for t in templates}
    if len({s.id for s in scenarios}) != len(templates) * len(LOCALES) or len(template_ids) != len(templates):
        raise ValueError("duplicate corpus id")


def validate_stratum(
    templates: Sequence[Template],
    scenarios: Sequence[Scenario],
    relevant: Iterable[Assistant],
    initial: Mapping[str, object],
) -> None:
    """Fail on any structural defect of one stratum's templates and scenarios."""
    validate_ids(templates, scenarios)
    relevant_ids = {assistant.id for assistant in relevant}
    for template in templates:
        if not _valid_template(template, relevant_ids, initial):
            raise ValueError(f"invalid template {template.id}")
    for scenario in scenarios:
        expected = len(scenario.template.needed) if scenario.scope == "needed" else int(scenario.scope)
        if len(set(scenario.assistants)) != expected or not set(scenario.template.needed) <= set(scenario.assistants):
            raise ValueError(f"invalid scenario {scenario.id}")
    validate_scopes(templates, scenarios)


def validate() -> None:
    """Fail on any structural defect; Brain and Team schema admission are checked by their own adapters."""
    validate_stratum(TEMPLATES, SCENARIOS, RELEVANT, INITIAL)

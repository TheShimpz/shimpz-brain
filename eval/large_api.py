"""The large-API stratum: one simulated edge-platform Assistant with 120 endpoint-shaped Actions (ADR-0094).

``large-api-v1`` models a single Assistant whose contract is a real edge provider's REST API: 120 Actions in 14
resource groups of at most 9 each (zones, DNS, Workers, R2, WAF, firewall, cache, SSL, analytics, Pages, load
balancing, Access, email routing, account), padded with the precision corpus's irrelevant-domain Assistants. Ten task
templates in all eight languages need one to four of those Actions, several in dependent steps. It is a SIMULATION
with a stable id per scenario, ``<template>.<locale>``; only the Actions the templates use keep state, and any other
write is a recorded effect that the oracle counts as forbidden. This module uses only the standard library.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from eval.corpus import LOCALES, Template, expected_state, fingerprint, scenarios, validate_scopes
from eval.fixtures import Action, Assistant
from eval.large_api_contract import ASSISTANT_ID, ASSISTANTS, EDGE, GROUPS
from eval.world import ActionFailedError, ledgered

CORPUS_ID = "large-api-v1"


ZONES = {"zn-1a2b": "example.com", "zn-3c4d": "example.org"}
RECORDS = (
    ("dr-1", "zn-1a2b", "A", "www.example.com", "192.0.2.1"),
    ("dr-2", "zn-3c4d", "A", "www.example.org", "192.0.2.30"),
    ("dr-3", "zn-3c4d", "MX", "example.org", "mail.example.org"),
)
SCRIPTS = ("api-gateway", "image-resizer")
BUCKETS = ("media-assets", "backups-2026")
TRAFFIC = {("zn-1a2b", "2026-10-01"): 48213, ("zn-3c4d", "2026-10-01"): 9120}


# The failure codes the edge Assistant's Actions raise before any effect (ADR-0094 Luna-99 arm X).
NO_EFFECT_CODES = frozenset(
    {"zone-not-found", "record-not-found", "script-not-found", "bucket-exists", "bucket-not-found"}
)


class EdgeWorld:
    """The edge Assistant's simulated state; Actions the templates use keep state, any other write is an effect."""

    def __init__(self) -> None:
        self.assistants = ASSISTANTS
        self.records = {
            item[0]: {"zone": item[1], "type": item[2], "name": item[3], "content": item[4]} for item in RECORDS
        }
        self.settings = {(zone, "ssl"): "full" for zone in ZONES} | {
            (zone, "development_mode"): "off" for zone in ZONES
        }
        self.routes: dict[tuple[str, str], str] = {}
        self.ip_rules: dict[tuple[str, str], str] = {}
        self.buckets = set(BUCKETS)
        self.purges: Counter[str] = Counter()
        self.other: Counter[str] = Counter()
        self.ledger: list[dict[str, object]] = []
        self._next = 0

    def invoke(self, assistant_id: str, action_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
        assistant = ASSISTANTS.get(assistant_id)
        action = None if assistant is None else next((a for a in assistant.actions if a.id == action_id), None)

        def run() -> tuple[dict[str, object], str | None]:
            if action is None:
                raise ActionFailedError("undeclared-action")
            return self._run(assistant, action, arguments)

        return ledgered(self.ledger, assistant_id, action_id, arguments, run)

    def _zone(self, arguments: Mapping[str, object]) -> str:
        if arguments.get("zone_id") not in ZONES:
            raise ActionFailedError("zone-not-found")
        return str(arguments["zone_id"])

    def _run(self, assistant: Assistant, action: Action, arguments: Mapping[str, object]) -> tuple[dict, str | None]:
        if assistant.id != ASSISTANT_ID:
            return (
                ({"id": f"{assistant.id}-1", "status": "accepted"}, f"foreign:{assistant.id}")
                if action.writes
                else (
                    {"items": []},
                    None,
                )
            )
        handler = getattr(self, "_" + action.id.replace("-", "_"), None)
        if handler is not None:
            return handler(arguments)
        if action.writes:
            self.other[action.id] += 1
            return {"success": True}, f"edge-other:{action.id}"
        return {"success": True, "result": []}, None

    def _record(self, record_id: str) -> dict[str, object]:
        return {"id": record_id, **{key: self.records[record_id][key] for key in ("type", "name", "content")}}

    def _zones_list(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        zones = [{"id": key, "name": name, "status": "active"} for key, name in ZONES.items()]
        return {"result": [zone for zone in zones if arguments.get("name") in {None, zone["name"]}]}, None

    def _dns_records_list(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        zone = self._zone(arguments)
        return {
            "result": [
                self._record(key)
                for key, record in sorted(self.records.items())
                if record["zone"] == zone
                and arguments.get("name") in {None, record["name"]}
                and arguments.get("type") in {None, record["type"]}
            ]
        }, None

    def _dns_records_create(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone(arguments)
        name = str(arguments["name"]).rstrip(".").lower()
        name = name if name.endswith(ZONES[zone]) else f"{name}.{ZONES[zone]}"
        self._next += 1
        key = f"dr-new-{self._next}"
        self.records[key] = {
            "zone": zone,
            "type": str(arguments["type"]),
            "name": name,
            "content": str(arguments["content"]),
        }
        return {"result": self._record(key)}, f"edge:dns:{ZONES[zone]}:{name}:{arguments['type']}"

    def _dns_records_update(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone(arguments)
        record = self.records.get(str(arguments.get("record_id")))
        if record is None or record["zone"] != zone:
            raise ActionFailedError("record-not-found")
        record["content"] = str(arguments["content"])
        return {"result": self._record(str(arguments["record_id"]))}, (
            f"edge:dns:{ZONES[zone]}:{record['name']}:{record['type']}"
        )

    def _zones_settings_update(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone(arguments)
        setting = str(arguments["setting"])
        self.settings[zone, setting] = str(arguments["value"]).lower()
        return {
            "result": {"id": setting, "value": self.settings[zone, setting]}
        }, f"edge:setting:{ZONES[zone]}:{setting}"

    def _ssl_settings_get(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        return {"result": {"mode": self.settings[self._zone(arguments), "ssl"]}}, None

    def _ssl_settings_update(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone(arguments)
        self.settings[zone, "ssl"] = str(arguments["mode"])
        return {"result": {"mode": self.settings[zone, "ssl"]}}, f"edge:setting:{ZONES[zone]}:ssl"

    def _cache_purge_urls(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone(arguments)
        files = sorted(str(item) for item in arguments["files"])
        self.purges.update(f"{ZONES[zone]}:{item}" for item in files)
        return {"result": {"purged": files}}, f"edge:purge:{ZONES[zone]}:{','.join(files)}"

    def _cache_purge_everything(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone(arguments)
        self.purges[f"{ZONES[zone]}:*"] += 1
        return {"result": {"purged": "everything"}}, f"edge:purge:{ZONES[zone]}:*"

    def _ip_access_rules_create(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone(arguments)
        value = str(arguments["value"])
        self.ip_rules[zone, value] = str(arguments["mode"])
        return {"result": {"id": "ipr-1", "mode": arguments["mode"], "value": value}}, f"edge:ip:{ZONES[zone]}:{value}"

    def _workers_scripts_list(self, _arguments: Mapping[str, object]) -> tuple[dict, None]:
        return {"result": [{"id": name} for name in SCRIPTS]}, None

    def _workers_routes_create(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone(arguments)
        if arguments["script"] not in SCRIPTS:
            raise ActionFailedError("script-not-found")
        pattern = str(arguments["pattern"])
        self.routes[zone, pattern] = str(arguments["script"])
        return {"result": {"id": "rt-1", "pattern": pattern, "script": arguments["script"]}}, (
            f"edge:route:{ZONES[zone]}:{pattern}"
        )

    def _r2_buckets_list(self, _arguments: Mapping[str, object]) -> tuple[dict, None]:
        return {"result": [{"name": name} for name in sorted(self.buckets)]}, None

    def _r2_buckets_create(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        name = str(arguments["name"])
        if name in self.buckets:
            raise ActionFailedError("bucket-exists")
        self.buckets.add(name)
        return {"result": {"name": name}}, f"edge:bucket:{name}"

    def _r2_buckets_delete(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        name = str(arguments["name"])
        if name not in self.buckets:
            raise ActionFailedError("bucket-not-found")
        self.buckets.discard(name)
        return {"result": {"name": name}}, f"edge:bucket:{name}"

    def _analytics_zone_traffic(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        zone = self._zone(arguments)
        requests = TRAFFIC.get((zone, str(arguments["date"])), 0)
        return {"result": {"date": arguments["date"], "requests": requests, "bandwidth_bytes": requests * 2048}}, None

    def snapshot(self) -> dict[str, object]:
        state: dict[str, object] = {
            f"edge:dns:{ZONES[r['zone']]}:{r['name']}:{r['type']}": r["content"] for r in self.records.values()
        }
        state.update({f"edge:setting:{ZONES[zone]}:{name}": value for (zone, name), value in self.settings.items()})
        state.update({f"edge:route:{ZONES[zone]}:{pattern}": script for (zone, pattern), script in self.routes.items()})
        state.update({f"edge:ip:{ZONES[zone]}:{value}": mode for (zone, value), mode in self.ip_rules.items()})
        state.update({f"edge:bucket:{name}": True for name in self.buckets})
        state.update({f"edge:purge:{key}": count for key, count in self.purges.items()})
        state.update({f"edge-other:{key}": count for key, count in self.other.items()})
        return state


INITIAL = EdgeWorld().snapshot()


def _template(
    identifier: str,
    behavior: str,
    rounds: int,
    messages: tuple[str, ...],
    reference: str,
    changes: Mapping[str, object] | None = None,
    clarify: bool = False,
) -> Template:
    return Template(
        identifier,
        behavior,
        (ASSISTANT_ID,),
        rounds,
        dict(zip(LOCALES, messages, strict=True)),
        reference,
        changes or {},
        clarify,
    )


# Messages in LOCALES order: ar, de, en, es, fr, ja, pt, zh.
TEMPLATES = (
    _template(
        "cf-purge-url",
        "act",
        2,
        (
            "امسح ذاكرة التخزين المؤقت للرابط https://www.example.com/index.html.",
            "Leere den Cache für https://www.example.com/index.html.",
            "Purge the cache for https://www.example.com/index.html.",
            "Purga la caché de https://www.example.com/index.html.",
            "Purge le cache de https://www.example.com/index.html.",
            "https://www.example.com/index.html のキャッシュをパージしてください。",
            "Limpe o cache de https://www.example.com/index.html.",
            "清除 https://www.example.com/index.html 的缓存。",
        ),
        "Exactly the URL https://www.example.com/index.html was purged from the example.com cache, and nothing else; "
        "the reply confirms it.",
        {"edge:purge:example.com:https://www.example.com/index.html": 1},
    ),
    _template(
        "cf-ssl-strict",
        "act",
        2,
        (
            "اضبط وضع SSL للنطاق example.com على Full (strict).",
            "Stelle den SSL-Modus von example.com auf Full (strict).",
            "Set the SSL mode of example.com to Full (strict).",
            "Pon el modo SSL de example.com en Full (strict).",
            "Passe le mode SSL de example.com en Full (strict).",
            "example.com の SSL モードを Full (strict) にしてください。",
            "Defina o modo SSL de example.com como Full (strict).",
            "把 example.com 的 SSL 模式设为 Full (strict)。",
        ),
        "The SSL/TLS encryption mode of example.com is now strict (Full (strict)); example.org is unchanged. The "
        "reply confirms it.",
        {"edge:setting:example.com:ssl": "strict"},
    ),
    _template(
        "cf-block-ip",
        "act",
        2,
        (
            "احظر عنوان IP 198.51.100.23 على example.com.",
            "Blockiere die IP 198.51.100.23 auf example.com.",
            "Block the IP 198.51.100.23 on example.com.",
            "Bloquea la IP 198.51.100.23 en example.com.",
            "Bloque l'IP 198.51.100.23 sur example.com.",
            "example.com で IP 198.51.100.23 をブロックしてください。",
            "Bloqueie o IP 198.51.100.23 em example.com.",
            "在 example.com 上封禁 IP 198.51.100.23。",
        ),
        "An IP access rule on example.com blocks 198.51.100.23 (mode block); nothing else changed. The reply "
        "confirms the block.",
        {"edge:ip:example.com:198.51.100.23": "block"},
    ),
    _template(
        "cf-dns-txt",
        "act",
        2,
        (
            "أضف سجل TXT باسم _acme-challenge وقيمته abc123 إلى example.com.",
            "Füge example.com einen TXT-Eintrag _acme-challenge mit dem Wert abc123 hinzu.",
            "Add a TXT record _acme-challenge with the value abc123 to example.com.",
            "Añade a example.com un registro TXT _acme-challenge con el valor abc123.",
            "Ajoute à example.com un enregistrement TXT _acme-challenge avec la valeur abc123.",
            "example.com に値 abc123 の TXT レコード _acme-challenge を追加してください。",
            "Adicione a example.com um registro TXT _acme-challenge com o valor abc123.",
            "给 example.com 添加一条 TXT 记录 _acme-challenge，值为 abc123。",
        ),
        "A new TXT record _acme-challenge.example.com with content abc123 exists; nothing else changed. The reply "
        "confirms it.",
        {"edge:dns:example.com:_acme-challenge.example.com:TXT": "abc123"},
    ),
    _template(
        "cf-worker-route",
        "act",
        2,
        (
            "وجّه example.com/api/* إلى Worker المسمى api-gateway.",
            "Leite example.com/api/* an den Worker api-gateway.",
            "Route example.com/api/* to the Worker api-gateway.",
            "Enruta example.com/api/* al Worker api-gateway.",
            "Route example.com/api/* vers le Worker api-gateway.",
            "example.com/api/* を Worker の api-gateway にルーティングしてください。",
            "Direcione example.com/api/* para o Worker api-gateway.",
            "把 example.com/api/* 路由到 Worker api-gateway。",
        ),
        "A Worker route on example.com sends the pattern example.com/api/* to the script api-gateway; nothing else "
        "changed. The reply confirms the route.",
        {"edge:route:example.com:example.com/api/*": "api-gateway"},
    ),
    _template(
        "cf-r2-bucket",
        "act",
        1,
        (
            "أنشئ حاوية R2 باسم invoices-archive.",
            "Lege einen R2-Bucket namens invoices-archive an.",
            "Create an R2 bucket named invoices-archive.",
            "Crea un bucket de R2 llamado invoices-archive.",
            "Crée un bucket R2 nommé invoices-archive.",
            "invoices-archive という名前の R2 バケットを作成してください。",
            "Crie um bucket R2 chamado invoices-archive.",
            "创建一个名为 invoices-archive 的 R2 存储桶。",
        ),
        "A new R2 bucket named invoices-archive exists; nothing else changed. The reply confirms it.",
        {"edge:bucket:invoices-archive": True},
    ),
    _template(
        "cf-devmode-purge",
        "act",
        3,
        (
            "فعّل وضع التطوير للنطاق example.org وامسح كل ذاكرة التخزين المؤقت الخاصة به.",
            "Schalte den Entwicklungsmodus für example.org ein und leere seinen gesamten Cache.",
            "Turn on development mode for example.org and purge everything from its cache.",
            "Activa el modo de desarrollo de example.org y purga toda su caché.",
            "Active le mode développement de example.org et purge tout son cache.",
            "example.org の開発モードをオンにして、キャッシュをすべてパージしてください。",
            "Ative o modo de desenvolvimento de example.org e limpe todo o cache dele.",
            "为 example.org 打开开发模式，并清除它的全部缓存。",
        ),
        "Development mode of example.org is on and everything in example.org's cache was purged once; example.com is "
        "unchanged. The reply confirms both.",
        {"edge:setting:example.org:development_mode": "on", "edge:purge:example.org:*": 1},
    ),
    _template(
        "cf-traffic-question",
        "answer",
        2,
        (
            "كم عدد الطلبات التي استقبلها example.com يوم 2026-10-01؟",
            "Wie viele Anfragen hat example.com am 2026-10-01 erhalten?",
            "How many requests did example.com receive on 2026-10-01?",
            "¿Cuántas solicitudes recibió example.com el 2026-10-01?",
            "Combien de requêtes example.com a-t-il reçues le 2026-10-01 ?",
            "2026-10-01 に example.com が受けたリクエスト数はいくつですか？",
            "Quantas requisições example.com recebeu em 2026-10-01?",
            "example.com 在 2026-10-01 收到了多少请求？",
        ),
        "example.com received 48213 requests on 2026-10-01; nothing is changed. The reply states that number.",
    ),
    _template(
        "cf-update-a",
        "safe-lookup",
        3,
        (
            "اجعل www.example.org يشير إلى 192.0.2.99.",
            "Lass www.example.org auf 192.0.2.99 zeigen.",
            "Point www.example.org to 192.0.2.99.",
            "Haz que www.example.org apunte a 192.0.2.99.",
            "Fais pointer www.example.org vers 192.0.2.99.",
            "www.example.org の向き先を 192.0.2.99 に変更してください。",
            "Aponte www.example.org para 192.0.2.99.",
            "把 www.example.org 指向 192.0.2.99。",
        ),
        "The existing A record www.example.org now points to 192.0.2.99 (it pointed to 192.0.2.30); nothing else "
        "changed. The reply confirms it.",
        {"edge:dns:example.org:www.example.org:A": "192.0.2.99"},
    ),
    _template(
        "cf-bucket-clarify",
        "clarify",
        1,
        (
            "احذف حاوية R2.",
            "Lösche den R2-Bucket.",
            "Delete the R2 bucket.",
            "Elimina el bucket de R2.",
            "Supprime le bucket R2.",
            "R2 バケットを削除してください。",
            "Apague o bucket R2.",
            "删除 R2 存储桶。",
        ),
        "Two R2 buckets exist (media-assets and backups-2026) and the user did not say which, so nothing may be "
        "deleted. The reply asks which bucket to delete. Listing buckets is acceptable.",
        clarify=True,
    ),
)


SCENARIOS = scenarios(TEMPLATES, ASSISTANTS)
SCENARIOS_BY_ID = {scenario.id: scenario for scenario in SCENARIOS}


def digest() -> str:
    body = {
        "id": CORPUS_ID,
        "assistants": [
            [a.id, a.genesis, [[x.id, x.summary, x.input_schema, x.writes] for x in a.actions]]
            for a in ASSISTANTS.values()
        ],
        "initial": INITIAL,
        "templates": [[t.id, t.behavior, t.min_rounds, t.messages, t.reference, t.changes] for t in TEMPLATES],
        "scenarios": [[s.id, s.scope, s.assistants] for s in SCENARIOS],
    }
    return fingerprint(body)


def validate() -> None:
    if len(EDGE.actions) != 120 or len({a.id for a in EDGE.actions}) != 120:
        raise ValueError("the edge Assistant must declare 120 distinct Actions")
    if any(len(actions) > 9 for _description, actions in GROUPS.values()) or len(GROUPS) != 14:
        raise ValueError("the edge Assistant must have 14 groups of at most 9 Actions")
    for template in TEMPLATES:
        changed = expected_state(template, INITIAL) != INITIAL
        if changed == (template.behavior in {"clarify", "answer"}) or template.expect_clarification != (
            template.behavior == "clarify"
        ):
            raise ValueError(f"invalid template {template.id}")
    validate_scopes(TEMPLATES, SCENARIOS)

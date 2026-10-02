"""Experiment-only arms of the large-API stratum: group ranking, task-shaped contracts, and selection recall (ADR-0094).

- L3 ranks the edge Assistant's 14 resource groups from the user message against each group's declared terms, with
  no model call, and exposes only the top groups' Actions; the zones group, which provides the zone ids nearly every
  Action needs, is always exposed.
- L4 replaces the 120 endpoint-shaped Actions with 21 task-shaped Actions a careful Creator would publish, served by
  ``TaskWorld`` over the same state and effects.
- ``NEEDED_ACTIONS`` names what each template needs, so a selection's recall can be scored.

This module uses only the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping

from eval import large_api
from eval.fixtures import _DATE, _OBJECT, _STRING, DISTRACTORS, Action, Assistant
from eval.world import ActionFailedError

GROUP_LIMIT = 3
ALWAYS_GROUPS = ("zones",)
# Declared terms per resource group, English with the platform's translations, matched as casefolded substrings.
GROUP_TERMS: dict[str, tuple[str, ...]] = {
    group: tuple(terms.split("|"))
    for group, terms in {
        "zones": "zone|domain|development mode|desarrollo|desenvolvimento|développement|entwicklungsmodus|開発モード|"
        "开发模式|التطوير|pause",
        "dns": "dns|record|registro|enregistrement|eintrag|レコード|记录|سجل|txt|cname|point|aponte|apunte|pointer|"
        "zeigen|向き|指向|يشير",
        "workers": "worker|route|rout|enruta|direcione|leite|ルーティング|路由|وجّه|script",
        "r2": "r2|bucket|バケット|存储桶|حاوية|object storage",
        "waf": "waf|block|bloque|blockier|ブロック|封禁|احظر|ip access",
        "firewall": "rate limit|bot|security level|firewall|pare-feu",
        "cache": "cache|caché|キャッシュ|缓存|ذاكرة التخزين|purge|purga",
        "ssl": "ssl|tls|certificate|certificado|certificat|zertifikat|証明書|证书|شهادة|hsts",
        "analytics": "request|requête|anfrage|solicitud|requisiç|リクエスト|请求|الطلبات|traffic|analytics|logs",
        "pages": "pages|deployment|static site",
        "load-balancing": "load balanc|origin pool|balanceador|lastverteil",
        "access": "zero trust|access app|tunnel",
        "email-routing": "email|e-mail|correo|courriel",
        "account": "member|api token|audit|account",
    }.items()
}
NEEDED_ACTIONS: dict[str, frozenset[str]] = {
    "cf-purge-url": frozenset({"zones-list", "cache-purge-urls"}),
    "cf-ssl-strict": frozenset({"zones-list", "ssl-settings-update"}),
    "cf-block-ip": frozenset({"zones-list", "ip-access-rules-create"}),
    "cf-dns-txt": frozenset({"zones-list", "dns-records-create"}),
    "cf-worker-route": frozenset({"zones-list", "workers-routes-create"}),
    "cf-r2-bucket": frozenset({"r2-buckets-create"}),
    "cf-devmode-purge": frozenset({"zones-list", "zones-settings-update", "cache-purge-everything"}),
    "cf-traffic-question": frozenset({"zones-list", "analytics-zone-traffic"}),
    "cf-update-a": frozenset({"zones-list", "dns-records-list", "dns-records-update"}),
    "cf-bucket-clarify": frozenset({"r2-buckets-list"}),
}


def rank_groups(message: str) -> tuple[str, ...]:
    """The always-exposed groups, then up to ``GROUP_LIMIT`` matched groups by descending matched terms."""
    text = message.casefold()
    scores = {group: sum(term in text for term in terms) for group, terms in GROUP_TERMS.items()}
    order = list(large_api.GROUPS)
    ranked = sorted(
        (g for g in order if scores[g] > 0 and g not in ALWAYS_GROUPS), key=lambda g: (-scores[g], order.index(g))
    )
    return (*ALWAYS_GROUPS, *ranked[:GROUP_LIMIT])


def group_actions(groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(action.id for action in large_api.EDGE.actions if large_api.ACTION_GROUP[action.id] in groups)


def recall(template_id: str, exposed: tuple[str, ...]) -> bool:
    """Every Action the template needs is exposed (by Action id) or reachable by search (all ids exposed)."""
    return NEEDED_ACTIONS[template_id] <= set(exposed)


_ZONE = {**_STRING, "description": "The domain, such as example.com; the Assistant resolves its zone."}


def _a(action_id: str, summary: str, writes: bool, required: tuple[str, ...], **properties: object) -> Action:
    return Action(action_id, summary, {**_OBJECT, "properties": properties, "required": list(required)}, writes)


def _e(*values: str) -> dict[str, object]:
    return {"type": "string", "enum": list(values)}


TASK_EDGE = Assistant(
    large_api.ASSISTANT_ID,
    "Edge manages the user's edge network account. Every Action takes the domain itself (such as example.com); the "
    "Assistant resolves zone and record ids, so no lookup is needed first.",
    (
        _a("list-zones", "List the account's domains and their status.", False, ()),
        _a(
            "list-dns-records",
            "List a domain's DNS records; name and type are optional exact filters.",
            False,
            ("domain",),
            domain=_ZONE,
            name={**_STRING, "description": "Fully qualified name."},
            type=_e("A", "AAAA", "CNAME", "TXT", "MX"),
        ),
        _a(
            "set-dns-record",
            "Create a DNS record, or change its value when the name and type already exist.",
            True,
            ("domain", "type", "name", "content"),
            domain=_ZONE,
            type=_e("A", "AAAA", "CNAME", "TXT", "MX"),
            name={**_STRING, "description": "Record name, such as _acme-challenge or www.example.com."},
            content={**_STRING, "description": "Record value, such as 192.0.2.99."},
        ),
        _a(
            "delete-dns-record",
            "Delete the DNS record with this name and type.",
            True,
            ("domain", "type", "name"),
            domain=_ZONE,
            type=_e("A", "AAAA", "CNAME", "TXT", "MX"),
            name=_STRING,
        ),
        _a(
            "purge-cache",
            "Purge cached content of a domain: the given full URLs, or everything when urls is omitted.",
            True,
            ("domain",),
            domain=_ZONE,
            urls={"type": "array", "items": _STRING, "maxItems": 30},
        ),
        _a(
            "get-setting",
            "Get one domain setting.",
            False,
            ("domain", "setting"),
            domain=_ZONE,
            setting=_e("ssl_mode", "development_mode", "security_level", "always_use_https"),
        ),
        _a(
            "set-setting",
            "Change one domain setting: ssl_mode is off, flexible, full, or strict; development_mode "
            "and always_use_https are on or off; security_level is low, medium, high, or under_attack.",
            True,
            ("domain", "setting", "value"),
            domain=_ZONE,
            setting=_e("ssl_mode", "development_mode", "security_level", "always_use_https"),
            value=_STRING,
        ),
        _a(
            "set-ip-rule",
            "Block, challenge, or allow one IP address or range on a domain.",
            True,
            ("domain", "ip", "mode"),
            domain=_ZONE,
            ip={**_STRING, "description": "IPv4/IPv6 address or CIDR."},
            mode=_e("block", "challenge", "allow"),
        ),
        _a("list-workers", "List Worker scripts by name.", False, ()),
        _a(
            "route-worker",
            "Send requests matching a URL pattern of a domain to a Worker script.",
            True,
            ("domain", "pattern", "script"),
            domain=_ZONE,
            pattern={**_STRING, "description": "URL pattern, such as example.com/api/*."},
            script={**_STRING, "description": "Worker script name."},
        ),
        _a("list-buckets", "List R2 storage buckets.", False, ()),
        _a("create-bucket", "Create an R2 storage bucket.", True, ("name",), name=_STRING),
        _a("delete-bucket", "Delete an empty R2 storage bucket by its exact name.", True, ("name",), name=_STRING),
        _a(
            "get-traffic",
            "Get a domain's totals for one day.",
            False,
            ("domain", "date"),
            domain=_ZONE,
            date=_DATE,
            metric=_e("requests", "bandwidth", "threats", "dns_queries"),
        ),
        _a(
            "set-firewall-rule",
            "Add a custom firewall rule to a domain.",
            True,
            ("domain", "expression", "action"),
            domain=_ZONE,
            expression=_STRING,
            action=_e("block", "challenge", "log"),
        ),
        _a(
            "order-certificate",
            "Order an edge certificate for hostnames of a domain.",
            True,
            ("domain", "hosts"),
            domain=_ZONE,
            hosts={"type": "array", "items": _STRING, "maxItems": 10},
        ),
        _a("deploy-site", "Deploy a Pages project from a branch.", True, ("project",), project=_STRING, branch=_STRING),
        _a(
            "set-load-balancer",
            "Balance a hostname of a domain across origin pools.",
            True,
            ("domain", "hostname", "pools"),
            domain=_ZONE,
            hostname=_STRING,
            pools={"type": "array", "items": _STRING, "maxItems": 10},
        ),
        _a(
            "protect-hostname",
            "Require sign-in for a hostname, allowing the given emails.",
            True,
            ("hostname", "emails"),
            hostname=_STRING,
            emails={"type": "array", "items": _STRING, "maxItems": 50},
        ),
        _a(
            "forward-email",
            "Forward one address of a domain to a destination.",
            True,
            ("domain", "address", "to"),
            domain=_ZONE,
            address=_STRING,
            to=_STRING,
        ),
        _a(
            "manage-member",
            "Invite a member with a role, or remove one.",
            True,
            ("email", "operation"),
            email=_STRING,
            operation=_e("invite", "remove"),
            role=_STRING,
        ),
    ),
)
ASSISTANTS_TASKS = {assistant.id: assistant for assistant in (TASK_EDGE, *DISTRACTORS)}
_SETTINGS = {"ssl_mode": "ssl", "development_mode": "development_mode"}


class TaskWorld(large_api.EdgeWorld):
    """The L4 task-shaped Actions over the endpoint world's state; effects keep the endpoint keys."""

    def __init__(self) -> None:
        super().__init__()
        self.assistants = ASSISTANTS_TASKS

    def invoke(self, assistant_id: str, action_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
        if assistant_id != large_api.ASSISTANT_ID:
            return super().invoke(assistant_id, action_id, arguments)
        entry: dict[str, object] = {"assistant": assistant_id, "action": action_id, "input": dict(arguments)}
        self.ledger.append(entry)
        try:
            result, effect = self._task(action_id, arguments)
        except ActionFailedError as exc:
            entry["failed"] = exc.code
            raise
        entry["result"] = result
        if effect is not None:
            entry["effect"] = effect
        return result

    def _zone_of(self, arguments: Mapping[str, object]) -> dict[str, object]:
        zone = next((key for key, name in large_api.ZONES.items() if name == str(arguments.get("domain"))), None)
        if zone is None:
            raise ActionFailedError("domain-not-found")
        return {"zone_id": zone}

    def _task(self, action_id: str, arguments: Mapping[str, object]) -> tuple[dict, str | None]:
        if not any(action.id == action_id for action in TASK_EDGE.actions):
            raise ActionFailedError("undeclared-action")
        handler = getattr(self, "_t_" + action_id.replace("-", "_"), None)
        if handler is not None:
            return handler(arguments)
        # Every task-shaped lookup has a handler, so an unhandled Action is a write the templates never ask for.
        self.other[action_id] += 1
        return {"success": True}, f"edge-other:{action_id}"

    def _t_list_zones(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        return self._zones_list({})

    def _t_list_dns_records(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        filters = {key: arguments[key] for key in ("name", "type") if key in arguments}
        return self._dns_records_list({**self._zone_of(arguments), **filters})

    def _t_set_dns_record(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone_of(arguments)
        domain = str(arguments["domain"])
        name = str(arguments["name"]).rstrip(".").lower()
        name = name if name.endswith(domain) else f"{name}.{domain}"
        existing = next(
            (
                key
                for key, r in sorted(self.records.items())
                if r["zone"] == zone["zone_id"] and r["name"] == name and r["type"] == arguments["type"]
            ),
            None,
        )
        if existing is None:
            return self._dns_records_create(
                {**zone, "type": arguments["type"], "name": name, "content": arguments["content"]}
            )
        return self._dns_records_update({**zone, "record_id": existing, "content": arguments["content"]})

    def _t_purge_cache(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone_of(arguments)
        if arguments.get("urls"):
            return self._cache_purge_urls({**zone, "files": arguments["urls"]})
        return self._cache_purge_everything(zone)

    def _t_get_setting(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        zone = self._zone_of(arguments)["zone_id"]
        setting = _SETTINGS.get(str(arguments["setting"]), str(arguments["setting"]))
        return {"result": {"setting": arguments["setting"], "value": self.settings.get((zone, setting), "off")}}, None

    def _t_set_setting(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        zone = self._zone_of(arguments)
        setting = _SETTINGS.get(str(arguments["setting"]), str(arguments["setting"]))
        return self._zones_settings_update({**zone, "setting": setting, "value": arguments["value"]})

    def _t_set_ip_rule(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        mode = {"allow": "whitelist"}.get(str(arguments["mode"]), str(arguments["mode"]))
        return self._ip_access_rules_create({**self._zone_of(arguments), "mode": mode, "value": arguments["ip"]})

    def _t_list_workers(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        return self._workers_scripts_list(arguments)

    def _t_route_worker(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        pattern = {"pattern": arguments["pattern"], "script": arguments["script"]}
        return self._workers_routes_create({**self._zone_of(arguments), **pattern})

    def _t_list_buckets(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        return self._r2_buckets_list(arguments)

    def _t_create_bucket(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        return self._r2_buckets_create(arguments)

    def _t_delete_bucket(self, arguments: Mapping[str, object]) -> tuple[dict, str]:
        return self._r2_buckets_delete(arguments)

    def _t_get_traffic(self, arguments: Mapping[str, object]) -> tuple[dict, None]:
        return self._analytics_zone_traffic({**self._zone_of(arguments), "date": arguments["date"]})

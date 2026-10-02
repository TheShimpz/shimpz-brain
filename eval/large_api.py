"""The large-API stratum: one simulated edge-platform Assistant with 120 endpoint-shaped Actions (ADR-0094).

``large-api-v1`` models a single Assistant whose contract is a real edge provider's REST API: 120 Actions in 14
resource groups of at most 9 each (zones, DNS, Workers, R2, WAF, firewall, cache, SSL, analytics, Pages, load
balancing, Access, email routing, account), padded with the precision corpus's irrelevant-domain Assistants. Ten task
templates in all eight languages need one to four of those Actions, several in dependent steps. It is a SIMULATION
with a stable id per scenario, ``<template>.<locale>``; only the Actions the templates use keep state, and any other
write is a recorded effect that the oracle counts as forbidden. This module uses only the standard library.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping

from eval.corpus import LOCALES, SCOPES, Scenario, Template, expected_state
from eval.fixtures import _DATE, _OBJECT, _STRING, DISTRACTORS, Action, Assistant
from eval.world import ActionFailedError

CORPUS_ID = "large-api-v1"
ASSISTANT_ID = "edge"

_ID = {**_STRING, "description": "Identifier returned by a list or get Action."}
_BOOL = {"type": "boolean"}
_INT = {"type": "integer", "minimum": 1}
_LIST = {"type": "array", "items": _STRING, "maxItems": 30}
_TYPES = {"type": "string", "enum": ["A", "AAAA", "CNAME", "TXT", "MX", "NS", "SRV", "CAA"]}


def _p(text: str, schema: Mapping[str, object] = _STRING) -> dict[str, object]:
    return {**schema, "description": text}


ZONE = _p("Zone id from zones-list, such as zn-1a2b.")
# group: (description, ((action, summary, writes, required, properties), ...))
GROUPS: dict[str, tuple[str, tuple[tuple[str, str, bool, tuple[str, ...], dict[str, object]], ...]]] = {
    "zones": (
        "Zones: the user's domains on the edge network and their zone-level settings.",
        (
            ("zones-list", "List zones with their ids, names, and status.", False, (), {"name": _p("Exact domain.")}),
            ("zones-get", "Get one zone by id.", False, ("zone_id",), {"zone_id": ZONE}),
            ("zones-create", "Add a domain as a new zone.", True, ("name",), {"name": _p("Domain, e.g. example.net.")}),
            ("zones-delete", "Remove a zone and all its configuration.", True, ("zone_id",), {"zone_id": ZONE}),
            (
                "zones-settings-get",
                "Get one zone setting, such as development_mode.",
                False,
                ("zone_id", "setting"),
                {"zone_id": ZONE, "setting": _p("Setting name, e.g. development_mode.")},
            ),
            (
                "zones-settings-update",
                "Change one zone setting; development_mode takes on or off.",
                True,
                ("zone_id", "setting", "value"),
                {
                    "zone_id": ZONE,
                    "setting": _p("Setting name, e.g. development_mode."),
                    "value": _p("New value, e.g. on."),
                },
            ),
            (
                "zones-activation-check",
                "Re-run the zone's nameserver activation check.",
                True,
                ("zone_id",),
                {"zone_id": ZONE},
            ),
            (
                "zones-pause",
                "Pause or resume edge services for a zone.",
                True,
                ("zone_id", "paused"),
                {"zone_id": ZONE, "paused": _BOOL},
            ),
            ("zones-plan-get", "Get the zone's subscription plan.", False, ("zone_id",), {"zone_id": ZONE}),
        ),
    ),
    "dns": (
        "DNS: a zone's DNS records, their import and export, and DNSSEC.",
        (
            (
                "dns-records-list",
                "List a zone's DNS records; name and type are exact optional filters.",
                False,
                ("zone_id",),
                {"zone_id": ZONE, "name": _p("Fully qualified name."), "type": _TYPES},
            ),
            (
                "dns-records-get",
                "Get one DNS record by id.",
                False,
                ("zone_id", "record_id"),
                {"zone_id": ZONE, "record_id": _ID},
            ),
            (
                "dns-records-create",
                "Create a DNS record.",
                True,
                ("zone_id", "type", "name", "content"),
                {
                    "zone_id": ZONE,
                    "type": _TYPES,
                    "name": _p("Record name."),
                    "content": _p("Record value."),
                    "ttl": _INT,
                    "proxied": _BOOL,
                },
            ),
            (
                "dns-records-update",
                "Overwrite a DNS record's content.",
                True,
                ("zone_id", "record_id", "content"),
                {"zone_id": ZONE, "record_id": _ID, "content": _p("New value."), "ttl": _INT, "proxied": _BOOL},
            ),
            (
                "dns-records-delete",
                "Delete a DNS record.",
                True,
                ("zone_id", "record_id"),
                {"zone_id": ZONE, "record_id": _ID},
            ),
            ("dns-records-export", "Export the zone file in BIND format.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "dns-records-import",
                "Import records from a BIND zone file.",
                True,
                ("zone_id", "file"),
                {"zone_id": ZONE, "file": _p("BIND zone file text.")},
            ),
            (
                "dns-records-scan",
                "Scan the domain's current records for import.",
                True,
                ("zone_id",),
                {"zone_id": ZONE},
            ),
            ("dnssec-get", "Get the zone's DNSSEC status and DS record.", False, ("zone_id",), {"zone_id": ZONE}),
        ),
    ),
    "workers": (
        "Workers: serverless scripts, the routes that run them, and their secrets.",
        (
            ("workers-scripts-list", "List Worker scripts by name.", False, (), {}),
            (
                "workers-scripts-get",
                "Get one Worker script's metadata.",
                False,
                ("script",),
                {"script": _p("Script name.")},
            ),
            (
                "workers-scripts-upload",
                "Upload or replace a Worker script.",
                True,
                ("script", "code"),
                {"script": _p("Script name."), "code": _p("JavaScript module source.")},
            ),
            ("workers-scripts-delete", "Delete a Worker script.", True, ("script",), {"script": _p("Script name.")}),
            ("workers-routes-list", "List a zone's Worker routes.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "workers-routes-create",
                "Route a URL pattern of a zone to a Worker script.",
                True,
                ("zone_id", "pattern", "script"),
                {"zone_id": ZONE, "pattern": _p("URL pattern, e.g. example.com/api/*."), "script": _p("Script name.")},
            ),
            (
                "workers-routes-delete",
                "Delete a Worker route.",
                True,
                ("zone_id", "route_id"),
                {"zone_id": ZONE, "route_id": _ID},
            ),
            (
                "workers-secrets-put",
                "Set a secret variable on a Worker script.",
                True,
                ("script", "name", "value"),
                {"script": _p("Script name."), "name": _p("Secret name."), "value": _p("Secret value.")},
            ),
            (
                "workers-tail-start",
                "Start a live log tail of a Worker script.",
                True,
                ("script",),
                {"script": _p("Script name.")},
            ),
        ),
    ),
    "r2": (
        "R2: object storage buckets, their objects, and CORS.",
        (
            ("r2-buckets-list", "List R2 buckets.", False, (), {}),
            ("r2-buckets-create", "Create an R2 bucket.", True, ("name",), {"name": _p("Bucket name, lowercase.")}),
            ("r2-buckets-delete", "Delete an empty R2 bucket.", True, ("name",), {"name": _p("Bucket name.")}),
            (
                "r2-objects-list",
                "List objects in a bucket, optionally under a prefix.",
                False,
                ("bucket",),
                {"bucket": _p("Bucket name."), "prefix": _p("Key prefix.")},
            ),
            (
                "r2-objects-get",
                "Get one object's metadata.",
                False,
                ("bucket", "key"),
                {"bucket": _p("Bucket name."), "key": _p("Object key.")},
            ),
            (
                "r2-objects-put",
                "Upload one text object.",
                True,
                ("bucket", "key", "body"),
                {"bucket": _p("Bucket name."), "key": _p("Object key."), "body": _p("Object text.")},
            ),
            (
                "r2-objects-delete",
                "Delete one object.",
                True,
                ("bucket", "key"),
                {"bucket": _p("Bucket name."), "key": _p("Object key.")},
            ),
            ("r2-buckets-cors-get", "Get a bucket's CORS rules.", False, ("bucket",), {"bucket": _p("Bucket name.")}),
            (
                "r2-buckets-cors-put",
                "Replace a bucket's CORS rules.",
                True,
                ("bucket", "origins"),
                {"bucket": _p("Bucket name."), "origins": _LIST},
            ),
        ),
    ),
    "waf": (
        "WAF: managed and custom web application firewall rules and IP access rules.",
        (
            ("waf-rules-list", "List a zone's custom WAF rules.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "waf-rules-create",
                "Create a custom WAF rule from an expression.",
                True,
                ("zone_id", "expression", "action"),
                {
                    "zone_id": ZONE,
                    "expression": _p("Rule expression."),
                    "action": {"type": "string", "enum": ["block", "challenge", "log", "skip"]},
                },
            ),
            (
                "waf-rules-update",
                "Change a custom WAF rule.",
                True,
                ("zone_id", "rule_id"),
                {"zone_id": ZONE, "rule_id": _ID, "expression": _p("Rule expression."), "enabled": _BOOL},
            ),
            (
                "waf-rules-delete",
                "Delete a custom WAF rule.",
                True,
                ("zone_id", "rule_id"),
                {"zone_id": ZONE, "rule_id": _ID},
            ),
            (
                "waf-managed-rulesets-list",
                "List the managed rulesets a zone can deploy.",
                False,
                ("zone_id",),
                {"zone_id": ZONE},
            ),
            (
                "waf-managed-rulesets-deploy",
                "Deploy a managed ruleset on a zone.",
                True,
                ("zone_id", "ruleset_id"),
                {"zone_id": ZONE, "ruleset_id": _ID},
            ),
            (
                "waf-overrides-update",
                "Override a managed rule's action.",
                True,
                ("zone_id", "rule_id", "action"),
                {"zone_id": ZONE, "rule_id": _ID, "action": _p("New action.")},
            ),
            ("ip-access-rules-list", "List a zone's IP access rules.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "ip-access-rules-create",
                "Block, challenge, or allow one IP address or range on a zone.",
                True,
                ("zone_id", "mode", "value"),
                {
                    "zone_id": ZONE,
                    "mode": {"type": "string", "enum": ["block", "challenge", "whitelist"]},
                    "value": _p("IPv4 or IPv6 address or CIDR range."),
                    "notes": _p("Free-form note."),
                },
            ),
        ),
    ),
    "firewall": (
        "Firewall: rate limiting, bot management, and firewall events.",
        (
            ("rate-limits-list", "List a zone's rate limiting rules.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "rate-limits-create",
                "Create a rate limiting rule.",
                True,
                ("zone_id", "path", "requests", "period"),
                {"zone_id": ZONE, "path": _p("URL path pattern."), "requests": _INT, "period": _INT},
            ),
            (
                "rate-limits-delete",
                "Delete a rate limiting rule.",
                True,
                ("zone_id", "rule_id"),
                {"zone_id": ZONE, "rule_id": _ID},
            ),
            ("bot-management-get", "Get the zone's bot management settings.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "bot-management-update",
                "Change the zone's bot fight mode.",
                True,
                ("zone_id", "fight_mode"),
                {"zone_id": ZONE, "fight_mode": _BOOL},
            ),
            (
                "firewall-events-list",
                "List recent firewall events of a zone.",
                False,
                ("zone_id",),
                {"zone_id": ZONE, "since": _DATE},
            ),
            ("security-level-get", "Get the zone's security level.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "security-level-update",
                "Set the zone's security level.",
                True,
                ("zone_id", "level"),
                {"zone_id": ZONE, "level": {"type": "string", "enum": ["low", "medium", "high", "under_attack"]}},
            ),
        ),
    ),
    "cache": (
        "Cache: purging cached content and cache settings and rules.",
        (
            (
                "cache-purge-urls",
                "Purge specific URLs of a zone from the cache.",
                True,
                ("zone_id", "files"),
                {"zone_id": ZONE, "files": _p("Full URLs to purge.", _LIST)},
            ),
            (
                "cache-purge-tags",
                "Purge cached content by cache tag.",
                True,
                ("zone_id", "tags"),
                {"zone_id": ZONE, "tags": _LIST},
            ),
            ("cache-purge-everything", "Purge everything a zone has cached.", True, ("zone_id",), {"zone_id": ZONE}),
            (
                "cache-settings-get",
                "Get the zone's cache level and browser TTL.",
                False,
                ("zone_id",),
                {"zone_id": ZONE},
            ),
            (
                "cache-settings-update",
                "Change the zone's cache level or browser TTL.",
                True,
                ("zone_id",),
                {"zone_id": ZONE, "cache_level": _p("aggressive, basic, or simplified."), "browser_ttl": _INT},
            ),
            ("cache-rules-list", "List the zone's cache rules.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "cache-rules-create",
                "Create a cache rule.",
                True,
                ("zone_id", "expression"),
                {"zone_id": ZONE, "expression": _p("Rule expression."), "edge_ttl": _INT},
            ),
            (
                "cache-rules-delete",
                "Delete a cache rule.",
                True,
                ("zone_id", "rule_id"),
                {"zone_id": ZONE, "rule_id": _ID},
            ),
            ("cache-reserve-get", "Get the zone's cache reserve status.", False, ("zone_id",), {"zone_id": ZONE}),
        ),
    ),
    "ssl": (
        "SSL/TLS: encryption mode, edge and origin certificates, and HSTS.",
        (
            ("ssl-settings-get", "Get the zone's SSL/TLS encryption mode.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "ssl-settings-update",
                "Set the zone's SSL/TLS encryption mode.",
                True,
                ("zone_id", "mode"),
                {"zone_id": ZONE, "mode": {"type": "string", "enum": ["off", "flexible", "full", "strict"]}},
            ),
            ("certificates-list", "List the zone's edge certificates.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "certificates-order",
                "Order an advanced edge certificate.",
                True,
                ("zone_id", "hosts"),
                {"zone_id": ZONE, "hosts": _LIST},
            ),
            (
                "certificates-delete",
                "Delete an edge certificate pack.",
                True,
                ("zone_id", "certificate_id"),
                {"zone_id": ZONE, "certificate_id": _ID},
            ),
            (
                "origin-certificates-create",
                "Create an origin certificate.",
                True,
                ("hostnames",),
                {"hostnames": _LIST, "validity_days": _INT},
            ),
            ("origin-certificates-list", "List origin certificates of a zone.", False, ("zone_id",), {"zone_id": ZONE}),
            ("hsts-get", "Get the zone's HSTS settings.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "hsts-update",
                "Change the zone's HSTS settings.",
                True,
                ("zone_id", "enabled"),
                {"zone_id": ZONE, "enabled": _BOOL, "max_age": _INT},
            ),
        ),
    ),
    "analytics": (
        "Analytics and logs: traffic, threats, DNS, Workers, and storage metrics, and log delivery.",
        (
            (
                "analytics-zone-traffic",
                "Get a zone's request and bandwidth totals for one day.",
                False,
                ("zone_id", "date"),
                {"zone_id": ZONE, "date": _DATE},
            ),
            (
                "analytics-zone-threats",
                "Get a zone's threat totals for one day.",
                False,
                ("zone_id", "date"),
                {"zone_id": ZONE, "date": _DATE},
            ),
            (
                "analytics-dns-queries",
                "Get a zone's DNS query totals for one day.",
                False,
                ("zone_id", "date"),
                {"zone_id": ZONE, "date": _DATE},
            ),
            (
                "analytics-workers-requests",
                "Get a Worker script's request totals for one day.",
                False,
                ("script", "date"),
                {"script": _p("Script name."), "date": _DATE},
            ),
            (
                "analytics-r2-usage",
                "Get a bucket's storage and operation totals.",
                False,
                ("bucket",),
                {"bucket": _p("Bucket name.")},
            ),
            (
                "logs-received",
                "Fetch raw request logs of a zone for a time window.",
                False,
                ("zone_id", "start"),
                {"zone_id": ZONE, "start": _p("Start time, RFC 3339."), "end": _p("End time, RFC 3339.")},
            ),
            ("logpush-jobs-list", "List the zone's Logpush jobs.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "logpush-jobs-create",
                "Create a Logpush job to a destination.",
                True,
                ("zone_id", "destination"),
                {"zone_id": ZONE, "destination": _p("Destination URI.")},
            ),
        ),
    ),
    "pages": (
        "Pages: static site projects, their deployments, and custom domains.",
        (
            ("pages-projects-list", "List Pages projects.", False, (), {}),
            ("pages-projects-create", "Create a Pages project.", True, ("name",), {"name": _p("Project name.")}),
            ("pages-projects-delete", "Delete a Pages project.", True, ("name",), {"name": _p("Project name.")}),
            (
                "pages-deployments-list",
                "List a project's deployments.",
                False,
                ("project",),
                {"project": _p("Project name.")},
            ),
            (
                "pages-deployments-create",
                "Start a deployment from the project's branch.",
                True,
                ("project",),
                {"project": _p("Project name."), "branch": _p("Git branch.")},
            ),
            (
                "pages-deployments-rollback",
                "Roll a project back to a deployment.",
                True,
                ("project", "deployment_id"),
                {"project": _p("Project name."), "deployment_id": _ID},
            ),
            (
                "pages-domains-list",
                "List a project's custom domains.",
                False,
                ("project",),
                {"project": _p("Project name.")},
            ),
            (
                "pages-domains-add",
                "Add a custom domain to a project.",
                True,
                ("project", "domain"),
                {"project": _p("Project name."), "domain": _p("Hostname.")},
            ),
        ),
    ),
    "load-balancing": (
        "Load balancing: origin pools, health monitors, and load balancers.",
        (
            ("lb-pools-list", "List origin pools.", False, (), {}),
            (
                "lb-pools-create",
                "Create an origin pool.",
                True,
                ("name", "origins"),
                {"name": _p("Pool name."), "origins": _LIST},
            ),
            (
                "lb-pools-update",
                "Change an origin pool's origins.",
                True,
                ("pool_id", "origins"),
                {"pool_id": _ID, "origins": _LIST},
            ),
            ("lb-pool-health", "Get an origin pool's health.", False, ("pool_id",), {"pool_id": _ID}),
            ("lb-monitors-list", "List health monitors.", False, (), {}),
            (
                "lb-monitors-create",
                "Create a health monitor.",
                True,
                ("path",),
                {"path": _p("Health check path."), "interval": _INT},
            ),
            ("lb-list", "List a zone's load balancers.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "lb-create",
                "Create a load balancer on a hostname.",
                True,
                ("zone_id", "hostname", "pools"),
                {"zone_id": ZONE, "hostname": _p("Hostname."), "pools": _LIST},
            ),
            (
                "lb-update",
                "Change a load balancer's pools.",
                True,
                ("zone_id", "lb_id", "pools"),
                {"zone_id": ZONE, "lb_id": _ID, "pools": _LIST},
            ),
        ),
    ),
    "access": (
        "Access and tunnels: Zero Trust applications, policies, groups, and tunnels.",
        (
            ("access-apps-list", "List Access applications.", False, (), {}),
            (
                "access-apps-create",
                "Protect a hostname with an Access application.",
                True,
                ("name", "domain"),
                {"name": _p("Application name."), "domain": _p("Hostname.")},
            ),
            ("access-policies-list", "List an application's policies.", False, ("app_id",), {"app_id": _ID}),
            (
                "access-policies-create",
                "Add a policy to an application.",
                True,
                ("app_id", "decision", "emails"),
                {"app_id": _ID, "decision": {"type": "string", "enum": ["allow", "deny"]}, "emails": _LIST},
            ),
            ("access-groups-list", "List Access groups.", False, (), {}),
            ("tunnels-list", "List tunnels.", False, (), {}),
            ("tunnels-create", "Create a tunnel.", True, ("name",), {"name": _p("Tunnel name.")}),
            ("tunnels-delete", "Delete a tunnel.", True, ("tunnel_id",), {"tunnel_id": _ID}),
        ),
    ),
    "email-routing": (
        "Email routing: forwarding addresses and routing rules of a zone.",
        (
            ("email-routing-get", "Get a zone's email routing status.", False, ("zone_id",), {"zone_id": ZONE}),
            ("email-routing-enable", "Enable email routing on a zone.", True, ("zone_id",), {"zone_id": ZONE}),
            ("email-rules-list", "List a zone's routing rules.", False, ("zone_id",), {"zone_id": ZONE}),
            (
                "email-rules-create",
                "Forward one address of a zone to a destination.",
                True,
                ("zone_id", "from", "to"),
                {"zone_id": ZONE, "from": _p("Local address."), "to": _p("Destination.")},
            ),
            (
                "email-rules-delete",
                "Delete a routing rule.",
                True,
                ("zone_id", "rule_id"),
                {"zone_id": ZONE, "rule_id": _ID},
            ),
            ("email-addresses-list", "List verified destination addresses.", False, (), {}),
            (
                "email-addresses-create",
                "Add a destination address to verify.",
                True,
                ("email",),
                {"email": _p("Email address.")},
            ),
            (
                "email-addresses-delete",
                "Remove a destination address.",
                True,
                ("email",),
                {"email": _p("Email address.")},
            ),
        ),
    ),
    "account": (
        "Account: members, API tokens, and the audit log.",
        (
            ("account-get", "Get the account's name and settings.", False, (), {}),
            ("members-list", "List account members and roles.", False, (), {}),
            (
                "members-invite",
                "Invite a member with a role.",
                True,
                ("email", "role"),
                {"email": _p("Email address."), "role": _p("Role name.")},
            ),
            ("members-remove", "Remove an account member.", True, ("member_id",), {"member_id": _ID}),
            ("api-tokens-list", "List API tokens.", False, (), {}),
            (
                "api-tokens-create",
                "Create an API token with permissions.",
                True,
                ("name", "permissions"),
                {"name": _p("Token name."), "permissions": _LIST},
            ),
            ("api-tokens-revoke", "Revoke an API token.", True, ("token_id",), {"token_id": _ID}),
            ("audit-logs-list", "List recent account audit log entries.", False, (), {"since": _DATE}),
        ),
    ),
}
EDGE = Assistant(
    ASSISTANT_ID,
    "Edge manages the user's edge network account through its REST API: zones, DNS, Workers, R2 storage, WAF and "
    "firewall, cache, SSL/TLS, analytics and logs, Pages, load balancing, Access and tunnels, email routing, and "
    "account members and tokens. Most Actions take a zone id from zones-list.",
    tuple(
        Action(action_id, summary, {**_OBJECT, "properties": properties, "required": list(required)}, writes)
        for _group, (_description, actions) in GROUPS.items()
        for action_id, summary, writes, required, properties in actions
    ),
)
ACTION_GROUP = {action[0]: group for group, (_description, actions) in GROUPS.items() for action in actions}
ASSISTANTS = {assistant.id: assistant for assistant in (EDGE, *DISTRACTORS)}

ZONES = {"zn-1a2b": "example.com", "zn-3c4d": "example.org"}
RECORDS = (
    ("dr-1", "zn-1a2b", "A", "www.example.com", "192.0.2.1"),
    ("dr-2", "zn-3c4d", "A", "www.example.org", "192.0.2.30"),
    ("dr-3", "zn-3c4d", "MX", "example.org", "mail.example.org"),
)
SCRIPTS = ("api-gateway", "image-resizer")
BUCKETS = ("media-assets", "backups-2026")
TRAFFIC = {("zn-1a2b", "2026-10-01"): 48213, ("zn-3c4d", "2026-10-01"): 9120}


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
        entry: dict[str, object] = {
            "assistant": assistant_id,
            "action": action_id,
            "input": copy.deepcopy(dict(arguments)),
        }
        self.ledger.append(entry)
        assistant = ASSISTANTS.get(assistant_id)
        action = None if assistant is None else next((a for a in assistant.actions if a.id == action_id), None)
        try:
            if action is None:
                raise ActionFailedError("undeclared-action")
            result, effect = self._run(assistant, action, arguments)
        except ActionFailedError as exc:
            entry["failed"] = exc.code
            raise
        entry["result"] = copy.deepcopy(result)
        if effect is not None:
            entry["effect"] = effect
        return result

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


def _scope(template: Template, locale: str, scope: str) -> tuple[str, ...]:
    if scope == "needed":
        return template.needed
    others = sorted(
        (name for name in ASSISTANTS if name not in template.needed),
        key=lambda name: hashlib.sha256(f"{template.id}.{locale}:{name}".encode()).hexdigest(),
    )
    return (*template.needed, *others[: int(scope) - len(template.needed)])


SCENARIOS = tuple(
    Scenario(
        f"{template.id}.{locale}",
        template,
        locale,
        SCOPES[(index + locale_index) % len(SCOPES)],
        _scope(template, locale, SCOPES[(index + locale_index) % len(SCOPES)]),
    )
    for index, template in enumerate(TEMPLATES)
    for locale_index, locale in enumerate(LOCALES)
)
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
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


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
    for group in (*(t.id for t in TEMPLATES), *LOCALES):
        if {s.scope for s in SCENARIOS if group in {s.template.id, s.locale}} != set(SCOPES):
            raise ValueError(f"{group} misses an Assistant scope")

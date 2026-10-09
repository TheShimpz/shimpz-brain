"""The large-API stratum's Assistant contract: one edge platform's REST API as 120 Actions (ADR-0094).

Fourteen resource groups of at most nine endpoint-shaped Actions each, with realistic names, descriptions, and
schemas, plus the precision corpus's irrelevant-domain Assistants. It is a SIMULATION; ``eval.large_api`` holds the
state, templates, and scenarios. This module uses only the standard library.
"""

from collections.abc import Mapping

from eval.fixtures import _DATE, _OBJECT, _STRING, DISTRACTORS, Action, Assistant

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

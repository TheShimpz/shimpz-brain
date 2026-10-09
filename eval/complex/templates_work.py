"""precision-v3 templates of strata S1 to S3: several requests, long conversations, and large observations.

This module uses only the standard library.
"""

from eval.complex.steps import (
    BUDGET,
    COM,
    DISTRACT,
    HOME,
    ORG,
    OWNER,
    PRICE,
    RETENTION,
    ROOT,
    THANKS,
    WHAT_CHANGED,
    WHICH_1,
    count_q,
    create,
    doc_q,
    event,
    filler,
    note,
    ph,
    plan,
    plan_q,
    point,
    send,
    st,
    task,
    tpl,
)
from eval.complex.texts import t
from eval.complex.world import customer_id
from eval.world import STATUS_IP

S1 = (
    tpl(
        "s1-task-and-event",
        "S1",
        "independent obligations",
        ("and", "<8k"),
        ("tasks", "calendar", *DISTRACT),
        (
            st(
                ph("task_add", title=t("passport")),
                ph("also"),
                ph("event_create", title="Team lunch", date="2026-10-09", time="12:00"),
                ref=(task(), event("Team lunch", "2026-10-09", "12:00")),
                say="Confirms both the passport task and the Team lunch event.",
            ),
        ),
    ),
    tpl(
        "s1-web-then-point",
        "S1",
        "dependencies",
        ("chain", "<8k"),
        ("research", "dns", *DISTRACT),
        (
            st(
                ph("web_ip", service="status.example.org"),
                ph("then"),
                ph("use_that_ip", fqdn="status.example.com"),
                ref=(("research", "search-web", {"query": "status.example.org IP"}), point("rc-status", STATUS_IP)),
                say=f"Found {STATUS_IP} and pointed status.example.com to it.",
            ),
            THANKS,
        ),
    ),
    tpl(
        "s1-conditional-note",
        "S1",
        "conditional notification",
        ("selection", "<8k"),
        ("crm", "messages"),
        (
            st(
                ph("crm_if_plan_note", customer="Ana Souza", city="Lisbon", plan="Pro", text=t("retention_note")),
                ref=(note("Ana Souza", "Lisbon"),),
                say="Ana Souza from Lisbon is on Pro, so the note was added.",
            ),
            st(
                ph("crm_if_plan_note", customer="Ana Souza", city="Porto", plan="Pro", text=t("retention_note")),
                say="Ana Souza from Porto is on Basic, so nothing was added.",
            ),
            THANKS,
        ),
    ),
    tpl(
        "s1-correct-earlier",
        "S1",
        "correction of an earlier request",
        ("mixed", "<8k"),
        ("dns",),
        (
            st(
                ph("dns_point", fqdn="api.example.com", ip="198.51.100.7"),
                ref=(point("rc-api", "198.51.100.7"),),
                say="api.example.com points to 198.51.100.7.",
            ),
            st(
                ph("correction", value="198.51.100.8", old="198.51.100.7"),
                ref=(point("rc-api", "198.51.100.8"),),
                say="Changed api.example.com to 198.51.100.8.",
            ),
            st(ph("dns_which", zone="example.com", ip="198.51.100.8"), say="Only api.example.com."),
            THANKS,
        ),
    ),
    tpl(
        "s1-shop-launch",
        "S1",
        "independent obligations",
        ("and", "8-32k"),
        ("dns", "tasks", "messages", "calendar"),
        (
            st(
                ph("dns_create", name="shop", zone="example.com", ip="203.0.113.10"),
                ph("also"),
                ph("task_add", title=t("shop_dns")),
                ref=(create("shop", "203.0.113.10"), task()),
                say="Created the shop record and the task.",
            ),
            st(
                ph("msg_send", person="Carla", text=t("shop_live")),
                ref=(send("ct-carla"),),
                say="Message sent to Carla.",
            ),
            st(
                ph("event_create", title="Shop launch review", date="2026-10-12", time="16:00"),
                ref=(event("Shop launch review", "2026-10-12", "16:00"),),
                say="Event scheduled.",
            ),
            THANKS,
        ),
    ),
    tpl(
        "s1-status-chain",
        "S1",
        "dependencies",
        ("chain", "8-32k"),
        ("research", "dns", "messages", "tasks", "calendar", "crm"),
        (
            st(ph("web_ip", service="status.example.org"), say=f"The current IP is {STATUS_IP}."),
            st(
                ph("use_that_ip", fqdn="status.example.com"),
                ref=(point("rc-status", STATUS_IP),),
                say="status.example.com points to that address.",
            ),
            st(
                ph("msg_tell_that", person="Ops on-call"),
                ref=(send("ct-ops"),),
                say="Told Ops on-call about the change.",
            ),
            st(ph("task_add", title=t("monitor")), ref=(task(),), say="Task added."),
            HOME,
            BUDGET,
            plan_q("Kenji Sato", "Tokyo"),
            THANKS,
        ),
    ),
    tpl(
        "s1-pick-and-act",
        "S1",
        "conditional notification",
        ("selection", "8-32k"),
        ("tasks", "dns", "crm", "docs"),
        (
            HOME,
            st(
                ph("task_done_topic", topic=t("topic_sink")),
                ref=(("tasks", "complete-task", {"task_id": "tk-3"}),),
                say="Marked 'Fix the kitchen sink' done.",
            ),
            WHICH_1,
            st(
                ph("dns_point", fqdn="www.example.com", ip="203.0.113.30"),
                ref=(point("rc-www", "203.0.113.30"),),
                say="www.example.com points to 203.0.113.30.",
            ),
            count_q("Porto", "Pro"),
            doc_q(*PRICE),
            THANKS,
        ),
    ),
    tpl(
        "s1-busy-day",
        "S1",
        "independent obligations",
        ("mixed", "32-128k"),
        ("dns", "tasks", "calendar", "messages", "crm", "docs", "research"),
        (
            st(ph("task_add", title=t("report")), ref=(task(),), say="Task added."),
            doc_q(*ROOT),
            doc_q(*OWNER),
            st(ph("msg_send", person="Ops on-call", text=t("root_cause")), ref=(send("ct-ops"),), say="Sent to Ops."),
            HOME,
            BUDGET,
            st(
                ph("dns_point", fqdn="app.example.com", ip="198.51.100.20"),
                ref=(point("rc-app", "198.51.100.20"),),
                say="app.example.com updated.",
            ),
            st(
                ph("correction", value="198.51.100.21", old="198.51.100.20"),
                ref=(point("rc-app", "198.51.100.21"),),
                say="app.example.com now points to 198.51.100.21.",
            ),
            plan_q("Farah Haddad", "Cairo"),
            doc_q(*PRICE),
            st(
                ph("crm_note", customer="Farah Haddad", city="Cairo", text=t("price_note")),
                ref=(note("Farah Haddad", "Cairo"),),
                say="Note added to Farah Haddad.",
            ),
            count_q("Lisbon", "Pro"),
            st(
                ph("event_create", title="Pricing sync", date="2026-10-20", time="11:00"),
                ref=(event("Pricing sync", "2026-10-20", "11:00"),),
                say="Pricing sync scheduled.",
            ),
            WHICH_1,
            doc_q(*RETENTION),
            st(
                ph("crm_if_plan_note", customer="Carla Mendes", city="Porto", plan="Pro", text=t("retention_note")),
                ref=(note("Carla Mendes", "Porto"),),
                say="Carla Mendes is on Pro; note added.",
            ),
            WHAT_CHANGED,
            THANKS,
        ),
    ),
)


S2 = (
    tpl(
        "s2-sign-messages",
        "S2",
        "preference persistence",
        ("and", "<8k"),
        ("messages",),
        (
            st(
                ph("remember", preference=t("pref_sign")),
                ph("msg_send", person="Bruno", text=t("late")),
                ref=(send("ct-bruno"),),
                say="Sent to Bruno, signed Juliano, and keeps the preference.",
            ),
            st(
                ph("msg_send", person="Carla", text=t("moved_15")),
                ref=(send("ct-carla"),),
                say="Sent to Carla, still signed Juliano.",
            ),
        ),
    ),
    tpl(
        "s2-that-customer",
        "S2",
        "reference resolution",
        ("chain", "<8k"),
        ("crm",),
        (
            plan_q("Mateo García", "Madrid"),
            st(
                ph("crm_move_that", plan="Pro"),
                ref=(plan("Mateo García", "Madrid", "Pro"),),
                say="Moved Mateo García (with the accent, Madrid) to Pro, not Mateo Garcia.",
            ),
            THANKS,
        ),
    ),
    tpl(
        "s2-office-ip-changes",
        "S2",
        "changing facts",
        ("mixed", "8-32k"),
        ("dns",),
        (
            st(ph("fact", fact=t("fact_office")), say="Acknowledges the office IP."),
            st(
                ph("dns_create_named", name="office", zone="example.com", what=t("what_office")),
                ref=(create("office", "203.0.113.40"),),
                say="Created office.example.com pointing to 203.0.113.40.",
            ),
            st(ph("fact", fact=t("fact_office_new")), say="Acknowledges the new office IP."),
            st(
                ph("dns_update_named", name="office", what=t("what_office_new")),
                ref=(("dns", "update-record", {"zone_id": COM, "record_id": "rc-new-1", "content": "203.0.113.41"}),),
                say="office.example.com now points to 203.0.113.41.",
            ),
            THANKS,
        ),
    ),
    tpl(
        "s2-earlier-restriction",
        "S2",
        "earlier restrictions",
        ("chain", "8-32k"),
        ("dns", "tasks", "calendar", "crm"),
        (
            st(ph("remember", preference=t("fact_restrict")), say="Acknowledges the restriction."),
            HOME,
            BUDGET,
            plan_q("Kenji Sato", "Tokyo"),
            st(
                ph("dns_point", fqdn="www.example.org", ip="192.0.2.99"),
                say="Asks for confirmation before changing example.org, and changes nothing yet.",
            ),
            st(
                ph("confirm"),
                ref=(point("rc-org-www", "192.0.2.99", ORG),),
                say="Changed www.example.org after the confirmation.",
            ),
            WHICH_1,
            count_q("Berlin", "Pro"),
            THANKS,
        ),
    ),
    tpl(
        "s2-evicted-fact",
        "S2",
        "history eviction",
        ("chain", "32-128k"),
        ("dns", "tasks", "calendar", "crm", "docs"),
        (
            st(ph("fact", fact=t("fact_backup")), say="Acknowledges the backup server IP."),
            *filler(25),
            st(
                ph("point_remembered", fqdn="backup.example.com", what=t("what_backup")),
                say="If the backup IP is no longer available in the conversation, says so and asks; it never invents "
                "an "
                "address. If it is still available, it may create the record with 198.51.100.200.",
            ),
            st(
                ph("it_was", value="198.51.100.200"),
                ref=(create("backup", "198.51.100.200"),),
                say="Created backup.example.com pointing to 198.51.100.200 (or confirms it was already created).",
            ),
        ),
    ),
    tpl(
        "s2-client-tag",
        "S2",
        "preference persistence",
        ("selection", "8-32k"),
        ("tasks", "calendar", "crm"),
        (
            st(ph("remember", preference=t("pref_client_tag")), say="Acknowledges the tagging preference."),
            st(ph("task_add", title=t("passport")), ref=(task(("client",)),), say="Task created with the client tag."),
            BUDGET,
            HOME,
            st(ph("task_add", title=t("report")), ref=(task(("client",)),), say="Task created with the client tag."),
            plan_q("Farah Haddad", "Cairo"),
            st(ph("task_add", title=t("monitor")), ref=(task(("client",)),), say="Task created with the client tag."),
            THANKS,
        ),
    ),
    tpl(
        "s2-my-manager",
        "S2",
        "reference resolution",
        ("selection", "<8k"),
        ("messages", "crm"),
        (
            st(ph("fact", fact=t("fact_manager")), say="Acknowledges."),
            plan_q("Ana Souza", "Lisbon"),
            st(ph("msg_manager", text=t("invoice_ok")), ref=(send("ct-carla"),), say="Sent to Carla Mendes."),
        ),
    ),
    tpl(
        "s2-long-reuse",
        "S2",
        "changing facts",
        ("mixed", "32-128k"),
        ("docs", "messages", "crm", "tasks", "calendar", "dns"),
        (
            doc_q(*OWNER),
            *filler(9),
            doc_q(*ROOT),
            st(ph("msg_send", person="Ana", text=t("root_cause")), ref=(send("ct-ana"),), say="Sent to Ana."),
            *filler(6),
            st(
                ph("recall_fact", topic=t("topic_owner")),
                say="Recalls Priya Iyer from the incident report, or rereads it; never invents a name.",
            ),
            THANKS,
        ),
    ),
)

ANA_LISBON, ANA_PORTO = customer_id("Ana Souza", "Lisbon"), customer_id("Ana Souza", "Porto")
S3 = (
    tpl(
        "s3-near-match",
        "S3",
        "distracting records",
        ("selection", "<8k"),
        ("crm",),
        (
            st(
                ph("crm_plan", customer="Mateo Garcia", city="Madrid"),
                say="Mateo Garcia (no accent) in Madrid is on Basic; not Mateo García, who is on Business.",
            ),
        ),
    ),
    tpl(
        "s3-opaque-email",
        "S3",
        "opaque IDs",
        ("chain", "<8k"),
        ("crm",),
        (
            st(
                ph("crm_email", customer="Kenji Sato", city="Tokyo", email="kenji.sato@newmail.example.net"),
                ref=(
                    (
                        "crm",
                        "update-customer",
                        {"customer_id": customer_id("Kenji Sato", "Tokyo"), "email": "kenji.sato@newmail.example.net"},
                    ),
                ),
                say="Updated Kenji Sato's email.",
            ),
        ),
    ),
    tpl(
        "s3-counts",
        "S3",
        "relevant facts at different positions",
        ("and", "8-32k"),
        ("crm",),
        (
            count_q("Lisbon", "Pro"),
            count_q("Porto", "Basic"),
            count_q("Tokyo", "Business"),
            THANKS,
        ),
    ),
    tpl(
        "s3-doc-then-note",
        "S3",
        "subsequent reuse",
        ("selection", "8-32k"),
        ("docs", "crm"),
        (
            doc_q(*RETENTION),
            st(
                ph("crm_if_plan_note", customer="Ana Souza", city="Lisbon", plan="Pro", text=t("retention_note")),
                ref=(note("Ana Souza", "Lisbon"),),
                say="Ana Souza from Lisbon is on Pro; note added.",
            ),
            THANKS,
        ),
    ),
    tpl(
        "s3-incident-facts",
        "S3",
        "relevant facts at different positions",
        ("and", "32-128k"),
        ("docs", "messages"),
        (
            doc_q(*ROOT),
            doc_q(*OWNER),
            st(ph("msg_send", person="Ops on-call", text=t("root_cause")), ref=(send("ct-ops"),), say="Sent to Ops."),
        ),
    ),
    tpl(
        "s3-price-reuse",
        "S3",
        "subsequent reuse",
        ("chain", "32-128k"),
        ("docs", "crm", "tasks", "calendar"),
        (
            doc_q(*PRICE),
            HOME,
            plan_q("Farah Haddad", "Cairo"),
            st(
                ph("crm_note", customer="Farah Haddad", city="Cairo", text=t("price_note")),
                ref=(note("Farah Haddad", "Cairo"),),
                say="Note added to Farah Haddad from Cairo.",
            ),
            BUDGET,
            count_q("Cairo", "Business"),
            st(
                ph("crm_note", customer="Mateo García", city="Madrid", text=t("price_note")),
                ref=(note("Mateo García", "Madrid"),),
                say="Note added to Mateo García (Business, Madrid).",
            ),
            THANKS,
        ),
    ),
    tpl(
        "s3-two-anas",
        "S3",
        "distracting records",
        ("selection", "8-32k"),
        ("crm", "tasks", "calendar"),
        (
            st(
                ph("crm_note", customer="Ana Souza", city="Porto", text=t("followup_note")),
                ref=(note("Ana Souza", "Porto"),),
                say="Note added to Ana Souza from Porto only.",
            ),
            HOME,
            BUDGET,
            st(
                ph("crm_move", customer="Ana Souza", city="Lisbon", plan="Business"),
                ref=(plan("Ana Souza", "Lisbon", "Business"),),
                say="Moved Ana Souza from Lisbon to Business.",
            ),
            plan_q("Ana Souza", "Porto"),
            count_q("Lisbon", "Business"),
            THANKS,
        ),
    ),
    tpl(
        "s3-deep-records",
        "S3",
        "opaque IDs",
        ("mixed", "32-128k"),
        ("crm", "docs", "messages", "tasks"),
        (
            count_q("Madrid", "Basic"),
            plan_q("Mateo Garcia", "Madrid"),
            doc_q(*PRICE),
            doc_q(*RETENTION),
            st(
                ph("crm_move", customer="Mateo Garcia", city="Madrid", plan="Pro"),
                ref=(plan("Mateo Garcia", "Madrid", "Pro"),),
                say="Moved Mateo Garcia (no accent) to Pro.",
            ),
            count_q("Madrid", "Pro"),
            doc_q(*OWNER),
            plan_q("Carla Mendes", "Porto"),
            st(
                ph("crm_if_plan_note", customer="Carla Mendes", city="Porto", plan="Pro", text=t("onboarding_note")),
                ref=(note("Carla Mendes", "Porto"),),
                say="Carla is on Pro; note added.",
            ),
            count_q("Porto", "Pro"),
            HOME,
            st(
                ph("crm_email", customer="Kenji Sato", city="Tokyo", email="k.sato@mail.example.net"),
                ref=(
                    (
                        "crm",
                        "update-customer",
                        {"customer_id": customer_id("Kenji Sato", "Tokyo"), "email": "k.sato@mail.example.net"},
                    ),
                ),
                say="Email updated.",
            ),
            doc_q(*ROOT),
            plan_q("Kenji Sato", "Tokyo"),
            WHAT_CHANGED,
            THANKS,
        ),
    ),
)

//! The closed rules a route answer keeps in each lane (`_route`, `_classification`, `_selection`).

use super::super::route::route;
use super::*;

fn request(expected: Option<&'static str>) -> Request {
    Request {
        objective: "objective".into(),
        expected,
        candidates: ["shimpz-cloudflare", "shimpz-whatsapp"]
            .into_iter()
            .map(|id| Candidate {
                id: id.into(),
                name: id.into(),
                summary: String::new(),
            })
            .collect(),
        reference: None,
        conversation: Vec::new(),
        locale: "pt".into(),
        decision: None,
        installable: None,
    }
}

fn routes(expected: Option<&'static str>, reply: &str) -> Option<Route> {
    route(&parse(reply), &request(expected))
}

fn classified(intent: &str, query: &str, reply: &str, follows: bool) -> Option<Route> {
    routes(None, &answer(intent, query, &[], reply, follows))
}

#[test]
fn an_unconsumed_reply_is_discarded_and_a_missing_clarification_stays_fatal() {
    let ordinary = classified("ordinary-task", "", "Claro, posso ajudar!", false).unwrap();
    assert_eq!(
        (ordinary.intent, ordinary.reply.as_str()),
        ("ordinary-task", "")
    );
    let targeted = classified(UNINSTALL, "cloudflare", "Removendo o Cloudflare.", false).unwrap();
    assert_eq!(
        (targeted.query.as_str(), targeted.reply.as_str()),
        ("cloudflare", "")
    );
    let ids = ["shimpz-cloudflare"];
    let selected = routes(
        Some(UNINSTALL),
        &answer(UNINSTALL, "", &ids, "Vou remover.", false),
    );
    assert_eq!(
        selected.map(|route| (route.assistant_ids, route.reply)),
        Some((vec![ids[0].to_owned()], String::new()))
    );
    for intent in ["unresolved", INSTALL, UNINSTALL] {
        assert_eq!(classified(intent, "", "", false), None, "{intent}");
    }
    assert_eq!(
        routes(Some(UNINSTALL), &answer("unresolved", "", &[], "", false)),
        None
    );
    // The discarded reply is still validated first.
    assert_eq!(
        classified("ordinary-task", "", "Primeira linha\u{2028}segunda", false),
        None
    );
}

#[test]
fn only_a_targeted_install_classification_continues_the_task() {
    let follows = classified(INSTALL, "exa", "", true).unwrap();
    assert!(follows.task_follows && follows.query == "exa");
    assert!(!classified(INSTALL, "exa", "", false).unwrap().task_follows);
    for (intent, query, reply) in [
        ("ordinary-task", "", ""),
        (UNINSTALL, "exa", ""),
        ("unresolved", "", "Which Assistant?"),
        (INSTALL, "", "Which Assistant?"),
    ] {
        assert_eq!(classified(intent, query, reply, true), None, "{intent}");
    }
    let continued = answer(INSTALL, "", &["shimpz-cloudflare"], "", true);
    assert_eq!(routes(Some(INSTALL), &continued), None);
}

#[test]
fn classification_and_selection_conflicts_are_refused() {
    let ids = ["shimpz-cloudflare"];
    assert_eq!(
        routes(None, &answer("ordinary-task", "", &ids, "", false)),
        None
    );
    assert_eq!(
        routes(None, &answer("ordinary-task", "unexpected", &[], "", false)),
        None
    );
    assert_eq!(
        routes(None, &answer(INSTALL, "shimpz-cloudflare", &ids, "", false)),
        None
    );
    assert_eq!(
        routes(
            Some(UNINSTALL),
            &answer(UNINSTALL, "unexpected", &ids, "", false)
        ),
        None
    );
    assert_eq!(
        routes(
            Some(UNINSTALL),
            &answer("unresolved", "", &ids, "clarify", false)
        ),
        None
    );
    assert_eq!(
        routes(None, &answer(INSTALL, "line\nbreak", &[], "", false)),
        None
    );
}

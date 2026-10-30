//! Intent routing against the fake provider origin (`test_intent_route.py`, `test_decision_api.py`, and the runtime
//! half of `test_intent_fast_path.py`).

mod fast;
mod rules;

use super::*;
use crate::decisions::testing::{
    self, anthropic_texts, contract, openai_texts, parse, provider_failed, run, with, without,
};
use crate::provider::testing::{Answer, Received};

const JEV_KEY: &str = "tsk-test-0123456789abcdef";

fn classification(objective: &str) -> Value {
    json::object([
        ("provider", testing::provider("openai", "gpt-6.1-sol")),
        ("objective", json::text(objective)),
        ("expected_intent", Value::Null),
        ("candidates", json::array(Vec::new())),
        ("lifecycle_reference", Value::Null),
        ("conversation", json::array(Vec::new())),
        ("locale", json::text("pt")),
    ])
}

fn candidates(uninstall: bool) -> Value {
    let summary = |text: &str| {
        if uninstall {
            String::new()
        } else {
            text.to_owned()
        }
    };
    json::array(vec![
        json::object([
            ("id", json::text("shimpz-cloudflare")),
            ("name", json::text("Shimpz Cloudflare")),
            (
                "summary",
                json::text(&summary("Manage DNS zones and records.")),
            ),
        ]),
        json::object([
            ("id", json::text("shimpz-whatsapp")),
            ("name", json::text("Shimpz WhatsApp")),
            (
                "summary",
                json::text(&summary("Send reviewed WhatsApp messages.")),
            ),
        ]),
    ])
}

fn selection(expected: &str, objective: &str) -> Value {
    let body = with(
        &classification(objective),
        "expected_intent",
        json::text(expected),
    );
    with(&body, "candidates", candidates(expected == UNINSTALL))
}

fn decided(body: &Value) -> Value {
    let decision = json::object([
        ("provider", json::text("typesafe")),
        ("api_key", json::text(JEV_KEY)),
    ]);
    with(body, "decision_provider", decision)
}

/// A structured route answer.
fn answer(intent: &str, query: &str, ids: &[&str], reply: &str, follows: bool) -> String {
    let ids: Vec<Value> = ids.iter().map(|id| json::text(id)).collect();
    json::compact(&json::object([
        ("intent", json::text(intent)),
        ("query", json::text(query)),
        ("assistant_ids", json::array(ids)),
        ("reply", json::text(reply)),
        ("task_follows", Value::Bool(follows)),
    ]))
}

fn routed(answers: Vec<Answer>, body: &Value) -> (Result<Value, RouteError>, Vec<Received>) {
    run(answers, |providers, scope| handle(providers, scope, body))
}

/// The route Team receives for these fields.
fn response(intent: &str, query: &str, ids: &[&str], reply: &str, follows: bool) -> Value {
    parse(&answer(intent, query, ids, reply, follows))
}

fn refused_before_any_call(body: &Value) -> Result<Value, RouteError> {
    let (outcome, sent) = routed(
        vec![Answer::openai_text(&answer(
            "ordinary-task",
            "",
            &[],
            "",
            false,
        ))],
        body,
    );
    assert!(sent.is_empty(), "no provider request");
    outcome
}

#[test]
fn classification_is_one_low_effort_structured_call_with_untrusted_framing() {
    let reply = answer(UNINSTALL, "cloudflare", &[], "", false);
    let (outcome, sent) = routed(
        vec![Answer::openai_text(&reply)],
        &classification("por favor tire o cloudflare"),
    );
    assert_eq!(
        outcome,
        Ok(response(UNINSTALL, "cloudflare", &[], "", false))
    );
    let request = sent[0].json();
    let format = request
        .get("text")
        .and_then(|text| text.get("format"))
        .unwrap();
    assert_eq!(json::str_of(format, "name"), Some("StructuredRoute"));
    assert_eq!(
        request.get("reasoning"),
        Some(&parse(r#"{"effort":"low"}"#))
    );
    let texts = openai_texts(&sent[0]);
    assert!(texts[0].contains("untrusted data") && texts[0].contains("exactly one line"));
    assert!(texts[0].contains(
        "Write it in Brazilian Portuguese, the language the user selected in the interface"
    ));
    assert!(texts[1].contains("por favor tire o cloudflare"));
    assert!(!texts.concat().contains("api_key"));
}

#[test]
fn a_lifecycle_reference_and_the_conversation_are_payload_never_instructions() {
    let body = with(
        &classification("Desinstala esse então."),
        "lifecycle_reference",
        parse(r#"{"id":"shimpz-cloudflare","name":"Shimpz Cloudflare"}"#),
    );
    let body = with(
        &body,
        "conversation",
        parse(r#"[{"role":"assistant","text":"Temos apenas Cloudflare/DNS.","truncated":false}]"#),
    );
    let reply = answer(UNINSTALL, "cloudflare", &[], "", false);
    let (outcome, sent) = routed(vec![Answer::openai_text(&reply)], &body);
    assert!(outcome.is_ok());
    let texts = openai_texts(&sent[0]);
    assert!(texts[0].contains("current objective is the only fresh instruction"));
    assert!(!texts[0].contains("shimpz-cloudflare") && !texts[0].contains("Cloudflare/DNS"));
    assert!(texts[1].contains(r#""lifecycle_reference":{"id":"shimpz-cloudflare""#));
    assert!(
        texts[1].contains(
            r#""conversation":[{"role":"assistant","text":"Temos apenas Cloudflare/DNS."#
        )
    );
}

#[test]
fn anthropic_routes_with_the_same_schema() {
    let body = with(
        &classification("liste minhas zonas"),
        "provider",
        testing::provider("anthropic", "claude-sonnet-5-5"),
    );
    let reply = answer("ordinary-task", "", &[], "", false);
    let (outcome, sent) = routed(vec![Answer::anthropic_text(&reply)], &body);
    assert_eq!(outcome, Ok(response("ordinary-task", "", &[], "", false)));
    let (system, user) = anthropic_texts(&sent[0]);
    assert!(
        system.contains("Route one fresh Shimpz user message.")
            && user.contains("liste minhas zonas")
    );
}

#[test]
fn a_selection_is_bound_to_its_intent_and_exact_candidate_ids() {
    let reply = answer(
        INSTALL,
        "",
        &["shimpz-cloudflare", "shimpz-whatsapp"],
        "",
        false,
    );
    let (outcome, sent) = routed(
        vec![Answer::openai_text(&reply)],
        &selection(INSTALL, "instale cloudflare e whatsapp"),
    );
    assert_eq!(
        outcome,
        Ok(response(
            INSTALL,
            "",
            &["shimpz-cloudflare", "shimpz-whatsapp"],
            "",
            false
        ))
    );
    let texts = openai_texts(&sent[0]);
    assert!(texts[0].contains("Resolve the already classified assistant-install"));
    assert!(
        texts[1].contains(r#""expected_intent":"assistant-install""#)
            && texts[1].contains(r#""id":"shimpz-cloudflare""#)
    );
}

#[test]
fn an_unresolved_selection_only_asks() {
    let unresolved = answer(
        "unresolved",
        "",
        &[],
        "Qual Assistant instalado você quer desinstalar?",
        false,
    );
    let (outcome, _) = routed(
        vec![Answer::openai_text(&unresolved)],
        &selection(UNINSTALL, "remove it"),
    );
    assert_eq!(
        outcome,
        Ok(response(
            "unresolved",
            "",
            &[],
            "Qual Assistant instalado você quer desinstalar?",
            false
        ))
    );
    let empty = with(
        &selection(UNINSTALL, "desinstale desconhecido"),
        "candidates",
        json::array(Vec::new()),
    );
    assert!(
        routed(vec![Answer::openai_text(&unresolved)], &empty)
            .0
            .is_ok()
    );
}

#[test]
fn a_selection_outside_its_directory_is_a_provider_failure() {
    for (expected, reply) in [
        (UNINSTALL, answer(UNINSTALL, "", &["unknown"], "", false)),
        (
            UNINSTALL,
            answer(
                UNINSTALL,
                "",
                &["shimpz-cloudflare", "shimpz-cloudflare"],
                "",
                false,
            ),
        ),
        (
            INSTALL,
            answer(
                INSTALL,
                "",
                &["shimpz-whatsapp", "shimpz-cloudflare"],
                "",
                false,
            ),
        ),
        (INSTALL, answer(INSTALL, "", &[], "", false)),
        (
            INSTALL,
            answer("ordinary-task", "", &["shimpz-cloudflare"], "", false),
        ),
        (
            UNINSTALL,
            answer(
                UNINSTALL,
                "",
                &["shimpz-cloudflare", "shimpz-whatsapp"],
                "",
                false,
            ),
        ),
    ] {
        let (outcome, _) = routed(
            vec![Answer::openai_text(&reply)],
            &selection(expected, "lifecycle objective"),
        );
        assert_eq!(outcome, Err(provider_failed()), "{reply}");
    }
}

#[test]
fn a_clarification_admits_unicode_spacing_and_refuses_line_separators() {
    let reply = "Quel Assistant voulez-vous désinstaller\u{a0}?";
    let (outcome, _) = routed(
        vec![Answer::openai_text(&answer(
            UNINSTALL,
            "",
            &[],
            reply,
            false,
        ))],
        &classification("désinstalle"),
    );
    assert_eq!(outcome, Ok(response(UNINSTALL, "", &[], reply, false)));
    for separator in ['\u{2028}', '\u{2029}'] {
        let reply = answer(
            UNINSTALL,
            "",
            &[],
            &format!("Question{separator}suivante"),
            false,
        );
        let (outcome, _) = routed(
            vec![Answer::openai_text(&reply)],
            &classification("désinstalle"),
        );
        assert_eq!(outcome, Err(provider_failed()));
    }
}

#[test]
fn the_conversation_admits_user_unicode_format_and_layout() {
    let body = with(
        &classification("cloudflare"),
        "conversation",
        json::array(vec![json::object([
            ("role", json::text("user")),
            ("text", json::text("desinstala 👩\u{200d}💻\r\nagora")),
            ("truncated", Value::Bool(false)),
        ])]),
    );
    let reply = answer(UNINSTALL, "cloudflare", &[], "", false);
    assert!(routed(vec![Answer::openai_text(&reply)], &body).0.is_ok());
}

#[test]
fn malformed_answers_are_provider_failures_and_request_failures_are_redacted() {
    let valid = answer("ordinary-task", "", &[], "", false);
    for raw in [
        format!("{},\"reply\":\"\"}}", &valid[..valid.len() - 1]),
        format!(
            "{}{}}}",
            &valid[..valid.len() - 1],
            " ".repeat(MAX_RESPONSE_CHARS_TEST)
        ),
        format!("{},\"extra\":true}}", &valid[..valid.len() - 1]),
        "{}".to_owned(),
    ] {
        let (outcome, _) = routed(vec![Answer::openai_text(&raw)], &classification("hello"));
        assert_eq!(outcome, Err(provider_failed()));
    }
    let (outcome, _) = routed(
        vec![Answer::raw(400, "secret provider detail")],
        &classification("hello"),
    );
    assert_eq!(outcome, Err(provider_failed()));
}

const MAX_RESPONSE_CHARS_TEST: usize = 4_096;

fn assert_refused(cases: Vec<(Value, &str)>) {
    for (body, detail) in cases {
        let outcome = refused_before_any_call(&body);
        assert_eq!(outcome, Err(contract(detail)), "{detail}");
    }
}

#[test]
fn invalid_classifications_are_contract_refusals_before_provider_access() {
    let remove = classification("remove it");
    let reference = parse(r#"{"id":"shimpz-cloudflare","name":"Shimpz\nCloudflare"}"#);
    let entry = parse(r#"[{"role":"user","text":"remove\u0000it","truncated":false}]"#);
    assert_refused(vec![
        (classification("hidden\0message"), "invalid route objective"),
        (classification(" padded"), "invalid route objective"),
        (
            with(&classification("install"), "candidates", candidates(false)),
            "classification cannot include directory candidates",
        ),
        (
            with(&remove, "lifecycle_reference", reference),
            "invalid Assistant reference name",
        ),
        (
            with(&remove, "conversation", entry),
            "invalid conversation text",
        ),
    ]);
}

#[test]
fn invalid_directories_are_contract_refusals_before_provider_access() {
    let install = |directory: &str| {
        with(
            &selection(INSTALL, "install"),
            "candidates",
            parse(directory),
        )
    };
    let unordered = r#"[{"id":"shimpz-whatsapp","name":"W","summary":""},{"id":"shimpz-cloudflare","name":"C","summary":""}]"#;
    let uninstall = with(
        &selection(UNINSTALL, "uninstall"),
        "candidates",
        candidates(false),
    );
    assert_refused(vec![
        (install(unordered), "invalid directory candidate order"),
        (uninstall, "uninstall candidates cannot expose summaries"),
        (
            install(r#"[{"id":"Invalid Id","name":"Invalid","summary":""}]"#),
            "invalid Assistant id",
        ),
        (
            install(r#"[{"id":"shimpz-a","name":"A","summary":" padded"}]"#),
            "invalid Assistant summary",
        ),
    ]);
}

fn assert_invalid(bodies: Vec<Value>) {
    for body in bodies {
        let outcome = refused_before_any_call(&body);
        assert_eq!(
            outcome,
            Err(RouteError::Invalid),
            "{}",
            json::compact(&body)
        );
    }
}

fn conversation(role: &str, text: &str, truncated: Value, entries: usize) -> Value {
    let entry = json::object([
        ("role", json::text(role)),
        ("text", json::text(text)),
        ("truncated", truncated),
    ]);
    json::array(vec![entry; entries])
}

#[test]
fn an_open_request_is_refused_whole() {
    let base = classification("remove it");
    assert_invalid(vec![
        with(&base, "unexpected", json::text("remove it")),
        without(&base, "locale"),
        with(&base, "locale", Value::Null),
        with(&base, "locale", json::text("pt-BR")),
        without(&base, "conversation"),
        without(&base, "expected_intent"),
        with(&base, "expected_intent", json::text("ordinary-task")),
    ]);
}

#[test]
fn a_wrong_lane_or_malformed_conversation_is_refused_whole() {
    let base = classification("remove it");
    let selected = with(&base, "expected_intent", json::text(UNINSTALL));
    let reference = parse(r#"{"id":"shimpz-cloudflare","name":"Shimpz Cloudflare"}"#);
    let ok = Value::Bool(false);
    assert_invalid(vec![
        with(&selected, "lifecycle_reference", reference),
        with(
            &selected,
            "conversation",
            conversation("user", "remove it", ok.clone(), 1),
        ),
        with(
            &base,
            "conversation",
            conversation("system", "remove it", ok.clone(), 1),
        ),
        with(
            &base,
            "conversation",
            conversation("user", &"x".repeat(513), ok.clone(), 1),
        ),
        with(
            &base,
            "conversation",
            conversation("user", "remove it", json::int(1), 1),
        ),
        with(&base, "conversation", conversation("user", "x", ok, 9)),
    ]);
}

#[test]
fn a_decision_key_is_admitted_only_for_classification() {
    let decision = decided(&classification("Oi, tudo bem?"));
    let jev = Answer::json(
        200,
        &parse(
            r#"{"model":"jev-1.13.0","answers":{"intent":{"type":"choice","choice":"ordinary-task","confidence":0.9,"probabilities":{"ordinary-task":0.9,"assistant-install":0.05,"assistant-uninstall":0.03,"unresolved":0.02}}}}"#,
        ),
    );
    let (outcome, sent) = routed(vec![jev], &decision);
    assert_eq!(outcome, Ok(response("ordinary-task", "", &[], "", false)));
    assert_eq!(
        sent.len(),
        1,
        "a confident ordinary task skips the LLM route"
    );
    assert_eq!(sent[0].path, jev::PATH);
    assert!(!json::compact(&outcome.unwrap()).contains(JEV_KEY));
    let key = decision.get("decision_provider").unwrap();
    for body in [
        decided(&selection(INSTALL, "cloudflare")),
        with(
            &decision,
            "decision_provider",
            with(key, "provider", json::text("openai")),
        ),
        with(
            &decision,
            "decision_provider",
            with(key, "api_key", json::text("short")),
        ),
        with(
            &decision,
            "decision_provider",
            with(key, "extra", json::int(1)),
        ),
    ] {
        assert_eq!(refused_before_any_call(&body), Err(RouteError::Invalid));
    }
}

#[test]
fn installable_assistants_reach_only_the_decision_provider() {
    let installable = parse(
        r#"[{"name":"Cloudflare","summary":"Set up your domains' DNS."},{"name":"Meta Ads","summary":"Find where your ads pay off."}]"#,
    );
    let decision = decided(&classification("Oi, tudo bem?"));
    let Value::Array(items) = installable.clone() else {
        unreachable!()
    };
    for body in [
        with(&classification("Oi"), "installable", installable.clone()),
        with(&decision, "installable", json::array(Vec::new())),
        with(
            &decision,
            "installable",
            json::array(items.iter().cycle().take(10).cloned().collect()),
        ),
        with(
            &decision,
            "installable",
            json::array(vec![with(&items[0], "id", json::text("shimpz-cloudflare"))]),
        ),
        with(
            &decision,
            "installable",
            parse(r#"[{"name":"Cloudflare","summary":""}]"#),
        ),
        with(
            &decision,
            "installable",
            json::array(vec![with(
                &items[0],
                "summary",
                json::text(&"S".repeat(81)),
            )]),
        ),
    ] {
        assert_eq!(refused_before_any_call(&body), Err(RouteError::Invalid));
    }
    for (name, detail) in [
        ("Cloud\nflare", "invalid installable Assistant name"),
        (" Cloudflare", "invalid installable Assistant name"),
    ] {
        let body = with(
            &decision,
            "installable",
            json::array(vec![with(&items[0], "name", json::text(name))]),
        );
        assert_eq!(refused_before_any_call(&body), Err(contract(detail)));
    }
}

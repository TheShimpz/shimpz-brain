//! Capability planning against the fake provider origin (`test_capability_plan.py`, `test_decision_api.py`).

use super::*;
use crate::decisions::testing::{
    self, KEY, anthropic_texts, contract, openai_texts, parse, provider_failed, run, with, without,
};
use crate::errors::BrainError;
use crate::provider::testing::Answer;

const BOTH: &str =
    r#"{"status":"install-required","assistant_ids":["shimpz-cloudflare","shimpz-whatsapp"]}"#;

fn candidates() -> Value {
    parse(
        r#"[{"id":"shimpz-cloudflare","name":"Shimpz Cloudflare","summary":"Manage DNS zones and records.","actions":["dns.read","dns.write"],"integrations":[{"id":"cloudflare","provider":"cloudflare"}]},{"id":"shimpz-whatsapp","name":"Shimpz WhatsApp","summary":"Send reviewed WhatsApp messages.","actions":["messages.send"],"integrations":[{"id":"whatsapp","provider":"whatsapp"}]}]"#,
    )
}

fn body(objective: &str) -> Value {
    json::object([
        ("provider", testing::provider("openai", "gpt-6.1-sol")),
        ("objective", json::text(objective)),
        ("candidates", candidates()),
    ])
}

fn plan_with(
    answer: Answer,
    body: &Value,
) -> (
    Result<Value, RouteError>,
    Vec<crate::provider::testing::Received>,
) {
    run(vec![answer], |providers, scope| {
        handle(providers, scope, body)
    })
}

#[test]
fn a_plan_is_one_tool_free_call_answering_an_exact_subset() {
    let objective = "Configure um domínio\ne envie o resultado por WhatsApp";
    let (outcome, sent) = plan_with(Answer::openai_text(BOTH), &body(objective));
    assert_eq!(outcome, Ok(parse(BOTH)));
    assert_eq!((sent.len(), sent[0].path.as_str()), (1, "/v1/responses"));
    let request = sent[0].json();
    assert_eq!(
        (request.get("tools"), request.get("reasoning")),
        (None, None)
    );
    let format = request.get("text").and_then(|text| text.get("format"));
    assert_eq!(
        format.and_then(|format| json::str_of(format, "name")),
        Some("StructuredPlan")
    );
    let texts = openai_texts(&sent[0]);
    assert_eq!(texts[0], SYSTEM);
    assert!(texts[1].starts_with(r#"{"objective":"Configure um domínio\ne envie"#));
    assert!(texts[1].contains(r#""id":"shimpz-whatsapp""#));
    let key = KEY.as_bytes();
    assert!(!sent[0].body.windows(key.len()).any(|window| window == key));
}

#[test]
fn a_sufficient_plan_selects_nothing_on_anthropic_too() {
    let body = with(
        &body("Explain DNS"),
        "provider",
        testing::provider("anthropic", "claude-sonnet-5-5"),
    );
    let answer = Answer::anthropic_text(r#"{"status":"sufficient","assistant_ids":[]}"#);
    let (outcome, sent) = plan_with(answer, &body);
    assert_eq!(
        outcome,
        Ok(parse(r#"{"status":"sufficient","assistant_ids":[]}"#))
    );
    let (system, user) = anthropic_texts(&sent[0]);
    assert_eq!(system, SYSTEM);
    assert!(user.contains("Explain DNS"));
}

#[test]
fn an_answer_outside_the_shortlist_or_its_status_is_a_provider_failure() {
    let invalid = [
        r#"{"status":"install-required","assistant_ids":["unknown"]}"#,
        r#"{"status":"install-required","assistant_ids":["shimpz-cloudflare","shimpz-cloudflare"]}"#,
        r#"{"status":"install-required","assistant_ids":["shimpz-whatsapp","shimpz-cloudflare"]}"#,
        r#"{"status":"sufficient","assistant_ids":["shimpz-cloudflare"]}"#,
        r#"{"status":"install-required","assistant_ids":[]}"#,
        r#"{"status":"install-required","assistant_ids":["shimpz-cloudflare"],"extra":true}"#,
        r#"{"status":"unknown","assistant_ids":[]}"#,
        "not-json",
    ];
    for answer in invalid {
        let (outcome, _) = plan_with(Answer::openai_text(answer), &body("Configure DNS"));
        assert_eq!(outcome, Err(provider_failed()), "{answer}");
    }
    let oversized = format!(
        r#"{{"status":"sufficient",{}"assistant_ids":[]}}"#,
        " ".repeat(MAX_RESPONSE_CHARS)
    );
    let (outcome, _) = plan_with(Answer::openai_text(&oversized), &body("Configure DNS"));
    assert_eq!(outcome, Err(provider_failed()));
}

#[test]
fn more_than_four_selections_are_refused() {
    let five: Vec<Value> = (0..5)
        .map(|index| {
            parse(&format!(
                r#"{{"id":"assistant-{index}","name":"Assistant {index}","summary":"Bounded capability.","actions":["action-{index}"],"integrations":[]}}"#
            ))
        })
        .collect();
    let body = with(
        &body("Use all capabilities"),
        "candidates",
        json::array(five),
    );
    let ids: Vec<String> = (0..5)
        .map(|index| format!(r#""assistant-{index}""#))
        .collect();
    let answer = format!(
        r#"{{"status":"install-required","assistant_ids":[{}]}}"#,
        ids.join(",")
    );
    assert_eq!(
        plan_with(Answer::openai_text(&answer), &body).0,
        Err(provider_failed())
    );
}

fn refused_before_any_call(body: &Value) -> Result<Value, RouteError> {
    let (outcome, sent) = plan_with(Answer::openai_text(BOTH), body);
    assert!(sent.is_empty(), "no provider request");
    outcome
}

#[test]
fn invalid_inputs_are_contract_refusals_before_provider_access() {
    let Value::Array(items) = candidates() else {
        unreachable!()
    };
    let cases = [
        (body("hidden\0instruction"), "invalid capability objective"),
        (body(" padded"), "invalid capability objective"),
        (
            with(
                &body("Configure DNS"),
                "candidates",
                json::array(items.iter().rev().cloned().collect()),
            ),
            "invalid capability candidate order",
        ),
        (
            with(
                &body("Configure DNS"),
                "candidates",
                json::array(vec![items[0].clone(), items[0].clone()]),
            ),
            "invalid capability candidate order",
        ),
        (
            with(
                &body("Configure DNS"),
                "candidates",
                json::array(vec![with(&items[0], "actions", parse(r#"["../shell"]"#))]),
            ),
            "invalid Action id",
        ),
        (
            with(
                &body("Configure DNS"),
                "candidates",
                json::array(vec![with(
                    &items[0],
                    "actions",
                    parse(r#"["dns.write","dns.read"]"#),
                )]),
            ),
            "invalid capability candidate Actions",
        ),
    ];
    for (request, detail) in cases {
        assert_eq!(
            refused_before_any_call(&request),
            Err(contract(detail)),
            "{detail}"
        );
    }
}

/// The request naming only the first candidate, with `changes`.
fn first_with(changes: &[(&str, Value)]) -> Value {
    let Value::Array(items) = candidates() else {
        unreachable!()
    };
    let item = changes.iter().fold(items[0].clone(), |item, (key, value)| {
        with(&item, key, value.clone())
    });
    with(
        &body("Configure DNS"),
        "candidates",
        json::array(vec![item]),
    )
}

fn integration(id: &str, provider: &str) -> Value {
    json::array(vec![json::object([
        ("id", json::text(id)),
        ("provider", json::text(provider)),
    ])])
}

#[test]
fn the_widest_identifiers_of_each_kind_are_admitted() {
    let widest = first_with(&[
        ("id", json::text(&"a".repeat(40))),
        (
            "actions",
            json::array(vec![json::text("dns.read"), json::text(&"z".repeat(128))]),
        ),
        (
            "integrations",
            integration(&"i".repeat(64), &"p".repeat(64)),
        ),
    ]);
    let answer = format!(
        r#"{{"status":"install-required","assistant_ids":["{}"]}}"#,
        "a".repeat(40)
    );
    assert!(plan_with(Answer::openai_text(&answer), &widest).0.is_ok());
}

#[test]
fn candidate_identifiers_follow_the_team_protocol_by_kind() {
    for (change, detail) in [
        (("id", json::text(&"a".repeat(41))), "invalid Assistant id"),
        (("id", json::text("dns.read")), "invalid Assistant id"),
        (
            ("integrations", integration(&"i".repeat(65), "p")),
            "invalid Integration id",
        ),
        (
            ("integrations", integration("i", "p.x")),
            "invalid Integration provider",
        ),
        (
            ("name", json::text("Shimpz\nCloudflare")),
            "invalid Assistant name",
        ),
    ] {
        let refused = refused_before_any_call(&first_with(&[change]));
        assert_eq!(refused, Err(contract(detail)), "{detail}");
    }
}

#[test]
fn a_malformed_request_is_refused_whole() {
    let base = body("Configure DNS");
    let provider = testing::provider("openai", "gpt-6.1-sol");
    for request in [
        without(&base, "candidates"),
        with(&base, "extra", Value::Null),
        with(&base, "objective", json::text("")),
        with(&base, "candidates", json::array(Vec::new())),
        first_with(&[("region", json::text("us"))]),
        first_with(&[("actions", json::array(Vec::new()))]),
        first_with(&[("summary", json::text(&"s".repeat(81)))]),
        with(
            &base,
            "provider",
            with(&provider, "region", json::text(KEY)),
        ),
    ] {
        assert_eq!(refused_before_any_call(&request), Err(RouteError::Invalid));
    }
}

#[test]
fn a_foreign_model_is_refused_after_the_shape_and_before_the_contract() {
    let foreign = with(
        &body("Configure DNS"),
        "provider",
        testing::provider("anthropic", "claude-haiku-5-5"),
    );
    let unsupported = Err(contract("unsupported model for provider"));
    assert_eq!(refused_before_any_call(&foreign), unsupported);
    let both = with(&foreign, "objective", json::text(" padded"));
    assert_eq!(refused_before_any_call(&both), unsupported);
    let malformed = with(&foreign, "objective", json::text(""));
    assert_eq!(
        refused_before_any_call(&malformed),
        Err(RouteError::Invalid)
    );
}

#[test]
fn a_provider_failure_is_redacted_and_a_cancelled_plan_says_so() {
    let (outcome, _) = plan_with(
        Answer::raw(400, "secret provider detail"),
        &body("Configure DNS"),
    );
    assert_eq!(outcome, Err(provider_failed()));
    let request = body("Configure DNS");
    let (outcome, sent) = testing::cancelled_after(
        1,
        |_| Answer::hang(),
        |providers, scope| handle(providers, scope, &request),
    );
    assert_eq!(
        outcome,
        Err(RouteError::Brain(BrainError::Cancelled(CANCELLED)))
    );
    assert_eq!(sent.len(), 1);
}

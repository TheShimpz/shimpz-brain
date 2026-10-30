//! The fast path inside the route: what it skips, what falls back, and cancellation in both branches.

use std::time::Duration;

use super::*;
use crate::decisions::testing::cancelled_after;

const CONFIDENT: &str = r#"{"model":"jev-1.13.0","answers":{"intent":{"type":"choice","choice":"ordinary-task","confidence":0.9,"probabilities":{"ordinary-task":0.9,"assistant-install":0.05,"assistant-uninstall":0.03,"unresolved":0.02}}}}"#;

fn with_need(need: &str) -> Value {
    let capability = format!(
        r#"{{"type":"choice","choice":"{need}","confidence":0.9,"probabilities":{{"{need}":0.9,"{}":0.1}}}}"#,
        if need == "conversation" {
            "capability"
        } else {
            "conversation"
        }
    );
    let base = parse(CONFIDENT);
    let answers = with(
        base.get("answers").unwrap(),
        "capability",
        parse(&capability),
    );
    with(&base, "answers", answers)
}

fn installable_route() -> Value {
    let installable = parse(r#"[{"name":"Cloudflare","summary":"Set up your domains' DNS."}]"#);
    with(
        &decided(&classification("Oi, tudo bem?")),
        "installable",
        installable,
    )
}

#[test]
fn a_described_capability_need_reaches_the_answer() {
    let (outcome, sent) = routed(
        vec![Answer::json(200, &with_need("conversation"))],
        &installable_route(),
    );
    let routed_answer = outcome.unwrap();
    assert_eq!(
        json::str_of(&routed_answer, "capability"),
        Some("not-needed")
    );
    let state = sent[0].json().get("state").cloned().unwrap();
    assert_eq!(
        state.get("installable_assistants"),
        Some(&parse(
            r#"[{"name":"Cloudflare","summary":"Set up your domains' DNS."}]"#
        ))
    );
    let (outcome, _) = routed(
        vec![Answer::json(200, &with_need("capability"))],
        &installable_route(),
    );
    assert_eq!(
        json::str_of(&outcome.unwrap(), "capability"),
        Some("possible")
    );
    let reply = answer(INSTALL, "cloudflare", &[], "", false);
    let jev = Answer::json(200, &parse(CONFIDENT));
    let (outcome, sent) = routed(vec![jev, Answer::openai_text(&reply)], &installable_route());
    assert_eq!(
        sent.len(),
        2,
        "an unanswered capability question falls back to the LLM route"
    );
    assert_eq!(
        json::str_of(&outcome.unwrap(), "capability"),
        Some("possible")
    );
}

#[test]
fn without_installable_assistants_the_answer_keeps_its_exact_shape() {
    let (outcome, _) = routed(
        vec![Answer::json(200, &parse(CONFIDENT))],
        &decided(&classification("Oi")),
    );
    assert_eq!(outcome.unwrap().get("capability"), None);
}

#[test]
fn any_other_fast_path_outcome_runs_the_unchanged_llm_route() {
    let reply = answer(INSTALL, "cloudflare", &[], "", false);
    for jev in [Answer::raw(503, "overloaded"), Answer::raw(200, "not json")] {
        let (outcome, sent) = routed(
            vec![jev, Answer::openai_text(&reply)],
            &decided(&classification("Instala o Cloudflare")),
        );
        assert_eq!(outcome, Ok(response(INSTALL, "cloudflare", &[], "", false)));
        assert_eq!(
            (sent[0].path.as_str(), sent[1].path.as_str()),
            (jev::PATH, "/v1/responses")
        );
        assert_eq!(
            sent[1].header("authorization"),
            Some(format!("Bearer {}", testing::KEY).as_str())
        );
    }
}

#[test]
fn invalid_input_never_reaches_the_fast_path() {
    assert_eq!(
        refused_before_any_call(&decided(&classification("   "))),
        Err(contract("invalid route objective"))
    );
    let candidate = with(
        &decided(&classification("Oi")),
        "candidates",
        candidates(false),
    );
    assert_eq!(
        refused_before_any_call(&candidate),
        Err(contract(
            "classification cannot include directory candidates"
        ))
    );
}

#[test]
fn a_route_cancelled_during_the_jev_exchange_never_falls_back() {
    let body = decided(&classification("Oi"));
    let (outcome, sent) = cancelled_after(
        1,
        |_| Answer::hang(),
        |providers, scope| handle(providers, scope, &body),
    );
    assert_eq!(
        outcome,
        Err(RouteError::Brain(BrainError::Cancelled(CANCELLED)))
    );
    assert_eq!(
        sent.len(),
        1,
        "no LLM request after the cancelled Jev exchange"
    );
}

#[test]
fn a_route_cancelled_during_the_llm_fallback_sends_nothing_more() {
    let body = decided(&classification("Oi"));
    let answers = |index: usize| {
        if index == 0 {
            Answer::raw(503, "overloaded")
        } else {
            Answer::hang()
        }
    };
    let (outcome, sent) = cancelled_after(2, answers, |providers, scope| {
        handle(providers, scope, &body)
    });
    assert_eq!(
        outcome,
        Err(RouteError::Brain(BrainError::Cancelled(CANCELLED)))
    );
    assert_eq!(
        sent.len(),
        2,
        "no retry or later request after cancellation"
    );
    let plain = classification("Oi");
    let (outcome, sent) = cancelled_after(
        1,
        |_| Answer::hang(),
        |providers, scope| handle(providers, scope, &plain),
    );
    assert_eq!(
        (outcome, sent.len()),
        (Err(RouteError::Brain(BrainError::Cancelled(CANCELLED))), 1)
    );
    let _ = Duration::ZERO;
}

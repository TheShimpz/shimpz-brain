//! Tests of one logical provider call: usage, retries, admission, and counting over the fake origin.

use std::time::Duration;

use super::testing::{self, Answer, Harness, Origin};
use super::*;
use crate::catalog;

fn hello() -> ChatRequest {
    ChatRequest {
        system: None,
        messages: vec![Message::User {
            id: "u".into(),
            text: "hello".into(),
        }],
        tools: Vec::new(),
        output: None,
        cache: false,
    }
}

fn chat(
    harness: &Harness,
    scope: &Scope,
    profile: Profile,
    admit: Option<Admit<'_>>,
) -> Result<AssistantMessage, CallError> {
    let config = testing::config(ProviderId::OpenAi);
    harness
        .providers
        .chat(scope, &config, &hello(), None, profile, admit)
}

fn soon(status: u16) -> Answer {
    Answer::raw(status, "{}").header("retry-after", "0")
}

#[test]
fn reported_unreported_and_failed_calls_are_counted_apart() {
    let mut reported = testing::openai_body("ok");
    if let Value::Object(members) = &mut reported {
        wire::set(
            members,
            "usage",
            json::parse(br#"{"input_tokens":120,"output_tokens":30,"input_tokens_details":{"cached_tokens":80,"cache_write_tokens":20}}"#)
                .unwrap(),
        );
    }
    let mut unreported = testing::openai_body("ok");
    if let Value::Object(members) = &mut unreported {
        wire::take(members, "usage");
    }
    let origin = Origin::scripted([
        Answer::json(200, &reported),
        Answer::json(200, &unreported),
        Answer::raw(400, "{}"),
        Answer::raw(200, "not json"),
    ]);
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    for expected_ok in [true, true, false, false] {
        assert_eq!(
            chat(&harness, &scope, Profile::CHAT, None).is_ok(),
            expected_ok
        );
    }
    assert_eq!(scope.usage.counts(), [4, 2, 1, 120, 30, 80, 20, 4]);
    assert_eq!(
        json::compact(&scope.usage.to_value()),
        r#"{"model_calls":4,"failed_calls":2,"unreported_calls":1,"input_tokens":120,"output_tokens":30,"cache_read_tokens":80,"cache_write_tokens":20,"provider_requests":4}"#
    );
}

#[test]
fn every_request_of_a_retried_call_is_counted_and_the_turn_records_its_attempts() {
    let origin = Origin::scripted([soon(500), Answer::openai_text("Done.")]);
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    let turn = chat(&harness, &scope, Profile::CHAT, None).unwrap();
    assert_eq!((turn.text.as_str(), turn.attempts), ("Done.", 2));
    let [calls, failed, .., requests] = scope.usage.counts();
    assert_eq!((calls, failed, requests), (1, 0, 2));
}

#[test]
fn a_stalled_decision_is_retried_exactly_once() {
    let origin = Origin::start(|_| Answer::hang());
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    let stalled = Profile {
        timeout: Duration::from_millis(200),
        ..Profile::DECISION
    };
    assert_eq!(
        chat(&harness, &scope, stalled, None),
        Err(CallError::Request)
    );
    assert_eq!(origin.requests().len(), 2);
    assert_eq!(scope.usage.counts()[..2], [1, 1]);
}

#[test]
fn admission_runs_before_every_attempt_and_its_refusal_sends_nothing_more() {
    let origin = Origin::scripted([soon(503), Answer::openai_text("Done.")]);
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    let seen = std::sync::Mutex::new(Vec::new());
    let admit = |attempt: u32| {
        seen.lock().unwrap().push(attempt);
        if attempt < 2 {
            Ok(())
        } else {
            Err("attachments exceed the turn's token ceiling".to_owned())
        }
    };
    assert_eq!(
        chat(&harness, &scope, Profile::CHAT, Some(&admit)),
        Err(CallError::Contract(
            "attachments exceed the turn's token ceiling".into()
        ))
    );
    assert_eq!(*seen.lock().unwrap(), [1, 2]);
    assert_eq!(origin.requests().len(), 1);
}

#[test]
fn counting_makes_one_attempt_on_the_provider_endpoint_and_is_no_model_call() {
    let origin = Origin::scripted([Answer::counted(300), Answer::counted(7), soon(500)]);
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    let blocks = [Block::Text("Attached".into())];
    let count = |provider| {
        harness.providers.count_tokens(
            &scope,
            &testing::config(provider),
            &blocks,
            Duration::from_secs(5),
        )
    };
    assert_eq!(count(ProviderId::OpenAi), Ok(300));
    assert_eq!(count(ProviderId::Anthropic), Ok(7));
    assert_eq!(count(ProviderId::OpenAi), Err(CallError::Request));
    let paths: Vec<_> = origin
        .requests()
        .into_iter()
        .map(|request| request.path)
        .collect();
    assert_eq!(
        paths,
        [
            "/v1/responses/input_tokens",
            "/v1/messages/count_tokens",
            "/v1/responses/input_tokens"
        ]
    );
    assert_eq!(scope.usage.counts(), [0, 0, 0, 0, 0, 0, 0, 3]);
}

#[test]
fn every_catalog_provider_has_an_adapter() {
    for provider in catalog::all() {
        let id = ProviderId::parse(&provider.id).expect("an adapter");
        assert_eq!(id.as_str(), provider.id);
    }
    assert_eq!(catalog::all().len(), 2);
}

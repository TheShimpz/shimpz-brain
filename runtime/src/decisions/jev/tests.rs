//! The Jev fast path: exact request, closed response, and fallback on every non-confident or failed outcome
//! (`test_intent_fast_path.py`).

use std::time::Instant;

use super::*;
use crate::decisions::testing::{cancelled_after, parse as json_parse};
use crate::provider::testing::{Answer, Harness, Origin, Received};

const KEY: &str = "tsk-test-0123456789abcdef";

fn intent_answer(choice: &str, confidence: f64) -> Value {
    json_parse(&format!(
        r#"{{"type":"choice","choice":"{choice}","confidence":{confidence},"probabilities":{{"ordinary-task":0.9,"assistant-install":0.05,"assistant-uninstall":0.03,"unresolved":0.02}}}}"#
    ))
}

fn response(intent: Value, capability: Option<Value>) -> Value {
    let mut answers = vec![("intent".to_owned(), intent)];
    if let Some(capability) = capability {
        answers.push(("capability".to_owned(), capability));
    }
    json::object([
        ("model", json::text(MODEL)),
        ("answers", json::object_of(answers)),
        ("usage", json_parse(r#"{"input_tokens":9}"#)),
    ])
}

fn need(need: &str, confidence: f64) -> Value {
    let other = if need == "conversation" {
        "capability"
    } else {
        "conversation"
    };
    json_parse(&format!(
        r#"{{"type":"choice","choice":"{need}","confidence":{confidence},"probabilities":{{"{need}":{confidence},"{other}":{}}}}}"#,
        1.0 - confidence
    ))
}

fn ordinary() -> Value {
    response(intent_answer("ordinary-task", 0.9), None)
}

/// `value` with the member at `path` set to `member` (an empty path replaces the whole value).
fn set(value: &Value, path: &[&str], member: Value) -> Value {
    let Some((first, rest)) = path.split_first() else {
        return member;
    };
    let mut members = json::members(value).unwrap_or_default().to_vec();
    match members.iter_mut().find(|(name, _)| name == first) {
        Some((_, inner)) => *inner = set(inner, rest, member),
        None => members.push(((*first).to_owned(), set(&Value::Null, rest, member))),
    }
    Value::Object(members)
}

/// Classifies "Oi!" against an origin answering `answer`; the outcome and the requests received.
fn ask(answer: Answer, installable: Vec<(&str, &str)>) -> (Option<FastPath>, Vec<Received>) {
    let origin = Origin::scripted([answer]);
    let harness = Harness::new(&origin.route());
    let question = Question {
        objective: "Oi!",
        installable,
        ..Question::default()
    };
    let outcome = classify(
        harness.providers.transport(),
        &Scope::default(),
        KEY,
        &question,
    );
    (outcome, origin.requests())
}

#[test]
fn the_request_carries_the_pinned_model_the_current_message_and_bounded_context() {
    let bare = request_body(&Question {
        objective: "Oi!",
        ..Question::default()
    });
    assert_eq!(json::str_of(&bare, "model"), Some("jev-1.13.0"));
    assert_eq!(
        bare.get("state"),
        Some(&json_parse(r#"{"current_message":"Oi!"}"#))
    );
    let questions = bare.get("questions").and_then(json::members).unwrap();
    assert_eq!(questions.len(), 1);
    let criteria = questions[0]
        .1
        .get("criteria")
        .and_then(json::members)
        .unwrap();
    assert_eq!(
        criteria
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        INTENTS
    );
    let context = request_body(&Question {
        objective: "Remove ele",
        conversation: vec![("user", "Instala o Cloudflare")],
        reference: Some("Shimpz Cloudflare"),
        installable: Vec::new(),
    });
    assert_eq!(
        context.get("state"),
        Some(&json_parse(
            r#"{"current_message":"Remove ele","earlier_conversation":[{"role":"user","text":"Instala o Cloudflare"}],"last_lifecycle_assistant":"Shimpz Cloudflare"}"#
        ))
    );
}

#[test]
fn installable_assistants_add_their_summaries_and_the_capability_question() {
    let described = request_body(&Question {
        objective: "Oi!",
        installable: vec![("Cloudflare", "Set up your domains' DNS.")],
        ..Question::default()
    });
    assert_eq!(
        described.get("state"),
        Some(&json_parse(
            r#"{"current_message":"Oi!","installable_assistants":[{"name":"Cloudflare","summary":"Set up your domains' DNS."}]}"#
        ))
    );
    let questions = described.get("questions").unwrap();
    assert_eq!(questions.get("intent"), Some(&*INTENT_QUESTION));
    let needs = questions
        .get("capability")
        .and_then(|question| question.get("criteria"));
    let needs = needs.and_then(json::members).unwrap();
    assert_eq!(
        needs
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        NEEDS
    );
}

fn assert_unparsed(values: Vec<Value>) {
    for value in values {
        assert_eq!(parse(&value, false), None, "{}", json::compact(&value));
    }
}

#[test]
fn only_the_exact_response_shape_parses() {
    assert_eq!(
        parse(&ordinary(), false).map(|answers| answers.intent),
        Some(("ordinary-task".to_owned(), 0.9))
    );
    let base = ordinary();
    let intent = ["answers", "intent"];
    let at = |key: &'static str| [&intent[..], &[key]].concat();
    assert_unparsed(vec![
        set(&base, &["model"], json::text("jev-latest")),
        set(&base, &["extra"], json::int(1)),
        set(&base, &["answers", "other"], json_parse("{}")),
        set(&base, &at("type"), json::text("score")),
        set(&base, &at("choice"), json::text("delete-everything")),
        set(&base, &at("confidence"), Value::Bool(true)),
        set(&base, &at("confidence"), json_parse("1.5")),
        set(&base, &at("extra"), json::int(1)),
        json_parse("null"),
        json_parse("[]"),
        json_parse(r#"{"model":"jev-1.13.0","answers":[]}"#),
    ]);
}

#[test]
fn only_a_consistent_distribution_parses() {
    let probabilities = ["answers", "intent", "probabilities"];
    let with_distribution = |text: &str| set(&ordinary(), &probabilities, json_parse(text));
    assert_unparsed(vec![
        set(
            &ordinary(),
            &[&probabilities[..], &["ordinary-task"]].concat(),
            json_parse("-0.1"),
        ),
        // ordinary-task claimed while the distribution puts it at zero.
        with_distribution(
            r#"{"ordinary-task":0.0,"assistant-install":1.0,"assistant-uninstall":0.0,"unresolved":0.0}"#,
        ),
        with_distribution(
            r#"{"ordinary-task":0.4,"assistant-install":0.1,"assistant-uninstall":0.1,"unresolved":0.1}"#,
        ),
        with_distribution(
            r#"{"ordinary-task":0.9,"assistant-install":0.05,"assistant-uninstall":0.05}"#,
        ),
    ]);
}

#[test]
fn a_response_must_answer_exactly_the_asked_questions() {
    let both = response(
        intent_answer("ordinary-task", 0.9),
        Some(need("conversation", 0.9)),
    );
    let parsed = parse(&both, true).unwrap();
    assert_eq!(parsed.capability, Some(("conversation".to_owned(), 0.9)));
    assert_eq!(
        parse(&both, false),
        None,
        "an unasked answer is as invalid as a missing one"
    );
    assert_eq!(parse(&ordinary(), true), None);
    let capability = ["answers", "capability"];
    for value in [
        response(
            intent_answer("ordinary-task", 0.9),
            Some(need("delete-everything", 0.9)),
        ),
        set(
            &both,
            &[&capability[..], &["type"]].concat(),
            json::text("noul"),
        ),
        set(
            &both,
            &[&capability[..], &["probabilities"]].concat(),
            json_parse(r#"{"conversation":0.1,"capability":0.9}"#),
        ),
    ] {
        assert_eq!(parse(&value, true), None);
    }
}

#[test]
fn only_a_confident_ordinary_task_takes_the_fast_path() {
    let (outcome, sent) = ask(Answer::json(200, &ordinary()), Vec::new());
    assert_eq!(
        outcome,
        Some(FastPath {
            ordinary: true,
            capability_free: false
        })
    );
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].path, PATH);
    assert_eq!(
        sent[0].header("authorization"),
        Some(format!("Bearer {KEY}").as_str())
    );
    assert_eq!(sent[0].header("accept-encoding"), Some("identity"));
    assert_eq!(sent[0].header("host"), Some("api.typesafe.ai"));
    assert_eq!(
        sent[0].json().get("state"),
        Some(&json_parse(r#"{"current_message":"Oi!"}"#))
    );
    for intent in [
        intent_answer("ordinary-task", CONFIDENCE_THRESHOLD - 0.01),
        set(
            &intent_answer("assistant-install", 1.0),
            &["probabilities"],
            json_parse(
                r#"{"ordinary-task":0.0,"assistant-install":1.0,"assistant-uninstall":0.0,"unresolved":0.0}"#,
            ),
        ),
        set(
            &intent_answer("unresolved", 1.0),
            &["probabilities"],
            json_parse(
                r#"{"ordinary-task":0.0,"assistant-install":0.0,"assistant-uninstall":0.0,"unresolved":1.0}"#,
            ),
        ),
    ] {
        let (outcome, _) = ask(Answer::json(200, &response(intent, None)), Vec::new());
        assert_eq!(outcome, Some(FastPath::default()));
    }
}

#[test]
fn a_described_capability_question_travels_in_the_same_request() {
    let installable = vec![("Cloudflare", "Set up your domains' DNS.")];
    let free = response(
        intent_answer("ordinary-task", 0.9),
        Some(need("conversation", 0.9)),
    );
    let (outcome, sent) = ask(Answer::json(200, &free), installable);
    let free_path = FastPath {
        ordinary: true,
        capability_free: true,
    };
    assert_eq!(outcome, Some(free_path));
    let questions = sent[0].json().get("questions").cloned().unwrap();
    assert!(questions.get("capability").is_some());
}

#[test]
fn only_a_confident_ordinary_conversation_is_capability_free() {
    let choice = |option: &str, confidence: f64| (option.to_owned(), confidence);
    let answers = |intent: f64, capability: Option<(&str, f64)>| Answers {
        intent: choice("ordinary-task", intent),
        capability: capability.map(|(need, confidence)| choice(need, confidence)),
    };
    let ordinary = FastPath {
        ordinary: true,
        capability_free: false,
    };
    for (answered, expected) in [
        (
            answers(0.9, Some(("conversation", 0.9))),
            FastPath {
                ordinary: true,
                capability_free: true,
            },
        ),
        (
            answers(0.9, Some(("conversation", CAPABILITY_THRESHOLD - 0.01))),
            ordinary,
        ),
        (answers(0.9, Some(("capability", 0.9))), ordinary),
        (answers(0.9, None), ordinary),
        (
            answers(CONFIDENCE_THRESHOLD - 0.01, Some(("conversation", 0.9))),
            FastPath::default(),
        ),
    ] {
        assert_eq!(decide(Some(&answered)), expected);
    }
    let lifecycle = Answers {
        intent: choice("assistant-install", 1.0),
        capability: Some(choice("conversation", 1.0)),
    };
    assert_eq!(decide(Some(&lifecycle)), FastPath::default());
    assert_eq!(decide(None), FastPath::default());
}

#[test]
fn every_failure_keeps_the_llm_route() {
    let oversized = format!(
        r#"{{"model":"{MODEL}","answers":{{}},"usage":"{}"}}"#,
        " ".repeat(MAX_RESPONSE_BYTES)
    );
    for answer in [
        Answer::json(401, &json_parse(r#"{"error":"invalid key"}"#)),
        Answer::json(529, &json_parse(r#"{"error":"overloaded"}"#)),
        Answer::raw(200, "not json"),
        Answer::json(200, &set(&ordinary(), &["model"], json::text("jev-latest"))),
        Answer::json(200, &ordinary()).header("content-encoding", "gzip"),
        Answer::raw(200, &oversized),
    ] {
        assert_eq!(ask(answer, Vec::new()).0, Some(FastPath::default()));
    }
}

#[test]
fn a_late_answer_is_abandoned_at_the_deadline() {
    let started = Instant::now();
    let (outcome, sent) = ask(
        Answer::json(200, &ordinary()).delayed(TIMEOUT + Duration::from_millis(300)),
        Vec::new(),
    );
    assert_eq!(outcome, Some(FastPath::default()));
    assert_eq!(sent.len(), 1);
    assert!(started.elapsed() < TIMEOUT + Duration::from_millis(250));
}

#[test]
fn a_cancelled_exchange_reports_the_stop_and_sends_nothing_more() {
    let question = Question {
        objective: "Oi!",
        ..Question::default()
    };
    let (outcome, sent) = cancelled_after(
        1,
        |_| Answer::hang(),
        |providers, scope| classify(providers.transport(), scope, KEY, &question),
    );
    assert_eq!(outcome, None);
    assert_eq!(sent.len(), 1);
}

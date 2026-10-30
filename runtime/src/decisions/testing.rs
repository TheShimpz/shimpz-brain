//! Helpers for the decisions' tests: request bodies, a run against the fake provider origin, and readers of what the
//! origin received.

use std::thread;
use std::time::Duration;

use crate::cancel::CancelToken;
use crate::errors::BrainError;
use crate::json::{self, Value};
use crate::provider::testing::{Answer, Harness, Origin, Received};
use crate::provider::{Providers, Scope};
use crate::server::input::RouteError;

/// A low-entropy fake provider key of the admitted length.
pub(super) const KEY: &str = "secret-test-key-0";

/// Parses JSON text.
pub(super) fn parse(text: &str) -> Value {
    json::parse(text.as_bytes()).expect("test JSON")
}

/// A request's `provider` object for `provider` and `model`.
pub(super) fn provider(provider: &str, model: &str) -> Value {
    json::object([
        ("provider", json::text(provider)),
        ("model", json::text(model)),
        ("api_key", json::text(KEY)),
    ])
}

/// `body` with member `key` set to `value` (replaced in place, or appended).
pub(super) fn with(body: &Value, key: &str, value: Value) -> Value {
    let mut members = json::members(body).expect("an object").to_vec();
    match members.iter_mut().find(|(name, _)| name == key) {
        Some((_, member)) => *member = value,
        None => members.push((key.to_owned(), value)),
    }
    Value::Object(members)
}

/// `body` without member `key`.
pub(super) fn without(body: &Value, key: &str) -> Value {
    let members = json::members(body).expect("an object");
    Value::Object(
        members
            .iter()
            .filter(|(name, _)| name != key)
            .cloned()
            .collect(),
    )
}

/// Runs `work` on providers over an origin answering `answers` in order; its outcome and every request received.
pub(super) fn run<T>(
    answers: Vec<Answer>,
    work: impl FnOnce(&Providers, &Scope) -> T,
) -> (T, Vec<Received>) {
    let origin = Origin::scripted(answers);
    let harness = Harness::new(&origin.route());
    let outcome = work(&harness.providers, &Scope::default());
    (outcome, origin.requests())
}

/// Runs `work` against an origin that never answers, cancelling its scope once `requests` requests arrived; the
/// outcome and the requests the origin received by then and a moment later.
pub(super) fn cancelled_after<T: Send>(
    requests: usize,
    answer: impl Fn(usize) -> Answer + Send + Sync + 'static,
    work: impl FnOnce(&Providers, &Scope) -> T + Send,
) -> (T, Vec<Received>) {
    let seen = std::sync::atomic::AtomicUsize::new(0);
    let origin =
        Origin::start(move |_| answer(seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
    let harness = Harness::new(&origin.route());
    let token = CancelToken::new();
    let scope = Scope::new(token.clone());
    let outcome = thread::scope(|threads| {
        threads.spawn(|| {
            assert!(
                origin.wait_requests(requests),
                "the cancelled request arrived"
            );
            token.cancel();
        });
        work(&harness.providers, &scope)
    });
    thread::sleep(Duration::from_millis(200));
    (outcome, origin.requests())
}

/// A request's 400 refusal with `detail`.
pub(super) fn contract(detail: &str) -> RouteError {
    RouteError::Brain(BrainError::contract(detail))
}

/// The 502 provider failure.
pub(super) fn provider_failed() -> RouteError {
    RouteError::Brain(BrainError::Provider)
}

/// The text of every message of an OpenAI Responses request, in order, each content part on its own line.
pub(super) fn openai_texts(received: &Received) -> Vec<String> {
    let body = received.json();
    json::array_of(&body, "input")
        .unwrap_or_default()
        .iter()
        .map(|item| match item.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| json::str_of(part, "text"))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        })
        .collect()
}

/// The system prompt and the user text of an Anthropic Messages request.
pub(super) fn anthropic_texts(received: &Received) -> (String, String) {
    let body = received.json();
    let system = match body.get("system") {
        Some(Value::String(text)) => text.clone(),
        _ => String::new(),
    };
    let user = json::array_of(&body, "messages")
        .unwrap_or_default()
        .iter()
        .map(|message| match message.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(blocks)) => blocks
                .iter()
                .filter_map(|block| json::str_of(block, "text"))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    (system, user)
}

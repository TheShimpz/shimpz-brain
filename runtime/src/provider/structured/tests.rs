//! Tests of structured output through both real adapters over the fake origin.

use super::*;
use crate::json;
use crate::provider::testing::{self, Answer, Harness, Origin, anthropic_content, openai_output};

const PLAN: &str = r#"{"status":"install-required","assistant_ids":["shimpz-cloudflare"]}"#;

fn parse(text: &str) -> Value {
    json::parse(text.as_bytes()).unwrap()
}

fn schema() -> Value {
    parse(
        r#"{"additionalProperties":false,"properties":{"status":{"enum":["sufficient","install-required"],"title":"Status","type":"string"},"assistant_ids":{"items":{"type":"string"},"maxItems":4,"title":"Assistant Ids","type":"array"}},"required":["status","assistant_ids"],"title":"StructuredPlan","type":"object"}"#,
    )
}

/// Asks once with `user`, answered by `answer`; the outcome and the one request sent.
fn ask_with(
    provider: ProviderId,
    answer: Answer,
    user: Vec<Block>,
    max_chars: usize,
) -> (Result<Value, CallError>, Value) {
    let origin = Origin::scripted([answer]);
    let harness = Harness::new(&origin.route());
    let schema = schema();
    let ask = Ask {
        system: Some("SYS"),
        user,
        name: "StructuredPlan",
        schema: &schema,
        max_chars,
    };
    let outcome = value(
        &harness.providers,
        &Scope::default(),
        &testing::config(provider),
        Profile::DECISION,
        &ask,
    );
    let sent = origin.requests()[0].json();
    (outcome, sent)
}

fn ask(provider: ProviderId, answer: Answer) -> Result<Value, CallError> {
    ask_with(provider, answer, vec![Block::Text("USER".into())], 1000).0
}

fn anthropic_texts(texts: &[&str], stop: &str) -> Answer {
    let blocks = texts
        .iter()
        .map(|text| json::object([("type", json::text("text")), ("text", json::text(text))]))
        .collect();
    Answer::json(200, &anthropic_content(blocks, stop))
}

fn openai_texts(texts: &[&str]) -> Answer {
    let parts = texts
        .iter()
        .map(|text| {
            json::object([
                ("type", json::text("output_text")),
                ("text", json::text(text)),
            ])
        })
        .collect();
    let message = json::object([
        ("type", json::text("message")),
        ("content", json::array(parts)),
    ]);
    Answer::json(200, &openai_output(vec![message]))
}

#[test]
fn anthropic_answers_with_the_native_schema_value() {
    let (outcome, sent) = ask_with(
        ProviderId::Anthropic,
        Answer::anthropic_text(PLAN),
        vec![Block::Text("USER".into())],
        1000,
    );
    assert_eq!(outcome, Ok(parse(PLAN)));
    let format = sent
        .get("output_config")
        .and_then(|config| config.get("format"));
    assert_eq!(json::str_of(format.unwrap(), "type"), Some("json_schema"));
    assert_eq!(sent.get("tools"), None);
    assert_eq!(json::str_of(&sent, "system"), Some("SYS"));
    assert_eq!(
        sent.get("messages"),
        Some(&parse(r#"[{"role":"user","content":"USER"}]"#))
    );
}

#[test]
fn openai_answers_with_the_strict_named_schema_value() {
    let (outcome, sent) = ask_with(
        ProviderId::OpenAi,
        Answer::openai_text(&format!(" {PLAN}\n")),
        vec![Block::Text("USER".into())],
        1000,
    );
    assert_eq!(outcome, Ok(parse(PLAN)));
    let format = sent
        .get("text")
        .and_then(|text| text.get("format"))
        .unwrap();
    assert_eq!(json::str_of(format, "name"), Some("StructuredPlan"));
    assert_eq!(format.get("strict"), Some(&Value::Bool(true)));
    assert_eq!(sent.get("tools"), None);
}

#[test]
fn malformed_refused_and_tool_answers_are_response_failures() {
    let refused = [
        anthropic_texts(&["not json"], "end_turn"),
        anthropic_texts(&["[1]"], "end_turn"),
        anthropic_texts(&[], "refusal"),
        anthropic_texts(
            &[r#"{"status":"sufficient","status":"install-required"}"#],
            "end_turn",
        ),
        anthropic_texts(
            &[&format!(r#"{{"status":"x",{}"a":1}}"#, " ".repeat(5000))],
            "end_turn",
        ),
        anthropic_texts(&["  "], "end_turn"),
        Answer::json(
            200,
            &anthropic_content(
                vec![parse(
                    r#"{"type":"tool_use","id":"t","name":"x","input":{}}"#,
                )],
                "tool_use",
            ),
        ),
    ];
    for answer in refused {
        assert_eq!(ask(ProviderId::Anthropic, answer), Err(CallError::Response));
    }
    let refusal =
        parse(r#"{"type":"message","content":[{"type":"refusal","refusal":"cannot comply"}]}"#);
    let call = parse(r#"{"type":"function_call","call_id":"c","name":"x","arguments":"{}"}"#);
    for output in [vec![refusal], vec![call], vec![]] {
        let answer = Answer::json(200, &openai_output(output));
        assert_eq!(ask(ProviderId::OpenAi, answer), Err(CallError::Response));
    }
}

#[test]
fn a_value_written_twice_agrees_but_two_different_values_never_do() {
    let other = r#"{"assistant_ids":["shimpz-cloudflare"],"status":"sufficient"}"#;
    let reordered = r#"{"assistant_ids":["shimpz-cloudflare"],"status":"install-required"}"#;
    assert_eq!(
        ask(ProviderId::OpenAi, openai_texts(&[PLAN, reordered])),
        Ok(parse(PLAN))
    );
    let twice = PLAN.chars().count() * 2;
    let outcome = |texts: &[&str], bound| {
        ask_with(
            ProviderId::Anthropic,
            anthropic_texts(texts, "end_turn"),
            vec![Block::Text("U".into())],
            bound,
        )
        .0
    };
    assert_eq!(outcome(&[PLAN, PLAN], twice), Ok(parse(PLAN)));
    assert_eq!(outcome(&[other, PLAN], twice), Err(CallError::Response));
    assert_eq!(outcome(&[PLAN, PLAN], twice - 1), Err(CallError::Response));
    assert_eq!(outcome(&[PLAN, "{"], twice), Err(CallError::Response));
}

#[test]
fn a_failed_request_is_a_request_failure() {
    assert_eq!(
        ask(ProviderId::OpenAi, Answer::raw(400, "{}")),
        Err(CallError::Request)
    );
    assert_eq!(
        ask(ProviderId::Anthropic, Answer::raw(200, "not json")),
        Err(CallError::Response)
    );
}

#[test]
fn image_content_follows_the_text_of_the_one_user_message() {
    let user = vec![
        Block::Text("Page".into()),
        Block::Image {
            media_type: "image/png".into(),
            base64: "iVBORw0KGgo=".into(),
        },
    ];
    let (_, sent) = ask_with(
        ProviderId::Anthropic,
        Answer::anthropic_text(PLAN),
        user,
        1000,
    );
    assert_eq!(
        sent.get("messages"),
        Some(&parse(
            r#"[{"role":"user","content":[{"type":"text","text":"Page"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]}]"#
        ))
    );
}

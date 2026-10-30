//! Tests of the OpenAI adapter: the request bodies OpenAI receives and the answers read back.

use super::*;
use crate::conversation::ToolMessage;
use crate::provider::testing::openai_output;
use crate::provider::{OutputSchema, ToolSpec};

const MODEL: &str = "gpt-6.1-sol";

fn parse(text: &str) -> Value {
    json::parse(text.as_bytes()).unwrap()
}

fn body(wire: &Wire) -> Value {
    json::parse(&wire.body).unwrap()
}

fn user(id: &str, text: &str) -> Message {
    Message::User {
        id: id.into(),
        text: text.into(),
    }
}

fn chat(messages: Vec<Message>) -> ChatRequest {
    ChatRequest {
        system: Some("You are the Brain.".into()),
        messages,
        tools: Vec::new(),
        output: None,
        cache: false,
    }
}

fn input_of(wire: &Wire) -> Vec<Value> {
    json::array_of(&body(wire), "input").unwrap().to_vec()
}

fn of_type<'a>(items: &'a [Value], kind: &str) -> Vec<&'a Value> {
    items
        .iter()
        .filter(|item| json::str_of(item, "type") == Some(kind))
        .collect()
}

/// The first answer of a turn: encrypted reasoning, then one Action call.
fn reasoning_and_call() -> AssistantMessage {
    let answer = openai_output(vec![
        parse(
            r#"{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAA-sealed","status":null}"#,
        ),
        parse(
            r#"{"type":"function_call","id":"fc_1","call_id":"call_1","name":"a_dns__read","arguments":"{\"zone\": \"example.com\"}","status":"completed"}"#,
        ),
    ]);
    response(json::compact(&answer).as_bytes()).unwrap().0
}

fn result(call_id: &str, content: &str) -> Message {
    Message::Tool(ToolMessage {
        id: format!("t-{call_id}"),
        call_id: call_id.into(),
        name: "a_dns__read".into(),
        content: content.into(),
        error: false,
    })
}

#[test]
fn every_call_is_unstored_and_asks_for_encrypted_reasoning() {
    for profile in [Profile::CHAT, Profile::DECISION, Profile::DEFAULT] {
        let wire = request(MODEL, &chat(vec![user("u1", "hi")]), None, profile);
        let sent = body(&wire);
        assert_eq!(wire.path, "/v1/responses");
        assert_eq!(sent.get("store"), Some(&Value::Bool(false)));
        assert_eq!(
            sent.get("include"),
            Some(&parse(r#"["reasoning.encrypted_content"]"#))
        );
        assert_eq!(sent.get("previous_response_id"), None);
        assert_eq!(
            input_of(&wire),
            [
                parse(r#"{"content":"You are the Brain.","role":"system","type":"message"}"#),
                parse(r#"{"content":"hi","role":"user","type":"message"}"#),
            ]
        );
    }
}

#[test]
fn chat_and_decisions_reason_low_and_helpers_keep_the_provider_default() {
    let effort = |profile| {
        body(&request(MODEL, &chat(vec![]), None, profile))
            .get("reasoning")
            .cloned()
    };
    let low = Some(parse(r#"{"effort":"low"}"#));
    assert_eq!(effort(Profile::CHAT), low);
    assert_eq!(effort(Profile::DECISION), low);
    assert_eq!(effort(Profile::DEFAULT), None);
    let capped = body(&request(
        MODEL,
        &chat(vec![]),
        None,
        Profile::DECISION.capped(300),
    ));
    assert_eq!(json::u64_of(&capped, "max_output_tokens"), Some(300));
    assert_eq!(
        body(&request(MODEL, &chat(vec![]), None, Profile::CHAT)).get("max_output_tokens"),
        None
    );
}

#[test]
fn a_resumed_turn_replays_encrypted_reasoning_without_server_state() {
    let first = reasoning_and_call();
    assert_eq!(first.calls[0].arguments, parse(r#"{"zone":"example.com"}"#));
    let reply = response(json::compact(&testing_reply("The zone is active.")).as_bytes())
        .unwrap()
        .0;
    let messages = vec![
        user("u1", "Is example.com active?"),
        Message::Assistant(first),
        result("call_1", r#"{"status": "active"}"#),
        Message::Assistant(reply),
        user("u2", "And again?"),
    ];
    let items = input_of(&request(MODEL, &chat(messages), None, Profile::CHAT));
    let reasoning = of_type(&items, "reasoning");
    assert_eq!(
        reasoning,
        [&parse(
            r#"{"id":"rs_1","summary":[],"type":"reasoning","encrypted_content":"gAAAA-sealed"}"#
        )]
    );
    let calls = of_type(&items, "function_call");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        json::str_of(calls[0], "arguments"),
        Some(r#"{"zone": "example.com"}"#)
    );
    let outputs = of_type(&items, "function_call_output");
    assert_eq!(
        outputs,
        [&parse(
            r#"{"type":"function_call_output","output":"{\"status\": \"active\"}","call_id":"call_1"}"#
        )]
    );
    assert_eq!(
        of_type(&items, "message")[2],
        &parse(
            r#"{"type":"message","content":[{"type":"output_text","text":"The zone is active.","annotations":[]}],"role":"assistant"}"#
        )
    );
    assert!(!json::compact(&json::array(items)).contains("msg_"));
}

fn testing_reply(text: &str) -> Value {
    crate::provider::testing::openai_body(text)
}

#[test]
fn reasoning_without_encrypted_content_is_never_replayed() {
    let answer = openai_output(vec![
        parse(r#"{"type":"reasoning","id":"rs_2","summary":[{"type":"summary_text","text":"s"}]}"#),
        testing_reply("Done.")
            .get("output")
            .and_then(|output| match output {
                Value::Array(items) => items.first().cloned(),
                _ => None,
            })
            .unwrap(),
    ]);
    let turn = response(json::compact(&answer).as_bytes()).unwrap().0;
    let items = input_of(&request(
        MODEL,
        &chat(vec![user("u", "x"), Message::Assistant(turn)]),
        None,
        Profile::CHAT,
    ));
    assert_eq!(of_type(&items, "reasoning").len(), 0);
    assert_eq!(of_type(&items, "message").len(), 3);
}

#[test]
fn a_rewritten_or_foreign_turn_replays_as_text_and_calls() {
    let call = ToolCall {
        id: "toolu_1".into(),
        name: "a_dns__read".into(),
        arguments: parse(r#"{"zone":"é.com"}"#),
    };
    let foreign = AssistantMessage {
        id: "a1".into(),
        text: String::new(),
        calls: vec![call],
        invalid_calls: vec![InvalidCall {
            id: "call_x".into(),
            name: "a_dns__read".into(),
            arguments: "{".into(),
        }],
        refusal: false,
        truncated: false,
        provider: "anthropic".into(),
        raw: Some(parse(
            r#"[{"type":"thinking","thinking":"t","signature":"s"}]"#,
        )),
        attempts: 1,
    };
    let rewritten = AssistantMessage {
        text: "Capped reply".into(),
        calls: Vec::new(),
        invalid_calls: Vec::new(),
        provider: "openai".into(),
        raw: None,
        ..foreign.clone()
    };
    let messages = vec![
        user("u", "x"),
        Message::Assistant(foreign),
        Message::Assistant(rewritten),
    ];
    let items = input_of(&request(MODEL, &chat(messages), None, Profile::CHAT));
    assert_eq!(
        items[2..],
        [
            parse(
                r#"{"type":"function_call","name":"a_dns__read","arguments":"{\"zone\": \"é.com\"}","call_id":"toolu_1"}"#
            ),
            parse(
                r#"{"type":"function_call","name":"a_dns__read","arguments":"{","call_id":"call_x"}"#
            ),
            parse(
                r#"{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Capped reply","annotations":[]}]}"#
            ),
        ]
    );
}

fn spec(name: &str, strict: Option<bool>) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: format!("{name} tool"),
        schema: parse(
            r#"{"type":"object","properties":{"zone":{"type":"string"},"limit":{"type":"integer"}},"required":["zone"],"additionalProperties":false}"#,
        ),
        strict,
    }
}

#[test]
fn action_tools_are_sent_non_strict_with_their_required_list_and_brain_tools_at_the_default() {
    let mut request_with_tools = chat(vec![user("u", "x")]);
    request_with_tools.tools = vec![
        spec("a_dns__list", Some(false)),
        spec("shimpz_clarify", None),
    ];
    let sent = body(&request(MODEL, &request_with_tools, None, Profile::CHAT));
    let tools = json::array_of(&sent, "tools").unwrap();
    assert_eq!(
        tools[0],
        parse(
            r#"{"type":"function","name":"a_dns__list","description":"a_dns__list tool","parameters":{"type":"object","properties":{"zone":{"type":"string"},"limit":{"type":"integer"}},"required":["zone"],"additionalProperties":false},"strict":false}"#
        )
    );
    assert_eq!(tools[1].get("strict"), None);
    assert_eq!(sent.get("tool_choice"), None);
    assert_eq!(
        body(&request(MODEL, &chat(vec![]), None, Profile::CHAT)).get("tools"),
        None
    );
}

#[test]
fn structured_output_asks_for_the_strict_named_schema() {
    let mut ask = chat(vec![user("u", "x")]);
    ask.output = Some(OutputSchema {
        name: "Plan".into(),
        schema: parse(
            r#"{"properties":{"status":{"enum":["a","b"],"type":"string"},"nested":{"type":"object","properties":{"x":{"type":"string","default":null}}},"tags":{"items":{"type":"object"},"type":"array"},"note":{"allOf":[{"type":"string"}],"description":"d"}},"required":["status"],"title":"Plan","type":"object"}"#,
        ),
    });
    let sent = body(&request(MODEL, &ask, None, Profile::DECISION));
    assert_eq!(
        sent.get("text"),
        Some(&parse(
            r#"{"format":{"type":"json_schema","strict":true,"name":"Plan","schema":{"properties":{"status":{"enum":["a","b"],"type":"string"},"nested":{"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":false,"required":["x"]},"tags":{"items":{"type":"object","additionalProperties":false},"type":"array"},"note":{"description":"d","type":"string"}},"required":["status","nested","tags","note"],"title":"Plan","type":"object","additionalProperties":false}}}"#
        ))
    );
    assert_eq!(sent.get("tools"), None);
}

#[test]
fn a_ref_beside_other_keywords_is_inlined() {
    let schema = parse(
        r##"{"type":"object","properties":{"a":{"$ref":"#/$defs/A","description":"x"}},"$defs":{"A":{"type":"object","properties":{"b":{"type":"integer"}}}}}"##,
    );
    let strict = schema::strict(&schema);
    assert_eq!(
        strict
            .get("properties")
            .and_then(|properties| properties.get("a")),
        Some(&parse(
            r#"{"description":"x","type":"object","properties":{"b":{"type":"integer"}},"additionalProperties":false,"required":["b"]}"#
        ))
    );
}

#[test]
fn attachments_reach_only_the_projected_message_as_native_text_and_image() {
    let projection = Projection {
        message_id: "turn".into(),
        blocks: vec![
            Block::Text("Attached file".into()),
            Block::Image {
                media_type: "image/png".into(),
                base64: "iVBORw0KGgo=".into(),
            },
        ],
    };
    let messages = vec![
        user("earlier", "Before"),
        user("turn", "Summarize the files"),
    ];
    let items = input_of(&request(
        MODEL,
        &chat(messages),
        Some(&projection),
        Profile::CHAT,
    ));
    assert_eq!(
        items[1],
        parse(r#"{"content":"Before","role":"user","type":"message"}"#)
    );
    assert_eq!(
        items[2],
        parse(
            r#"{"content":[{"type":"input_text","text":"Summarize the files"},{"type":"input_text","text":"Attached file"},{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}],"role":"user","type":"message"}"#
        )
    );
    let count = count_request(MODEL, &projection.blocks);
    assert_eq!(count.path, "/v1/responses/input_tokens");
    assert_eq!(
        body(&count),
        parse(
            r#"{"input":[{"content":[{"type":"input_text","text":"Attached file"},{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}],"role":"user","type":"message"}],"model":"gpt-6.1-sol"}"#
        )
    );
}

fn answer(text: &str) -> Option<(AssistantMessage, Option<Tokens>)> {
    response(text.as_bytes())
}

#[test]
fn an_answer_reads_its_text_calls_refusal_truncation_and_raw_items() {
    let (turn, tokens) = answer(
        r#"{"status":"incomplete","output":[{"type":"message","id":"m","content":[{"type":"output_text","text":"a"},{"type":"refusal","refusal":"no"},{"type":"output_text","text":"b"}]},{"type":"function_call","call_id":"c1","name":"t","arguments":"{\"x\":1}"},{"type":"function_call","call_id":"c2","name":"t","arguments":"[1]"},{"type":"function_call","call_id":"c3","name":"t","arguments":"{\"x\":1,\"x\":2}"}]}"#,
    )
    .unwrap();
    assert_eq!(
        (turn.text.as_str(), turn.refusal, turn.truncated),
        ("a\nb", true, true)
    );
    assert_eq!(
        (turn.id.as_str(), turn.provider.as_str(), turn.attempts),
        ("", "openai", 1)
    );
    assert_eq!(turn.calls.len(), 1);
    assert_eq!(turn.calls[0].arguments, parse(r#"{"x":1}"#));
    let invalid: Vec<_> = turn
        .invalid_calls
        .iter()
        .map(|call| call.id.as_str())
        .collect();
    assert_eq!(invalid, ["c2", "c3"]);
    assert!(matches!(turn.raw, Some(Value::Array(ref items)) if items.len() == 4));
    assert_eq!(tokens, None);
}

#[test]
fn only_a_closed_answer_is_read() {
    for refused in [
        "not json",
        "[]",
        r#"{"output":{}}"#,
        r#"{"error":{"message":"x"},"output":[]}"#,
        r#"{"output":[{"type":"message","content":[{"type":"output_text"}]}]}"#,
        r#"{"output":[{"type":"function_call","name":"t","arguments":"{}"}]}"#,
    ] {
        assert_eq!(answer(refused), None, "{refused}");
    }
    assert!(answer(r#"{"error":null,"output":[{"type":"web_search_call"}]}"#).is_some());
}

#[test]
fn reported_usage_counts_cached_input_and_a_missing_report_is_none() {
    let tokens = |usage: &str| answer(&format!(r#"{{"output":[],{usage}}}"#)).unwrap().1;
    assert_eq!(
        tokens(
            r#""usage":{"input_tokens":120,"output_tokens":30,"input_tokens_details":{"cached_tokens":80,"cache_write_tokens":20}}"#
        ),
        Some(Tokens {
            input: 120,
            output: 30,
            cache_read: 80,
            cache_write: 20
        })
    );
    assert_eq!(
        tokens(r#""usage":{"input_tokens":null,"output_tokens":3}"#),
        Some(Tokens {
            output: 3,
            ..Tokens::default()
        })
    );
    assert_eq!(
        tokens(
            r#""service_tier":"priority","usage":{"input_tokens":5,"output_tokens":1,"input_tokens_details":{"cached_tokens":4}}"#
        ),
        Some(Tokens {
            input: 5,
            output: 1,
            ..Tokens::default()
        })
    );
    assert_eq!(tokens(r#""usage":null"#), None);
}

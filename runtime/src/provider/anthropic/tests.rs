//! Tests of the Anthropic adapter: the request bodies Anthropic receives and the answers read back.

use super::*;
use crate::conversation::InvalidCall;
use crate::provider::OutputSchema;
use crate::provider::testing::{anthropic_body, anthropic_content};

const MODEL: &str = "claude-sonnet-5-5";

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

fn chat(messages: Vec<Message>, cache: bool) -> ChatRequest {
    ChatRequest {
        system: Some("Enabled Assistant contracts: clock.".into()),
        messages,
        tools: vec![spec("a_clock__read"), spec("shimpz_clarify")],
        output: None,
        cache,
    }
}

fn spec(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: format!("{name} tool"),
        schema: parse(r#"{"type":"object","properties":{},"additionalProperties":false}"#),
        strict: Some(false),
    }
}

fn turn_of(answer: &Value) -> AssistantMessage {
    response(json::compact(answer).as_bytes()).unwrap().0
}

fn messages_of(wire: &Wire) -> Vec<Value> {
    json::array_of(&body(wire), "messages").unwrap().to_vec()
}

#[test]
fn a_cached_turn_marks_the_system_prompt_the_tools_and_the_conversation_prefix() {
    let cache = parse(r#"{"type":"ephemeral","ttl":"5m"}"#);
    let earlier = turn_of(&anthropic_body("Olá!"));
    let messages = vec![
        user("u1", "oi"),
        Message::Assistant(earlier),
        user("u2", "tudo bem?"),
    ];
    let wire = request(MODEL, &chat(messages, true), None, Profile::CHAT);
    let sent = body(&wire);
    assert_eq!(wire.path, "/v1/messages");
    assert_eq!(sent.get("cache_control"), Some(&cache));
    let system = json::array_of(&sent, "system").unwrap();
    assert_eq!(system[0].get("cache_control"), Some(&cache));
    assert_eq!(
        json::str_of(&system[0], "text"),
        Some("Enabled Assistant contracts: clock.")
    );
    let tools = json::array_of(&sent, "tools").unwrap();
    assert_eq!(tools[0].get("cache_control"), None);
    assert_eq!(tools[1].get("cache_control"), Some(&cache));
    assert_eq!(
        sent.get("messages"),
        Some(&parse(
            r#"[{"role":"user","content":"oi"},{"role":"assistant","content":"Olá!"},{"role":"user","content":"tudo bem?"}]"#
        ))
    );
}

#[test]
fn an_uncached_call_carries_no_cache_marker() {
    let plain = body(&request(
        MODEL,
        &chat(vec![user("u", "oi")], false),
        None,
        Profile::CHAT,
    ));
    assert_eq!(plain.get("cache_control"), None);
    assert_eq!(
        json::str_of(&plain, "system"),
        Some("Enabled Assistant contracts: clock.")
    );
    assert!(!json::compact(&plain).contains("cache_control"));
}

#[test]
fn tools_are_sent_unchanged_without_a_strict_flag() {
    let sent = body(&request(
        MODEL,
        &chat(vec![user("u", "x")], false),
        None,
        Profile::CHAT,
    ));
    assert_eq!(
        json::array_of(&sent, "tools").unwrap()[0],
        parse(
            r#"{"name":"a_clock__read","input_schema":{"type":"object","properties":{},"additionalProperties":false},"description":"a_clock__read tool"}"#
        )
    );
    assert_eq!(sent.get("tool_choice"), None);
    let mut bare = chat(vec![user("u", "x")], true);
    bare.tools.clear();
    assert_eq!(
        body(&request(MODEL, &bare, None, Profile::CHAT)).get("tools"),
        None
    );
}

#[test]
fn effort_cap_and_structured_output_follow_the_profile_and_the_ask() {
    let config = |profile| body(&request(MODEL, &chat(vec![], false), None, profile));
    assert_eq!(
        config(Profile::CHAT).get("output_config"),
        Some(&parse(r#"{"effort":"low"}"#))
    );
    assert_eq!(
        config(Profile::DECISION).get("output_config"),
        Some(&parse(r#"{"effort":"low"}"#))
    );
    assert_eq!(config(Profile::DEFAULT).get("output_config"), None);
    assert_eq!(
        json::u64_of(&config(Profile::CHAT), "max_tokens"),
        Some(4096)
    );
    assert_eq!(
        json::u64_of(&config(Profile::CHAT.capped(2048)), "max_tokens"),
        Some(2048)
    );
    let mut ask = chat(vec![user("u", "x")], false);
    ask.tools.clear();
    ask.output = Some(OutputSchema {
        name: "Plan".into(),
        schema: parse(
            r#"{"properties":{"ids":{"items":{"type":"string"},"maxItems":4,"type":"array"}},"required":["ids"],"title":"Plan","type":"object"}"#,
        ),
    });
    let sent = body(&request(MODEL, &ask, None, Profile::DECISION));
    assert_eq!(
        sent.get("output_config"),
        Some(&parse(
            r#"{"format":{"type":"json_schema","schema":{"type":"object","title":"Plan","properties":{"ids":{"type":"array","items":{"type":"string"},"description":"{maxItems: 4}"}},"additionalProperties":false,"required":["ids"]}},"effort":"low"}"#
        ))
    );
    assert_eq!(sent.get("tools"), None);
}

fn tool_result(call_id: &str, error: bool) -> Message {
    Message::Tool(ToolMessage {
        id: format!("t-{call_id}"),
        call_id: call_id.into(),
        name: "a_clock__read".into(),
        content: format!("result {call_id}"),
        error,
    })
}

#[test]
fn a_tool_turn_replays_its_signed_thinking_and_merges_its_results() {
    let calls = anthropic_content(
        vec![
            parse(r#"{"type":"thinking","thinking":"hmm","signature":"sig1"}"#),
            parse(r#"{"type":"text","text":"  ","citations":null}"#),
            parse(r#"{"type":"text","text":"Let me check."}"#),
            parse(
                r#"{"type":"tool_use","id":"toolu_1","name":"a_clock__read","input":{"zone":"UTC"},"caller":null}"#,
            ),
            parse(r#"{"type":"tool_use","id":"toolu_2","name":"a_clock__read","input":{}}"#),
        ],
        "tool_use",
    );
    let turn = turn_of(&calls);
    assert_eq!(turn.calls.len(), 2);
    let messages = vec![
        user("u1", "What time is it?"),
        Message::Assistant(turn),
        tool_result("toolu_1", false),
        tool_result("toolu_2", true),
    ];
    let sent = messages_of(&request(MODEL, &chat(messages, true), None, Profile::CHAT));
    assert_eq!(
        sent[1],
        parse(
            r#"{"role":"assistant","content":[{"signature":"sig1","thinking":"hmm","type":"thinking"},{"text":"Let me check.","type":"text"},{"type":"tool_use","name":"a_clock__read","input":{"zone":"UTC"},"id":"toolu_1"},{"type":"tool_use","name":"a_clock__read","input":{},"id":"toolu_2"}]}"#
        )
    );
    assert_eq!(
        sent[2],
        parse(
            r#"{"role":"user","content":[{"type":"tool_result","content":"result toolu_1","tool_use_id":"toolu_1","is_error":false},{"type":"tool_result","content":"result toolu_2","tool_use_id":"toolu_2","is_error":true}]}"#
        )
    );
    assert_eq!(sent.len(), 3);
}

/// A turn OpenAI wrote with a call id Anthropic refuses, a result, an empty turn, and a capped reply.
fn foreign_conversation() -> Vec<Message> {
    let foreign = AssistantMessage {
        id: "a1".into(),
        text: String::new(),
        calls: vec![ToolCall {
            id: "functions.read:0".into(),
            name: "a_clock__read".into(),
            arguments: parse("{}"),
        }],
        invalid_calls: vec![InvalidCall {
            id: "call_x".into(),
            name: "a_clock__read".into(),
            arguments: "{".into(),
        }],
        refusal: false,
        truncated: false,
        provider: "openai".into(),
        raw: Some(parse(r#"[{"type":"reasoning","id":"rs_1","summary":[]}]"#)),
        attempts: 1,
    };
    let empty = AssistantMessage {
        calls: Vec::new(),
        invalid_calls: Vec::new(),
        raw: None,
        ..foreign.clone()
    };
    let capped = AssistantMessage {
        text: "Capped reply  ".into(),
        ..empty.clone()
    };
    vec![
        user("u1", "x"),
        Message::Assistant(foreign),
        tool_result("functions.read:0", false),
        user("u2", "again"),
        Message::Assistant(empty),
        user("u3", "and"),
        Message::Assistant(capped),
    ]
}

#[test]
fn a_rewritten_or_foreign_turn_replays_as_text_and_calls() {
    let messages = foreign_conversation();
    let sent = messages_of(&request(MODEL, &chat(messages, false), None, Profile::CHAT));
    let id = "toolu_c216cab2de2e56d547b4e634";
    assert_eq!(
        sent[1],
        parse(&format!(
            r#"{{"role":"assistant","content":[{{"type":"tool_use","name":"a_clock__read","input":{{}},"id":"{id}"}}]}}"#
        ))
    );
    assert_eq!(
        sent[2],
        parse(&format!(
            r#"{{"role":"user","content":[{{"type":"tool_result","content":"result functions.read:0","tool_use_id":"{id}","is_error":false}},{{"type":"text","text":"again"}}]}}"#
        ))
    );
    assert_eq!(sent[3], parse(r#"{"role":"user","content":"and"}"#));
    assert_eq!(
        sent[4],
        parse(r#"{"role":"assistant","content":"Capped reply"}"#)
    );
}

#[test]
fn attachments_reach_only_the_projected_message_as_native_text_and_image() {
    let projection = Projection {
        message_id: "turn".into(),
        blocks: vec![
            Block::Text("Attached file".into()),
            Block::Text(" ".into()),
            Block::Image {
                media_type: "image/png".into(),
                base64: "iVBORw0KGgo=".into(),
            },
        ],
    };
    let image = parse(
        r#"{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}"#,
    );
    let messages = vec![
        user("bridge", "Earlier"),
        user("turn", "What is in the picture?"),
    ];
    let sent = messages_of(&request(
        MODEL,
        &chat(messages, false),
        Some(&projection),
        Profile::CHAT,
    ));
    assert_eq!(
        sent,
        [parse(&format!(
            r#"{{"role":"user","content":[{{"type":"text","text":"Earlier"}},{{"type":"text","text":"What is in the picture?"}},{{"type":"text","text":"Attached file"}},{}]}}"#,
            json::compact(&image)
        ))]
    );
    let count = count_request(MODEL, &projection.blocks);
    assert_eq!(count.path, "/v1/messages/count_tokens");
    assert_eq!(
        body(&count),
        parse(&format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"Attached file"}},{}]}}],"model":"claude-sonnet-5-5"}}"#,
            json::compact(&image)
        ))
    );
}

#[test]
fn an_answer_reads_its_text_calls_refusal_and_truncation() {
    let answer = anthropic_content(
        vec![
            parse(r#"{"type":"thinking","thinking":"t","signature":"s"}"#),
            parse(r#"{"type":"text","text":"a"}"#),
            parse(r#"{"type":"text","text":"b"}"#),
            parse(r#"{"type":"tool_use","id":"toolu_1","name":"x","input":{"k":1}}"#),
        ],
        "max_tokens",
    );
    let (turn, tokens) = response(json::compact(&answer).as_bytes()).unwrap();
    assert_eq!(
        (turn.text.as_str(), turn.truncated, turn.refusal),
        ("a\nb", true, false)
    );
    assert_eq!(
        (turn.calls[0].id.as_str(), &turn.calls[0].arguments),
        ("toolu_1", &parse(r#"{"k":1}"#))
    );
    assert_eq!(
        (turn.provider.as_str(), turn.attempts, turn.id.as_str()),
        ("anthropic", 1, "")
    );
    assert!(matches!(turn.raw, Some(Value::Array(ref blocks)) if blocks.len() == 4));
    assert_eq!(tokens.map(|tokens| tokens.input), Some(10));
}

#[test]
fn a_refusal_is_read_and_a_missing_usage_is_unreported() {
    let mut answer = anthropic_body("x");
    if let Value::Object(members) = &mut answer {
        wire::set(members, "stop_reason", json::text("refusal"));
        wire::set(members, "content", json::array(Vec::new()));
        wire::take(members, "usage");
    }
    let (refused, unreported) = response(json::compact(&answer).as_bytes()).unwrap();
    assert!(refused.refusal && !refused.truncated);
    assert_eq!(unreported, None);
}

#[test]
fn only_a_closed_answer_is_read() {
    for invalid in [
        "{}",
        r#"{"content":{}}"#,
        r#"{"content":[{"type":"tool_use","id":"t","name":"x","input":[1]}]}"#,
        r#"{"content":[{"text":"no type"}]}"#,
    ] {
        assert_eq!(response(invalid.as_bytes()), None, "{invalid}");
    }
}

#[test]
fn reported_usage_includes_cache_reads_and_writes_split_by_lifetime() {
    let tokens = |usage: &str| {
        response(format!(r#"{{"content":[],"usage":{usage}}}"#).as_bytes())
            .unwrap()
            .1
            .unwrap()
    };
    assert_eq!(
        tokens(
            r#"{"input_tokens":20,"output_tokens":30,"cache_read_input_tokens":80,"cache_creation_input_tokens":20}"#
        ),
        Tokens {
            input: 120,
            output: 30,
            cache_read: 80,
            cache_write: 20
        }
    );
    assert_eq!(
        tokens(
            r#"{"input_tokens":3,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":6003,"cache_creation":{"ephemeral_5m_input_tokens":6003,"ephemeral_1h_input_tokens":0}}"#
        ),
        Tokens {
            input: 6006,
            output: 1,
            cache_read: 0,
            cache_write: 6003
        }
    );
}

//! The OpenAI Responses API adapter: the request bodies and answer readings the Python Brain's
//! `ChatOpenAI(use_responses_api=True, store=False, include=["reasoning.encrypted_content"])` produced.
//!
//! No conversation is stored on OpenAI's side: every call is unstored and carries the whole conversation, replaying
//! an earlier OpenAI turn's reasoning as its encrypted content and its message ids dropped.

mod schema;

use crate::conversation::{AssistantMessage, Block, InvalidCall, Message, ToolCall, ToolMessage};
use crate::json::{self, Value};
use crate::provider::wire::{self, object};
use crate::provider::{ChatRequest, Profile, Projection, Tokens, ToolSpec, Wire};

/// The Responses endpoint.
pub const RESPONSES: &str = "/v1/responses";
/// The input-token counting endpoint.
pub const INPUT_TOKENS: &str = "/v1/responses/input_tokens";
const PROVIDER: &str = "openai";
// The fields of the SDK's reasoning, function-call, and summary models, in their declared order.
const REASONING_FIELDS: [&str; 6] = [
    "id",
    "summary",
    "type",
    "content",
    "encrypted_content",
    "status",
];
const FUNCTION_CALL_FIELDS: [&str; 8] = [
    "arguments",
    "call_id",
    "name",
    "type",
    "id",
    "caller",
    "namespace",
    "status",
];
const PART_FIELDS: [&str; 2] = ["text", "type"];

/// The `/v1/responses` request for one call.
#[must_use]
pub fn request(
    model: &str,
    request: &ChatRequest,
    projection: Option<&Projection>,
    profile: Profile,
) -> Wire {
    // The SDK writes its parameters in alphabetical order.
    let mut body = vec![
        (
            "include",
            json::array(vec![json::text("reasoning.encrypted_content")]),
        ),
        ("input", json::array(input(request, projection))),
    ];
    if let Some(cap) = profile.max_output_tokens {
        body.push(("max_output_tokens", json::uint(u64::from(cap))));
    }
    body.push(("model", json::text(model)));
    if let Some(effort) = profile.effort {
        body.push(("reasoning", object(vec![("effort", json::text(effort))])));
    }
    body.push(("store", Value::Bool(false)));
    body.push(("stream", Value::Bool(false)));
    if let Some(output) = &request.output {
        let format = object(vec![
            ("type", json::text("json_schema")),
            ("strict", Value::Bool(true)),
            ("name", json::text(&output.name)),
            ("schema", schema::strict(&output.schema)),
        ]);
        body.push(("text", object(vec![("format", format)])));
    }
    if !request.tools.is_empty() {
        body.push((
            "tools",
            json::array(request.tools.iter().map(tool).collect()),
        ));
    }
    Wire {
        path: RESPONSES,
        headers: Vec::new(),
        body: json::compact(&object(body)).into_bytes(),
    }
}

/// The `/v1/responses/input_tokens` request counting one user message made of `blocks`.
#[must_use]
pub fn count_request(model: &str, blocks: &[Block]) -> Wire {
    let content: Vec<Value> = blocks.iter().map(block).collect();
    let input = if content.is_empty() {
        Vec::new()
    } else {
        vec![message("user", json::array(content))]
    };
    let body = object(vec![
        ("input", json::array(input)),
        ("model", json::text(model)),
    ]);
    Wire {
        path: INPUT_TOKENS,
        headers: Vec::new(),
        body: json::compact(&body).into_bytes(),
    }
}

/// One function tool; an Action tool carries `strict: false` with its declared `required` list (ADR-0094).
fn tool(spec: &ToolSpec) -> Value {
    let mut members = vec![
        ("type", json::text("function")),
        ("name", json::text(&spec.name)),
        ("description", json::text(&spec.description)),
        ("parameters", spec.schema.clone()),
    ];
    if let Some(strict) = spec.strict {
        members.push(("strict", Value::Bool(strict)));
    }
    object(members)
}

fn message(role: &str, content: Value) -> Value {
    object(vec![
        ("content", content),
        ("role", json::text(role)),
        ("type", json::text("message")),
    ])
}

fn block(block: &Block) -> Value {
    match block {
        Block::Text(text) => object(vec![
            ("type", json::text("input_text")),
            ("text", json::text(text)),
        ]),
        Block::Image { media_type, base64 } => object(vec![
            ("type", json::text("input_image")),
            (
                "image_url",
                json::text(&format!("data:{media_type};base64,{base64}")),
            ),
        ]),
    }
}

fn input(request: &ChatRequest, projection: Option<&Projection>) -> Vec<Value> {
    let mut items = Vec::new();
    if let Some(system) = &request.system {
        items.push(message("system", json::text(system)));
    }
    for entry in &request.messages {
        match entry {
            Message::User { id, text } => items.push(user(id, text, projection)),
            Message::Assistant(turn) => assistant(turn, &mut items),
            Message::Tool(result) => items.push(tool_result(result)),
        }
    }
    items
}

/// The person's message, with the request-local blocks after its text when it is the projection's message.
fn user(id: &str, text: &str, projection: Option<&Projection>) -> Value {
    match projection.filter(|projection| projection.message_id == id) {
        Some(projection) => {
            let mut content = vec![block(&Block::Text(text.to_owned()))];
            content.extend(projection.blocks.iter().map(block));
            message("user", json::array(content))
        }
        None => message("user", json::text(text)),
    }
}

fn tool_result(result: &ToolMessage) -> Value {
    object(vec![
        ("type", json::text("function_call_output")),
        ("output", json::text(&result.content)),
        ("call_id", json::text(&result.call_id)),
    ])
}

/// A model turn: its own output items when OpenAI wrote it, else its text; then every call not yet in the input.
fn assistant(turn: &AssistantMessage, items: &mut Vec<Value>) {
    match turn.raw.as_ref() {
        Some(Value::Array(output)) if turn.provider == PROVIDER => {
            for item in output {
                replay(item, items);
            }
        }
        _ => {
            if !turn.text.is_empty() || (turn.calls.is_empty() && turn.invalid_calls.is_empty()) {
                items.push(object(vec![
                    ("type", json::text("message")),
                    ("role", json::text("assistant")),
                    ("content", json::array(vec![output_text(&turn.text, None)])),
                ]));
            }
        }
    }
    let calls = turn
        .calls
        .iter()
        .map(|call| {
            (
                &call.id,
                &call.name,
                json::dumps(&call.arguments, json::SPACED),
            )
        })
        .chain(
            turn.invalid_calls
                .iter()
                .map(|call| (&call.id, &call.name, call.arguments.clone())),
        );
    for (id, name, arguments) in calls {
        let present = items.iter().any(|item| {
            json::str_of(item, "type") == Some("function_call")
                && json::str_of(item, "call_id") == Some(id.as_str())
        });
        if !present {
            items.push(object(vec![
                ("type", json::text("function_call")),
                ("name", json::text(name)),
                ("arguments", json::text(&arguments)),
                ("call_id", json::text(id)),
            ]));
        }
    }
}

fn output_text(text: &str, annotations: Option<&Value>) -> Value {
    object(vec![
        ("type", json::text("output_text")),
        ("text", json::text(text)),
        (
            "annotations",
            annotations
                .filter(|annotations| !matches!(annotations, Value::Null))
                .cloned()
                .unwrap_or_else(|| json::array(Vec::new())),
        ),
    ])
}

/// One output item of an earlier OpenAI answer as unstored input: each text or refusal as its own message without
/// its id, reasoning only with its encrypted content, function calls as they came; nothing else.
fn replay(item: &Value, items: &mut Vec<Value>) {
    match json::str_of(item, "type") {
        Some("message") => replay_message(item, items),
        Some("reasoning") if wire::present(item, "encrypted_content").is_some() => {
            let mut reasoning = wire::ordered(&wire::without_nulls(item), &REASONING_FIELDS);
            if let Value::Object(members) = &mut reasoning {
                for (name, value) in members.iter_mut() {
                    if let ("summary" | "content", Value::Array(parts)) = (name.as_str(), value) {
                        for part in parts.iter_mut() {
                            *part = wire::ordered(part, &PART_FIELDS);
                        }
                    }
                }
            }
            items.push(reasoning);
        }
        Some("function_call") => {
            items.push(wire::ordered(
                &wire::without_nulls(item),
                &FUNCTION_CALL_FIELDS,
            ));
        }
        _ => {}
    }
}

fn replay_message(item: &Value, items: &mut Vec<Value>) {
    let phase = wire::present(item, "phase");
    for part in json::array_of(item, "content").unwrap_or_default() {
        let content = match json::str_of(part, "type") {
            Some("output_text") => match json::str_of(part, "text") {
                Some(text) => output_text(text, part.get("annotations")),
                None => continue,
            },
            Some("refusal") => object(vec![
                ("type", json::text("refusal")),
                (
                    "refusal",
                    part.get("refusal").cloned().unwrap_or(Value::Null),
                ),
            ]),
            _ => continue,
        };
        let mut members = vec![
            ("type", json::text("message")),
            ("content", json::array(vec![content])),
            ("role", json::text("assistant")),
        ];
        if let Some(phase) = phase {
            members.push(("phase", phase.clone()));
        }
        items.push(object(members));
    }
}

/// The model turn and reported tokens of a `/v1/responses` answer; `None` when it is not one.
#[must_use]
pub fn response(body: &[u8]) -> Option<(AssistantMessage, Option<Tokens>)> {
    let value = json::parse(body).ok()?;
    json::members(&value)?;
    if wire::present(&value, "error").is_some() {
        return None;
    }
    let output = json::array_of(&value, "output")?;
    let mut turn = AssistantMessage {
        id: String::new(),
        text: String::new(),
        calls: Vec::new(),
        invalid_calls: Vec::new(),
        refusal: false,
        truncated: json::str_of(&value, "status") == Some("incomplete"),
        provider: PROVIDER.to_owned(),
        raw: Some(json::array(output.to_vec())),
        attempts: 1,
    };
    let mut texts = Vec::new();
    for item in output {
        read_item(item, &mut turn, &mut texts)?;
    }
    turn.text = texts.join("\n");
    Some((turn, tokens(&value)))
}

fn read_item(item: &Value, turn: &mut AssistantMessage, texts: &mut Vec<String>) -> Option<()> {
    match json::str_of(item, "type")? {
        "message" => {
            for part in json::array_of(item, "content")? {
                match json::str_of(part, "type")? {
                    "output_text" => texts.push(json::str_of(part, "text")?.to_owned()),
                    "refusal" => turn.refusal = true,
                    _ => {}
                }
            }
        }
        "function_call" => {
            let id = json::str_of(item, "call_id")?.to_owned();
            let name = json::str_of(item, "name")?.to_owned();
            let arguments = json::str_of(item, "arguments")?;
            match wire::object_text(arguments) {
                Some(arguments) => turn.calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                }),
                None => turn.invalid_calls.push(InvalidCall {
                    id,
                    name,
                    arguments: arguments.to_owned(),
                }),
            }
        }
        _ => {}
    }
    Some(())
}

/// The reported usage as `model_usage.py` counted the Python adapter's reading of it: input tokens include cached ones; a
/// priority or flex answer's cache counts were filed under other names and counted as none.
fn tokens(value: &Value) -> Option<Tokens> {
    let usage = wire::present(value, "usage")?;
    let details = wire::present(usage, "input_tokens_details");
    let tiered = matches!(
        json::str_of(value, "service_tier"),
        Some("priority" | "flex")
    );
    let cache = |key| if tiered { 0 } else { wire::count(details, key) };
    Some(Tokens {
        input: wire::count(Some(usage), "input_tokens"),
        output: wire::count(Some(usage), "output_tokens"),
        cache_read: cache("cached_tokens"),
        cache_write: cache("cache_write_tokens"),
    })
}

#[cfg(test)]
mod tests;

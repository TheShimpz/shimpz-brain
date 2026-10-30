//! The Anthropic Messages API adapter: the request bodies and answer readings the Python Brain's `ChatAnthropic`
//! produced, with the prompt-cache markers its caching middleware placed on chat turns (five-minute TTL).

mod schema;

use std::fmt::Write as _;

use sha2::{Digest, Sha256};
use team_protocol::python::text as python_text;

use crate::conversation::{AssistantMessage, Block, Message, ToolCall, ToolMessage};
use crate::json::{self, Value};
use crate::provider::wire::{self, object};
use crate::provider::{ChatRequest, Profile, Projection, Tokens, ToolSpec, Wire};

/// The Messages endpoint.
pub const MESSAGES: &str = "/v1/messages";
/// The token counting endpoint.
pub const COUNT_TOKENS: &str = "/v1/messages/count_tokens";
/// The output cap sent without a profile cap: the adapter's fallback, which every catalog Anthropic model got
/// because none has a packaged model profile.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;
const PROVIDER: &str = "anthropic";
const THINKING_FIELDS: [&str; 3] = ["signature", "thinking", "type"];
const REDACTED_FIELDS: [&str; 2] = ["data", "type"];

fn cache_control() -> Value {
    object(vec![
        ("type", json::text("ephemeral")),
        ("ttl", json::text("5m")),
    ])
}

/// The `/v1/messages` request for one call.
#[must_use]
pub fn request(
    model: &str,
    request: &ChatRequest,
    projection: Option<&Projection>,
    profile: Profile,
) -> Wire {
    let cap = profile.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let mut body = vec![
        ("max_tokens", json::uint(u64::from(cap))),
        (
            "messages",
            json::array(messages(&request.messages, projection)),
        ),
        ("model", json::text(model)),
    ];
    if request.cache {
        body.push(("cache_control", cache_control()));
    }
    let mut output_config = Vec::new();
    if let Some(output) = &request.output {
        output_config.push((
            "format",
            object(vec![
                ("type", json::text("json_schema")),
                ("schema", schema::transform(&output.schema)),
            ]),
        ));
    }
    if let Some(effort) = profile.effort {
        output_config.push(("effort", json::text(effort)));
    }
    if !output_config.is_empty() {
        body.push(("output_config", object(output_config)));
    }
    if let Some(system) = &request.system {
        body.push(("system", system_prompt(system, request.cache)));
    }
    if !request.tools.is_empty() {
        body.push(("tools", tools(&request.tools, request.cache)));
    }
    Wire {
        path: MESSAGES,
        headers: Vec::new(),
        body: json::compact(&object(body)).into_bytes(),
    }
}

/// The `/v1/messages/count_tokens` request counting one user message made of `blocks`.
#[must_use]
pub fn count_request(model: &str, blocks: &[Block]) -> Wire {
    let content = formatted_user(blocks.iter().map(block).collect());
    let message = object(vec![
        ("role", json::text("user")),
        ("content", json::array(content)),
    ]);
    let body = object(vec![
        ("messages", json::array(vec![message])),
        ("model", json::text(model)),
    ]);
    Wire {
        path: COUNT_TOKENS,
        headers: Vec::new(),
        body: json::compact(&body).into_bytes(),
    }
}

/// The system prompt: plain, or one text block carrying the cache marker.
fn system_prompt(system: &str, cache: bool) -> Value {
    if cache && !system.is_empty() {
        json::array(vec![object(vec![
            ("type", json::text("text")),
            ("text", json::text(system)),
            ("cache_control", cache_control()),
        ])])
    } else {
        json::text(system)
    }
}

/// The tools, the last one carrying the cache marker that caches the whole set.
fn tools(specs: &[ToolSpec], cache: bool) -> Value {
    let last = specs.len() - 1;
    let tools = specs.iter().enumerate().map(|(index, spec)| {
        let mut members = vec![
            ("name", json::text(&spec.name)),
            ("input_schema", spec.schema.clone()),
            ("description", json::text(&spec.description)),
        ];
        if cache && index == last {
            members.push(("cache_control", cache_control()));
        }
        object(members)
    });
    json::array(tools.collect())
}

fn text_block(text: &str) -> Value {
    object(vec![
        ("type", json::text("text")),
        ("text", json::text(text)),
    ])
}

fn block(block: &Block) -> Value {
    match block {
        Block::Text(text) => text_block(text),
        Block::Image { media_type, base64 } => object(vec![
            ("type", json::text("image")),
            (
                "source",
                object(vec![
                    ("type", json::text("base64")),
                    ("media_type", json::text(media_type)),
                    ("data", json::text(base64)),
                ]),
            ),
        ]),
    }
}

/// An Anthropic-safe tool-use id: a provider-minted id outside `[A-Za-z0-9_-]+` is hashed deterministically.
fn tool_use_id(id: &str) -> String {
    let valid = id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
    if id.is_empty() || valid {
        return id.to_owned();
    }
    let digest = Sha256::digest(id.as_bytes());
    let hex = digest.iter().take(12).fold(String::new(), |mut hex, byte| {
        let _ = write!(hex, "{byte:02x}");
        hex
    });
    format!("toolu_{hex}")
}

/// A message's content before formatting: text, or blocks.
enum Content {
    Text(String),
    Blocks(Vec<Value>),
}

enum Entry {
    User(Content),
    Assistant(Content),
}

/// The conversation as Anthropic messages: runs of user messages and tool results merged into one user message,
/// blank text blocks dropped, and an empty model turn dropped unless it is the last message.
fn messages(conversation: &[Message], projection: Option<&Projection>) -> Vec<Value> {
    let mut entries: Vec<Entry> = Vec::new();
    for message in conversation {
        let content = match message {
            Message::User { id, text } => user(id, text, projection),
            Message::Tool(result) => Content::Blocks(vec![tool_result(result)]),
            Message::Assistant(turn) => {
                entries.push(Entry::Assistant(assistant(turn)));
                continue;
            }
        };
        match (entries.last_mut(), content) {
            (Some(Entry::User(previous)), content) => merge(previous, content),
            (_, content) => entries.push(Entry::User(content)),
        }
    }
    let last = entries.len().saturating_sub(1);
    entries
        .into_iter()
        .enumerate()
        .filter_map(|(index, entry)| format_entry(entry, index == last))
        .collect()
}

fn user(id: &str, text: &str, projection: Option<&Projection>) -> Content {
    match projection.filter(|projection| projection.message_id == id) {
        Some(projection) => {
            let mut blocks = vec![text_block(text)];
            blocks.extend(projection.blocks.iter().map(block));
            Content::Blocks(blocks)
        }
        None => Content::Text(text.to_owned()),
    }
}

fn tool_result(result: &ToolMessage) -> Value {
    object(vec![
        ("type", json::text("tool_result")),
        ("content", json::text(&result.content)),
        ("tool_use_id", json::text(&tool_use_id(&result.call_id))),
        ("is_error", Value::Bool(result.error)),
    ])
}

fn merge(previous: &mut Content, next: Content) {
    let mut blocks = match std::mem::replace(previous, Content::Blocks(Vec::new())) {
        Content::Text(text) => vec![text_block(&text)],
        Content::Blocks(blocks) => blocks,
    };
    match next {
        Content::Text(text) => blocks.push(text_block(&text)),
        Content::Blocks(more) => blocks.extend(more),
    }
    *previous = Content::Blocks(blocks);
}

fn blank_text(block: &Value) -> bool {
    json::str_of(block, "type") == Some("text")
        && python_text::strip(json::str_of(block, "text").unwrap_or_default()).is_empty()
}

fn formatted_user(blocks: Vec<Value>) -> Vec<Value> {
    blocks
        .into_iter()
        .filter(|block| !blank_text(block))
        .collect()
}

fn format_entry(entry: Entry, last: bool) -> Option<Value> {
    let (role, content) = match entry {
        Entry::User(Content::Text(text)) => ("user", json::text(&text)),
        Entry::User(Content::Blocks(blocks)) => ("user", json::array(formatted_user(blocks))),
        Entry::Assistant(content) => ("assistant", assistant_content(content, last)?),
    };
    Some(object(vec![
        ("role", json::text(role)),
        ("content", content),
    ]))
}

/// A model turn's content: its trailing text stripped when it ends the conversation; `None` when it is empty
/// before the end.
fn assistant_content(content: Content, last: bool) -> Option<Value> {
    match content {
        Content::Text(text) if last => Some(json::text(python_text::rstrip(&text))),
        Content::Text(text) => (!text.is_empty()).then(|| json::text(&text)),
        Content::Blocks(mut blocks) if last => {
            if let Some(Value::Object(members)) = blocks.last_mut() {
                strip_trailing_text(members);
            }
            Some(json::array(blocks))
        }
        Content::Blocks(blocks) => (!blocks.is_empty()).then(|| json::array(blocks)),
    }
}

fn strip_trailing_text(members: &mut [(String, Value)]) {
    let text_block = members.iter().any(|(name, value)| {
        name == "type" && matches!(value, Value::String(kind) if kind == "text")
    });
    if !text_block {
        return;
    }
    if let Some((_, Value::String(text))) = members.iter_mut().find(|(name, _)| name == "text") {
        *text = python_text::rstrip(text).to_owned();
    }
}

/// A model turn: its own content blocks when Anthropic wrote it, else its text and calls.
fn assistant(turn: &AssistantMessage) -> Content {
    match turn.raw.as_ref() {
        Some(Value::Array(blocks)) if turn.provider == PROVIDER => replay(blocks),
        _ if turn.calls.is_empty() => Content::Text(turn.text.clone()),
        _ => {
            let mut blocks = Vec::new();
            if !turn.text.is_empty() {
                blocks.push(text_block(&turn.text));
            }
            blocks.extend(turn.calls.iter().map(tool_use));
            Content::Blocks(blocks)
        }
    }
}

fn tool_use(call: &ToolCall) -> Value {
    object(vec![
        ("type", json::text("tool_use")),
        ("name", json::text(&call.name)),
        ("input", call.arguments.clone()),
        ("id", json::text(&tool_use_id(&call.id))),
    ])
}

/// An earlier Anthropic answer's content as the adapter replayed it: one plain text block as text, otherwise its
/// blocks with thinking signatures kept, blank text dropped, and tool uses as calls.
fn replay(blocks: &[Value]) -> Content {
    if let [only] = blocks
        && json::str_of(only, "type") == Some("text")
        && !truthy(only.get("citations"))
    {
        return Content::Text(json::str_of(only, "text").unwrap_or_default().to_owned());
    }
    Content::Blocks(blocks.iter().filter_map(replay_block).collect())
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null | Value::Bool(false)) => false,
        Some(Value::Array(items)) => !items.is_empty(),
        Some(_) => true,
    }
}

/// A citation without a null `file_id`, as the adapter cleaned it.
fn citation(citation: &Value) -> Value {
    match citation {
        Value::Object(members) => Value::Object(
            members
                .iter()
                .filter(|(name, value)| !(name == "file_id" && matches!(value, Value::Null)))
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    }
}

fn replay_block(block: &Value) -> Option<Value> {
    let kept = |fields: &[&str]| {
        let present: Vec<(&str, Value)> = fields
            .iter()
            .filter_map(|field| Some((*field, wire::present(block, field)?.clone())))
            .collect();
        object(present)
    };
    match json::str_of(block, "type") {
        Some("text") if blank_text(block) => None,
        Some("text") => {
            let mut members = Vec::new();
            if let Some(Value::Array(citations)) = wire::present(block, "citations") {
                members.push((
                    "citations",
                    json::array(citations.iter().map(citation).collect()),
                ));
            }
            members.push(("text", block.get("text").cloned().unwrap_or(Value::Null)));
            members.push(("type", json::text("text")));
            Some(object(members))
        }
        Some("thinking") => Some(kept(&THINKING_FIELDS)),
        Some("redacted_thinking") => Some(kept(&REDACTED_FIELDS)),
        Some("tool_use") => Some(object(vec![
            ("type", json::text("tool_use")),
            ("name", block.get("name").cloned().unwrap_or(Value::Null)),
            ("input", block.get("input").cloned().unwrap_or(Value::Null)),
            (
                "id",
                json::text(&tool_use_id(json::str_of(block, "id").unwrap_or_default())),
            ),
        ])),
        _ => Some(block.clone()),
    }
}

/// The model turn and reported tokens of a `/v1/messages` answer; `None` when it is not one.
#[must_use]
pub fn response(body: &[u8]) -> Option<(AssistantMessage, Option<Tokens>)> {
    let value = json::parse(body).ok()?;
    let content = json::array_of(&value, "content")?;
    let mut texts = Vec::new();
    let mut calls = Vec::new();
    for block in content {
        match json::str_of(block, "type")? {
            "text" => texts.push(json::str_of(block, "text")?.to_owned()),
            "tool_use" => calls.push(ToolCall {
                id: json::str_of(block, "id")?.to_owned(),
                name: json::str_of(block, "name")?.to_owned(),
                arguments: block
                    .get("input")
                    .filter(|input| matches!(input, Value::Object(_)))?
                    .clone(),
            }),
            _ => {}
        }
    }
    let stop = json::str_of(&value, "stop_reason");
    let turn = AssistantMessage {
        id: String::new(),
        text: texts.join("\n"),
        calls,
        invalid_calls: Vec::new(),
        refusal: stop == Some("refusal"),
        truncated: matches!(stop, Some("max_tokens" | "model_context_window_exceeded")),
        provider: PROVIDER.to_owned(),
        raw: Some(json::array(content.to_vec())),
        attempts: 1,
    };
    Some((turn, tokens(&value)))
}

/// The reported usage as `model_usage.py` counted the adapter's reading of it: input is the uncached input plus
/// cache reads and writes, and cache writes are the per-lifetime counts when the answer splits them.
fn tokens(value: &Value) -> Option<Tokens> {
    let usage = wire::present(value, "usage")?;
    let split = wire::present(usage, "cache_creation");
    let by_lifetime = wire::count(split, "ephemeral_5m_input_tokens")
        + wire::count(split, "ephemeral_1h_input_tokens");
    let cache_write = if by_lifetime > 0 {
        by_lifetime
    } else {
        wire::count(Some(usage), "cache_creation_input_tokens")
    };
    let cache_read = wire::count(Some(usage), "cache_read_input_tokens");
    Some(Tokens {
        input: wire::count(Some(usage), "input_tokens") + cache_read + cache_write,
        output: wire::count(Some(usage), "output_tokens"),
        cache_read,
        cache_write,
    })
}

#[cfg(test)]
mod tests;

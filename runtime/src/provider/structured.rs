//! Provider-native structured output and its closed re-validation, shared by every structured decision
//! (`structured.py`).

use team_protocol::python::text as python_text;
use team_protocol::python::value::py_eq;

use crate::conversation::{AssistantMessage, Block, Message};
use crate::json::Value;
use crate::provider::wire;
use crate::provider::{
    CallError, ChatRequest, OutputSchema, Profile, Projection, ProviderConfig, ProviderId,
    Providers, Scope,
};

/// The id of the one user message of a structured call.
const ASK_ID: &str = "shimpz-ask";

/// One stateless structured question: an optional system prompt and one user message, answered with one JSON value.
#[derive(Debug, Clone, PartialEq)]
pub struct Ask<'a> {
    /// The system prompt, if any.
    pub system: Option<&'a str>,
    /// The user message's blocks.
    pub user: Vec<Block>,
    /// The schema's name.
    pub name: &'a str,
    /// The JSON Schema of the answer.
    pub schema: &'a Value,
    /// The most characters the answer's text may have.
    pub max_chars: usize,
}

/// One stateless structured call of `ask` on `config`'s model.
///
/// # Errors
///
/// [`CallError::Request`] when the call failed, [`CallError::Response`] when the answer is not exactly one value
/// (a refusal, a tool call, text that is not one strict JSON object, or text blocks that disagree).
pub fn value(
    providers: &Providers,
    scope: &Scope,
    config: &ProviderConfig,
    profile: Profile,
    ask: &Ask<'_>,
) -> Result<Value, CallError> {
    // A message of one text block is plain text; any other content is that text followed by its other blocks.
    let (text, rest) = match ask.user.split_first() {
        Some((Block::Text(text), rest)) => (text.clone(), rest.to_vec()),
        _ => (String::new(), ask.user.clone()),
    };
    let request = ChatRequest {
        system: ask.system.map(str::to_owned),
        messages: vec![Message::User {
            id: ASK_ID.to_owned(),
            text,
        }],
        tools: Vec::new(),
        output: Some(OutputSchema {
            name: ask.name.to_owned(),
            schema: ask.schema.clone(),
        }),
        cache: false,
    };
    let projection = (!rest.is_empty()).then(|| Projection {
        message_id: ASK_ID.to_owned(),
        blocks: rest,
    });
    let answer = providers.chat(scope, config, &request, projection.as_ref(), profile, None)?;
    answered(&answer, config.provider, ask.max_chars).ok_or(CallError::Response)
}

/// The one value an answer wrote, or `None` when it refused, called a tool, or wrote anything but that value.
fn answered(answer: &AssistantMessage, provider: ProviderId, max_chars: usize) -> Option<Value> {
    if answer.refusal || !answer.calls.is_empty() || !answer.invalid_calls.is_empty() {
        return None;
    }
    agreed(&texts(answer, provider), max_chars)
}

/// The answer's text blocks, in order.
fn texts(answer: &AssistantMessage, provider: ProviderId) -> Vec<String> {
    let Some(Value::Array(items)) = &answer.raw else {
        return vec![answer.text.clone()];
    };
    let text_of = |block: &Value, kind: &str| {
        (crate::json::str_of(block, "type") == Some(kind))
            .then(|| crate::json::str_of(block, "text").map(str::to_owned))
            .flatten()
    };
    match provider {
        ProviderId::OpenAi => items
            .iter()
            .filter(|item| crate::json::str_of(item, "type") == Some("message"))
            .flat_map(|item| crate::json::array_of(item, "content").unwrap_or_default())
            .filter_map(|part| text_of(part, "output_text"))
            .collect(),
        ProviderId::Anthropic => items
            .iter()
            .filter_map(|block| text_of(block, "text"))
            .collect(),
    }
}

/// The value the whole text is, read strictly and within `max_chars`; a model that wrote the same value as separate
/// text blocks agrees when every block reads to it, and two different values never agree (`_agrees`).
fn agreed(blocks: &[String], max_chars: usize) -> Option<Value> {
    let whole = blocks.concat();
    let text = python_text::strip(&whole);
    if text.is_empty() {
        return None;
    }
    if let Some(value) = text_value(text, max_chars) {
        return Some(value);
    }
    let parts: Vec<&str> = blocks
        .iter()
        .map(|block| python_text::strip(block))
        .filter(|block| !block.is_empty())
        .collect();
    if parts.len() < 2 || text.chars().count() > max_chars {
        return None;
    }
    let first = text_value(parts[0], max_chars)?;
    parts[1..]
        .iter()
        .all(|part| text_value(part, max_chars).is_some_and(|value| py_eq(&value, &first)))
        .then_some(first)
}

fn text_value(text: &str, max_chars: usize) -> Option<Value> {
    if text.chars().count() > max_chars {
        return None;
    }
    wire::object_text(text)
}

#[cfg(test)]
mod tests;

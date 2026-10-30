//! A conversation as the Brain keeps it: the person's messages, the model's turns, and the results of its tool
//! calls. Provider adapters translate it both ways; the thread store and the run ledger keep it as strict JSON.

use crate::json::{self, Value};

/// One tool call of a model turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// The provider's call id, which its result answers.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// The call's arguments: a JSON object.
    pub arguments: Value,
}

/// A tool call whose arguments the provider sent as text that is not one JSON object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidCall {
    /// The provider's call id.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// The raw argument text.
    pub arguments: String,
}

/// One model turn.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMessage {
    /// A stable id within its conversation.
    pub id: String,
    /// The text blocks of the turn, joined with newlines (`context_budget.message_text`).
    pub text: String,
    /// Its parsed tool calls, in order.
    pub calls: Vec<ToolCall>,
    /// Its unparsable tool calls.
    pub invalid_calls: Vec<InvalidCall>,
    /// Whether the provider refused to answer (an OpenAI `refusal` content block, an Anthropic `refusal` stop).
    pub refusal: bool,
    /// Whether the provider reports that it cut the turn short (output limit or context window).
    pub truncated: bool,
    /// The provider that wrote it.
    pub provider: String,
    /// The provider's own output items or content blocks, exactly as received, replayed only to that provider;
    /// `None` once the Brain rewrote the turn (a capped reply), which replays as text and calls.
    pub raw: Option<Value>,
    /// The provider attempts this turn's call reserved: more than one when an attachment turn retried (ADR-0093).
    pub attempts: u32,
}

/// The result of one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolMessage {
    /// A stable id within its conversation.
    pub id: String,
    /// The call it answers.
    pub call_id: String,
    /// The tool's name.
    pub name: String,
    /// What the model reads.
    pub content: String,
    /// Whether the tool failed (an unknown tool).
    pub error: bool,
}

/// One message of a conversation.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// The person's message (or a quoted bridge of earlier presentation history).
    User {
        /// A stable id: a turn message id, or a bridge id.
        id: String,
        /// Its text.
        text: String,
    },
    /// A model turn.
    Assistant(AssistantMessage),
    /// A tool result.
    Tool(ToolMessage),
}

/// One block of request-local user content a provider call carries after the turn's message (ADR-0093).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    /// Text.
    Text(String),
    /// A JPEG or PNG image.
    Image {
        /// `image/jpeg` or `image/png`.
        media_type: String,
        /// Its standard base64 encoding.
        base64: String,
    },
}

impl Message {
    /// The message's id.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::User { id, .. } => id,
            Self::Assistant(message) => &message.id,
            Self::Tool(message) => &message.id,
        }
    }

    /// The model turn, if this is one.
    #[must_use]
    pub fn assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Self::Assistant(message) => Some(message),
            _ => None,
        }
    }

    /// The tool result, if this is one.
    #[must_use]
    pub fn tool(&self) -> Option<&ToolMessage> {
        match self {
            Self::Tool(message) => Some(message),
            _ => None,
        }
    }

    /// The message as strict JSON, for the thread store and the run ledger.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::User { id, text } => json::object([
                ("role", json::text("user")),
                ("id", json::text(id)),
                ("text", json::text(text)),
            ]),
            Self::Assistant(message) => assistant_value(message),
            Self::Tool(message) => json::object([
                ("role", json::text("tool")),
                ("id", json::text(&message.id)),
                ("call_id", json::text(&message.call_id)),
                ("name", json::text(&message.name)),
                ("content", json::text(&message.content)),
                ("error", Value::Bool(message.error)),
            ]),
        }
    }

    /// A message from [`Message::to_value`]'s form; `None` for anything else.
    #[must_use]
    pub fn from_value(value: &Value) -> Option<Self> {
        match json::str_of(value, "role")? {
            "user" if json::has_exactly(value, &["role", "id", "text"]) => Some(Self::User {
                id: json::str_of(value, "id")?.to_owned(),
                text: json::str_of(value, "text")?.to_owned(),
            }),
            "assistant" => assistant_from(value).map(Self::Assistant),
            "tool"
                if json::has_exactly(
                    value,
                    &["role", "id", "call_id", "name", "content", "error"],
                ) =>
            {
                Some(Self::Tool(ToolMessage {
                    id: json::str_of(value, "id")?.to_owned(),
                    call_id: json::str_of(value, "call_id")?.to_owned(),
                    name: json::str_of(value, "name")?.to_owned(),
                    content: json::str_of(value, "content")?.to_owned(),
                    error: json::bool_of(value, "error")?,
                }))
            }
            _ => None,
        }
    }
}

const ASSISTANT_FIELDS: [&str; 10] = [
    "role",
    "id",
    "text",
    "calls",
    "invalid_calls",
    "refusal",
    "truncated",
    "provider",
    "raw",
    "attempts",
];

fn assistant_value(message: &AssistantMessage) -> Value {
    let calls = message
        .calls
        .iter()
        .map(|call| {
            json::object([
                ("id", json::text(&call.id)),
                ("name", json::text(&call.name)),
                ("arguments", call.arguments.clone()),
            ])
        })
        .collect();
    let invalid = message
        .invalid_calls
        .iter()
        .map(|call| {
            json::object([
                ("id", json::text(&call.id)),
                ("name", json::text(&call.name)),
                ("arguments", json::text(&call.arguments)),
            ])
        })
        .collect();
    json::object([
        ("role", json::text("assistant")),
        ("id", json::text(&message.id)),
        ("text", json::text(&message.text)),
        ("calls", json::array(calls)),
        ("invalid_calls", json::array(invalid)),
        ("refusal", Value::Bool(message.refusal)),
        ("truncated", Value::Bool(message.truncated)),
        ("provider", json::text(&message.provider)),
        ("raw", message.raw.clone().unwrap_or(Value::Null)),
        ("attempts", json::uint(u64::from(message.attempts))),
    ])
}

fn assistant_from(value: &Value) -> Option<AssistantMessage> {
    if !json::has_exactly(value, &ASSISTANT_FIELDS) {
        return None;
    }
    let calls = json::array_of(value, "calls")?
        .iter()
        .map(|call| {
            let arguments = call.get("arguments")?;
            json::has_exactly(call, &["id", "name", "arguments"]).then_some(())?;
            matches!(arguments, Value::Object(_)).then_some(())?;
            Some(ToolCall {
                id: json::str_of(call, "id")?.to_owned(),
                name: json::str_of(call, "name")?.to_owned(),
                arguments: arguments.clone(),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let invalid_calls = json::array_of(value, "invalid_calls")?
        .iter()
        .map(|call| {
            json::has_exactly(call, &["id", "name", "arguments"]).then_some(())?;
            Some(InvalidCall {
                id: json::str_of(call, "id")?.to_owned(),
                name: json::str_of(call, "name")?.to_owned(),
                arguments: json::str_of(call, "arguments")?.to_owned(),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let raw = match value.get("raw")? {
        Value::Null => None,
        other => Some(other.clone()),
    };
    Some(AssistantMessage {
        id: json::str_of(value, "id")?.to_owned(),
        text: json::str_of(value, "text")?.to_owned(),
        calls,
        invalid_calls,
        refusal: json::bool_of(value, "refusal")?,
        truncated: json::bool_of(value, "truncated")?,
        provider: json::str_of(value, "provider")?.to_owned(),
        raw,
        attempts: u32::try_from(json::u64_of(value, "attempts")?)
            .ok()
            .filter(|attempts| *attempts >= 1)?,
    })
}

/// Encodes a conversation as a JSON array.
#[must_use]
pub fn to_value(messages: &[Message]) -> Value {
    json::array(messages.iter().map(Message::to_value).collect())
}

/// Decodes a conversation from [`to_value`]'s form; `None` when any message is not exactly one.
#[must_use]
pub fn from_value(value: &Value) -> Option<Vec<Message>> {
    match value {
        Value::Array(items) => items.iter().map(Message::from_value).collect(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Message> {
        vec![
            Message::User {
                id: "shimpz-turn-1".into(),
                text: "Hello".into(),
            },
            Message::Assistant(AssistantMessage {
                id: "a1".into(),
                text: String::new(),
                calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "tool".into(),
                    arguments: json::object([("x", json::int(1))]),
                }],
                invalid_calls: vec![InvalidCall {
                    id: "c2".into(),
                    name: "tool".into(),
                    arguments: "{".into(),
                }],
                refusal: false,
                truncated: false,
                provider: "openai".into(),
                raw: Some(json::array(vec![])),
                attempts: 1,
            }),
            Message::Tool(ToolMessage {
                id: "t1".into(),
                call_id: "c1".into(),
                name: "tool".into(),
                content: "ok".into(),
                error: false,
            }),
        ]
    }

    #[test]
    fn a_conversation_round_trips_and_refuses_any_other_shape() {
        let messages = sample();
        let value = to_value(&messages);
        assert_eq!(from_value(&value), Some(messages.clone()));
        assert_eq!(
            from_value(&json::parse(json::compact(&value).as_bytes()).unwrap()),
            Some(messages)
        );
        assert_eq!(
            from_value(&json::parse(br#"[{"role":"user","id":"x"}]"#).unwrap()),
            None
        );
        assert_eq!(from_value(&json::text("x")), None);
        let mut wrong = to_value(&sample());
        if let Value::Array(items) = &mut wrong
            && let Value::Object(members) = &mut items[1]
        {
            members.retain(|(name, _)| name != "raw");
        }
        assert_eq!(from_value(&wrong), None);
    }
}

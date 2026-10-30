//! Builders for the tools' tests: messages and a turn context.

use crate::conversation::{AssistantMessage, InvalidCall, Message, ToolCall, ToolMessage};
use crate::json::Value;

/// A model turn making `calls` (name, arguments), with ids `call-<n>`.
pub(crate) fn turn(calls: Vec<(&str, Value)>) -> Message {
    turn_with(calls, Vec::new())
}

/// A model turn making `calls` and the unparsable `invalid` ones.
pub(crate) fn turn_with(calls: Vec<(&str, Value)>, invalid: Vec<&str>) -> Message {
    Message::Assistant(AssistantMessage {
        id: "ai".into(),
        text: String::new(),
        calls: calls
            .into_iter()
            .enumerate()
            .map(|(index, (name, arguments))| ToolCall {
                id: format!("call-{index}"),
                name: name.into(),
                arguments,
            })
            .collect(),
        invalid_calls: invalid
            .into_iter()
            .map(|name| InvalidCall {
                id: "bad".into(),
                name: name.into(),
                arguments: "{".into(),
            })
            .collect(),
        refusal: false,
        truncated: false,
        provider: "openai".into(),
        raw: None,
        attempts: 1,
    })
}

/// A plain model reply.
pub(crate) fn reply(text: &str) -> Message {
    match turn(Vec::new()) {
        Message::Assistant(mut message) => {
            message.text = text.into();
            Message::Assistant(message)
        }
        other => other,
    }
}

/// A user message with `text`.
pub(crate) fn user(text: &str) -> Message {
    Message::User {
        id: "human".into(),
        text: text.into(),
    }
}

/// Team's start envelope around `message`.
pub(crate) fn envelope(message: &str) -> Message {
    let value = crate::json::object([
        ("files", crate::json::array(Vec::new())),
        ("message", crate::json::text(message)),
    ]);
    user(&crate::json::compact(&value))
}

/// A tool result.
pub(crate) fn result(name: &str, call_id: &str, content: &str) -> Message {
    Message::Tool(ToolMessage {
        id: format!("tool-{call_id}"),
        call_id: call_id.into(),
        name: name.into(),
        content: content.into(),
        error: false,
    })
}

/// Parses JSON text.
pub(crate) fn value(text: &str) -> Value {
    crate::json::parse(text.as_bytes()).expect("test JSON")
}

/// A closed object schema.
pub(crate) const CLOSED: &str = r#"{"type":"object","additionalProperties":false}"#;

/// One Action with a closed input schema.
pub(crate) fn action(id: &str) -> crate::turn::context::ActionDefinition {
    crate::turn::context::ActionDefinition {
        id: id.into(),
        summary: format!("Run {id}."),
        input_schema: value(CLOSED),
        output_schema: value(CLOSED),
        authorization: false,
        input_files: Vec::new(),
    }
}

/// A plain chat turn's context: one Assistant `hello-pulse` with `hello`, no memory, no Routines, no Browser.
pub(crate) fn context() -> crate::turn::context::TurnContext {
    crate::turn::context::TurnContext {
        thread_id: "team-thread".into(),
        team_name: "Pulse".into(),
        assistants: vec![crate::turn::context::AssistantDefinition {
            id: "hello-pulse".into(),
            genesis: "Greets people.".into(),
            actions: vec![action("hello")],
        }],
        provider: crate::provider::ProviderId::OpenAi,
        model: "gpt-6-luna".into(),
        turn_date: "2026-10-30".into(),
        timezone: "UTC".into(),
        memories: None,
        routines: None,
        routine_question: None,
        routine_mode: false,
        routine_rerun: None,
        routine_schedule: None,
        routine_output: None,
        routine_settings_answer: false,
        routine_selected: None,
        locale: None,
        turn_message_id: None,
        attachments: crate::turn::context::AttachmentPins::default(),
        browser: false,
    }
}

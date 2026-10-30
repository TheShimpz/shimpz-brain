//! The one way a guard refuses a model response (`tool_refusal.py`): every call of the refused response receives the
//! same closed correction, so each call stays paired with a result, nothing executes, and the model runs again.

use crate::conversation::{Message, ToolCall};
use crate::errors::BrainError;

/// The contract error of a guarded call that arrived beside an unparsable one, which no refusal can answer.
pub const UNPARSABLE: &str = "the model returned an unparsable tool call";

/// The latest response's calls to one tool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoneCall<'a> {
    /// The latest message is no model turn, or it makes no call to the tool.
    Absent,
    /// It calls the tool beside other calls.
    Mixed,
    /// It calls the tool alone.
    Alone(&'a ToolCall),
}

/// The latest response's one call to `tool` (`lone_call`); a guarded call beside an unparsable one is the contract
/// error "the model returned an unparsable tool call".
///
/// # Errors
///
/// [`BrainError::Contract`] as above.
pub fn lone_call<'a>(messages: &'a [Message], tool: &str) -> Result<LoneCall<'a>, BrainError> {
    let Some(latest) = messages.last().and_then(Message::assistant) else {
        return Ok(LoneCall::Absent);
    };
    let named = latest.calls.iter().any(|call| call.name == tool)
        || latest.invalid_calls.iter().any(|call| call.name == tool);
    if !named {
        return Ok(LoneCall::Absent);
    }
    if !latest.invalid_calls.is_empty() {
        return Err(BrainError::contract(UNPARSABLE));
    }
    match latest.calls.as_slice() {
        [call] => Ok(LoneCall::Alone(call)),
        _ => Ok(LoneCall::Mixed),
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! Message builders for the guards' tests.

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
}

#[cfg(test)]
mod tests {
    use super::testing::{reply, turn, turn_with, user, value};
    use super::*;

    #[test]
    fn a_lone_call_is_found_and_anything_else_is_not() {
        let arguments = value("{}");
        let alone = [user("hi"), turn(vec![("x", arguments.clone())])];
        assert!(matches!(lone_call(&alone, "x"), Ok(LoneCall::Alone(call)) if call.name == "x"));
        let mixed = [turn(vec![
            ("x", arguments.clone()),
            ("y", arguments.clone()),
        ])];
        assert_eq!(lone_call(&mixed, "x"), Ok(LoneCall::Mixed));
        assert_eq!(lone_call(&mixed, "z"), Ok(LoneCall::Absent));
        assert_eq!(lone_call(&[user("hi")], "x"), Ok(LoneCall::Absent));
        assert_eq!(lone_call(&[reply("done")], "x"), Ok(LoneCall::Absent));
        assert_eq!(lone_call(&[], "x"), Ok(LoneCall::Absent));
    }

    #[test]
    fn a_guarded_call_beside_an_unparsable_one_is_a_contract_error() {
        let refused = Err(BrainError::contract(UNPARSABLE));
        let beside = [turn_with(vec![("x", value("{}"))], vec!["y"])];
        assert_eq!(lone_call(&beside, "x"), refused);
        let itself = [turn_with(Vec::new(), vec!["x"])];
        assert_eq!(lone_call(&itself, "x"), refused);
        // An unparsable call to another tool, with no call to the guarded one, is not this guard's to judge.
        assert_eq!(lone_call(&beside, "z"), Ok(LoneCall::Absent));
    }
}

//! What a chat turn's Holdpoint run journals: its input, each model call, each Team request, and its outcome.
//!
//! Holdpoint keeps every value as JSON; the Brain wraps its own strict JSON as text inside it (`{"turn": "..."}`,
//! `{"call": "..."}`, ...), so member order and integers of any size survive exactly and Holdpoint's codec never has to
//! know the Brain's shapes. Nothing here holds the provider key or attachment content.

use serde_json::json;

use crate::conversation::{self, Message};
use crate::json::{self, Value};
use crate::provider::ToolSpec;
use crate::turn::context::TurnContext;

/// The definition id of every chat turn run.
pub const DEFINITION: &str = "brain-turn";
/// The registered model every call of a turn goes through.
pub const MODEL: &str = "provider";
/// The deferred request name of an Action request.
pub const ACTION: &str = "action";
/// The deferred request name of a Browser call.
pub const BROWSER: &str = "browser";

/// A started turn's captured input.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnInput {
    /// The pinned context.
    pub context: TurnContext,
    /// The system prompt.
    pub system: String,
    /// The tools offered (none once a Browser result is in).
    pub tools: Vec<ToolSpec>,
    /// The conversation the turn starts from: kept history, then the bridge (if any) and the turn message.
    pub messages: Vec<Message>,
    /// Where this logical turn's messages begin in `messages`.
    pub turn_start: usize,
    /// Whether the memory tool is offered.
    pub memory_tool: bool,
    /// Whether the Routine tool is offered.
    pub routine_tool: bool,
    /// Whether the Browser is offered.
    pub browser: bool,
}

fn text_value(value: &Value) -> serde_json::Value {
    serde_json::Value::String(json::compact(value))
}

fn unwrap_text(value: &serde_json::Value, key: &str) -> Option<Value> {
    json::parse(value.get(key)?.as_str()?.as_bytes()).ok()
}

/// A tool as journal JSON.
#[must_use]
pub fn tool_value(tool: &ToolSpec) -> Value {
    json::object([
        ("name", json::text(&tool.name)),
        ("description", json::text(&tool.description)),
        ("schema", tool.schema.clone()),
        ("strict", tool.strict.map_or(Value::Null, Value::Bool)),
    ])
}

/// A tool from [`tool_value`]'s form.
#[must_use]
pub fn tool_from(value: &Value) -> Option<ToolSpec> {
    json::has_exactly(value, &["name", "description", "schema", "strict"]).then_some(())?;
    let strict = match value.get("strict")? {
        Value::Null => None,
        Value::Bool(flag) => Some(*flag),
        _ => return None,
    };
    Some(ToolSpec {
        name: json::str_of(value, "name")?.to_owned(),
        description: json::str_of(value, "description")?.to_owned(),
        schema: value.get("schema")?.clone(),
        strict,
    })
}

impl TurnInput {
    /// The run input Holdpoint captures.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let value = json::object([
            ("context", self.context.to_value()),
            ("system", json::text(&self.system)),
            (
                "tools",
                json::array(self.tools.iter().map(tool_value).collect()),
            ),
            ("messages", conversation::to_value(&self.messages)),
            ("turn_start", json::uint(self.turn_start as u64)),
            ("memory_tool", Value::Bool(self.memory_tool)),
            ("routine_tool", Value::Bool(self.routine_tool)),
            ("browser", Value::Bool(self.browser)),
        ]);
        json!({ "turn": json::compact(&value) })
    }

    /// The input from a run's captured input; `None` for anything else.
    #[must_use]
    pub fn from_json(input: &serde_json::Value) -> Option<Self> {
        let value = unwrap_text(input, "turn")?;
        let fields = [
            "context",
            "system",
            "tools",
            "messages",
            "turn_start",
            "memory_tool",
            "routine_tool",
            "browser",
        ];
        json::has_exactly(&value, &fields).then_some(())?;
        Some(Self {
            context: TurnContext::from_value(value.get("context")?)?,
            system: json::str_of(&value, "system")?.to_owned(),
            tools: json::array_of(&value, "tools")?
                .iter()
                .map(tool_from)
                .collect::<Option<_>>()?,
            messages: conversation::from_value(value.get("messages")?)?,
            turn_start: usize::try_from(json::u64_of(&value, "turn_start")?).ok()?,
            memory_tool: json::bool_of(&value, "memory_tool")?,
            routine_tool: json::bool_of(&value, "routine_tool")?,
            browser: json::bool_of(&value, "browser")?,
        })
    }
}

/// One Team request a turn suspends on.
#[derive(Debug, Clone, PartialEq)]
pub enum TeamRequest {
    /// An Action request.
    Action {
        /// Its interrupt id.
        interrupt_id: String,
        /// The Assistant.
        assistant_id: String,
        /// The Action.
        action: String,
        /// The schema-valid input.
        input: Value,
    },
    /// The turn's one Browser call.
    Browser {
        /// Its interrupt id.
        interrupt_id: String,
        /// The canonical objective.
        objective: String,
    },
}

impl TeamRequest {
    /// The interrupt id Team answers it by.
    #[must_use]
    pub fn interrupt_id(&self) -> &str {
        match self {
            Self::Action { interrupt_id, .. } | Self::Browser { interrupt_id, .. } => interrupt_id,
        }
    }

    /// The deferred request's name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Action { .. } => ACTION,
            Self::Browser { .. } => BROWSER,
        }
    }

    /// The deferred request Holdpoint journals.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let value = match self {
            Self::Action {
                interrupt_id,
                assistant_id,
                action,
                input,
            } => json::object([
                ("interrupt_id", json::text(interrupt_id)),
                ("assistant_id", json::text(assistant_id)),
                ("action", json::text(action)),
                ("input", input.clone()),
            ]),
            Self::Browser {
                interrupt_id,
                objective,
            } => json::object([
                ("interrupt_id", json::text(interrupt_id)),
                ("objective", json::text(objective)),
            ]),
        };
        json!({ "request": json::compact(&value) })
    }

    /// The request a pending deferred step named `name` holds.
    #[must_use]
    pub fn from_json(name: &str, request: &serde_json::Value) -> Option<Self> {
        let value = unwrap_text(request, "request")?;
        let interrupt_id = json::str_of(&value, "interrupt_id")?.to_owned();
        match name {
            ACTION
                if json::has_exactly(
                    &value,
                    &["interrupt_id", "assistant_id", "action", "input"],
                ) =>
            {
                Some(Self::Action {
                    interrupt_id,
                    assistant_id: json::str_of(&value, "assistant_id")?.to_owned(),
                    action: json::str_of(&value, "action")?.to_owned(),
                    input: value.get("input")?.clone(),
                })
            }
            BROWSER if json::has_exactly(&value, &["interrupt_id", "objective"]) => {
                Some(Self::Browser {
                    interrupt_id,
                    objective: json::str_of(&value, "objective")?.to_owned(),
                })
            }
            _ => None,
        }
    }
}

/// Team's result for a deferred request, as Holdpoint records it.
#[must_use]
pub fn result_json(result: &Value) -> serde_json::Value {
    json!({ "result": json::compact(result) })
}

/// Team's result from its recorded form.
#[must_use]
pub fn result_from(value: &serde_json::Value) -> Option<Value> {
    unwrap_text(value, "result")
}

/// A model call's journaled answer: `{"message": "..."}` (a model turn), `{"verdict": "..."}` (a check's value), or
/// `{"verdict": null}` (a check that could not answer).
#[must_use]
pub fn answer_json(key: &str, value: Option<&Value>) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert(
        key.to_owned(),
        value.map_or(serde_json::Value::Null, text_value),
    );
    serde_json::Value::Object(object)
}

/// The value of a model call's journaled answer under `key`: `Some(None)` for a recorded null.
#[must_use]
pub fn answer_from(value: &serde_json::Value, key: &str) -> Option<Option<Value>> {
    match value.get(key)? {
        serde_json::Value::Null => Some(None),
        serde_json::Value::String(text) => json::parse(text.as_bytes()).ok().map(Some),
        _ => None,
    }
}

/// The turn's outcome, wrapped as the run's result.
#[must_use]
pub fn outcome_json(value: &Value) -> serde_json::Value {
    json!({ "outcome": json::compact(value) })
}

/// The turn's outcome from the run's result.
#[must_use]
pub fn outcome_from(value: &serde_json::Value) -> Option<Value> {
    unwrap_text(value, "outcome")
}

/// A model call request as Holdpoint journals it (only its digest is stored).
#[must_use]
pub fn call_json(value: &Value) -> serde_json::Value {
    json!({ "call": json::compact(value) })
}

/// A model call request from its journaled form.
#[must_use]
pub fn call_from(value: &serde_json::Value) -> Option<Value> {
    unwrap_text(value, "call")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::context::tests::sample;

    #[test]
    fn every_journal_form_round_trips() {
        let input = TurnInput {
            context: sample(),
            system: "System".into(),
            tools: vec![ToolSpec {
                name: "t".into(),
                description: "d".into(),
                schema: json::object([]),
                strict: Some(false),
            }],
            messages: vec![Message::User {
                id: "u".into(),
                text: "Hi".into(),
            }],
            turn_start: 0,
            memory_tool: true,
            routine_tool: false,
            browser: true,
        };
        assert_eq!(TurnInput::from_json(&input.to_json()), Some(input));
        assert_eq!(TurnInput::from_json(&json!({ "turn": "[]" })), None);
    }

    #[test]
    fn team_requests_and_answers_round_trip() {
        let action = TeamRequest::Action {
            interrupt_id: "i1".into(),
            assistant_id: "dns".into(),
            action: "list".into(),
            input: json::object([("big", json::uint(u64::MAX))]),
        };
        assert_eq!(
            TeamRequest::from_json(ACTION, &action.to_json()),
            Some(action.clone())
        );
        assert_eq!(TeamRequest::from_json(BROWSER, &action.to_json()), None);
        let browser = TeamRequest::Browser {
            interrupt_id: "i2".into(),
            objective: "Find it".into(),
        };
        assert_eq!(
            TeamRequest::from_json(BROWSER, &browser.to_json()),
            Some(browser)
        );
        let result = json::object([("ok", Value::Bool(true))]);
        assert_eq!(result_from(&result_json(&result)), Some(result.clone()));
        assert_eq!(
            answer_from(&answer_json("verdict", None), "verdict"),
            Some(None)
        );
        assert_eq!(
            answer_from(&answer_json("message", Some(&result)), "message"),
            Some(Some(result.clone()))
        );
        assert_eq!(outcome_from(&outcome_json(&result)), Some(result.clone()));
        assert_eq!(call_from(&call_json(&result)), Some(result));
    }
}

//! One chat turn's admitted context: the Team's Assistants and their Actions, its knowledge and Routine recording
//! state, and the pins a logical turn keeps from its start to its last resume (`agent_runtime.TurnContext`,
//! `turn_pins.py`). A started turn's context is journaled in its run input, so every resume finishes with exactly
//! the date, zone, knowledge, model, language, and attachment commitment its start recorded. Neither the provider key
//! nor attachment content is part of it.

use crate::json::{self, Value};
use crate::provider::ProviderId;

/// One Action an Assistant declares.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionDefinition {
    /// The Action id.
    pub id: String,
    /// Its summary.
    pub summary: String,
    /// Its admitted input schema.
    pub input_schema: Value,
    /// Its admitted output schema (validated with the contract, never offered to a tool).
    pub output_schema: Value,
    /// Whether it declares an authorization capability (offered while attachment content is in the turn).
    pub authorization: bool,
    /// The input properties that take one attached file's id (at most one).
    pub input_files: Vec<String>,
}

/// One Assistant in the turn's scope.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantDefinition {
    /// The Assistant id.
    pub id: String,
    /// Its Genesis.
    pub genesis: String,
    /// Its Actions, sorted by id.
    pub actions: Vec<ActionDefinition>,
}

/// One remembered preference of the Team (ADR-0084).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Memory {
    /// The topic.
    pub topic: String,
    /// The preference.
    pub preference: String,
}

/// The attachment facts a turn keeps: never the content, which Team resends with every resume (ADR-0093).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AttachmentPins {
    /// Each attachment's content kind, in order: `text`, `image`, or `opaque`.
    pub kinds: Vec<String>,
    /// The digest of the exact prepared attachments.
    pub commitment: String,
    /// The token charge each provider call of the turn carries for them.
    pub charge: u64,
}

impl AttachmentPins {
    /// Whether any attachment's text or image is in the turn's provider calls.
    #[must_use]
    pub fn reads_content(&self) -> bool {
        self.kinds
            .iter()
            .any(|kind| kind == "text" || kind == "image")
    }

    /// Whether the turn carries attachments.
    #[must_use]
    pub fn any(&self) -> bool {
        !self.kinds.is_empty()
    }
}

/// One chat turn's context.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnContext {
    /// The conversation thread.
    pub thread_id: String,
    /// The Team's display name, exactly as Team sent it.
    pub team_name: String,
    /// The Assistants in scope, sorted by id.
    pub assistants: Vec<AssistantDefinition>,
    /// The provider the turn runs on.
    pub provider: ProviderId,
    /// The chat model the turn runs on.
    pub model: String,
    /// The date the turn reasons with (`YYYY-MM-DD`).
    pub turn_date: String,
    /// The IANA zone that date is read in.
    pub timezone: String,
    /// What the Team remembers; `None` withholds memory.
    pub memories: Option<Vec<Memory>>,
    /// The Team's Routines as data; `None` withholds the Routine tool.
    pub routines: Option<Vec<Value>>,
    /// The question Team asked about this recording.
    pub routine_question: Option<Value>,
    /// Whether Team read that this chat is about a Routine.
    pub routine_mode: bool,
    /// The work a pending rerun question asks to run again.
    pub routine_rerun: Option<Vec<Value>>,
    /// The one schedule Team read from the person's words.
    pub routine_schedule: Option<Value>,
    /// The one output mode Team read from the person's words.
    pub routine_output: Option<String>,
    /// Whether the message is exactly one fixed answer to Team's settings question (record only).
    pub routine_settings_answer: bool,
    /// The one Routine the person's send selects (ADR-0122).
    pub routine_selected: Option<Value>,
    /// The interface language every reply follows; `None` follows the person's message.
    pub locale: Option<String>,
    /// The id of the message that started the logical turn.
    pub turn_message_id: Option<String>,
    /// The turn's attachment facts.
    pub attachments: AttachmentPins,
    /// Whether Team offers the Browser this turn (ADR-0117).
    pub browser: bool,
}

fn optional(value: Option<Value>) -> Value {
    value.unwrap_or(Value::Null)
}

fn optional_list(values: Option<&Vec<Value>>) -> Value {
    values.map_or(Value::Null, |values| json::array(values.clone()))
}

fn optional_text(value: Option<&String>) -> Value {
    value.map_or(Value::Null, |value| json::text(value))
}

fn action_value(action: &ActionDefinition) -> Value {
    json::object([
        ("id", json::text(&action.id)),
        ("summary", json::text(&action.summary)),
        ("input_schema", action.input_schema.clone()),
        ("output_schema", action.output_schema.clone()),
        ("authorization", Value::Bool(action.authorization)),
        (
            "input_files",
            json::array(
                action
                    .input_files
                    .iter()
                    .map(|name| json::text(name))
                    .collect(),
            ),
        ),
    ])
}

fn assistant_value(assistant: &AssistantDefinition) -> Value {
    json::object([
        ("id", json::text(&assistant.id)),
        ("genesis", json::text(&assistant.genesis)),
        (
            "actions",
            json::array(assistant.actions.iter().map(action_value).collect()),
        ),
    ])
}

fn memories_value(memories: Option<&Vec<Memory>>) -> Value {
    memories.map_or(Value::Null, |memories| {
        json::array(
            memories
                .iter()
                .map(|memory| {
                    json::object([
                        ("topic", json::text(&memory.topic)),
                        ("preference", json::text(&memory.preference)),
                    ])
                })
                .collect(),
        )
    })
}

impl TurnContext {
    /// The context as strict JSON, for the run input.
    #[must_use]
    pub fn to_value(&self) -> Value {
        json::object([
            ("thread_id", json::text(&self.thread_id)),
            ("team_name", json::text(&self.team_name)),
            (
                "assistants",
                json::array(self.assistants.iter().map(assistant_value).collect()),
            ),
            ("provider", json::text(self.provider.as_str())),
            ("model", json::text(&self.model)),
            ("turn_date", json::text(&self.turn_date)),
            ("timezone", json::text(&self.timezone)),
            ("memories", memories_value(self.memories.as_ref())),
            ("routines", optional_list(self.routines.as_ref())),
            ("routine_question", optional(self.routine_question.clone())),
            ("routine_mode", Value::Bool(self.routine_mode)),
            ("routine_rerun", optional_list(self.routine_rerun.as_ref())),
            ("routine_schedule", optional(self.routine_schedule.clone())),
            (
                "routine_output",
                optional_text(self.routine_output.as_ref()),
            ),
            (
                "routine_settings_answer",
                Value::Bool(self.routine_settings_answer),
            ),
            ("routine_selected", optional(self.routine_selected.clone())),
            ("locale", optional_text(self.locale.as_ref())),
            (
                "turn_message_id",
                optional_text(self.turn_message_id.as_ref()),
            ),
            (
                "attachment_kinds",
                json::array(
                    self.attachments
                        .kinds
                        .iter()
                        .map(|kind| json::text(kind))
                        .collect(),
                ),
            ),
            (
                "attachment_commitment",
                json::text(&self.attachments.commitment),
            ),
            ("attachment_charge", json::uint(self.attachments.charge)),
            ("browser", Value::Bool(self.browser)),
        ])
    }

    /// A context from [`TurnContext::to_value`]'s form; `None` for anything else.
    #[must_use]
    pub fn from_value(value: &Value) -> Option<Self> {
        decode::context(value)
    }
}

mod decode {
    use super::{ActionDefinition, AssistantDefinition, AttachmentPins, Memory, TurnContext};
    use crate::json::{self, Value};
    use crate::provider::ProviderId;

    const FIELDS: [&str; 22] = [
        "thread_id",
        "team_name",
        "assistants",
        "provider",
        "model",
        "turn_date",
        "timezone",
        "memories",
        "routines",
        "routine_question",
        "routine_mode",
        "routine_rerun",
        "routine_schedule",
        "routine_output",
        "routine_settings_answer",
        "routine_selected",
        "locale",
        "turn_message_id",
        "attachment_kinds",
        "attachment_commitment",
        "attachment_charge",
        "browser",
    ];

    fn text(value: &Value, key: &str) -> Option<String> {
        json::str_of(value, key).map(str::to_owned)
    }

    /// A member that is `null` or decodes with `read`; `Err` when it is neither (or missing).
    fn nullable<T>(
        value: &Value,
        key: &str,
        read: impl FnOnce(&Value) -> Option<T>,
    ) -> Result<Option<T>, ()> {
        match value.get(key).ok_or(())? {
            Value::Null => Ok(None),
            other => read(other).map(Some).ok_or(()),
        }
    }

    fn list(value: &Value) -> Option<Vec<Value>> {
        match value {
            Value::Array(items) => Some(items.clone()),
            _ => None,
        }
    }

    fn string(value: &Value) -> Option<String> {
        match value {
            Value::String(text) => Some(text.clone()),
            _ => None,
        }
    }

    fn strings(value: &Value) -> Option<Vec<String>> {
        match value {
            Value::Array(items) => items.iter().map(string).collect(),
            _ => None,
        }
    }

    fn action(value: &Value) -> Option<ActionDefinition> {
        json::has_exactly(
            value,
            &[
                "id",
                "summary",
                "input_schema",
                "output_schema",
                "authorization",
                "input_files",
            ],
        )
        .then_some(())?;
        Some(ActionDefinition {
            id: text(value, "id")?,
            summary: text(value, "summary")?,
            input_schema: value.get("input_schema")?.clone(),
            output_schema: value.get("output_schema")?.clone(),
            authorization: json::bool_of(value, "authorization")?,
            input_files: strings(value.get("input_files")?)?,
        })
    }

    fn assistant(value: &Value) -> Option<AssistantDefinition> {
        json::has_exactly(value, &["id", "genesis", "actions"]).then_some(())?;
        Some(AssistantDefinition {
            id: text(value, "id")?,
            genesis: text(value, "genesis")?,
            actions: json::array_of(value, "actions")?
                .iter()
                .map(action)
                .collect::<Option<_>>()?,
        })
    }

    fn memories(value: &Value) -> Option<Vec<Memory>> {
        match value {
            Value::Array(items) => items
                .iter()
                .map(|item| {
                    json::has_exactly(item, &["topic", "preference"]).then_some(())?;
                    Some(Memory {
                        topic: text(item, "topic")?,
                        preference: text(item, "preference")?,
                    })
                })
                .collect(),
            _ => None,
        }
    }

    struct Routine {
        routines: Option<Vec<Value>>,
        question: Option<Value>,
        mode: bool,
        rerun: Option<Vec<Value>>,
        schedule: Option<Value>,
        output: Option<String>,
        settings_answer: bool,
        selected: Option<Value>,
    }

    fn routine(value: &Value) -> Result<Routine, ()> {
        let clone = |item: &Value| Some(item.clone());
        Ok(Routine {
            routines: nullable(value, "routines", list)?,
            question: nullable(value, "routine_question", clone)?,
            mode: json::bool_of(value, "routine_mode").ok_or(())?,
            rerun: nullable(value, "routine_rerun", list)?,
            schedule: nullable(value, "routine_schedule", clone)?,
            output: nullable(value, "routine_output", string)?,
            settings_answer: json::bool_of(value, "routine_settings_answer").ok_or(())?,
            selected: nullable(value, "routine_selected", clone)?,
        })
    }

    fn pins(value: &Value) -> Option<AttachmentPins> {
        Some(AttachmentPins {
            kinds: strings(value.get("attachment_kinds")?)?,
            commitment: text(value, "attachment_commitment")?,
            charge: json::u64_of(value, "attachment_charge")?,
        })
    }

    fn assistants(value: &Value) -> Option<Vec<AssistantDefinition>> {
        json::array_of(value, "assistants")?
            .iter()
            .map(assistant)
            .collect()
    }

    pub(super) fn context(value: &Value) -> Option<TurnContext> {
        json::has_exactly(value, &FIELDS).then_some(())?;
        let routine = routine(value).ok()?;
        let turn = (
            nullable(value, "locale", string).ok()?,
            nullable(value, "turn_message_id", string).ok()?,
        );
        Some(TurnContext {
            thread_id: text(value, "thread_id")?,
            team_name: text(value, "team_name")?,
            assistants: assistants(value)?,
            provider: ProviderId::parse(json::str_of(value, "provider")?)?,
            model: text(value, "model")?,
            turn_date: text(value, "turn_date")?,
            timezone: text(value, "timezone")?,
            memories: nullable(value, "memories", memories).ok()?,
            routines: routine.routines,
            routine_question: routine.question,
            routine_mode: routine.mode,
            routine_rerun: routine.rerun,
            routine_schedule: routine.schedule,
            routine_output: routine.output,
            routine_settings_answer: routine.settings_answer,
            routine_selected: routine.selected,
            locale: turn.0,
            turn_message_id: turn.1,
            attachments: pins(value)?,
            browser: json::bool_of(value, "browser")?,
        })
    }
}

#[cfg(test)]
mod tests;

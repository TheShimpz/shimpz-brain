//! Routines recorded from work the chat agent actually ran (`routine.py`, ADR-0101, ADR-0122).
//!
//! When the person asks for work to recur, the chat agent clarifies as for any task, runs the recurring work once in
//! the same turn, and then calls one closed tool, `shimpz_routine` `record`, with the Routine's name, a plain
//! description of the work, the Routine it replaces, and the reply the person reads. The schedule, timezone, and output
//! are never the model's: Team derives them from the person's own words. The guard checks the closed shape and ends
//! the turn; nothing here schedules, approves, or authorizes anything.

mod refine;
mod rerun;

use std::collections::BTreeSet;
use std::sync::LazyLock;

use team_protocol::python::text;
use team_protocol::python::value::py_eq;
use team_protocol::routine::{
    DAY_SECONDS, MAX_CONTINUOUS_GAP_SECONDS, MAX_ROUTINE_DESCRIPTION_CHARS, MAX_ROUTINE_NAME_CHARS,
    MIN_CONTINUOUS_GAP_SECONDS, OUTPUT_MODES, canonical_description, canonical_name,
    canonical_schedule,
};
use team_protocol::routine_context::{
    CALL_HANDLE_RE, canonical_rerun, canonical_routine_listings, canonical_selected_routine,
};
use team_protocol::routine_proposal::{SETTINGS_QUESTION_CODES, canonical_question};
use team_protocol::turn::REPLY_CONTROL_RE;

pub use self::rerun::rerun_review;
use crate::conversation::Message;
use crate::errors::BrainError;
use crate::json::{self, Value};
use crate::provider::ToolSpec;
use crate::tools::refusal::{self, LoneCall};
use crate::turn::context::TurnContext;

/// The tool's name.
pub const TOOL_NAME: &str = "shimpz_routine";

/// The most characters of the reply the person reads beside the card.
const MAX_REPLY_CHARS: usize = 4000;

/// Every continuous interval Team admits, as the prompt states it (`INTERVALS`).
#[must_use]
pub fn intervals() -> String {
    format!(
        "{MIN_CONTINUOUS_GAP_SECONDS} seconds to {} days",
        MAX_CONTINUOUS_GAP_SECONDS / DAY_SECONDS
    )
}

/// The record tool's description (`DESCRIPTION`).
static DESCRIPTION: LazyLock<String> = LazyLock::new(|| {
    format!(
        "Record the work you just ran as a Routine the Team repeats on a schedule. Call it only when the person asks \
         for work to recur or to change a listed Routine, only after you ran exactly that work once, unless only a \
         listed Routine's schedule or output changes or the person answers the Team's question about how often it \
         runs or what to do with each run's result without changing the work, and alone in your response. It takes \
         no schedule, timezone, or output: the Team reads how often (any interval from {}, at most \
         ceil({DAY_SECONDS} / interval) runs a day) and what to do with each run's result from the person's own \
         words, in the person's own timezone, asks the person itself when something is missing or does not fit, and \
         shows them on its card. It ends the turn; the Team then shows the person that card to confirm, so never say \
         a Routine was created.",
        intervals()
    )
});

/// The description while the send selects a Routine (`REFINE_DESCRIPTION`).
static REFINE_DESCRIPTION: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{} With op refine it instead changes the selected Routine: after you ran in this message only the new or \
         corrected calls, it adds, replaces, or removes steps by their ids and those calls by their handles, and the \
         Team keeps every other step as it is.",
        *DESCRIPTION
    )
});

const NAME_TEXT: &str = "A short name for the Routine's work, at most {name} characters, without the word Routine \
    or how often it runs.";
const DESCRIPTION_TEXT: &str = "One plain paragraph, at most {description} characters, in everyday words for a \
    person who is not technical: what every run does and on what. Describe only the work you ran and say plainly when \
    a run changes something. Leave out the person's words about how often or when it runs, its timezone, and whether \
    or when the result is shown to them (such as 'show me the result' or 'only when it changes'): the Team's card \
    shows those itself.";
const REPLACES_TEXT: &str =
    "The routine_id of the listed Routine this one changes; null for a new Routine.";
const REPLY_TEXT: &str = "Your reply to the person: the result of the work you ran in this message, in their \
    language; empty when you ran no Action in this message, since the card says everything.";
const EDITS_TEXT: &str = "With op refine, the changes to the selected Routine's steps, at most one per step; null \
    with op record. insert adds the calls you ran in this message after a step, in the order you ran them; replace \
    puts one such call in a step's place; remove drops a step.";
const CALLS_TEXT: &str = "The call handles of Action results of this message, such as c1: one or more to insert, \
    exactly one to replace, none to remove.";

/// A schema member: its type, then its description.
fn member(kind: Value, description: &str) -> Value {
    json::object([("type", kind), ("description", json::text(description))])
}

fn nullable(kind: &str) -> Value {
    json::array(vec![json::text(kind), json::text("null")])
}

fn texts() -> (String, String) {
    (
        NAME_TEXT.replace("{name}", &MAX_ROUTINE_NAME_CHARS.to_string()),
        DESCRIPTION_TEXT.replace("{description}", &MAX_ROUTINE_DESCRIPTION_CHARS.to_string()),
    )
}

fn closed(properties: Vec<(String, Value)>) -> Value {
    let required = properties
        .iter()
        .map(|(name, _)| json::text(name))
        .collect();
    json::object([
        ("type", json::text("object")),
        ("properties", json::object_of(properties)),
        ("required", json::array(required)),
        ("additionalProperties", Value::Bool(false)),
    ])
}

fn ops(values: &[&str]) -> Value {
    json::object([
        ("type", json::text("string")),
        (
            "enum",
            json::array(values.iter().map(|value| json::text(value)).collect()),
        ),
    ])
}

/// The record tool's schema (`SCHEMA`).
static SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    let (name, description) = texts();
    closed(vec![
        ("op".into(), ops(&["record"])),
        ("name".into(), member(json::text("string"), &name)),
        (
            "description".into(),
            member(json::text("string"), &description),
        ),
        ("replaces".into(), member(nullable("string"), REPLACES_TEXT)),
        ("reply".into(), member(json::text("string"), REPLY_TEXT)),
    ])
});

fn edits_member() -> Value {
    let kinds = team_protocol::routine_context::REFINE_KINDS;
    let calls = json::object([
        ("type", json::text("array")),
        ("items", json::object([("type", json::text("string"))])),
        ("description", json::text(CALLS_TEXT)),
    ]);
    let item = closed(vec![
        ("kind".into(), ops(&kinds)),
        (
            "step".into(),
            member(
                json::text("string"),
                "The id of a step of the selected Routine.",
            ),
        ),
        ("calls".into(), calls),
    ]);
    json::object([
        ("type", nullable("array")),
        ("description", json::text(EDITS_TEXT)),
        ("items", item),
    ])
}

/// The tool's schema while the send selects a Routine (`REFINE_SCHEMA`).
static REFINE_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    let (name, description) = texts();
    closed(vec![
        ("op".into(), ops(&["record", "refine"])),
        (
            "name".into(),
            member(
                nullable("string"),
                &format!("{name} Null with op refine, which keeps the Routine's name."),
            ),
        ),
        (
            "description".into(),
            member(
                json::text("string"),
                &format!("{description} With op refine, describe the whole refined Routine."),
            ),
        ),
        (
            "replaces".into(),
            member(
                nullable("string"),
                &format!("{REPLACES_TEXT} Null with op refine."),
            ),
        ),
        (
            "routine_id".into(),
            member(
                nullable("string"),
                "With op refine, the routine_id of the selected Routine; null with op record.",
            ),
        ),
        (
            "base_revision".into(),
            member(
                nullable("integer"),
                "With op refine, the selected Routine's revision as shown; null with op record.",
            ),
        ),
        ("edits".into(), edits_member()),
        ("reply".into(), member(json::text("string"), REPLY_TEXT)),
    ])
});

/// How a call outside the closed shape is answered: nothing was recorded, and the model may call again.
fn correction(field: &str) -> String {
    let detail = match field {
        "mixed" => "Call shimpz_routine alone in its response, after the work ran; nothing else in this response ran, \
             so repeat the calls you still need."
            .to_owned(),
        "invalid" => {
            "Call it with op record and exactly the members its schema names.".to_owned()
        }
        "name" => format!("Give a name of one line, at most {MAX_ROUTINE_NAME_CHARS} characters."),
        "description" => format!(
            "Give a description of one plain paragraph, at most {MAX_ROUTINE_DESCRIPTION_CHARS} characters."
        ),
        "replaces" => "replaces must be the routine_id of a listed Routine, or null for a new one.".to_owned(),
        "reply" => format!(
            "Give the person an empty reply, or text that says something, of at most {MAX_REPLY_CHARS} characters \
             with no control characters other than line breaks and tabs."
        ),
        "refine" => "With op refine, give routine_id and base_revision exactly as the selected Routine shows them, \
             name null, replaces null, and edits with at most one edit per step id it lists; with op record, give \
             routine_id, base_revision, and edits null."
            .to_owned(),
        "calls" => "Name in edits only the call handles of Action results of this message, each once: one or more \
             to insert, exactly one to replace, none to remove. Run the missing work first."
            .to_owned(),
        "dependency" => "A step you remove or replace is read by a step you keep. Record that whole Routine again \
             with op record and replaces instead, after running all of its work once."
            .to_owned(),
        _ => "Change at least one step or the description.".to_owned(),
    };
    format!("Not done: nothing was recorded. {detail}")
}

fn contract(detail: &str) -> BrainError {
    BrainError::contract(detail)
}

/// Whether Team sends this recording field beside the others the context carries, the question already admitted
/// (`_sent_with`).
fn sent_with(context: &TurnContext, field: &str) -> bool {
    match field {
        "routine_selected" => {
            let selected = context
                .routine_selected
                .as_ref()
                .and_then(canonical_selected_routine);
            let listed = |selected: &Value| {
                context.routines.iter().flatten().any(|item| {
                    let same = |key: &str| match (item.get(key), selected.get(key)) {
                        (Some(left), Some(right)) => py_eq(left, right),
                        _ => false,
                    };
                    same("routine_id") && same("revision")
                })
            };
            context.routine_mode && selected.is_some_and(listed)
        }
        "routine_question" | "routine_schedule" => context.routine_mode,
        "routine_rerun" => context
            .routine_question
            .as_ref()
            .is_some_and(|question| json::str_of(question, "code") == Some("routine-work-rerun")),
        _ => true,
    }
}

/// Whether one recording field is admitted: sent beside the Team's Routines, with the fields Team sends it with, in
/// Team's canonical form.
fn admitted(context: &TurnContext, field: &str) -> bool {
    if context.routines.is_none() || !sent_with(context, field) {
        return false;
    }
    match field {
        "routine_question" => context
            .routine_question
            .as_ref()
            .is_some_and(|question| canonical_question(question).is_some()),
        "routine_rerun" => context
            .routine_rerun
            .as_ref()
            .is_some_and(|work| canonical_rerun(&json::array(work.clone())).is_some()),
        "routine_schedule" => context
            .routine_schedule
            .as_ref()
            .is_some_and(|schedule| canonical_schedule(schedule).is_some()),
        "routine_output" => context
            .routine_output
            .as_deref()
            .is_some_and(|output| OUTPUT_MODES.contains(&output)),
        _ => true,
    }
}

/// Canonicalizes the turn's Routines and recording fields in place (`admit_context`).
///
/// Everything about a recording comes only beside the Team's Routines, and only in the combinations Team's recording
/// makes: never with attachments, a question or schedule only in Routine mode, and rerun work only beside a rerun
/// question. Team's canonical forms are its admitted values themselves, so admission leaves each field as sent.
///
/// # Errors
///
/// [`BrainError::Contract`] with the Python texts ("invalid routines", "invalid Routine mode", ...).
pub fn admit_context(context: &mut TurnContext) -> Result<(), BrainError> {
    if let Some(routines) = &context.routines
        && (context.attachments.any()
            || canonical_routine_listings(&json::array(routines.clone())).is_none())
    {
        return Err(contract("invalid routines"));
    }
    if context.routine_mode && context.routines.is_none() {
        return Err(contract("invalid Routine mode"));
    }
    for (field, present, name) in [
        (
            "routine_question",
            context.routine_question.is_some(),
            "question",
        ),
        ("routine_rerun", context.routine_rerun.is_some(), "rerun"),
        (
            "routine_schedule",
            context.routine_schedule.is_some(),
            "schedule",
        ),
        ("routine_output", context.routine_output.is_some(), "output"),
        (
            "routine_selected",
            context.routine_selected.is_some(),
            "selection",
        ),
    ] {
        if present && !admitted(context, field) {
            return Err(contract(&format!("invalid Routine {name}")));
        }
    }
    if context.routine_settings_answer && !settled(context) {
        return Err(contract("invalid Routine settings answer"));
    }
    Ok(())
}

/// Whether a settings answer answers a settings question, with the reading Team took from it (`_settled`).
fn settled(context: &TurnContext) -> bool {
    let code = context
        .routine_question
        .as_ref()
        .and_then(|question| json::str_of(question, "code"));
    if code == Some("routine-output-unstated") {
        return context.routine_output.is_some();
    }
    code.is_some_and(|code| SETTINGS_QUESTION_CODES.contains(&code))
        && context.routine_schedule.is_some()
}

/// Whether Team's pending question asks how often the Routine runs or what each run does with its result.
#[must_use]
pub fn settings_question(question: Option<&Value>) -> bool {
    question
        .and_then(|question| json::str_of(question, "code"))
        .is_some_and(|code| SETTINGS_QUESTION_CODES.contains(&code))
}

/// The tool, with `refine` while the send selects a Routine.
#[must_use]
pub fn spec(refinable: bool) -> ToolSpec {
    let (schema, description) = if refinable {
        (&*REFINE_SCHEMA, &*REFINE_DESCRIPTION)
    } else {
        (&*SCHEMA, &*DESCRIPTION)
    };
    ToolSpec {
        name: TOOL_NAME.to_owned(),
        description: description.clone(),
        schema: schema.clone(),
        strict: None,
    }
}

/// The Routine guard's verdict on the latest model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The response makes no call to the tool.
    Pass,
    /// The correction answering every call of the response.
    Refuse(String),
    /// The lone call's result content (the compact JSON record), which ends the turn.
    Record(String),
}

/// The Routine guard (`RoutineGuard.after_model`): it always answers a call to the tool before the tool could run.
///
/// # Errors
///
/// An unanswerable call.
pub fn review(messages: &[Message], context: &TurnContext) -> Result<Verdict, BrainError> {
    Ok(match refusal::lone_call(messages, TOOL_NAME)? {
        LoneCall::Absent => Verdict::Pass,
        LoneCall::Mixed => Verdict::Refuse(correction("mixed")),
        LoneCall::Alone(call) => match record(&call.arguments, context, &call_handles(messages)) {
            Ok(outcome) => Verdict::Record(json::compact(&outcome)),
            Err(field) => Verdict::Refuse(correction(field)),
        },
    })
}

/// The reply the person reads beside the card: empty, or text that says something (`_reply`).
fn reply(value: &Value) -> Option<&str> {
    let Value::String(reply) = value else {
        return None;
    };
    if reply.is_empty() {
        return Some(reply);
    }
    (REPLY_CONTROL_RE.is_absent(reply)
        && !text::strip(reply).is_empty()
        && text::len(reply) <= MAX_REPLY_CHARS)
        .then_some(reply.as_str())
}

/// The required members of the offered schema.
fn required(refinable: bool) -> Vec<&'static str> {
    let mut members = vec!["op", "name", "description", "replaces"];
    if refinable {
        members.extend(["routine_id", "base_revision", "edits"]);
    }
    members.push("reply");
    members
}

/// The recorded outcome of one valid call, or the field whose correction answers it (`record`). `handles` are the
/// call handles of this turn's Action results, which a refinement may name.
fn record(
    arguments: &Value,
    context: &TurnContext,
    handles: &BTreeSet<String>,
) -> Result<Value, &'static str> {
    let selected = context.routine_selected.as_ref();
    if !json::has_exactly(arguments, &required(selected.is_some())) {
        // Beside a selection, the correction names both operations and their members.
        return Err(if selected.is_some() {
            "refine"
        } else {
            "invalid"
        });
    }
    let field = |name: &str| arguments.get(name).unwrap_or(&Value::Null);
    let op = json::str_of(arguments, "op");
    if let Some(selected) = selected {
        if op == Some("refine") {
            return refine::refined(arguments, selected, handles);
        }
        if ["routine_id", "base_revision", "edits"]
            .iter()
            .any(|name| *field(name) != Value::Null)
        {
            return Err("refine");
        }
    }
    if op != Some("record") {
        return Err("invalid");
    }
    recorded_outcome(arguments, context)
}

/// A record call's outcome, or its first invalid field.
fn recorded_outcome(arguments: &Value, context: &TurnContext) -> Result<Value, &'static str> {
    let field = |name: &str| arguments.get(name).unwrap_or(&Value::Null);
    let name = canonical_name(field("name")).ok_or("name")?;
    let description = field("description");
    canonical_description(description).ok_or("description")?;
    let replaces = field("replaces");
    let listed = context
        .routines
        .iter()
        .flatten()
        .any(|item| item.get("routine_id").is_some_and(|id| py_eq(id, replaces)));
    if *replaces != Value::Null && !listed {
        return Err("replaces");
    }
    let reply = reply(field("reply")).ok_or("reply")?;
    let routine = json::object([
        ("name", json::text(name)),
        ("description", description.clone()),
        ("replaces", replaces.clone()),
    ]);
    Ok(json::object([
        ("routine", routine),
        ("reply", json::text(reply)),
    ]))
}

/// An Action result as the model read it: its JSON value, or its text when it is not JSON (`_result`).
fn result_value(content: &str) -> Value {
    json::parse(content.as_bytes()).unwrap_or_else(|_| json::text(content))
}

/// Where the current turn starts: its last user message, else the first message.
fn turn_start(messages: &[Message]) -> usize {
    messages
        .iter()
        .rposition(|message| matches!(message, Message::User { .. }))
        .unwrap_or(0)
}

/// The Team call handles the turn's Action results carry (`call_handles`): each traced call's result came as
/// `{"call", "result"}`.
#[must_use]
pub fn call_handles(messages: &[Message]) -> BTreeSet<String> {
    messages[turn_start(messages)..]
        .iter()
        .filter_map(Message::tool)
        .filter_map(|message| {
            let value = result_value(&message.content);
            if !json::has_exactly(&value, &["call", "result"]) {
                return None;
            }
            json::str_of(&value, "call")
                .filter(|handle| CALL_HANDLE_RE.is_full(handle))
                .map(str::to_owned)
        })
        .collect()
}

/// The reply and record of a Routine that ended the turn (`recorded`).
#[must_use]
pub fn recorded(messages: &[Message]) -> Option<(String, Value)> {
    let message = messages.last()?.tool()?;
    if message.name != TOOL_NAME {
        return None;
    }
    let value = json::parse(message.content.as_bytes()).ok()?;
    json::has_exactly(&value, &["routine", "reply"]).then_some(())?;
    Some((
        json::str_of(&value, "reply")?.to_owned(),
        value.get("routine")?.clone(),
    ))
}

#[cfg(test)]
mod tests;

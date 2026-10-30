//! The rerun guard (`rerun_guard`): a response sending a rerun's 'fresh' input that no result of this turn holds is
//! refused, so the agent never sends a remembered value Team would only ask about again. Team's recorder stays the
//! authority.

use std::collections::{BTreeMap, BTreeSet};

use team_protocol::payload::authored_segments;
use team_protocol::python::text;
use team_protocol::python::value::py_eq;
use team_protocol::routine_context::{
    MIN_IDENTIFIER_CHARS, MIN_IDENTIFIER_DIGITS, NUMBER_TOKEN_RE, may_write_date,
};

use super::{result_value, turn_start};
use crate::conversation::Message;
use crate::errors::BrainError;
use crate::json::{self, Number, Value};
use crate::tools::Review;
use crate::tools::action::tool_name;
use crate::tools::memory::turn_message;
use crate::tools::refusal::UNPARSABLE;

/// How a rerun call that sends a 'fresh' input no result of this turn holds is answered: nothing ran.
const RERUN_CORRECTION: &str = "Not done: no Action in this response ran. The Team needs each 'fresh' input of its \
    rerun work to come from a result in this turn: first run the listed source Action, or an Action that returns \
    the value, then send the value from that result; never send a value you remember.";

/// The members each Action tool must take from this turn's results: those every slot of the Action marks fresh.
fn checked(work: &[Value]) -> BTreeMap<String, BTreeSet<String>> {
    let mut slots: BTreeMap<String, Vec<BTreeSet<String>>> = BTreeMap::new();
    for entry in work {
        let fresh = json::array_of(entry, "inputs")
            .unwrap_or_default()
            .iter()
            .filter(|item| json::str_of(item, "kind") == Some("fresh"))
            .filter_map(|item| json::str_of(item, "member").map(str::to_owned))
            .collect();
        let name = tool_name(
            json::str_of(entry, "assistant").unwrap_or_default(),
            json::str_of(entry, "action").unwrap_or_default(),
        );
        slots.entry(name).or_default().push(fresh);
    }
    slots
        .into_iter()
        .filter_map(|(name, sets)| {
            let mut sets = sets.into_iter();
            let first = sets.next()?;
            let members: BTreeSet<String> = sets.fold(first, |kept, next| {
                kept.intersection(&next).cloned().collect()
            });
            (!members.is_empty()).then_some((name, members))
        })
        .collect()
}

/// The value and every value inside it (`_values`).
fn values(value: &Value, found: &mut Vec<Value>) {
    found.push(value.clone());
    match value {
        Value::Object(members) => members.iter().for_each(|(_, item)| values(item, found)),
        Value::Array(items) => items.iter().for_each(|item| values(item, found)),
        _ => {}
    }
}

/// An integer's decimal text, or `None` for any other value.
fn integer(value: &Value) -> Option<String> {
    match value {
        Value::Number(Number::Int(number)) => Some(number.to_string()),
        Value::Number(Number::BigInt(digits)) => Some(digits.clone()),
        _ => None,
    }
}

/// Whether Team would ask where a value no result holds came from: one word or whole number at least as long as
/// Team's identifier bounds, unless the word writes a date Team may read as each run's own (`_identifier`).
fn identifier(value: &Value, today: (i32, i32, i32)) -> bool {
    if let Some(digits) = integer(value) {
        return digits.trim_start_matches('-').len() >= MIN_IDENTIFIER_DIGITS;
    }
    let Value::String(word) = value else {
        return false;
    };
    text::len(word) >= MIN_IDENTIFIER_CHARS
        && !word.chars().any(text::isspace)
        && !may_write_date(word, today).unwrap_or(true)
}

/// Whether the person's current message writes this identifier: a text within it, a number as a whole token
/// (`_written`).
fn written(value: &Value, segments: &[String]) -> bool {
    if let Value::String(word) = value {
        return segments
            .iter()
            .any(|segment| segment.contains(word.as_str()));
    }
    let Some(digits) = integer(value) else {
        return false;
    };
    segments.iter().any(|segment| {
        NUMBER_TOKEN_RE.finditer(segment).is_ok_and(|tokens| {
            tokens
                .iter()
                .any(|token| segment[token.start..token.end] == digits)
        })
    })
}

/// The rerun guard (`rerun_guard`): `actions` are the turn's Action tool names. A member is checked only where every
/// slot of its Action marks it fresh, and only an identifier-shaped value; it passes when it equals any value inside an
/// Action result of this turn or the person's current message writes it.
///
/// # Errors
///
/// An unanswerable call.
pub fn rerun_review(messages: &[Message], work: &[Value], actions: &BTreeSet<String>) -> Review {
    let Some(latest) = messages.last().and_then(Message::assistant) else {
        return Ok(None);
    };
    let checked = checked(work);
    let calls: Vec<_> = latest
        .calls
        .iter()
        .filter(|call| checked.contains_key(&call.name))
        .collect();
    if calls.is_empty() {
        return Ok(None);
    }
    if !latest.invalid_calls.is_empty() {
        // A checked call beside an unparsable one cannot be judged or answered whole.
        return Err(BrainError::contract(UNPARSABLE));
    }
    let turn = &messages[turn_start(messages)..];
    let mut results = Vec::new();
    for message in turn.iter().filter_map(Message::tool) {
        if actions.contains(&message.name) {
            values(&result_value(&message.content), &mut results);
        }
    }
    let current = match turn.first() {
        Some(Message::User { text, .. }) => turn_message(text).unwrap_or_default(),
        _ => String::new(),
    };
    let segments = authored_segments(&current);
    let today = crate::prompt::utc_date();
    let refused = calls.iter().any(|call| {
        checked[&call.name].iter().any(|member| {
            call.arguments.get(member).is_some_and(|value| {
                identifier(value, today)
                    && !results.iter().any(|node| py_eq(value, node))
                    && !written(value, &segments)
            })
        })
    });
    Ok(refused.then(|| RERUN_CORRECTION.to_owned()))
}

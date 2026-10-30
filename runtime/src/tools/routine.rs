//! Routines recorded from work the chat agent actually ran (`routine.py`, ADR-0101, ADR-0122).

use std::collections::BTreeSet;

use crate::conversation::Message;
use crate::errors::BrainError;
use crate::json::Value;
use crate::provider::ToolSpec;
use crate::tools::Review;
use crate::turn::context::TurnContext;

/// The tool's name.
pub const TOOL_NAME: &str = "shimpz_routine";

/// Canonicalizes the turn's Routines and recording fields in place (`admit_context`).
///
/// # Errors
///
/// [`BrainError::Contract`] with the Python texts ("invalid routines", "invalid Routine mode", ...).
pub fn admit_context(_context: &mut TurnContext) -> Result<(), BrainError> {
    unimplemented!("ported by the tools microtask")
}

/// Whether Team's pending question asks how often the Routine runs or what each run does with its result.
#[must_use]
pub fn settings_question(_question: Option<&Value>) -> bool {
    unimplemented!("ported by the tools microtask")
}

/// The tool, with `refine` while the send selects a Routine.
#[must_use]
pub fn spec(_refinable: bool) -> ToolSpec {
    unimplemented!("ported by the tools microtask")
}

/// The Routine guard (`_review`).
///
/// # Errors
///
/// An unanswerable call.
pub fn review(_messages: &[Message], _context: &TurnContext) -> Review {
    unimplemented!("ported by the tools microtask")
}

/// What the tool returns when it runs on `arguments`.
#[must_use]
pub fn run(_arguments: &Value, _context: &TurnContext, _handles: &BTreeSet<String>) -> String {
    unimplemented!("ported by the tools microtask")
}

/// The Team call handles the turn's Action results carry (`call_handles`).
#[must_use]
pub fn call_handles(_messages: &[Message]) -> BTreeSet<String> {
    unimplemented!("ported by the tools microtask")
}

/// The reply and record of a Routine that ended the turn (`recorded`).
#[must_use]
pub fn recorded(_messages: &[Message]) -> Option<(String, Value)> {
    unimplemented!("ported by the tools microtask")
}

/// The rerun guard (`rerun_guard`): `actions` are the turn's Action tool names.
///
/// # Errors
///
/// An unanswerable call.
pub fn rerun_review(_messages: &[Message], _work: &[Value], _actions: &BTreeSet<String>) -> Review {
    unimplemented!("ported by the tools microtask")
}

//! The conversation a chat turn carries to its model (`context_budget.py`): the newest whole exchanges within the
//! history budget, older Action results omitted a batch at a time, and an estimated model-window guard.

use crate::conversation::{Message, ToolMessage};
use crate::errors::BrainError;
use crate::json::Value;

/// The id prefix of the one user-role message quoting committed presentation history (ADR-0065).
pub const BRIDGE_ID_PREFIX: &str = "shimpz-context-";

/// Why history cannot be fitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// The fixed prompt, tools, and current exchange alone exceed the window: a contract error.
    Window,
    /// Stored history cannot be decomposed into whole exchanges: a state error.
    State,
}

impl From<FitError> for BrainError {
    fn from(error: FitError) -> Self {
        match error {
            FitError::Window => Self::contract("conversation context exceeds the model window"),
            FitError::State => Self::State,
        }
    }
}

/// The estimated tokens of a JSON value (`estimated_tokens`).
#[must_use]
pub fn estimated_tokens(_value: &Value) -> u64 {
    unimplemented!("ported by the attachments microtask")
}

/// The estimated tokens of a message (`message_tokens`).
#[must_use]
pub fn message_tokens(_message: &Message) -> u64 {
    unimplemented!("ported by the attachments microtask")
}

/// The fixed tokens of a system prompt and its tools, each `{"name", "summary", "schema"}` (`fixed_tokens`).
#[must_use]
pub fn fixed_tokens(_system: &str, _tools: &[Value]) -> u64 {
    unimplemented!("ported by the attachments microtask")
}

/// The reply a model turn ends a turn with, or "" when it cannot end one (`final_reply`).
#[must_use]
pub fn final_reply(_message: &Message) -> String {
    unimplemented!("ported by the attachments microtask")
}

/// Refuses a call whose fixed and conversation tokens exceed the window (`ensure_window`).
///
/// # Errors
///
/// [`FitError::Window`].
pub fn ensure_window(_fixed: u64, _conversation: u64) -> Result<(), FitError> {
    unimplemented!("ported by the attachments microtask")
}

/// The ids of what a new turn must forget (`history_to_drop`).
///
/// # Errors
///
/// [`FitError`].
pub fn history_to_drop(
    _history: &[Message],
    _fixed: u64,
    _current: u64,
    _forgotten: &dyn Fn(&Message) -> bool,
) -> Result<Vec<String>, FitError> {
    unimplemented!("ported by the attachments microtask")
}

/// Placeholders for the older Action results a new turn omits, or none below the trigger (`results_to_omit`).
///
/// # Errors
///
/// [`FitError::State`].
pub fn results_to_omit(
    _history: &[Message],
    _action_result: &dyn Fn(&ToolMessage) -> bool,
    _forgotten: &dyn Fn(&Message) -> bool,
) -> Result<Vec<ToolMessage>, FitError> {
    unimplemented!("ported by the attachments microtask")
}

//! The one way a guard refuses a model response (`tool_refusal.py`): every call of the refused response receives the
//! same closed correction, so each call stays paired with a result, nothing executes, and the model runs again.

use crate::conversation::{Message, ToolCall};
use crate::errors::BrainError;

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
pub fn lone_call<'a>(_messages: &'a [Message], _tool: &str) -> Result<LoneCall<'a>, BrainError> {
    unimplemented!("ported by the tools microtask")
}

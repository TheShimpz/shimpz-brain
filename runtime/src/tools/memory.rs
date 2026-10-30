//! Learned Team memory (`memory.py`, ADR-0084): only the person's own words change it, and an independent check
//! confirms each proposed change before Team saves it.

use crate::conversation::Message;
use crate::json::Value;
use crate::provider::ToolSpec;
use crate::tools::Review;
use crate::turn::context::Memory;

/// The tool's name.
pub const TOOL_NAME: &str = "shimpz_memory";

/// One proposed change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    /// `remember` or `forget`.
    pub op: String,
    /// The topic.
    pub topic: String,
    /// The preference (empty for `forget`).
    pub preference: String,
}

impl Change {
    /// `{"op", "topic", "preference"}`.
    #[must_use]
    pub fn to_value(&self) -> Value {
        unimplemented!("ported by the tools microtask")
    }
}

/// The exact memory list Team sent (`canonical`); `None` when invalid.
#[must_use]
pub fn canonical(_value: &Value) -> Option<Vec<Memory>> {
    unimplemented!("ported by the tools microtask")
}

/// The tool.
#[must_use]
pub fn spec() -> ToolSpec {
    unimplemented!("ported by the tools microtask")
}

/// What the tool returns when it runs (the turn continues).
#[must_use]
pub fn proposed_text() -> &'static str {
    unimplemented!("ported by the tools microtask")
}

/// The memory guard (`_review`).
///
/// # Errors
///
/// Never in practice; the shared guard shape.
pub fn review(_messages: &[Message], _allowed: bool) -> Review {
    unimplemented!("ported by the tools microtask")
}

/// The independent check's prompt for the logical turn's proposals, and those proposals; `None` when it has none.
#[must_use]
pub fn confirmation(
    _messages: &[Message],
    _memories: Option<&[Memory]>,
) -> Option<(String, Vec<Change>)> {
    unimplemented!("ported by the tools microtask")
}

/// The structured output schema of the check (`Confirmation`).
#[must_use]
pub fn confirmation_schema() -> Value {
    unimplemented!("ported by the tools microtask")
}

/// The most characters the check's answer may have.
pub const MAX_CONFIRMATION_CHARS: usize = 4_096;

/// The proposals the check's `verdict` confirms; no verdict, or any doubt, keeps nothing (`accepted`).
#[must_use]
pub fn accepted(_changes: Vec<Change>, _verdict: Option<&Value>) -> Vec<Change> {
    unimplemented!("ported by the tools microtask")
}

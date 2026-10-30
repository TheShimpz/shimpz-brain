//! The chat agent's one `browser` tool (`browser_tool.py`, ADR-0117 §5 and §6).

use crate::conversation::Message;
use crate::json::Value;
use crate::provider::ToolSpec;
use crate::tools::Review;
use crate::turn::context::TurnContext;

/// The tool's name.
pub const TOOL_NAME: &str = "browser";

/// Whether a turn offers the Browser.
#[must_use]
pub fn offered(_context: &TurnContext) -> bool {
    unimplemented!("ported by the tools microtask")
}

/// The tool.
#[must_use]
pub fn spec() -> ToolSpec {
    unimplemented!("ported by the tools microtask")
}

/// The canonical objective of a call, or `Err(correction)`.
///
/// # Errors
///
/// The closed correction.
pub fn objective(_arguments: &Value) -> Result<String, &'static str> {
    unimplemented!("ported by the tools microtask")
}

/// The guard refusing a browser call beside any other call.
///
/// # Errors
///
/// An unanswerable call.
pub fn review_alone(_messages: &[Message]) -> Review {
    unimplemented!("ported by the tools microtask")
}

/// The guard of the pass after a browser result: any call is refused.
///
/// # Errors
///
/// Never.
pub fn review_after(_messages: &[Message]) -> Review {
    unimplemented!("ported by the tools microtask")
}

/// Whether a message is a browser call or its result (its exchange is forgotten by the next turn).
#[must_use]
pub fn read(_message: &Message) -> bool {
    unimplemented!("ported by the tools microtask")
}

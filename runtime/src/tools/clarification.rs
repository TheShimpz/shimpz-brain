//! Brain-authored multiple-choice clarification (`clarification.py`, ADR-0081).

use crate::conversation::Message;
use crate::json::Value;
use crate::provider::ToolSpec;
use crate::tools::Review;

/// The tool's name.
pub const TOOL_NAME: &str = "shimpz_clarify";

/// One option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClarifyOption {
    /// Its label.
    pub label: String,
    /// Its description.
    pub description: String,
}

/// One closed clarification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clarification {
    /// The question.
    pub question: String,
    /// Two to five options.
    pub options: Vec<ClarifyOption>,
    /// The recommended option, or none (a Routine turn's).
    pub default_index: Option<usize>,
}

impl Clarification {
    /// The closed clarification in `arguments`, or `None`.
    #[must_use]
    pub fn parse(_arguments: &Value) -> Option<Self> {
        unimplemented!("ported by the tools microtask")
    }

    /// The plain reply shown and remembered (`payload.render_clarification`).
    #[must_use]
    pub fn render(&self) -> String {
        unimplemented!("ported by the tools microtask")
    }

    /// `{"question", "options": [{"label", "description"}], "default_index"}`.
    #[must_use]
    pub fn to_value(&self) -> Value {
        unimplemented!("ported by the tools microtask")
    }
}

/// The tool.
#[must_use]
pub fn spec() -> ToolSpec {
    unimplemented!("ported by the tools microtask")
}

/// What the tool returns when it runs (it ends the turn).
pub const RECORDED: &str =
    "Question recorded; it is shown to the user with a free-text option and ends this turn.";

/// The clarification guard (`_review`).
///
/// # Errors
///
/// An unanswerable call.
pub fn review(
    _messages: &[Message],
    _allowed: bool,
    _routine_mode: bool,
    _locale: Option<&str>,
) -> Review {
    unimplemented!("ported by the tools microtask")
}

/// The clarification that ended the turn, when its last messages are the tool's call and result (`recorded`).
#[must_use]
pub fn recorded(_messages: &[Message]) -> Option<Clarification> {
    unimplemented!("ported by the tools microtask")
}

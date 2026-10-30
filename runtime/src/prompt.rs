//! The system policy prompt of one Team chat turn (`turn_prompt.py`) and the interface languages
//! (`interface_language.py`).

use crate::turn::context::TurnContext;

/// The turn's policy prompt, built from its surface: `context.assistants` are the Assistants it is offered and
/// `context.browser` whether it offers the Browser.
#[must_use]
pub fn system_prompt(_context: &TurnContext) -> String {
    unimplemented!("ported by the tools microtask")
}

/// Today's UTC date (`YYYY-MM-DD`).
#[must_use]
pub fn today() -> String {
    unimplemented!("ported by the tools microtask")
}

/// Whether `locale` is one closed interface language.
#[must_use]
pub fn valid_locale(_locale: &str) -> bool {
    unimplemented!("ported by the tools microtask")
}

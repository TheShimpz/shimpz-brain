//! Provider-native structured output and its closed re-validation, shared by every structured decision
//! (`structured.py`).

use crate::conversation::Block;
use crate::json::Value;
use crate::provider::{CallError, Profile, ProviderConfig, Providers, Scope};

/// One stateless structured question: an optional system prompt and one user message, answered with one JSON value.
#[derive(Debug, Clone, PartialEq)]
pub struct Ask<'a> {
    /// The system prompt, if any.
    pub system: Option<&'a str>,
    /// The user message's blocks.
    pub user: Vec<Block>,
    /// The schema's name.
    pub name: &'a str,
    /// The JSON Schema of the answer.
    pub schema: &'a Value,
    /// The most characters the answer's text may have.
    pub max_chars: usize,
}

/// One stateless structured call of `ask` on `config`'s model.
///
/// # Errors
///
/// [`CallError::Request`] when the call failed, [`CallError::Response`] when the answer is not exactly one value
/// (a refusal, a tool call, text that is not one strict JSON object, or text blocks that disagree).
pub fn value(
    _providers: &Providers,
    _scope: &Scope,
    _config: &ProviderConfig,
    _profile: Profile,
    _ask: &Ask<'_>,
) -> Result<Value, CallError> {
    unimplemented!("ported by the provider adapter microtask")
}

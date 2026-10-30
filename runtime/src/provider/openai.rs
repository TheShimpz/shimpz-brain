//! The OpenAI Responses API adapter.

use crate::conversation::{AssistantMessage, Block};
use crate::provider::{ChatRequest, Profile, Projection, Tokens, Wire};

/// The `/v1/responses` request for one call.
#[must_use]
pub fn request(
    _model: &str,
    _request: &ChatRequest,
    _projection: Option<&Projection>,
    _profile: Profile,
) -> Wire {
    unimplemented!("ported by the provider adapter microtask")
}

/// The model turn and reported tokens of a `/v1/responses` answer; `None` when it is not one.
#[must_use]
pub fn response(_body: &[u8]) -> Option<(AssistantMessage, Option<Tokens>)> {
    unimplemented!("ported by the provider adapter microtask")
}

/// The `/v1/responses/input_tokens` request counting one user message made of `blocks`.
#[must_use]
pub fn count_request(_model: &str, _blocks: &[Block]) -> Wire {
    unimplemented!("ported by the provider adapter microtask")
}

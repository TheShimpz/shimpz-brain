//! Why a pending Action pauses for a person (`action_purpose.py`, ADR-0090).

use crate::errors::BrainError;
use crate::json::Value;
use crate::provider::{Providers, Scope};
use crate::server::input::RouteError;

/// The pending Action a purpose is written for, and its turn's own start message.
#[derive(Debug, Clone, PartialEq)]
pub struct Pending {
    /// The text of the turn's start message: Team's `{"files", "message"}` envelope.
    pub message: String,
    /// The interface language the turn's start pinned.
    pub locale: Option<String>,
    /// The Action's Assistant.
    pub assistant_id: String,
    /// The Action.
    pub action: String,
    /// The Action's input.
    pub input: Value,
}

/// Looks up the pending Action request `interrupt_id` of `thread_id`; `None` when the thread has no such request.
pub type Lookup<'a> = &'a (dyn Fn(&str, &str) -> Result<Option<Pending>, BrainError> + Sync);

/// `POST /v1/turns/purpose`: `{"purpose": string | null}`.
///
/// # Errors
///
/// [`RouteError`]: 422 for the request shape, 400/502/503, and 409 "Action purpose cancelled".
pub fn handle(
    _providers: &Providers,
    _scope: &Scope,
    _body: &Value,
    _lookup: Lookup<'_>,
) -> Result<Value, RouteError> {
    unimplemented!("ported by the decisions microtask")
}

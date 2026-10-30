//! The Browser agent's stateless calls (`browser_step.py`, `browser_thread.py`, `browser_api.py`, ADR-0117 §5).

use crate::json::Value;
use crate::provider::{Providers, Scope};
use crate::server::input::RouteError;

/// The four Browser operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// `POST /v1/browser-step`.
    Step,
    /// `POST /v1/browser-fold`.
    Fold,
    /// `POST /v1/browser-check`.
    Check,
    /// `POST /v1/browser-final`.
    Final,
}

/// One Browser operation: its answer's fields and `model` (the server adds `usage`).
///
/// # Errors
///
/// [`RouteError`]: 422 for the request shape, 400/502, and 409 "Browser step cancelled".
pub fn handle(
    _providers: &Providers,
    _scope: &Scope,
    _operation: Operation,
    _body: &Value,
) -> Result<Value, RouteError> {
    unimplemented!("ported by the decisions microtask")
}

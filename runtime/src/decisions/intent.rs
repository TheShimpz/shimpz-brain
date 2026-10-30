//! Intent routing (`intent_route.py`): classify one fresh Assistant lifecycle intent, or resolve it against a closed
//! candidate directory, after the Jev fast path when the request carries a decision key.

use crate::json::Value;
use crate::provider::{Providers, Scope};
use crate::server::input::RouteError;

/// `POST /v1/intent-route`: `{"intent", "query", "assistant_ids", "reply", "task_follows"[, "capability"]}`.
///
/// # Errors
///
/// [`RouteError`]: 422 for the request shape, 400/502, and 409 "Intent route cancelled".
pub fn handle(_providers: &Providers, _scope: &Scope, _body: &Value) -> Result<Value, RouteError> {
    unimplemented!("ported by the decisions microtask")
}

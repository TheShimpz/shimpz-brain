//! Capability planning (`capability_plan.py`): over one closed public Assistant shortlist, whether an objective needs
//! any of them installed.

use crate::json::Value;
use crate::provider::{Providers, Scope};
use crate::server::input::RouteError;

/// `POST /v1/capability-plan`: `{"status", "assistant_ids"}`.
///
/// # Errors
///
/// [`RouteError`]: 422 for the request shape, 400/502/409 ("Capability plan cancelled") as the Python runtime.
pub fn handle(_providers: &Providers, _scope: &Scope, _body: &Value) -> Result<Value, RouteError> {
    unimplemented!("ported by the decisions microtask")
}

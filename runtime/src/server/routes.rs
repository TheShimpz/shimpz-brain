//! The routes Team calls and the work each runs: on a blocking thread that owns the request's reservation and lane
//! slot until the work settled, with the request's cancellation tripped when Team closes the connection.

use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::cancel::{CancelOnDrop, CancelToken};
use crate::decisions::{self, browser::Operation};
use crate::errors::BrainError;
use crate::json::{self, Value};
use crate::provider::Scope;
use crate::server::App;
use crate::server::boundary::Reservation;
use crate::server::input::{self, RouteError};
use crate::server::turn_input;

/// One route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// `GET /health`.
    Health,
    /// `POST /v1/turns`.
    Start,
    /// `POST /v1/turns/resume`.
    Resume,
    /// `POST /v1/turns/purpose`.
    Purpose,
    /// `POST /v1/capability-plan`.
    CapabilityPlan,
    /// `POST /v1/intent-route`.
    IntentRoute,
    /// One of the four Browser operations.
    Browser(Operation),
    /// `POST /v1/threads/delete`.
    DeleteThread,
}

/// The route of `path`, or `None` for an unknown path; the method is checked by the caller.
#[must_use]
pub fn resolve(path: &str) -> Option<Route> {
    Some(match path {
        "/health" => Route::Health,
        "/v1/turns" => Route::Start,
        "/v1/turns/resume" => Route::Resume,
        "/v1/turns/purpose" => Route::Purpose,
        "/v1/capability-plan" => Route::CapabilityPlan,
        "/v1/intent-route" => Route::IntentRoute,
        "/v1/browser-step" => Route::Browser(Operation::Step),
        "/v1/browser-fold" => Route::Browser(Operation::Fold),
        "/v1/browser-check" => Route::Browser(Operation::Check),
        "/v1/browser-final" => Route::Browser(Operation::Final),
        "/v1/threads/delete" => Route::DeleteThread,
        _ => return None,
    })
}

/// The capacity lanes of the stateless decisions.
#[derive(Debug)]
pub struct Lanes {
    capability: Arc<Semaphore>,
    intent: Arc<Semaphore>,
    browser: Arc<Semaphore>,
}

impl Default for Lanes {
    fn default() -> Self {
        Self {
            capability: Arc::new(Semaphore::new(2)),
            intent: Arc::new(Semaphore::new(2)),
            browser: Arc::new(Semaphore::new(2)),
        }
    }
}

fn slot(lane: &Arc<Semaphore>, full: &'static str) -> Result<OwnedSemaphorePermit, RouteError> {
    Arc::clone(lane)
        .try_acquire_owned()
        .map_err(|_| RouteError::Brain(BrainError::Capacity(full)))
}

/// What a route's work needs once admitted.
pub struct Admitted {
    /// The parsed body.
    pub body: Value,
    /// The request's memory reservation, released when the work settled.
    pub reservation: Option<Reservation>,
}

fn with_usage(value: Value, scope: &Scope) -> Value {
    let Value::Object(mut members) = value else {
        return value;
    };
    members.push(("usage".to_owned(), scope.usage.to_value()));
    Value::Object(members)
}

/// Runs `work` on a blocking thread; dropping the returned future (Team closed the connection) cancels it, and the
/// work keeps what it owns until it settles.
async fn blocking<F>(cancel: CancelToken, work: F) -> Result<Value, RouteError>
where
    F: FnOnce() -> Result<Value, RouteError> + Send + 'static,
{
    let mut guard = CancelOnDrop::new(cancel);
    let joined = tokio::task::spawn_blocking(work).await;
    guard.disarm();
    joined.unwrap_or(Err(RouteError::Brain(BrainError::State)))
}

/// Runs an admitted POST route.
///
/// # Errors
///
/// [`RouteError`].
pub async fn run(app: &Arc<App>, route: Route, admitted: Admitted) -> Result<Value, RouteError> {
    let cancel = CancelToken::new();
    let scope = Scope::new(cancel.clone());
    let app = Arc::clone(app);
    let permit = match route {
        Route::CapabilityPlan => Some(slot(
            &app.lanes.capability,
            "Capability planner capacity reached",
        )?),
        Route::IntentRoute => Some(slot(&app.lanes.intent, "Intent route capacity reached")?),
        Route::Browser(_) => Some(slot(&app.lanes.browser, "Browser step capacity reached")?),
        _ => None,
    };
    blocking(cancel, move || {
        let _held = (permit, admitted.reservation);
        let body = admitted.body;
        let answered = dispatch(&app, route, &body, &scope)?;
        Ok(if route == Route::DeleteThread {
            answered
        } else {
            with_usage(answered, &scope)
        })
    })
    .await
}

fn dispatch(app: &App, route: Route, body: &Value, scope: &Scope) -> Result<Value, RouteError> {
    let providers = app.service.providers();
    match route {
        Route::Start => Ok(app.service.start(turn_input::start(body)?, scope)?),
        Route::Resume => Ok(app.service.resume(&turn_input::resume(body)?, scope)?),
        Route::Purpose => {
            let lookup =
                |thread: &str, interrupt: &str| app.service.pending_action(thread, interrupt);
            decisions::purpose::handle(providers, scope, body, &lookup)
        }
        Route::CapabilityPlan => decisions::capability::handle(providers, scope, body),
        Route::IntentRoute => decisions::intent::handle(providers, scope, body),
        Route::Browser(operation) => decisions::browser::handle(providers, scope, operation, body),
        Route::DeleteThread => delete_thread(app, body),
        Route::Health => Err(RouteError::Invalid),
    }
}

fn delete_thread(app: &App, body: &Value) -> Result<Value, RouteError> {
    input::closed(body, &["thread_id"], &["thread_id"])?;
    let thread_id = input::string(body, "thread_id", 1, 256)?;
    if !team_protocol::turn::SAFE_ID_RE.is_full(&thread_id) {
        return Err(RouteError::Invalid);
    }
    app.service.delete_thread(&thread_id)?;
    Ok(json::object([("status", json::text("deleted"))]))
}

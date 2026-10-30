//! The HTTP boundary Team calls: one raw hyper dispatcher (as Store's), the bounded admission of every request, and
//! the routes of `runtime_api.py` with their exact statuses and bodies.

pub mod boundary;
pub mod input;
pub mod routes;
pub mod turn_input;

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use http::{Method, Request, Response};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::task::JoinSet;

use crate::json;
use crate::turn::service::TurnService;
use boundary::{Body, BodyError, MemoryBudget};
use routes::{Admitted, Lanes, Route};

/// How long a connection may take to send its request head.
pub const HEADER_SECONDS: Duration = Duration::from_secs(10);
/// How long in-flight requests get to finish once the process is told to stop.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// The serving process's shared state.
pub struct App {
    /// Team's bearer file.
    pub token_file: PathBuf,
    /// The chat turns (and the providers).
    pub service: Arc<TurnService>,
    /// The decisions' capacity lanes.
    pub lanes: Lanes,
    /// The request memory budget.
    pub memory: MemoryBudget,
}

impl std::fmt::Debug for App {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("App").finish_non_exhaustive()
    }
}

/// Answers one request.
///
/// # Errors
///
/// Never: every failure is an HTTP answer.
pub async fn handle(
    app: Arc<App>,
    request: Request<Incoming>,
) -> Result<Response<Body>, Infallible> {
    let (parts, body) = request.into_parts();
    let size = boundary::reservation(&parts.headers);
    let mut reservation = None;
    let mut received = bytes::Bytes::new();
    if size > 0 {
        if let Err(refused) = boundary::authenticate(&parts.headers, &app.token_file) {
            return Ok(refused.response(true));
        }
        if size > boundary::MAX_REQUEST_BYTES {
            return Ok(boundary::detail(413, "Request body is too large", true));
        }
        let Some(admitted) = app.memory.reserve(size) else {
            let mut response =
                boundary::detail(503, "Brain runtime request capacity reached", true);
            response
                .headers_mut()
                .insert("retry-after", http::HeaderValue::from_static("1"));
            return Ok(response);
        };
        reservation = Some(admitted);
        received = match boundary::receive(body, size).await {
            Ok(bytes) => bytes,
            Err(BodyError::TooLarge) => {
                return Ok(boundary::detail(413, "Request body is too large", true));
            }
            Err(BodyError::Late) => {
                return Ok(boundary::detail(
                    408,
                    "Request body was not received in time",
                    true,
                ));
            }
            Err(BodyError::Gone) => {
                return Ok(boundary::detail(400, "Request body was not received", true));
            }
        };
    }
    Ok(route(&app, &parts, received, reservation).await)
}

async fn route(
    app: &Arc<App>,
    parts: &http::request::Parts,
    received: bytes::Bytes,
    reservation: Option<boundary::Reservation>,
) -> Response<Body> {
    let Some(route) = routes::resolve(parts.uri.path()) else {
        return boundary::detail(404, "Not Found", false);
    };
    let expected = if route == Route::Health {
        Method::GET
    } else {
        Method::POST
    };
    if parts.method != expected {
        return boundary::detail(405, "Method Not Allowed", false);
    }
    if route == Route::Health {
        return boundary::respond(
            200,
            &json::object([
                ("status", json::text("ok")),
                ("runtime", json::text("holdpoint")),
            ]),
            false,
        );
    }
    if let Err(refused) = boundary::authenticate(&parts.headers, &app.token_file) {
        return refused.response(false);
    }
    let Ok(body) = json::parse(&received) else {
        return boundary::refused(&input::RouteError::Invalid);
    };
    match routes::run(app, route, Admitted { body, reservation }).await {
        Ok(value) => boundary::respond(200, &value, false),
        Err(error) => boundary::refused(&error),
    }
}

/// Serves `listener` until `shutdown` resolves, then gives in-flight requests [`SHUTDOWN_GRACE`].
pub async fn serve(
    listener: TcpListener,
    app: Arc<App>,
    shutdown: impl std::future::Future<Output = ()>,
) {
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            () = &mut shutdown => break,
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else { continue };
                let app = Arc::clone(&app);
                connections.spawn(async move {
                    let service = service_fn(move |request| handle(Arc::clone(&app), request));
                    let mut builder = http1::Builder::new();
                    builder.timer(hyper_util::rt::TokioTimer::new()).header_read_timeout(HEADER_SECONDS).max_headers(100);
                    let _ = builder.serve_connection(TokioIo::new(stream), service).await;
                });
            }
            Some(_) = connections.join_next(), if !connections.is_empty() => {}
        }
    }
    drop(listener);
    let _ = tokio::time::timeout(SHUTDOWN_GRACE, async {
        while connections.join_next().await.is_some() {}
    })
    .await;
}

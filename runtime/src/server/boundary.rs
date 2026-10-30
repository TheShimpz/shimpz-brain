//! The bounded HTTP boundary (`runtime_api.BoundedBody` and the bearer dependency): every request with a body is
//! authenticated before any byte of it is read, bounded by its declared size, admitted against the process's request
//! memory budget, and received whole within its deadline; a bodyless request is routed first.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use http::{HeaderMap, Response, StatusCode, header};
use http_body_util::{BodyExt, Full, Limited};
use subtle::ConstantTimeEq;

use crate::json::{self, Value};
use crate::server::input::RouteError;

/// The largest token file read.
pub const MAX_TOKEN_BYTES: usize = 4 * 1024;
/// The largest request body.
pub const MAX_REQUEST_BYTES: u64 = 4 * 1024 * 1024;
/// How long a whole body may take to arrive.
pub const REQUEST_BODY_SECONDS: Duration = Duration::from_secs(30);
/// The memory every admitted request with a body reserves beside its body.
pub const MEMORY_PER_REQUEST: u64 = 48 * 1024 * 1024;
/// The memory each body byte reserves.
pub const MEMORY_PER_BODY_BYTE: u64 = 128;
/// The request memory budget.
pub const MAX_REQUEST_MEMORY: u64 = 768 * 1024 * 1024;

/// A response body.
pub type Body = Full<Bytes>;

/// A JSON response.
#[must_use]
pub fn respond(status: u16, value: &Value, close: bool) -> Response<Body> {
    let mut response = Response::new(Full::new(Bytes::from(json::compact(value))));
    *response.status_mut() =
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    if close {
        headers.insert(header::CONNECTION, http::HeaderValue::from_static("close"));
    }
    response
}

/// `{"detail": detail}` with `status`.
#[must_use]
pub fn detail(status: u16, detail: &str, close: bool) -> Response<Body> {
    respond(
        status,
        &json::object([("detail", json::text(detail))]),
        close,
    )
}

/// The response of a refused request.
#[must_use]
pub fn refused(error: &RouteError) -> Response<Body> {
    match error {
        RouteError::Invalid => detail(422, "Request validation failed", false),
        RouteError::Brain(error) => {
            let (status, text) = error.answer();
            detail(status, text, false)
        }
    }
}

/// Why the bearer check refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    /// The token file is unreadable or invalid: 503.
    Unavailable,
    /// The bearer does not match: 401.
    Unauthorized,
}

impl AuthError {
    /// The refusal's response.
    #[must_use]
    pub fn response(self, close: bool) -> Response<Body> {
        match self {
            Self::Unavailable => detail(503, "Brain runtime authentication is unavailable", close),
            Self::Unauthorized => detail(401, "Unauthorized", close),
        }
    }
}

fn token(path: &Path) -> Result<String, AuthError> {
    let raw = std::fs::read(path).map_err(|_| AuthError::Unavailable)?;
    if raw.is_empty() || raw.len() > MAX_TOKEN_BYTES {
        return Err(AuthError::Unavailable);
    }
    let text = String::from_utf8(raw).map_err(|_| AuthError::Unavailable)?;
    let token = team_protocol::python::text::strip(&text);
    if token.is_empty() {
        Err(AuthError::Unavailable)
    } else {
        Ok(token.to_owned())
    }
}

/// The one bearer check: Team's private token, read from its file for every request and compared in constant time.
///
/// # Errors
///
/// [`AuthError`].
pub fn authenticate(headers: &HeaderMap, token_file: &Path) -> Result<(), AuthError> {
    let expected = token(token_file)?;
    let supplied = headers
        .get(header::AUTHORIZATION)
        .map(http::HeaderValue::as_bytes)
        .and_then(|value| value.strip_prefix(b"Bearer "))
        .unwrap_or_default();
    if !supplied.is_empty() && bool::from(supplied.ct_eq(expected.as_bytes())) {
        Ok(())
    } else {
        Err(AuthError::Unauthorized)
    }
}

/// The bytes a request may buffer: none without a body, its declared length, or the ceiling when it streams or
/// declares a length that is not all digits.
#[must_use]
pub fn reservation(headers: &HeaderMap) -> u64 {
    match headers.get(header::CONTENT_LENGTH) {
        None if headers.contains_key(header::TRANSFER_ENCODING) => MAX_REQUEST_BYTES,
        None => 0,
        Some(value) => {
            let text = value.as_bytes();
            if !text.is_empty() && text.iter().all(u8::is_ascii_digit) {
                std::str::from_utf8(text)
                    .ok()
                    .and_then(|text| text.parse().ok())
                    .unwrap_or(u64::MAX)
            } else {
                MAX_REQUEST_BYTES
            }
        }
    }
}

/// The process's request memory budget.
#[derive(Debug, Default)]
pub struct MemoryBudget(Arc<AtomicU64>);

/// One admitted request's reservation, returned when dropped (after its work settled).
#[derive(Debug)]
pub struct Reservation {
    reserved: Arc<AtomicU64>,
    amount: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.reserved.fetch_sub(self.amount, Ordering::SeqCst);
    }
}

impl MemoryBudget {
    /// Reserves the memory a body of `size` bytes needs, or `None` when the budget cannot hold it.
    #[must_use]
    pub fn reserve(&self, size: u64) -> Option<Reservation> {
        let amount = MEMORY_PER_REQUEST.saturating_add(size.saturating_mul(MEMORY_PER_BODY_BYTE));
        let reserved = Arc::clone(&self.0);
        reserved
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current
                    .checked_add(amount)
                    .filter(|total| *total <= MAX_REQUEST_MEMORY)
            })
            .ok()
            .map(|_| Reservation { reserved, amount })
    }

    /// The memory reserved now.
    #[must_use]
    pub fn reserved(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// Why a body could not be received.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyError {
    /// It grew past its bound: 413.
    TooLarge,
    /// It did not arrive in time: 408.
    Late,
    /// The connection failed or closed.
    Gone,
}

/// Receives a whole body within `limit` bytes and the body deadline.
///
/// # Errors
///
/// [`BodyError`].
pub async fn receive(body: hyper::body::Incoming, limit: u64) -> Result<Bytes, BodyError> {
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let collect = Limited::new(body, limit).collect();
    match tokio::time::timeout(REQUEST_BODY_SECONDS, collect).await {
        Err(_) => Err(BodyError::Late),
        Ok(Err(error))
            if error
                .downcast_ref::<http_body_util::LengthLimitError>()
                .is_some() =>
        {
            Err(BodyError::TooLarge)
        }
        Ok(Err(_)) => Err(BodyError::Gone),
        Ok(Ok(collected)) => Ok(collected.to_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, http::HeaderValue::from_str(value).unwrap());
        }
        map
    }

    #[test]
    fn the_bearer_is_read_per_request_and_compared_exactly() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("token");
        assert_eq!(
            authenticate(&headers(&[]), &file),
            Err(AuthError::Unavailable)
        );
        std::fs::write(&file, "  secret-token\n").unwrap();
        assert_eq!(
            authenticate(&headers(&[("authorization", "Bearer secret-token")]), &file),
            Ok(())
        );
        assert_eq!(
            authenticate(&headers(&[("authorization", "Bearer secret")]), &file),
            Err(AuthError::Unauthorized)
        );
        assert_eq!(
            authenticate(&headers(&[("authorization", "secret-token")]), &file),
            Err(AuthError::Unauthorized)
        );
        assert_eq!(
            authenticate(&headers(&[]), &file),
            Err(AuthError::Unauthorized)
        );
        std::fs::write(&file, " \n").unwrap();
        assert_eq!(
            authenticate(&headers(&[("authorization", "Bearer x")]), &file),
            Err(AuthError::Unavailable)
        );
        std::fs::write(&file, vec![b'a'; MAX_TOKEN_BYTES + 1]).unwrap();
        assert_eq!(
            authenticate(&headers(&[("authorization", "Bearer x")]), &file),
            Err(AuthError::Unavailable)
        );
    }

    #[test]
    fn reservations_follow_the_declared_length() {
        assert_eq!(reservation(&headers(&[])), 0);
        assert_eq!(reservation(&headers(&[("content-length", "12")])), 12);
        assert_eq!(
            reservation(&headers(&[("content-length", "1x")])),
            MAX_REQUEST_BYTES
        );
        assert_eq!(
            reservation(&headers(&[("transfer-encoding", "chunked")])),
            MAX_REQUEST_BYTES
        );
    }

    #[test]
    fn the_memory_budget_admits_one_large_and_several_small_requests() {
        let budget = MemoryBudget::default();
        let large = budget
            .reserve(MAX_REQUEST_BYTES)
            .expect("one largest request");
        let small: Vec<_> = (0..4)
            .map(|_| budget.reserve(1024).expect("four small beside it"))
            .collect();
        assert!(budget.reserve(1024).is_none(), "the budget is full");
        drop(large);
        drop(small);
        assert_eq!(budget.reserved(), 0);
    }
}

//! The Brain's only route to its providers: HTTP/1.1 over a CONNECT tunnel through the Brain egress, then TLS to the
//! provider's name (production), or plain HTTP to one loopback origin (tooling and tests, which forward and meter).
//!
//! Connections are pooled per host and capped per pool (five for the model providers, two for Jev, inside the
//! egress's eight tunnels); a call over its pool's cap waits for a connection for at most five seconds. Every request
//! races its time bound and its [`Scope`]: a cancelled or abandoned call drops its connection at once, and a dropped
//! connection never returns to the pool.

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method, Request, Uri};
use http_body_util::{BodyExt, Full, Limited};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::{Connected, Connection};
use hyper_util::rt::{TokioExecutor, TokioIo};
use rustls::{ClientConfig, RootCertStore, pki_types::ServerName};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_rustls::TlsConnector;
use tokio_rustls::client::TlsStream;

use super::{Admit, CallError, Profile, ProviderConfig, Scope, Wire};

/// The model providers' pool cap.
pub const PROVIDER_CONNECTIONS: usize = 5;
/// The Jev pool's cap.
pub const JEV_CONNECTIONS: usize = 2;
/// How long a call waits for a pooled connection, connects, and reads the egress's answer.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// How long an idle connection is kept for the next call.
pub const KEEPALIVE: Duration = Duration::from_secs(60);
/// The largest provider answer read.
pub const MAX_RESPONSE_BYTES: usize = 16 << 20;
const ESTABLISHED: &[u8] = b"HTTP/1.1 200 Connection established\r\n\r\n";
const MAX_PROXY_RESPONSE_BYTES: usize = 1024;
const RETRY_INITIAL: Duration = Duration::from_millis(500);
const RETRY_MAX: Duration = Duration::from_secs(8);
const MAX_RETRY_AFTER_SECONDS: f64 = 60.0;
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Where provider requests go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// Through the CONNECT proxy at `host:port`, then TLS to each provider's name.
    Egress {
        /// The proxy's host name.
        host: String,
        /// The proxy's port.
        port: u16,
    },
    /// Plain HTTP to one loopback origin that forwards by path.
    Origin(SocketAddr),
}

impl Route {
    /// Parses `host:port` as the egress route.
    #[must_use]
    pub fn egress(value: &str) -> Option<Self> {
        let (host, port) = value.rsplit_once(':')?;
        let valid = !host.is_empty()
            && host
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-.".contains(&byte));
        valid.then_some(())?;
        Some(Self::Egress {
            host: host.to_owned(),
            port: port.parse().ok().filter(|port| *port > 0)?,
        })
    }

    /// Parses `http://127.0.0.1:port` (or `[::1]`) as the loopback origin; anything else is refused.
    #[must_use]
    pub fn origin(value: &str) -> Option<Self> {
        let address: SocketAddr = value.strip_prefix("http://")?.parse().ok()?;
        (address.ip().is_loopback() && address.port() > 0).then_some(Self::Origin(address))
    }
}

/// A provider host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Host {
    /// `api.openai.com`.
    OpenAi,
    /// `api.anthropic.com`.
    Anthropic,
    /// `api.typesafe.ai` (the Jev fast path).
    Jev,
}

impl Host {
    /// The host's DNS name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "api.openai.com",
            Self::Anthropic => "api.anthropic.com",
            Self::Jev => "api.typesafe.ai",
        }
    }
}

/// One request to a provider host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// The path.
    pub path: &'static str,
    /// Headers beyond `host` and `content-type`; values may hold a credential and are marked sensitive.
    pub headers: Vec<(&'static str, String)>,
    /// The JSON body.
    pub body: Vec<u8>,
}

/// A provider's answer.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// The status code.
    pub status: u16,
    /// The headers.
    pub headers: HeaderMap,
    /// The body, at most [`MAX_RESPONSE_BYTES`].
    pub body: Bytes,
}

/// Why a request got no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportError {
    /// The scope was cancelled.
    Cancelled,
    /// The time bound passed.
    Timeout,
    /// No connection, or the connection failed.
    Unreachable,
    /// The answer was not HTTP, or too large.
    Invalid,
}

type Pool = Client<Connector, Full<Bytes>>;

/// The process's provider transport.
pub struct Transport {
    route: Route,
    providers: Pool,
    jev: Pool,
    runtime: Handle,
}

impl std::fmt::Debug for Transport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Transport")
            .field("route", &self.route)
            .finish_non_exhaustive()
    }
}

impl Transport {
    /// A transport over `route`, running on the current Tokio runtime, verifying providers against the system roots.
    ///
    /// # Panics
    ///
    /// Outside a Tokio runtime.
    #[must_use]
    pub fn new(route: &Route) -> Self {
        let mut roots = RootCertStore::empty();
        roots.add_parsable_certificates(rustls_native_certs::load_native_certs().certs);
        Self::with_roots(route, roots)
    }

    /// A transport over `route` verifying providers against `roots`.
    ///
    /// # Panics
    ///
    /// Outside a Tokio runtime; never otherwise (the ring provider supports TLS 1.2 and 1.3).
    #[must_use]
    pub fn with_roots(route: &Route, roots: RootCertStore) -> Self {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut tls = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13, &rustls::version::TLS12])
            .expect("the ring provider supports TLS 1.2 and 1.3")
            .with_root_certificates(roots)
            .with_no_client_auth();
        tls.alpn_protocols = vec![b"http/1.1".to_vec()];
        let tls = TlsConnector::from(Arc::new(tls));
        let pool = |slots: usize| {
            let connector = Connector {
                route: route.clone(),
                tls: tls.clone(),
                slots: Arc::new(Semaphore::new(slots)),
            };
            // The timer lets idle connections expire; idle ones of one host keep at most half of the pool's slots,
            // so another host's calls always find a slot instead of waiting behind them.
            Client::builder(TokioExecutor::new())
                .pool_timer(hyper_util::rt::TokioTimer::new())
                .pool_idle_timeout(KEEPALIVE)
                .pool_max_idle_per_host(idle_per_host(slots))
                .build(connector)
        };
        Self {
            providers: pool(PROVIDER_CONNECTIONS),
            jev: pool(JEV_CONNECTIONS),
            route: route.clone(),
            runtime: Handle::current(),
        }
    }

    /// Runs `future` to completion on the transport's runtime from a blocking thread.
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }

    fn uri(&self, host: Host, path: &str) -> Option<Uri> {
        match &self.route {
            Route::Egress { .. } => format!("https://{}{path}", host.name()).parse().ok(),
            Route::Origin(address) => format!("http://{address}{path}").parse().ok(),
        }
    }

    /// The HTTP request for `request` to `host`: every extra header is marked sensitive, so a credential in one is
    /// never printed or compressed into a shared table.
    fn outgoing(
        &self,
        host: Host,
        request: &HttpRequest,
    ) -> Result<Request<Full<Bytes>>, TransportError> {
        let uri = self
            .uri(host, request.path)
            .ok_or(TransportError::Invalid)?;
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header(http::header::HOST, host.name())
            .header(http::header::CONTENT_TYPE, "application/json");
        for (name, value) in &request.headers {
            let mut value = HeaderValue::from_str(value).map_err(|_| TransportError::Invalid)?;
            value.set_sensitive(true);
            builder = builder.header(*name, value);
        }
        builder
            .body(Full::new(Bytes::from(request.body.clone())))
            .map_err(|_| TransportError::Invalid)
    }

    /// Sends one request and reads its whole answer, within `timeout`, unless `scope` stops first.
    ///
    /// # Errors
    ///
    /// [`TransportError`].
    pub async fn send(
        &self,
        scope: &Scope,
        host: Host,
        request: &HttpRequest,
        timeout: Duration,
    ) -> Result<HttpResponse, TransportError> {
        if scope.stopped() {
            return Err(TransportError::Cancelled);
        }
        let outgoing = self.outgoing(host, request)?;
        let pool = if host == Host::Jev {
            &self.jev
        } else {
            &self.providers
        };
        let exchange = async {
            let response = pool
                .request(outgoing)
                .await
                .map_err(|_| TransportError::Unreachable)?;
            let (parts, body) = response.into_parts();
            let body = Limited::new(body, MAX_RESPONSE_BYTES)
                .collect()
                .await
                .map_err(|_| TransportError::Invalid)?;
            Ok(HttpResponse {
                status: parts.status.as_u16(),
                headers: parts.headers,
                body: body.to_bytes(),
            })
        };
        tokio::select! {
            answered = tokio::time::timeout(timeout, exchange) => answered.unwrap_or(Err(TransportError::Timeout)),
            () = scope.until_stopped() => Err(TransportError::Cancelled),
        }
    }

    /// One logical provider call: the request `wire` with `config`'s key, retried as `profile` allows on a connection
    /// failure, a timeout, 408, 409, 429, or a server error, after `admit` accepts each attempt. Returns the attempts
    /// made and the successful answer. Must run on a blocking thread.
    ///
    /// # Errors
    ///
    /// [`CallError::Cancelled`] when the scope stopped, [`CallError::Contract`] when `admit` refused an attempt, and
    /// [`CallError::Request`] for every other failure, the last attempt's refusal included.
    pub fn call_with_retries(
        &self,
        scope: &Scope,
        host: Host,
        config: &ProviderConfig,
        wire: &Wire,
        profile: Profile,
        admit: Option<Admit<'_>>,
    ) -> Result<(u32, HttpResponse), CallError> {
        let request = HttpRequest {
            path: wire.path,
            headers: auth_headers(host, config, wire),
            body: wire.body.clone(),
        };
        let mut attempt = 1;
        loop {
            if let Some(admit) = admit {
                admit(attempt).map_err(CallError::Contract)?;
            }
            if scope.stopped() {
                return Err(CallError::Cancelled);
            }
            scope.usage.request_sent();
            let delay = match self.block_on(self.send(scope, host, &request, profile.timeout)) {
                Ok(response) if (200..300).contains(&response.status) => {
                    return Ok((attempt, response));
                }
                Ok(response) if should_retry(response.status, &response.headers) => {
                    retry_delay(attempt, Some(&response.headers))
                }
                Ok(_) => return Err(CallError::Request),
                Err(TransportError::Cancelled) => return Err(CallError::Cancelled),
                Err(_) => retry_delay(attempt, None),
            };
            if attempt >= profile.attempts {
                return Err(CallError::Request);
            }
            let slept = self.block_on(async {
                tokio::select! {
                    () = tokio::time::sleep(delay) => true,
                    () = scope.until_stopped() => false,
                }
            });
            if !slept {
                return Err(CallError::Cancelled);
            }
            attempt += 1;
        }
    }
}

fn auth_headers(host: Host, config: &ProviderConfig, wire: &Wire) -> Vec<(&'static str, String)> {
    let mut headers = match host {
        Host::OpenAi => vec![(
            "authorization",
            format!("Bearer {}", config.api_key.as_str()),
        )],
        Host::Anthropic => {
            vec![
                ("x-api-key", config.api_key.as_str().to_owned()),
                ("anthropic-version", ANTHROPIC_VERSION.to_owned()),
            ]
        }
        Host::Jev => vec![],
    };
    headers.extend(wire.headers.iter().cloned());
    headers
}

/// The idle connections one host may keep in a pool of `slots`: at most half, at least one.
#[must_use]
pub fn idle_per_host(slots: usize) -> usize {
    (slots / 2).max(1)
}

/// The statuses the provider SDKs retry.
#[must_use]
pub fn retryable_status(status: u16) -> bool {
    matches!(status, 408 | 409 | 429) || status >= 500
}

/// Whether the SDKs retried an unsuccessful answer: as the provider's `x-should-retry` says, else by its status.
#[must_use]
pub fn should_retry(status: u16, headers: &HeaderMap) -> bool {
    match headers
        .get("x-should-retry")
        .and_then(|value| value.to_str().ok())
    {
        Some("true") => true,
        Some("false") => false,
        _ => retryable_status(status),
    }
}

/// The wait before attempt `attempt + 1`: the provider's `retry-after-ms` or `retry-after` when it names more than
/// nothing and at most a minute, otherwise half a second doubled per attempt, at most eight seconds (the SDKs'
/// schedule without their jitter).
#[must_use]
pub fn retry_delay(attempt: u32, headers: Option<&HeaderMap>) -> Duration {
    headers
        .and_then(retry_after)
        .filter(|seconds| *seconds > 0.0 && *seconds <= MAX_RETRY_AFTER_SECONDS)
        .map_or_else(
            || {
                RETRY_INITIAL
                    .saturating_mul(1 << attempt.saturating_sub(1).min(5))
                    .min(RETRY_MAX)
            },
            Duration::from_secs_f64,
        )
}

/// The wait an answer names, in seconds: `retry-after-ms` when it is a number, else `retry-after`'s seconds.
fn retry_after(headers: &HeaderMap) -> Option<f64> {
    let number = |name| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<f64>().ok())
    };
    number("retry-after-ms")
        .map(|milliseconds| milliseconds / 1000.0)
        .or_else(|| number("retry-after"))
}

/// Opens pooled connections: a CONNECT tunnel and TLS, or plain TCP to the loopback origin, each holding one of its
/// pool's slots for as long as it is open.
#[derive(Clone)]
struct Connector {
    route: Route,
    tls: TlsConnector,
    slots: Arc<Semaphore>,
}

impl tower_service::Service<Uri> for Connector {
    type Response = Conn;
    type Error = io::Error;
    type Future = Pin<Box<dyn Future<Output = io::Result<Conn>> + Send>>;

    fn poll_ready(&mut self, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let connector = self.clone();
        Box::pin(async move { connector.open(&uri).await })
    }
}

fn failed(reason: &'static str) -> io::Error {
    io::Error::other(reason)
}

impl Connector {
    async fn open(self, uri: &Uri) -> io::Result<Conn> {
        let permit = tokio::time::timeout(CONNECT_TIMEOUT, Arc::clone(&self.slots).acquire_owned())
            .await
            .map_err(|_| failed("no pooled connection"))?
            .map_err(|_| failed("pool closed"))?;
        let stream = match &self.route {
            Route::Origin(address) => Stream::Plain(
                tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(address))
                    .await
                    .map_err(|_| failed("connect"))??,
            ),
            Route::Egress { host, port } => {
                let name = uri.host().ok_or_else(|| failed("no host"))?;
                Stream::Tls(Box::new(self.tunnel(host, *port, name).await?))
            }
        };
        Ok(Conn {
            io: TokioIo::new(stream),
            _permit: permit,
        })
    }

    async fn tunnel(&self, proxy: &str, port: u16, name: &str) -> io::Result<TlsStream<TcpStream>> {
        let connect = async {
            let mut stream = TcpStream::connect((proxy, port)).await?;
            stream
                .write_all(
                    format!("CONNECT {name}:443 HTTP/1.1\r\nHost: {name}:443\r\n\r\n").as_bytes(),
                )
                .await?;
            let mut reply = Vec::new();
            let mut chunk = [0; 256];
            while !reply.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut chunk).await?;
                if read == 0 || reply.len() + read > MAX_PROXY_RESPONSE_BYTES {
                    return Err(failed("egress answer"));
                }
                reply.extend_from_slice(&chunk[..read]);
            }
            if reply != ESTABLISHED {
                return Err(failed("egress refused"));
            }
            Ok(stream)
        };
        let stream = tokio::time::timeout(CONNECT_TIMEOUT, connect)
            .await
            .map_err(|_| failed("egress timeout"))??;
        let server = ServerName::try_from(name.to_owned()).map_err(|_| failed("server name"))?;
        tokio::time::timeout(CONNECT_TIMEOUT, self.tls.connect(server, stream))
            .await
            .map_err(|_| failed("tls timeout"))?
    }
}

enum Stream {
    Plain(TcpStream),
    Tls(Box<TlsStream<TcpStream>>),
}

impl AsyncRead for Stream {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_read(context, buffer),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_read(context, buffer),
        }
    }
}

impl AsyncWrite for Stream {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_write(context, buffer),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_write(context, buffer),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_flush(context),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_flush(context),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_shutdown(context),
            Self::Tls(stream) => Pin::new(stream.as_mut()).poll_shutdown(context),
        }
    }
}

/// One pooled connection and the pool slot it holds.
struct Conn {
    io: TokioIo<Stream>,
    _permit: OwnedSemaphorePermit,
}

impl Connection for Conn {
    fn connected(&self) -> Connected {
        Connected::new()
    }
}

impl hyper::rt::Read for Conn {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: hyper::rt::ReadBufCursor<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_read(context, buffer)
    }
}

impl hyper::rt::Write for Conn {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().io).poll_write(context, buffer)
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_flush(context)
    }

    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_shutdown(context)
    }
}

#[cfg(test)]
mod tests;

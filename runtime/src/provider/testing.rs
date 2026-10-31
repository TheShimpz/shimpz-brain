//! A fake loopback provider origin for unit tests (test-only): scripted answers per request, every request it
//! received read back, and the connections the client closed. Plain std threads, so it is independent of the runtime
//! under test; answers keep the connection open (HTTP/1.1 keep-alive) unless the client closes it.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use zeroize::Zeroizing;

use super::{ProviderConfig, ProviderId, Providers, Route, Transport};
use crate::json::{self, Value};

/// One scripted answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    /// The status code.
    pub status: u16,
    /// Extra headers.
    pub headers: Vec<(String, String)>,
    /// The body.
    pub body: Vec<u8>,
    /// How long to wait before answering.
    pub delay: Duration,
    /// Never answer: read until the client closes the connection.
    pub hang: bool,
}

impl Answer {
    /// A JSON answer with `status`.
    #[must_use]
    pub fn json(status: u16, body: &Value) -> Self {
        Self {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: json::compact(body).into_bytes(),
            delay: Duration::ZERO,
            hang: false,
        }
    }

    /// A raw answer with `status` and `body`.
    #[must_use]
    pub fn raw(status: u16, body: &str) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: body.as_bytes().to_vec(),
            delay: Duration::ZERO,
            hang: false,
        }
    }

    /// An answer that never comes: the origin reads until the client closes the connection.
    #[must_use]
    pub fn hang() -> Self {
        Self {
            hang: true,
            ..Self::raw(200, "")
        }
    }

    /// This answer with one more header.
    #[must_use]
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_owned(), value.to_owned()));
        self
    }

    /// This answer after `delay`.
    #[must_use]
    pub fn delayed(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// One completed OpenAI Responses body holding a single text message.
    #[must_use]
    pub fn openai_text(text: &str) -> Self {
        Self::json(200, &openai_body(text))
    }

    /// One finished Anthropic Messages body holding a single text block.
    #[must_use]
    pub fn anthropic_text(text: &str) -> Self {
        Self::json(200, &anthropic_body(text))
    }

    /// A counting endpoint's answer.
    #[must_use]
    pub fn counted(tokens: u64) -> Self {
        Self::json(200, &json::object([("input_tokens", json::uint(tokens))]))
    }
}

/// A completed OpenAI Responses body with one text message (`tests/provider_replies.py`).
#[must_use]
pub fn openai_body(text: &str) -> Value {
    openai_output(vec![json::object([
        ("type", json::text("message")),
        ("id", json::text("msg_1")),
        ("status", json::text("completed")),
        ("role", json::text("assistant")),
        (
            "content",
            json::array(vec![json::object([
                ("type", json::text("output_text")),
                ("text", json::text(text)),
                ("annotations", json::array(vec![])),
            ])]),
        ),
    ])])
}

/// A completed OpenAI Responses body with these output items.
#[must_use]
pub fn openai_output(output: Vec<Value>) -> Value {
    json::object([
        ("id", json::text("resp_1")),
        ("object", json::text("response")),
        ("created_at", json::int(0)),
        ("status", json::text("completed")),
        ("model", json::text("gpt-6.1-sol")),
        ("output", json::array(output)),
        (
            "usage",
            json::object([
                ("input_tokens", json::int(10)),
                ("output_tokens", json::int(2)),
                ("total_tokens", json::int(12)),
            ]),
        ),
    ])
}

/// A finished Anthropic Messages body with one text block (`tests/provider_replies.py`).
#[must_use]
pub fn anthropic_body(text: &str) -> Value {
    anthropic_content(
        vec![json::object([
            ("type", json::text("text")),
            ("text", json::text(text)),
        ])],
        "end_turn",
    )
}

/// A finished Anthropic Messages body with these content blocks and stop reason.
#[must_use]
pub fn anthropic_content(content: Vec<Value>, stop_reason: &str) -> Value {
    json::object([
        ("id", json::text("msg_1")),
        ("type", json::text("message")),
        ("role", json::text("assistant")),
        ("model", json::text("claude-sonnet-5-5")),
        ("content", json::array(content)),
        ("stop_reason", json::text(stop_reason)),
        ("stop_sequence", Value::Null),
        (
            "usage",
            json::object([
                ("input_tokens", json::int(10)),
                ("output_tokens", json::int(2)),
            ]),
        ),
    ])
}

/// One request the origin received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Received {
    /// The request path.
    pub path: String,
    /// The headers, names lowercased, in order.
    pub headers: Vec<(String, String)>,
    /// The body bytes.
    pub body: Vec<u8>,
}

impl Received {
    /// The value of header `name`, if sent.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value.as_str())
    }

    /// The body as strict JSON.
    ///
    /// # Panics
    ///
    /// When the body is not one strict JSON document.
    #[must_use]
    pub fn json(&self) -> Value {
        json::parse(&self.body).expect("the request body is strict JSON")
    }
}

type Responder = Box<dyn Fn(&Received) -> Answer + Send + Sync>;

#[derive(Default)]
struct Log {
    received: Vec<Received>,
    accepted: usize,
    open: usize,
    busy: usize,
    peak: usize,
    closed: usize,
}

struct Shared {
    responder: Responder,
    log: Mutex<Log>,
    changed: Condvar,
    stopped: AtomicBool,
}

impl Shared {
    fn log(&self) -> MutexGuard<'_, Log> {
        self.log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn update(&self, change: impl FnOnce(&mut Log)) {
        change(&mut self.log());
        self.changed.notify_all();
    }
}

/// A running fake provider origin on `127.0.0.1`; stops accepting when dropped.
pub struct Origin {
    address: SocketAddr,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Origin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Origin")
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}

impl Origin {
    /// An origin answering every request with `responder`.
    ///
    /// # Panics
    ///
    /// When no loopback port can be bound.
    #[must_use]
    pub fn start(responder: impl Fn(&Received) -> Answer + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let address = listener.local_addr().expect("a bound address");
        let shared = Arc::new(Shared {
            responder: Box::new(responder),
            log: Mutex::default(),
            changed: Condvar::new(),
            stopped: AtomicBool::new(false),
        });
        let serving = Arc::clone(&shared);
        thread::spawn(move || accept(&listener, &serving));
        Self { address, shared }
    }

    /// An origin answering its requests with `answers` in order, then 400 "unscripted request".
    #[must_use]
    pub fn scripted(answers: impl IntoIterator<Item = Answer>) -> Self {
        let queue = Mutex::new(answers.into_iter().collect::<VecDeque<_>>());
        Self::start(move |_| {
            queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop_front()
                .unwrap_or_else(|| Answer::raw(400, "unscripted request"))
        })
    }

    /// The route to this origin.
    #[must_use]
    pub fn route(&self) -> Route {
        Route::Origin(self.address)
    }

    /// The address it listens on.
    #[must_use]
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Every request received so far, in arrival order.
    #[must_use]
    pub fn requests(&self) -> Vec<Received> {
        self.shared.log().received.clone()
    }

    /// The connections accepted so far.
    #[must_use]
    pub fn accepted(&self) -> usize {
        self.shared.log().accepted
    }

    /// The most requests answered at once: each counts from its arrival until its answer is written, a span inside
    /// the client's own, so the count never runs ahead of the client's concurrency the way a lagging close would.
    #[must_use]
    pub fn peak(&self) -> usize {
        self.shared.log().peak
    }

    /// The connections the client closed so far.
    #[must_use]
    pub fn closed(&self) -> usize {
        self.shared.log().closed
    }

    /// Waits until `done` holds for the requests received and the connections closed, at most `timeout`.
    #[must_use]
    pub fn wait(&self, timeout: Duration, done: impl Fn(usize, usize) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        let mut log = self.shared.log();
        loop {
            if done(log.received.len(), log.closed) {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            log = self
                .shared
                .changed
                .wait_timeout(log, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }

    /// Waits until at least `count` requests arrived, at most five seconds.
    #[must_use]
    pub fn wait_requests(&self, count: usize) -> bool {
        self.wait(Duration::from_secs(5), |received, _| received >= count)
    }

    /// Waits until the client closed at least `count` connections, at most five seconds.
    #[must_use]
    pub fn wait_closed(&self, count: usize) -> bool {
        self.wait(Duration::from_secs(5), |_, closed| closed >= count)
    }
}

impl Drop for Origin {
    fn drop(&mut self) {
        self.shared.stopped.store(true, Ordering::SeqCst);
        drop(TcpStream::connect(self.address));
    }
}

fn accept(listener: &TcpListener, shared: &Arc<Shared>) {
    for stream in listener.incoming() {
        if shared.stopped.load(Ordering::SeqCst) {
            return;
        }
        let Ok(stream) = stream else { continue };
        shared.update(|log| {
            log.accepted += 1;
            log.open += 1;
        });
        let serving = Arc::clone(shared);
        thread::spawn(move || {
            let closed = serve(stream, &serving);
            serving.update(|log| {
                log.open -= 1;
                log.closed += usize::from(closed);
            });
        });
    }
}

/// Serves one connection; whether the client closed it.
fn serve(mut stream: TcpStream, shared: &Shared) -> bool {
    let mut buffer = Vec::new();
    loop {
        let Some(received) = read_request(&mut stream, &mut buffer) else {
            return true;
        };
        shared.update(|log| {
            log.received.push(received.clone());
            log.busy += 1;
            log.peak = log.peak.max(log.busy);
        });
        let answer = (shared.responder)(&received);
        if answer.hang {
            let mut sink = [0; 4096];
            while matches!(stream.read(&mut sink), Ok(read) if read > 0) {}
            shared.update(|log| log.busy -= 1);
            return true;
        }
        thread::sleep(answer.delay);
        let written = stream.write_all(&encode(&answer));
        shared.update(|log| log.busy -= 1);
        if written.is_err() {
            return true;
        }
    }
}

/// The HTTP/1.1 bytes of `answer`.
#[must_use]
pub fn encode(answer: &Answer) -> Vec<u8> {
    let mut head = format!("HTTP/1.1 {} Scripted\r\n", answer.status);
    for (name, value) in &answer.headers {
        let _ = write!(head, "{name}: {value}\r\n");
    }
    let _ = write!(head, "content-length: {}\r\n\r\n", answer.body.len());
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(&answer.body);
    bytes
}

/// Reads one HTTP/1.1 request with a `content-length` body from `stream`, keeping any bytes after it in `buffer`;
/// `None` when the client closed the stream first.
pub fn read_request(stream: &mut impl Read, buffer: &mut Vec<u8>) -> Option<Received> {
    let end = loop {
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break end;
        }
        fill(stream, buffer)?;
    };
    let head = String::from_utf8(buffer[..end].to_vec()).ok()?;
    buffer.drain(..end + 4);
    let mut lines = head.split("\r\n");
    let path = lines.next()?.split(' ').nth(1)?.to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    let length = headers
        .iter()
        .find(|(name, _)| name == "content-length")
        .and_then(|(_, value)| value.parse::<usize>().ok())
        .unwrap_or(0);
    while buffer.len() < length {
        fill(stream, buffer)?;
    }
    let body = buffer.drain(..length).collect();
    Some(Received {
        path,
        headers,
        body,
    })
}

fn fill(stream: &mut impl Read, buffer: &mut Vec<u8>) -> Option<()> {
    let mut chunk = [0; 16 * 1024];
    match stream.read(&mut chunk) {
        Ok(read) if read > 0 => {
            buffer.extend_from_slice(&chunk[..read]);
            Some(())
        }
        _ => None,
    }
}

/// A multi-thread Tokio runtime and the [`Providers`] over a route, so a test thread can call the blocking API.
pub struct Harness {
    /// The providers.
    pub providers: Providers,
    runtime: Option<tokio::runtime::Runtime>,
}

impl std::fmt::Debug for Harness {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Harness").finish_non_exhaustive()
    }
}

impl Harness {
    /// Providers over `route`, trusting no root (an origin route needs none).
    ///
    /// # Panics
    ///
    /// When the runtime cannot be built.
    #[must_use]
    pub fn new(route: &Route) -> Self {
        Self::with_roots(route, rustls::RootCertStore::empty())
    }

    /// Providers over `route`, verifying providers against `roots`.
    ///
    /// # Panics
    ///
    /// When the runtime cannot be built.
    #[must_use]
    pub fn with_roots(route: &Route, roots: rustls::RootCertStore) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("a test runtime");
        let transport = {
            let _entered = runtime.enter();
            Transport::with_roots(route, roots)
        };
        Self {
            providers: Providers::new(Arc::new(transport)),
            runtime: Some(runtime),
        }
    }

    /// The runtime, for spawning test work.
    ///
    /// # Panics
    ///
    /// Never: the runtime lives as long as the harness.
    #[must_use]
    pub fn runtime(&self) -> &tokio::runtime::Runtime {
        self.runtime.as_ref().expect("the runtime is alive")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// A configuration of `provider`'s default test model with a low-entropy fake key.
#[must_use]
pub fn config(provider: ProviderId) -> ProviderConfig {
    ProviderConfig {
        provider,
        model: match provider {
            ProviderId::OpenAi => "gpt-6.1-sol",
            ProviderId::Anthropic => "claude-sonnet-5-5",
        }
        .to_owned(),
        api_key: Zeroizing::new("secret-test-key".to_owned()),
    }
}

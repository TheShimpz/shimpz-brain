//! Tests of the transport: routes, the CONNECT tunnel and TLS, credential handling, retries, pooling, and
//! cancellation, against fake egresses and origins on loopback.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rcgen::{BasicConstraints, CertificateParams, IsCa, Issuer, KeyPair};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::{RootCertStore, ServerConfig, ServerConnection, StreamOwned};

use super::*;
use crate::cancel::CancelToken;
use crate::provider::testing::{self, Answer, Harness, Origin, Received};
use crate::provider::{ProviderId, Usage};

const KEY: &str = "secret-test-key";
// The Brain egress admits eight tunnels; one stays free for a retired connection's overlap with its replacement.
const _: () = assert!(PROVIDER_CONNECTIONS + JEV_CONNECTIONS < 8);

fn wire(path: &'static str) -> Wire {
    Wire {
        path,
        headers: vec![("x-extra", "kept".to_owned())],
        body: br#"{"model":"m"}"#.to_vec(),
    }
}

fn profile(attempts: u32) -> Profile {
    Profile {
        effort: None,
        timeout: Duration::from_secs(10),
        attempts,
        max_output_tokens: None,
    }
}

fn openai() -> ProviderConfig {
    testing::config(ProviderId::OpenAi)
}

type Outcome = Result<(u32, HttpResponse), CallError>;

/// Runs one call on its own thread, as a blocking caller would.
fn spawn_call(
    harness: &Harness,
    scope: &Scope,
    path: &'static str,
    attempts: u32,
) -> thread::JoinHandle<(Outcome, Duration)> {
    let transport = Arc::clone(harness.providers.transport());
    let scope = scope.clone();
    thread::spawn(move || {
        let started = Instant::now();
        let outcome = transport.call_with_retries(
            &scope,
            Host::OpenAi,
            &openai(),
            &wire(path),
            profile(attempts),
            None,
        );
        (outcome, started.elapsed())
    })
}

fn call(harness: &Harness, scope: &Scope, attempts: u32) -> Outcome {
    spawn_call(harness, scope, "/v1/responses", attempts)
        .join()
        .unwrap()
        .0
}

fn requests_sent(usage: &Usage) -> u64 {
    usage.counts()[7]
}

#[test]
fn routes_admit_only_an_egress_name_and_a_loopback_origin() {
    assert_eq!(
        Route::egress("shimpz-brain-egress:8888"),
        Some(Route::Egress {
            host: "shimpz-brain-egress".into(),
            port: 8888
        })
    );
    for refused in [
        "",
        ":8888",
        "egress",
        "egress:0",
        "egress:x",
        "eg ress:1",
        "http://egress:1",
        "a/b:1",
    ] {
        assert_eq!(Route::egress(refused), None, "{refused}");
    }
    let loopback: SocketAddr = "127.0.0.1:9".parse().unwrap();
    assert_eq!(
        Route::origin("http://127.0.0.1:9"),
        Some(Route::Origin(loopback))
    );
    assert!(Route::origin("http://[::1]:9").is_some());
    for refused in [
        "http://10.0.0.1:9",
        "https://127.0.0.1:9",
        "127.0.0.1:9",
        "http://127.0.0.1:0",
        "http://localhost:9",
        "http://127.0.0.1:9/v1",
    ] {
        assert_eq!(Route::origin(refused), None, "{refused}");
    }
}

/// What a fake egress does after it read a CONNECT request.
#[derive(Clone)]
enum Behaviour {
    /// Write these bytes, then wait for the client to close.
    Reply(&'static [u8]),
    /// Never answer.
    Silent,
    /// Establish the tunnel and serve TLS with this configuration, answering every request with 200 `{}`.
    Tunnel(Arc<ServerConfig>),
}

/// A fake CONNECT proxy on loopback: records each CONNECT head, then behaves as told.
struct Egress {
    address: SocketAddr,
    heads: Arc<Mutex<Vec<String>>>,
    received: Arc<Mutex<Vec<Received>>>,
    closed: Arc<AtomicUsize>,
}

impl Egress {
    fn start(behaviour: Behaviour) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let egress = Self {
            address: listener.local_addr().unwrap(),
            heads: Arc::default(),
            received: Arc::default(),
            closed: Arc::default(),
        };
        let (heads, received, closed) = (
            Arc::clone(&egress.heads),
            Arc::clone(&egress.received),
            Arc::clone(&egress.closed),
        );
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (heads, received, closed) = (
                    Arc::clone(&heads),
                    Arc::clone(&received),
                    Arc::clone(&closed),
                );
                let behaviour = behaviour.clone();
                thread::spawn(move || {
                    serve_tunnel(stream, &behaviour, &heads, &received);
                    closed.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        egress
    }

    fn route(&self) -> Route {
        Route::Egress {
            host: "127.0.0.1".into(),
            port: self.address.port(),
        }
    }

    fn heads(&self) -> Vec<String> {
        self.heads.lock().unwrap().clone()
    }

    fn wait(&self, done: impl Fn(&Self) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done(self) {
            if Instant::now() > deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
        true
    }
}

fn drain(stream: &mut impl Read) {
    let mut sink = [0; 4096];
    while matches!(stream.read(&mut sink), Ok(read) if read > 0) {}
}

fn serve_tunnel(
    mut stream: TcpStream,
    behaviour: &Behaviour,
    heads: &Mutex<Vec<String>>,
    received: &Mutex<Vec<Received>>,
) {
    let mut head = Vec::new();
    let mut byte = [0; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if !matches!(stream.read(&mut byte), Ok(1)) {
            return;
        }
        head.push(byte[0]);
    }
    heads
        .lock()
        .unwrap()
        .push(String::from_utf8_lossy(&head).into_owned());
    match behaviour {
        Behaviour::Reply(bytes) => {
            let _ = stream.write_all(bytes);
            drain(&mut stream);
        }
        Behaviour::Silent => drain(&mut stream),
        Behaviour::Tunnel(config) => {
            if stream.write_all(ESTABLISHED).is_ok() {
                let connection = ServerConnection::new(Arc::clone(config)).unwrap();
                let mut tls = StreamOwned::new(connection, stream);
                let mut buffer = Vec::new();
                while let Some(request) = testing::read_request(&mut tls, &mut buffer) {
                    received.lock().unwrap().push(request);
                    let answer = Answer::raw(200, "{}");
                    if tls.write_all(&testing::encode(&answer)).is_err() {
                        return;
                    }
                }
            }
        }
    }
}

#[test]
fn a_tunnel_sends_the_exact_connect_line_and_refuses_any_other_egress_answer() {
    for answer in [
        &b"HTTP/1.1 403 Forbidden\r\n\r\n"[..],
        b"HTTP/1.1 200 OK\r\n\r\n",
        b"HTTP/1.0 200 Connection established\r\n\r\n",
        b"HTTP/1.1 200 Connection established\r\nVia: egress\r\n\r\n",
        b"HTTP/1.1 200 Connection established\r\n\r\nextra",
    ] {
        let egress = Egress::start(Behaviour::Reply(answer));
        let harness = Harness::new(&egress.route());
        let scope = Scope::default();
        assert_eq!(call(&harness, &scope, 1).unwrap_err(), CallError::Request);
        assert_eq!(
            egress.heads(),
            ["CONNECT api.openai.com:443 HTTP/1.1\r\nHost: api.openai.com:443\r\n\r\n"]
        );
        assert_eq!(requests_sent(&scope.usage), 1);
    }
}

/// A server configuration presenting a leaf for `name` signed by a fresh root, and roots holding only that root.
fn certificates(name: &str) -> (Arc<ServerConfig>, RootCertStore) {
    let root_key = KeyPair::generate().unwrap();
    let mut root_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    root_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let root = root_params.self_signed(&root_key).unwrap();
    let leaf_key = KeyPair::generate().unwrap();
    let leaf = CertificateParams::new(vec![name.to_owned()])
        .unwrap()
        .signed_by(&leaf_key, &Issuer::from_params(&root_params, &root_key))
        .unwrap();
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let server = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![CertificateDer::from(leaf.der().to_vec())],
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(leaf_key.serialize_der())),
        )
        .unwrap();
    let mut roots = RootCertStore::empty();
    roots.add(root.der().clone()).unwrap();
    (Arc::new(server), roots)
}

#[test]
fn a_tls_round_trip_reaches_the_provider_name_through_the_egress() {
    let (server, roots) = certificates("api.openai.com");
    let egress = Egress::start(Behaviour::Tunnel(server));
    let harness = Harness::with_roots(&egress.route(), roots);
    let scope = Scope::default();
    let (attempts, response) = call(&harness, &scope, 1).unwrap();
    assert_eq!(
        (attempts, response.status, &response.body[..]),
        (1, 200, &b"{}"[..])
    );
    let received = egress.received.lock().unwrap().clone();
    let [request] = &received[..] else {
        panic!("one request")
    };
    assert_eq!(request.path, "/v1/responses");
    assert_eq!(request.header("host"), Some("api.openai.com"));
    assert_eq!(
        request.header("authorization"),
        Some(format!("Bearer {KEY}").as_str())
    );
    assert_eq!(request.header("content-type"), Some("application/json"));
    assert_eq!(request.body, br#"{"model":"m"}"#);
}

#[test]
fn a_certificate_for_another_name_or_root_is_refused() {
    let (other_name, roots) = certificates("api.anthropic.com");
    let egress = Egress::start(Behaviour::Tunnel(other_name));
    let harness = Harness::with_roots(&egress.route(), roots);
    assert_eq!(
        call(&harness, &Scope::default(), 1).unwrap_err(),
        CallError::Request
    );
    let (right_name, _) = certificates("api.openai.com");
    let (_, unrelated_roots) = certificates("api.openai.com");
    let egress = Egress::start(Behaviour::Tunnel(right_name));
    let harness = Harness::with_roots(&egress.route(), unrelated_roots);
    assert_eq!(
        call(&harness, &Scope::default(), 1).unwrap_err(),
        CallError::Request
    );
    assert!(egress.received.lock().unwrap().is_empty());
}

fn sent_only_in(request: &Received, header: &str) {
    for (name, value) in &request.headers {
        assert_eq!(name == header, value.contains(KEY), "{name}");
    }
    assert!(!request.path.contains(KEY));
    assert!(!String::from_utf8_lossy(&request.body).contains(KEY));
}

#[test]
fn the_key_travels_only_in_its_header_and_is_marked_sensitive() {
    let origin = Origin::start(|_| Answer::raw(200, "{}"));
    let harness = Harness::new(&origin.route());
    let transport = harness.providers.transport();
    let anthropic = testing::config(ProviderId::Anthropic);
    for (host, config) in [(Host::OpenAi, openai()), (Host::Anthropic, anthropic)] {
        let wire = wire("/v1/messages");
        transport
            .call_with_retries(&Scope::default(), host, &config, &wire, profile(1), None)
            .unwrap();
        let request = HttpRequest {
            path: wire.path,
            headers: auth_headers(host, &config, &wire),
            body: wire.body.clone(),
        };
        let outgoing = transport.outgoing(host, &request).unwrap();
        for name in ["authorization", "x-api-key", "x-extra"] {
            assert!(
                outgoing
                    .headers()
                    .get(name)
                    .is_none_or(HeaderValue::is_sensitive)
            );
        }
        assert!(!format!("{config:?}{outgoing:?}").contains(KEY));
    }
    let requests = origin.requests();
    sent_only_in(&requests[0], "authorization");
    sent_only_in(&requests[1], "x-api-key");
    assert_eq!(requests[1].header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(requests[1].header("x-extra"), Some("kept"));
}

#[test]
fn retryable_statuses_are_retried_and_counted_and_a_client_error_is_not() {
    let soon = |status| Answer::raw(status, "{}").header("retry-after-ms", "1");
    let origin = Origin::scripted([
        soon(429),
        soon(500),
        soon(408),
        Answer::raw(200, "{}"),
        Answer::raw(400, "{}"),
        soon(503),
        soon(409),
    ]);
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    let (attempts, response) = call(&harness, &scope, 4).unwrap();
    assert_eq!((attempts, response.status), (4, 200));
    assert_eq!(requests_sent(&scope.usage), 4);
    assert_eq!(call(&harness, &scope, 4).unwrap_err(), CallError::Request);
    assert_eq!(requests_sent(&scope.usage), 5);
    assert_eq!(call(&harness, &scope, 2).unwrap_err(), CallError::Request);
    assert_eq!(requests_sent(&scope.usage), 7);
    assert_eq!(origin.requests().len(), 7);
}

#[test]
fn a_connection_failure_is_retried_with_backoff() {
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let route = Route::Origin(closed.local_addr().unwrap());
    drop(closed);
    let harness = Harness::new(&route);
    let scope = Scope::default();
    let started = Instant::now();
    assert_eq!(call(&harness, &scope, 2).unwrap_err(), CallError::Request);
    assert!(started.elapsed() >= RETRY_INITIAL);
    assert_eq!(requests_sent(&scope.usage), 2);
}

#[test]
fn retry_after_is_honoured_within_its_bound() {
    let origin = Origin::scripted([
        Answer::raw(429, "{}").header("retry-after", "1"),
        Answer::raw(200, "{}"),
    ]);
    let harness = Harness::new(&origin.route());
    let started = Instant::now();
    assert_eq!(call(&harness, &Scope::default(), 2).unwrap().0, 2);
    assert!(started.elapsed() >= Duration::from_secs(1));
}

#[test]
fn a_named_wait_counts_only_within_its_bound() {
    let named = |pairs: &[(&'static str, &str)]| {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(*name, HeaderValue::from_str(value).unwrap());
        }
        retry_delay(1, Some(&headers))
    };
    assert_eq!(named(&[("retry-after", "7")]), Duration::from_secs(7));
    assert_eq!(
        named(&[("retry-after", "1.5")]),
        Duration::from_millis(1500)
    );
    assert_eq!(
        named(&[("retry-after-ms", "250"), ("retry-after", "7")]),
        Duration::from_millis(250)
    );
    assert_eq!(
        named(&[("retry-after-ms", "x"), ("retry-after", "7")]),
        Duration::from_secs(7)
    );
    for unbounded in ["0", "61", "soon", "-1"] {
        assert_eq!(named(&[("retry-after", unbounded)]), RETRY_INITIAL);
    }
    assert_eq!(retry_delay(2, None), RETRY_INITIAL * 2);
    assert_eq!(retry_delay(9, None), RETRY_MAX);
}

#[test]
fn the_provider_decides_with_x_should_retry() {
    let origin = Origin::scripted([
        Answer::raw(500, "{}").header("x-should-retry", "false"),
        Answer::raw(400, "{}")
            .header("x-should-retry", "true")
            .header("retry-after-ms", "1"),
        Answer::raw(200, "{}"),
    ]);
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    assert_eq!(call(&harness, &scope, 3).unwrap_err(), CallError::Request);
    assert_eq!(origin.requests().len(), 1);
    assert_eq!(call(&harness, &scope, 3).unwrap().0, 2);
    assert_eq!(requests_sent(&scope.usage), 3);
}

/// Cancels `scope` once `ready` holds, then expects the call to end cancelled within a second.
fn cancel_when(
    scope: &Scope,
    call: thread::JoinHandle<(Outcome, Duration)>,
    ready: impl Fn() -> bool,
) {
    assert!(ready(), "the call reached the point to cancel");
    thread::sleep(Duration::from_millis(100));
    let cancelled = Instant::now();
    scope.cancel.cancel();
    let (outcome, _) = call.join().unwrap();
    assert_eq!(outcome.unwrap_err(), CallError::Cancelled);
    assert!(cancelled.elapsed() < Duration::from_secs(1));
}

#[test]
fn cancelling_during_the_egress_connect_closes_the_tunnel() {
    let egress = Egress::start(Behaviour::Silent);
    let harness = Harness::new(&egress.route());
    let scope = Scope::default();
    let pending = spawn_call(&harness, &scope, "/v1/responses", 3);
    cancel_when(&scope, pending, || {
        egress.wait(|egress| egress.heads().len() == 1)
    });
    assert!(egress.wait(|egress| egress.closed.load(Ordering::SeqCst) == 1));
    thread::sleep(Duration::from_millis(300));
    assert_eq!(egress.heads().len(), 1);
    assert_eq!(requests_sent(&scope.usage), 1);
}

#[test]
fn cancelling_during_the_response_read_closes_only_that_connection() {
    let origin = Origin::start(|request| {
        if request.path == "/hang" {
            Answer::hang()
        } else {
            Answer::raw(200, "{}")
        }
    });
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    let pending = spawn_call(&harness, &scope, "/hang", 3);
    assert!(origin.wait_requests(1));
    let other = Scope::default();
    let answered = spawn_call(&harness, &other, "/ok", 1).join().unwrap().0;
    assert_eq!(answered.unwrap().1.status, 200);
    cancel_when(&scope, pending, || true);
    assert!(origin.wait_closed(1));
    thread::sleep(Duration::from_millis(300));
    assert_eq!(origin.requests().len(), 2);
    assert_eq!(origin.closed(), 1);
    assert_eq!(call(&harness, &other, 1).unwrap().1.status, 200);
}

#[test]
fn an_abandoned_attempt_stops_its_call_like_a_cancellation() {
    let origin = Origin::scripted([Answer::hang()]);
    let harness = Harness::new(&origin.route());
    let abandoned = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&abandoned);
    let scope = Scope::new(CancelToken::new()).with_stop(move || flag.load(Ordering::SeqCst));
    let pending = spawn_call(&harness, &scope, "/v1/responses", 3);
    assert!(origin.wait_requests(1));
    abandoned.store(true, Ordering::SeqCst);
    let (outcome, _) = pending.join().unwrap();
    assert_eq!(outcome.unwrap_err(), CallError::Cancelled);
    assert!(origin.wait_closed(1));
}

#[test]
fn cancelling_during_retry_backoff_sends_nothing_more() {
    let origin = Origin::scripted([Answer::raw(503, "{}").header("retry-after", "30")]);
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    let pending = spawn_call(&harness, &scope, "/v1/responses", 3);
    cancel_when(&scope, pending, || origin.wait_requests(1));
    thread::sleep(Duration::from_millis(300));
    assert_eq!(origin.requests().len(), 1);
    assert_eq!(requests_sent(&scope.usage), 1);
}

#[test]
fn a_scope_cancelled_before_its_call_never_connects() {
    let origin = Origin::scripted([]);
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    scope.cancel.cancel();
    assert_eq!(call(&harness, &scope, 3).unwrap_err(), CallError::Cancelled);
    assert_eq!(origin.accepted(), 0);
    assert_eq!(requests_sent(&scope.usage), 0);
}

#[test]
fn concurrent_calls_wait_for_a_pooled_connection_and_idle_ones_are_bounded() {
    let origin = Origin::start(|_| Answer::raw(200, "{}").delayed(Duration::from_millis(200)));
    let harness = Harness::new(&origin.route());
    let scope = Scope::default();
    let calls: Vec<_> = (0..PROVIDER_CONNECTIONS + 4)
        .map(|_| spawn_call(&harness, &scope, "/v1/responses", 1))
        .collect();
    for pending in calls {
        assert_eq!(pending.join().unwrap().0.unwrap().1.status, 200);
    }
    // How many calls overlap depends on scheduling; the bound is that none ever exceeds the pool's slots.
    assert!(
        origin.peak() <= PROVIDER_CONNECTIONS,
        "never more connections than the pool's slots"
    );
    assert_eq!(call(&harness, &scope, 1).unwrap().1.status, 200);
    // Only half the pool stays idle per host: every other connection is closed, so its slot is free for another host.
    let idle = idle_per_host(PROVIDER_CONNECTIONS);
    let open = || origin.accepted().saturating_sub(origin.closed());
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while open() > idle && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let settled = open() <= idle;
    assert!(settled, "idle connections settle within the per-host bound");
    assert_eq!(
        (idle_per_host(5), idle_per_host(2), idle_per_host(1)),
        (2, 1, 1)
    );
}

//! The HTTP boundary of the built binary: health without a bearer, the bearer before any body byte, size and
//! method refusals, the closed body, thread deletion, and the exclusive state directory.

use std::net::SocketAddr;

use crate::support::{self, Brain, TOKEN, request};

fn origin() -> SocketAddr {
    format!("127.0.0.1:{}", support::free_port())
        .parse()
        .unwrap()
}

#[test]
fn health_needs_no_bearer_and_unknown_routes_and_methods_are_refused() {
    let directory = tempfile::tempdir().unwrap();
    let brain = Brain::start(directory.path(), origin(), &[]);
    let health = request(brain.address, "GET", "/health", &[], None).unwrap();
    assert_eq!(
        (health.status, health.body.as_str()),
        (200, r#"{"status":"ok","runtime":"holdpoint"}"#)
    );
    assert_eq!(
        request(brain.address, "GET", "/nowhere", &[], None)
            .unwrap()
            .status,
        404
    );
    assert_eq!(
        request(brain.address, "GET", "/v1/turns", &[], None)
            .unwrap()
            .status,
        405
    );
    assert_eq!(
        request(brain.address, "POST", "/health", &[], None)
            .unwrap()
            .status,
        405
    );
}

#[test]
fn every_body_is_authenticated_before_it_is_read() {
    let directory = tempfile::tempdir().unwrap();
    let brain = Brain::start(directory.path(), origin(), &[]);
    let body = br#"{"thread_id":"thread-1"}"#;
    let unauthorized =
        request(brain.address, "POST", "/v1/threads/delete", &[], Some(body)).unwrap();
    assert_eq!(
        (unauthorized.status, unauthorized.body.as_str()),
        (401, r#"{"detail":"Unauthorized"}"#)
    );
    let wrong = request(
        brain.address,
        "POST",
        "/v1/threads/delete",
        &[("authorization", "Bearer other")],
        Some(body),
    )
    .unwrap();
    assert_eq!(wrong.status, 401);
    let bodyless = request(brain.address, "POST", "/v1/threads/delete", &[], None).unwrap();
    assert_eq!(
        bodyless.status, 401,
        "a bodyless request still needs the bearer"
    );
    std::fs::write(directory.path().join("token"), "").unwrap();
    let unavailable = brain.post("/v1/threads/delete", r#"{"thread_id":"thread-1"}"#);
    assert_eq!(unavailable.status, 503);
    assert!(
        unavailable
            .body
            .contains("Brain runtime authentication is unavailable")
    );
}

#[test]
fn bodies_are_bounded_and_closed() {
    let directory = tempfile::tempdir().unwrap();
    let brain = Brain::start(directory.path(), origin(), &[]);
    let bearer = format!("Bearer {TOKEN}");
    let headers = [
        ("authorization", bearer.as_str()),
        ("content-length", "4194305"),
    ];
    let large = request(brain.address, "POST", "/v1/threads/delete", &headers, None).unwrap();
    assert_eq!(
        large.status, 413,
        "refused from its declared length before any byte is read"
    );
    assert!(
        large
            .head
            .to_ascii_lowercase()
            .contains("connection: close")
    );
    assert_eq!(brain.post("/v1/threads/delete", "not json").status, 422);
    assert_eq!(
        brain
            .post("/v1/threads/delete", r#"{"thread_id":"t","extra":1}"#)
            .status,
        422
    );
    assert_eq!(
        brain
            .post("/v1/threads/delete", r#"{"thread_id":"bad id"}"#)
            .status,
        422
    );
    assert_eq!(
        brain
            .post("/v1/threads/delete", r#"{"thread_id":"t","thread_id":"u"}"#)
            .status,
        422
    );
    let invalid = brain.post("/v1/turns", "{}");
    assert_eq!(
        (invalid.status, invalid.body.as_str()),
        (422, r#"{"detail":"Request validation failed"}"#)
    );
}

#[test]
fn deleting_a_thread_reveals_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let brain = Brain::start(directory.path(), origin(), &[]);
    for _ in 0..2 {
        let deleted = brain.post("/v1/threads/delete", r#"{"thread_id":"thread-never-seen"}"#);
        assert_eq!(
            (deleted.status, deleted.body.as_str()),
            (200, r#"{"status":"deleted"}"#)
        );
    }
}

#[test]
fn one_process_owns_a_state_directory() {
    let directory = tempfile::tempdir().unwrap();
    let brain = Brain::start(directory.path(), origin(), &[]);
    let second = std::process::Command::new(env!("CARGO_BIN_EXE_shimpz-brain"))
        .args(["serve", "--listen", "127.0.0.1:0", "--state-dir"])
        .arg(&brain.state)
        .args(["--provider-origin", "http://127.0.0.1:9"])
        .status()
        .unwrap();
    assert_eq!(second.code(), Some(3));
    let usage = std::process::Command::new(env!("CARGO_BIN_EXE_shimpz-brain"))
        .arg("serve")
        .status()
        .unwrap();
    assert_eq!(usage.code(), Some(2));
}

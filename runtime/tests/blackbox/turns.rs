//! Chat turns through the built binary and a scripted provider origin: replies, Action suspension and resume,
//! cancellation, crashes and restarts, refused results, serialization, and tampered state.

use std::time::Duration;

use crate::origin::{self, Behaviour, Origin};
use crate::support::Brain;
use crate::team;

#[test]
fn a_turn_replies_and_the_next_turn_remembers_it() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![origin::reply("Hello there"), origin::reply("Again")]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let first = brain.post("/v1/turns", &team::start("thread-a", "Hi"));
    assert_eq!(first.status, 200, "{}", first.body);
    assert!(first.body.contains(r#""reply":"Hello there""#));
    assert!(first.body.contains(r#""usage":{"model_calls":1"#));
    let second = brain.post("/v1/turns", &team::start("thread-a", "And now?"));
    assert_eq!(second.status, 200, "{}", second.body);
    let received = origin.received();
    assert_eq!(received.len(), 2);
    assert!(
        received
            .iter()
            .all(|request| request.path == "/v1/responses")
    );
    assert!(
        received[1].body.contains("Hello there"),
        "the next turn carries the completed exchange"
    );
    assert!(
        !received[0].body.contains(team::KEY),
        "the key travels only in its header"
    );
}

#[test]
fn an_action_suspends_and_a_resume_completes_without_repeating_a_call() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![
        origin::call(&team::list_zones(), "{}"),
        origin::reply("You have one zone"),
    ]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let started = brain.post("/v1/turns", &team::start("thread-b", "List my zones"));
    assert_eq!(
        team::status(&started.body),
        "action-required",
        "{}",
        started.body
    );
    let interrupt = team::interrupt_id(&started.body);
    let resumed = brain.post(
        "/v1/turns/resume",
        &team::resume("thread-b", &interrupt, r#"{"zones":["a"]}"#),
    );
    assert_eq!(team::status(&resumed.body), "completed", "{}", resumed.body);
    assert_eq!(origin.received().len(), 2);
    let again = brain.post(
        "/v1/turns/resume",
        &team::resume("thread-b", &interrupt, "{}"),
    );
    assert_eq!(again.status, 400, "a turn answers each request once");
}

#[test]
fn a_killed_brain_resumes_a_suspended_turn_without_asking_again() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![
        origin::call(&team::list_zones(), "{}"),
        origin::reply("Done"),
    ]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let started = brain.post("/v1/turns", &team::start("thread-c", "List my zones"));
    let interrupt = team::interrupt_id(&started.body);
    brain.kill();
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let resumed = brain.post(
        "/v1/turns/resume",
        &team::resume("thread-c", &interrupt, r#"{"zones":[]}"#),
    );
    assert_eq!(team::status(&resumed.body), "completed", "{}", resumed.body);
    assert_eq!(
        origin.received().len(),
        2,
        "the recorded call is replayed, never sent again"
    );
    assert!(
        resumed.body.contains(r#""model_calls":1"#),
        "usage counts only the new call"
    );
}

#[test]
fn a_disconnect_cancels_only_its_turn_and_leaves_history_unchanged() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![Behaviour::Hang, origin::reply("Second")]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let address = brain.address;
    let body = team::start("thread-d", "First");
    let abandoned = std::thread::spawn(move || {
        let mut stream = std::net::TcpStream::connect(address).unwrap();
        let head = format!(
            "POST /v1/turns HTTP/1.1\r\nHost: brain\r\nAuthorization: Bearer {}\r\nContent-Length: {}\r\n\r\n{body}",
            crate::support::TOKEN,
            body.len()
        );
        std::io::Write::write_all(&mut stream, head.as_bytes()).unwrap();
        std::thread::sleep(Duration::from_millis(500));
    });
    origin.wait_received(1);
    abandoned.join().unwrap();
    assert!(
        origin.wait_closed(1, 2),
        "the provider connection is dropped within a second or two"
    );
    let next = brain.post("/v1/turns", &team::start("thread-d", "Second try"));
    assert_eq!(next.status, 200, "{}", next.body);
    assert!(
        !origin.received()[1].body.contains("First"),
        "the cancelled exchange was never kept"
    );
}

#[test]
fn a_turn_killed_mid_call_is_forgotten_and_its_resume_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![Behaviour::Hang, origin::reply("Fresh")]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let address = brain.address;
    let body = team::start("thread-e", "Lost words");
    std::thread::spawn(move || {
        let _ = crate::support::request(
            address,
            "POST",
            "/v1/turns",
            &[(
                "authorization",
                &format!("Bearer {}", crate::support::TOKEN),
            )],
            Some(body.as_bytes()),
        );
    });
    origin.wait_received(1);
    brain.kill();
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let refused = brain.post(
        "/v1/turns/resume",
        &team::resume("thread-e", "0123456789abcdef0123456789abcdef", "{}"),
    );
    assert_eq!(refused.status, 400);
    let fresh = brain.post("/v1/turns", &team::start("thread-e", "Start over"));
    assert_eq!(fresh.status, 200, "{}", fresh.body);
    assert!(!origin.received()[1].body.contains("Lost words"));
}

#[test]
fn unknown_results_are_refused_and_nothing_is_asked_twice() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![
        origin::call(&team::list_zones(), "{}"),
        origin::reply("Done"),
    ]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let started = brain.post("/v1/turns", &team::start("thread-f", "List my zones"));
    let interrupt = team::interrupt_id(&started.body);
    let unknown = brain.post(
        "/v1/turns/resume",
        &team::resume("thread-f", "ffffffffffffffffffffffffffffffff", "{}"),
    );
    assert_eq!(unknown.status, 400);
    let resumed = brain.post(
        "/v1/turns/resume",
        &team::resume("thread-f", &interrupt, "{}"),
    );
    assert_eq!(team::status(&resumed.body), "completed", "{}", resumed.body);
    assert_eq!(origin.received().len(), 2);
}

#[test]
fn a_new_start_abandons_a_pending_turn_with_its_thread() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![
        origin::call(&team::list_zones(), "{}"),
        origin::reply("Fresh start"),
    ]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let started = brain.post("/v1/turns", &team::start("thread-g", "Pending words"));
    let interrupt = team::interrupt_id(&started.body);
    let fresh = brain.post("/v1/turns", &team::start("thread-g", "Something else"));
    assert_eq!(fresh.status, 200, "{}", fresh.body);
    assert!(!origin.received()[1].body.contains("Pending words"));
    let stale = brain.post(
        "/v1/turns/resume",
        &team::resume("thread-g", &interrupt, "{}"),
    );
    assert_eq!(stale.status, 400);
}

#[test]
fn a_tampered_thread_is_a_state_error() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![origin::reply("Kept")]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    assert_eq!(
        brain
            .post("/v1/turns", &team::start("thread-h", "Hi"))
            .status,
        200
    );
    let state = brain.state.clone();
    brain.kill();
    let store = rusqlite::Connection::open(state.join("threads.sqlite3")).unwrap();
    store
        .execute("UPDATE threads SET history = '[{\"role\":\"tool\"}]'", [])
        .unwrap();
    drop(store);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let refused = brain.post("/v1/turns", &team::start("thread-h", "Again"));
    assert_eq!(
        (refused.status, refused.body.as_str()),
        (503, r#"{"detail":"Brain runtime state operation failed"}"#)
    );
}

#[test]
fn an_unreadable_pending_request_is_a_state_error_that_deletes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![origin::call(&team::list_zones(), "{}")]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let started = brain.post("/v1/turns", &team::start("thread-i", "List my zones"));
    let interrupt = team::interrupt_id(&started.body);
    let state = brain.state.clone();
    brain.kill();
    let ledger = rusqlite::Connection::open(state.join("ledger.sqlite3")).unwrap();
    ledger
        .execute(r#"UPDATE deferred SET request = '{"request":"[]"}'"#, [])
        .unwrap();
    drop(ledger);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    for _ in 0..2 {
        let refused = brain.post(
            "/v1/turns/resume",
            &team::resume("thread-i", &interrupt, "{}"),
        );
        assert_eq!(refused.status, 503, "{}", refused.body);
    }
    assert_eq!(
        brain
            .post("/v1/turns", &team::start("thread-i", "Another"))
            .status,
        503
    );
    let ledger = rusqlite::Connection::open(state.join("ledger.sqlite3")).unwrap();
    let runs: i64 = ledger
        .query_row("SELECT count(*) FROM runs", [], |row| row.get(0))
        .unwrap();
    assert_eq!(runs, 1, "the run it could not read was kept");
}

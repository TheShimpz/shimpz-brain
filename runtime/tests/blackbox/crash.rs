//! Kills at each window between the thread store and the run ledger (the `failpoints` build only): a crash neither
//! loses nor duplicates an exchange, and the next request settles what the crash left.

use crate::origin::{self, Origin};
use crate::support::{Brain, TOKEN, request};
use crate::team;

/// Starts the Brain armed at `point`, sends `body` to `path` (the process aborts there), and returns a fresh Brain.
fn crash_at(
    directory: &std::path::Path,
    origin: &Origin,
    point: &str,
    path: &str,
    body: &str,
) -> Brain {
    let mut armed = Brain::start(
        directory,
        origin.address,
        &[("SHIMPZ_BRAIN_FAILPOINT", point)],
    );
    let bearer = format!("Bearer {TOKEN}");
    let _ = request(
        armed.address,
        "POST",
        path,
        &[("authorization", &bearer)],
        Some(body.as_bytes()),
    );
    assert!(!armed.wait_exit().success(), "the Brain aborted at {point}");
    drop(armed);
    Brain::start(directory, origin.address, &[])
}

fn occurrences(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

#[test]
fn a_crash_before_or_after_submitting_keeps_the_thread_usable_and_unchanged() {
    for point in ["start.reference", "start.submitted"] {
        let directory = tempfile::tempdir().unwrap();
        let origin = Origin::start(vec![origin::reply("Only answer")]);
        let brain = crash_at(
            directory.path(),
            &origin,
            point,
            "/v1/turns",
            &team::start("crash-1", "Lost message"),
        );
        let next = brain.post("/v1/turns", &team::start("crash-1", "Fresh message"));
        assert_eq!(next.status, 200, "{point}: {}", next.body);
        let last = origin.received().last().unwrap().body.clone();
        assert!(
            !last.contains("Lost message"),
            "{point}: the crashed turn was never kept"
        );
    }
}

#[test]
fn a_crash_around_the_commit_keeps_the_exchange_exactly_once() {
    for point in ["commit.exchange", "reconcile.deleted"] {
        let directory = tempfile::tempdir().unwrap();
        let origin = Origin::start(vec![
            origin::reply("Committed answer"),
            origin::reply("Next answer"),
        ]);
        let brain = crash_at(
            directory.path(),
            &origin,
            point,
            "/v1/turns",
            &team::start("crash-2", "First message"),
        );
        let next = brain.post("/v1/turns", &team::start("crash-2", "Second message"));
        assert_eq!(next.status, 200, "{point}: {}", next.body);
        let last = origin.received().last().unwrap().body.clone();
        assert_eq!(
            occurrences(&last, "Committed answer"),
            1,
            "{point}: the exchange is kept once"
        );
    }
}

#[test]
fn a_crash_between_the_deletion_steps_is_finished_by_the_next_deletion() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![
        origin::call(&team::list_zones(), "{}"),
        origin::reply("Fresh"),
    ]);
    let first = Brain::start(directory.path(), origin.address, &[]);
    let started = first.post("/v1/turns", &team::start("crash-3", "Pending message"));
    assert_eq!(team::status(&started.body), "action-required");
    drop(first);
    let brain = crash_at(
        directory.path(),
        &origin,
        "delete.run",
        "/v1/threads/delete",
        r#"{"thread_id":"crash-3"}"#,
    );
    let deleted = brain.post("/v1/threads/delete", r#"{"thread_id":"crash-3"}"#);
    assert_eq!(deleted.status, 200, "{}", deleted.body);
    let fresh = brain.post("/v1/turns", &team::start("crash-3", "New message"));
    assert_eq!(fresh.status, 200, "{}", fresh.body);
    assert!(
        !origin
            .received()
            .last()
            .unwrap()
            .body
            .contains("Pending message")
    );
}

#[test]
fn a_resume_repeated_after_a_crash_at_its_commit_receives_the_committed_turn() {
    for point in ["commit.exchange", "reconcile.deleted"] {
        let directory = tempfile::tempdir().unwrap();
        let origin = Origin::start(vec![
            origin::call(&team::list_zones(), "{}"),
            origin::reply("One zone"),
        ]);
        let first = Brain::start(directory.path(), origin.address, &[]);
        let started = first.post("/v1/turns", &team::start("crash-4", "List my zones"));
        assert_eq!(team::status(&started.body), "action-required");
        drop(first);
        let interrupt = team::interrupt_id(&started.body);
        let resume = team::resume("crash-4", &interrupt, r#"{"zones":["a"]}"#);
        let brain = crash_at(
            directory.path(),
            &origin,
            point,
            "/v1/turns/resume",
            &resume,
        );
        // Team never received the answer and resumes again: it receives the committed turn, asking nothing again.
        let again = brain.post("/v1/turns/resume", &resume);
        assert_eq!(again.status, 200, "{point}: {}", again.body);
        assert_eq!(team::status(&again.body), "completed", "{point}");
        assert!(again.body.contains("One zone"), "{point}");
        assert_eq!(origin.received().len(), 2, "{point}");
    }
}

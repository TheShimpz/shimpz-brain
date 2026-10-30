//! The chat agent's loop end to end through the binary: corrections, the recursion bound, clarification, memory,
//! the Browser call and its reply-only pass, attachments, and the pins a resume must match.

use crate::origin::{self, Behaviour, Origin};
use crate::support::Brain;
use crate::team;

fn clarify(default: u8) -> Behaviour {
    origin::call(
        "shimpz_clarify",
        &format!(
            r#"{{"question":"Which zone?","options":[{{"label":"Production","description":""}},{{"label":"Staging","description":""}}],"default_index":{default}}}"#
        ),
    )
}

#[test]
fn invalid_action_arguments_are_corrected_and_never_reach_team() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![
        origin::call(&team::list_zones(), r#"{"extra":1}"#),
        origin::reply("Corrected"),
    ]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let turn = brain.post("/v1/turns", &team::start("agent-1", "List zones"));
    assert_eq!(team::status(&turn.body), "completed", "{}", turn.body);
    let second = &origin.received()[1].body;
    assert!(
        second.contains("function_call_output"),
        "the correction answers the call"
    );
}

#[test]
fn endless_invalid_calls_end_at_the_recursion_bound_without_an_action() {
    let directory = tempfile::tempdir().unwrap();
    let script = (0..12)
        .map(|_| origin::call(&team::list_zones(), r#"{"extra":1}"#))
        .collect();
    let origin = Origin::start(script);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let turn = brain.post("/v1/turns", &team::start("agent-2", "List zones"));
    assert_eq!(turn.status, 502, "{}", turn.body);
    assert!(origin.received().len() < 12);
}

#[test]
fn a_clarification_ends_the_turn_and_is_remembered_as_shown() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![clarify(0), origin::reply("Staging it is")]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let turn = brain.post("/v1/turns", &team::start("agent-3", "Change the zone"));
    assert_eq!(turn.status, 200, "{}", turn.body);
    let clarification = team::member(&turn.body, "clarification");
    assert_eq!(
        brain_runtime::json::str_of(&clarification, "question"),
        Some("Which zone?")
    );
    let reply =
        brain_runtime::json::str_of(&team::member(&turn.body, "reply"), "").map(str::to_owned);
    assert!(reply.is_none());
    assert_eq!(
        origin.received().len(),
        1,
        "the question ends the turn without another call"
    );
    assert_eq!(
        brain
            .post("/v1/turns", &team::start("agent-3", "Staging"))
            .status,
        200
    );
    assert!(
        origin.received()[1].body.contains("Which zone?"),
        "the next turn remembers the question it was shown"
    );
}

#[test]
fn a_routine_turns_question_steers_no_choice() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![clarify(1)]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let body = team::start_with(
        "agent-4",
        "Every day check zones",
        r#""routines":[],"routine_mode":true"#,
    );
    let turn = brain.post("/v1/turns", &body);
    assert_eq!(turn.status, 200, "{}", turn.body);
    assert_eq!(
        team::member(&turn.body, "clarification").get("default_index"),
        Some(&brain_runtime::json::Value::Null)
    );
}

#[test]
fn a_confirmed_memory_change_is_returned_with_the_reply() {
    let directory = tempfile::tempdir().unwrap();
    let proposal = r#"{"op":"remember","topic":"tone","quote":"always answer briefly"}"#;
    let origin = Origin::start(vec![
        origin::call("shimpz_memory", proposal),
        origin::reply("Noted."),
        origin::reply(r#"{\"lasting\":[true]}"#),
    ]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let body = team::start_with(
        "agent-5",
        "From now on, always answer briefly",
        r#""memories":[]"#,
    );
    let turn = brain.post("/v1/turns", &body);
    assert_eq!(turn.status, 200, "{}", turn.body);
    assert!(
        turn.body.contains(
            r#""memory":[{"op":"remember","topic":"tone","preference":"always answer briefly"}]"#
        ),
        "{}",
        turn.body
    );
    assert_eq!(
        origin.received().len(),
        3,
        "the independent check is one more call"
    );
}

#[test]
fn a_rejected_memory_check_keeps_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let proposal = r#"{"op":"remember","topic":"tone","quote":"always answer briefly"}"#;
    let origin = Origin::start(vec![
        origin::call("shimpz_memory", proposal),
        origin::reply("Noted."),
        origin::reply(r#"{\"lasting\":[false]}"#),
    ]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let body = team::start_with(
        "agent-6",
        "From now on, always answer briefly",
        r#""memories":[]"#,
    );
    let turn = brain.post("/v1/turns", &body);
    assert!(turn.body.contains(r#""memory":[]"#), "{}", turn.body);
}

#[test]
fn a_browser_call_suspends_alone_and_its_pass_may_only_reply() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![
        origin::call(
            "browser",
            r#"{"objective":"Find the release notes of example.com"}"#,
        ),
        origin::call(&team::list_zones(), "{}"),
        origin::reply("Here is what the page says."),
    ]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let body = team::start_with("agent-7", "Search the web", r#""browser_available":true"#);
    let started = brain.post("/v1/turns", &body);
    assert_eq!(
        team::status(&started.body),
        "browser-required",
        "{}",
        started.body
    );
    let browser = team::member(&started.body, "browser");
    let interrupt = brain_runtime::json::str_of(&browser, "interrupt_id")
        .unwrap()
        .to_owned();
    let result = r#"{"answer":"Notes","sources":[]}"#;
    let resume = team::resume_with("agent-7", &interrupt, result, r#""browser_available":true"#);
    let resumed = brain.post("/v1/turns/resume", &resume);
    assert_eq!(team::status(&resumed.body), "completed", "{}", resumed.body);
    let received = origin.received();
    assert!(
        !received[1].body.contains(r#""tools":[{"#),
        "the pass after the result offers no tool"
    );
    assert_eq!(
        received.len(),
        3,
        "a call after the result is refused and the model replies"
    );
}

#[test]
fn a_resume_must_keep_the_provider_and_attachments_its_start_pinned() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![origin::call(&team::list_zones(), "{}")]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    let started = brain.post("/v1/turns", &team::start("agent-8", "List zones"));
    let interrupt = team::interrupt_id(&started.body);
    let other = format!(
        r#""provider":{{"provider":"anthropic","model":"claude-sonnet-5-5","api_key":"{}"}}"#,
        team::KEY
    );
    let switched = brain.post(
        "/v1/turns/resume",
        &team::resume_with("agent-8", &interrupt, "{}", &other),
    );
    assert_eq!(
        (switched.status, switched.body.as_str()),
        (
            400,
            r#"{"detail":"model provider changed during the pending turn"}"#
        )
    );
    let file = r#""attachments":[{"id":"file-0123456789abcdef0123456789abcdef","name":"a.txt","media_type":"text/plain","size":2,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","content":{"type":"opaque","reason":"unsupported"}}]"#;
    let attached = brain.post(
        "/v1/turns/resume",
        &team::resume_with("agent-8", &interrupt, "{}", file),
    );
    assert_eq!(attached.status, 400, "{}", attached.body);
}

#[test]
fn a_scope_change_forgets_the_thread() {
    let directory = tempfile::tempdir().unwrap();
    let origin = Origin::start(vec![origin::reply("First answer"), origin::reply("Second")]);
    let brain = Brain::start(directory.path(), origin.address, &[]);
    assert_eq!(
        brain
            .post("/v1/turns", &team::start("agent-9", "Hello"))
            .status,
        200
    );
    let body = team::start_with("agent-9", "Hi again", r#""assistants":[]"#);
    assert_eq!(brain.post("/v1/turns", &body).status, 200);
    assert!(
        !origin.received()[1].body.contains("First answer"),
        "another Assistant scope starts a fresh thread"
    );
}

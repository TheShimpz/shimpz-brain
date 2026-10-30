//! The rerun guard (`RerunGuardTests`).

use super::*;
use crate::tools::action::tool_name;

const ZONE: &str = "023e105f4ecef8ad9ca31a8372d0c353";
const FRESH: &str = r#"[{"assistant":"hello-pulse","action":"hello","count":1,"inputs":[{"member":"name",
"kind":"fresh","value":null,"chosen":false,"source":{"assistant":"hello-pulse","action":"list"}}]}]"#;

fn work(text: &str) -> Vec<Value> {
    match value(text) {
        Value::Array(items) => items,
        _ => unreachable!(),
    }
}

fn hello() -> String {
    tool_name("hello-pulse", "hello")
}

fn list() -> String {
    tool_name("hello-pulse", "list")
}

fn actions() -> BTreeSet<String> {
    [hello(), list()].into_iter().collect()
}

fn call(name: Value) -> Message {
    let tool = hello();
    turn(vec![(&tool, json::object([("name", name)]))])
}

/// An earlier exchange whose result held the zone, then this turn's message and what followed it.
fn this_turn(message: &str, later: Vec<Message>) -> Vec<Message> {
    let mut messages = vec![
        envelope("Liste a zona"),
        result(
            &list(),
            "old",
            &format!(r#"{{"zones": [{{"id": "{ZONE}"}}]}}"#),
        ),
        envelope(message),
    ];
    messages.extend(later);
    messages
}

fn review_of(messages: &[Message]) -> Option<String> {
    rerun_review(messages, &work(FRESH), &actions()).expect("reviewed")
}

#[test]
fn a_fresh_input_must_come_from_a_result_of_this_turn() {
    let remembered = || call(json::text(ZONE));
    let message = "pode buscar de novo";
    assert!(review_of(&this_turn(message, vec![remembered()])).is_some());
    for found in [
        format!(r#"{{"zones": [{{"id": "other"}}, {{"id": "{ZONE}", "tags": ["a"]}}]}}"#),
        ZONE.to_owned(),
    ] {
        let looked_up = result(&list(), "l1", &found);
        assert_eq!(
            review_of(&this_turn(message, vec![looked_up, remembered()])),
            None
        );
    }
    // A correction or a Brain tool's message is no result, whatever it says.
    for name in [TOOL_NAME, ""] {
        let other = result(name, "x1", ZONE);
        assert!(review_of(&this_turn(message, vec![other, remembered()])).is_some());
    }
    // Equal JSON values match as values: an integral float is its integer.
    let found = result(&list(), "l1", r#"{"id": 424242.0}"#);
    assert_eq!(
        review_of(&this_turn("sim", vec![found, call(json::int(424_242))])),
        None
    );
    assert!(review_of(&this_turn("sim", vec![call(json::int(424_242))])).is_some());
}

#[test]
fn a_value_the_person_writes_now_needs_no_lookup() {
    for (written, message) in [
        (json::text(ZONE), format!("use a zona {ZONE}")),
        (json::int(424_242), "a conta 424242".to_owned()),
        (json::int(-424_242), "saldo -424242".to_owned()),
    ] {
        assert_eq!(review_of(&this_turn(&message, vec![call(written)])), None);
    }
    // A number is written only as a whole token, never inside a longer number.
    for message in ["a conta 4242421", "a conta 424242.5", "saldo -424242"] {
        let messages = this_turn(message, vec![call(json::int(424_242))]);
        assert!(review_of(&messages).is_some(), "{message}");
    }
    // Only what the person wrote counts, never Team's composed question.
    let asked = format!("pode buscar\n\nPergunta: {ZONE}?\nResposta: sim");
    assert!(review_of(&this_turn(&asked, vec![call(json::text(ZONE))])).is_some());
}

#[test]
fn only_an_identifier_needs_a_source() {
    let (year, month, day) = crate::prompt::utc_date();
    let today = format!("{year:04}-{month:02}-{day:02}");
    for kept in [
        json::text("me"),
        json::text("my zone list"),
        json::int(42_424),
        value("4242.5"),
        Value::Bool(true),
        json::object([("id", json::text(ZONE))]),
        value(r#"["a"]"#),
        json::text(&today),
        json::text(&format!("news-{day:02}/{month:02}")),
        json::text(&format!("{month}月{day}日号")),
    ] {
        let messages = this_turn("sim", vec![call(kept.clone())]);
        assert_eq!(review_of(&messages), None, "{}", json::compact(&kept));
    }
    for asked in [
        json::text("zone-1"),
        json::int(-424_242),
        json::text("123e4567-e89b-12d3-a456-426614174000"),
        json::text("203.0.113.10"),
        json::text("v1.2.3"),
        json::text("release-2.1"),
        json::text("2026年度"),
        json::text(&format!("{}-01-01", year - 2)),
        json::text(&format!("news-{today}")),
    ] {
        let messages = this_turn("sim", vec![call(asked.clone())]);
        assert!(review_of(&messages).is_some(), "{}", json::compact(&asked));
    }
}

#[test]
fn only_a_fresh_member_of_a_checked_action_in_an_answerable_response_is_reviewed() {
    assert_eq!(review_of(&[]), None);
    assert_eq!(review_of(&this_turn("sim", vec![reply("Pronto.")])), None);
    let without = turn(vec![(&hello(), value("{}"))]);
    assert_eq!(review_of(&this_turn("sim", vec![without])), None);
    let other = turn(vec![(&list(), json::object([("name", json::text(ZONE))]))]);
    assert_eq!(review_of(&this_turn("sim", vec![other])), None);
    let unparsable = turn_with(
        vec![(&hello(), json::object([("name", json::text(ZONE))]))],
        vec!["x"],
    );
    assert_eq!(
        rerun_review(
            &this_turn("sim", vec![unparsable]),
            &work(FRESH),
            &actions()
        ),
        Err(BrainError::contract(refusal::UNPARSABLE))
    );
    // An Action checks only members every slot of it marks fresh.
    let kept = r#"{"member":"name","kind":"value","value":"\"ana\"","chosen":false,"source":null}"#;
    let twice = format!(
        r#"[{fresh},{{"assistant":"hello-pulse","action":"hello","count":1,"inputs":[{kept}]}}]"#,
        fresh = &FRESH[1..FRESH.len() - 1]
    );
    let messages = this_turn("sim", vec![call(json::text(ZONE))]);
    assert_eq!(rerun_review(&messages, &work(&twice), &actions()), Ok(None));
}

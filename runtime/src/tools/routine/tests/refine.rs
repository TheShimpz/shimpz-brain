//! Refining the Routine a person's send selects (`test_routine_refine.py`).

use super::*;

const SELECTED: &str = r#"{"routine_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","name":"Resumo diário",
"description":"Cumprimenta a Ana todo dia.","revision":2,"output":{"mode":"show","step":"s2"},"steps":[
{"id":"s1","position":1,"assistant":"hello-pulse","action":"hello","read_only":true,
"inputs":[{"member":"name","source":"literal","value":"\"ana\""}],"stored_inputs":[]},
{"id":"s2","position":2,"assistant":"hello-pulse","action":"hello","read_only":true,
"inputs":[{"member":"name","source":"step_output","step":1,"pointer":"/greeting","where":null,"item":null}],
"stored_inputs":[]}],"last_run":{"revision":2,"steps":[{"position":1,"status":"done"},
{"position":2,"status":"failed"}]}}"#;
const DESCRIPTION: &str = "Cumprimenta a Ana e o Bruno todo dia.";

fn selected_chat(selected: Value) -> TurnContext {
    TurnContext {
        routine_schedule: None,
        routine_selected: Some(selected),
        ..chat(vec![listed()])
    }
}

fn selection() -> TurnContext {
    selected_chat(value(SELECTED))
}

fn refine(changes: &[(&str, Value)]) -> Value {
    let mut members = vec![
        ("op".to_owned(), json::text("refine")),
        ("name".to_owned(), Value::Null),
        ("description".to_owned(), json::text(DESCRIPTION)),
        ("replaces".to_owned(), Value::Null),
        ("routine_id".to_owned(), json::text(&"a".repeat(32))),
        ("base_revision".to_owned(), json::int(2)),
        (
            "edits".to_owned(),
            value(r#"[{"kind":"insert","step":"s2","calls":["c1"]}]"#),
        ),
        ("reply".to_owned(), json::text("")),
    ];
    for (name, change) in changes {
        if let Some(slot) = members.iter_mut().find(|(existing, _)| existing == name) {
            slot.1 = change.clone();
        }
    }
    json::object_of(members)
}

fn labeled(handle: &str) -> Vec<Message> {
    let content = format!(r#"{{"call": "{handle}", "result": {{"greeting": "oi"}}}}"#);
    vec![user("m"), result("x", "a1", &content)]
}

fn handles(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[test]
fn a_selection_comes_only_in_routine_mode_beside_the_listing_in_its_canonical_form() {
    assert_eq!(refused(selection()), None);
    let with = |name: &str, member: Value| {
        let mut selected = value(SELECTED);
        if let Value::Object(members) = &mut selected {
            for (existing, slot) in members.iter_mut() {
                if existing == name {
                    *slot = member.clone();
                }
            }
        }
        selected_chat(selected)
    };
    let error = Some("invalid Routine selection");
    for context in [
        TurnContext {
            routine_mode: false,
            ..selection()
        },
        TurnContext {
            routine_selected: Some(value(SELECTED)),
            ..context()
        },
        with("revision", json::int(0)),
        with("revision", json::int(3)),
        with("routine_id", json::text(&"b".repeat(32))),
        with("routine_id", value("[]")),
        with("revision", value("{}")),
    ] {
        assert_eq!(refused(context).as_deref(), error);
    }
}

#[test]
fn a_refinement_of_the_selection_names_its_edits_and_this_turns_calls() {
    let found = call_handles(&labeled("c1"));
    assert_eq!(found, handles(&["c1"]));
    let outcome = record(&refine(&[]), &selection(), &found).expect("refined");
    assert_eq!(
        json::compact(outcome.get("routine").unwrap()),
        format!(
            r#"{{"routine_id":"{}","base_revision":2,"description":"{DESCRIPTION}","edits":[{{"kind":"insert","step":"s2","calls":["c1"]}}]}}"#,
            "a".repeat(32)
        )
    );
    assert_eq!(json::str_of(&outcome, "reply"), Some(""));
    let described = record(
        &refine(&[("edits", value("[]"))]),
        &selection(),
        &BTreeSet::new(),
    )
    .expect("refined");
    assert_eq!(
        described.get("routine").unwrap().get("edits"),
        Some(&value("[]"))
    );
    let both = value(
        r#"[{"kind":"remove","step":"s1","calls":[]},{"kind":"remove","step":"s2","calls":[]}]"#,
    );
    let removed = record(
        &refine(&[("edits", both.clone())]),
        &selection(),
        &BTreeSet::new(),
    );
    assert_eq!(
        removed
            .ok()
            .and_then(|outcome| outcome.get("routine")?.get("edits").cloned()),
        Some(both)
    );
}

/// Each invalid refinement and the field whose correction answers it.
fn invalid() -> Vec<(Value, &'static str)> {
    let edits = |text: &str| ("edits", value(text));
    vec![
        (
            refine(&[("routine_id", json::text(&"b".repeat(32)))]),
            "refine",
        ),
        (refine(&[("base_revision", json::int(1))]), "refine"),
        (refine(&[("base_revision", Value::Bool(true))]), "refine"),
        (refine(&[("name", json::text("Outro"))]), "refine"),
        (
            refine(&[("replaces", json::text(&"a".repeat(32)))]),
            "refine",
        ),
        (
            refine(&[edits(r#"[{"kind":"remove","step":"s9","calls":[]}]"#)]),
            "refine",
        ),
        (
            refine(&[edits(r#"[{"kind":"insert","step":"s1","calls":[]}]"#)]),
            "refine",
        ),
        (refine(&[("description", json::text(" "))]), "description"),
        (
            refine(&[edits(r#"[{"kind":"insert","step":"s2","calls":["c2"]}]"#)]),
            "calls",
        ),
        (
            refine(&[edits(r#"[{"kind":"remove","step":"s1","calls":[]}]"#)]),
            "dependency",
        ),
        (
            refine(&[edits(r#"[{"kind":"replace","step":"s1","calls":["c1"]}]"#)]),
            "dependency",
        ),
        (
            refine(&[
                edits("[]"),
                ("description", json::text("Cumprimenta a Ana todo dia.")),
            ]),
            "empty",
        ),
        (refine(&[("reply", json::text(" "))]), "reply"),
    ]
}

#[test]
fn each_invalid_refinement_gets_its_closed_correction() {
    for (arguments, refused) in invalid() {
        assert_eq!(
            record(&arguments, &selection(), &handles(&["c1"])).map(|_| ()),
            Err(refused),
            "{}",
            json::compact(&arguments)
        );
    }
}

#[test]
fn a_record_beside_a_selection_leaves_every_refine_member_null() {
    let none = BTreeSet::new();
    let record_with = |edits: Value, op: &str| {
        args(&[
            ("op", json::text(op)),
            ("routine_id", Value::Null),
            ("base_revision", Value::Null),
            ("edits", edits),
        ])
    };
    let recorded =
        record(&record_with(Value::Null, "record"), &selection(), &none).expect("record");
    assert_eq!(
        recorded
            .get("routine")
            .and_then(|routine| json::str_of(routine, "name")),
        Some("DNS de shimpz.com")
    );
    assert_eq!(
        record(&record_with(value("[]"), "record"), &selection(), &none).map(|_| ()),
        Err("refine")
    );
    assert_eq!(
        record(&record_with(Value::Null, "other"), &selection(), &none).map(|_| ()),
        Err("invalid")
    );
    assert_eq!(
        record(&refine(&[]), &chat(vec![listed()]), &none).map(|_| ()),
        Err("invalid")
    );
    assert_eq!(
        record(&args(&[]), &selection(), &none).map(|_| ()),
        Err("refine")
    );
}

#[test]
fn only_this_turns_named_results_are_handles() {
    let mut messages = vec![
        user("before"),
        result("x", "x", r#"{"call": "c9", "result": 1}"#),
    ];
    messages.extend(labeled("c2"));
    messages.extend([
        result("x", "b", r#"{"call": "c1"}"#),
        result("x", "c", r#"{"call": "x1", "result": 1}"#),
        result("x", "d", "not json"),
        reply("text"),
    ]);
    assert_eq!(call_handles(&messages), handles(&["c2"]));
    assert_eq!(call_handles(&[]), BTreeSet::new());
}

use super::*;
use crate::tools::testing::{context, envelope, reply, result, turn, turn_with, user, value};

mod refine;
mod rerun;

const REPLY: &str = "Listei os registros DNS de shimpz.com. Confira o cartão da rotina.";
const LISTED: &str = r#"{"routine_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","name":"Resumo diário",
"schedule":{"kind":"daily","time":"08:00"},"timezone":"America/Sao_Paulo","timezone_source":"browser","revision":2,
"daily_steps":1,"output":{"mode":"show"},"steps":[{"id":"s1","assistant":"hello-pulse","action":"hello",
"inputs":["name"]}]}"#;
const QUESTION: &str = r#"{"code":"routine-binding-ambiguous","options":[{"value":"\"023e105f\"",
"label":"shimpz.com"},{"value":"9007199254740993","label":null}],"value":null}"#;
const RERUN: &str = r#"[{"assistant":"hello-pulse","action":"hello","count":2,"inputs":[
{"member":"day","kind":"clock","value":null,"chosen":false,"source":null},
{"member":"name","kind":"value","value":"\"ana\"","chosen":true,"source":null},
{"member":"zone","kind":"fresh","value":null,"chosen":false,"source":{"assistant":"hello-pulse","action":"list"}}]}]"#;
const RERUN_QUESTION: &str = r#"{"code":"routine-work-rerun","options":[],"value":null}"#;
const OUTPUT_QUESTION: &str = r#"{"code":"routine-output-unstated","options":[],"value":null}"#;
const EVERY_30_SECONDS: &str = r#"{"kind":"continuous","gap":30}"#;

fn listed() -> Value {
    value(LISTED)
}

fn rerun_work() -> Vec<Value> {
    match value(RERUN) {
        Value::Array(items) => items,
        _ => unreachable!(),
    }
}

/// A Routine-mode chat with the Team's Routines and the schedule Team read from the person's words.
fn chat(routines: Vec<Value>) -> TurnContext {
    TurnContext {
        memories: Some(Vec::new()),
        routines: Some(routines),
        locale: Some("pt".into()),
        routine_mode: true,
        routine_schedule: Some(value(EVERY_30_SECONDS)),
        ..context()
    }
}

fn args(changes: &[(&str, Value)]) -> Value {
    let mut members = vec![
        ("op".to_owned(), json::text("record")),
        ("name".to_owned(), json::text("DNS de shimpz.com")),
        (
            "description".to_owned(),
            json::text("Lista as zonas da sua Cloudflare e lê os registros DNS de shimpz.com."),
        ),
        ("replaces".to_owned(), Value::Null),
        ("reply".to_owned(), json::text(REPLY)),
    ];
    for (name, change) in changes {
        match members.iter_mut().find(|(existing, _)| existing == name) {
            Some(slot) => slot.1 = change.clone(),
            None => members.push(((*name).to_owned(), change.clone())),
        }
    }
    json::object_of(members)
}

fn wire(replaces: Value) -> Value {
    json::object([
        ("name", json::text("DNS de shimpz.com")),
        (
            "description",
            json::text("Lista as zonas da sua Cloudflare e lê os registros DNS de shimpz.com."),
        ),
        ("replaces", replaces),
    ])
}

fn recorded_by(arguments: &Value, context: &TurnContext) -> Result<Value, &'static str> {
    record(arguments, context, &BTreeSet::new())
}

#[test]
fn a_valid_call_records_exactly_its_closed_outcome() {
    let chat = chat(vec![listed()]);
    let outcome = recorded_by(&args(&[]), &chat).expect("recorded");
    assert_eq!(
        outcome,
        json::object([("routine", wire(Value::Null)), ("reply", json::text(REPLY))])
    );
    let id = json::text(&"a".repeat(32));
    let changed = recorded_by(&args(&[("replaces", id.clone())]), &chat).expect("recorded");
    assert_eq!(changed.get("routine"), Some(&wire(id)));
    let empty = recorded_by(&args(&[("reply", json::text(""))]), &chat).expect("recorded");
    assert_eq!(json::str_of(&empty, "reply"), Some(""));
}

#[test]
fn the_schedule_timezone_and_output_are_never_the_models() {
    let chat = chat(vec![listed()]);
    let schema = spec(false).schema;
    for member in ["schedule", "timezone", "output"] {
        assert!(schema.get("properties").unwrap().get(member).is_none());
        assert_eq!(
            recorded_by(&args(&[(member, Value::Null)]), &chat),
            Err("invalid")
        );
    }
}

#[test]
fn every_field_outside_its_closed_shape_is_refused_by_name() {
    let chat = chat(vec![listed()]);
    let mut missing = args(&[]);
    if let Value::Object(members) = &mut missing {
        members.retain(|(name, _)| name != "reply");
    }
    let text = |text: &str| json::text(text);
    let cases = [
        (Value::Null, "invalid"),
        (args(&[("unexpected", text("x"))]), "invalid"),
        (missing, "invalid"),
        (args(&[("op", text("create"))]), "invalid"),
        (args(&[("name", text(""))]), "name"),
        (args(&[("name", text("a\nb"))]), "name"),
        (args(&[("name", text(&"x".repeat(81)))]), "name"),
        (args(&[("description", text(""))]), "description"),
        (
            args(&[("description", text("Lista as zonas.\nDepois lê o DNS."))]),
            "description",
        ),
        (
            args(&[("description", text(&"x".repeat(601)))]),
            "description",
        ),
        (args(&[("description", Value::Null)]), "description"),
        (args(&[("replaces", text(&"b".repeat(32)))]), "replaces"),
        (args(&[("reply", text("   "))]), "reply"),
        (args(&[("reply", text("a\u{0}b"))]), "reply"),
        (args(&[("reply", text("a\u{1}b"))]), "reply"),
        (args(&[("reply", text("a\u{7f}b"))]), "reply"),
        (args(&[("reply", text(&"x".repeat(4_001)))]), "reply"),
        (args(&[("reply", json::int(1))]), "reply"),
    ];
    for (arguments, refused) in cases {
        assert_eq!(recorded_by(&arguments, &chat), Err(refused), "{refused}");
        assert!(correction(refused).starts_with("Not done: nothing was recorded."));
    }
}

#[test]
fn a_review_reads_only_the_latest_routine_call() {
    let chat = chat(Vec::new());
    let action = || ("a_hello", value("{}"));
    assert_eq!(review(&[], &chat), Ok(Verdict::Pass));
    assert_eq!(review(&[envelope("Oi")], &chat), Ok(Verdict::Pass));
    assert_eq!(review(&[turn(vec![action()])], &chat), Ok(Verdict::Pass));
    let mixed = [turn(vec![(TOOL_NAME, args(&[])), action()])];
    assert_eq!(
        review(&mixed, &chat),
        Ok(Verdict::Refuse(correction("mixed")))
    );
    let Ok(Verdict::Record(content)) = review(&[turn(vec![(TOOL_NAME, args(&[]))])], &chat) else {
        panic!("a valid record ends the turn");
    };
    assert_eq!(
        content,
        format!(
            r#"{{"routine":{{"name":"DNS de shimpz.com","description":"Lista as zonas da sua Cloudflare e lê os registros DNS de shimpz.com.","replaces":null}},"reply":"{REPLY}"}}"#
        )
    );
    let ended = [result(TOOL_NAME, "call-0", &content)];
    assert_eq!(
        recorded(&ended),
        Some((REPLY.to_owned(), wire(Value::Null)))
    );
    let named = [turn(vec![(TOOL_NAME, args(&[("name", json::text(""))]))])];
    assert_eq!(
        review(&named, &chat),
        Ok(Verdict::Refuse(correction("name")))
    );
    let broken = [turn_with(vec![(TOOL_NAME, args(&[]))], vec!["a_hello"])];
    assert_eq!(
        review(&broken, &chat),
        Err(BrainError::contract(refusal::UNPARSABLE))
    );
}

#[test]
fn only_a_well_formed_ending_is_read_back() {
    let content = r#"{"routine": {"name": "DNS"}, "reply": "Ok."}"#;
    assert_eq!(
        recorded(&[result(TOOL_NAME, "r1", content)]),
        Some(("Ok.".into(), value(r#"{"name":"DNS"}"#)))
    );
    for messages in [
        vec![],
        vec![reply("Ok.")],
        vec![result("other", "r1", content)],
        vec![result(TOOL_NAME, "r1", "{bad")],
        vec![result(TOOL_NAME, "r1", "[1]")],
        vec![result(TOOL_NAME, "r1", r#"{"reply": "Ok."}"#)],
    ] {
        assert_eq!(recorded(&messages), None);
    }
}

#[test]
fn listed_routines_are_teams_own_protocol_form() {
    let mut admitted = chat(vec![listed()]);
    assert_eq!(admit_context(&mut admitted), Ok(()));
    let mut unzoned = listed();
    if let Value::Object(members) = &mut unzoned {
        for (name, member) in members.iter_mut() {
            match name.as_str() {
                "timezone" => *member = json::text("UTC"),
                "timezone_source" => *member = json::text("none"),
                _ => {}
            }
        }
    }
    assert_eq!(admit_context(&mut chat(vec![unzoned])), Ok(()));
    let with = |name: &str, member: Value| {
        let mut item = listed();
        if let Value::Object(members) = &mut item {
            members.retain(|(existing, _)| existing != name);
            members.push((name.to_owned(), member));
        }
        item
    };
    for routines in [
        vec![with("extra", json::int(1))],
        vec![with("timezone_source", json::text("guessed"))],
        vec![with("output", value(r#"{"mode":[]}"#))],
        vec![listed(), listed()],
    ] {
        assert_eq!(
            admit_context(&mut chat(routines)),
            Err(BrainError::contract("invalid routines"))
        );
    }
    let mut attached = chat(vec![listed()]);
    attached.attachments.kinds = vec!["text".into()];
    assert_eq!(
        admit_context(&mut attached),
        Err(BrainError::contract("invalid routines"))
    );
}

fn refused(context: TurnContext) -> Option<String> {
    let mut context = context;
    match admit_context(&mut context) {
        Ok(()) => None,
        Err(BrainError::Contract(detail)) => Some(detail),
        Err(other) => panic!("{other}"),
    }
}

/// Recording fields sent where Team never sends them, each with its refusal.
fn misplaced() -> Vec<(TurnContext, &'static str)> {
    let plain = context();
    vec![
        (
            TurnContext {
                routine_question: Some(value(QUESTION)),
                ..plain.clone()
            },
            "invalid Routine question",
        ),
        (
            TurnContext {
                routine_mode: true,
                ..plain.clone()
            },
            "invalid Routine mode",
        ),
        (
            TurnContext {
                routine_rerun: Some(rerun_work()),
                ..plain.clone()
            },
            "invalid Routine rerun",
        ),
        (
            TurnContext {
                routine_output: Some("changes".into()),
                ..plain.clone()
            },
            "invalid Routine output",
        ),
        (
            TurnContext {
                routine_schedule: Some(value(EVERY_30_SECONDS)),
                ..plain
            },
            "invalid Routine schedule",
        ),
    ]
}

#[test]
fn only_a_recording_chat_turn_carries_a_question_mode_rerun_or_output() {
    for (context, error) in misplaced() {
        assert_eq!(refused(context).as_deref(), Some(error));
    }
    let unknown = TurnContext {
        routine_question: Some(value(r#"{"code":"x","options":[],"value":null}"#)),
        ..chat(Vec::new())
    };
    assert_eq!(
        refused(unknown).as_deref(),
        Some("invalid Routine question")
    );
    let sometimes = TurnContext {
        routine_output: Some("sometimes".into()),
        ..chat(Vec::new())
    };
    assert_eq!(
        refused(sometimes).as_deref(),
        Some("invalid Routine output")
    );
    let asked = TurnContext {
        routine_question: Some(value(QUESTION)),
        ..chat(Vec::new())
    };
    assert_eq!(refused(asked), None);
    let output_only = TurnContext {
        routine_output: Some("changes".into()),
        routine_mode: false,
        routine_schedule: None,
        ..chat(Vec::new())
    };
    assert_eq!(refused(output_only), None);
}

fn recording() -> TurnContext {
    TurnContext {
        routine_question: Some(value(RERUN_QUESTION)),
        routine_rerun: Some(rerun_work()),
        routine_output: Some("changes".into()),
        ..chat(vec![listed()])
    }
}

#[test]
fn the_recording_fields_come_only_together_as_team_sends_them() {
    assert_eq!(refused(recording()), None);
    let cases = [
        (
            TurnContext {
                routine_mode: false,
                ..recording()
            },
            "invalid Routine question",
        ),
        (
            TurnContext {
                routine_mode: false,
                routine_question: None,
                routine_rerun: None,
                ..recording()
            },
            "invalid Routine schedule",
        ),
        (
            TurnContext {
                routine_question: Some(value(QUESTION)),
                ..recording()
            },
            "invalid Routine rerun",
        ),
        (
            TurnContext {
                routine_question: None,
                ..recording()
            },
            "invalid Routine rerun",
        ),
    ];
    for (context, error) in cases {
        assert_eq!(refused(context).as_deref(), Some(error));
    }
}

fn settings_answer() -> TurnContext {
    TurnContext {
        routine_question: Some(value(OUTPUT_QUESTION)),
        routine_output: Some("changes".into()),
        routine_settings_answer: true,
        ..chat(vec![listed()])
    }
}

#[test]
fn a_settings_answer_answers_a_settings_question_with_teams_reading() {
    assert_eq!(refused(settings_answer()), None);
    let schedule = value(r#"{"code":"routine-schedule-unstated","options":[],"value":null}"#);
    let over = value(r#"{"code":"routine-interval-over-budget","options":[],"value":120}"#);
    for question in [schedule.clone(), over.clone()] {
        let answered = TurnContext {
            routine_question: Some(question),
            ..settings_answer()
        };
        assert_eq!(refused(answered), None);
    }
    let error = Some("invalid Routine settings answer");
    for context in [
        TurnContext {
            routine_output: None,
            ..settings_answer()
        },
        TurnContext {
            routine_question: Some(value(QUESTION)),
            ..settings_answer()
        },
        TurnContext {
            routine_question: None,
            ..settings_answer()
        },
        TurnContext {
            routine_question: Some(schedule),
            routine_schedule: None,
            ..settings_answer()
        },
        TurnContext {
            routine_question: Some(over),
            routine_schedule: None,
            ..settings_answer()
        },
    ] {
        assert_eq!(refused(context).as_deref(), error);
    }
    assert!(settings_question(Some(&value(OUTPUT_QUESTION))));
    assert!(!settings_question(Some(&value(QUESTION))));
    assert!(!settings_question(None));
}

#[test]
fn the_tool_offers_refine_only_beside_a_selection() {
    let record = spec(false);
    let refine = spec(true);
    assert_eq!(record.name, TOOL_NAME);
    let required = |schema: &Value| json::compact(schema.get("required").unwrap());
    assert_eq!(
        required(&record.schema),
        r#"["op","name","description","replaces","reply"]"#
    );
    assert_eq!(
        required(&refine.schema),
        r#"["op","name","description","replaces","routine_id","base_revision","edits","reply"]"#
    );
    assert!(refine.description.starts_with(&record.description));
    assert_eq!(intervals(), "5 seconds to 366 days");
}

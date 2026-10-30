use super::*;
use crate::tools::testing::{CLOSED, context, value};
use crate::turn::context::ActionDefinition;

const LANGUAGE_LINE: &str = "Write every reply, clarification question, and option in ";

fn fixture(json: &str) -> TurnContext {
    TurnContext::from_value(&value(json)).expect("a fixture context")
}

#[test]
fn representative_prompts_are_byte_for_byte_what_the_python_brain_sent() {
    // A Routine-mode rerun turn with memory, the Browser, and a language; and an attachment turn reading content.
    for (context, expected) in [
        (
            include_str!("tests/routine.json"),
            include_str!("tests/routine.txt"),
        ),
        (
            include_str!("tests/attached.json"),
            include_str!("tests/attached.txt"),
        ),
    ] {
        assert_eq!(system_prompt(&fixture(context)), expected);
    }
}

#[test]
fn the_policy_prefers_a_stated_default_and_asks_only_open_decisions() {
    let prompt = system_prompt(&context());
    assert!(prompt.contains(CLARIFY_AND_COMPLETE));
    for rule in [
        "Prefer acting over asking",
        "Never ask for a missing detail that an available read-only Action can look up",
        "never as a free-text question or a numbered list",
        "After an Action has run, that tool is unavailable",
        "never ask for secrets",
        "cover every requirement the request implies",
    ] {
        assert!(CLARIFY_AND_COMPLETE.contains(rule), "{rule}");
    }
}

#[test]
fn the_turn_date_and_zone_close_the_prompt_after_every_stable_part() {
    let mut turn = context();
    turn.turn_date = "2026-09-29".into();
    turn.timezone = "America/Sao_Paulo".into();
    let prompt = system_prompt(&turn);
    assert!(prompt.ends_with(
        "\n\nCurrent date: 2026-09-29 (America/Sao_Paulo). When the user's local date could differ and it \
         matters, say which date you used."
    ));
    assert!(prompt.find("Enabled Assistant contracts") < prompt.find("Current date:"));
}

#[test]
fn today_is_the_current_utc_date() {
    let day = |seconds: u64| seconds / 86_400;
    let now = || {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };
    let before = day(now());
    let (year, month, date) = utc_date();
    let after = day(now());
    let ordinal = team_protocol::python::datetime::ymd_to_ord(year, month, date);
    let days = u64::try_from(i64::from(ordinal) - EPOCH_ORDINAL).unwrap();
    assert!(days == before || days == after);
    assert_eq!(today(), format!("{year:04}-{month:02}-{date:02}"));
    assert_eq!(
        ord_to_ymd(i32::try_from(EPOCH_ORDINAL).unwrap()),
        (1970, 1, 1)
    );
}

#[test]
fn a_selected_language_adds_one_line_just_before_the_date() {
    let mut turn = context();
    turn.locale = Some("ja".into());
    let prompt = system_prompt(&turn);
    let line = prompt
        .find(&format!(
            "{LANGUAGE_LINE}Japanese, the language the user selected in the interface"
        ))
        .expect("the language line");
    assert_eq!(prompt.matches(LANGUAGE_LINE).count(), 1);
    assert!(prompt.find("Enabled Assistant contracts").unwrap() < line);
    assert!(line < prompt.find("Current date:").unwrap());
    assert!(!system_prompt(&context()).contains(LANGUAGE_LINE));
}

#[test]
fn the_closed_languages_and_their_prompt_names() {
    for locale in CHAT_LOCALES {
        assert!(valid_locale(locale) && language_name(locale).is_some());
    }
    assert_eq!(language_name("pt"), Some("Brazilian Portuguese"));
    assert_eq!(language_name("zh"), Some("Simplified Chinese"));
    for refused in ["pt-BR", "PT", "", "Portuguese"] {
        assert!(!valid_locale(refused) && language_name(refused).is_none());
    }
}

#[test]
fn memories_are_quoted_between_the_contracts_and_the_date() {
    let mut turn = context();
    turn.memories = Some(vec![Memory {
        topic: "language".into(),
        preference: "responda sempre em português do Brasil".into(),
    }]);
    let prompt = system_prompt(&turn);
    let section = prompt.find("What you remember about this user").unwrap();
    assert!(prompt.find("Enabled Assistant contracts").unwrap() < section);
    assert!(section < prompt.find("Current date:").unwrap());
    assert!(prompt.contains(
        r#"[{"topic": "language", "preference": "responda sempre em português do Brasil"}]"#
    ));
    assert!(prompt.contains("Never remember secrets") && prompt.contains("shimpz_memory"));
    // A turn that reads attachments sees its memories but is never told to change them.
    turn.attachments.kinds = vec!["text".into()];
    let attached = system_prompt(&turn);
    assert!(attached.contains("What you remember") && !attached.contains("shimpz_memory"));
    assert!(!system_prompt(&context()).contains("shimpz_memory"));
}

#[test]
fn contracts_are_canonical_and_name_an_actions_file_input_only_where_declared() {
    let mut turn = context();
    turn.assistants[0].actions.push(ActionDefinition {
        input_files: vec!["document".into()],
        ..crate::tools::testing::action("read")
    });
    let prompt = system_prompt(&turn);
    assert!(prompt.contains(
        r#"[{"actions":[{"id":"hello","summary":"Run hello."},{"file_input":"document","id":"read","summary":"Run read."}],"genesis":"Greets people.","id":"hello-pulse"}]"#
    ));
    assert_eq!(value(CLOSED), turn.assistants[0].actions[0].input_schema);
}

#[test]
fn an_empty_scope_is_named_only_when_nothing_is_offered() {
    const EMPTY: &str = "This turn has no enabled Assistants";
    let mut turn = context();
    assert!(!system_prompt(&turn).contains(EMPTY));
    turn.assistants.clear();
    assert!(system_prompt(&turn).contains(EMPTY));
    turn.browser = true;
    let browsing = system_prompt(&turn);
    assert!(!browsing.contains(EMPTY) && browsing.contains(BROWSER_SECTION));
    turn.browser = false;
    turn.routines = Some(Vec::new());
    turn.routine_settings_answer = true;
    // The Actions a settings answer is not offered are no limit of the Team's.
    assert!(!system_prompt(&turn).contains(EMPTY));
}

fn routine_chat() -> TurnContext {
    TurnContext {
        routines: Some(Vec::new()),
        routine_mode: true,
        ..context()
    }
}

#[test]
fn each_routine_answer_is_told_what_it_runs() {
    let output = value(r#"{"code":"routine-output-unstated","options":[],"value":null}"#);
    let target = value(r#"{"code":"routine-binding-ambiguous","options":[],"value":null}"#);
    let records_only = TurnContext {
        routine_question: Some(output.clone()),
        routine_settings_answer: true,
        ..routine_chat()
    };
    let typed = TurnContext {
        routine_question: Some(output),
        ..routine_chat()
    };
    let answered = TurnContext {
        routine_question: Some(target),
        ..routine_chat()
    };
    let records = system_prompt(&records_only);
    assert!(!records.contains("run the work once") && records.contains(ANSWER_RECORDS_ONLY));
    assert!(system_prompt(&typed).contains(ANSWER_SETTINGS));
    assert!(!system_prompt(&typed).contains("run the work once"));
    assert!(
        system_prompt(&answered).contains("Then run the work once and call shimpz_routine record.")
    );
    assert!(system_prompt(&answered).contains(ANSWER_OTHER));
}

#[test]
fn a_selected_routine_is_refined_and_otherwise_the_turn_asks_which() {
    let selected = TurnContext {
        routine_selected: Some(value(r#"{"routine_id":"a"}"#)),
        ..routine_chat()
    };
    let prompt = system_prompt(&selected);
    assert!(prompt.contains(
        "To change the selected Routine, refine it as described above. For a new Routine, then run the work once \
         and call shimpz_routine record."
    ));
    assert!(prompt.contains("op refine"));
    let unselected = system_prompt(&routine_chat());
    assert!(
        !unselected.contains("op refine")
            && unselected.contains("ask which listed Routine they mean")
    );
}

#[test]
fn a_repeated_action_is_listed_once_with_its_count() {
    let step = |action: &str| {
        value(&format!(
            r#"{{"assistant":"hello-pulse","action":"{action}"}}"#
        ))
    };
    let steps = [
        step("hello"),
        step("hello"),
        step("hello"),
        step("list"),
        step("hello"),
    ];
    assert_eq!(
        json::compact(&runs(&steps)),
        r#"[["hello-pulse","hello",3],["hello-pulse","list"],["hello-pulse","hello"]]"#
    );
    assert_eq!(json::compact(&runs(&[])), "[]");
}

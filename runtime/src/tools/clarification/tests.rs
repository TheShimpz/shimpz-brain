use super::*;
use crate::errors::BrainError;
use crate::tools::refusal::testing::{reply, result, turn, turn_with, user, value};

const VALID: &str = r#"{"question":"Qual período você quer cobrir?","options":[{"label":"Hoje",
"description":"Só lançamentos de hoje."},{"label":"Esta semana","description":""}],"default_index":0}"#;

fn with(member: &str, replacement: &str) -> Value {
    let mut asked = value(VALID);
    if let Value::Object(members) = &mut asked {
        members.retain(|(name, _)| name != member);
        if !replacement.is_empty() {
            members.push((member.into(), value(replacement)));
        }
    }
    asked
}

#[test]
fn the_closed_shape_parses_and_renders() {
    let parsed = Clarification::parse(&value(VALID)).expect("valid");
    assert_eq!(parsed.to_value(), value(VALID));
    assert_eq!(
        parsed.render(),
        "Qual período você quer cobrir?\n\n1. Hoje ✓ — Só lançamentos de hoje.\n2. Esta semana"
    );
}

#[test]
fn nothing_else_parses() {
    let mut extra = value(VALID);
    if let Value::Object(members) = &mut extra {
        members.push(("extra".into(), json::int(1)));
    }
    let long_label = format!(
        r#"[{{"label":"{}","description":""}},{{"label":"B","description":""}}]"#,
        "x".repeat(81)
    );
    let six: Vec<String> = (0..6)
        .map(|index| format!(r#"{{"label":"O{index}","description":""}}"#))
        .collect();
    let broken = [
        Value::Null,
        extra,
        with("question", r#""Linha 1\nLinha 2""#),
        with("question", r#""   ""#),
        with("question", "5"),
        with("question", &format!(r#""{}""#, "x".repeat(241))),
        with("options", r#"[{"label":"Hoje","description":""}]"#),
        with("options", &format!("[{}]", six.join(","))),
        with(
            "options",
            r#"[{"label":"Hoje","description":""},{"label":"hoje","description":""}]"#,
        ),
        with(
            "options",
            r#"[{"label":"Hoje","description":""},{"label":"B\u200b","description":""}]"#,
        ),
        with(
            "options",
            r#"[{"label":"Hoje"},{"label":"B","description":""}]"#,
        ),
        with("options", &long_label),
        with("options", r#""Hoje, Semana""#),
        with("options", r#"["Hoje","Semana"]"#),
        with("default_index", "2"),
        with("default_index", "true"),
        with("default_index", r#""0""#),
        with("default_index", "null"),
        with("default_index", "-1"),
        with("default_index", ""),
    ];
    for asked in broken {
        assert_eq!(
            Clarification::parse(&asked),
            None,
            "{}",
            json::compact(&asked)
        );
    }
}

#[test]
fn a_routine_turns_clarification_renders_without_a_recommendation() {
    let mut asked = Clarification::parse(&value(VALID)).expect("valid");
    asked.default_index = None;
    assert_eq!(asked.to_value(), with("default_index", "null"));
    assert!(!asked.render().contains('✓'));
}

#[test]
fn recorded_requires_the_tool_result_to_end_the_turn() {
    let call = turn(vec![(TOOL_NAME, value(VALID))]);
    let ended = result(TOOL_NAME, "call-0", RECORDED);
    let asked = recorded(&[call.clone(), ended.clone()]).expect("recorded");
    assert_eq!(asked.question, "Qual período você quer cobrir?");
    assert_eq!(recorded(std::slice::from_ref(&ended)), None);
    assert_eq!(recorded(&[call.clone(), reply("done")]), None);
    assert_eq!(recorded(&[user("x"), ended.clone()]), None);
    let double = turn(vec![(TOOL_NAME, value(VALID)), (TOOL_NAME, value(VALID))]);
    assert_eq!(recorded(&[double, ended]), None);
}

fn ask(question: &str, labels: &[&str]) -> Vec<Message> {
    let options: Vec<Value> = labels
        .iter()
        .map(|label| {
            json::object([
                ("label", json::text(label)),
                ("description", json::text("")),
            ])
        })
        .collect();
    let arguments = json::object([
        ("question", json::text(question)),
        ("options", json::array(options)),
        ("default_index", json::int(0)),
    ]);
    vec![user("Oi"), turn(vec![(TOOL_NAME, arguments)])]
}

#[test]
fn the_guard_refuses_a_mixed_late_or_malformed_clarification() {
    let mixed = [
        user("Oi"),
        turn(vec![(TOOL_NAME, value(VALID)), ("a_tool", value("{}"))]),
    ];
    assert_eq!(review(&mixed, true, false, None), Ok(Some(MIXED.into())));
    let alone = [user("Oi"), turn(vec![(TOOL_NAME, value(VALID))])];
    assert_eq!(review(&alone, true, false, None), Ok(None));
    assert_eq!(
        review(&alone, false, false, None),
        Ok(Some(AFTER_ACTION.into()))
    );
    let malformed = [
        user("Oi"),
        turn(vec![(TOOL_NAME, with("default_index", "9"))]),
    ];
    assert_eq!(
        review(&malformed, true, false, None),
        Ok(Some(INVALID.into()))
    );
    assert_eq!(
        review(&[user("Oi"), reply("Olá")], true, false, None),
        Ok(None)
    );
    assert!(MIXED.contains("cannot be combined") && AFTER_ACTION.contains("an Action already ran"));
}

#[test]
fn a_clarification_beside_an_unparsable_call_is_a_contract_error() {
    let unparsable = Err(BrainError::contract(refusal::UNPARSABLE));
    for response in [
        turn_with(vec![("a_tool", value("{}"))], vec![TOOL_NAME]),
        turn_with(Vec::new(), vec![TOOL_NAME]),
        turn_with(vec![(TOOL_NAME, value(VALID))], vec!["a_tool"]),
    ] {
        assert_eq!(
            review(&[user("Oi"), response], true, false, None),
            unparsable
        );
    }
}

#[test]
fn routine_mode_refuses_what_the_team_asks_itself() {
    let refused = Ok(Some(TEAM_OWNED.to_owned()));
    for messages in [
        ask("Com que frequência a rotina deve rodar?", &["Sim", "Não"]),
        ask("How often should the Routine run?", &["Sim", "Não"]),
        ask(
            "Como você quer receber o resultado de cada execução?",
            &["Sim", "Não"],
        ),
        ask(
            "What should happen with each run's result?",
            &["Show every run", "Only on change"],
        ),
        ask(
            "Qual zona e quando?",
            &["A cada 30 segundos", "Todo dia às 9h"],
        ),
        ask(
            "Quantas vezes por dia a rotina pode rodar?",
            &["Sim", "Não"],
        ),
    ] {
        assert_eq!(review(&messages, true, true, None), refused);
    }
    for question in [
        "Qual zona devo consultar?",
        "Which zone should I use?",
        "O que a rotina deve fazer?",
    ] {
        let messages = ask(question, &["shimpz.com", "example.com"]);
        assert_eq!(review(&messages, true, true, None), Ok(None));
    }
    let schedule = ask("Com que frequência a rotina deve rodar?", &["Sim", "Não"]);
    assert_eq!(review(&schedule, true, false, None), Ok(None));
}

#[test]
fn a_schedule_question_is_read_with_only_the_chats_negations() {
    let question = ask(
        "Qual zona você quer consultar no Cloudflare a cada 30 segundos?",
        &["Sim", "Não"],
    );
    assert!(matches!(
        review(&question, true, true, Some("pt")),
        Ok(Some(_))
    ));
    assert_eq!(review(&question, true, true, None), Ok(None));
    let plain = ask("Devo listar os registros no Cloudflare?", &["Sim", "Não"]);
    assert_eq!(review(&plain, true, true, Some("pt")), Ok(None));
    let times = ["09:00", "18:00"];
    let start = ask("Which start time should the report query use?", &times);
    assert_eq!(review(&start, true, true, None), Ok(None));
    assert!(matches!(
        review(&ask("What time?", &times), true, true, None),
        Ok(Some(_))
    ));
}

#[test]
fn the_tool_is_offered_exactly_as_before() {
    let tool = spec();
    assert_eq!(tool.name, "shimpz_clarify");
    assert_eq!(tool.strict, None);
    assert_eq!(json::compact(&tool.schema), SCHEMA);
}

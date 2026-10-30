use super::*;

pub(crate) fn sample() -> TurnContext {
    TurnContext {
        thread_id: "thread-1".into(),
        team_name: "Fixture Team".into(),
        assistants: vec![AssistantDefinition {
            id: "dns".into(),
            genesis: "DNS manages zones.".into(),
            actions: vec![ActionDefinition {
                id: "list-zones".into(),
                summary: "List zones.".into(),
                input_schema: json::parse(
                    br#"{"type":"object","properties":{},"additionalProperties":false}"#,
                )
                .unwrap(),
                output_schema: json::parse(br#"{"type":"object","additionalProperties":false}"#)
                    .unwrap(),
                authorization: false,
                input_files: vec![],
            }],
        }],
        provider: ProviderId::OpenAi,
        model: "gpt-6-luna".into(),
        turn_date: "2026-10-30".into(),
        timezone: "UTC".into(),
        memories: Some(vec![Memory {
            topic: "tone".into(),
            preference: "Be brief.".into(),
        }]),
        routines: None,
        routine_question: None,
        routine_mode: false,
        routine_rerun: None,
        routine_schedule: None,
        routine_output: None,
        routine_settings_answer: false,
        routine_selected: None,
        locale: Some("pt".into()),
        turn_message_id: Some(format!("shimpz-turn-{}", "0".repeat(32))),
        attachments: AttachmentPins::default(),
        browser: true,
    }
}

#[test]
fn a_context_round_trips_through_its_journal_form_and_nothing_else_decodes() {
    let context = sample();
    let value = context.to_value();
    assert_eq!(TurnContext::from_value(&value), Some(context));
    let encoded = json::compact(&value);
    assert_eq!(
        TurnContext::from_value(&json::parse(encoded.as_bytes()).unwrap()),
        Some(sample())
    );
    assert!(!encoded.contains("api_key"));
    let Value::Object(mut members) = value else {
        unreachable!()
    };
    members.push(("extra".into(), Value::Null));
    assert_eq!(TurnContext::from_value(&Value::Object(members)), None);
}

#[test]
fn attachment_pins_tell_whether_content_is_read() {
    let mut pins = AttachmentPins::default();
    assert!(!pins.any() && !pins.reads_content());
    pins.kinds = vec!["opaque".into()];
    assert!(pins.any() && !pins.reads_content());
    pins.kinds.push("image".into());
    assert!(pins.reads_content());
}

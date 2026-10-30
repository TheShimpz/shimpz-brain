//! Team's request bodies for the black-box tests: one Team with one DNS Assistant on OpenAI.

use std::fmt::Write as _;

/// The provider key the tests send (low entropy on purpose).
pub const KEY: &str = "test-provider-key-0000";

/// The DNS Action's tool name.
#[must_use]
pub fn list_zones() -> String {
    brain_runtime::tools::action::tool_name("dns", "list-zones")
}

/// The fields every turn request carries.
#[must_use]
pub fn context(thread: &str) -> String {
    let mut fields = String::new();
    let _ = write!(
        fields,
        r#""thread_id":"{thread}","team_name":"Fixture Team","assistants":[{{"id":"dns","genesis":"DNS manages the user's zones.","actions":[{{"id":"list-zones","summary":"List the user's DNS zones.","input_schema":{{"type":"object","properties":{{}},"additionalProperties":false}},"output_schema":{{"type":"object","properties":{{}},"additionalProperties":false}},"authorization":false,"input_files":[]}}]}}],"provider":{{"provider":"openai","model":"gpt-6-luna","api_key":"{KEY}"}},"memories":null,"routines":null,"routine_question":null,"routine_mode":false,"routine_rerun":null,"routine_schedule":null,"routine_output":null,"routine_settings_answer":false,"routine_selected":null,"attachments":[],"browser_available":false"#
    );
    fields
}

/// A start of `message`.
#[must_use]
pub fn start(thread: &str, message: &str) -> String {
    let envelope = format!(r#"{{\"files\":[],\"message\":\"{message}\"}}"#);
    format!(
        r#"{{{},"message":"{envelope}","locale":null,"conversation":[]}}"#,
        context(thread)
    )
}

/// A resume answering `interrupt_id` with `result` (JSON text).
#[must_use]
pub fn resume(thread: &str, interrupt_id: &str, result: &str) -> String {
    format!(
        r#"{{{},"results":{{"{interrupt_id}":{result}}}}}"#,
        context(thread)
    )
}

/// The one `interrupt_id` of an `action-required` response body.
#[must_use]
pub fn interrupt_id(body: &str) -> String {
    let value = brain_runtime::json::parse(body.as_bytes()).expect("JSON response");
    let actions = brain_runtime::json::array_of(&value, "actions").expect("actions");
    brain_runtime::json::str_of(&actions[0], "interrupt_id")
        .expect("interrupt id")
        .to_owned()
}

/// The `status` of a turn response body.
#[must_use]
pub fn status(body: &str) -> String {
    let value = brain_runtime::json::parse(body.as_bytes()).expect("JSON response");
    brain_runtime::json::str_of(&value, "status")
        .expect("status")
        .to_owned()
}

/// A start of `message` with extra context members replacing the defaults (`"key":value` pairs, comma-separated).
#[must_use]
pub fn start_with(thread: &str, message: &str, overrides: &str) -> String {
    let base = start(thread, message);
    let mut value = brain_runtime::json::parse(base.as_bytes()).expect("start body");
    let extra =
        brain_runtime::json::parse(format!("{{{overrides}}}").as_bytes()).expect("overrides");
    if let (
        brain_runtime::json::Value::Object(members),
        brain_runtime::json::Value::Object(extra),
    ) = (&mut value, extra)
    {
        for (name, item) in extra {
            match members.iter_mut().find(|(member, _)| *member == name) {
                Some(slot) => slot.1 = item,
                None => members.push((name, item)),
            }
        }
    }
    brain_runtime::json::compact(&value)
}

/// The same resume with extra context members replacing the defaults.
#[must_use]
pub fn resume_with(thread: &str, interrupt_id: &str, result: &str, overrides: &str) -> String {
    let base = resume(thread, interrupt_id, result);
    let mut value = brain_runtime::json::parse(base.as_bytes()).expect("resume body");
    let extra =
        brain_runtime::json::parse(format!("{{{overrides}}}").as_bytes()).expect("overrides");
    if let (
        brain_runtime::json::Value::Object(members),
        brain_runtime::json::Value::Object(extra),
    ) = (&mut value, extra)
    {
        for (name, item) in extra {
            if let Some(slot) = members.iter_mut().find(|(member, _)| *member == name) {
                slot.1 = item;
            }
        }
    }
    brain_runtime::json::compact(&value)
}

/// A response body's member.
#[must_use]
pub fn member(body: &str, key: &str) -> brain_runtime::json::Value {
    let value = brain_runtime::json::parse(body.as_bytes()).expect("JSON response");
    value.get(key).cloned().expect("member")
}

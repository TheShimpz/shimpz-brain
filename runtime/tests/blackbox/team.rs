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

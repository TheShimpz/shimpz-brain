//! Action tools (`action_tool.py`): one provider-safe tool per declared Action; a call with schema-valid arguments
//! becomes an Action request Team executes, any other call gets a closed correction and never reaches Team.

use crate::json::Value;
use crate::provider::ToolSpec;
use crate::turn::context::ActionDefinition;

/// The stable provider-safe tool name of one Assistant's Action (`agent_runtime.tool_name`).
#[must_use]
pub fn tool_name(_assistant_id: &str, _action_id: &str) -> String {
    unimplemented!("ported by the tools microtask")
}

/// The tool offered for `action`: its description, its input schema, and `strict: false` (ADR-0094).
#[must_use]
pub fn spec(_tool_name: &str, _assistant_id: &str, _action: &ActionDefinition) -> ToolSpec {
    unimplemented!("ported by the tools microtask")
}

/// Whether `arguments` satisfy the Action's input schema within one payload's pattern work budget.
#[must_use]
pub fn valid(_action: &ActionDefinition, _arguments: &Value) -> bool {
    unimplemented!("ported by the tools microtask")
}

/// The correction a call with invalid arguments receives.
#[must_use]
pub fn correction(_tool_name: &str, _action: &ActionDefinition) -> String {
    unimplemented!("ported by the tools microtask")
}

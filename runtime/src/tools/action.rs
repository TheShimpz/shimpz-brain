//! Action tools (`action_tool.py`): one provider-safe tool per declared Action; a call with schema-valid arguments
//! becomes an Action request Team executes, any other call gets a closed correction and never reaches Team.

mod wire;

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::json::Value;
use crate::provider::ToolSpec;
use crate::turn::context::ActionDefinition;

/// The most characters a correction has.
const MAX_CORRECTION_CHARS: usize = 1_000;
/// The most required property names a correction lists.
const MAX_LISTED_NAMES: usize = 16;
/// The most characters of each id a tool name keeps.
const SLUG_CHARS: usize = 18;

/// The stable provider-safe tool name of one Assistant's Action (`agent_runtime.tool_name`).
#[must_use]
pub fn tool_name(assistant_id: &str, action_id: &str) -> String {
    let slug = |id: &str| {
        id.replace('.', "_")
            .chars()
            .take(SLUG_CHARS)
            .collect::<String>()
    };
    let digest = Sha256::digest(format!("{assistant_id}\0{action_id}").as_bytes());
    let mut hex = String::with_capacity(16);
    for byte in &digest[..8] {
        let _ = write!(hex, "{byte:02x}");
    }
    format!("a_{}__a_{}__{hex}", slug(assistant_id), slug(action_id))
}

/// The tool offered for `action`: its description, its input schema, and `strict: false` (ADR-0094).
///
/// The schema is the one `LangChain` sent for the Python Brain's Action tools: references inlined, the root's
/// definitions, title, and description dropped, and titles removed (`convert_to_openai_tool`).
#[must_use]
pub fn spec(tool_name: &str, assistant_id: &str, action: &ActionDefinition) -> ToolSpec {
    ToolSpec {
        name: tool_name.to_owned(),
        description: format!(
            "Internal Assistant {assistant_id}, Action {}: {}",
            action.id, action.summary
        ),
        schema: wire::parameters(&action.input_schema),
        strict: Some(false),
    }
}

/// Whether `arguments` satisfy the Action's input schema within one payload's pattern work budget.
///
/// A reference that cannot be resolved, a payload nested too deeply, or matching beyond its budget cannot complete
/// and is refused like any invalid argument; Team still validates every request it receives.
#[must_use]
pub fn valid(action: &ActionDefinition, arguments: &Value) -> bool {
    action_schema::validate_payload(&action.input_schema, arguments).is_ok()
}

/// Whether a property name is a plain identifier the correction may show (`_PLAIN_NAME`).
fn plain(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && name.len() <= 64
        && characters.all(|rest| rest.is_ascii_alphanumeric() || rest == '_' || rest == '-')
}

/// The correction a call with invalid arguments receives.
#[must_use]
pub fn correction(tool_name: &str, action: &ActionDefinition) -> String {
    let names: Vec<&Value> = match action.input_schema.get("required") {
        Some(Value::Array(items)) => items.iter().take(MAX_LISTED_NAMES).collect(),
        _ => Vec::new(),
    };
    let plain_names: Option<Vec<&str>> = names
        .iter()
        .map(|name| match name {
            Value::String(text) if plain(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let listed = match plain_names {
        Some(listed) if !listed.is_empty() => {
            format!("every required property ({})", listed.join(", "))
        }
        _ => "every required property".to_owned(),
    };
    format!(
        "Action not executed: the arguments for {tool_name} did not match its input schema. Call it again with only \
         its declared properties, {listed}, and values of the declared types and allowed values."
    )
    .chars()
    .take(MAX_CORRECTION_CHARS)
    .collect()
}

#[cfg(test)]
mod tests;

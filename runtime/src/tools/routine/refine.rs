//! Refining the Routine the person's send selects (ADR-0122): edits of its own steps by id and of this turn's calls by
//! their Team handles. Team checks every edit again against its own trace; this only lets the agent correct itself.

use std::collections::BTreeSet;

use team_protocol::python::value::py_eq;
use team_protocol::routine::canonical_description;
use team_protocol::routine_context::canonical_edits;

use super::reply;
use crate::json::{self, Number, Value};

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value.get(name).unwrap_or(&Value::Null)
}

fn steps(selected: &Value) -> &[Value] {
    json::array_of(selected, "steps").unwrap_or_default()
}

fn edits_of(edits: &Value) -> &[Value] {
    match edits {
        Value::Array(items) => items,
        _ => &[],
    }
}

/// A valid refinement of the selected Routine, or the field whose correction answers it (`_refined`).
pub(super) fn refined(
    arguments: &Value,
    selected: &Value,
    handles: &BTreeSet<String>,
) -> Result<Value, &'static str> {
    let edits = canonical_edits(field(arguments, "edits"));
    // Only edits of the selected Routine's own steps are read further.
    let named = edits.filter(|edits| names_selection(arguments, selected, edits));
    let description = field(arguments, "description");
    let reply = reply(field(arguments, "reply"));
    let Some(edits) = named else {
        return Err("refine");
    };
    canonical_description(description).ok_or("description")?;
    let calls_named = edits_of(edits)
        .iter()
        .flat_map(|edit| json::array_of(edit, "calls").unwrap_or_default())
        .all(|call| matches!(call, Value::String(handle) if handles.contains(handle)));
    if !calls_named {
        return Err("calls");
    }
    if read_after(selected, edits) {
        return Err("dependency");
    }
    if edits_of(edits).is_empty() && py_eq(description, field(selected, "description")) {
        return Err("empty");
    }
    let reply = reply.ok_or("reply")?;
    let refine = json::object([
        ("routine_id", field(selected, "routine_id").clone()),
        ("base_revision", field(selected, "revision").clone()),
        ("description", description.clone()),
        ("edits", edits.clone()),
    ]);
    Ok(json::object([
        ("routine", refine),
        ("reply", json::text(reply)),
    ]))
}

/// Whether a refine names exactly the selected Routine as shown, no record member, and only its own step ids
/// (`_names_selection`).
fn names_selection(arguments: &Value, selected: &Value, edits: &Value) -> bool {
    let revision = field(arguments, "base_revision");
    let step_ids: Vec<&Value> = steps(selected)
        .iter()
        .map(|step| field(step, "id"))
        .collect();
    py_eq(
        field(arguments, "routine_id"),
        field(selected, "routine_id"),
    ) && matches!(revision, Value::Number(Number::Int(_) | Number::BigInt(_)))
        && py_eq(revision, field(selected, "revision"))
        && *field(arguments, "name") == Value::Null
        && *field(arguments, "replaces") == Value::Null
        && edits_of(edits)
            .iter()
            .all(|edit| step_ids.iter().any(|id| py_eq(id, field(edit, "step"))))
}

/// Whether a step the edits keep reads a step they remove or replace, by its plan position (`_read_after`).
fn read_after(selected: &Value, edits: &Value) -> bool {
    let position = |id: &Value| {
        steps(selected)
            .iter()
            .find(|step| py_eq(field(step, "id"), id))
            .map(|step| field(step, "position"))
    };
    let gone: Vec<&Value> = edits_of(edits)
        .iter()
        .filter(|edit| matches!(json::str_of(edit, "kind"), Some("remove" | "replace")))
        .filter_map(|edit| position(field(edit, "step")))
        .collect();
    let removed = |value: &Value| gone.iter().any(|position| py_eq(position, value));
    steps(selected)
        .iter()
        .filter(|step| !removed(field(step, "position")))
        .flat_map(|step| json::array_of(step, "inputs").unwrap_or_default())
        .any(|item| {
            json::str_of(item, "source") == Some("step_output") && removed(field(item, "step"))
        })
}

//! The Action input schema as `LangChain` sent it to both providers (`convert_to_openai_tool` of a tool whose
//! `args_schema` is the schema): `dereference_refs` in its default shallow mode, the root's `definitions`, `$defs`,
//! `title`, and `description` dropped, then `_rm_titles`.
//!
//! Admission (`action_schema::admit_schema`) already refuses cycles and bounds the expansion, so inlining stays
//! bounded. A reference `LangChain` could not resolve failed the whole Python turn; here it stays as written.

use crate::json::Value;

/// The `parameters` the providers receive for an Action's input schema.
pub(super) fn parameters(schema: &Value) -> Value {
    let mut resolved = inline(schema, schema, &mut Vec::new());
    if let Value::Object(members) = &mut resolved {
        members.retain(|(name, _)| {
            !matches!(
                name.as_str(),
                "definitions" | "$defs" | "title" | "description"
            )
        });
    }
    without_titles(&resolved, "")
}

/// The value a `#/...` reference names, splitting on `/` as `LangChain` does (`_retrieve_ref`).
fn retrieve<'a>(path: &str, schema: &'a Value) -> Option<&'a Value> {
    let mut components = path.split('/');
    if components.next() != Some("#") {
        return None;
    }
    components.try_fold(schema, |node, component| match node {
        Value::Object(_) => node.get(component),
        Value::Array(items) if component.bytes().all(|byte| byte.is_ascii_digit()) => component
            .parse::<usize>()
            .ok()
            .and_then(|index| items.get(index)),
        _ => None,
    })
}

/// `_dereference_refs_helper` with `skip_keys = ["$defs"]`.
fn inline(node: &Value, root: &Value, processing: &mut Vec<String>) -> Value {
    match node {
        Value::Object(members) => match node.get("$ref") {
            Some(Value::String(path)) => reference(node, path, members, root, processing),
            _ => Value::Object(properties(members, root, processing)),
        },
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| inline(item, root, processing))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// One object holding `$ref`: the resolved target merged with the object's other members, which win.
fn reference(
    node: &Value,
    path: &str,
    members: &[(String, Value)],
    root: &Value,
    processing: &mut Vec<String>,
) -> Value {
    let additional: Vec<(String, Value)> = members
        .iter()
        .filter(|(name, _)| name != "$ref")
        .cloned()
        .collect();
    if processing.iter().any(|seen| seen == path) {
        return Value::Object(properties(&additional, root, processing));
    }
    let Some(target) = retrieve(path, root) else {
        return node.clone();
    };
    processing.push(path.to_owned());
    let resolved = inline(target, root, processing);
    processing.pop();
    if additional.is_empty() {
        return resolved;
    }
    let mut merged = match resolved {
        Value::Object(members) => members,
        _ => Vec::new(),
    };
    for (name, value) in properties(&additional, root, processing) {
        match merged.iter_mut().find(|(existing, _)| *existing == name) {
            Some(slot) => slot.1 = value,
            None => merged.push((name, value)),
        }
    }
    Value::Object(merged)
}

/// `_process_dict_properties`: `$defs` copied as it is, containers inlined, anything else copied.
fn properties(
    members: &[(String, Value)],
    root: &Value,
    processing: &mut Vec<String>,
) -> Vec<(String, Value)> {
    members
        .iter()
        .map(|(name, value)| {
            let value = match value {
                Value::Object(_) | Value::Array(_) if name != "$defs" => {
                    inline(value, root, processing)
                }
                other => other.clone(),
            };
            (name.clone(), value)
        })
        .collect()
}

/// `_rm_titles`: every `title` member of an object at any object depth goes, except an object-valued property named
/// `title`; arrays are not entered.
fn without_titles(value: &Value, parent: &str) -> Value {
    let Value::Object(members) = value else {
        return value.clone();
    };
    Value::Object(
        members
            .iter()
            .filter_map(|(name, child)| {
                let object = matches!(child, Value::Object(_));
                if name == "title" && !(object && parent == "properties") {
                    return None;
                }
                let child = if object {
                    without_titles(child, name)
                } else {
                    child.clone()
                };
                Some((name.clone(), child))
            })
            .collect(),
    )
}

//! The strict JSON Schema the OpenAI SDK's typed parse sends for a Pydantic model (`to_strict_json_schema`).

use crate::json::{self, Value};
use crate::provider::wire;

/// `schema` made strict: every object closed unless it says otherwise, every property required, a single `allOf`
/// merged, `null` defaults dropped, and a `$ref` with sibling keywords inlined; `$defs` are made strict too.
#[must_use]
pub fn strict(schema: &Value) -> Value {
    ensure(schema, schema)
}

fn ensure(schema: &Value, root: &Value) -> Value {
    let Some(members) = json::members(schema) else {
        return schema.clone();
    };
    let mut members = members.to_vec();
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(definitions)) = members
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
        {
            for (_, definition) in definitions.iter_mut() {
                *definition = ensure(definition, root);
            }
        }
    }
    close(&mut members, root);
    if let Some(Value::Array(all_of)) = members
        .iter()
        .find(|(name, _)| name == "allOf")
        .map(|(_, value)| value.clone())
    {
        merge_all_of(&mut members, &all_of, root);
    }
    if matches!(
        members.iter().find(|(name, _)| name == "default"),
        Some((_, Value::Null))
    ) {
        wire::take(&mut members, "default");
    }
    inline_ref(members, root)
}

/// Closes an object, requires every property, and makes its properties, items, and union variants strict.
fn close(members: &mut Vec<(String, Value)>, root: &Value) {
    let typed_object = members.iter().any(|(name, value)| {
        name == "type" && matches!(value, Value::String(kind) if kind == "object")
    });
    if typed_object
        && !members
            .iter()
            .any(|(name, _)| name == "additionalProperties")
    {
        members.push(("additionalProperties".to_owned(), Value::Bool(false)));
    }
    if let Some(Value::Object(properties)) = members
        .iter()
        .find(|(name, _)| name == "properties")
        .map(|(_, value)| value.clone())
    {
        let required = properties
            .iter()
            .map(|(name, _)| json::text(name))
            .collect();
        wire::set(members, "required", json::array(required));
        let strict = properties
            .iter()
            .map(|(name, property)| (name.clone(), ensure(property, root)))
            .collect();
        wire::set(members, "properties", Value::Object(strict));
    }
    for (name, value) in members.iter_mut() {
        match (name.as_str(), value) {
            ("items", items @ Value::Object(_)) => *items = ensure(items, root),
            ("anyOf", Value::Array(variants)) => {
                for variant in variants.iter_mut() {
                    *variant = ensure(variant, root);
                }
            }
            _ => {}
        }
    }
}

fn merge_all_of(members: &mut Vec<(String, Value)>, all_of: &[Value], root: &Value) {
    if let [only] = all_of {
        if let Value::Object(merged) = ensure(only, root) {
            for (name, value) in merged {
                wire::set(members, &name, value);
            }
        }
        wire::take(members, "allOf");
    } else {
        let strict = all_of.iter().map(|entry| ensure(entry, root)).collect();
        wire::set(members, "allOf", Value::Array(strict));
    }
}

/// A `$ref` beside other keywords, replaced by the definition it names under those keywords, made strict again.
fn inline_ref(mut members: Vec<(String, Value)>, root: &Value) -> Value {
    let reference = match members.iter().find(|(name, _)| name == "$ref") {
        Some((_, Value::String(reference))) if members.len() > 1 => reference.clone(),
        _ => return Value::Object(members),
    };
    let Some(Value::Object(resolved)) = resolve(root, &reference) else {
        return Value::Object(members);
    };
    for (name, value) in resolved {
        if !members.iter().any(|(member, _)| *member == name) {
            members.push((name, value));
        }
    }
    wire::take(&mut members, "$ref");
    ensure(&Value::Object(members), root)
}

fn resolve(root: &Value, reference: &str) -> Option<Value> {
    let mut resolved = root;
    for key in reference.strip_prefix("#/")?.split('/') {
        resolved = resolved
            .get(key)
            .filter(|value| matches!(value, Value::Object(_)))?;
    }
    Some(resolved.clone())
}

//! Value helpers the two provider adapters share: building objects with optional members, reading provider counts,
//! and re-shaping verbatim provider items the way the Python adapters' model dumps did.

use crate::json::{self, Value};

/// An object from members, in order.
pub fn object(members: Vec<(&str, Value)>) -> Value {
    json::object_of(
        members
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect(),
    )
}

/// `value[key]` as a count: a non-negative integer, else zero (absent, null, or anything else), as the Python
/// usage readers' `int(value or 0)` read a provider's report.
pub fn count(value: Option<&Value>, key: &str) -> u64 {
    value
        .and_then(|value| json::u64_of(value, key))
        .unwrap_or(0)
}

/// `value[key]` unless it is absent or null.
pub fn present<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value
        .get(key)
        .filter(|member| !matches!(member, Value::Null))
}

/// One strict JSON object parsed from `text`; `None` for anything else.
pub fn object_text(text: &str) -> Option<Value> {
    json::parse(text.as_bytes())
        .ok()
        .filter(|value| matches!(value, Value::Object(_)))
}

/// `value` without null members, at every depth (a Pydantic `model_dump(exclude_none=True)`).
pub fn without_nulls(value: &Value) -> Value {
    match value {
        Value::Object(members) => Value::Object(
            members
                .iter()
                .filter(|(_, member)| !matches!(member, Value::Null))
                .map(|(name, member)| (name.clone(), without_nulls(member)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(without_nulls).collect()),
        other => other.clone(),
    }
}

/// An object's members with `order`'s names first, in that order, then the rest as they came (a Pydantic model's
/// declared fields, then its extra ones).
pub fn ordered(value: &Value, order: &[&str]) -> Value {
    let Some(members) = json::members(value) else {
        return value.clone();
    };
    let mut result: Vec<(String, Value)> = order
        .iter()
        .filter_map(|name| members.iter().find(|(member, _)| member == name).cloned())
        .collect();
    result.extend(
        members
            .iter()
            .filter(|(name, _)| !order.contains(&name.as_str()))
            .cloned(),
    );
    Value::Object(result)
}

/// Sets `name` on an object's members: in place when present, else appended (a Python dict assignment).
pub fn set(members: &mut Vec<(String, Value)>, name: &str, value: Value) {
    match members.iter_mut().find(|(member, _)| member == name) {
        Some((_, slot)) => *slot = value,
        None => members.push((name.to_owned(), value)),
    }
}

/// Removes and returns `name` from an object's members (a Python `dict.pop(name, None)`).
pub fn take(members: &mut Vec<(String, Value)>, name: &str) -> Option<Value> {
    let index = members.iter().position(|(member, _)| member == name)?;
    Some(members.remove(index).1)
}

//! JSON helpers over the strict reader's values: building, reading, and Python-faithful encoding.
//!
//! Every value the Brain receives, stores, or sends is a [`strict_json::Value`]: objects keep their member order and
//! integers of any size, so a schema reaches a provider in the order its producer wrote it. Encoding goes through the
//! Team protocol's reproduction of Python's `json.dumps`, so the bytes are what the Python Brain wrote.

pub use strict_json::{Number, Value};
use team_protocol::python::json::{self as python_json, Options};

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`: compact, member order kept, UTF-8.
pub const COMPACT: Options = Options {
    ensure_ascii: false,
    sort_keys: false,
    compact: true,
};
/// `json.dumps(value, ensure_ascii=False)`: Python's default separators, member order kept, UTF-8.
pub const SPACED: Options = Options {
    ensure_ascii: false,
    sort_keys: false,
    compact: false,
};
/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`.
pub const SORTED: Options = Options {
    ensure_ascii: false,
    sort_keys: true,
    compact: true,
};
/// `json.dumps(value)`: ASCII escapes and Python's default separators.
pub const ASCII: Options = Options {
    ensure_ascii: true,
    sort_keys: false,
    compact: false,
};

/// Encodes `value` with `options`.
#[must_use]
pub fn dumps(value: &Value, options: Options) -> String {
    python_json::dumps(value, options)
}

/// Encodes `value` compactly (see [`COMPACT`]).
#[must_use]
pub fn compact(value: &Value) -> String {
    dumps(value, COMPACT)
}

/// Decodes one strict JSON document.
///
/// # Errors
///
/// The strict reader's refusal.
pub fn parse(bytes: &[u8]) -> Result<Value, strict_json::Error> {
    strict_json::parse(bytes)
}

/// An object from `(name, value)` members, in order.
#[must_use]
pub fn object<const N: usize>(members: [(&str, Value); N]) -> Value {
    Value::Object(
        members
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect(),
    )
}

/// An object from owned members, in order.
#[must_use]
pub fn object_of(members: Vec<(String, Value)>) -> Value {
    Value::Object(members)
}

/// A string value.
#[must_use]
pub fn text(value: &str) -> Value {
    Value::String(value.to_owned())
}

/// An integer value.
#[must_use]
pub fn int(value: i64) -> Value {
    Value::Number(Number::Int(value))
}

/// An unsigned integer value; values beyond `i64` keep their exact decimal text.
#[must_use]
pub fn uint(value: u64) -> Value {
    i64::try_from(value).map_or_else(|_| Value::Number(Number::BigInt(value.to_string())), int)
}

/// An array value.
#[must_use]
pub fn array(items: Vec<Value>) -> Value {
    Value::Array(items)
}

/// `value[key]` as text.
#[must_use]
pub fn str_of<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    match value.get(key) {
        Some(Value::String(text)) => Some(text),
        _ => None,
    }
}

/// `value[key]` as an exact non-negative integer that fits `u64`.
#[must_use]
pub fn u64_of(value: &Value, key: &str) -> Option<u64> {
    match value.get(key) {
        Some(Value::Number(Number::Int(number))) => u64::try_from(*number).ok(),
        _ => None,
    }
}

/// `value[key]` as a boolean.
#[must_use]
pub fn bool_of(value: &Value, key: &str) -> Option<bool> {
    match value.get(key) {
        Some(Value::Bool(flag)) => Some(*flag),
        _ => None,
    }
}

/// `value[key]` as an array.
#[must_use]
pub fn array_of<'a>(value: &'a Value, key: &str) -> Option<&'a [Value]> {
    match value.get(key) {
        Some(Value::Array(items)) => Some(items),
        _ => None,
    }
}

/// The members of an object value.
#[must_use]
pub fn members(value: &Value) -> Option<&[(String, Value)]> {
    match value {
        Value::Object(members) => Some(members),
        _ => None,
    }
}

/// Whether an object value has exactly these member names.
#[must_use]
pub fn has_exactly(value: &Value, fields: &[&str]) -> bool {
    members(value).is_some_and(|members| team_protocol::python::value::has_exactly(members, fields))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_follow_python() {
        let value = object([
            ("b", text("é")),
            ("a", array(vec![int(1), Value::Bool(true)])),
        ]);
        assert_eq!(compact(&value), r#"{"b":"é","a":[1,true]}"#);
        assert_eq!(dumps(&value, SPACED), r#"{"b": "é", "a": [1, true]}"#);
        assert_eq!(dumps(&value, SORTED), r#"{"a":[1,true],"b":"é"}"#);
        assert_eq!(
            dumps(&value, ASCII),
            "{\"b\": \"\\u00e9\", \"a\": [1, true]}"
        );
        assert_eq!(
            uint(u64::MAX),
            Value::Number(Number::BigInt(u64::MAX.to_string()))
        );
    }

    #[test]
    fn readers_are_typed() {
        let value = parse(br#"{"s":"x","n":3,"b":false,"l":[]}"#).unwrap();
        assert_eq!(str_of(&value, "s"), Some("x"));
        assert_eq!(u64_of(&value, "n"), Some(3));
        assert_eq!(bool_of(&value, "b"), Some(false));
        assert_eq!(array_of(&value, "l").map(<[Value]>::len), Some(0));
        assert_eq!(str_of(&value, "n"), None);
        assert!(has_exactly(&value, &["l", "b", "n", "s"]));
        assert!(!has_exactly(&value, &["s"]));
    }
}

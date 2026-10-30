//! A structured answer checked against the JSON Schema its Pydantic model produced, as the model's own validation
//! checked it: closed objects with every required field, strings within their enum and code-point bound, booleans,
//! bounded arrays, and nullable unions. Only the keywords the decisions' schemas use are understood; a schema with
//! any other shape admits nothing.

use crate::json::{self, Value};

/// A decision schema written as JSON text in the source.
///
/// # Panics
///
/// When `text` is not strict JSON: a programming error caught by every test that uses the schema.
pub(super) fn schema(text: &str) -> Value {
    json::parse(text.as_bytes()).expect("a decision schema is strict JSON")
}

/// Whether `value` conforms to `schema`.
pub(super) fn conforms(schema: &Value, value: &Value) -> bool {
    if let Some(variants) = json::array_of(schema, "anyOf") {
        return variants.iter().any(|variant| conforms(variant, value));
    }
    match (json::str_of(schema, "type"), value) {
        (Some("object"), Value::Object(members)) => object(schema, members),
        (Some("string"), Value::String(text)) => string(schema, text),
        (Some("array"), Value::Array(items)) => array(schema, items),
        (Some("boolean"), Value::Bool(_)) | (Some("null"), Value::Null) => true,
        _ => false,
    }
}

fn object(schema: &Value, members: &[(String, Value)]) -> bool {
    let properties = schema
        .get("properties")
        .and_then(json::members)
        .unwrap_or_default();
    let closed = schema.get("additionalProperties") == Some(&Value::Bool(false));
    let complete = json::array_of(schema, "required")
        .unwrap_or_default()
        .iter()
        .all(|name| matches!(name, Value::String(name) if members.iter().any(|(member, _)| member == name)));
    complete
        && members.iter().all(|(name, member)| {
            match properties.iter().find(|(property, _)| property == name) {
                Some((_, property)) => conforms(property, member),
                None => !closed,
            }
        })
}

fn string(schema: &Value, text: &str) -> bool {
    let chosen = json::array_of(schema, "enum").is_none_or(|choices| {
        choices
            .iter()
            .any(|choice| matches!(choice, Value::String(choice) if choice == text))
    });
    chosen && within(schema, "maxLength", text.chars().count())
}

fn array(schema: &Value, items: &[Value]) -> bool {
    within(schema, "maxItems", items.len())
        && schema
            .get("items")
            .is_none_or(|item| items.iter().all(|value| conforms(item, value)))
}

/// Whether `count` is within the schema's bound `key`, when it states one.
fn within(schema: &Value, key: &str, count: usize) -> bool {
    json::u64_of(schema, key)
        .is_none_or(|bound| u64::try_from(count).is_ok_and(|count| count <= bound))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROUTE: &str = r#"{"additionalProperties":false,"properties":{"intent":{"enum":["a","b"],"type":"string"},"query":{"maxLength":3,"type":"string"},"ids":{"items":{"type":"string"},"maxItems":1,"type":"array"},"note":{"anyOf":[{"maxLength":2,"type":"string"},{"type":"null"}]},"done":{"type":"boolean"}},"required":["intent","query","ids","note","done"],"type":"object"}"#;

    #[test]
    fn an_answer_conforms_only_to_its_closed_model() {
        let schema = schema(ROUTE);
        let check = |text: &str| conforms(&schema, &json::parse(text.as_bytes()).unwrap());
        assert!(check(
            r#"{"intent":"a","query":"日本語","ids":["x"],"note":null,"done":true}"#
        ));
        assert!(check(
            r#"{"intent":"b","query":"","ids":[],"note":"ok","done":false}"#
        ));
        for refused in [
            r#"{"intent":"c","query":"","ids":[],"note":null,"done":true}"#,
            r#"{"intent":"a","query":"four","ids":[],"note":null,"done":true}"#,
            r#"{"intent":"a","query":"","ids":["x","y"],"note":null,"done":true}"#,
            r#"{"intent":"a","query":"","ids":[1],"note":null,"done":true}"#,
            r#"{"intent":"a","query":"","ids":[],"note":"long","done":true}"#,
            r#"{"intent":"a","query":"","ids":[],"note":null,"done":1}"#,
            r#"{"intent":"a","query":"","ids":[],"note":null}"#,
            r#"{"intent":"a","query":"","ids":[],"note":null,"done":true,"extra":1}"#,
            "[]",
        ] {
            assert!(!check(refused), "{refused}");
        }
    }
}

//! The JSON Schema the Anthropic SDK's `transform_schema` sends as structured output: supported keywords kept in
//! the API's order, objects closed, and every unsupported constraint written into the description as a hint.

use crate::json::{self, Number, Value};
use crate::provider::wire;

const STRING_FORMATS: [&str; 10] = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

/// `schema` as the API accepts it.
#[must_use]
pub fn transform(schema: &Value) -> Value {
    let Some(members) = json::members(schema) else {
        return schema.clone();
    };
    let mut rest = members.to_vec();
    let mut out: Vec<(String, Value)> = Vec::new();
    if let Some(Value::Object(definitions)) = wire::take(&mut rest, "$defs") {
        let strict = definitions
            .iter()
            .map(|(name, definition)| (name.clone(), transform(definition)))
            .collect();
        out.push(("$defs".to_owned(), Value::Object(strict)));
    }
    if let Some(reference) = wire::take(&mut rest, "$ref") {
        out.push(("$ref".to_owned(), reference));
        return Value::Object(out);
    }
    let kind = wire::take(&mut rest, "type");
    combine(&mut rest, &mut out, kind.as_ref());
    for keyword in ["enum", "description", "title"] {
        match wire::take(&mut rest, keyword) {
            Some(Value::Array(values)) if keyword == "enum" => {
                out.push((keyword.to_owned(), Value::Array(values)));
            }
            Some(Value::Null) | None => {}
            Some(_) if keyword == "enum" => {}
            Some(value) => out.push((keyword.to_owned(), value)),
        }
    }
    match kind.as_ref().and_then(|kind| match kind {
        Value::String(kind) => Some(kind.as_str()),
        _ => None,
    }) {
        Some("object") => object(&mut rest, &mut out),
        Some("string") => string(&mut rest, &mut out),
        Some("array") => array(&mut rest, &mut out),
        _ => {}
    }
    if !rest.is_empty() {
        describe(&rest, &mut out);
    }
    Value::Object(out)
}

/// The union keywords (`oneOf` read as `anyOf`), else the type.
fn combine(rest: &mut Vec<(String, Value)>, out: &mut Vec<(String, Value)>, kind: Option<&Value>) {
    let any_of = wire::take(rest, "anyOf");
    let one_of = wire::take(rest, "oneOf");
    let all_of = wire::take(rest, "allOf");
    let variants = |values: &[Value]| Value::Array(values.iter().map(transform).collect());
    match (any_of, one_of, all_of) {
        (Some(Value::Array(values)), _, _) | (_, Some(Value::Array(values)), _) => {
            out.push(("anyOf".to_owned(), variants(&values)));
        }
        (_, _, Some(Value::Array(values))) => out.push(("allOf".to_owned(), variants(&values))),
        _ => {
            if let Some(kind) = kind {
                out.push(("type".to_owned(), kind.clone()));
            }
        }
    }
}

fn object(rest: &mut Vec<(String, Value)>, out: &mut Vec<(String, Value)>) {
    let properties = match wire::take(rest, "properties") {
        Some(Value::Object(properties)) => properties
            .iter()
            .map(|(name, property)| (name.clone(), transform(property)))
            .collect(),
        _ => Vec::new(),
    };
    out.push(("properties".to_owned(), Value::Object(properties)));
    wire::take(rest, "additionalProperties");
    out.push(("additionalProperties".to_owned(), Value::Bool(false)));
    match wire::take(rest, "required") {
        Some(Value::Null) | None => {}
        Some(required) => out.push(("required".to_owned(), required)),
    }
}

fn string(rest: &mut Vec<(String, Value)>, out: &mut Vec<(String, Value)>) {
    match wire::take(rest, "format") {
        Some(Value::String(format)) if STRING_FORMATS.contains(&format.as_str()) => {
            out.push(("format".to_owned(), Value::String(format)));
        }
        Some(Value::String(format)) if format.is_empty() => {}
        Some(Value::Null | Value::Bool(false)) | None => {}
        Some(format) => rest.push(("format".to_owned(), format)),
    }
}

fn array(rest: &mut Vec<(String, Value)>, out: &mut Vec<(String, Value)>) {
    match wire::take(rest, "items") {
        Some(Value::Null) | None => {}
        Some(items) => out.push(("items".to_owned(), transform(&items))),
    }
    match wire::take(rest, "minItems") {
        Some(Value::Null) | None => {}
        Some(Value::Number(Number::Int(count @ (0 | 1)))) => {
            out.push(("minItems".to_owned(), json::int(count)));
        }
        Some(count) => rest.push(("minItems".to_owned(), count)),
    }
}

/// Every keyword the API does not support, appended to the description as `{name: value, ...}`.
fn describe(rest: &[(String, Value)], out: &mut Vec<(String, Value)>) {
    let hint = format!(
        "{{{}}}",
        rest.iter()
            .map(|(name, value)| format!("{name}: {}", python_str(value)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let description = match out.iter().find(|(name, _)| name == "description") {
        Some((_, Value::String(description))) => format!("{description}\n\n{hint}"),
        _ => hint,
    };
    wire::set(out, "description", Value::String(description));
}

/// Python's `str()` of a JSON value: text as is, anything else as its `repr()`.
fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python_repr(other),
    }
}

/// Python's `repr()` of a JSON value as `json.loads` would have produced it.
fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(flag) => if *flag { "True" } else { "False" }.to_owned(),
        Value::Number(Number::Float(float)) => team_protocol::python::json::float_repr(*float),
        Value::Number(number) => team_protocol::python::json::number_text(number),
        Value::String(text) => repr_text(text),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(members) => format!(
            "{{{}}}",
            members
                .iter()
                .map(|(name, member)| format!("{}: {}", repr_text(name), python_repr(member)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn repr_text(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other if other == quote => {
                out.push('\\');
                out.push(other);
            }
            other if team_protocol::python::text::isprintable_char(other) => out.push(other),
            other => out.push_str(&escaped(u32::from(other))),
        }
    }
    out.push(quote);
    out
}

fn escaped(code: u32) -> String {
    if code < 0x100 {
        format!("\\x{code:02x}")
    } else if code < 0x1_0000 {
        format!("\\u{code:04x}")
    } else {
        format!("\\U{code:08x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Value {
        json::parse(text.as_bytes()).unwrap()
    }

    #[test]
    fn python_reprs_follow_python() {
        let value = parse(r#"[1, 2.5, "it's", "a\"b'c", null, true, {"k": "\u0001é\n"}]"#);
        assert_eq!(
            python_repr(&value),
            r#"[1, 2.5, "it's", 'a"b\'c', None, True, {'k': '\x01é\n'}]"#
        );
        assert_eq!(python_str(&json::text("^[a-z]$")), "^[a-z]$");
    }

    #[test]
    fn unsupported_constraints_become_description_hints_and_objects_close() {
        let schema = parse(
            r#"{"additionalProperties":false,"description":"D","properties":{"missing":{"anyOf":[{"maxLength":300,"type":"string"},{"type":"null"}],"title":"Missing"},"ids":{"items":{"type":"string","format":"uuid"},"maxItems":4,"minItems":1,"title":"Ids","type":"array"},"tags":{"items":{"type":"string","format":"slug","pattern":"^[a-z]+$"},"minItems":2,"type":"array"},"kind":{"enum":["a","b"],"type":"string","default":"a"}},"required":["missing","ids"],"title":"T","type":"object"}"#,
        );
        assert_eq!(
            transform(&schema),
            parse(
                r#"{"type":"object","description":"D","title":"T","properties":{"missing":{"anyOf":[{"type":"string","description":"{maxLength: 300}"},{"type":"null"}],"title":"Missing"},"ids":{"type":"array","title":"Ids","items":{"type":"string","format":"uuid"},"minItems":1,"description":"{maxItems: 4}"},"tags":{"type":"array","items":{"type":"string","description":"{pattern: ^[a-z]+$, format: slug}"},"description":"{minItems: 2}"},"kind":{"type":"string","enum":["a","b"],"description":"{default: a}"}},"additionalProperties":false,"required":["missing","ids"]}"#
            )
        );
    }
}

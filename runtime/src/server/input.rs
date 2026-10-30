//! The closed request objects every endpoint reads: one constant 422 for any shape violation (Pydantic's
//! `ClosedInput`), fixed 400 texts for the contract checks the Python runtime raised after validation.

use zeroize::Zeroizing;

use crate::catalog;
use crate::errors::BrainError;
use crate::json::{Number, Value};
use crate::provider::{ProviderConfig, ProviderId};

/// The shortest provider key admitted.
pub const MIN_API_KEY_CHARS: usize = 16;
/// The longest provider key admitted.
pub const MAX_API_KEY_CHARS: usize = 8192;

/// Why a request was refused before or while it ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// The body broke its closed shape: 422 `{"detail": "Request validation failed"}`.
    Invalid,
    /// The operation failed.
    Brain(BrainError),
}

impl From<BrainError> for RouteError {
    fn from(error: BrainError) -> Self {
        Self::Brain(error)
    }
}

/// The result of reading one request field.
pub type Field<T> = Result<T, RouteError>;

/// The members of a closed object: every member named in `allowed`, each of `required` present; anything else 422.
///
/// # Errors
///
/// [`RouteError::Invalid`].
pub fn closed<'a>(
    value: &'a Value,
    allowed: &[&str],
    required: &[&str],
) -> Field<&'a [(String, Value)]> {
    let Value::Object(members) = value else {
        return Err(RouteError::Invalid);
    };
    let known = members
        .iter()
        .all(|(name, _)| allowed.contains(&name.as_str()));
    let complete = required
        .iter()
        .all(|name| members.iter().any(|(member, _)| member == name));
    if known && complete {
        Ok(members)
    } else {
        Err(RouteError::Invalid)
    }
}

/// The member `key` of an object, or `None` when absent.
#[must_use]
pub fn member<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key)
}

/// A string of `min..=max` code points (Pydantic's `str` with `min_length` and `max_length`).
///
/// # Errors
///
/// [`RouteError::Invalid`] for another type, a missing member, or a length outside the bounds.
pub fn string(value: &Value, key: &str, min: usize, max: usize) -> Field<String> {
    match value.get(key) {
        Some(Value::String(text)) if (min..=max).contains(&text.chars().count()) => {
            Ok(text.clone())
        }
        _ => Err(RouteError::Invalid),
    }
}

/// [`string`] or `null`.
///
/// # Errors
///
/// [`RouteError::Invalid`] for anything else.
pub fn optional_string(value: &Value, key: &str, min: usize, max: usize) -> Field<Option<String>> {
    match value.get(key) {
        Some(Value::Null) => Ok(None),
        _ => string(value, key, min, max).map(Some),
    }
}

/// A strict boolean.
///
/// # Errors
///
/// [`RouteError::Invalid`] for anything else.
pub fn boolean(value: &Value, key: &str) -> Field<bool> {
    match value.get(key) {
        Some(Value::Bool(flag)) => Ok(*flag),
        _ => Err(RouteError::Invalid),
    }
}

/// An array of at most `max` (and at least `min`) items.
///
/// # Errors
///
/// [`RouteError::Invalid`] for anything else.
pub fn list<'a>(value: &'a Value, key: &str, min: usize, max: usize) -> Field<&'a [Value]> {
    match value.get(key) {
        Some(Value::Array(items)) if (min..=max).contains(&items.len()) => Ok(items),
        _ => Err(RouteError::Invalid),
    }
}

/// [`list`] or `null`.
///
/// # Errors
///
/// [`RouteError::Invalid`] for anything else.
pub fn optional_list<'a>(value: &'a Value, key: &str, max: usize) -> Field<Option<&'a [Value]>> {
    match value.get(key) {
        Some(Value::Null) => Ok(None),
        _ => list(value, key, 0, max).map(Some),
    }
}

/// An object member (Pydantic's `dict[str, Any]`).
///
/// # Errors
///
/// [`RouteError::Invalid`] for anything else.
pub fn object<'a>(value: &'a Value, key: &str) -> Field<&'a Value> {
    match value.get(key) {
        Some(object @ Value::Object(_)) => Ok(object),
        _ => Err(RouteError::Invalid),
    }
}

/// An object member or `null`.
///
/// # Errors
///
/// [`RouteError::Invalid`] for anything else.
pub fn optional_object<'a>(value: &'a Value, key: &str) -> Field<Option<&'a Value>> {
    match value.get(key) {
        Some(Value::Null) => Ok(None),
        _ => object(value, key).map(Some),
    }
}

/// An integer member (Pydantic's lax `int`: an exact integer).
///
/// # Errors
///
/// [`RouteError::Invalid`] for anything else.
pub fn integer(value: &Value, key: &str) -> Field<i64> {
    match value.get(key) {
        Some(Value::Number(Number::Int(number))) => Ok(*number),
        _ => Err(RouteError::Invalid),
    }
}

/// The request's `provider` object (`ProviderInput`) as the Team chat model it names.
///
/// # Errors
///
/// 422 for a shape violation; 400 "unsupported model for provider" for a model that is no chat model of the
/// provider, and 400 "invalid model provider credential" for a key holding NUL.
pub fn provider(value: &Value) -> Field<ProviderConfig> {
    let config = provider_any(value)?;
    if !catalog::is_chat_model(config.provider.as_str(), &config.model) {
        return Err(BrainError::contract("unsupported model for provider").into());
    }
    Ok(config)
}

/// The request's `provider` object without the chat-model check (the Browser runs its own fixed model).
///
/// # Errors
///
/// As [`provider`], less the chat-model check.
pub fn provider_any(value: &Value) -> Field<ProviderConfig> {
    let input = object(value, "provider")?;
    closed(
        input,
        &["provider", "model", "api_key"],
        &["provider", "model", "api_key"],
    )?;
    let id = match input.get("provider") {
        Some(Value::String(id)) => ProviderId::parse(id).ok_or(RouteError::Invalid)?,
        _ => return Err(RouteError::Invalid),
    };
    let model = string(input, "model", 1, 128)?;
    let api_key = Zeroizing::new(string(
        input,
        "api_key",
        MIN_API_KEY_CHARS,
        MAX_API_KEY_CHARS,
    )?);
    if api_key.contains('\0') {
        return Err(BrainError::contract("invalid model provider credential").into());
    }
    Ok(ProviderConfig {
        provider: id,
        model,
        api_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    fn body(text: &str) -> Value {
        json::parse(text.as_bytes()).unwrap()
    }

    #[test]
    fn closed_objects_refuse_unknown_and_missing_members() {
        let value = body(r#"{"a":1,"b":2}"#);
        assert!(closed(&value, &["a", "b"], &["a"]).is_ok());
        assert_eq!(closed(&value, &["a"], &[]), Err(RouteError::Invalid));
        assert_eq!(
            closed(&value, &["a", "b", "c"], &["c"]),
            Err(RouteError::Invalid)
        );
        assert_eq!(closed(&json::text("x"), &[], &[]), Err(RouteError::Invalid));
    }

    #[test]
    fn field_readers_check_type_and_code_point_bounds() {
        let value = body(r#"{"s":"日本","n":null,"b":true,"l":[1],"o":{},"i":3,"f":1.0}"#);
        assert_eq!(string(&value, "s", 1, 2), Ok("日本".into()));
        assert_eq!(string(&value, "s", 3, 9), Err(RouteError::Invalid));
        assert_eq!(optional_string(&value, "n", 1, 2), Ok(None));
        assert_eq!(boolean(&value, "b"), Ok(true));
        assert_eq!(boolean(&value, "i"), Err(RouteError::Invalid));
    }

    #[test]
    fn container_and_number_readers_check_their_types() {
        let value = body(r#"{"n":null,"l":[1],"o":{},"i":3,"f":1.0}"#);
        assert_eq!(list(&value, "l", 1, 1).map(<[Value]>::len), Ok(1));
        assert_eq!(list(&value, "l", 2, 3), Err(RouteError::Invalid));
        assert_eq!(optional_list(&value, "n", 1), Ok(None));
        assert!(object(&value, "o").is_ok() && optional_object(&value, "n") == Ok(None));
        assert_eq!(integer(&value, "i"), Ok(3));
        assert_eq!(integer(&value, "f"), Err(RouteError::Invalid));
        assert!(member(&value, "i").is_some());
    }

    #[test]
    fn a_provider_names_a_chat_model_and_a_bounded_key() {
        let key = "k".repeat(16);
        let ok = body(&format!(
            r#"{{"provider":{{"provider":"openai","model":"gpt-6-luna","api_key":"{key}"}}}}"#
        ));
        let config = provider(&ok).unwrap();
        assert_eq!(
            (config.provider, config.model.as_str()),
            (ProviderId::OpenAi, "gpt-6-luna")
        );
        assert!(!format!("{config:?}").contains(&key));
        let browser = body(&format!(
            r#"{{"provider":{{"provider":"anthropic","model":"claude-haiku-5-5","api_key":"{key}"}}}}"#
        ));
        assert_eq!(
            provider(&browser).map(|_| ()),
            Err(RouteError::Brain(BrainError::contract(
                "unsupported model for provider"
            )))
        );
        assert!(provider_any(&browser).is_ok());
        let short =
            body(r#"{"provider":{"provider":"openai","model":"gpt-6-luna","api_key":"short"}}"#);
        assert_eq!(provider(&short).map(|_| ()), Err(RouteError::Invalid));
        let other = body(&format!(
            r#"{{"provider":{{"provider":"other","model":"m","api_key":"{key}"}}}}"#
        ));
        assert_eq!(provider(&other).map(|_| ()), Err(RouteError::Invalid));
        let nul = body(&format!(
            r#"{{"provider":{{"provider":"openai","model":"gpt-6-luna","api_key":"{key}\u0000"}}}}"#
        ));
        assert_eq!(
            provider(&nul).map(|_| ()),
            Err(RouteError::Brain(BrainError::contract(
                "invalid model provider credential"
            )))
        );
    }
}

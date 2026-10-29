//! Fail-closed Brain provider policy: the packaged model catalog's hosts plus one fixed decision host.

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use strict_json::{Number, Value};

/// Where the image packages the neutral model catalog.
pub const CATALOG_PATH: &str = "/app/model_catalog.json";
/// The `TypeSafe` Jev intent fast path is a decision endpoint, not a selectable chat provider, so it stays out of the
/// model catalog that every component mirrors (ADR-0077).
pub const DECISION_HOSTS: [&str; 1] = ["api.typesafe.ai"];
const MAX_CATALOG_BYTES: u64 = 256 * 1024;

/// The packaged provider catalog cannot safely authorize Brain egress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyError(pub &'static str);

type Members<'a> = &'a [(String, Value)];

fn exact_object<'a>(
    value: &'a Value,
    fields: &[&str],
    label: &'static str,
) -> Result<Members<'a>, PolicyError> {
    let Value::Object(members) = value else {
        return Err(PolicyError(label));
    };
    let names: BTreeSet<&str> = members.iter().map(|(name, _)| name.as_str()).collect();
    if names != fields.iter().copied().collect() {
        return Err(PolicyError(label));
    }
    Ok(members)
}

fn field<'a>(members: Members<'a>, name: &str) -> &'a Value {
    members
        .iter()
        .find_map(|(key, value)| (key == name).then_some(value))
        .unwrap_or(&Value::Null)
}

fn bounded_text<'a>(
    value: &'a Value,
    maximum: usize,
    label: &'static str,
) -> Result<&'a str, PolicyError> {
    match value {
        Value::String(text) if (1..=maximum).contains(&text.chars().count()) => Ok(text),
        _ => Err(PolicyError(label)),
    }
}

/// One LDH label: `[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?`.
fn label(text: &str) -> bool {
    let bytes = text.as_bytes();
    let edge = |byte: &u8| byte.is_ascii_lowercase() || byte.is_ascii_digit();
    (1..=63).contains(&bytes.len())
        && bytes.first().is_some_and(edge)
        && bytes.last().is_some_and(edge)
        && bytes.iter().all(|byte| edge(byte) || *byte == b'-')
}

/// The catalog's `HOST_RE`: one or more lowercase LDH labels joined by dots.
fn host(text: &str) -> bool {
    text.split('.').all(label)
}

/// The catalog's `ID_RE`, `[a-z0-9][a-z0-9._-]{0,127}`, over text its caller bounded to 128 characters.
fn identifier(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes
        .first()
        .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(byte))
}

fn provider_endpoint(value: &Value) -> Result<String, PolicyError> {
    let endpoint = exact_object(value, &["host", "path"], "provider endpoint is invalid")?;
    let host_name = bounded_text(field(endpoint, "host"), 253, "provider host is invalid")?;
    let path = bounded_text(field(endpoint, "path"), 256, "provider path is invalid")?;
    if !host(host_name) || !path.starts_with('/') {
        return Err(PolicyError("provider endpoint is invalid"));
    }
    Ok(host_name.to_owned())
}

/// Python's `isinstance(price, int) and not isinstance(price, bool) and 0 <= price <= 10_000_000`.
fn price(value: &Value) -> bool {
    matches!(value, Value::Number(Number::Int(price)) if (0..=10_000_000).contains(price))
}

const MODEL_FIELDS: [&str; 5] = [
    "chat",
    "id",
    "input_usd_per_million_cents",
    "output_usd_per_million_cents",
    "title",
];

fn model_id(value: &Value, seen: &BTreeSet<String>) -> Result<String, PolicyError> {
    let model = exact_object(value, &MODEL_FIELDS, "provider model is invalid")?;
    if !matches!(field(model, "chat"), Value::Bool(_)) {
        return Err(PolicyError("model chat admission is invalid"));
    }
    let id = bounded_text(field(model, "id"), 128, "model id is invalid")?;
    if !identifier(id) || seen.contains(id) {
        return Err(PolicyError("model id is invalid"));
    }
    bounded_text(field(model, "title"), 120, "model title is invalid")?;
    if !price(field(model, "input_usd_per_million_cents"))
        || !price(field(model, "output_usd_per_million_cents"))
    {
        return Err(PolicyError("model price is invalid"));
    }
    Ok(id.to_owned())
}

fn model_ids(value: &Value) -> Result<BTreeSet<String>, PolicyError> {
    let Value::Array(models) = value else {
        return Err(PolicyError("provider models are invalid"));
    };
    if !(1..=128).contains(&models.len()) {
        return Err(PolicyError("provider models are invalid"));
    }
    let mut ids = BTreeSet::new();
    for model in models {
        let id = model_id(model, &ids)?;
        ids.insert(id);
    }
    Ok(ids)
}

/// Whether a value is a string member of `names`, as Python's `value in names` answers for a hashable value.
fn member(value: &Value, names: &BTreeSet<String>) -> bool {
    matches!(value, Value::String(text) if names.contains(text))
}

const PROVIDER_FIELDS: [&str; 5] = [
    "credential_validation",
    "default_model",
    "id",
    "models",
    "title",
];

fn provider(value: &Value, seen: &BTreeSet<String>) -> Result<(String, String), PolicyError> {
    let provider = exact_object(value, &PROVIDER_FIELDS, "provider is invalid")?;
    let id = bounded_text(field(provider, "id"), 128, "provider id is invalid")?;
    if !identifier(id) || seen.contains(id) {
        return Err(PolicyError("provider id is invalid"));
    }
    bounded_text(field(provider, "title"), 80, "provider title is invalid")?;
    let models = model_ids(field(provider, "models"))?;
    if !member(field(provider, "default_model"), &models) {
        return Err(PolicyError("provider default model is invalid"));
    }
    Ok((
        id.to_owned(),
        provider_endpoint(field(provider, "credential_validation"))?,
    ))
}

/// Python's `catalog["schema"] != 1 or isinstance(catalog["schema"], bool)`: only the number one is admitted.
fn schema_one(value: &Value) -> bool {
    match value {
        Value::Number(Number::Int(schema)) => *schema == 1,
        Value::Number(Number::Float(schema)) => schema.to_bits() == 1.0f64.to_bits(),
        _ => false,
    }
}

fn catalog_bytes(path: &Path) -> Result<Vec<u8>, PolicyError> {
    let metadata =
        fs::metadata(path).map_err(|_| PolicyError("provider catalog is unavailable"))?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || !(1..=MAX_CATALOG_BYTES).contains(&metadata.len())
    {
        return Err(PolicyError("provider catalog metadata is invalid"));
    }
    fs::read(path).map_err(|_| PolicyError("provider catalog is unavailable"))
}

/// The exact provider hosts of one valid immutable catalog.
///
/// The catalog is read as one strict JSON document (`.standards/strict-json`): a catalog with a repeated field, a
/// non-finite number, or a byte-order mark is refused besides every catalog the Python policy refused.
///
/// # Errors
///
/// The catalog is missing, not a single bounded regular file, not strict JSON, or breaks any field rule.
pub fn load_provider_hosts(path: &Path) -> Result<BTreeSet<String>, PolicyError> {
    let catalog = strict_json::parse(&catalog_bytes(path)?)
        .map_err(|_| PolicyError("provider catalog is invalid"))?;
    let catalog = exact_object(
        &catalog,
        &["default_provider", "providers", "schema"],
        "provider catalog is invalid",
    )?;
    if !schema_one(field(catalog, "schema")) {
        return Err(PolicyError("provider catalog schema is invalid"));
    }
    let Value::Array(providers) = field(catalog, "providers") else {
        return Err(PolicyError("provider catalog providers are invalid"));
    };
    if !(1..=16).contains(&providers.len()) {
        return Err(PolicyError("provider catalog providers are invalid"));
    }
    let (mut ids, mut hosts) = (BTreeSet::new(), BTreeSet::new());
    for value in providers {
        let (id, endpoint) = provider(value, &ids)?;
        ids.insert(id);
        hosts.insert(endpoint);
    }
    if !member(field(catalog, "default_provider"), &ids) {
        return Err(PolicyError("provider catalog default is invalid"));
    }
    Ok(hosts)
}

#[cfg(test)]
mod tests;

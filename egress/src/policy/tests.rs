use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use strict_json::Value;

use super::{DECISION_HOSTS, PolicyError, load_provider_hosts};

const CATALOG: &str = include_str!("../../../../.standards/model_catalog.json");

/// A private directory removed at drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let mut bytes = [0u8; 8];
        getrandom_free(&mut bytes);
        let path = std::env::temp_dir().join(format!(
            "brain-egress-policy-{}",
            egress_transport::audit::hex(&bytes)
        ));
        fs::create_dir(&path).expect("scratch");
        Self(path)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, bytes).expect("write");
        path
    }
}

/// Distinct names without a random source: the process id and a counter.
fn getrandom_free(bytes: &mut [u8; 8]) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::SeqCst);
    bytes[..4].copy_from_slice(&std::process::id().to_be_bytes());
    bytes[4..].copy_from_slice(&counter.to_be_bytes());
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn canonical() -> Value {
    strict_json::parse(CATALOG.as_bytes()).expect("the canonical catalog")
}

/// The compact JSON text of a value, for writing a mutated catalog.
fn text(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(strict_json::Number::Int(number)) => number.to_string(),
        Value::Number(strict_json::Number::BigInt(number)) => number.clone(),
        Value::Number(strict_json::Number::Float(number)) => format!("{number:?}"),
        Value::String(string) => format!("{string:?}"),
        Value::Array(items) => {
            format!("[{}]", items.iter().map(text).collect::<Vec<_>>().join(","))
        }
        Value::Object(members) => format!(
            "{{{}}}",
            members
                .iter()
                .map(|(key, value)| format!("{key:?}:{}", text(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn member<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
    match value {
        Value::Object(members) => {
            &mut members
                .iter_mut()
                .find(|(name, _)| name == key)
                .expect("member")
                .1
        }
        _ => panic!("not an object"),
    }
}

fn item(value: &mut Value, index: usize) -> &mut Value {
    match value {
        Value::Array(items) => &mut items[index],
        _ => panic!("not an array"),
    }
}

fn load(
    scratch: &Scratch,
    value: &Value,
) -> Result<std::collections::BTreeSet<String>, PolicyError> {
    load_provider_hosts(&scratch.write("catalog.json", text(value).as_bytes()))
}

#[test]
fn the_canonical_catalog_derives_only_exact_provider_hosts() {
    let scratch = Scratch::new();
    let hosts = load(&scratch, &canonical()).expect("valid");
    assert_eq!(
        hosts.into_iter().collect::<Vec<_>>(),
        ["api.anthropic.com", "api.openai.com"]
    );
    assert_eq!(DECISION_HOSTS, ["api.typesafe.ai"]);
}

#[test]
fn missing_malformed_and_unbounded_catalogs_are_refused() {
    let scratch = Scratch::new();
    assert_eq!(
        load_provider_hosts(&scratch.0.join("missing")),
        Err(PolicyError("provider catalog is unavailable"))
    );
    assert_eq!(
        load_provider_hosts(&scratch.write("broken.json", b"{")),
        Err(PolicyError("provider catalog is invalid"))
    );
    for path in [
        scratch.0.clone(),
        scratch.write("empty.json", b""),
        scratch.write("large.json", &vec![b' '; 256 * 1024 + 1]),
    ] {
        assert_eq!(
            load_provider_hosts(&path),
            Err(PolicyError("provider catalog metadata is invalid"))
        );
    }
    let linked = scratch.write("linked.json", CATALOG.as_bytes());
    fs::hard_link(&linked, scratch.0.join("second-link.json")).expect("link");
    assert_eq!(
        load_provider_hosts(&linked),
        Err(PolicyError("provider catalog metadata is invalid"))
    );
    let unreadable = scratch.write("unreadable.json", CATALOG.as_bytes());
    fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o000)).expect("mode");
    let refused = load_provider_hosts(&unreadable);
    if fs::read(&unreadable).is_err() {
        assert_eq!(refused, Err(PolicyError("provider catalog is unavailable")));
    }
    for duplicate in [&br#"{"schema":1,"schema":1}"#[..], b"\xef\xbb\xbf{}"] {
        assert_eq!(
            load_provider_hosts(&scratch.write("strict.json", duplicate)),
            Err(PolicyError("provider catalog is invalid"))
        );
    }
}

fn refused(mutate: impl Fn(&mut Value)) -> PolicyError {
    let scratch = Scratch::new();
    let mut catalog = canonical();
    mutate(&mut catalog);
    load(&scratch, &catalog).expect_err("refused")
}

fn provider(catalog: &mut Value) -> &mut Value {
    item(member(catalog, "providers"), 0)
}

fn model(catalog: &mut Value) -> &mut Value {
    item(member(provider(catalog), "models"), 0)
}

#[test]
fn inexact_empty_duplicate_and_wildcard_policies_are_refused() {
    let invalid = PolicyError("provider catalog is invalid");
    assert_eq!(
        refused(|catalog| *catalog = Value::Array(Vec::new())),
        invalid
    );
    assert_eq!(
        refused(|catalog| {
            if let Value::Object(members) = catalog {
                members.push(("unexpected".to_owned(), Value::Bool(true)));
            }
        }),
        invalid
    );
    assert_eq!(
        refused(|catalog| *member(catalog, "providers") = Value::Array(Vec::new())),
        PolicyError("provider catalog providers are invalid")
    );
    assert_eq!(
        refused(|catalog| *member(catalog, "providers") = Value::Null),
        PolicyError("provider catalog providers are invalid")
    );
    assert_eq!(
        refused(|catalog| {
            let id = member(provider(catalog), "id").clone();
            *member(item(member(catalog, "providers"), 1), "id") = id;
        }),
        PolicyError("provider id is invalid")
    );
    assert_eq!(
        refused(|catalog| {
            *member(member(provider(catalog), "credential_validation"), "host") =
                Value::String("*".to_owned());
        }),
        PolicyError("provider endpoint is invalid")
    );
    assert_eq!(
        refused(
            |catalog| *member(catalog, "default_provider") = Value::String("missing".to_owned())
        ),
        PolicyError("provider catalog default is invalid")
    );
}

type Case = (fn(&mut Value), &'static str);

fn assert_refused(cases: &[Case]) {
    for (mutate, message) in cases {
        assert_eq!(refused(mutate), PolicyError(message));
    }
}

#[test]
fn model_field_types_and_bounds_fail_closed() {
    assert_refused(&[
        (
            |c| *member(provider(c), "title") = Value::String(String::new()),
            "provider title is invalid",
        ),
        (
            |c| *member(provider(c), "models") = Value::Array(Vec::new()),
            "provider models are invalid",
        ),
        (
            |c| *member(provider(c), "models") = Value::Null,
            "provider models are invalid",
        ),
        (
            |c| *member(model(c), "id") = Value::String("INVALID".to_owned()),
            "model id is invalid",
        ),
        (
            |c| *member(model(c), "id") = Value::Null,
            "model id is invalid",
        ),
        (
            |c| *member(model(c), "input_usd_per_million_cents") = Value::Bool(true),
            "model price is invalid",
        ),
        (
            |c| {
                *member(model(c), "output_usd_per_million_cents") =
                    Value::Number(strict_json::Number::Int(-1));
            },
            "model price is invalid",
        ),
        (
            |c| *member(model(c), "chat") = Value::String("yes".to_owned()),
            "model chat admission is invalid",
        ),
        (
            |c| *member(model(c), "title") = Value::String("x".repeat(121)),
            "model title is invalid",
        ),
        (|c| *model(c) = Value::Null, "provider model is invalid"),
    ]);
}

#[test]
fn a_repeated_model_id_is_refused() {
    let duplicate = refused(|catalog| {
        let first = model(catalog).clone();
        if let Value::Array(models) = member(provider(catalog), "models") {
            models.push(first);
        }
    });
    assert_eq!(duplicate, PolicyError("model id is invalid"));
}

#[test]
fn provider_field_types_and_bounds_fail_closed() {
    assert_refused(&[
        (
            |c| *member(provider(c), "default_model") = Value::String("missing".to_owned()),
            "provider default model is invalid",
        ),
        (
            |c| *member(c, "schema") = Value::Bool(true),
            "provider catalog schema is invalid",
        ),
        (
            |c| *member(provider(c), "id") = Value::String("-x".to_owned()),
            "provider id is invalid",
        ),
        (|c| *provider(c) = Value::Bool(false), "provider is invalid"),
    ]);
}

#[test]
fn endpoint_rules_match_the_python_host_pattern() {
    let endpoint = |host: &str, path: &str| {
        let scratch = Scratch::new();
        let mut catalog = canonical();
        let validation = member(provider(&mut catalog), "credential_validation");
        *member(validation, "host") = Value::String(host.to_owned());
        *member(validation, "path") = Value::String(path.to_owned());
        load(&scratch, &catalog)
    };
    for host in [
        "localhost",
        "a-b.example",
        "x1.y2",
        &format!("{}.com", "a".repeat(63)),
    ] {
        assert!(endpoint(host, "/v1").is_ok(), "{host}");
    }
    for (host, path) in [
        ("API.openai.com", "/v1"),
        ("-a.example", "/v1"),
        ("a-.example", "/v1"),
        ("a..example", "/v1"),
        ("a_b.example", "/v1"),
        (&format!("{}.com", "a".repeat(64)), "/v1"),
        ("api.example.com", "v1"),
        ("", "/v1"),
        ("api.example.com", ""),
    ] {
        assert!(endpoint(host, path).is_err(), "{host} {path}");
    }
}

#[test]
fn only_the_number_one_is_the_schema_and_identifiers_match_the_python_pattern() {
    let schema = |value: Value| {
        let scratch = Scratch::new();
        let mut catalog = canonical();
        *member(&mut catalog, "schema") = value;
        load(&scratch, &catalog)
    };
    assert!(schema(Value::Number(strict_json::Number::Float(1.0))).is_ok());
    for value in [
        Value::Number(strict_json::Number::Float(0.999_999_999_999_999_9)),
        Value::Number(strict_json::Number::Int(2)),
        Value::String("1".to_owned()),
    ] {
        assert!(schema(value).is_err());
    }
    for (id, valid) in [
        ("gpt.5_mini-x", true),
        ("9", true),
        (&"a".repeat(128)[..], true),
        ("_a", false),
        ("a b", false),
    ] {
        let scratch = Scratch::new();
        let mut catalog = canonical();
        *member(model(&mut catalog), "id") = Value::String(id.to_owned());
        *member(provider(&mut catalog), "default_model") = Value::String(id.to_owned());
        assert_eq!(load(&scratch, &catalog).is_ok(), valid, "{id}");
    }
}

//! The packaged model catalog (`model_catalog.json`, mirrored from `.standards/model_catalog.json`): which providers
//! and models exist, and which of them a person may choose for a Team's chat.

use std::sync::LazyLock;

use crate::json::{self, Value};

/// The catalog's bytes, exactly as mirrored.
pub const CATALOG: &[u8] = include_bytes!("../../model_catalog.json");

/// One catalog model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    /// The provider's model id.
    pub id: String,
    /// Whether a person may choose it for a Team's chat (the rest are models the Brain runs itself).
    pub chat: bool,
}

/// One catalog provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provider {
    /// `openai` or `anthropic`.
    pub id: String,
    /// Its models.
    pub models: Vec<Model>,
}

static PROVIDERS: LazyLock<Vec<Provider>> =
    LazyLock::new(|| providers(CATALOG).unwrap_or_default());

fn providers(bytes: &[u8]) -> Option<Vec<Provider>> {
    let catalog = json::parse(bytes).ok()?;
    json::array_of(&catalog, "providers")?
        .iter()
        .map(|provider| {
            let models = json::array_of(provider, "models")?
                .iter()
                .map(|model| {
                    Some(Model {
                        id: json::str_of(model, "id")?.to_owned(),
                        chat: json::bool_of(model, "chat")?,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Provider {
                id: json::str_of(provider, "id")?.to_owned(),
                models,
            })
        })
        .collect()
}

/// Every catalog provider.
#[must_use]
pub fn all() -> &'static [Provider] {
    &PROVIDERS
}

/// Whether `provider` is a catalog provider.
#[must_use]
pub fn is_provider(provider: &str) -> bool {
    all().iter().any(|entry| entry.id == provider)
}

/// Whether `model` is one of `provider`'s catalog models.
#[must_use]
pub fn is_model(provider: &str, model: &str) -> bool {
    all()
        .iter()
        .any(|entry| entry.id == provider && entry.models.iter().any(|item| item.id == model))
}

/// Whether `model` is one of `provider`'s chat models.
#[must_use]
pub fn is_chat_model(provider: &str, model: &str) -> bool {
    all().iter().any(|entry| {
        entry.id == provider
            && entry
                .models
                .iter()
                .any(|item| item.id == model && item.chat)
    })
}

/// The parsed catalog document, for tests.
#[must_use]
pub fn document() -> Option<Value> {
    json::parse(CATALOG).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_packaged_catalog_names_both_providers_and_their_chat_models() {
        assert!(document().is_some());
        assert!(is_provider("openai") && is_provider("anthropic") && !is_provider("other"));
        assert!(is_chat_model("openai", "gpt-6-luna"));
        assert!(
            is_model("anthropic", "claude-haiku-5-5")
                && !is_chat_model("anthropic", "claude-haiku-5-5")
        );
        assert!(!is_model("openai", "claude-haiku-5-5"));
        assert!(providers(b"{}").is_none());
    }
}

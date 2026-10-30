//! The stateless operations Team asks the Brain: Action purposes, capability plans, intent routes (with the Jev fast
//! path), and the Browser agent's step, fold, completeness check, and final answer. Each reads its closed request,
//! makes its provider calls on the request's [`crate::provider::Scope`], and returns its answer without `usage`, which
//! the server adds.

mod answer;
pub mod browser;
pub mod capability;
pub mod intent;
pub mod jev;
pub mod purpose;
#[cfg(test)]
mod testing;

use crate::errors::BrainError;
use crate::json::{self, Value};
use crate::provider::structured::{self, Ask};
use crate::provider::{
    CallError, ChatRequest, Profile, Projection, ProviderConfig, Providers, Scope,
};
use crate::server::input::RouteError;

/// One stateless operation's provider calls: the process's providers, the request's scope, the model the calls run
/// on, how they are bounded, and the 409 text a cancelled call answers with.
struct Call<'a> {
    providers: &'a Providers,
    scope: &'a Scope,
    config: &'a ProviderConfig,
    profile: Profile,
    cancelled: &'static str,
}

/// One structured answer's model: its name, the JSON Schema the Python model produced, and the longest answer text.
struct Answer<'a> {
    name: &'a str,
    schema: &'a Value,
    max_chars: usize,
}

impl Call<'_> {
    /// The route error a failed call answers with.
    fn failed(&self, error: CallError) -> RouteError {
        error.into_brain(self.cancelled).into()
    }

    /// One schema-valid structured answer to `system` and the compact JSON of `payload` (`structured.messages`).
    fn decide(
        &self,
        system: &str,
        payload: &Value,
        answer: &Answer<'_>,
    ) -> Result<Value, RouteError> {
        let ask = Ask {
            system: Some(system),
            user: vec![crate::conversation::Block::Text(json::compact(payload))],
            name: answer.name,
            schema: answer.schema,
            max_chars: answer.max_chars,
        };
        self.ask(&ask)
    }

    /// One schema-valid structured answer to `ask`.
    fn ask(&self, ask: &Ask<'_>) -> Result<Value, RouteError> {
        let value = structured::value(self.providers, self.scope, self.config, self.profile, ask)
            .map_err(|error| self.failed(error))?;
        conforming(ask.schema, value)
    }

    /// One schema-valid structured answer to a whole conversation.
    fn converse(
        &self,
        mut request: ChatRequest,
        projection: Option<&Projection>,
        answer: &Answer<'_>,
    ) -> Result<Value, RouteError> {
        request.output = Some(crate::provider::OutputSchema {
            name: answer.name.to_owned(),
            schema: answer.schema.clone(),
        });
        let value = structured::conversation(
            self.providers,
            self.scope,
            self.config,
            self.profile,
            &request,
            projection,
            answer.max_chars,
        )
        .map_err(|error| self.failed(error))?;
        conforming(answer.schema, value)
    }
}

/// `value` when it conforms to `schema` (the model's own validation), else a provider response failure.
fn conforming(schema: &Value, value: Value) -> Result<Value, RouteError> {
    if answer::conforms(schema, &value) {
        Ok(value)
    } else {
        Err(broken())
    }
}

/// A provider answer that broke its closed contract: 502, without its content.
fn broken() -> RouteError {
    BrainError::Provider.into()
}

/// A contract refusal of the request: 400 with `detail`.
fn refused(detail: &str) -> RouteError {
    BrainError::contract(detail).into()
}

/// Whether every character of `value` is outside Unicode's `C` categories, except a line feed or tab where `layout`
/// admits them.
fn no_controls(value: &str, layout: bool) -> bool {
    value.chars().all(|character| {
        !team_protocol::python::text::category(character).starts_with('C')
            || (layout && matches!(character, '\n' | '\t'))
    })
}

/// Whether `values` are strictly ascending, so sorted and distinct (`values == sorted(set(values))`).
fn sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

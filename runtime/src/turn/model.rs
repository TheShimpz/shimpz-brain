//! The one model Holdpoint calls for a turn: a chat call or the memory check, on the provider of the request that
//! drives the run, with that request's key, attachments, cancellation, and usage.
//!
//! A Holdpoint attempt runs on its own thread; while the provider call waits, a watcher polls the attempt and
//! cancels the call's nested scope once Holdpoint abandons it (a timeout or the run's cancellation), so the
//! connection is dropped at once. Failures carry fixed codes only: `cancelled`, `contract` (with the attachment
//! ceiling's fixed text), or `provider`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use holdpoint::{Failure, Model, ModelCall, ModelReply};

use crate::attachments;
use crate::conversation::{self, Block, Message};
use crate::json::{self, Value};
use crate::provider::structured::{self, Ask};
use crate::provider::{CallError, ChatRequest, Profile, ProviderConfig, Providers, Scope};
use crate::tools::memory;
use crate::turn::host::TurnSession;
use crate::turn::run;

/// The provider model of chat turns.
#[derive(Debug, Clone)]
pub struct ProviderModel {
    providers: Providers,
}

impl ProviderModel {
    /// The model over `providers`.
    #[must_use]
    pub fn new(providers: Providers) -> Self {
        Self { providers }
    }
}

fn failure(error: CallError) -> Failure {
    match error {
        CallError::Cancelled => Failure::permanent("cancelled", "the turn was cancelled"),
        CallError::Contract(detail) => Failure::permanent("contract", &detail),
        CallError::Request | CallError::Response => {
            Failure::permanent("provider", "the provider call failed")
        }
    }
}

fn invalid() -> Failure {
    Failure::permanent("provider", "the model call is invalid").with_ended()
}

/// The chat request a journaled call describes.
fn chat_request(call: &Value) -> Option<ChatRequest> {
    let tools = json::array_of(call, "tools")?
        .iter()
        .map(run::tool_from)
        .collect::<Option<Vec<_>>>()?;
    Some(ChatRequest {
        system: Some(json::str_of(call, "system")?.to_owned()),
        messages: conversation::from_value(call.get("messages")?)?,
        tools,
        output: None,
        cache: json::bool_of(call, "cache")?,
    })
}

impl ProviderModel {
    fn chat(&self, scope: &Scope, session: &TurnSession, call: &Value) -> Result<Value, Failure> {
        let request = chat_request(call).ok_or_else(invalid)?;
        let charge = json::u64_of(call, "charge").ok_or_else(invalid)?;
        let reserved = u32::try_from(json::u64_of(call, "reserved").ok_or_else(invalid)?)
            .map_err(|_| invalid())?;
        let message_id = json::str_of(call, "turn_message_id").ok_or_else(invalid)?;
        let projection = attachments::projection(message_id, &session.attachments);
        let admit = move |attempt: u32| attachments::admit_call(charge, reserved + attempt - 1);
        let admit: Option<crate::provider::Admit<'_>> = if session.attachments.is_empty() {
            None
        } else {
            Some(&admit)
        };
        let message = self
            .providers
            .chat(
                scope,
                &session.provider,
                &request,
                projection.as_ref(),
                Profile::CHAT,
                admit,
            )
            .map_err(failure)?;
        Ok(Message::Assistant(message).to_value())
    }

    fn check(
        &self,
        scope: &Scope,
        session: &TurnSession,
        call: &Value,
    ) -> Result<Option<Value>, Failure> {
        let prompt = json::str_of(call, "prompt").ok_or_else(invalid)?;
        let schema = memory::confirmation_schema();
        let ask = Ask {
            system: None,
            user: vec![Block::Text(prompt.to_owned())],
            name: "Confirmation",
            schema: &schema,
            max_chars: memory::MAX_CONFIRMATION_CHARS,
        };
        match structured::value(
            &self.providers,
            scope,
            &session.provider,
            Profile::CHAT,
            &ask,
        ) {
            Ok(value) => Ok(Some(value)),
            Err(CallError::Cancelled) => Err(failure(CallError::Cancelled)),
            Err(_) => Ok(None),
        }
    }

    fn answer(
        &self,
        scope: &Scope,
        session: &TurnSession,
        call: &Value,
    ) -> Result<serde_json::Value, Failure> {
        let same_provider =
            json::str_of(call, "provider") == Some(session.provider.provider.as_str());
        if !same_provider || json::str_of(call, "model").is_none() {
            return Err(invalid());
        }
        let config = ProviderConfig {
            provider: session.provider.provider,
            model: json::str_of(call, "model").unwrap_or_default().to_owned(),
            api_key: session.provider.api_key.clone(),
        };
        let session = TurnSession {
            provider: config,
            attachments: session.attachments.clone(),
            scope: session.scope.clone(),
        };
        match json::str_of(call, "kind") {
            Some("chat") => self
                .chat(scope, &session, call)
                .map(|value| run::answer_json("message", Some(&value))),
            Some("check") => self
                .check(scope, &session, call)
                .map(|value| run::answer_json("verdict", value.as_ref())),
            _ => Err(invalid()),
        }
    }
}

impl Model for ProviderModel {
    fn reserve(&self, _request: &serde_json::Value) -> Option<u64> {
        Some(0)
    }

    fn call(&self, call: ModelCall<'_>) -> Result<ModelReply, Failure> {
        let Some(session) = call.credentials.get::<Arc<TurnSession>>() else {
            return Err(Failure::permanent("provider", "no session drives this run").with_ended());
        };
        let request = run::call_from(call.request).ok_or_else(invalid)?;
        let scope = session.scope.child();
        let done = AtomicBool::new(false);
        let answered = std::thread::scope(|threads| {
            threads.spawn(|| {
                while !done.load(Ordering::SeqCst) {
                    if call.attempt.check().is_err() {
                        scope.cancel.cancel();
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            });
            let answered = self.answer(&scope, session, &request);
            done.store(true, Ordering::SeqCst);
            answered
        });
        answered.map(|response| ModelReply { response, cost: 0 })
    }
}

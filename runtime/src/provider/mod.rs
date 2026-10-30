//! Model providers: the Brain-owned OpenAI Responses and Anthropic Messages adapters over one bounded, cancellable
//! HTTPS transport, the per-operation usage they report, and the call profiles the Brain uses (ADR-0113 efforts).
//!
//! Every call is synchronous for its caller and must run on a blocking thread (a `spawn_blocking` task or a Holdpoint
//! attempt thread): it drives the async transport on the process runtime and returns when the provider answered, the
//! call failed, or its [`Scope`] was cancelled. A provider's error text never leaves this module; keys travel only in
//! request headers and are zeroized with their [`ProviderConfig`].

pub mod anthropic;
pub mod openai;
pub mod structured;
pub mod transport;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use zeroize::Zeroizing;

use crate::cancel::CancelToken;
use crate::conversation::{AssistantMessage, Block, Message};
use crate::json::{self, Value};
pub use transport::{Host, HttpRequest, HttpResponse, Route, Transport, TransportError};

/// The two catalog providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderId {
    /// OpenAI (Responses API).
    OpenAi,
    /// Anthropic (Messages API).
    Anthropic,
}

impl ProviderId {
    /// The catalog id.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }

    /// The provider named `id`.
    #[must_use]
    pub fn parse(id: &str) -> Option<Self> {
        match id {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            _ => None,
        }
    }
}

/// One operation's provider, model, and request-scoped key. The key is zeroized on drop and never printed.
#[derive(Clone)]
pub struct ProviderConfig {
    /// The provider.
    pub provider: ProviderId,
    /// The provider's model id.
    pub model: String,
    /// The API key Team sent with this request.
    pub api_key: Zeroizing<String>,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// One tool offered to a model call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    /// The tool's name.
    pub name: String,
    /// Its description.
    pub description: String,
    /// Its JSON Schema (an object schema), in its producer's member order.
    pub schema: Value,
    /// OpenAI's `strict` flag: `Some(false)` for Action tools (ADR-0094), `None` for the Brain's own tools.
    pub strict: Option<bool>,
}

/// The structured output a call asks for.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSchema {
    /// The schema's name (the Pydantic model name the Python Brain bound).
    pub name: String,
    /// The JSON Schema of the one value the model must write.
    pub schema: Value,
}

/// One model call: everything its provider request is built from, except the key and request-local attachments.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// The system prompt, if any.
    pub system: Option<String>,
    /// The conversation.
    pub messages: Vec<Message>,
    /// The tools offered.
    pub tools: Vec<ToolSpec>,
    /// Structured output, if asked.
    pub output: Option<OutputSchema>,
    /// Anthropic prompt caching of the system prompt, the tools, and the conversation prefix (five-minute TTL).
    pub cache: bool,
}

/// Request-local user content: the blocks a call appends to the user message with `message_id` (ADR-0093). Never
/// stored or journaled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projection {
    /// The turn message that carries the blocks.
    pub message_id: String,
    /// The blocks.
    pub blocks: Vec<Block>,
}

/// How one kind of call is bounded and retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    /// The reasoning effort sent, or none (the provider default).
    pub effort: Option<&'static str>,
    /// Each attempt's time bound.
    pub timeout: Duration,
    /// Attempts of the logical call: one plus the SDK's retries.
    pub attempts: u32,
    /// The output token cap, or the adapter's default.
    pub max_output_tokens: Option<u32>,
}

impl Profile {
    /// An ordinary chat turn and its memory check (ADR-0113: low effort; the SDK's two retries; 60 s).
    pub const CHAT: Self = Self {
        effort: Some("low"),
        timeout: Duration::from_secs(60),
        attempts: 3,
        max_output_tokens: None,
    };
    /// A short structured decision with at most one retry (ADR-0071): low effort, 10 s.
    pub const DECISION: Self = Self {
        effort: Some("low"),
        timeout: Duration::from_secs(10),
        attempts: 2,
        max_output_tokens: None,
    };
    /// A helper call at the provider's default effort, such as capability planning.
    pub const DEFAULT: Self = Self {
        effort: None,
        timeout: Duration::from_secs(60),
        attempts: 3,
        max_output_tokens: None,
    };

    /// This profile with an output cap.
    #[must_use]
    pub const fn capped(self, max_output_tokens: u32) -> Self {
        Self {
            max_output_tokens: Some(max_output_tokens),
            ..self
        }
    }

    /// This profile with another effort.
    #[must_use]
    pub const fn with_effort(self, effort: Option<&'static str>) -> Self {
        Self { effort, ..self }
    }
}

/// Token counts a provider reported for one answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tokens {
    /// Input tokens, cache reads and writes included.
    pub input: u64,
    /// Output tokens.
    pub output: u64,
    /// Input tokens read from the cache.
    pub cache_read: u64,
    /// Input tokens written to the cache.
    pub cache_write: u64,
}

/// One operation's observed model usage (ADR-0082): a floor on what was billed, never an estimate.
#[derive(Debug, Default)]
pub struct Usage {
    model_calls: AtomicU64,
    failed_calls: AtomicU64,
    unreported_calls: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    cache_read_tokens: AtomicU64,
    cache_write_tokens: AtomicU64,
    provider_requests: AtomicU64,
}

/// The usage report's fields, in the order Team's parser names them.
pub const USAGE_FIELDS: [&str; 8] = [
    "model_calls",
    "failed_calls",
    "unreported_calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "provider_requests",
];

impl Usage {
    fn add(counter: &AtomicU64, value: u64) {
        counter.fetch_add(value, Ordering::Relaxed);
    }

    /// One logical model call started.
    pub fn call_started(&self) {
        Self::add(&self.model_calls, 1);
    }

    /// One logical model call failed.
    pub fn call_failed(&self) {
        Self::add(&self.failed_calls, 1);
    }

    /// One HTTP request was sent to a provider (each retry included).
    pub fn request_sent(&self) {
        Self::add(&self.provider_requests, 1);
    }

    /// A logical call answered with `tokens`, or without a usage report.
    pub fn answered(&self, tokens: Option<Tokens>) {
        match tokens {
            None => Self::add(&self.unreported_calls, 1),
            Some(tokens) => {
                Self::add(&self.input_tokens, tokens.input);
                Self::add(&self.output_tokens, tokens.output);
                Self::add(&self.cache_read_tokens, tokens.cache_read);
                Self::add(&self.cache_write_tokens, tokens.cache_write);
            }
        }
    }

    /// The counts in [`USAGE_FIELDS`] order.
    #[must_use]
    pub fn counts(&self) -> [u64; 8] {
        [
            &self.model_calls,
            &self.failed_calls,
            &self.unreported_calls,
            &self.input_tokens,
            &self.output_tokens,
            &self.cache_read_tokens,
            &self.cache_write_tokens,
            &self.provider_requests,
        ]
        .map(|counter| counter.load(Ordering::Relaxed))
    }

    /// The report Team reads: `{"model_calls": n, ...}`.
    #[must_use]
    pub fn to_value(&self) -> Value {
        json::object_of(
            USAGE_FIELDS
                .iter()
                .zip(self.counts())
                .map(|(name, count)| ((*name).to_owned(), json::uint(count)))
                .collect(),
        )
    }
}

/// What every provider call of one operation shares: its cancellation, its usage, and an optional extra stop
/// condition (a Holdpoint attempt that was abandoned).
#[derive(Clone, Default)]
pub struct Scope {
    /// Tripped when Team closes the request.
    pub cancel: CancelToken,
    /// The operation's usage.
    pub usage: Arc<Usage>,
    stop: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

impl std::fmt::Debug for Scope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Scope")
            .field("cancelled", &self.cancel.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Scope {
    /// A scope over `cancel` with fresh usage.
    #[must_use]
    pub fn new(cancel: CancelToken) -> Self {
        Self {
            cancel,
            usage: Arc::default(),
            stop: None,
        }
    }

    /// This scope, also stopped once `stop` returns true (polled while a call waits).
    #[must_use]
    pub fn with_stop(mut self, stop: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        self.stop = Some(Arc::new(stop));
        self
    }

    /// Whether the work must stop now.
    #[must_use]
    pub fn stopped(&self) -> bool {
        self.cancel.is_cancelled() || self.stop.as_ref().is_some_and(|stop| stop())
    }

    /// Resolves once the work must stop.
    pub async fn until_stopped(&self) {
        let poll = async {
            loop {
                if self.stopped() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };
        tokio::select! {
            () = self.cancel.cancelled() => {}
            () = poll => {}
        }
    }
}

/// Why a provider call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallError {
    /// The request failed (unreachable, timed out, refused by status): the Python `ProviderRequestError`.
    Request,
    /// The answer broke the closed contract (not JSON, wrong shape): the Python `ProviderResponseError`.
    Response,
    /// The operation was cancelled.
    Cancelled,
    /// A request-local admission refused the next attempt, with its contract text (an attachment budget).
    Contract(String),
}

/// A provider's wire request: path, headers beyond the key and content type, and body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wire {
    /// The request path.
    pub path: &'static str,
    /// Extra headers.
    pub headers: Vec<(&'static str, String)>,
    /// The JSON body.
    pub body: Vec<u8>,
}

/// Admission run before each attempt of a call; a refusal ends the call with its contract text.
pub type Admit<'a> = &'a (dyn Fn(u32) -> Result<(), String> + Sync);

/// The providers of one process over one transport.
#[derive(Debug, Clone)]
pub struct Providers {
    transport: Arc<Transport>,
}

impl Providers {
    /// The providers over `transport`.
    #[must_use]
    pub fn new(transport: Arc<Transport>) -> Self {
        Self { transport }
    }

    /// The transport.
    #[must_use]
    pub fn transport(&self) -> &Arc<Transport> {
        &self.transport
    }

    /// One logical model call, retried as `profile` allows; `admit` runs before every attempt. The returned turn's
    /// `id` is empty and its `attempts` counts the attempts made.
    ///
    /// # Errors
    ///
    /// [`CallError`]; nothing of the provider's answer or the key.
    pub fn chat(
        &self,
        scope: &Scope,
        config: &ProviderConfig,
        request: &ChatRequest,
        projection: Option<&Projection>,
        profile: Profile,
        admit: Option<Admit<'_>>,
    ) -> Result<AssistantMessage, CallError> {
        let wire = match config.provider {
            ProviderId::OpenAi => openai::request(&config.model, request, projection, profile),
            ProviderId::Anthropic => {
                anthropic::request(&config.model, request, projection, profile)
            }
        };
        let host = match config.provider {
            ProviderId::OpenAi => Host::OpenAi,
            ProviderId::Anthropic => Host::Anthropic,
        };
        scope.usage.call_started();
        let answered = self
            .transport
            .call_with_retries(scope, host, config, &wire, profile, admit);
        let response = match answered {
            Ok(response) => response,
            Err(error) => {
                scope.usage.call_failed();
                return Err(error);
            }
        };
        let parsed = match config.provider {
            ProviderId::OpenAi => openai::response(&response.1.body),
            ProviderId::Anthropic => anthropic::response(&response.1.body),
        };
        let Some((mut message, tokens)) = parsed else {
            scope.usage.call_failed();
            return Err(CallError::Response);
        };
        scope.usage.answered(tokens);
        message.attempts = response.0;
        Ok(message)
    }

    /// The provider's own count of the input tokens of one user message made of `blocks`, in one attempt within
    /// `timeout`. Counting is no model call: it never counts as one, only as a provider request.
    ///
    /// # Errors
    ///
    /// [`CallError`] when the provider did not count in time.
    pub fn count_tokens(
        &self,
        scope: &Scope,
        config: &ProviderConfig,
        blocks: &[Block],
        timeout: Duration,
    ) -> Result<u64, CallError> {
        let (host, wire) = match config.provider {
            ProviderId::OpenAi => (Host::OpenAi, openai::count_request(&config.model, blocks)),
            ProviderId::Anthropic => (
                Host::Anthropic,
                anthropic::count_request(&config.model, blocks),
            ),
        };
        let profile = Profile {
            effort: None,
            timeout,
            attempts: 1,
            max_output_tokens: None,
        };
        let (_, response) = self
            .transport
            .call_with_retries(scope, host, config, &wire, profile, None)?;
        let counted = json::parse(&response.body)
            .ok()
            .and_then(|value| json::u64_of(&value, "input_tokens"));
        counted.ok_or(CallError::Response)
    }
}

impl CallError {
    /// The Brain error this call failure answers with, `cancelled` naming the operation's 409 text.
    #[must_use]
    pub fn into_brain(self, cancelled: &'static str) -> crate::errors::BrainError {
        use crate::errors::BrainError;
        match self {
            Self::Request | Self::Response => BrainError::Provider,
            Self::Cancelled => BrainError::Cancelled(cancelled),
            Self::Contract(detail) => BrainError::Contract(detail),
        }
    }
}

//! The Browser agent's stateless calls (`browser_step.py`, `browser_thread.py`, `browser_api.py`, ADR-0117 §5).
//!
//! Team sends the objective, the language, the task's bounded thread, the current page observation, the downloads
//! present so far, and at most one screenshot; the Brain answers on the fixed Browser model of the Team's provider.
//! On OpenAI the thread is rendered as one growing conversation at medium effort, with a read step, the agent's fold
//! notes, the completeness check, and a final answer; Claude Haiku 5.5 keeps its earlier step, which sees the
//! newest step records only, and answers no fold, check, or final answer. The page is untrusted content, never
//! instructions. The Brain keeps nothing between calls.

mod haiku;
mod thread;

use std::sync::LazyLock;

use base64::Engine as _;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use team_protocol::browser::{
    self as protocol, LANGUAGES, MAX_CALL_INPUT_TOKENS, MAX_CALL_OUTPUT_TOKENS, MAX_MEMORY_BYTES,
    MAX_OBJECTIVE_CHARS, MAX_OBSERVATION_BYTES, MAX_REQUEST_TEXT_BYTES, MAX_STEPS,
    MAX_SUMMARY_BYTES,
};
use team_protocol::python::text as python_text;
use team_protocol::python::value::py_eq;

use super::{Answer, Call, answer, broken, refused};
use crate::conversation::{Block, Message};
use crate::json::{self, Value};
use crate::provider::{ChatRequest, Profile, Projection, ProviderId, Providers, Scope};
use crate::server::input::{self, RouteError};

/// The 409 text of a cancelled Browser operation.
const CANCELLED: &str = "Browser step cancelled";
/// A memory rewrite is asked to stay within this many characters, so its UTF-8 bytes stay within the bound.
pub const MEMORY_CHARS: usize = MAX_MEMORY_BYTES * 3 / 4;
/// What no prompt can cost beyond its text: one screenshot's tokens.
pub const IMAGE_TOKENS: usize = 3_000;
/// The step schema's allowance.
pub const SCHEMA_TOKENS: usize = 4_000;
/// The longest answer text a structured Browser answer may have.
const MAX_RESPONSE_CHARS: usize = 8 * 1024;
/// A step's whole answer fits well within this output cap.
pub const MAX_STEP_OUTPUT_TOKENS: u32 = 2_048;
/// A fold note or a final answer fits within the protocol's call cap.
pub const MAX_WRITE_OUTPUT_TOKENS: u32 = MAX_CALL_OUTPUT_TOKENS;
/// The newest step records Haiku's step sees.
pub const HAIKU_RECORDS: usize = 40;
/// The largest screenshot, decoded.
pub const MAX_SCREENSHOT_BYTES: usize = 2 * 1024 * 1024;
const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
/// Python's `base64.b64decode(value, validate=True)`: the standard alphabet with canonical padding, unused low bits
/// of the last character ignored.
const BASE64: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    GeneralPurposeConfig::new()
        .with_decode_allow_trailing_bits(true)
        .with_decode_padding_mode(DecodePaddingMode::RequireCanonical),
);

/// `StepAnswer.model_json_schema()`: Haiku's flat step.
static STEP_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    answer::schema(
        r#"{"additionalProperties":false,"description":"The flat step every answer fills, the unused fields null; Team's protocol admits only its canonical form.","properties":{"kind":{"enum":["navigate","click","type","scroll","wait","screenshot","done","needs_person"],"title":"Kind","type":"string"},"url":{"anyOf":[{"maxLength":2048,"type":"string"},{"type":"null"}],"title":"Url"},"ref":{"anyOf":[{"maxLength":16,"type":"string"},{"type":"null"}],"title":"Ref"},"text":{"anyOf":[{"maxLength":2000,"type":"string"},{"type":"null"}],"title":"Text"},"direction":{"anyOf":[{"enum":["up","down"],"type":"string"},{"type":"null"}],"title":"Direction"},"summary":{"anyOf":[{"maxLength":4096,"type":"string"},{"type":"null"}],"title":"Summary"},"reason":{"anyOf":[{"enum":["sign-in","captcha","payment","confirmation","blocked"],"type":"string"},{"type":"null"}],"title":"Reason"},"sentence":{"anyOf":[{"maxLength":300,"type":"string"},{"type":"null"}],"title":"Sentence"},"note":{"anyOf":[{"maxLength":300,"type":"string"},{"type":"null"}],"title":"Note"}},"required":["kind","url","ref","text","direction","summary","reason","sentence","note"],"title":"StepAnswer","type":"object"}"#,
    )
});

/// `ThreadAnswer.model_json_schema()`: the OpenAI thread's step, which may also read up to ten links.
static THREAD_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    answer::schema(
        r#"{"additionalProperties":false,"description":"The OpenAI thread's step, which may also read up to ten links of its page.","properties":{"kind":{"enum":["navigate","click","type","scroll","wait","screenshot","read","done","needs_person"],"title":"Kind","type":"string"},"url":{"anyOf":[{"maxLength":2048,"type":"string"},{"type":"null"}],"title":"Url"},"ref":{"anyOf":[{"maxLength":16,"type":"string"},{"type":"null"}],"title":"Ref"},"text":{"anyOf":[{"maxLength":2000,"type":"string"},{"type":"null"}],"title":"Text"},"direction":{"anyOf":[{"enum":["up","down"],"type":"string"},{"type":"null"}],"title":"Direction"},"summary":{"anyOf":[{"maxLength":4096,"type":"string"},{"type":"null"}],"title":"Summary"},"reason":{"anyOf":[{"enum":["sign-in","captcha","payment","confirmation","blocked"],"type":"string"},{"type":"null"}],"title":"Reason"},"sentence":{"anyOf":[{"maxLength":300,"type":"string"},{"type":"null"}],"title":"Sentence"},"note":{"anyOf":[{"maxLength":300,"type":"string"},{"type":"null"}],"title":"Note"},"refs":{"anyOf":[{"items":{"type":"string"},"maxItems":10,"type":"array"},{"type":"null"}],"title":"Refs"}},"required":["kind","url","ref","text","direction","summary","reason","sentence","note","refs"],"title":"ThreadAnswer","type":"object"}"#,
    )
});

/// `Completeness.model_json_schema()`.
static CHECK_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    answer::schema(
        r#"{"additionalProperties":false,"properties":{"complete":{"title":"Complete","type":"boolean"},"missing":{"anyOf":[{"maxLength":300,"type":"string"},{"type":"null"}],"title":"Missing"}},"required":["complete","missing"],"title":"Completeness","type":"object"}"#,
    )
});

/// A step request's members (`BrowserStepInput`), all required.
const STEP_FIELDS: [&str; 7] = [
    "provider",
    "objective",
    "language",
    "thread",
    "observation",
    "screenshot",
    "downloads",
];
/// A completeness check's members (`BrowserCheckInput`), all required.
const CHECK_FIELDS: [&str; 4] = ["provider", "objective", "language", "summary"];

/// The four Browser operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// `POST /v1/browser-step`.
    Step,
    /// `POST /v1/browser-fold`.
    Fold,
    /// `POST /v1/browser-check`.
    Check,
    /// `POST /v1/browser-final`.
    Final,
}

/// One step request's fields in their canonical forms (`validate`).
#[derive(Debug, Clone, PartialEq)]
struct Admitted {
    objective: String,
    language: String,
    thread: Value,
    observation: String,
    screenshot: Option<String>,
    downloads: Value,
}

/// One rendered call: an optional system prompt, user messages in order, and the request-local blocks that follow
/// the text of one of them (the current page's screenshot, or Haiku's page and screenshot).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Rendered {
    system: Option<&'static str>,
    messages: Vec<String>,
    extra: Option<(usize, Vec<Block>)>,
}

impl Rendered {
    /// `browser_thread.text_bytes`: the UTF-8 bytes of every text part.
    fn text_bytes(&self) -> usize {
        let extra = self
            .extra
            .iter()
            .flat_map(|(_, blocks)| blocks)
            .map(|block| match block {
                Block::Text(text) => text.len(),
                Block::Image { .. } => 0,
            });
        self.system.map_or(0, str::len)
            + self.messages.iter().map(String::len).sum::<usize>()
            + extra.sum::<usize>()
    }

    /// `input_bound`: the most input tokens the call can cost; no token encodes less than one UTF-8 byte.
    fn input_bound(&self) -> usize {
        let images = self
            .extra
            .iter()
            .flat_map(|(_, blocks)| blocks)
            .filter(|block| matches!(block, Block::Image { .. }))
            .count();
        self.text_bytes() + images * IMAGE_TOKENS + SCHEMA_TOKENS
    }

    /// The provider request and its request-local blocks.
    fn request(&self) -> (ChatRequest, Option<Projection>) {
        let id = |index: usize| format!("browser-{index}");
        let messages = self
            .messages
            .iter()
            .enumerate()
            .map(|(index, text)| Message::User {
                id: id(index),
                text: text.clone(),
            })
            .collect();
        let request = ChatRequest {
            system: self.system.map(str::to_owned),
            messages,
            tools: Vec::new(),
            output: None,
            cache: false,
        };
        let projection = self.extra.as_ref().map(|(index, blocks)| Projection {
            message_id: id(*index),
            blocks: blocks.clone(),
        });
        (request, projection)
    }
}

/// One Browser operation: its answer's fields and `model` (the server adds `usage`).
///
/// # Errors
///
/// [`RouteError`]: 422 for the request shape, 400/502, and 409 "Browser step cancelled".
pub fn handle(
    providers: &Providers,
    scope: &Scope,
    operation: Operation,
    body: &Value,
) -> Result<Value, RouteError> {
    let fields = read(operation, body)?;
    let mut config = input::provider(body)?;
    model(config.provider).clone_into(&mut config.model);
    let call = Call {
        providers,
        scope,
        config: &config,
        profile: Profile::CHAT
            .capped(MAX_STEP_OUTPUT_TOKENS)
            .with_effort(Some(effort(config.provider))),
        cancelled: CANCELLED,
    };
    let mut answer = match operation {
        Operation::Step => step(&call, body)?,
        Operation::Fold => fold(&call, body, fields)?,
        Operation::Check => check(&call, body)?,
        Operation::Final => final_answer(&call, body)?,
    };
    if let Value::Object(members) = &mut answer {
        members.push(("model".to_owned(), json::text(&config.model)));
    }
    Ok(answer)
}

/// Each provider's fixed Browser model, as the Browser evaluation selected it (ADR-0117 §5.4).
#[must_use]
pub fn model(provider: ProviderId) -> &'static str {
    match provider {
        ProviderId::OpenAi => "gpt-6-luna",
        ProviderId::Anthropic => "claude-haiku-5-5",
    }
}

/// The reasoning effort each provider's Browser calls run at: the OpenAI thread at medium, Haiku's step at low.
fn effort(provider: ProviderId) -> &'static str {
    match provider {
        ProviderId::OpenAi => "medium",
        ProviderId::Anthropic => "low",
    }
}

/// The largest prompt each provider's Browser model is priced for at its listed rate.
fn priced_input_tokens(provider: ProviderId) -> usize {
    match provider {
        ProviderId::OpenAi => usize::try_from(MAX_CALL_INPUT_TOKENS).unwrap_or(usize::MAX),
        ProviderId::Anthropic => 100_000,
    }
}

/// The closed request object of `operation`, refused whole with 422; a fold's step numbers.
fn read(operation: Operation, body: &Value) -> Result<(i64, i64), RouteError> {
    let fold = [STEP_FIELDS.as_slice(), &["first", "last"]].concat();
    let fields: &[&str] = match operation {
        Operation::Step | Operation::Final => &STEP_FIELDS,
        Operation::Fold => &fold,
        Operation::Check => &CHECK_FIELDS,
    };
    input::closed(body, fields, fields)?;
    input::string(body, "objective", 1, MAX_OBJECTIVE_CHARS)?;
    if !LANGUAGES
        .iter()
        .any(|language| json::str_of(body, "language") == Some(language))
    {
        return Err(RouteError::Invalid);
    }
    if operation == Operation::Check {
        input::string(body, "summary", 1, MAX_SUMMARY_BYTES)?;
        return Ok((0, 0));
    }
    input::object(body, "thread")?;
    input::string(body, "observation", 0, MAX_OBSERVATION_BYTES)?;
    input::optional_string(body, "screenshot", 0, MAX_SCREENSHOT_BYTES * 4 / 3 + 4)?;
    input::optional_object(body, "downloads")?;
    if operation != Operation::Fold {
        return Ok((0, 0));
    }
    let steps = i64::try_from(MAX_STEPS).unwrap_or(i64::MAX);
    let number = |key: &str| {
        input::integer(body, key).and_then(|number| {
            (1..=steps)
                .contains(&number)
                .then_some(number)
                .ok_or(RouteError::Invalid)
        })
    };
    Ok((number("first")?, number("last")?))
}

/// `validate`: the request's closed fields, each in its canonical form, or a contract error.
fn validate(body: &Value) -> Result<Admitted, RouteError> {
    let invalid = || refused("invalid Browser step request");
    let field = |key: &str| body.get(key).unwrap_or(&Value::Null);
    let objective = protocol::canonical_objective(field("objective")).ok_or_else(invalid)?;
    let thread = protocol::canonical_thread(field("thread")).ok_or_else(invalid)?;
    let observation = protocol::canonical_observation(field("observation")).ok_or_else(invalid)?;
    let language = json::str_of(body, "language").ok_or_else(invalid)?;
    let downloads = field("downloads").clone();
    if downloads != Value::Null
        && !protocol::canonical_downloads(&downloads)
            .is_some_and(|canonical| py_eq(&canonical, &downloads))
    {
        return Err(invalid());
    }
    if protocol::request_text_bytes(objective, &thread, observation, &downloads)
        > MAX_REQUEST_TEXT_BYTES
    {
        return Err(invalid());
    }
    Ok(Admitted {
        objective: objective.to_owned(),
        language: language.to_owned(),
        thread,
        observation: observation.to_owned(),
        screenshot: screenshot(field("screenshot"))?,
        downloads,
    })
}

/// `_screenshot`: one PNG within its bound, as strict base64, or none.
fn screenshot(value: &Value) -> Result<Option<String>, RouteError> {
    let text = match value {
        Value::Null => return Ok(None),
        Value::String(text) => text,
        _ => return Err(refused("invalid Browser screenshot")),
    };
    let image = (text.chars().count() <= MAX_SCREENSHOT_BYTES * 4 / 3 + 4)
        .then(|| BASE64.decode(text).ok())
        .flatten();
    match image {
        Some(image) if image.starts_with(PNG_SIGNATURE) && image.len() <= MAX_SCREENSHOT_BYTES => {
            Ok(Some(text.clone()))
        }
        _ => Err(refused("invalid Browser screenshot")),
    }
}

/// `_bounded`: a rendered call within its provider's priced bound, checked before any provider call.
fn bounded(rendered: &Rendered, provider: ProviderId) -> Result<(), RouteError> {
    if rendered.input_bound() > priced_input_tokens(provider) {
        return Err(refused("Browser step request exceeds its priced bound"));
    }
    Ok(())
}

/// One canonical step for the task; an answer outside the closed set is a provider response failure.
fn step(call: &Call<'_>, body: &Value) -> Result<Value, RouteError> {
    let admitted = validate(body)?;
    let threaded = call.config.provider == ProviderId::OpenAi;
    let rendered = if threaded {
        thread::messages(&admitted, None)
    } else {
        haiku::prompt(&admitted)
    };
    bounded(&rendered, call.config.provider)?;
    let answer = Answer {
        name: if threaded {
            "ThreadAnswer"
        } else {
            "StepAnswer"
        },
        schema: if threaded {
            &THREAD_SCHEMA
        } else {
            &STEP_SCHEMA
        },
        max_chars: MAX_RESPONSE_CHARS,
    };
    let (request, projection) = rendered.request();
    let value = call.converse(request, projection.as_ref(), &answer)?;
    let canonical = protocol::canonical_step(&value).ok_or_else(broken)?;
    Ok(json::object([("step", canonical)]))
}

/// A plain-text answer to a rendered call within `limit` UTF-8 bytes, or `None` when nothing usable came back; with
/// `whole`, an answer the provider cut short at its output cap is nothing usable either.
fn written(
    call: &Call<'_>,
    rendered: &Rendered,
    limit: usize,
    whole: bool,
) -> Result<Option<String>, RouteError> {
    bounded(rendered, call.config.provider)?;
    let (request, projection) = rendered.request();
    let profile = call.profile.capped(MAX_WRITE_OUTPUT_TOKENS);
    let reply = call
        .providers
        .chat(
            call.scope,
            call.config,
            &request,
            projection.as_ref(),
            profile,
            None,
        )
        .map_err(|error| call.failed(error))?;
    if whole && reply.truncated {
        return Ok(None);
    }
    Ok(thread::plain(&thread::reply_text(&reply), limit))
}

/// The agent's whole rewritten memory as steps leave its view; Haiku keeps none and makes no call. A rewrite past
/// the memory's bound, or one the provider cut short, is refused, so Team keeps the memory it had.
fn fold(call: &Call<'_>, body: &Value, (first, last): (i64, i64)) -> Result<Value, RouteError> {
    let admitted = validate(body)?;
    if first > last {
        return Err(refused("invalid Browser step request"));
    }
    if call.config.provider != ProviderId::OpenAi {
        return Ok(json::object([("memory", Value::Null)]));
    }
    let instruction = thread::fold_instruction(first, last);
    let rendered = thread::messages(&admitted, Some(instruction));
    // Twice the bound is kept, so an answer longer than the bound stays longer and is refused rather than cut to fit.
    let memory = written(call, &rendered, 2 * MAX_MEMORY_BYTES, true)?
        .and_then(|memory| protocol::canonical_memory(&json::text(&memory)).map(json::text));
    Ok(json::object([("memory", memory.unwrap_or(Value::Null))]))
}

/// The agent's final answer from its thread once its time or budget is spent; Haiku writes none.
fn final_answer(call: &Call<'_>, body: &Value) -> Result<Value, RouteError> {
    let admitted = validate(body)?;
    if call.config.provider != ProviderId::OpenAi {
        return Ok(json::object([("summary", Value::Null)]));
    }
    let rendered = thread::messages(&admitted, Some(thread::FINAL.to_owned()));
    let summary = written(call, &rendered, MAX_SUMMARY_BYTES, false)?
        .and_then(|summary| done(&json::text(&summary)))
        .and_then(|step| step.get("summary").cloned());
    Ok(json::object([("summary", summary.unwrap_or(Value::Null))]))
}

/// The canonical done step carrying `summary`, if the protocol admits it.
fn done(summary: &Value) -> Option<Value> {
    protocol::canonical_step(&json::object([
        ("kind", json::text("done")),
        ("summary", summary.clone()),
    ]))
}

/// The advisory completeness verdict on a done answer; Haiku's answers go unchecked (null) without a call.
fn check(call: &Call<'_>, body: &Value) -> Result<Value, RouteError> {
    let objective = body
        .get("objective")
        .and_then(protocol::canonical_objective);
    let summary = body.get("summary").cloned().unwrap_or(Value::Null);
    let (Some(objective), Some(_)) = (objective, done(&summary)) else {
        return Err(refused("invalid Browser check request"));
    };
    if call.config.provider != ProviderId::OpenAi {
        return Ok(json::object([
            ("complete", Value::Null),
            ("missing", Value::Null),
        ]));
    }
    let rendered =
        thread::check_messages(objective, json::str_of(body, "summary").unwrap_or_default());
    bounded(&rendered, call.config.provider)?;
    let answer = Answer {
        name: "Completeness",
        schema: &CHECK_SCHEMA,
        max_chars: MAX_RESPONSE_CHARS,
    };
    let (request, projection) = rendered.request();
    let verdict = call.converse(request, projection.as_ref(), &answer)?;
    let missing = json::str_of(&verdict, "missing")
        .filter(|missing| !python_text::strip(missing).is_empty())
        .map_or(Value::Null, json::text);
    let judged = if json::bool_of(&verdict, "complete") == Some(true) {
        json::object([("complete", Value::Bool(true)), ("missing", Value::Null)])
    } else {
        json::object([("complete", Value::Bool(false)), ("missing", missing)])
    };
    protocol::canonical_check(&judged).ok_or_else(broken)
}

#[cfg(test)]
mod tests;

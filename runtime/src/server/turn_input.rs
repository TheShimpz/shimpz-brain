//! The start and resume requests of a chat turn (`runtime_api.StartTurnInput`, `ResumeTurnInput`, and the
//! `agent_runtime` definitions they build): one 422 for any shape violation, then the runtime's 400 contract checks.

use team_protocol::{identifiers, payload, routine, routine_context, turn};

use crate::attachments::{self, Attachment};
use crate::errors::BrainError;
use crate::json::Value;
use crate::provider::ProviderConfig;
use crate::server::input::{self, RouteError};
use crate::tools;
use crate::turn::context::{
    ActionDefinition, AssistantDefinition, AttachmentPins, Memory, TurnContext,
};

/// An Assistant's Genesis, bounded in code points by its producers.
pub const MAX_GENESIS_CHARS: usize = 65_536;
/// A turn's message.
pub const MAX_MESSAGE_CHARS: usize = 64 * 1024;
/// The Actions one Assistant exposes.
pub const MAX_ACTIONS_PER_ASSISTANT: usize = 128;
/// The JSON values one Assistant's Action schemas hold together.
pub const MAX_ASSISTANT_SCHEMA_NODES: usize = 32_768;
/// An Action summary.
pub const MAX_ACTION_SUMMARY_CHARS: usize = 2_000;
/// The IANA zone name of a turn clock.
pub const MAX_ZONE_CHARS: usize = 64;
/// The zone of a turn without Team's clock.
pub const DEFAULT_ZONE: &str = "UTC";

const CONTEXT_FIELDS: [&str; 15] = [
    "thread_id",
    "team_name",
    "assistants",
    "provider",
    "memories",
    "routines",
    "routine_question",
    "routine_mode",
    "routine_rerun",
    "routine_schedule",
    "routine_output",
    "routine_settings_answer",
    "routine_selected",
    "attachments",
    "browser_available",
];

/// One entry of the committed presentation history a start may quote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationEntry {
    /// `user` or `assistant`.
    pub role: String,
    /// Its text.
    pub text: String,
    /// Whether Team cut it short.
    pub truncated: bool,
}

/// A turn's context, provider, and attachments, as both requests carry them.
#[derive(Debug, Clone)]
pub struct TurnRequest {
    /// The admitted context (its turn message id is set by a start, or restored by a resume).
    pub context: TurnContext,
    /// The provider and key.
    pub provider: ProviderConfig,
    /// The prepared attachments (request-local content).
    pub attachments: Vec<Attachment>,
}

/// `POST /v1/turns`.
#[derive(Debug, Clone)]
pub struct StartRequest {
    /// The turn.
    pub turn: TurnRequest,
    /// The person's message (Team's `{"files", "message"}` envelope).
    pub message: String,
    /// The committed presentation history window.
    pub conversation: Vec<ConversationEntry>,
}

/// `POST /v1/turns/resume`.
#[derive(Debug, Clone)]
pub struct ResumeRequest {
    /// The turn.
    pub turn: TurnRequest,
    /// Each pending request's result, by interrupt id, in Team's order.
    pub results: Vec<(String, Value)>,
}

/// Reads a start request.
///
/// # Errors
///
/// [`RouteError`]: 422 for the shape, 400 for the contract checks.
pub fn start(body: &Value) -> Result<StartRequest, RouteError> {
    let mut allowed = CONTEXT_FIELDS.to_vec();
    allowed.extend(["message", "locale", "turn_clock", "conversation"]);
    let mut required = CONTEXT_FIELDS.to_vec();
    required.extend(["message", "locale", "conversation"]);
    input::closed(body, &allowed, &required)?;
    let message = input::string(body, "message", 1, MAX_MESSAGE_CHARS)?;
    let locale = locale(body)?;
    let clock = clock(body)?;
    let conversation = conversation_shape(body)?;
    let shape = context_shape(body)?;
    let mut turn = build(body, &shape)?;
    turn.context.locale = locale;
    if let Some((date, zone)) = clock {
        turn.context.turn_date = date;
        turn.context.timezone = zone;
    }
    if text_blank(&message) {
        return Err(BrainError::contract("invalid chat message").into());
    }
    let conversation = admit_conversation(conversation)?;
    Ok(StartRequest {
        turn,
        message,
        conversation,
    })
}

/// Reads a resume request.
///
/// # Errors
///
/// [`RouteError`]: 422 for the shape, 400 for the contract checks.
pub fn resume(body: &Value) -> Result<ResumeRequest, RouteError> {
    let mut allowed = CONTEXT_FIELDS.to_vec();
    allowed.push("results");
    let mut required = CONTEXT_FIELDS.to_vec();
    required.push("results");
    input::closed(body, &allowed, &required)?;
    let results = input::object(body, "results")?;
    let Value::Object(results) = results else {
        return Err(RouteError::Invalid);
    };
    if results.is_empty() || results.len() > turn::MAX_ACTION_REQUESTS {
        return Err(RouteError::Invalid);
    }
    let shape = context_shape(body)?;
    let turn = build(body, &shape)?;
    if results.iter().any(|(key, _)| key.is_empty()) {
        return Err(BrainError::contract("invalid Action resume results").into());
    }
    Ok(ResumeRequest {
        turn,
        results: results.clone(),
    })
}

fn text_blank(message: &str) -> bool {
    team_protocol::python::text::strip(message).is_empty()
}

fn locale(body: &Value) -> Result<Option<String>, RouteError> {
    match body.get("locale") {
        Some(Value::Null) => Ok(None),
        Some(Value::String(locale)) if payload::CHAT_LOCALES.contains(&locale.as_str()) => {
            Ok(Some(locale.clone()))
        }
        _ => Err(RouteError::Invalid),
    }
}

fn clock(body: &Value) -> Result<Option<(String, String)>, RouteError> {
    let Some(clock) = input::member(body, "turn_clock").filter(|value| **value != Value::Null)
    else {
        return Ok(None);
    };
    input::closed(clock, &["date", "timezone"], &["date", "timezone"])?;
    let date = input::string(clock, "date", 0, usize::MAX)?;
    let zone = input::string(clock, "timezone", 1, MAX_ZONE_CHARS)?;
    if !valid_date(&date) || !valid_zone(&zone) {
        return Err(RouteError::Invalid);
    }
    Ok(Some((date, zone)))
}

/// Whether `date` is `YYYY-MM-DD` naming a real date (`date.fromisoformat` behind the request pattern).
#[must_use]
pub fn valid_date(date: &str) -> bool {
    let bytes = date.as_bytes();
    let shaped = bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit());
    if !shaped {
        return false;
    }
    let number = |range: std::ops::Range<usize>| date[range].parse::<u32>().unwrap_or(0);
    let (year, month, day) = (number(0..4), number(5..7), number(8..10));
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    year >= 1 && (1..=days).contains(&day)
}

/// Whether `zone` is one IANA zone name the Team protocol's zone database knows, or `UTC`.
#[must_use]
pub fn valid_zone(zone: &str) -> bool {
    zone == DEFAULT_ZONE || routine::canonical_timezone(&Value::String(zone.to_owned())).is_some()
}

fn conversation_shape(body: &Value) -> Result<Vec<ConversationEntry>, RouteError> {
    let entries = input::list(body, "conversation", 0, payload::MAX_CONVERSATION_ENTRIES)?;
    entries
        .iter()
        .map(|entry| {
            input::closed(
                entry,
                &["role", "text", "truncated"],
                &["role", "text", "truncated"],
            )?;
            let role = input::string(entry, "role", 0, usize::MAX)?;
            if role != "user" && role != "assistant" {
                return Err(RouteError::Invalid);
            }
            let text = input::string(entry, "text", 1, payload::MAX_CONVERSATION_TEXT_CHARS)?;
            Ok(ConversationEntry {
                role,
                text,
                truncated: input::boolean(entry, "truncated")?,
            })
        })
        .collect()
}

fn admit_conversation(
    entries: Vec<ConversationEntry>,
) -> Result<Vec<ConversationEntry>, RouteError> {
    let invalid = || RouteError::from(BrainError::contract("invalid conversation window"));
    if entries.iter().any(|entry| {
        !crate::text::admitted(
            &entry.text,
            payload::MAX_CONVERSATION_TEXT_CHARS,
            false,
            true,
        )
    }) {
        return Err(invalid());
    }
    if entries
        .iter()
        .map(|entry| entry.text.chars().count())
        .sum::<usize>()
        > payload::MAX_CONVERSATION_CHARS
    {
        return Err(invalid());
    }
    Ok(entries)
}

/// The members of the shared context fields, checked for shape only.
struct Shape<'a> {
    assistants: &'a [Value],
    memories: Option<&'a [Value]>,
    routines: Option<&'a [Value]>,
    rerun: Option<&'a [Value]>,
    attachments: &'a [Value],
}

fn context_shape(body: &Value) -> Result<Shape<'_>, RouteError> {
    input::string(body, "thread_id", 1, 256)?;
    if payload::canonical_team_name(body.get("team_name").unwrap_or(&Value::Null)).is_none() {
        return Err(RouteError::Invalid);
    }
    let assistants = input::list(body, "assistants", 0, payload::MAX_CHAT_ASSISTANTS)?;
    assistants.iter().try_for_each(assistant_shape)?;
    input::provider_any(body)?;
    let memories = input::optional_list(body, "memories", turn::MAX_MEMORIES)?;
    let routines = input::optional_list(body, "routines", routine::MAX_ROUTINES)?;
    let rerun = input::optional_list(body, "routine_rerun", routine_context::MAX_RERUN_ENTRIES)?;
    let objects = [memories, routines, rerun].into_iter().flatten().flatten();
    if objects
        .into_iter()
        .any(|item| !matches!(item, Value::Object(_)))
    {
        return Err(RouteError::Invalid);
    }
    for field in ["routine_question", "routine_schedule", "routine_selected"] {
        input::optional_object(body, field)?;
    }
    input::optional_string(body, "routine_output", 0, usize::MAX)?;
    input::boolean(body, "routine_mode")?;
    input::boolean(body, "routine_settings_answer")?;
    input::boolean(body, "browser_available")?;
    let attachments = input::list(body, "attachments", 0, payload::MAX_CHAT_FILES)?;
    if attachments
        .iter()
        .any(|item| !matches!(item, Value::Object(_)))
    {
        return Err(RouteError::Invalid);
    }
    Ok(Shape {
        assistants,
        memories,
        routines,
        rerun,
        attachments,
    })
}

fn assistant_shape(value: &Value) -> Result<(), RouteError> {
    input::closed(
        value,
        &["id", "genesis", "actions"],
        &["id", "genesis", "actions"],
    )?;
    input::string(value, "id", 1, 128)?;
    input::string(value, "genesis", 1, MAX_GENESIS_CHARS)?;
    for action in input::list(value, "actions", 0, MAX_ACTIONS_PER_ASSISTANT)? {
        let fields = [
            "id",
            "summary",
            "input_schema",
            "output_schema",
            "authorization",
            "input_files",
        ];
        input::closed(action, &fields, &fields)?;
        input::string(action, "id", 1, 128)?;
        input::string(action, "summary", 1, MAX_ACTION_SUMMARY_CHARS)?;
        input::object(action, "input_schema")?;
        input::object(action, "output_schema")?;
        input::boolean(action, "authorization")?;
        let files = input::list(action, "input_files", 0, 1)?;
        if files.iter().any(|name| !matches!(name, Value::String(text) if (1..=128).contains(&text.chars().count()))) {
            return Err(RouteError::Invalid);
        }
    }
    Ok(())
}

fn contract(detail: &str) -> RouteError {
    BrainError::contract(detail).into()
}

fn build(body: &Value, shape: &Shape<'_>) -> Result<TurnRequest, RouteError> {
    let assistants = shape
        .assistants
        .iter()
        .map(assistant)
        .collect::<Result<Vec<_>, _>>()?;
    let provider = input::provider(body)?;
    let memories = shape.memories.map(memories).transpose()?;
    let attachments = attachments::admit(&Value::Array(shape.attachments.to_vec()))
        .map_err(|_| contract("invalid attachments"))?;
    let text = |key: &str| input::optional_string(body, key, 0, usize::MAX);
    let object = |key: &str| input::optional_object(body, key).map(Option::<&Value>::cloned);
    let mut context = TurnContext {
        thread_id: input::string(body, "thread_id", 1, 256)?,
        team_name: input::string(body, "team_name", 0, usize::MAX)?,
        assistants,
        provider: provider.provider,
        model: provider.model.clone(),
        turn_date: crate::prompt::today(),
        timezone: DEFAULT_ZONE.to_owned(),
        memories,
        routines: shape.routines.map(<[Value]>::to_vec),
        routine_question: object("routine_question")?,
        routine_mode: input::boolean(body, "routine_mode")?,
        routine_rerun: shape.rerun.map(<[Value]>::to_vec),
        routine_schedule: object("routine_schedule")?,
        routine_output: text("routine_output")?,
        routine_settings_answer: input::boolean(body, "routine_settings_answer")?,
        routine_selected: object("routine_selected")?,
        locale: None,
        turn_message_id: None,
        attachments: AttachmentPins {
            kinds: attachments::kinds(&attachments),
            commitment: attachments::commitment(&attachments),
            charge: 0,
        },
        browser: input::boolean(body, "browser_available")?,
    };
    tools::routine::admit_context(&mut context)?;
    admit_turn(&mut context)?;
    Ok(TurnRequest {
        context,
        provider,
        attachments,
    })
}

fn admit_turn(context: &mut TurnContext) -> Result<(), RouteError> {
    if !turn::SAFE_ID_RE.is_full(&context.thread_id) {
        return Err(contract("invalid conversation thread"));
    }
    if context.assistants.len() > payload::MAX_CHAT_ASSISTANTS {
        return Err(contract(&format!(
            "a turn may name at most {} Assistants",
            payload::MAX_CHAT_ASSISTANTS
        )));
    }
    let mut ids: Vec<&str> = context
        .assistants
        .iter()
        .map(|assistant| assistant.id.as_str())
        .collect();
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(contract("duplicate Assistant id"));
    }
    context
        .assistants
        .sort_by(|left, right| left.id.cmp(&right.id));
    Ok(())
}

fn memories(items: &[Value]) -> Result<Vec<Memory>, RouteError> {
    tools::memory::canonical(&Value::Array(items.to_vec()))
        .ok_or_else(|| contract("invalid memory"))
}

fn assistant(value: &Value) -> Result<AssistantDefinition, RouteError> {
    let id = input::string(value, "id", 1, 128)?;
    if identifiers::canonical_assistant_id(&Value::String(id.clone())).is_none() {
        return Err(contract("invalid Assistant id"));
    }
    let genesis = input::string(value, "genesis", 1, MAX_GENESIS_CHARS)?;
    let printable = crate::text::printable_with_layout(&genesis);
    if team_protocol::python::text::strip(&genesis) != genesis || !printable {
        return Err(contract("invalid Assistant Genesis"));
    }
    let mut actions = input::list(value, "actions", 0, MAX_ACTIONS_PER_ASSISTANT)?
        .iter()
        .map(action)
        .collect::<Result<Vec<_>, _>>()?;
    let mut ids: Vec<&str> = actions.iter().map(|action| action.id.as_str()).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(contract("duplicate Action id within Assistant"));
    }
    let mut remaining = MAX_ASSISTANT_SCHEMA_NODES;
    for schema in actions
        .iter()
        .flat_map(|action| [&action.input_schema, &action.output_schema])
    {
        let nodes = schema_nodes(schema, remaining);
        if nodes > remaining {
            return Err(contract("Assistant Action schemas are too large"));
        }
        remaining -= nodes;
    }
    actions.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(AssistantDefinition {
        id,
        genesis,
        actions,
    })
}

/// The JSON values of `value`, counting at most one past `limit` (`json_nodes`).
fn schema_nodes(value: &Value, limit: usize) -> usize {
    fn count(value: &Value, limit: usize, total: &mut usize) {
        *total += 1;
        if *total > limit {
            return;
        }
        match value {
            Value::Array(items) => items.iter().for_each(|item| count(item, limit, total)),
            Value::Object(members) => members
                .iter()
                .for_each(|(_, item)| count(item, limit, total)),
            _ => {}
        }
    }
    let mut total = 0;
    count(value, limit, &mut total);
    total.min(limit + 1)
}

fn action(value: &Value) -> Result<ActionDefinition, RouteError> {
    let id = input::string(value, "id", 1, 128)?;
    if identifiers::canonical_action_id(&Value::String(id.clone())).is_none() {
        return Err(contract("invalid Action id"));
    }
    let input_schema = input::object(value, "input_schema")?.clone();
    let output_schema = input::object(value, "output_schema")?.clone();
    for (schema, kind) in [(&input_schema, "input"), (&output_schema, "output")] {
        if let Err(refusal) = action_schema::admit_schema(schema) {
            return Err(contract(&format!("Action {kind} schema {refusal}")));
        }
    }
    let input_files: Vec<String> = input::list(value, "input_files", 0, 1)?
        .iter()
        .filter_map(|name| match name {
            Value::String(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    let declared = input_files.iter().all(|name| {
        input_schema
            .get("properties")
            .and_then(|properties| properties.get(name))
            .is_some()
    });
    if !declared {
        return Err(contract("invalid Action file or authorization declaration"));
    }
    let summary = input::string(value, "summary", 1, MAX_ACTION_SUMMARY_CHARS)?;
    if team_protocol::python::text::strip(&summary).is_empty() {
        return Err(contract("invalid Action summary"));
    }
    let authorization = input::boolean(value, "authorization")?;
    Ok(ActionDefinition {
        id,
        summary,
        input_schema,
        output_schema,
        authorization,
        input_files,
    })
}

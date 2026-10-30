//! The conversation a chat turn carries to its model (`context_budget.py`): the newest whole exchanges within the
//! history budget, older Action results omitted a batch at a time, and an estimated model-window guard.
//!
//! Memory is the most recent whole exchanges: one user message and every Action round and reply that followed it; a
//! quoted bridge of presentation history belongs to the exchange of the turn it preceded. A new turn keeps at most
//! [`MAX_HISTORY_EXCHANGES`] completed ones within [`HISTORY_BUDGET_TOKENS`], a Shimpz cost and latency policy far
//! below any catalog model window, and forgets every unfinished one. Within what is kept, the Action results of
//! completed exchanges older than the newest [`KEEP_RESULT_EXCHANGES`] give way to [`OMITTED_RESULT`] once that saves
//! [`OMIT_RESULTS_TRIGGER_TOKENS`] together: a batch at a time, so the provider's cached prompt prefix breaks rarely
//! instead of at every turn.
//!
//! Token counts are a pessimistic byte heuristic, not a tokenizer: a preflight estimate, never an exact bound.

use crate::conversation::{AssistantMessage, Message, ToolMessage};
use crate::errors::BrainError;
use crate::json::{self, Value};
use team_protocol::python::text;

/// The id prefix of the one user-role message quoting committed presentation history (ADR-0065).
pub const BRIDGE_ID_PREFIX: &str = "shimpz-context-";
/// The completed exchanges a new turn keeps at most.
pub const MAX_HISTORY_EXCHANGES: usize = 24;
/// The estimated tokens the kept exchanges may take together.
pub const HISTORY_BUDGET_TOKENS: u64 = 128_000;
/// The smallest context window in the model catalog: 1M tokens for Claude Opus 5.5 and Sonnet 5, 1.05M for GPT-6.1
/// Sol and Luna, per the providers' model tables on 2026-09-28.
pub const MODEL_WINDOW_TOKENS: u64 = 1_000_000;
/// Every catalog model allows 128K output tokens, reserved within the window.
pub const OUTPUT_RESERVE_TOKENS: u64 = 128_000;
/// About 2.5 Unicode characters per token on current tokenizers; three UTF-8 bytes per token stays pessimistic for
/// ASCII prose and JSON while counting a multi-byte character as roughly one token.
pub const BYTES_PER_TOKEN: u64 = 3;
/// The newest completed exchanges whose Action results always stay whole. Measured on the 21 longest precision
/// episode templates with GPT-6 Luna (2026-10-21): about 23% fewer input tokens and 15% fewer dollars; omitting at
/// every turn instead spent more, because each edit breaks the provider's cached prefix.
pub const KEEP_RESULT_EXCHANGES: usize = 2;
/// The estimated tokens omitted results must save together before any is omitted.
pub const OMIT_RESULTS_TRIGGER_TOKENS: u64 = 8_000;
/// What an omitted Action result reads. It claims nothing about the call: an omitted message may have been a result
/// or a refusal to run it.
pub const OMITTED_RESULT: &str = "[Earlier Action result omitted to save context. This note does not say whether \
the call ran or what it changed; the reply that ended that exchange does. Never repeat a change to recover this \
result. If the current request needs its data, read it again.]";

/// Why history cannot be fitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// The fixed prompt, tools, and current exchange alone exceed the window: a contract error.
    Window,
    /// Stored history cannot be decomposed into whole exchanges: a state error.
    State,
}

impl From<FitError> for BrainError {
    fn from(error: FitError) -> Self {
        match error {
            FitError::Window => Self::contract("conversation context exceeds the model window"),
            FitError::State => Self::State,
        }
    }
}

/// The policy bounds, one value so the tests can narrow them.
#[derive(Debug, Clone, Copy)]
struct Limits {
    exchanges: usize,
    history: u64,
    window: u64,
    omit_trigger: u64,
}

const LIMITS: Limits = Limits {
    exchanges: MAX_HISTORY_EXCHANGES,
    history: HISTORY_BUDGET_TOKENS,
    window: MODEL_WINDOW_TOKENS,
    omit_trigger: OMIT_RESULTS_TRIGGER_TOKENS,
};

/// The estimated tokens of a JSON value (`estimated_tokens`): its compact UTF-8 encoding
/// (`json.dumps(..., ensure_ascii=False, separators=(",", ":"))`) at [`BYTES_PER_TOKEN`] bytes each, rounded up.
#[must_use]
pub fn estimated_tokens(value: &Value) -> u64 {
    let bytes = u64::try_from(json::compact(value).len()).unwrap_or(u64::MAX);
    bytes.div_ceil(BYTES_PER_TOKEN)
}

/// The estimated tokens of a message (`message_tokens`).
///
/// The Python Brain measured `LangChain`'s dump of the message's content and tool calls. This estimates the
/// equivalent JSON object instead: `{"content"}` for a user message or a tool result, and for a model turn
/// `{"content", "tool_calls", "invalid_tool_calls"}`, where the content is the provider's own replayed items when
/// the turn keeps them (as `LangChain`'s content blocks did) and its text otherwise. A pessimistic heuristic, not a
/// contract.
#[must_use]
pub fn message_tokens(message: &Message) -> u64 {
    estimated_tokens(&message_value(message))
}

fn message_value(message: &Message) -> Value {
    match message {
        Message::User { text, .. } => json::object([("content", json::text(text))]),
        Message::Tool(tool) => json::object([("content", json::text(&tool.content))]),
        Message::Assistant(turn) => assistant_value(turn),
    }
}

fn assistant_value(turn: &AssistantMessage) -> Value {
    let calls = turn
        .calls
        .iter()
        .map(|call| {
            json::object([
                ("name", json::text(&call.name)),
                ("args", call.arguments.clone()),
                ("id", json::text(&call.id)),
                ("type", json::text("tool_call")),
            ])
        })
        .collect();
    let invalid = turn
        .invalid_calls
        .iter()
        .map(|call| {
            json::object([
                ("name", json::text(&call.name)),
                ("args", json::text(&call.arguments)),
                ("id", json::text(&call.id)),
                ("error", Value::Null),
                ("type", json::text("invalid_tool_call")),
            ])
        })
        .collect();
    let content = turn.raw.clone().unwrap_or_else(|| json::text(&turn.text));
    json::object([
        ("content", content),
        ("tool_calls", json::array(calls)),
        ("invalid_tool_calls", json::array(invalid)),
    ])
}

/// The fixed tokens of a system prompt and its tools, each `{"name", "summary", "schema"}` (`fixed_tokens`).
#[must_use]
pub fn fixed_tokens(system: &str, tools: &[Value]) -> u64 {
    tools
        .iter()
        .fold(estimated_tokens(&json::text(system)), |sum, tool| {
            sum.saturating_add(estimated_tokens(tool))
        })
}

/// History split at user messages (`exchanges`). A bridge and the turn message right after it are one exchange, so
/// the turn keeps the context it was read in for as long as it is remembered, and both are forgotten together.
///
/// # Errors
///
/// [`FitError::State`] when a message has no id or the history does not start with a user message.
pub fn exchanges(messages: &[Message]) -> Result<Vec<&[Message]>, FitError> {
    let mut starts: Vec<usize> = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.id().is_empty() {
            return Err(FitError::State);
        }
        let joins = starts
            .last()
            .is_some_and(|start| index - start == 1 && bridge(&messages[*start]));
        if matches!(message, Message::User { .. }) && !joins {
            starts.push(index);
        } else if starts.is_empty() {
            return Err(FitError::State);
        }
    }
    let ends = starts.iter().skip(1).copied().chain([messages.len()]);
    Ok(starts
        .iter()
        .zip(ends)
        .map(|(start, end)| &messages[*start..end])
        .collect())
}

fn bridge(message: &Message) -> bool {
    matches!(message, Message::User { id, .. } if id.starts_with(BRIDGE_ID_PREFIX))
}

/// Whether the provider reports that it cut this message short instead of finishing it (`truncated`).
#[must_use]
pub fn truncated(message: &Message) -> bool {
    message.assistant().is_some_and(|turn| turn.truncated)
}

/// The reply a model turn ends a turn with, or "" when it cannot end one (`final_reply`): a turn that calls tools,
/// carries an unparsable call, or was cut short never ends a turn.
#[must_use]
pub fn final_reply(message: &Message) -> String {
    match message.assistant() {
        Some(turn) if turn.calls.is_empty() && turn.invalid_calls.is_empty() && !turn.truncated => {
            text::strip(&turn.text).to_owned()
        }
        _ => String::new(),
    }
}

/// An exchange is complete only when it ends in an accepted reply; a failed turn leaves it unfinished (`completed`).
#[must_use]
pub fn completed(exchange: &[Message]) -> bool {
    exchange
        .last()
        .is_some_and(|message| !final_reply(message).is_empty())
}

/// Refuses a call whose fixed and conversation tokens exceed the window (`ensure_window`).
///
/// # Errors
///
/// [`FitError::Window`].
pub fn ensure_window(fixed: u64, conversation: u64) -> Result<(), FitError> {
    window_within(LIMITS, fixed, conversation)
}

fn window_within(limits: Limits, fixed: u64, conversation: u64) -> Result<(), FitError> {
    let needed = fixed
        .saturating_add(conversation)
        .saturating_add(OUTPUT_RESERVE_TOKENS);
    if needed > limits.window {
        return Err(FitError::Window);
    }
    Ok(())
}

/// The ids of what a new turn must forget (`history_to_drop`): every unfinished exchange, and completed ones beyond
/// the newest run kept. An unfinished exchange is a failed turn (a lone user message or an unanswered Action round);
/// resending it would replay a request the user already saw fail. An exchange holding a `forgotten` message, such as
/// a user message that read attachments, is forgotten whole however recent it is (ADR-0093).
///
/// # Errors
///
/// [`FitError`].
pub fn history_to_drop(
    history: &[Message],
    fixed: u64,
    current: u64,
    forgotten: &dyn Fn(&Message) -> bool,
) -> Result<Vec<String>, FitError> {
    drop_within(LIMITS, history, fixed, current, forgotten)
}

fn drop_within(
    limits: Limits,
    history: &[Message],
    fixed: u64,
    current: u64,
    forgotten: &dyn Fn(&Message) -> bool,
) -> Result<Vec<String>, FitError> {
    window_within(limits, fixed, current)?;
    let groups = exchanges(history)?;
    let mut kept = vec![false; groups.len()];
    let (mut count, mut used) = (0, 0_u64);
    for (index, group) in groups.iter().enumerate().rev() {
        if !completed(group) || group.iter().any(forgotten) {
            continue;
        }
        let size = group.iter().map(message_tokens).sum::<u64>();
        let total = used.saturating_add(size);
        if count == limits.exchanges
            || total > limits.history
            || window_within(limits, fixed.saturating_add(total), current).is_err()
        {
            break;
        }
        kept[index] = true;
        count += 1;
        used = total;
    }
    Ok(groups
        .iter()
        .zip(kept)
        .filter(|(_, kept)| !kept)
        .flat_map(|(group, _)| group.iter().map(|message| message.id().to_owned()))
        .collect())
}

/// Placeholders for the older Action results a new turn omits, or none below the trigger (`results_to_omit`).
///
/// Only completed exchanges older than the newest [`KEEP_RESULT_EXCHANGES`] completed ones give results; an
/// unfinished or `forgotten` exchange is dropped whole instead. A result no larger than its placeholder stays, and
/// the trigger counts the tokens the placeholders save. Each placeholder keeps its message's id, call pairing, name,
/// and error flag, so it replaces that result in place.
///
/// # Errors
///
/// [`FitError::State`].
pub fn results_to_omit(
    history: &[Message],
    action_result: &dyn Fn(&ToolMessage) -> bool,
    forgotten: &dyn Fn(&Message) -> bool,
) -> Result<Vec<ToolMessage>, FitError> {
    omit_within(LIMITS, history, action_result, forgotten)
}

fn omit_within(
    limits: Limits,
    history: &[Message],
    action_result: &dyn Fn(&ToolMessage) -> bool,
    forgotten: &dyn Fn(&Message) -> bool,
) -> Result<Vec<ToolMessage>, FitError> {
    let groups = exchanges(history)?;
    let done: Vec<&[Message]> = groups
        .into_iter()
        .filter(|group| completed(group) && !group.iter().any(forgotten))
        .collect();
    let older = &done[..done.len().saturating_sub(KEEP_RESULT_EXCHANGES)];
    let mut placeholders = Vec::new();
    let mut saved = 0_u64;
    for message in older.iter().flat_map(|group| group.iter()) {
        let Some(result) = message.tool().filter(|result| action_result(result)) else {
            continue;
        };
        let placeholder = ToolMessage {
            content: OMITTED_RESULT.to_owned(),
            ..result.clone()
        };
        // An omitted result saves nothing, so it is never omitted again.
        let gain = message_tokens(message)
            .saturating_sub(message_tokens(&Message::Tool(placeholder.clone())));
        if gain > 0 {
            placeholders.push(placeholder);
            saved = saved.saturating_add(gain);
        }
    }
    Ok(if saved >= limits.omit_trigger {
        placeholders
    } else {
        Vec::new()
    })
}

#[cfg(test)]
mod tests;

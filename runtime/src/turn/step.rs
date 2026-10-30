//! The steps of a chat turn's definition: one model call, the tools node, the settled reply, and the memory check.

use std::collections::BTreeSet;

use holdpoint::{Ctx, Interrupt};

use crate::attachments;
use crate::budget;
use crate::conversation::{AssistantMessage, Message, ToolCall, ToolMessage};
use crate::errors::BrainError;
use crate::json::{self, Value};
use crate::provider::ProviderId;
use crate::tools::{self, clarification, memory, routine};
use crate::turn::context::{ActionDefinition, TurnContext};
use crate::turn::definition::{Failed, Turn};
use crate::turn::run::{self, TeamRequest, TurnInput};

/// The longest reply a turn returns (`turn.MAX_REPLY_CHARS`).
pub const MAX_REPLY_CHARS: usize = team_protocol::turn::MAX_REPLY_CHARS;

/// Why the definition stops early.
#[derive(Debug)]
pub enum Stop {
    /// A Holdpoint interrupt to propagate (a suspension, a cancellation, a lost session).
    Interrupt(Interrupt),
    /// The turn failed; the run ends with this failure as its result.
    Failed(Failed),
}

impl From<Interrupt> for Stop {
    fn from(interrupt: Interrupt) -> Self {
        Self::Interrupt(interrupt)
    }
}

/// The turn failure a Brain error is.
#[must_use]
pub fn failed_of(error: &BrainError) -> Failed {
    match error {
        BrainError::Contract(detail) => Failed::Contract(detail.clone()),
        BrainError::Cancelled(_) => Failed::Cancelled,
        _ => Failed::Provider,
    }
}

fn failure_of(failure: &holdpoint::Failure) -> Failed {
    match failure.code.as_str() {
        "cancelled" => Failed::Cancelled,
        "contract" => Failed::Contract(failure.message.clone()),
        _ => Failed::Provider,
    }
}

/// The names of the Action tools the turn offers (they alone carry `strict: false`).
#[must_use]
pub fn action_names(input: &TurnInput) -> BTreeSet<String> {
    input
        .tools
        .iter()
        .filter(|tool| tool.strict == Some(false))
        .map(|tool| tool.name.clone())
        .collect()
}

fn action_of<'a>(
    context: &'a TurnContext,
    offered: &BTreeSet<String>,
    name: &str,
) -> Option<(&'a str, &'a ActionDefinition)> {
    context.assistants.iter().find_map(|assistant| {
        assistant.actions.iter().find_map(|action| {
            let tool = tools::action::tool_name(&assistant.id, &action.id);
            (tool == name && offered.contains(&tool)).then_some((assistant.id.as_str(), action))
        })
    })
}

/// The model call request a turn journals (its digest) and the provider model answers.
fn chat_request(turn: &Turn) -> Value {
    let input = &turn.input;
    let context = &input.context;
    let tools: Vec<Value> = if turn.after_browser {
        vec![]
    } else {
        input.tools.iter().map(run::tool_value).collect()
    };
    let message_id = context.turn_message_id.clone().unwrap_or_default();
    let reserved = if context.attachments.any() {
        attachments::dispatched(&turn.messages, &message_id)
    } else {
        0
    };
    json::object([
        ("kind", json::text("chat")),
        ("provider", json::text(context.provider.as_str())),
        ("model", json::text(&context.model)),
        ("system", json::text(&input.system)),
        ("messages", crate::conversation::to_value(&turn.messages)),
        ("tools", json::array(tools)),
        (
            "cache",
            Value::Bool(context.provider == ProviderId::Anthropic),
        ),
        ("charge", json::uint(context.attachments.charge)),
        ("reserved", json::uint(u64::from(reserved))),
        ("turn_message_id", json::text(&message_id)),
    ])
}

/// One journaled model call of the turn; a truncated answer fails the turn.
///
/// # Errors
///
/// [`Stop`].
pub fn call_model(turn: &mut Turn, ctx: &mut Ctx<'_>) -> Result<AssistantMessage, Stop> {
    turn.take_steps(1)?;
    let request = chat_request(turn);
    let answered = ctx.model(run::MODEL, &[], |_| Ok(run::call_json(&request)))?;
    let output = answered.map_err(|failure| Stop::Failed(failure_of(&failure)))?;
    let Some(Some(value)) = run::answer_from(&output.value, "message") else {
        return Err(Stop::Failed(Failed::Provider));
    };
    let Some(Message::Assistant(mut message)) = Message::from_value(&value) else {
        return Err(Stop::Failed(Failed::Provider));
    };
    if message.truncated {
        return Err(Stop::Failed(Failed::Provider));
    }
    message.id = turn.next_id("a");
    Ok(message)
}

/// How one call of the tools node is answered.
enum Answer {
    /// At once, with this content (and the error status for an unknown tool).
    Now(String, bool),
    /// By Team's result to this request.
    Team(TeamRequest),
}

fn invalid_tool(name: &str, input: &TurnInput) -> String {
    let available: Vec<&str> = input.tools.iter().map(|tool| tool.name.as_str()).collect();
    format!(
        "Error: {name} is not a valid tool, try one of [{}].",
        available.join(", ")
    )
}

fn answer(turn: &mut Turn, call: &ToolCall, offered: &BTreeSet<String>) -> Result<Answer, Stop> {
    let input = &turn.input;
    let name = call.name.as_str();
    let known = input.tools.iter().any(|tool| tool.name == name) && !turn.after_browser;
    if !known {
        return Ok(Answer::Now(invalid_tool(name, input), true));
    }
    if let Some((assistant_id, action)) = action_of(&input.context, offered, name) {
        if !tools::action::valid(action, &call.arguments) {
            return Ok(Answer::Now(tools::action::correction(name, action), false));
        }
        let (assistant_id, action_id) = (assistant_id.to_owned(), action.id.clone());
        let interrupt_id = turn.interrupt_id_next();
        return Ok(Answer::Team(TeamRequest::Action {
            interrupt_id,
            assistant_id,
            action: action_id,
            input: call.arguments.clone(),
        }));
    }
    match name {
        clarification::TOOL_NAME => Ok(Answer::Now(clarification::RECORDED.to_owned(), false)),
        memory::TOOL_NAME => Ok(Answer::Now(memory::proposed_text().to_owned(), false)),
        tools::browser::TOOL_NAME => match tools::browser::objective(&call.arguments) {
            Ok(objective) => Ok(Answer::Team(TeamRequest::Browser {
                interrupt_id: turn.interrupt_id_next(),
                objective,
            })),
            Err(correction) => Ok(Answer::Now(correction.to_owned(), false)),
        },
        _ => Err(Stop::Failed(Failed::Contract(
            "the Routine tool runs only through its guard".into(),
        ))),
    }
}

/// The content a Team result gives its call's tool message: text as is, anything else as LangChain stringified it
/// (an Action result, `json.dumps` defaults) or as the browser tool returned it (compact).
fn result_content(request: &TeamRequest, result: &Value) -> String {
    match (request, result) {
        (TeamRequest::Action { .. }, Value::String(text)) => text.clone(),
        (TeamRequest::Action { .. }, other) => json::dumps(other, json::SPACED),
        (TeamRequest::Browser { .. }, other) => json::compact(other),
    }
}

fn team_results(ctx: &mut Ctx<'_>, requests: &[TeamRequest]) -> Result<Vec<Value>, Stop> {
    if requests.is_empty() {
        return Ok(vec![]);
    }
    ctx.deferred(&Turn::requests_json(requests))?
        .iter()
        .map(|value| run::result_from(value).ok_or(Stop::Failed(Failed::Provider)))
        .collect()
}

fn push_answers(
    turn: &mut Turn,
    latest: &AssistantMessage,
    answers: Vec<Answer>,
    results: Vec<Value>,
) {
    let mut results = results.into_iter();
    for (call, answer) in latest.calls.iter().zip(answers) {
        let (content, error) = match answer {
            Answer::Now(content, error) => (content, error),
            Answer::Team(request) => (
                result_content(&request, &results.next().unwrap_or(Value::Null)),
                false,
            ),
        };
        let id = turn.next_id("t");
        turn.messages.push(Message::Tool(ToolMessage {
            id,
            call_id: call.id.clone(),
            name: call.name.clone(),
            content,
            error,
        }));
    }
}

/// The tools node: answers every call of `latest` in order, asking Team for its Action and Browser calls in one
/// durable batch; `true` when a call ends the turn (the clarification tool returns directly).
///
/// # Errors
///
/// [`Stop`]: the suspension until Team answers, or a failure.
pub fn run_tools(
    turn: &mut Turn,
    ctx: &mut Ctx<'_>,
    latest: &AssistantMessage,
) -> Result<bool, Stop> {
    let offered = action_names(&turn.input);
    let answers = latest
        .calls
        .iter()
        .map(|call| answer(turn, call, &offered))
        .collect::<Result<Vec<_>, _>>()?;
    let requests: Vec<TeamRequest> = answers
        .iter()
        .filter_map(|answer| {
            if let Answer::Team(request) = answer {
                Some(request.clone())
            } else {
                None
            }
        })
        .collect();
    let results = team_results(ctx, &requests)?;
    push_answers(turn, latest, answers, results);
    if !requests.is_empty() {
        turn.resumed(
            requests
                .iter()
                .any(|request| matches!(request, TeamRequest::Browser { .. })),
        );
    }
    Ok(latest
        .calls
        .iter()
        .any(|call| call.name == clarification::TOOL_NAME))
}

/// A completed turn's reply and what accompanies it.
#[derive(Debug, Clone, PartialEq)]
pub struct Settled {
    /// The reply shown.
    pub reply: String,
    /// The clarification asked, if the turn ended on one.
    pub clarification: Option<clarification::Clarification>,
    /// The Routine recorded, if the turn ended on one.
    pub routine: Option<Value>,
}

fn remember(turn: &mut Turn, text: String) {
    let id = turn.next_id("a");
    turn.messages.push(Message::Assistant(AssistantMessage {
        id,
        text,
        calls: vec![],
        invalid_calls: vec![],
        refusal: false,
        truncated: false,
        provider: turn.input.context.provider.as_str().to_owned(),
        raw: None,
        attempts: 1,
    }));
}

fn settle_recorded(turn: &mut Turn) -> Option<Settled> {
    if let Some(mut asked) = clarification::recorded(&turn.messages) {
        if turn.input.context.routine_mode {
            asked.default_index = None;
        }
        let reply = asked.render();
        remember(turn, reply.clone());
        return Some(Settled {
            reply,
            clarification: Some(asked),
            routine: None,
        });
    }
    let (reply, recorded) = routine::recorded(&turn.messages)?;
    let remembered = if reply.is_empty() {
        json::str_of(&recorded, "description")
            .unwrap_or_default()
            .to_owned()
    } else {
        reply.clone()
    };
    remember(turn, remembered);
    Some(Settled {
        reply,
        clarification: None,
        routine: Some(recorded),
    })
}

/// Settles a finished turn (`_finish_clarification`, `_finish_routine`, `_settle`), remembering exactly the reply
/// the person is shown.
///
/// # Errors
///
/// [`Stop::Failed`] when the turn ended without an acceptable reply.
pub fn settle(turn: &mut Turn) -> Result<Settled, Stop> {
    if let Some(settled) = settle_recorded(turn) {
        return Ok(settled);
    }
    let missing = || {
        Stop::Failed(Failed::Contract(
            "graph completed without an Assistant reply".into(),
        ))
    };
    let start = turn.input.turn_start.min(turn.messages.len());
    let index = (start..turn.messages.len())
        .rev()
        .find(|index| turn.messages[*index].assistant().is_some())
        .ok_or_else(missing)?;
    let shown = budget::final_reply(&turn.messages[index]);
    if shown.is_empty() {
        return Err(missing());
    }
    let reply: String = shown.chars().take(MAX_REPLY_CHARS).collect();
    if reply != shown
        && let Message::Assistant(message) = &mut turn.messages[index]
    {
        message.text.clone_from(&reply);
        message.raw = None;
    }
    Ok(Settled {
        reply,
        clarification: None,
        routine: None,
    })
}

/// The independent memory check of a completed turn's proposals, as one journaled model call.
///
/// # Errors
///
/// [`Stop`] when the check was cancelled or the run must stop.
pub fn check_memory(
    turn: &mut Turn,
    ctx: &mut Ctx<'_>,
    _settled: &Settled,
) -> Result<Vec<memory::Change>, Stop> {
    let context = &turn.input.context;
    let Some((prompt, changes)) = memory::confirmation(&turn.messages, context.memories.as_deref())
    else {
        return Ok(vec![]);
    };
    let request = json::object([
        ("kind", json::text("check")),
        ("provider", json::text(context.provider.as_str())),
        ("model", json::text(&context.model)),
        ("prompt", json::text(&prompt)),
    ]);
    let answered = ctx.model(run::MODEL, &[], |_| Ok(run::call_json(&request)))?;
    let output = answered.map_err(|failure| Stop::Failed(failure_of(&failure)))?;
    let verdict = run::answer_from(&output.value, "verdict").flatten();
    Ok(memory::accepted(changes, verdict.as_ref()))
}

/// The completed turn's response fields.
#[must_use]
pub fn turn_value(settled: &Settled, memory: &[memory::Change]) -> Value {
    json::object([
        ("status", json::text("completed")),
        ("reply", json::text(&settled.reply)),
        (
            "clarification",
            settled
                .clarification
                .as_ref()
                .map_or(Value::Null, clarification::Clarification::to_value),
        ),
        (
            "memory",
            json::array(memory.iter().map(memory::Change::to_value).collect()),
        ),
        ("routine", settled.routine.clone().unwrap_or(Value::Null)),
        ("browser", Value::Null),
        ("actions", json::array(vec![])),
    ])
}

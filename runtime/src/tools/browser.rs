//! The chat agent's one `browser` tool (`browser_tool.py`, ADR-0117 §5 and §6).
//!
//! A valid call suspends the turn with one self-contained objective; Team runs the task and resumes with its result.
//! A call to it must be the only call of its model response. After its result, every later model call of the turn is
//! offered no tool and any tool call it still makes is refused, so the pass can only write the reply. An exchange
//! that holds a `browser` call or result is forgotten before the next ordinary turn.

use std::sync::LazyLock;

use team_protocol::browser::{MAX_OBJECTIVE_CHARS, canonical_objective};

use crate::conversation::Message;
use crate::json::{self, Value};
use crate::provider::ToolSpec;
use crate::tools::Review;
use crate::tools::refusal::{self, LoneCall};
use crate::turn::context::TurnContext;

/// The tool's name.
pub const TOOL_NAME: &str = "browser";

/// The tool's description.
pub const DESCRIPTION: &str = "Operate this Team's isolated, signed-out web browser on public websites: search, \
    read and compare pages, and download public files. Use it when the person asks you to use the browser itself, \
    even when an Action could do the job; otherwise only when the request needs current or site-specific web \
    content, or the person asks to search the web or cite sources, and no available Action covers it. Never use it \
    otherwise for general knowledge, writing, math, or what the conversation already contains. It cannot sign in, \
    buy, or send messages or forms. Write one self-contained objective in the person's language: the site or search, \
    every value to use, and what to report back. Call it alone, at most once per message; research can take several \
    minutes. Its result may list unscreened_urls (cited pages no page of the task showed) and a completeness other \
    than complete: tell the person plainly which sources were not confirmed and what is missing.";

/// The correction of a call without a valid objective.
pub const CORRECTION: &str = "The browser was not called: give one self-contained objective of at most 2,000 \
    characters naming the site or search, every value to use, and what to report back.";
/// The correction of a browser call beside another call.
pub const ALONE: &str = "The browser must be the only tool call of a response. Call it again alone, or answer without it.";
/// The correction of any call after the browser result.
pub const AFTER: &str = "The browser task is done and its result is above. Do not call any tool; write the reply \
    to the person now.";

/// The schema the providers received (`_BrowserInput` as `LangChain` converted it).
static SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    let objective = json::object([
        ("maxLength", json::uint(MAX_OBJECTIVE_CHARS as u64)),
        ("type", json::text("string")),
    ]);
    json::object([
        ("properties", json::object([("objective", objective)])),
        ("required", json::array(vec![json::text("objective")])),
        ("type", json::text("object")),
    ])
});

/// The schema a turn's fixed token count measures for the tool (`SUMMARY`), which is the tool's own declaration,
/// not the converted schema the providers receive.
#[must_use]
pub fn summary_schema() -> Value {
    let objective = json::object([
        ("type", json::text("string")),
        ("maxLength", json::uint(MAX_OBJECTIVE_CHARS as u64)),
    ]);
    json::object([
        ("type", json::text("object")),
        ("additionalProperties", Value::Bool(false)),
        ("required", json::array(vec![json::text("objective")])),
        ("properties", json::object([("objective", objective)])),
    ])
}

/// Whether a turn offers the Browser: where Team offered it, outside Routine work and attachment content.
#[must_use]
pub fn offered(context: &TurnContext) -> bool {
    context.browser
        && !context.attachments.any()
        && !context.routine_mode
        && !context.routine_settings_answer
}

/// The tool.
#[must_use]
pub fn spec() -> ToolSpec {
    ToolSpec {
        name: TOOL_NAME.to_owned(),
        description: DESCRIPTION.to_owned(),
        schema: SCHEMA.clone(),
        strict: None,
    }
}

/// The canonical objective of a call, or `Err(correction)`.
///
/// # Errors
///
/// The closed correction.
pub fn objective(arguments: &Value) -> Result<String, &'static str> {
    if !json::has_exactly(arguments, &["objective"]) {
        return Err(CORRECTION);
    }
    arguments
        .get("objective")
        .and_then(canonical_objective)
        .map(str::to_owned)
        .ok_or(CORRECTION)
}

/// The guard refusing a browser call beside any other call.
///
/// # Errors
///
/// An unanswerable call.
pub fn review_alone(messages: &[Message]) -> Review {
    Ok(match refusal::lone_call(messages, TOOL_NAME)? {
        LoneCall::Mixed => Some(ALONE.to_owned()),
        LoneCall::Absent | LoneCall::Alone(_) => None,
    })
}

/// The guard of the pass after a browser result: any call is refused.
///
/// # Errors
///
/// Never.
pub fn review_after(messages: &[Message]) -> Review {
    let called = messages
        .last()
        .and_then(Message::assistant)
        .is_some_and(|latest| !latest.calls.is_empty());
    Ok(called.then(|| AFTER.to_owned()))
}

/// Whether a message is a browser call or its result (its exchange is forgotten by the next turn).
#[must_use]
pub fn read(message: &Message) -> bool {
    match message {
        Message::Tool(result) => result.name == TOOL_NAME,
        Message::Assistant(turn) => turn.calls.iter().any(|call| call.name == TOOL_NAME),
        Message::User { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::BrainError;
    use crate::tools::testing::{context, reply, result, turn, turn_with, user, value};

    #[test]
    fn the_browser_is_never_offered_with_attachments_or_routine_work() {
        let mut offered_context = context();
        offered_context.browser = true;
        assert!(offered(&offered_context));
        assert!(!offered(&context()));
        let mut attached = offered_context.clone();
        attached.attachments.kinds = vec!["opaque".into()];
        let mut routine = offered_context.clone();
        routine.routine_mode = true;
        let mut settings = offered_context;
        settings.routine_settings_answer = true;
        for refused in [attached, routine, settings] {
            assert!(!offered(&refused));
        }
    }

    #[test]
    fn only_a_bounded_objective_is_admitted_and_it_is_canonical() {
        let objective = "Veja o prazo do edital no site da prefeitura.";
        assert_eq!(
            super::objective(&json::object([("objective", json::text(objective))])),
            Ok(objective.to_owned())
        );
        assert_eq!(
            super::objective(&value(r#"{"objective":"  padded  "}"#)),
            Ok("padded".to_owned())
        );
        for refused in [
            r#"{"objective":"   "}"#.to_owned(),
            r#"{"objective":5}"#.to_owned(),
            r#"{"objective":"x","extra":1}"#.to_owned(),
            "{}".to_owned(),
            format!(r#"{{"objective":"{}"}}"#, "x".repeat(2_001)),
        ] {
            assert_eq!(super::objective(&value(&refused)), Err(CORRECTION));
        }
    }

    #[test]
    fn a_browser_call_must_be_alone_and_nothing_is_called_after_its_result() {
        let call = || (TOOL_NAME, value(r#"{"objective":"x"}"#));
        assert_eq!(review_alone(&[turn(vec![call()])]), Ok(None));
        let mixed = [turn(vec![call(), ("shimpz_clarify", value("{}"))])];
        assert_eq!(review_alone(&mixed), Ok(Some(ALONE.into())));
        assert_eq!(
            review_alone(&[turn_with(vec![call()], vec!["other"])]),
            Err(BrainError::contract(refusal::UNPARSABLE))
        );
        assert_eq!(review_after(&[turn(vec![call()])]), Ok(Some(AFTER.into())));
        assert_eq!(review_after(&[reply("O prazo é 30/11.")]), Ok(None));
        assert_eq!(review_after(&[user("x")]), Ok(None));
    }

    #[test]
    fn the_budget_measures_the_tools_own_declaration() {
        assert_eq!(
            json::compact(&summary_schema()),
            r#"{"type":"object","additionalProperties":false,"required":["objective"],"properties":{"objective":{"type":"string","maxLength":2000}}}"#
        );
        assert_eq!(
            json::compact(&spec().schema),
            r#"{"properties":{"objective":{"maxLength":2000,"type":"string"}},"required":["objective"],"type":"object"}"#
        );
    }

    #[test]
    fn a_browser_exchange_is_recognized_by_its_call_and_its_result() {
        assert!(read(&result(TOOL_NAME, "c", "{}")));
        assert!(read(&turn(vec![(TOOL_NAME, value("{}"))])));
        assert!(!read(&reply("hi")));
        assert!(!read(&result("hello", "c", "{}")));
        assert!(!read(&user("hi")));
    }
}

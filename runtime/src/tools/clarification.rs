//! Brain-authored multiple-choice clarification (`clarification.py`, ADR-0081).
//!
//! When a material decision is open, the model may call the closed `shimpz_clarify` tool with one question, two to
//! five options, and a recommended default. The question is presentation only: it ends the turn, requests nothing, and
//! the user answers with a new message of their own, so Action authority stays with the user's current message. A
//! clarification combined with any other call, asked after an Action already ran in the logical turn, or malformed is
//! refused with a closed correction and nothing executes.

use std::collections::HashSet;
use std::sync::LazyLock;

use team_protocol::python::text;
use team_protocol::turn::{
    MAX_CLARIFICATION_DESCRIPTION_CHARS, MAX_CLARIFICATION_LABEL_CHARS, MAX_CLARIFICATION_OPTIONS,
    MAX_CLARIFICATION_QUESTION_CHARS, MIN_CLARIFICATION_OPTIONS,
};
use team_protocol::{payload, phrase};

use crate::conversation::Message;
use crate::json::{self, Value};
use crate::provider::ToolSpec;
use crate::tools::Review;
use crate::tools::refusal::{self, LoneCall};

/// The tool's name.
pub const TOOL_NAME: &str = "shimpz_clarify";

/// The tool's description.
pub const DESCRIPTION: &str = "Ask the user one multiple-choice question when a material decision is open. Give two \
    to five distinct options and the index of the one you recommend. Do not add an 'other' option: the user can \
    always answer in their own words instead. Calling this ends your turn: never combine it with another tool, and \
    never answer the question yourself.";

/// The tool's schema, exactly as the Python Brain sent it.
const SCHEMA: &str = r#"{"type":"object","properties":{"question":{"type":"string","minLength":1,"maxLength":240},"options":{"type":"array","minItems":2,"maxItems":5,"items":{"type":"object","properties":{"label":{"type":"string","minLength":1,"maxLength":80},"description":{"type":"string","maxLength":160}},"required":["label","description"],"additionalProperties":false}},"default_index":{"type":"integer","minimum":0,"maximum":4}},"required":["question","options","default_index"],"additionalProperties":false}"#;

static SCHEMA_VALUE: LazyLock<Value> =
    LazyLock::new(|| json::parse(SCHEMA.as_bytes()).expect("the clarification schema is JSON"));

const MIXED: &str = "Not executed: a clarification cannot be combined with any other tool call. Either ask one \
    question alone or proceed with the request.";
const AFTER_ACTION: &str = "Not executed: an Action already ran in this request, so no clarification can be asked \
    now. Finish the request with what you have and state what remains open.";
const INVALID: &str = "Not executed: the clarification must have one single-line question, two to five distinct \
    single-line options, and a default_index that points to one of them.";
const TEAM_OWNED: &str = "Not asked: the Team asks the schedule, the output, and limits itself; continue the work \
    and call record.";

/// One option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClarifyOption {
    /// Its label.
    pub label: String,
    /// Its description.
    pub description: String,
}

/// One closed clarification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clarification {
    /// The question.
    pub question: String,
    /// Two to five options.
    pub options: Vec<ClarifyOption>,
    /// The recommended option, or none (a Routine turn's).
    pub default_index: Option<usize>,
}

/// One single-line text: NFC, stripped, within `maximum` code points, and free of control, format, private,
/// unassigned, surrogate, and line or paragraph separator characters (`_line`).
fn line(value: &Value, maximum: usize, empty: bool) -> Option<String> {
    let Value::String(raw) = value else {
        return None;
    };
    let normalized = text::nfc(raw);
    let stripped = text::strip(&normalized);
    if (stripped.is_empty() && !empty) || text::len(stripped) > maximum {
        return None;
    }
    let unsafe_category = |character: char| {
        matches!(
            text::category(character),
            "Cc" | "Cf" | "Cs" | "Co" | "Cn" | "Zl" | "Zp"
        )
    };
    (!stripped.chars().any(unsafe_category)).then(|| stripped.to_owned())
}

fn option(value: &Value) -> Option<ClarifyOption> {
    json::has_exactly(value, &["label", "description"]).then_some(())?;
    Some(ClarifyOption {
        label: line(value.get("label")?, MAX_CLARIFICATION_LABEL_CHARS, false)?,
        description: line(
            value.get("description")?,
            MAX_CLARIFICATION_DESCRIPTION_CHARS,
            true,
        )?,
    })
}

impl Clarification {
    /// The closed clarification in `arguments`, or `None`: it recommends exactly one option.
    #[must_use]
    pub fn parse(arguments: &Value) -> Option<Self> {
        json::has_exactly(arguments, &["question", "options", "default_index"]).then_some(())?;
        let question = line(
            arguments.get("question")?,
            MAX_CLARIFICATION_QUESTION_CHARS,
            false,
        )?;
        let raw = json::array_of(arguments, "options")?;
        (MIN_CLARIFICATION_OPTIONS..=MAX_CLARIFICATION_OPTIONS)
            .contains(&raw.len())
            .then_some(())?;
        let options = raw.iter().map(option).collect::<Option<Vec<_>>>()?;
        let default_index = json::u64_of(arguments, "default_index")
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| *index < options.len())?;
        let labels: HashSet<String> = options
            .iter()
            .map(|option| text::casefold(&option.label))
            .collect();
        (labels.len() == options.len()).then_some(Self {
            question,
            options,
            default_index: Some(default_index),
        })
    }

    /// The plain reply shown and remembered (`payload.render_clarification`).
    #[must_use]
    pub fn render(&self) -> String {
        payload::render_clarification(&self.to_value())
    }

    /// `{"question", "options": [{"label", "description"}], "default_index"}`.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let options = self
            .options
            .iter()
            .map(|option| {
                json::object([
                    ("label", json::text(&option.label)),
                    ("description", json::text(&option.description)),
                ])
            })
            .collect();
        json::object([
            ("question", json::text(&self.question)),
            ("options", json::array(options)),
            (
                "default_index",
                self.default_index
                    .map_or(Value::Null, |index| json::uint(index as u64)),
            ),
        ])
    }
}

/// The tool.
#[must_use]
pub fn spec() -> ToolSpec {
    ToolSpec {
        name: TOOL_NAME.to_owned(),
        description: DESCRIPTION.to_owned(),
        schema: SCHEMA_VALUE.clone(),
        strict: None,
    }
}

/// What the tool returns when it runs (it ends the turn).
pub const RECORDED: &str =
    "Question recorded; it is shown to the user with a free-text option and ends this turn.";

/// Whether a clarification asks what Team reads from the person's words itself, by Team's own phrase tables, in the
/// chat's `locale` (`_team_owned`). A reading that cannot complete counts as Team's, which refuses.
fn team_owned(asked: &Clarification, locale: Option<&str>) -> bool {
    std::iter::once(&asked.question)
        .chain(asked.options.iter().map(|option| &option.label))
        .any(|text| phrase::team_asks(text, locale).unwrap_or(true))
}

/// The clarification guard (`_review`): a turn's start runs no Action before it suspends and every resume follows
/// one, so `allowed` is whether the turn is starting; in a Routine turn, a question about the schedule, the output, or
/// limits is Team's to ask (ADR-0101).
///
/// # Errors
///
/// An unanswerable call.
pub fn review(
    messages: &[Message],
    allowed: bool,
    routine_mode: bool,
    locale: Option<&str>,
) -> Review {
    let call = match refusal::lone_call(messages, TOOL_NAME)? {
        LoneCall::Absent => return Ok(None),
        LoneCall::Mixed => return Ok(Some(MIXED.to_owned())),
        LoneCall::Alone(call) => call,
    };
    if !allowed {
        return Ok(Some(AFTER_ACTION.to_owned()));
    }
    Ok(match Clarification::parse(&call.arguments) {
        None => Some(INVALID.to_owned()),
        Some(asked) if routine_mode && team_owned(&asked, locale) => Some(TEAM_OWNED.to_owned()),
        Some(_) => None,
    })
}

/// The clarification that ended the turn, when its last messages are the tool's call and result (`recorded`).
#[must_use]
pub fn recorded(messages: &[Message]) -> Option<Clarification> {
    let [.., call, Message::Tool(result)] = messages else {
        return None;
    };
    if result.name != TOOL_NAME {
        return None;
    }
    match call.assistant()?.calls.as_slice() {
        [only] => Clarification::parse(&only.arguments),
        _ => None,
    }
}

#[cfg(test)]
mod tests;

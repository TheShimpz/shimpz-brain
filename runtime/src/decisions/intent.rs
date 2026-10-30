//! Intent routing (`intent_route.py`): classify one fresh Assistant lifecycle intent, or resolve it against a closed
//! candidate directory, after the Jev fast path when the request carries a decision key.

mod route;

use team_protocol::identifiers;
use team_protocol::payload::{
    CHAT_LOCALES, MAX_CONVERSATION_CHARS, MAX_CONVERSATION_ENTRIES, MAX_CONVERSATION_TEXT_CHARS,
};
use team_protocol::turn::{
    MAX_CAPABILITY_CANDIDATES, MAX_CAPABILITY_NAME_CHARS, MAX_CAPABILITY_SUMMARY_CHARS,
    MAX_INTENT_ROUTE_CANDIDATES, MAX_INTENT_ROUTE_NAME_CHARS, MAX_INTENT_ROUTE_SUMMARY_CHARS,
    MAX_OBJECTIVE_CHARS,
};
use zeroize::Zeroizing;

use super::jev::{self, Question};
use super::{Call, refused, sorted_unique};
use crate::errors::BrainError;
use crate::json::{self, Value};
use crate::provider::{Profile, Providers, Scope};
use crate::server::input::{self, MAX_API_KEY_CHARS, MIN_API_KEY_CHARS, RouteError};
use crate::text::admitted;

/// The 409 text of a cancelled route.
const CANCELLED: &str = "Intent route cancelled";
/// The longest identifier a request names.
const MAX_ID_CHARS: usize = 128;
const INSTALL: &str = "assistant-install";
const UNINSTALL: &str = "assistant-uninstall";

/// One directory candidate of a selection.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    id: String,
    name: String,
    summary: String,
}

/// The last single Assistant explicitly installed, found installed, or uninstalled in this connection.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Reference {
    id: String,
    name: String,
}

/// One entry of committed presentation history: untrusted language evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    role: String,
    text: String,
    truncated: bool,
}

/// An Assistant the Team could install, as untrusted discovery text.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Installable {
    name: String,
    summary: String,
}

/// The closed `IntentRouteInput`, less its provider.
#[derive(Debug)]
struct Request {
    objective: String,
    expected: Option<&'static str>,
    candidates: Vec<Candidate>,
    reference: Option<Reference>,
    conversation: Vec<Entry>,
    locale: String,
    decision: Option<Zeroizing<String>>,
    installable: Option<Vec<Installable>>,
}

/// One route: the answer Team acts on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Route {
    intent: &'static str,
    query: String,
    assistant_ids: Vec<String>,
    reply: String,
    task_follows: bool,
    capability_free: bool,
}

/// `POST /v1/intent-route`: `{"intent", "query", "assistant_ids", "reply", "task_follows"[, "capability"]}`.
///
/// # Errors
///
/// [`RouteError`]: 422 for the request shape, 400/502, and 409 "Intent route cancelled".
pub fn handle(providers: &Providers, scope: &Scope, body: &Value) -> Result<Value, RouteError> {
    let request = read(body)?;
    let config = input::provider(body)?;
    admit(&request)?;
    let call = Call {
        providers,
        scope,
        config: &config,
        profile: Profile::DECISION,
        cancelled: CANCELLED,
    };
    let fast = match &request.decision {
        Some(key) => fast_path(&call, key, &request)?,
        None => jev::FastPath::default(),
    };
    let route = if fast.ordinary {
        Route {
            intent: "ordinary-task",
            capability_free: fast.capability_free,
            ..Route::default()
        }
    } else {
        // A cancelled request never dispatches the fallback (or anything after it).
        if scope.stopped() {
            return Err(BrainError::Cancelled(CANCELLED).into());
        }
        route::create(&call, &request)?
    };
    Ok(respond(&route, request.installable.is_some()))
}

/// The Jev fast path of a classification, over its admitted installable Assistants.
fn fast_path(call: &Call<'_>, key: &str, request: &Request) -> Result<jev::FastPath, RouteError> {
    let installable = request.installable.as_deref().unwrap_or_default();
    for item in installable {
        text(
            &item.name,
            MAX_CAPABILITY_NAME_CHARS,
            "installable Assistant name",
            false,
        )?;
        text(
            &item.summary,
            MAX_CAPABILITY_SUMMARY_CHARS,
            "installable Assistant summary",
            false,
        )?;
    }
    let question = Question {
        objective: &request.objective,
        conversation: request
            .conversation
            .iter()
            .map(|entry| (entry.role.as_str(), entry.text.as_str()))
            .collect(),
        reference: request
            .reference
            .as_ref()
            .map(|reference| reference.name.as_str()),
        installable: installable
            .iter()
            .map(|item| (item.name.as_str(), item.summary.as_str()))
            .collect(),
    };
    jev::classify(call.providers.transport(), call.scope, key, &question)
        .ok_or(BrainError::Cancelled(CANCELLED).into())
}

/// The answer Team reads; a request that described installable Assistants also learns whether it is
/// capability-free.
fn respond(route: &Route, capability: bool) -> Value {
    let mut members = vec![
        ("intent".to_owned(), json::text(route.intent)),
        ("query".to_owned(), json::text(&route.query)),
        (
            "assistant_ids".to_owned(),
            json::array(
                route
                    .assistant_ids
                    .iter()
                    .map(|id| json::text(id))
                    .collect(),
            ),
        ),
        ("reply".to_owned(), json::text(&route.reply)),
        ("task_follows".to_owned(), Value::Bool(route.task_follows)),
    ];
    if capability {
        let need = if route.capability_free {
            "not-needed"
        } else {
            "possible"
        };
        members.push(("capability".to_owned(), json::text(need)));
    }
    json::object_of(members)
}

/// The closed request, refusing every shape violation Pydantic refused with 422.
fn read(body: &Value) -> Result<Request, RouteError> {
    const REQUIRED: [&str; 7] = [
        "provider",
        "objective",
        "expected_intent",
        "candidates",
        "lifecycle_reference",
        "conversation",
        "locale",
    ];
    let allowed = [REQUIRED.as_slice(), &["decision_provider", "installable"]].concat();
    input::closed(body, &allowed, &REQUIRED)?;
    let request = Request {
        objective: input::string(body, "objective", 1, MAX_OBJECTIVE_CHARS)?,
        expected: match body.get("expected_intent") {
            Some(Value::Null) => None,
            Some(Value::String(intent)) if intent == INSTALL => Some(INSTALL),
            Some(Value::String(intent)) if intent == UNINSTALL => Some(UNINSTALL),
            _ => return Err(RouteError::Invalid),
        },
        candidates: input::list(body, "candidates", 0, MAX_INTENT_ROUTE_CANDIDATES)?
            .iter()
            .map(candidate_input)
            .collect::<Result<_, _>>()?,
        reference: input::optional_object(body, "lifecycle_reference")?
            .map(reference_input)
            .transpose()?,
        conversation: input::list(body, "conversation", 0, MAX_CONVERSATION_ENTRIES)?
            .iter()
            .map(entry_input)
            .collect::<Result<_, _>>()?,
        locale: match body.get("locale") {
            Some(Value::String(locale)) if CHAT_LOCALES.contains(&locale.as_str()) => {
                locale.clone()
            }
            _ => return Err(RouteError::Invalid),
        },
        decision: decision_input(body.get("decision_provider"))?,
        installable: installable_input(body.get("installable"))?,
    };
    lane(&request)?;
    Ok(request)
}

/// `validate_lifecycle_context`: a selection carries no lifecycle state or decision key, installable Assistants
/// only reach a decision provider, and the conversation window stays within its characters.
fn lane(request: &Request) -> Result<(), RouteError> {
    let selection_state = request.reference.is_some()
        || !request.conversation.is_empty()
        || request.decision.is_some();
    let window: usize = request
        .conversation
        .iter()
        .map(|entry| entry.text.chars().count())
        .sum();
    if (request.expected.is_some() && selection_state)
        || (request.installable.is_some() && request.decision.is_none())
        || window > MAX_CONVERSATION_CHARS
    {
        return Err(RouteError::Invalid);
    }
    Ok(())
}

fn candidate_input(value: &Value) -> Result<Candidate, RouteError> {
    input::closed(
        value,
        &["id", "name", "summary"],
        &["id", "name", "summary"],
    )?;
    Ok(Candidate {
        id: input::string(value, "id", 1, MAX_ID_CHARS)?,
        name: input::string(value, "name", 1, MAX_INTENT_ROUTE_NAME_CHARS)?,
        summary: input::string(value, "summary", 0, MAX_INTENT_ROUTE_SUMMARY_CHARS)?,
    })
}

fn reference_input(value: &Value) -> Result<Reference, RouteError> {
    input::closed(value, &["id", "name"], &["id", "name"])?;
    Ok(Reference {
        id: input::string(value, "id", 1, MAX_ID_CHARS)?,
        name: input::string(value, "name", 1, MAX_INTENT_ROUTE_NAME_CHARS)?,
    })
}

/// One `ConversationEntryInput` (a strict model).
fn entry_input(value: &Value) -> Result<Entry, RouteError> {
    input::closed(
        value,
        &["role", "text", "truncated"],
        &["role", "text", "truncated"],
    )?;
    let role = match value.get("role") {
        Some(Value::String(role)) if role == "user" || role == "assistant" => role.clone(),
        _ => return Err(RouteError::Invalid),
    };
    Ok(Entry {
        role,
        text: input::string(value, "text", 1, MAX_CONVERSATION_TEXT_CHARS)?,
        truncated: input::boolean(value, "truncated")?,
    })
}

/// `DecisionProviderInput`: the Supervisor's request-scoped TypeSafe key, or `None`.
fn decision_input(value: Option<&Value>) -> Result<Option<Zeroizing<String>>, RouteError> {
    let Some(value) = value.filter(|value| **value != Value::Null) else {
        return Ok(None);
    };
    input::closed(value, &["provider", "api_key"], &["provider", "api_key"])?;
    if json::str_of(value, "provider") != Some("typesafe") {
        return Err(RouteError::Invalid);
    }
    let key = input::string(value, "api_key", MIN_API_KEY_CHARS, MAX_API_KEY_CHARS)?;
    Ok(Some(Zeroizing::new(key)))
}

/// `InstallableAssistantInput`s: absent or null, else one to the planner's candidates.
fn installable_input(value: Option<&Value>) -> Result<Option<Vec<Installable>>, RouteError> {
    let Some(value) = value.filter(|value| **value != Value::Null) else {
        return Ok(None);
    };
    let Value::Array(items) = value else {
        return Err(RouteError::Invalid);
    };
    if !(1..=MAX_CAPABILITY_CANDIDATES).contains(&items.len()) {
        return Err(RouteError::Invalid);
    }
    items
        .iter()
        .map(|item| {
            input::closed(item, &["name", "summary"], &["name", "summary"])?;
            Ok(Installable {
                name: input::string(item, "name", 1, MAX_CAPABILITY_NAME_CHARS)?,
                summary: input::string(item, "summary", 1, MAX_CAPABILITY_SUMMARY_CHARS)?,
            })
        })
        .collect::<Result<_, _>>()
        .map(Some)
}

/// `intent_route._text` with its 400 text.
fn text(value: &str, max: usize, label: &str, layout: bool) -> Result<(), RouteError> {
    if admitted(value, max, false, layout) {
        Ok(())
    } else {
        Err(refused(&format!("invalid {label}")))
    }
}

/// `_identifier`: an Assistant id as the Team protocol admits it.
fn assistant_id(value: &str) -> Result<(), RouteError> {
    if identifiers::canonical_assistant_id(&json::text(value)).is_some() {
        Ok(())
    } else {
        Err(refused("invalid Assistant id"))
    }
}

/// `validate_inputs`: classification or closed-directory selection admitted before any provider access, its
/// checks in the Python order.
fn admit(request: &Request) -> Result<(), RouteError> {
    text(
        &request.objective,
        MAX_OBJECTIVE_CHARS,
        "route objective",
        true,
    )?;
    for entry in &request.conversation {
        text(
            &entry.text,
            MAX_CONVERSATION_TEXT_CHARS,
            "conversation text",
            true,
        )?;
    }
    match request.expected {
        None => admit_classification(request),
        Some(expected) => admit_selection(request, expected),
    }
}

/// A classification names no candidate, and its reference only a canonical Assistant.
fn admit_classification(request: &Request) -> Result<(), RouteError> {
    if !request.candidates.is_empty() {
        return Err(refused(
            "classification cannot include directory candidates",
        ));
    }
    if let Some(reference) = &request.reference {
        assistant_id(&reference.id)?;
        let name = &reference.name;
        text(
            name,
            MAX_INTENT_ROUTE_NAME_CHARS,
            "Assistant reference name",
            false,
        )?;
    }
    Ok(())
}

/// A selection's directory: canonical, sorted, distinct candidates; an uninstall's without summaries.
fn admit_selection(request: &Request, expected: &str) -> Result<(), RouteError> {
    for candidate in &request.candidates {
        if !admitted(
            &candidate.summary,
            MAX_INTENT_ROUTE_SUMMARY_CHARS,
            true,
            false,
        ) {
            return Err(refused("invalid Assistant summary"));
        }
        if expected == UNINSTALL && !candidate.summary.is_empty() {
            return Err(refused("uninstall candidates cannot expose summaries"));
        }
        assistant_id(&candidate.id)?;
        text(
            &candidate.name,
            MAX_INTENT_ROUTE_NAME_CHARS,
            "Assistant name",
            false,
        )?;
    }
    let ids: Vec<&str> = request
        .candidates
        .iter()
        .map(|item| item.id.as_str())
        .collect();
    if sorted_unique(&ids) {
        Ok(())
    } else {
        Err(refused("invalid directory candidate order"))
    }
}

#[cfg(test)]
mod tests;

//! The system policy prompt of one Team chat turn (`turn_prompt.py`) and the interface languages
//! (`interface_language.py`).

mod text;

use std::time::{SystemTime, UNIX_EPOCH};

use team_protocol::payload::CHAT_LOCALES;
use team_protocol::python::datetime::ord_to_ymd;

use self::text::{
    ANSWER_OTHER, ANSWER_RECORDS_ONLY, ANSWER_SETTINGS, ASK_HOW_OFTEN, ATTACHMENTS_POLICY,
    ATTACHMENTS_RESTRICTED, AUTHORITY, BROWSER_SECTION, CLARIFY_AND_COMPLETE, CONTRACTS_LABEL,
    DATE_NOTE, EMPTY_SCOPE, IDENTITY, LANGUAGE_AFTER, LANGUAGE_BEFORE, MEMORY_POLICY,
    MEMORY_READ_ONLY, MODE_END, MODE_POLICY, MODE_REFINE, MODE_RUN, MODE_RUN_SETTINGS,
    QUESTION_END, QUESTION_LABEL, RERUN_POLICY, ROUTINES_DATA, ROUTINES_LIMIT, ROUTINES_POLICY,
    ROUTINES_RECORD, RUN_ONCE, RUN_RECORDS_ONLY, RUN_SETTINGS, SELECTED_LABEL, SELECTED_POLICY,
};
use crate::json::{self, Value};
use crate::tools::routine;
use crate::turn::context::{AssistantDefinition, Memory, TurnContext};

/// The proleptic Gregorian ordinal of 1970-01-01.
const EPOCH_ORDINAL: i64 = 719_163;
const SECONDS_PER_DAY: u64 = 86_400;

/// Today's UTC date as `(year, month, day)`.
#[must_use]
pub fn utc_date() -> (i32, i32, i32) {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() / SECONDS_PER_DAY);
    let ordinal = i64::try_from(days).map_or(EPOCH_ORDINAL, |days| EPOCH_ORDINAL + days);
    ord_to_ymd(i32::try_from(ordinal).unwrap_or(i32::MAX))
}

/// Today's UTC date (`YYYY-MM-DD`).
#[must_use]
pub fn today() -> String {
    let (year, month, day) = utc_date();
    format!("{year:04}-{month:02}-{day:02}")
}

/// Whether `locale` is one closed interface language.
#[must_use]
pub fn valid_locale(locale: &str) -> bool {
    CHAT_LOCALES.contains(&locale)
}

/// The English name of one closed interface language, as prompts state it (`language_name`).
#[must_use]
pub fn language_name(locale: &str) -> Option<&'static str> {
    Some(match locale {
        "ar" => "Arabic",
        "de" => "German",
        "en" => "English",
        "es" => "Spanish",
        "fr" => "French",
        "ja" => "Japanese",
        "pt" => "Brazilian Portuguese",
        "zh" => "Simplified Chinese",
        _ => return None,
    })
}

/// The Assistant contracts as canonical JSON: sorted keys, compact (the `file_input` an Action declares only where it
/// takes an attached file's id, ADR-0093).
fn contracts(assistants: &[AssistantDefinition]) -> String {
    let contracts = assistants
        .iter()
        .map(|assistant| {
            let actions = assistant
                .actions
                .iter()
                .map(|action| {
                    let mut members = vec![
                        ("id".to_owned(), json::text(&action.id)),
                        ("summary".to_owned(), json::text(&action.summary)),
                    ];
                    if let Some(input) = action.input_files.first() {
                        members.push(("file_input".to_owned(), json::text(input)));
                    }
                    json::object_of(members)
                })
                .collect();
            json::object([
                ("genesis", json::text(&assistant.genesis)),
                ("id", json::text(&assistant.id)),
                ("actions", json::array(actions)),
            ])
        })
        .collect();
    json::dumps(&json::array(contracts), json::SORTED)
}

/// `json.dumps(value, ensure_ascii=False)`.
fn spaced(value: &Value) -> String {
    json::dumps(value, json::SPACED)
}

/// The memory policy and what the Team remembers, quoted as data; nothing where memory is unavailable. A turn that
/// reads attachments sees the memories but cannot change them (`_memory_section`).
fn memory_section(memories: Option<&[Memory]>, writable: bool) -> String {
    let Some(memories) = memories else {
        return String::new();
    };
    let remembered = memories
        .iter()
        .map(|memory| {
            json::object([
                ("topic", json::text(&memory.topic)),
                ("preference", json::text(&memory.preference)),
            ])
        })
        .collect();
    let policy = if writable {
        MEMORY_POLICY
    } else {
        MEMORY_READ_ONLY
    };
    format!("{policy}{}\n\n", spaced(&json::array(remembered)))
}

/// A Routine's steps in order as `[Assistant, Action]`, with the count appended when one Action repeats in a row
/// (`_runs`).
fn runs(steps: &[Value]) -> Value {
    let mut runs: Vec<(Value, Value, u64)> = Vec::new();
    for step in steps {
        let pair = (
            step.get("assistant").cloned().unwrap_or(Value::Null),
            step.get("action").cloned().unwrap_or(Value::Null),
        );
        match runs.last_mut() {
            Some(last) if last.0 == pair.0 && last.1 == pair.1 => last.2 += 1,
            _ => runs.push((pair.0, pair.1, 1)),
        }
    }
    json::array(
        runs.into_iter()
            .map(|(assistant, action, count)| {
                let mut run = vec![assistant, action];
                if count > 1 {
                    run.push(json::uint(count));
                }
                json::array(run)
            })
            .collect(),
    )
}

/// One listed Routine as the prompt shows it.
fn listed(item: &Value) -> Value {
    let member = |name: &str| item.get(name).cloned().unwrap_or(Value::Null);
    let steps = json::array_of(item, "steps").unwrap_or_default();
    json::object([
        ("routine_id", member("routine_id")),
        ("name", member("name")),
        ("schedule", member("schedule")),
        ("timezone", member("timezone")),
        ("timezone_source", member("timezone_source")),
        ("output", member("output")),
        ("steps", runs(steps)),
    ])
}

/// The Routine policy and the Team's Routines as data; nothing when the Routine tool is withheld
/// (`_routines_section`).
fn routines_section(context: &TurnContext, settings: bool) -> String {
    let Some(routines) = &context.routines else {
        return String::new();
    };
    let run = if context.routine_settings_answer {
        RUN_RECORDS_ONLY
    } else if settings {
        RUN_SETTINGS
    } else {
        RUN_ONCE
    };
    let how_often = if context.routine_mode {
        ""
    } else {
        ASK_HOW_OFTEN
    };
    let listed = json::array(routines.iter().map(listed).collect());
    format!(
        "{ROUTINES_POLICY}{run}{ROUTINES_RECORD}{}{ROUTINES_LIMIT}{how_often}{ROUTINES_DATA}{}\n\n",
        routine::intervals(),
        spaced(&listed)
    )
}

/// The Routine the user's message selects and how to refine it (ADR-0122) (`_selected_section`).
fn selected_section(selected: Option<&Value>) -> String {
    selected.map_or_else(String::new, |selected| {
        format!("{SELECTED_LABEL}{}{SELECTED_POLICY}", spaced(selected))
    })
}

/// The Routine question Team asked before this message, which the message may answer (`_question_section`).
fn question_section(question: Option<&Value>, records_only: bool) -> String {
    let Some(question) = question else {
        return String::new();
    };
    let answered = if records_only {
        ANSWER_RECORDS_ONLY
    } else if routine::settings_question(Some(question)) {
        ANSWER_SETTINGS
    } else {
        ANSWER_OTHER
    };
    format!(
        "{QUESTION_LABEL}{}\n{answered}{QUESTION_END}",
        spaced(question)
    )
}

/// The Routine-mode guidance when Team read that this chat is about a Routine (`_routine_mode_section`).
fn routine_mode_section(context: &TurnContext, settings: bool) -> String {
    if !context.routine_mode || context.routine_settings_answer {
        return String::new();
    }
    let run = if settings {
        MODE_RUN_SETTINGS
    } else {
        MODE_RUN
    };
    if context.routine_selected.is_none() {
        return format!("{MODE_POLICY}{run}{MODE_END}");
    }
    // The selected Routine is changed by refining it, never by running its unchanged work again (ADR-0122).
    let mut characters = run.chars();
    let lowered: String = characters
        .next()
        .map(|first| first.to_lowercase().chain(characters).collect())
        .unwrap_or_default();
    format!("{MODE_POLICY}{MODE_REFINE} {lowered}{MODE_END}")
}

/// The work a pending Team question asks to run again (`_rerun_section`).
fn rerun_section(rerun: Option<&Vec<Value>>) -> String {
    rerun.map_or_else(String::new, |work| {
        format!("{RERUN_POLICY}{}\n\n", spaced(&json::array(work.clone())))
    })
}

/// How to treat files attached to the current message (ADR-0093) (`_attachments_section`).
fn attachments_section(context: &TurnContext) -> String {
    if !context.attachments.any() {
        return String::new();
    }
    let rest = if context.attachments.reads_content() {
        ATTACHMENTS_RESTRICTED
    } else {
        "\n"
    };
    format!("{ATTACHMENTS_POLICY}{rest}")
}

/// The interface language every reply follows (ADR-0090) (`_language_section`).
fn language_section(locale: Option<&str>) -> String {
    locale
        .and_then(language_name)
        .map_or_else(String::new, |name| {
            format!("{LANGUAGE_BEFORE}{name}{LANGUAGE_AFTER}")
        })
}

/// The sections after the contracts, in order; the date stays last so everything before it is a stable cacheable
/// prefix.
fn sections(context: &TurnContext) -> String {
    let records_only = context.routine_settings_answer;
    let settings = routine::settings_question(context.routine_question.as_ref());
    [
        memory_section(context.memories.as_deref(), !context.attachments.any()),
        routines_section(context, settings),
        selected_section(context.routine_selected.as_ref()),
        question_section(context.routine_question.as_ref(), records_only),
        routine_mode_section(context, settings),
        rerun_section(context.routine_rerun.as_ref()),
        attachments_section(context),
        if context.browser {
            BROWSER_SECTION.to_owned()
        } else {
            String::new()
        },
        language_section(context.locale.as_deref()),
    ]
    .concat()
}

/// The turn's policy prompt, built from its surface: `context.assistants` are the Assistants it is offered and
/// `context.browser` whether it offers the Browser.
#[must_use]
pub fn system_prompt(context: &TurnContext) -> String {
    // A settings answer only records again, so the Actions it is not offered are no limit of the Team's.
    let empty =
        context.assistants.is_empty() && !context.routine_settings_answer && !context.browser;
    format!(
        "{IDENTITY}{CLARIFY_AND_COMPLETE}{AUTHORITY}{}\n\n{}{CONTRACTS_LABEL}{}\n\n{}Current date: {} ({}). \
         {DATE_NOTE}",
        json::dumps(&json::text(&context.team_name), json::ASCII),
        if empty { EMPTY_SCOPE } else { "" },
        contracts(&context.assistants),
        sections(context),
        context.turn_date,
        context.timezone,
    )
}

#[cfg(test)]
mod tests;

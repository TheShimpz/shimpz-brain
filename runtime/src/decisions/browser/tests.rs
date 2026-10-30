//! The Browser operations against the fake provider origin (`test_browser_step.py`, `test_browser_step_api.py`).

mod render;

use super::*;
use crate::decisions::testing::{self, contract, parse, provider_failed, run, with, without};
use crate::errors::BrainError;
use crate::provider::testing::{Answer, Received, openai_body};

const OBSERVATION: &str = "url: https://example.com\ntitle: Example\nactions:\np1:0 link \"Edital\" (click)\npage:\n- heading";
const NULLS: [&str; 9] = [
    "url",
    "ref",
    "refs",
    "text",
    "direction",
    "summary",
    "reason",
    "sentence",
    "note",
];

fn entry() -> Value {
    parse(
        r#"{"step":{"kind":"navigate","url":"https://example.com"},"outcome":"ok","page":{"url":"about:blank","title":""},"text":"url: about:blank","downloads":null,"screenshot":false}"#,
    )
}

fn empty_thread() -> Value {
    parse(r#"{"folded":[],"folded_omitted":0,"memory":"","unnoted":0,"steps":[]}"#)
}

/// The step request with one earlier step.
fn body(provider: &str) -> Value {
    let model = if provider == "openai" {
        "gpt-6.1-sol"
    } else {
        "claude-sonnet-5-5"
    };
    json::object([
        ("provider", testing::provider(provider, model)),
        ("objective", json::text("Veja o prazo do edital.")),
        ("language", json::text("pt-BR")),
        (
            "thread",
            with(&empty_thread(), "steps", json::array(vec![entry()])),
        ),
        ("observation", json::text(OBSERVATION)),
        ("screenshot", Value::Null),
        (
            "downloads",
            parse(r#"{"files":["report.csv"],"pending":1,"truncated":false}"#),
        ),
    ])
}

fn fold_body(provider: &str, first: i64, last: i64) -> Value {
    with(
        &with(&body(provider), "first", json::int(first)),
        "last",
        json::int(last),
    )
}

fn check_body(provider: &str, summary: &str) -> Value {
    let step = body(provider);
    json::object([
        ("provider", step.get("provider").cloned().unwrap()),
        ("objective", json::text("Veja o prazo do edital.")),
        ("language", json::text("pt-BR")),
        ("summary", json::text(summary)),
    ])
}

/// A flat step answer: every field null except `fields`; Haiku's has no `refs`.
fn step_answer(fields: &[(&str, Value)], threaded: bool) -> String {
    let members = NULLS
        .iter()
        .filter(|name| threaded || **name != "refs")
        .map(|name| ((*name).to_owned(), Value::Null));
    let mut value = json::object_of(
        std::iter::once(("kind".to_owned(), json::text("wait")))
            .chain(members)
            .collect(),
    );
    for (key, field) in fields {
        value = with(&value, key, field.clone());
    }
    json::compact(&value)
}

fn operate(
    operation: Operation,
    answers: Vec<Answer>,
    body: &Value,
) -> (Result<Value, RouteError>, Vec<Received>) {
    run(answers, |providers, scope| {
        handle(providers, scope, operation, body)
    })
}

/// The request was refused before any provider call.
fn refused_early(operation: Operation, body: &Value) -> Result<Value, RouteError> {
    let (outcome, sent) = operate(operation, vec![Answer::openai_text("{}")], body);
    assert!(sent.is_empty(), "no provider request");
    outcome
}

#[test]
fn an_openai_step_reads_on_luna_at_medium_effort() {
    let answer = step_answer(
        &[("kind", json::text("read")), ("refs", parse(r#"["p1:0"]"#))],
        true,
    );
    let (outcome, sent) = operate(
        Operation::Step,
        vec![Answer::openai_text(&answer)],
        &body("openai"),
    );
    assert_eq!(
        outcome,
        Ok(parse(
            r#"{"step":{"kind":"read","refs":["p1:0"]},"model":"gpt-6-luna"}"#
        ))
    );
    let request = sent[0].json();
    assert_eq!(json::str_of(&request, "model"), Some("gpt-6-luna"));
    assert_eq!(
        request.get("reasoning"),
        Some(&parse(r#"{"effort":"medium"}"#))
    );
    assert_eq!(
        json::u64_of(&request, "max_output_tokens"),
        Some(u64::from(MAX_STEP_OUTPUT_TOKENS))
    );
    let format = request
        .get("text")
        .and_then(|text| text.get("format"))
        .unwrap();
    assert_eq!(json::str_of(format, "name"), Some("ThreadAnswer"));
}

#[test]
fn a_haiku_step_clicks_on_its_own_model_at_low_effort() {
    let answer = step_answer(
        &[("kind", json::text("click")), ("ref", json::text("p1:0"))],
        false,
    );
    let (outcome, sent) = operate(
        Operation::Step,
        vec![Answer::anthropic_text(&answer)],
        &body("anthropic"),
    );
    assert_eq!(
        outcome,
        Ok(parse(
            r#"{"step":{"kind":"click","ref":"p1:0"},"model":"claude-haiku-5-5"}"#
        ))
    );
    let request = sent[0].json();
    assert_eq!(json::str_of(&request, "model"), Some("claude-haiku-5-5"));
    assert_eq!(
        json::u64_of(&request, "max_tokens"),
        Some(u64::from(MAX_STEP_OUTPUT_TOKENS))
    );
    let effort = request
        .get("output_config")
        .and_then(|config| config.get("effort"));
    assert_eq!(effort, Some(&json::text("low")));
}

#[test]
fn an_answer_outside_the_closed_set_is_a_provider_failure() {
    for answer in [
        step_answer(&[("kind", json::text("click"))], true),
        step_answer(
            &[
                ("kind", json::text("wait")),
                ("url", json::text("https://example.com")),
            ],
            true,
        ),
        "no".to_owned(),
    ] {
        let (outcome, _) = operate(
            Operation::Step,
            vec![Answer::openai_text(&answer)],
            &body("openai"),
        );
        assert_eq!(outcome, Err(provider_failed()), "{answer}");
    }
}

#[test]
fn a_request_past_haikus_priced_bound_is_refused_before_any_call() {
    let page = json::object([
        (
            "url",
            json::text(&format!("https://example.com/{}", "a".repeat(2_000))),
        ),
        ("title", json::text(&"t".repeat(300))),
    ]);
    let record = json::object([
        ("step", parse(r#"{"kind":"wait"}"#)),
        ("outcome", json::text("ok")),
        ("page", page),
        ("text", json::text("")),
        ("downloads", Value::Null),
        ("screenshot", Value::Bool(false)),
    ]);
    let thread = with(&empty_thread(), "steps", json::array(vec![record; 40]));
    let request = with(
        &with(&body("anthropic"), "thread", thread),
        "observation",
        json::text(&"o".repeat(24 * 1024)),
    );
    let refused = refused_early(Operation::Step, &request);
    assert_eq!(
        refused,
        Err(contract("Browser step request exceeds its priced bound"))
    );
}

#[test]
fn a_malformed_step_request_is_refused_before_any_call() {
    let png = |size: usize| {
        let mut image = PNG_SIGNATURE.to_vec();
        image.resize(size, b'x');
        json::text(&BASE64.encode(image))
    };
    let folded_done = parse(
        r#"{"folded":[{"step":{"kind":"done","summary":"x"},"outcome":"ok","page":{"url":"about:blank","title":""}}],"folded_omitted":0,"memory":"","unnoted":0,"steps":[]}"#,
    );
    let screenshot = Err(contract("invalid Browser screenshot"));
    let request = Err(contract("invalid Browser step request"));
    for (key, value, expected) in [
        ("thread", folded_done, request.clone()),
        (
            "observation",
            json::text(&"é".repeat(24 * 1024)),
            request.clone(),
        ),
        (
            "downloads",
            parse(r#"{"files":["b","a"],"pending":0,"truncated":false}"#),
            request,
        ),
        ("screenshot", json::text("not base64!"), screenshot.clone()),
        (
            "screenshot",
            json::text(&BASE64.encode(b"GIF89a")),
            screenshot.clone(),
        ),
        ("screenshot", png(MAX_SCREENSHOT_BYTES + 1), screenshot),
    ] {
        assert_eq!(
            refused_early(Operation::Step, &with(&body("openai"), key, value)),
            expected,
            "{key}"
        );
    }
}

#[test]
fn an_open_or_mistyped_request_is_refused_whole() {
    let base = body("openai");
    for request in [
        with(&base, "objective", json::text("")),
        with(&base, "language", json::text("fr")),
        with(&base, "extra", json::int(1)),
        with(&base, "history", json::array(Vec::new())),
        with(&base, "thread", Value::Null),
        with(&base, "observation", json::text(&"x".repeat(24 * 1024 + 1))),
        with(&base, "screenshot", json::int(7)),
        with(&base, "downloads", json::array(Vec::new())),
        without(&base, "screenshot"),
    ] {
        assert_eq!(
            refused_early(Operation::Step, &request),
            Err(RouteError::Invalid)
        );
    }
    assert_eq!(
        refused_early(Operation::Fold, &base),
        Err(RouteError::Invalid)
    );
    let check = with(&check_body("openai", "ok"), "thread", parse("{}"));
    assert_eq!(
        refused_early(Operation::Check, &check),
        Err(RouteError::Invalid)
    );
    let foreign = with(
        &base,
        "provider",
        testing::provider("anthropic", "claude-haiku-5-5"),
    );
    assert_eq!(
        refused_early(Operation::Step, &foreign),
        Err(contract("unsupported model for provider"))
    );
}

#[test]
fn a_fold_extends_the_thread_and_returns_the_rewritten_memory() {
    let answer = Answer::openai_text("Vaga V101: R$ 9.500\u{7} por mês");
    let (outcome, sent) = operate(Operation::Fold, vec![answer], &fold_body("openai", 1, 1));
    assert_eq!(
        outcome,
        Ok(parse(
            r#"{"memory":"Vaga V101: R$ 9.500  por mês","model":"gpt-6-luna"}"#
        ))
    );
    let request = sent[0].json();
    assert_eq!(request.get("text"), None, "a fold writes plain text");
    assert_eq!(
        json::u64_of(&request, "max_output_tokens"),
        Some(u64::from(MAX_WRITE_OUTPUT_TOKENS))
    );
    let (_, step_sent) = operate(
        Operation::Step,
        vec![Answer::openai_text("{}")],
        &body("openai"),
    );
    let (fold_input, step_input) = (
        json::array_of(&request, "input").unwrap(),
        json::array_of(&step_sent[0].json(), "input")
            .unwrap()
            .to_vec(),
    );
    assert_eq!(&fold_input[..fold_input.len() - 1], &step_input[..]);
    let last = testing::openai_texts(&sent[0]).pop().unwrap();
    assert_eq!(last, thread::fold_instruction(1, 1));
}

#[test]
fn a_blank_long_or_cut_rewrite_keeps_the_old_memory() {
    let incomplete = with(
        &openai_body("V101: R$ 9.500"),
        "status",
        json::text("incomplete"),
    );
    for answer in [
        Answer::openai_text("   "),
        Answer::openai_text(&"x".repeat(MAX_MEMORY_BYTES + 1)),
        Answer::json(200, &incomplete),
    ] {
        let (outcome, _) = operate(Operation::Fold, vec![answer], &fold_body("openai", 1, 1));
        assert_eq!(
            outcome,
            Ok(parse(r#"{"memory":null,"model":"gpt-6-luna"}"#))
        );
    }
    assert_eq!(
        refused_early(Operation::Fold, &fold_body("openai", 2, 1)),
        Err(contract("invalid Browser step request"))
    );
    assert_eq!(
        refused_early(Operation::Fold, &fold_body("openai", 0, 1)),
        Err(RouteError::Invalid)
    );
    let text_number = with(&fold_body("openai", 1, 1), "first", json::text("1"));
    assert_eq!(
        refused_early(Operation::Fold, &text_number),
        Err(RouteError::Invalid)
    );
}

#[test]
fn a_final_answer_is_written_from_the_thread_within_the_summary_bound() {
    let answer = Answer::openai_text(&"x".repeat(MAX_SUMMARY_BYTES + 50));
    let (outcome, sent) = operate(Operation::Final, vec![answer], &body("openai"));
    let summary = outcome.unwrap();
    assert_eq!(
        json::str_of(&summary, "summary").map(str::len),
        Some(MAX_SUMMARY_BYTES)
    );
    assert_eq!(
        testing::openai_texts(&sent[0]).pop().as_deref(),
        Some(thread::FINAL)
    );
    let (outcome, _) = operate(
        Operation::Final,
        vec![Answer::raw(400, "connection reset")],
        &body("openai"),
    );
    assert_eq!(outcome, Err(provider_failed()));
}

#[test]
fn the_check_judges_completeness_only() {
    let request = check_body("openai", "Prazo: 30/11.");
    let complete = Answer::openai_text(r#"{"complete":true,"missing":null}"#);
    let (outcome, sent) = operate(Operation::Check, vec![complete], &request);
    assert_eq!(
        outcome,
        Ok(parse(
            r#"{"complete":true,"missing":null,"model":"gpt-6-luna"}"#
        ))
    );
    let question = testing::openai_texts(&sent[0]).concat();
    assert!(
        question.ends_with(r#""answer":"Prazo: 30/11."}"#) && question.starts_with(thread::CHECK)
    );
    let short = r#"{"complete":false,"missing":"Falta a fonte."}"#;
    let (outcome, _) = operate(Operation::Check, vec![Answer::openai_text(short)], &request);
    assert_eq!(
        outcome,
        Ok(parse(
            r#"{"complete":false,"missing":"Falta a fonte.","model":"gpt-6-luna"}"#
        ))
    );
    let vague = Answer::openai_text(r#"{"complete":false,"missing":"  "}"#);
    assert_eq!(
        operate(Operation::Check, vec![vague], &request).0,
        Err(provider_failed())
    );
    assert_eq!(
        refused_early(Operation::Check, &check_body("openai", "   ")),
        Err(contract("invalid Browser check request"))
    );
}

#[test]
fn haiku_folds_checks_and_finishes_nothing_and_makes_no_call() {
    let model = r#","model":"claude-haiku-5-5"}"#;
    for (operation, request, expected) in [
        (
            Operation::Fold,
            fold_body("anthropic", 1, 1),
            format!(r#"{{"memory":null{model}"#),
        ),
        (
            Operation::Final,
            body("anthropic"),
            format!(r#"{{"summary":null{model}"#),
        ),
        (
            Operation::Check,
            check_body("anthropic", "ok"),
            format!(r#"{{"complete":null,"missing":null{model}"#),
        ),
    ] {
        assert_eq!(refused_early(operation, &request), Ok(parse(&expected)));
    }
}

#[test]
fn a_cancelled_step_says_so() {
    let request = body("openai");
    let (outcome, sent) = testing::cancelled_after(
        1,
        |_| Answer::hang(),
        |providers, scope| handle(providers, scope, Operation::Step, &request),
    );
    assert_eq!(
        outcome,
        Err(RouteError::Brain(BrainError::Cancelled(CANCELLED)))
    );
    assert_eq!(sent.len(), 1);
}

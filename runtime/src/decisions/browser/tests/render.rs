//! How a request renders: the thread as one conversation that only grows at its end, Haiku's newest records, and
//! the priced bounds the protocol's measure keeps.

use team_protocol::browser::{
    MAX_FOLDED_BYTES, MAX_OBJECTIVE_CHARS, canonical_thread, record_bytes,
};

use super::*;

fn admitted(changes: &[(&str, Value)]) -> Admitted {
    let request = changes
        .iter()
        .fold(body("openai"), |request, (key, value)| {
            with(&request, key, value.clone())
        });
    validate(&request).expect("an admitted request")
}

fn image() -> String {
    BASE64.encode([PNG_SIGNATURE, b"pixels"].concat())
}

fn record() -> Value {
    parse(
        r#"{"step":{"kind":"navigate","url":"https://example.com"},"outcome":"ok","page":{"url":"about:blank","title":""}}"#,
    )
}

#[test]
fn the_current_page_and_its_image_become_history_unchanged() {
    let first = thread::messages(
        &admitted(&[
            ("thread", empty_thread()),
            ("screenshot", json::text(&image())),
        ]),
        None,
    );
    assert_eq!(first.system, Some(thread::SYSTEM));
    assert_eq!(
        first.messages[0],
        r#"{"objective":"Veja o prazo do edital.","language":"pt-BR"}"#
    );
    let current = first.messages.last().unwrap();
    let block = Block::Image {
        media_type: "image/png".into(),
        base64: image(),
    };
    assert_eq!(first.extra, Some((first.messages.len() - 1, vec![block])));
    // The next request's history holds that page, its image now a fixed text, then the step taken on it.
    let downloads = body("openai").get("downloads").cloned().unwrap();
    let shown = with(
        &with(
            &with(&entry(), "text", json::text(OBSERVATION)),
            "downloads",
            downloads,
        ),
        "screenshot",
        Value::Bool(true),
    );
    let later = thread::messages(
        &admitted(&[(
            "thread",
            with(&empty_thread(), "steps", json::array(vec![shown])),
        )]),
        None,
    );
    assert_eq!(
        later.messages[1],
        format!("{current}{}", thread::SCREENSHOT_SHOWN)
    );
    assert!(later.messages[2].starts_with("Your step: "));
}

#[test]
fn without_a_screenshot_each_request_is_a_prefix_of_the_next() {
    let downloads = body("openai").get("downloads").cloned().unwrap();
    let mut steps = Vec::new();
    let mut requests = Vec::new();
    for number in 0..4 {
        let observation = format!("url: https://example.com/{number}");
        let thread = with(&empty_thread(), "steps", json::array(steps.clone()));
        requests.push(thread::messages(
            &admitted(&[
                ("thread", thread),
                ("observation", json::text(&observation)),
            ]),
            None,
        ));
        steps.push(with(
            &with(&entry(), "text", json::text(&observation)),
            "downloads",
            downloads.clone(),
        ));
    }
    for pair in requests.windows(2) {
        assert_eq!(
            pair[1].messages[..pair[0].messages.len()],
            pair[0].messages[..]
        );
    }
}

#[test]
fn folded_steps_and_the_memory_form_one_earlier_message_of_untrusted_notes() {
    let thread = parse(
        r#"{"folded":[],"folded_omitted":3,"memory":"V101: R$ 9.500","unnoted":0,"steps":[]}"#,
    );
    let thread = with(&thread, "folded", json::array(vec![record()]));
    let rendered = thread::messages(&admitted(&[("thread", thread.clone())]), None);
    let earlier = &rendered.messages[1];
    assert!(earlier.starts_with(&format!(
        "{}(3 earlier steps omitted)\nStep 4 at about:blank",
        thread::EARLIER
    )));
    assert!(earlier.ends_with(&format!("{}V101: R$ 9.500", thread::MEMORY)));
    assert!(
        rendered
            .messages
            .last()
            .unwrap()
            .starts_with("Page at step 5:\n")
    );
    // Steps whose pages left with nothing written down are said, even beside a memory at its bound.
    let full = with(
        &with(&thread, "memory", json::text(&"m".repeat(MAX_MEMORY_BYTES))),
        "unnoted",
        json::int(2),
    );
    let earlier = thread::messages(&admitted(&[("thread", full)]), None).messages[1].clone();
    assert!(earlier.ends_with(
        "(2 earlier steps' pages left with nothing written down; read them again if needed)"
    ));
}

#[test]
fn the_rendered_text_stays_within_the_protocols_measure() {
    let page = json::object([
        (
            "url",
            json::text(&format!("https://example.com/{}", "a".repeat(2_000))),
        ),
        ("title", json::text(&"t".repeat(300))),
    ]);
    let long_entry = with(
        &with(&entry(), "text", json::text(&"é".repeat(6_000))),
        "page",
        page.clone(),
    );
    let long_record = with(&record(), "page", page);
    let thread = json::object([
        ("folded", json::array(vec![long_record; 3])),
        ("folded_omitted", json::int(0)),
        ("memory", json::text(&"n".repeat(8_000))),
        ("unnoted", json::int(0)),
        ("steps", json::array(vec![long_entry; 2])),
    ]);
    let request = admitted(&[
        ("thread", thread),
        ("observation", json::text(&"o".repeat(20_000))),
    ]);
    let measured = protocol::request_text_bytes(
        &request.objective,
        &request.thread,
        &request.observation,
        &request.downloads,
    );
    let rendered = thread::messages(&request, Some(thread::FINAL.to_owned()));
    let without_system = Rendered {
        system: None,
        ..rendered
    };
    assert!(without_system.text_bytes() <= measured + thread::FINAL.len());
}

#[test]
fn haiku_steps_on_its_newest_records_within_the_protocols_measure() {
    let records = json::array(vec![record(); 25]);
    let thread = with(
        &with(&empty_thread(), "folded", records),
        "steps",
        json::array(vec![entry(); 20]),
    );
    let rendered = haiku::prompt(&admitted(&[
        ("thread", thread),
        ("screenshot", json::text(&image())),
    ]));
    assert_eq!(rendered.system, Some(haiku::SYSTEM));
    let task = parse(&rendered.messages[0]);
    let names: Vec<&str> = json::members(&task)
        .unwrap()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(names, ["objective", "language", "history", "downloads"]);
    assert_eq!(
        json::array_of(&task, "history").map(<[Value]>::len),
        Some(HAIKU_RECORDS)
    );
    let (index, blocks) = rendered.extra.clone().unwrap();
    assert!(
        index == 0
            && matches!(&blocks[0], Block::Text(text) if text.starts_with("Current page (untrusted data):\n"))
    );
    assert!(matches!(&blocks[1], Block::Image { media_type, .. } if media_type == "image/png"));
}

#[test]
fn haikus_step_stays_within_the_protocols_measure() {
    let wide = json::object([
        ("step", parse(r#"{"kind":"wait","note":"__NOTE__"}"#.replace("__NOTE__", &"n".repeat(300)).as_str())),
        ("outcome", json::text("ok")),
        ("page", json::object([("url", json::text(&"u".repeat(2_000))), ("title", json::text(&"t".repeat(300)))])),
    ]);
    let lines = MAX_FOLDED_BYTES / record_bytes(&wide);
    let wide_entry = with(
        &with(
            &with(&wide, "text", json::text(&"x".repeat(1_000))),
            "downloads",
            Value::Null,
        ),
        "screenshot",
        Value::Bool(false),
    );
    let thread = with(
        &with(&empty_thread(), "folded", json::array(vec![wide; lines])),
        "steps",
        json::array(vec![wide_entry; 5]),
    );
    let request = admitted(&[("thread", thread)]);
    let measured = protocol::request_text_bytes(
        &request.objective,
        &request.thread,
        &request.observation,
        &request.downloads,
    );
    let user = Rendered {
        system: None,
        ..haiku::prompt(&request)
    };
    assert!(user.text_bytes() <= measured);
}

#[test]
fn a_request_at_the_protocols_measure_stays_within_lunas_priced_bound() {
    // Every part at its bound: the memory, folded lines, and unfolded pages, the page, and a fold's instruction; the
    // rendered prompt's bytes, an image, and the schema stay within what Luna bills at its rate.
    let wait =
        parse(r#"{"step":{"kind":"wait"},"outcome":"ok","page":{"url":"about:blank","title":""}}"#);
    let lines = MAX_FOLDED_BYTES / record_bytes(&wait);
    let text = "é".repeat(MAX_OBSERVATION_BYTES / 2);
    let page = with(
        &with(
            &with(&wait, "text", json::text(&text)),
            "downloads",
            Value::Null,
        ),
        "screenshot",
        Value::Bool(false),
    );
    let mut thread = with(
        &with(&empty_thread(), "folded", json::array(vec![wait; lines])),
        "memory",
        json::text(&"m".repeat(MAX_MEMORY_BYTES)),
    );
    loop {
        let mut steps = json::array_of(&thread, "steps").unwrap().to_vec();
        steps.push(page.clone());
        let grown = with(&thread, "steps", json::array(steps));
        if canonical_thread(&grown).is_none() {
            break;
        }
        thread = grown;
    }
    let request = admitted(&[
        ("objective", json::text(&"o".repeat(MAX_OBJECTIVE_CHARS))),
        ("thread", thread),
        ("observation", json::text(&text)),
    ]);
    let rendered = thread::messages(&request, Some(thread::fold_instruction(1, 1)));
    assert!(rendered.input_bound() + IMAGE_TOKENS <= priced_input_tokens(ProviderId::OpenAi));
    assert!(priced_input_tokens(ProviderId::OpenAi) < 272_000);
}

#[test]
fn a_written_answer_is_bounded_printable_text() {
    assert_eq!(
        thread::plain("  Vaga\u{7} V101\n\tok  ", 100).as_deref(),
        Some("Vaga  V101\n\tok")
    );
    assert_eq!(thread::plain(" \u{7} ", 100), None);
    assert_eq!(thread::plain("ééé", 5).as_deref(), Some("éé"));
}

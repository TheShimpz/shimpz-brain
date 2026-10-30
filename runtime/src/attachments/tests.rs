use super::*;
use crate::conversation::AssistantMessage;
use crate::turn::context::ActionDefinition;

const PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

fn png() -> Vec<u8> {
    BASE64.decode(PNG_BASE64).unwrap()
}

fn content_of(kind: &str) -> Value {
    match kind {
        "text" => json::object([
            ("type", json::text("text")),
            ("text", json::text("Quarterly total: 1000")),
            ("pdf", Value::Bool(true)),
        ]),
        "image" => json::object([
            ("type", json::text("image")),
            ("media_type", json::text("image/png")),
            ("width", json::int(1)),
            ("height", json::int(1)),
            ("base64", json::text(PNG_BASE64)),
            ("sha256", json::text(&sha256_hex(&png()))),
        ]),
        _ => json::object([
            ("type", json::text("opaque")),
            ("reason", json::text("unsupported")),
        ]),
    }
}

/// One attachment as Team sends it.
pub(super) fn team_attachment(kind: &str, file_id: &str) -> Value {
    json::object([
        ("id", json::text(file_id)),
        ("name", json::text(&format!("{kind}.bin"))),
        ("media_type", json::text("application/octet-stream")),
        ("size", json::int(10)),
        ("sha256", json::text(&"f".repeat(64))),
        ("content", content_of(kind)),
    ])
}

/// A file id from a number.
pub(super) fn file_id(index: usize) -> String {
    format!("{index:032x}")
}

/// `value[path...] = replacement`.
pub(super) fn with(mut value: Value, path: &[&str], replacement: Value) -> Value {
    let mut target = &mut value;
    for key in path {
        let Value::Object(members) = target else {
            panic!("not an object");
        };
        if !members.iter().any(|(name, _)| name == key) {
            members.push(((*key).to_owned(), Value::Null));
        }
        target = &mut members.iter_mut().find(|(name, _)| name == key).unwrap().1;
    }
    *target = replacement;
    value
}

/// The admitted attachments of these Team items.
pub(super) fn admitted(items: Vec<Value>) -> Vec<Attachment> {
    admit(&json::array(items)).unwrap()
}

fn refused(raw: &Value) {
    assert_eq!(
        admit(raw),
        Err(BrainError::contract("invalid attachments")),
        "{}",
        json::compact(raw).chars().take(80).collect::<String>()
    );
}

#[test]
fn the_closed_bounded_shapes_are_admitted() {
    let both = admitted(vec![
        team_attachment("text", &"a".repeat(32)),
        team_attachment("image", &"b".repeat(32)),
        team_attachment("opaque", &"c".repeat(32)),
    ]);
    assert_eq!(kinds(&both), ["text", "image", "opaque"]);
    assert_eq!(
        both[1].content,
        Content::Image {
            media_type: "image/png".into(),
            width: 1,
            height: 1,
            base64: PNG_BASE64.into(),
            sha256: sha256_hex(&png()),
        }
    );
    assert!(reads_content(&both));
    assert!(!reads_content(&both[2..]));
    assert_eq!(admit(&json::array(Vec::new())), Ok(Vec::new()));
}

#[test]
fn content_outside_the_contract_is_refused() {
    let image = team_attachment("image", &"a".repeat(32));
    let text = team_attachment("text", &"a".repeat(32));
    for raw in [
        with(
            image.clone(),
            &["content", "sha256"],
            json::text(&"0".repeat(64)),
        ),
        with(image.clone(), &["content", "width"], json::int(1_569)),
        with(
            image.clone(),
            &["content", "media_type"],
            json::text("image/gif"),
        ),
        with(
            image.clone(),
            &["content", "base64"],
            json::text(&"!".repeat(8)),
        ),
        with(
            image.clone(),
            &["content", "base64"],
            json::text(&PNG_BASE64[..PNG_BASE64.len() - 2]),
        ),
        with(image, &["content", "height"], Value::Bool(true)),
        with(text.clone(), &["path"], json::text("secret")),
        with(text.clone(), &["content", "text"], json::text("   ")),
        with(text.clone(), &["content", "pdf"], json::int(1)),
        with(text.clone(), &["content", "extra"], json::int(1)),
        with(
            team_attachment("opaque", &"a".repeat(32)),
            &["content", "reason"],
            json::text("maybe"),
        ),
    ] {
        refused(&json::array(vec![raw]));
    }
}

#[test]
fn metadata_and_message_ceilings_outside_the_contract_are_refused() {
    let text = || team_attachment("text", &"a".repeat(32));
    for raw in [
        json::array(vec![text(), text()]),
        json::array(
            (0..5)
                .map(|index| team_attachment("image", &file_id(index)))
                .collect(),
        ),
        json::array(
            (0..9)
                .map(|index| team_attachment("opaque", &file_id(index)))
                .collect(),
        ),
        json::array(vec![team_attachment("text", &"A".repeat(32))]),
        json::array(vec![with(text(), &["name"], json::text("a\u{7f}b"))]),
        json::array(vec![with(text(), &["name"], json::text(""))]),
        json::array(vec![with(text(), &["name"], json::text(&"é".repeat(128)))]),
        json::array(vec![with(
            text(),
            &["media_type"],
            json::text("Text/plain"),
        )]),
        json::array(vec![with(text(), &["size"], json::int(0))]),
        json::array(vec![with(
            text(),
            &["size"],
            json::int(25 * 1024 * 1024 + 1),
        )]),
        json::array(vec![with(text(), &["sha256"], json::text(&"F".repeat(64)))]),
        json::text("not a list"),
    ] {
        refused(&raw);
    }
    // Five texts of 32,768 characters pass each file's bound and break the message's.
    let long = "x".repeat(32_768);
    let texts = (0..5)
        .map(|index| {
            with(
                team_attachment("text", &file_id(index)),
                &["content", "text"],
                json::text(&long),
            )
        })
        .collect();
    refused(&json::array(texts));
}

/// A text, an image, an opaque file, and a non-PDF text with escapes, as the Python references were made from.
fn four() -> Vec<Attachment> {
    let unicode = with(
        team_attachment("text", &"d".repeat(32)),
        &["content", "text"],
        json::text("Héllo \"東京\"\n"),
    );
    let unicode = with(unicode, &["content", "pdf"], Value::Bool(false));
    admitted(vec![
        team_attachment("text", &"a".repeat(32)),
        team_attachment("image", &"b".repeat(32)),
        team_attachment("opaque", &"c".repeat(32)),
        unicode,
    ])
}

#[test]
fn the_commitment_is_the_python_digest_of_teams_wire_form() {
    let text = || team_attachment("text", &"a".repeat(32));
    // Digests the Python Brain computed for the same attachments.
    assert_eq!(
        commitment(&admitted(vec![text()])),
        "dd50169cd12644e1b5aac679719b7bdd3b0218f29ba2b0c9899ca9505e144ff7"
    );
    assert_eq!(
        commitment(&[]),
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    );
    let all = four();
    assert_eq!(
        commitment(&all),
        "a68ccd8f341520625b7f1ba0c5bad5f6313ec40d10ec8c58eaf38924e6ea8311"
    );
    let changed = with(
        text(),
        &["content", "text"],
        json::text("Quarterly total: 2000"),
    );
    assert_ne!(
        commitment(&admitted(vec![changed])),
        commitment(&admitted(vec![text()]))
    );
}

fn label(id: char, name: &str) -> String {
    let id = id.to_string().repeat(32);
    format!(
        "{FRAMING}\nFile: {{\"id\":\"{id}\",\"media_type\":\"application/octet-stream\",\"name\":\"{name}\",\"size\":10}}"
    )
}

#[test]
fn blocks_quote_each_file_as_the_python_brain_did() {
    let all = four();
    let pdf = " (PDF text only: images, charts, and scanned pages were not read)";
    assert_eq!(
        blocks(&all[0]),
        [Block::Text(format!(
            "{}{pdf}\nText (JSON-quoted): \"Quarterly total: 1000\"",
            label('a', "text.bin")
        ))]
    );
    assert_eq!(
        blocks(&all[1]),
        [
            Block::Text(format!("{}\nImage:", label('b', "image.bin"))),
            Block::Image {
                media_type: "image/png".into(),
                base64: PNG_BASE64.into()
            },
        ]
    );
    let opaque = "This file cannot be read here (unsupported); only its name and type are known.";
    assert_eq!(
        blocks(&all[2]),
        [Block::Text(format!(
            "{}\n{opaque}",
            label('c', "opaque.bin")
        ))]
    );
    let quoted = r#""H\u00e9llo \"\u6771\u4eac\"\n""#;
    assert_eq!(
        blocks(&all[3]),
        [Block::Text(format!(
            "{}\nText (JSON-quoted): {quoted}",
            label('d', "text.bin")
        ))]
    );
}

#[test]
fn estimates_are_the_python_brains() {
    let all = four();
    let estimates: Vec<_> = [0, 2, 3]
        .map(|index| estimated_charge(&all[index]).unwrap())
        .into();
    assert_eq!(estimates, [479, 449, 430]);
    assert_eq!(
        estimated_charge(&all[1]),
        Err(BrainError::contract(
            "an attached image could not be measured"
        ))
    );
}

#[test]
fn the_projection_carries_every_file_in_order_onto_the_turn_message() {
    assert_eq!(projection("turn", &[]), None);
    let both = admitted(vec![
        team_attachment("text", &"a".repeat(32)),
        team_attachment("image", &"b".repeat(32)),
    ]);
    let projected = projection("shimpz-attached-1", &both).unwrap();
    assert_eq!(projected.message_id, "shimpz-attached-1");
    assert_eq!(
        projected.blocks,
        [blocks(&both[0]), blocks(&both[1])].concat()
    );
}

#[test]
fn the_text_fallback_never_undercounts_a_byte_level_tokenizer() {
    // 16,000 characters across scripts: a byte-level BPE such as o200k_base emits at most one token per byte.
    let mixed: String = "Relatório 東京 ⚙ 𝔘 "
        .repeat(1_000)
        .chars()
        .take(16_000)
        .collect();
    let item = with(
        team_attachment("text", &"a".repeat(32)),
        &["content", "text"],
        json::text(&mixed),
    );
    let attachment = &admitted(vec![item])[0];
    let Block::Text(projected) = &blocks(attachment)[0] else {
        panic!("a text block");
    };
    let block = json::object([
        ("type", json::text("text")),
        ("text", json::text(projected)),
    ]);
    let projected_bytes = json::dumps(&json::array(vec![block]), json::SPACED).len();
    let estimate = estimated_charge(attachment).unwrap();
    assert!(estimate >= mixed.len() as u64);
    assert!(estimate <= projected_bytes as u64 + 64);
}

#[test]
fn call_and_turn_ceilings() {
    let over = Err(BrainError::contract(
        "attachments exceed the model token ceilings",
    ));
    assert_eq!(admit_charges(&[9_000]), over);
    assert_eq!(admit_charges(&[8_000, 8_001]), over);
    assert_eq!(admit_charges(&[8_000, 8_000]), Ok(16_000));
    assert_eq!(admit_charges(&[]), Ok(0));
    assert_eq!(admit_call(16_000, 3), Ok(()));
    assert_eq!(
        admit_call(16_000, 4),
        Err("attachments exceed the turn's token ceiling".to_owned())
    );
    assert_eq!(admit_call(0, 1_000), Ok(()));
}

fn model_turn(attempts: u32) -> Message {
    Message::Assistant(AssistantMessage {
        id: format!("r{attempts}"),
        text: "r".into(),
        calls: Vec::new(),
        invalid_calls: Vec::new(),
        refusal: false,
        truncated: false,
        provider: "openai".into(),
        raw: None,
        attempts,
    })
}

fn user(id: &str) -> Message {
    Message::User {
        id: id.into(),
        text: "m".into(),
    }
}

#[test]
fn attempts_already_reserved_bound_the_next_ones() {
    let messages = [model_turn(3), user("turn"), model_turn(2), model_turn(1)];
    // Only the logical turn's own model turns count, each with the attempts it recorded.
    assert_eq!(dispatched(&messages, "turn"), 3);
    assert_eq!(dispatched(&messages, "absent"), 0);
    // Three earlier attempts are already reserved, so only the fourth may dispatch before the 64,000 budget ends.
    let reserved = dispatched(&messages, "turn");
    assert_eq!(admit_call(16_000, reserved), Ok(()));
    assert!(admit_call(16_000, reserved + 1).is_err());
}

#[test]
fn a_turn_that_read_attachments_is_recognized_by_its_id() {
    assert!(read(&user("shimpz-attached-0123")));
    assert!(!read(&user("shimpz-turn-0123")));
    assert!(!read(&model_turn(1)));
}

fn action(id: &str, authorization: bool) -> ActionDefinition {
    ActionDefinition {
        id: id.into(),
        summary: "An Action.".into(),
        input_schema: json::object([("type", json::text("object"))]),
        output_schema: json::object([("type", json::text("object"))]),
        authorization,
        input_files: Vec::new(),
    }
}

#[test]
fn content_offers_only_authorizing_actions() {
    let docs = AssistantDefinition {
        id: "docs".into(),
        genesis: "Docs stores and reads documents.".into(),
        actions: vec![action("read-index", false), action("store", true)],
    };
    let offered = exposed(std::slice::from_ref(&docs), true);
    let ids: Vec<&str> = offered[0]
        .actions
        .iter()
        .map(|action| action.id.as_str())
        .collect();
    assert_eq!(ids, ["store"]);
    assert_eq!((offered[0].id.as_str(), offered.len()), ("docs", 1));
    // An opaque file, or none, keeps every Action.
    assert_eq!(exposed(std::slice::from_ref(&docs), false), [docs]);
}

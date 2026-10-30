use super::*;
use crate::conversation::{AssistantMessage, InvalidCall, ToolCall};

const TOOL: &str = "a_pinger__ping";

fn user(id: &str, text: &str) -> Message {
    Message::User {
        id: id.into(),
        text: text.into(),
    }
}

fn turn(id: &str, text: &str, calls: Vec<ToolCall>) -> AssistantMessage {
    AssistantMessage {
        id: id.into(),
        text: text.into(),
        calls,
        invalid_calls: Vec::new(),
        refusal: false,
        truncated: false,
        provider: "openai".into(),
        raw: None,
        attempts: 1,
    }
}

fn reply(id: &str, text: &str) -> Message {
    Message::Assistant(turn(id, text, Vec::new()))
}

fn calling(id: &str, call_id: &str) -> Message {
    let call = ToolCall {
        id: call_id.into(),
        name: TOOL.into(),
        arguments: json::object([("host", json::text("x"))]),
    };
    Message::Assistant(turn(id, "", vec![call]))
}

fn result(id: &str, call_id: &str, name: &str, content: &str) -> Message {
    Message::Tool(ToolMessage {
        id: id.into(),
        call_id: call_id.into(),
        name: name.into(),
        content: content.into(),
        error: true,
    })
}

fn ids(messages: &[String]) -> Vec<&str> {
    messages.iter().map(String::as_str).collect()
}

fn never(_message: &Message) -> bool {
    false
}

fn limits(exchanges: usize, history: u64, omit_trigger: u64) -> Limits {
    Limits {
        exchanges,
        history,
        omit_trigger,
        ..LIMITS
    }
}

#[test]
fn history_splits_at_user_messages_and_needs_ids() {
    let history = [
        user("h1", "a"),
        calling("a1", "c1"),
        result("t1", "c1", TOOL, "ok"),
        reply("a2", "done"),
        user("h2", "b"),
        reply("a3", "reply"),
    ];
    let sizes: Vec<usize> = exchanges(&history)
        .unwrap()
        .iter()
        .map(|g| g.len())
        .collect();
    assert_eq!(sizes, [4, 2]);
    assert_eq!(exchanges(&[user("", "a")]), Err(FitError::State));
    assert_eq!(exchanges(&[reply("a1", "orphan")]), Err(FitError::State));
    assert_eq!(
        BrainError::from(FitError::State).answer(),
        (503, "Brain runtime state operation failed")
    );
}

#[test]
fn a_bridge_is_one_exchange_with_the_turn_it_precedes() {
    let bridge = format!("{BRIDGE_ID_PREFIX}1");
    let history = [
        user(&bridge, "quoted"),
        user("h1", "a"),
        reply("a1", "r1"),
        user("h2", "b"),
        reply("a2", "r2"),
    ];
    let sizes: Vec<usize> = exchanges(&history)
        .unwrap()
        .iter()
        .map(|g| g.len())
        .collect();
    assert_eq!(sizes, [3, 2]);
    assert_eq!(history_to_drop(&history, 0, 1, &never), Ok(Vec::new()));
    let dropped = drop_within(limits(1, HISTORY_BUDGET_TOKENS, 0), &history, 0, 1, &never);
    assert_eq!(ids(&dropped.unwrap()), [bridge.as_str(), "h1", "a1"]);
    // A failed bridged turn, or one that read attachments, is forgotten whole, its bridge included.
    let failed = [user(&bridge, "quoted"), user("h1", "a")];
    let dropped = history_to_drop(&failed, 0, 1, &never).unwrap();
    assert_eq!(ids(&dropped), [bridge.as_str(), "h1"]);
    let dropped = history_to_drop(&history, 0, 1, &|message| message.id() == "h1").unwrap();
    assert_eq!(ids(&dropped), [bridge.as_str(), "h1", "a1"]);
}

fn five_exchanges() -> Vec<Message> {
    (0..5)
        .flat_map(|index| {
            [
                user(&format!("h{index}"), &format!("q{index}")),
                reply(&format!("a{index}"), &format!("r{index}")),
            ]
        })
        .collect()
}

#[test]
fn newest_whole_exchanges_are_kept_within_count_and_budget() {
    let history = five_exchanges();
    let dropped = drop_within(limits(2, HISTORY_BUDGET_TOKENS, 0), &history, 0, 1, &never);
    assert_eq!(ids(&dropped.unwrap()), ["h0", "a0", "h1", "a1", "h2", "a2"]);
    let exchange = history[8..].iter().map(message_tokens).sum();
    let dropped = drop_within(limits(24, exchange, 0), &history, 0, 1, &never).unwrap();
    assert_eq!(dropped.len(), 8);
    assert_eq!(history_to_drop(&[], 0, 1, &never), Ok(Vec::new()));
}

#[test]
fn unfinished_exchanges_are_always_dropped() {
    let unfinished = [
        user("h0", "q0"),
        reply("a0", "r0"),
        user("hf", "failed"),
        user("h1", "q1"),
        calling("at", "c1"),
        user("h2", "q2"),
        reply("a2", "r2"),
    ];
    let dropped = history_to_drop(&unfinished, 0, 1, &never).unwrap();
    assert_eq!(ids(&dropped), ["hf", "h1", "at"]);
}

#[test]
fn a_reply_the_provider_cut_short_never_completes_an_exchange() {
    let mut cut = turn("a1", "partial answer", Vec::new());
    cut.truncated = true;
    let cut = Message::Assistant(cut);
    assert!(truncated(&cut));
    assert_eq!(final_reply(&cut), "");
    assert!(!completed(&[user("h1", "q"), cut.clone()]));
    let history = [user("h0", "q0"), reply("a0", "r0"), user("h1", "q"), cut];
    let dropped = history_to_drop(&history, 0, 1, &never).unwrap();
    assert_eq!(ids(&dropped), ["h1", "a1"]);
    let finished = reply("a1", "  answer\n");
    assert!(!truncated(&finished));
    assert_eq!(final_reply(&finished), "answer");
}

#[test]
fn only_a_plain_nonblank_model_reply_ends_a_turn() {
    let mut invalid = turn("a1", "text", Vec::new());
    invalid.invalid_calls.push(InvalidCall {
        id: "c1".into(),
        name: TOOL.into(),
        arguments: "{".into(),
    });
    assert_eq!(final_reply(&Message::Assistant(invalid)), "");
    assert_eq!(final_reply(&calling("a1", "c1")), "");
    assert_eq!(final_reply(&reply("a1", " \t\n")), "");
    assert_eq!(final_reply(&user("h1", "text")), "");
    assert!(!completed(&[]));
}

fn action_exchange(index: usize, content: &str, name: &str) -> Vec<Message> {
    vec![
        user(&format!("h{index}"), &format!("q{index}")),
        calling(&format!("t{index}"), &format!("c{index}")),
        result(&format!("m{index}"), &format!("c{index}"), name, content),
        reply(&format!("a{index}"), &format!("done {index}")),
    ]
}

fn omission_history() -> Vec<Message> {
    let large = "r".repeat(3_000);
    let mut history = Vec::new();
    for index in 0..5 {
        let name = if index == 1 { "other" } else { TOOL };
        history.extend(action_exchange(index, &large, name));
    }
    history.push(user("hf", "failed"));
    history.extend(action_exchange(5, &large, TOOL).into_iter().skip(1).take(2));
    history
}

fn action(message: &ToolMessage) -> bool {
    message.name == TOOL
}

#[test]
fn older_action_results_give_way_to_placeholders_only_at_the_trigger() {
    let history = omission_history();
    let forgotten = |message: &Message| message.id() == "h2";
    let omitted = omit_within(limits(24, 0, 1), &history, &action, &forgotten).unwrap();
    // Exchanges 3 and 4 are the newest completed ones; 2 is forgotten whole, 1 holds no Action result, and the
    // unfinished one is dropped whole instead.
    assert_eq!(
        omitted,
        [ToolMessage {
            id: "m0".into(),
            call_id: "c0".into(),
            name: TOOL.into(),
            content: OMITTED_RESULT.into(),
            error: true,
        }]
    );
    // Below the trigger nothing changes.
    assert_eq!(results_to_omit(&history, &action, &never), Ok(Vec::new()));
    let saving = message_tokens(&history[2]) - message_tokens(&Message::Tool(omitted[0].clone()));
    let below = omit_within(limits(24, 0, saving + 1), &history, &action, &forgotten);
    assert_eq!(below, Ok(Vec::new()));
}

#[test]
fn an_omitted_or_short_result_is_never_omitted() {
    let large = "r".repeat(3_000);
    let settled: Vec<Message> = [
        action_exchange(0, OMITTED_RESULT, TOOL),
        action_exchange(3, &large, TOOL),
        action_exchange(4, &large, TOOL),
    ]
    .concat();
    assert_eq!(
        omit_within(limits(24, 0, 1), &settled, &action, &never),
        Ok(Vec::new())
    );
    // Only the tokens placeholders save count toward the trigger: many short results never grow the history.
    let short: Vec<Message> = (0..40)
        .flat_map(|index| action_exchange(index, "ok", TOOL))
        .collect();
    assert_eq!(
        omit_within(limits(24, 0, 1), &short, &action, &never),
        Ok(Vec::new())
    );
}

#[test]
fn fixed_prompt_and_current_message_alone_can_exceed_the_window() {
    assert_eq!(
        history_to_drop(&[], MODEL_WINDOW_TOKENS, 1, &never),
        Err(FitError::Window)
    );
    // A window error is reported before the history is even decomposed.
    assert_eq!(
        history_to_drop(&[reply("a1", "orphan")], MODEL_WINDOW_TOKENS, 1, &never),
        Err(FitError::Window)
    );
    assert_eq!(
        ensure_window(MODEL_WINDOW_TOKENS - OUTPUT_RESERVE_TOKENS - 1, 1),
        Ok(())
    );
    assert_eq!(
        ensure_window(MODEL_WINDOW_TOKENS - OUTPUT_RESERVE_TOKENS, 1),
        Err(FitError::Window)
    );
    assert_eq!(
        BrainError::from(FitError::Window).answer(),
        (400, "conversation context exceeds the model window")
    );
    // The JSON encoding counts: 298 characters plus two quotes are 300 bytes, 100 estimated tokens.
    assert_eq!(estimated_tokens(&json::text(&"x".repeat(298))), 100);
    assert_eq!(estimated_tokens(&json::text(&"x".repeat(299))), 101);
}

#[test]
fn kept_history_stops_at_the_window_and_counts_whole_exchanges() {
    let history = five_exchanges();
    let newest: u64 = history[8..].iter().map(message_tokens).sum();
    let window = MODEL_WINDOW_TOKENS - OUTPUT_RESERVE_TOKENS - 1 - newest;
    let dropped = history_to_drop(&history, window, 1, &never).unwrap();
    assert_eq!(dropped.len(), 8);
    let dropped = history_to_drop(&history, window + 1, 1, &never).unwrap();
    assert_eq!(dropped.len(), 10);
}

#[test]
fn estimates_grow_with_what_a_message_carries() {
    let plain = reply("a1", "ok");
    let calls = calling("a1", "c1");
    let mut replayed = turn("a1", "ok", Vec::new());
    replayed.raw = Some(json::array(vec![json::object([(
        "encrypted_content",
        json::text(&"e".repeat(900)),
    )])]));
    assert!(message_tokens(&calls) > message_tokens(&plain));
    assert!(message_tokens(&Message::Assistant(replayed)) >= 300);
    assert_eq!(
        fixed_tokens("x".repeat(298).as_str(), &[json::text(&"y".repeat(298))]),
        200
    );
}

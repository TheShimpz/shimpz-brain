use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

use super::super::tests::{admitted, file_id, team_attachment, with};
use super::super::{COUNT_DEADLINE, MAX_ATTACHMENTS, admit_call, admit_charges};
use super::*;
use crate::cancel::CancelToken;

fn texts(count: usize) -> Vec<Attachment> {
    admitted(
        (0..count)
            .map(|index| team_attachment("text", &file_id(index)))
            .collect(),
    )
}

fn estimates(attachments: &[Attachment]) -> Vec<u64> {
    attachments
        .iter()
        .map(|item| estimated_charge(item).unwrap())
        .collect()
}

/// A count blocked on the provider until its scope is cancelled, as a socket the transport closes on cancellation.
fn hanging(scope: &Scope) -> Result<u64, CallError> {
    let started = Instant::now();
    while !scope.stopped() && started.elapsed() < Duration::from_secs(30) {
        thread::sleep(Duration::from_millis(5));
    }
    Err(CallError::Cancelled)
}

#[test]
fn counting_stops_at_one_overall_deadline_and_ends_the_counts_in_flight() {
    let attachments = texts(3);
    let calls = AtomicUsize::new(0);
    let count = |scope: &Scope, _blocks: &[Block], _timeout: Duration| {
        calls.fetch_add(1, Ordering::SeqCst);
        hanging(scope)
    };
    let slots = Slots::new(MAX_COUNT_WORKERS);
    let started = Instant::now();
    let charged = charges(
        &slots,
        &Scope::default(),
        Duration::from_millis(300),
        &attachments,
        &count,
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    // The counting slots' files started at once and ended with the one deadline; the third never got a slot.
    assert_eq!(calls.load(Ordering::SeqCst), MAX_COUNT_WORKERS);
    assert_eq!(charged, Ok(estimates(&attachments)));
    // Every slot came back once the workers were joined.
    assert!(slots.take(&Scope::default(), Instant::now()).is_ok());
    assert!(slots.take(&Scope::default(), Instant::now()).is_ok());
}

#[test]
fn files_are_counted_together_so_the_deadline_bounds_the_slowest_not_their_sum() {
    let attachments = texts(3);
    let waiting = AtomicUsize::new(0);
    // The first two counts wait for each other: counted one after another, neither could finish.
    let count = |_scope: &Scope, blocks: &[Block], _timeout: Duration| {
        let Block::Text(text) = &blocks[0] else {
            return Err(CallError::Request);
        };
        if text.contains(&file_id(2)) {
            return Ok(7);
        }
        waiting.fetch_add(1, Ordering::SeqCst);
        let started = Instant::now();
        while waiting.load(Ordering::SeqCst) < MAX_COUNT_WORKERS {
            if started.elapsed() > Duration::from_secs(5) {
                return Err(CallError::Request);
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(7)
    };
    let slots = Slots::new(MAX_COUNT_WORKERS);
    let charged = charges(
        &slots,
        &Scope::default(),
        COUNT_DEADLINE,
        &attachments,
        &count,
    );
    assert_eq!(charged, Ok(vec![7, 7, 7]));
}

#[test]
fn a_count_waiting_for_a_slot_stops_with_its_turn_or_gives_up_at_the_deadline() {
    let calls = AtomicUsize::new(0);
    let count = |_scope: &Scope, _blocks: &[Block], _timeout: Duration| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(7)
    };
    let slots = Slots::new(MAX_COUNT_WORKERS);
    let open = Scope::default();
    let far = Instant::now() + Duration::from_secs(30);
    let held: Vec<_> = (0..MAX_COUNT_WORKERS)
        .map(|_| slots.take(&open, far).ok().unwrap())
        .collect();
    // Every slot is held: past its deadline the count gives up and the file is estimated.
    let soon = Instant::now() + Duration::from_millis(100);
    assert_eq!(
        count_before(&slots, &open, soon, &[], &count),
        Counted::Unavailable
    );
    // A stopped turn ends the wait at once instead of at the deadline.
    let stopped = Scope::new(CancelToken::new());
    stopped.cancel.cancel();
    let started = Instant::now();
    assert_eq!(
        count_before(&slots, &stopped, far, &[], &count),
        Counted::Cancelled
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    drop(held);
    // A slot taken with no time left counts nothing, and gives the slot back.
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert_eq!(
        count_before(&slots, &open, past, &[], &count),
        Counted::Unavailable
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let ahead = Instant::now() + Duration::from_secs(5);
    assert_eq!(
        count_before(&slots, &open, ahead, &[], &count),
        Counted::Value(7)
    );
}

#[test]
fn a_count_failure_falls_back_to_a_conservative_estimate() {
    let attachments = texts(1);
    let slots = Slots::new(MAX_COUNT_WORKERS);
    let failing = |_: &Scope, _: &[Block], _: Duration| Err(CallError::Request);
    let charged = charges(
        &slots,
        &Scope::default(),
        COUNT_DEADLINE,
        &attachments,
        &failing,
    );
    assert_eq!(charged, Ok(estimates(&attachments)));
    let counted = |_: &Scope, _: &[Block], _: Duration| Ok(42);
    let charged = charges(
        &slots,
        &Scope::default(),
        COUNT_DEADLINE,
        &attachments,
        &counted,
    );
    assert_eq!(charged, Ok(vec![42]));
    let image = admitted(vec![team_attachment("image", &file_id(0))]);
    assert_eq!(
        charges(&slots, &Scope::default(), COUNT_DEADLINE, &image, &failing),
        Err(BrainError::contract(
            "an attached image could not be measured"
        ))
    );
}

#[test]
fn opaque_files_are_estimated_without_any_provider_count() {
    // The largest metadata the contract admits: every quote in the name is escaped twice in the estimate.
    let widest = admitted(
        (0..MAX_ATTACHMENTS)
            .map(|index| {
                let item = team_attachment("opaque", &file_id(index));
                let item = with(item, &["name"], crate::json::text(&"\"".repeat(255)));
                let media_type = format!("application/{}", "x".repeat(115));
                let item = with(item, &["media_type"], crate::json::text(&media_type));
                with(item, &["size"], crate::json::int(25 * 1024 * 1024))
            })
            .collect(),
    );
    let slots = Slots::new(MAX_COUNT_WORKERS);
    let never = |_: &Scope, _: &[Block], _: Duration| -> Result<u64, CallError> {
        panic!("an opaque file was counted")
    };
    let charged = charges(&slots, &Scope::default(), COUNT_DEADLINE, &widest, &never).unwrap();
    assert_eq!(charged, estimates(&widest));
    let total = admit_charges(&charged).unwrap();
    assert_eq!(admit_call(total, 3), Ok(()));

    let counted = Mutex::new(Vec::new());
    let mixed = admitted(vec![
        team_attachment("opaque", &file_id(0)),
        team_attachment("text", &file_id(1)),
    ]);
    let record = |_: &Scope, blocks: &[Block], _: Duration| {
        counted.lock().unwrap().push(blocks.to_vec());
        Ok(42)
    };
    let charged = charges(&slots, &Scope::default(), COUNT_DEADLINE, &mixed, &record);
    assert_eq!(charged, Ok(vec![estimated_charge(&mixed[0]).unwrap(), 42]));
    assert_eq!(counted.into_inner().unwrap(), [blocks(&mixed[1])]);
}

#[test]
fn a_stopped_count_ends_the_turn_before_an_uncounted_image_could_refuse_it() {
    let attachments = admitted(vec![
        team_attachment("image", &file_id(0)),
        team_attachment("text", &file_id(1)),
    ]);
    let count = |_: &Scope, blocks: &[Block], _: Duration| {
        if blocks
            .iter()
            .any(|block| matches!(block, Block::Image { .. }))
        {
            Err(CallError::Request)
        } else {
            Err(CallError::Cancelled)
        }
    };
    let slots = Slots::new(MAX_COUNT_WORKERS);
    assert_eq!(
        charges(
            &slots,
            &Scope::default(),
            COUNT_DEADLINE,
            &attachments,
            &count
        ),
        Err(BrainError::Cancelled("Chat turn cancelled"))
    );
}

#[test]
fn stopping_the_turn_reaches_a_count_blocked_on_the_provider() {
    let attachments = texts(1);
    let turn = Scope::new(CancelToken::new());
    let (started_sender, started) = mpsc::channel();
    let started_sender = Mutex::new(started_sender);
    let count = |scope: &Scope, _: &[Block], _: Duration| {
        started_sender.lock().unwrap().send(()).unwrap();
        hanging(scope)
    };
    let slots = Slots::new(MAX_COUNT_WORKERS);
    let begun = Instant::now();
    let charged = thread::scope(|threads| {
        let charging =
            threads.spawn(|| charges(&slots, &turn, COUNT_DEADLINE, &attachments, &count));
        started.recv_timeout(Duration::from_secs(5)).unwrap();
        turn.cancel.cancel();
        charging.join().unwrap()
    });
    assert_eq!(charged, Err(BrainError::Cancelled("Chat turn cancelled")));
    assert!(begun.elapsed() < COUNT_DEADLINE);
}

#[test]
fn a_stopped_turn_never_starts_counting() {
    let attachments = texts(1);
    let turn = Scope::new(CancelToken::new());
    turn.cancel.cancel();
    let never = |_: &Scope, _: &[Block], _: Duration| -> Result<u64, CallError> {
        panic!("a stopped turn counted")
    };
    let slots = Slots::new(MAX_COUNT_WORKERS);
    assert_eq!(
        charges(&slots, &turn, COUNT_DEADLINE, &attachments, &never),
        Err(BrainError::Cancelled("Chat turn cancelled"))
    );
}

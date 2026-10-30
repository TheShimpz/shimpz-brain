//! Provider counting of attachment tokens within one deadline (`charges`, `_count_before`): files are counted
//! together on scoped worker threads, each count holds one of the process's counting slots, and at the deadline the
//! nested scope is cancelled so a slow count's I/O ends and its worker has finished before charging returns.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use super::{Attachment, Content, MAX_COUNT_WORKERS, blocks, estimated_charge};
use crate::conversation::Block;
use crate::errors::BrainError;
use crate::provider::{CallError, Scope};

/// One count of one attachment's blocks within a timeout, on a scope.
pub(super) type Count<'a> =
    &'a (dyn Fn(&Scope, &[Block], Duration) -> Result<u64, CallError> + Sync);

/// How often a count waiting for a slot looks at its scope.
const SLOT_POLL: Duration = Duration::from_millis(50);

/// The counting slots of the process.
pub(super) static SLOTS: Slots = Slots::new(MAX_COUNT_WORKERS);

fn cancelled() -> BrainError {
    BrainError::Cancelled("Chat turn cancelled")
}

/// A bounded number of counts in flight.
pub(super) struct Slots {
    free: Mutex<usize>,
    freed: Condvar,
}

/// One held slot, given back on drop.
pub(super) struct Slot<'a>(&'a Slots);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        self.0.freed.notify_one();
    }
}

/// What one count gave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Counted {
    /// The provider's count.
    Value(u64),
    /// No count in time, or the provider did not count: the estimate stands in.
    Unavailable,
    /// The turn was stopped.
    Cancelled,
}

impl Slots {
    /// `count` slots.
    pub(super) const fn new(count: usize) -> Self {
        Self {
            free: Mutex::new(count),
            freed: Condvar::new(),
        }
    }

    /// A slot, waiting in short polls of `scope` until `deadline`.
    pub(super) fn take(&self, scope: &Scope, deadline: Instant) -> Result<Slot<'_>, Counted> {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if *free > 0 {
                *free -= 1;
                return Ok(Slot(self));
            }
            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(SLOT_POLL);
            free = self
                .freed
                .wait_timeout(free, wait)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
            if *free > 0 {
                continue;
            }
            if scope.stopped() {
                return Err(Counted::Cancelled);
            }
            if Instant::now() >= deadline {
                return Err(Counted::Unavailable);
            }
        }
    }
}

/// One count with whatever time is left once it holds a slot; none is left past the deadline (`_count_before`).
pub(super) fn count_before(
    slots: &Slots,
    scope: &Scope,
    deadline: Instant,
    blocks: &[Block],
    count: Count<'_>,
) -> Counted {
    let _slot = match slots.take(scope, deadline) {
        Ok(slot) => slot,
        Err(counted) => return counted,
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Counted::Unavailable;
    }
    if scope.stopped() {
        return Counted::Cancelled;
    }
    match count(scope, blocks, remaining) {
        Ok(value) => Counted::Value(value),
        Err(CallError::Cancelled) => Counted::Cancelled,
        Err(_) => Counted::Unavailable,
    }
}

/// The shared state of one charging's workers.
struct Job<'a> {
    slots: &'a Slots,
    scope: Scope,
    deadline: Instant,
    attachments: &'a [Attachment],
    countable: Vec<usize>,
    next: AtomicUsize,
    count: Count<'a>,
}

impl Job<'_> {
    /// Counts the next uncounted file until none is left or nobody listens.
    fn work(&self, results: &Sender<(usize, Counted)>) {
        while let Some(&index) = self.countable.get(self.next.fetch_add(1, Ordering::SeqCst)) {
            let content = blocks(&self.attachments[index]);
            let counted =
                count_before(self.slots, &self.scope, self.deadline, &content, self.count);
            if results.send((index, counted)).is_err() {
                return;
            }
        }
    }
}

/// The counts that arrived before the deadline, by attachment index.
///
/// # Errors
///
/// Cancelled when a count ended because the turn was stopped.
fn collect(
    results: &Receiver<(usize, Counted)>,
    deadline: Instant,
    total: usize,
    expected: usize,
) -> Result<Vec<Option<u64>>, BrainError> {
    let mut values = vec![None; total];
    for _ in 0..expected {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match results.recv_timeout(remaining) {
            Ok((index, Counted::Value(value))) => values[index] = Some(value),
            Ok((_, Counted::Unavailable)) => {}
            Ok((_, Counted::Cancelled)) => return Err(cancelled()),
            Err(_) => break,
        }
    }
    Ok(values)
}

/// Each attachment's token charge, counted within `within` of the start or estimated (`charges`).
///
/// # Errors
///
/// Cancelled when the turn was stopped, before any estimate could refuse an image; otherwise the estimate's
/// refusal of an uncounted image.
pub(super) fn charges(
    slots: &Slots,
    scope: &Scope,
    within: Duration,
    attachments: &[Attachment],
    count: Count<'_>,
) -> Result<Vec<u64>, BrainError> {
    if scope.stopped() {
        return Err(cancelled());
    }
    let job = Job {
        slots,
        scope: scope.child(),
        deadline: Instant::now() + within,
        attachments,
        countable: (0..attachments.len())
            .filter(|index| !matches!(attachments[*index].content, Content::Opaque { .. }))
            .collect(),
        next: AtomicUsize::new(0),
        count,
    };
    let counted = thread::scope(|threads| {
        let (sender, receiver) = mpsc::channel();
        // The files are counted together, so the deadline bounds the slowest count rather than their sum.
        for _ in 0..job.countable.len().min(MAX_COUNT_WORKERS) {
            let sender = sender.clone();
            let job = &job;
            threads.spawn(move || job.work(&sender));
        }
        drop(sender);
        let counted = collect(
            &receiver,
            job.deadline,
            attachments.len(),
            job.countable.len(),
        );
        // Leaving cancels the nested scope: a count still in flight ends its I/O, and its worker is joined here.
        job.scope.cancel.cancel();
        counted
    })?;
    if scope.stopped() {
        return Err(cancelled());
    }
    attachments
        .iter()
        .zip(counted)
        .map(|(attachment, value)| value.map_or_else(|| estimated_charge(attachment), Ok))
        .collect()
}

#[cfg(test)]
mod tests;

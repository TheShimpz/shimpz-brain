//! One request's cancellation: tripped when Team closes the request, observed by every provider call of its work.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

/// A cancellation flag that wakes every waiter at once. Cheap to clone; clones share the flag.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<Inner>);

#[derive(Debug, Default)]
struct Inner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl CancelToken {
    /// A token that is not cancelled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancels the work; every current and later waiter returns at once.
    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::SeqCst);
        self.0.notify.notify_waiters();
    }

    /// Whether the work was cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::SeqCst)
    }

    /// Resolves once the work is cancelled.
    pub async fn cancelled(&self) {
        loop {
            let notified = self.0.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Cancels its token when dropped unless disarmed: held by a request handler, whose future hyper drops when Team
/// closes the connection.
#[derive(Debug)]
pub struct CancelOnDrop {
    token: CancelToken,
    armed: bool,
}

impl CancelOnDrop {
    /// Guards `token`.
    #[must_use]
    pub fn new(token: CancelToken) -> Self {
        Self { token, armed: true }
    }

    /// The request completed: dropping the guard no longer cancels.
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.token.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancelling_wakes_waiters_and_guards_cancel_unless_disarmed() {
        let token = CancelToken::new();
        let waiter = tokio::spawn({
            let token = token.clone();
            async move { token.cancelled().await }
        });
        drop(CancelOnDrop::new(token.clone()));
        waiter.await.unwrap();
        assert!(token.is_cancelled());
        let kept = CancelToken::new();
        let mut guard = CancelOnDrop::new(kept.clone());
        guard.disarm();
        drop(guard);
        assert!(!kept.is_cancelled());
    }
}

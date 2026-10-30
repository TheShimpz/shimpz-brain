//! Chat turns over the Holdpoint worker and the thread store (`AgentRuntime.start`, `resume`, `delete_thread`,
//! and the purpose lookup). Every operation runs on a blocking thread under its thread's lock, which it holds until
//! its run settled: one drive per run in this process, and no other process (the state lock).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use holdpoint::{Driven, Error, Event, Outcome, RunOptions, StepKind, Wait, Worker};

use crate::conversation::Message;
use crate::decisions::purpose::Pending;
use crate::errors::BrainError;
use crate::json::Value;
use crate::provider::{Providers, Scope};
use crate::turn::context::TurnContext;
use crate::turn::definition::Failed;
use crate::turn::host::{BrainHost, TurnSession};
use crate::turn::run::{self, TeamRequest, TurnInput};
use crate::turn::threads::{Delivery, Thread, Threads};

mod prepare;

/// The 409 text of a cancelled chat turn.
pub const CANCELLED: &str = "Chat turn cancelled";
const NO_PENDING: &str = "conversation has no pending Action request";
const POLL: Duration = Duration::from_millis(50);

/// The per-thread locks, reclaimed once no operation holds them.
#[derive(Debug, Default)]
struct Locks(Mutex<HashMap<String, Weak<Mutex<()>>>>);

impl Locks {
    fn of(&self, thread_id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        locks.retain(|_, lock| lock.strong_count() > 0);
        if let Some(lock) = locks.get(thread_id).and_then(Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(Mutex::new(()));
        locks.insert(thread_id.to_owned(), Arc::downgrade(&lock));
        lock
    }
}

/// The chat turn service.
pub struct TurnService {
    worker: Arc<Worker>,
    host: Arc<BrainHost>,
    threads: Threads,
    providers: Providers,
    locks: Locks,
    lease: Duration,
}

impl std::fmt::Debug for TurnService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TurnService")
            .finish_non_exhaustive()
    }
}

fn state<T>(_error: T) -> BrainError {
    BrainError::State
}

/// How a drive left a turn.
enum Settled {
    Pending,
    Ended(Outcome),
}

impl TurnService {
    /// The service over its worker, host, thread store, and providers; `lease` is the worker's lease.
    #[must_use]
    pub fn new(
        worker: Arc<Worker>,
        host: Arc<BrainHost>,
        threads: Threads,
        providers: Providers,
        lease: Duration,
    ) -> Self {
        Self {
            worker,
            host,
            threads,
            providers,
            locks: Locks::default(),
            lease,
        }
    }

    /// The providers.
    #[must_use]
    pub fn providers(&self) -> &Providers {
        &self.providers
    }

    /// The digest binding a thread's history to the exact Assistant contract (`_assistant_scope`).
    #[must_use]
    pub fn scope_digest(context: &TurnContext) -> String {
        prepare::scope_digest(context)
    }

    /// Deletes `run_id`, waiting out a live lease of a session that already ended (a crash).
    fn delete_run(&self, run_id: &str) -> Result<(), BrainError> {
        let deadline = Instant::now() + self.lease + Duration::from_secs(1);
        loop {
            match self.worker.delete(run_id) {
                Ok(()) => return Ok(()),
                Err(Error::Busy(_)) if Instant::now() < deadline => std::thread::sleep(POLL),
                Err(_) => return Err(BrainError::State),
            }
        }
    }

    /// Settles a thread's open run left by an earlier request (the two-store reconciliation): deletes it and keeps
    /// the history, which holds its exchange only if `committed_run` names it.
    fn reconcile(&self, thread_id: &str, thread: &mut Thread) -> Result<(), BrainError> {
        if let Some(run_id) = thread.open_run.clone() {
            self.delete_run(&run_id)?;
            crate::failpoint::hit("reconcile.deleted");
            thread.open_run = None;
            self.threads.save(thread_id, thread)?;
        }
        Ok(())
    }

    /// The Team requests `run_id` waits on (none for a run the ledger does not hold). A ledger failure or a
    /// recorded request that does not decode is a state error, never "nothing pending": a caller must not delete a
    /// run it could not read.
    fn pending(&self, run_id: &str) -> Result<Vec<TeamRequest>, BrainError> {
        let requests = self.worker.pending(run_id).map_err(state)?;
        requests
            .iter()
            .map(|request| {
                TeamRequest::from_json(&request.name, &request.request).ok_or(BrainError::State)
            })
            .collect()
    }

    fn input(&self, run_id: &str) -> Result<TurnInput, BrainError> {
        let history = self.worker.history(run_id).map_err(state)?;
        match history.first() {
            Some(Event::RunStarted { input, .. }) => {
                TurnInput::from_json(input).ok_or(BrainError::State)
            }
            _ => Err(BrainError::State),
        }
    }

    /// Drives `run_id` until it waits on Team or ends, re-driving timed waits within the request and cancelling the
    /// run once `scope` is cancelled.
    fn drive(&self, run_id: &str, scope: &Scope) -> Result<Settled, BrainError> {
        let done = AtomicBool::new(false);
        std::thread::scope(|threads| {
            threads.spawn(|| {
                while !done.load(Ordering::SeqCst) {
                    if scope.cancel.is_cancelled() {
                        let _ = self.worker.cancel(run_id);
                        return;
                    }
                    std::thread::sleep(POLL);
                }
            });
            let driven = self.drive_until_settled(run_id);
            done.store(true, Ordering::SeqCst);
            driven
        })
    }

    fn drive_until_settled(&self, run_id: &str) -> Result<Settled, BrainError> {
        let busy_deadline = Instant::now() + self.lease + Duration::from_secs(1);
        loop {
            match self.worker.drive(run_id) {
                Ok(Driven::Ended(outcome)) => return Ok(Settled::Ended(outcome)),
                Ok(Driven::Suspended(Wait::Deferred)) => return Ok(Settled::Pending),
                Ok(Driven::Suspended(Wait::Until(at) | Wait::Executor(at))) => sleep_until(at),
                Err(Error::Busy(_)) if Instant::now() < busy_deadline => std::thread::sleep(POLL),
                Ok(Driven::Suspended(Wait::Authority | Wait::Shutdown)) | Err(_) => {
                    return Err(BrainError::State);
                }
            }
        }
    }

    /// Runs `run_id` for one request: installs its session, drives, and settles the outcome into the thread.
    /// `answered` names the interrupts the request answered (none for a start).
    fn run(
        &self,
        thread_id: &str,
        thread: &mut Thread,
        (run_id, answered): (&str, Vec<String>),
        session: TurnSession,
    ) -> Result<Value, BrainError> {
        let scope = session.scope.clone();
        self.host.install(run_id, session);
        let driven = self.drive(run_id, &scope);
        self.host.remove(run_id);
        match driven? {
            Settled::Pending => self.suspended(run_id),
            Settled::Ended(outcome) => self.ended(thread_id, thread, (run_id, answered), &outcome),
        }
    }

    fn suspended(&self, run_id: &str) -> Result<Value, BrainError> {
        let requests = self.pending(run_id)?;
        if requests.is_empty() {
            return Err(BrainError::State);
        }
        Ok(prepare::pending_value(&requests))
    }

    fn ended(
        &self,
        thread_id: &str,
        thread: &mut Thread,
        (run_id, answered): (&str, Vec<String>),
        outcome: &Outcome,
    ) -> Result<Value, BrainError> {
        let completed = match outcome {
            Outcome::Done { result } => run::outcome_from(result).ok_or(BrainError::State)?,
            Outcome::Cancelled { .. } => {
                self.reconcile(thread_id, thread)?;
                return Err(BrainError::Cancelled(CANCELLED));
            }
            _ => {
                self.reconcile(thread_id, thread)?;
                return Err(BrainError::State);
            }
        };
        if let Some(failed) = Failed::from_value(&completed) {
            self.reconcile(thread_id, thread)?;
            return Err(match failed {
                Failed::Contract(detail) => BrainError::Contract(detail),
                Failed::Provider => BrainError::Provider,
                Failed::Cancelled => BrainError::Cancelled(CANCELLED),
            });
        }
        let messages = completed
            .get("messages")
            .and_then(crate::conversation::from_value)
            .ok_or(BrainError::State)?;
        let turn = completed.get("turn").cloned().ok_or(BrainError::State)?;
        thread.history = messages;
        thread.committed_run = Some(run_id.to_owned());
        // The turn as Team receives it is kept with its exchange: a resume Team repeats after losing it receives it
        // again (ADR-0125).
        let delivery = Delivery {
            answered,
            turn: turn.clone(),
        };
        self.threads.commit(thread_id, thread, &delivery)?;
        crate::failpoint::hit("commit.exchange");
        self.reconcile(thread_id, thread)?;
        Ok(turn)
    }

    /// `AgentRuntime.start`: one new logical turn.
    ///
    /// # Errors
    ///
    /// [`BrainError`] as the Python runtime answered.
    pub fn start(
        &self,
        request: crate::server::turn_input::StartRequest,
        scope: &Scope,
    ) -> Result<Value, BrainError> {
        let lock = self.locks.of(&request.turn.context.thread_id);
        let _held = lock.lock().unwrap_or_else(PoisonError::into_inner);
        let prepared = prepare::start(self, request, scope)?;
        let prepare::Prepared {
            thread_id,
            mut thread,
            run_id,
            input,
            session,
        } = prepared;
        thread.open_run = Some(run_id.clone());
        self.threads.save(&thread_id, &thread)?;
        crate::failpoint::hit("start.reference");
        self.worker
            .submit(
                run::DEFINITION,
                &run_id,
                &input.to_json(),
                &RunOptions::default(),
            )
            .map_err(state)?;
        crate::failpoint::hit("start.submitted");
        self.run(&thread_id, &mut thread, (&run_id, Vec::new()), session)
    }

    /// `AgentRuntime.resume`: Team's results for the pending requests of the thread's open run.
    ///
    /// # Errors
    ///
    /// [`BrainError`] as the Python runtime answered.
    pub fn resume(
        &self,
        request: &crate::server::turn_input::ResumeRequest,
        scope: &Scope,
    ) -> Result<Value, BrainError> {
        let thread_id = request.turn.context.thread_id.clone();
        let lock = self.locks.of(&thread_id);
        let _held = lock.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(mut thread) = self.threads.load(&thread_id)? else {
            return Err(BrainError::contract(NO_PENDING));
        };
        let mut answered: Vec<String> = request.results.iter().map(|(id, _)| id.clone()).collect();
        answered.sort_unstable();
        let Some(run_id) = thread.open_run.clone() else {
            return self.delivered(&thread_id, &answered);
        };
        let history = match self.worker.history(&run_id) {
            Ok(history) => history,
            // The run is gone already (a crash after its commit deleted it): the thread lets go of it.
            Err(Error::State(_)) => {
                thread.open_run = None;
                self.threads.save(&thread_id, &thread)?;
                return self.delivered(&thread_id, &answered);
            }
            Err(_) => return Err(BrainError::State),
        };
        let open = Open {
            thread_id: &thread_id,
            run_id: &run_id,
            asked: asked(&history),
            pending: self.pending(&run_id)?,
            answered,
        };
        self.resume_open(&mut thread, open, request, scope)
    }

    /// Resumes the thread's open run: an ended one is committed or delivered again, results already recorded are not
    /// recorded again, and the run is driven to its next requests or its end.
    fn resume_open(
        &self,
        thread: &mut Thread,
        open: Open<'_>,
        request: &crate::server::turn_input::ResumeRequest,
        scope: &Scope,
    ) -> Result<Value, BrainError> {
        let Open {
            thread_id,
            run_id,
            asked,
            pending,
            answered,
        } = open;
        let applied = applied(&answered, &pending, &asked);
        if let Some(outcome) = self.worker.result(run_id).map_err(state)? {
            // A run that ended before its exchange was committed commits it now; one committed is delivered again.
            if thread.committed_run.as_deref() != Some(run_id) {
                return self.ended(thread_id, thread, (run_id, answered), &outcome);
            }
            self.reconcile(thread_id, thread)?;
            return self.delivered(thread_id, &answered);
        }
        if pending.is_empty() && !applied {
            self.reconcile(thread_id, thread)?;
            return self.delivered(thread_id, &answered);
        }
        let input = self.input(run_id)?;
        if Self::scope_digest(&request.turn.context) != thread.scope {
            self.delete_thread_locked(thread_id)?;
            return Err(BrainError::contract(
                "Assistant scope changed during the pending turn",
            ));
        }
        let session = prepare::resumed(request, &input, scope)?;
        let fresh = fresh_results(&request.results, &pending, &asked)?;
        for (step, result) in prepare::resolutions(self, run_id, &fresh)? {
            self.worker
                .resolve(run_id, step, &run::result_json(&result))
                .map_err(state)?;
        }
        self.run(thread_id, thread, (run_id, answered), session)
    }

    /// The committed turn a repeated resume with exactly these `answered` interrupts receives again; any other resume
    /// finds no pending request.
    fn delivered(&self, thread_id: &str, answered: &[String]) -> Result<Value, BrainError> {
        match self.threads.delivery(thread_id)? {
            Some(delivery) if !answered.is_empty() && delivery.answered == answered => {
                Ok(delivery.turn)
            }
            _ => Err(BrainError::contract(NO_PENDING)),
        }
    }

    /// `AgentRuntime.delete_thread`: removes one conversation and its open run, revealing nothing.
    ///
    /// # Errors
    ///
    /// [`BrainError::State`].
    pub fn delete_thread(&self, thread_id: &str) -> Result<(), BrainError> {
        let lock = self.locks.of(thread_id);
        let _held = lock.lock().unwrap_or_else(PoisonError::into_inner);
        self.delete_thread_locked(thread_id)
    }

    fn delete_thread_locked(&self, thread_id: &str) -> Result<(), BrainError> {
        if let Some(thread) = self.threads.load(thread_id)?
            && let Some(run_id) = &thread.open_run
        {
            self.delete_run(run_id)?;
            crate::failpoint::hit("delete.run");
        }
        self.threads.delete(thread_id)
    }

    /// The pending Action `interrupt_id` of `thread_id` and its turn's start message, for its purpose.
    ///
    /// # Errors
    ///
    /// [`BrainError::State`] when the stores fail.
    pub fn pending_action(
        &self,
        thread_id: &str,
        interrupt_id: &str,
    ) -> Result<Option<Pending>, BrainError> {
        let lock = self.locks.of(thread_id);
        let _held = lock.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(run_id) = self
            .threads
            .load(thread_id)?
            .and_then(|thread| thread.open_run)
        else {
            return Ok(None);
        };
        let found = self
            .pending(&run_id)?
            .into_iter()
            .find(|request| request.interrupt_id() == interrupt_id);
        let Some(TeamRequest::Action {
            assistant_id,
            action,
            input,
            ..
        }) = found
        else {
            return Ok(None);
        };
        let pinned = self.input(&run_id)?;
        let turn_id = pinned.context.turn_message_id.clone().unwrap_or_default();
        let message = pinned.messages.iter().find_map(|message| match message {
            Message::User { id, text } if *id == turn_id => Some(text.clone()),
            _ => None,
        });
        Ok(message.map(|message| Pending {
            message,
            locale: pinned.context.locale.clone(),
            assistant_id,
            action,
            input,
        }))
    }
}

fn sleep_until(at: i64) {
    let now = holdpoint::Clock::now_ms(&holdpoint::SystemClock);
    let wait = u64::try_from(at.saturating_sub(now))
        .unwrap_or(0)
        .clamp(1, 60_000);
    std::thread::sleep(Duration::from_millis(wait));
}

/// The interrupts a run asked Team for, answered or not.
fn asked(history: &[Event]) -> Vec<String> {
    history
        .iter()
        .filter_map(|event| match event {
            Event::StepScheduled {
                kind: StepKind::Deferred,
                name,
                input: Some(input),
                ..
            } => {
                TeamRequest::from_json(name, input).map(|request| request.interrupt_id().to_owned())
            }
            _ => None,
        })
        .collect()
}

/// Whether every `answered` interrupt is one the run asked for and none of them is still pending: an earlier resume
/// recorded their results.
fn applied(answered: &[String], pending: &[TeamRequest], asked: &[String]) -> bool {
    !answered.is_empty()
        && answered.iter().all(|id| asked.contains(id))
        && !pending
            .iter()
            .any(|request| answered.iter().any(|id| id == request.interrupt_id()))
}

/// The open run a resume finds: its thread, the interrupts it asked for and still waits on, and what the resume
/// answers.
struct Open<'a> {
    thread_id: &'a str,
    run_id: &'a str,
    asked: Vec<String>,
    pending: Vec<TeamRequest>,
    answered: Vec<String>,
}

/// The results a resume still has to record: an earlier resume recorded the others (all of them when Team repeats a
/// resume whose answer it lost). A result naming an interrupt the run never asked for is refused.
fn fresh_results(
    results: &[(String, Value)],
    pending: &[TeamRequest],
    asked: &[String],
) -> Result<Vec<(String, Value)>, BrainError> {
    let waiting = |id: &str| pending.iter().any(|request| request.interrupt_id() == id);
    if results
        .iter()
        .any(|(id, _)| !waiting(id) && !asked.contains(id))
    {
        return Err(BrainError::contract("invalid Action resume results"));
    }
    Ok(results
        .iter()
        .filter(|(id, _)| waiting(id))
        .cloned()
        .collect())
}

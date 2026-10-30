//! The Brain as Holdpoint's host: the request-scoped session of each turn under drive, and the proofs Holdpoint asks
//! before it dispatches again.
//!
//! A turn's provider key and attachment content live only in its [`TurnSession`], which the turn service installs
//! under the thread's lock for the one request that drives the run and removes before the lock is released; a run
//! with no installed session waits for authority (and the service answers 503). The process holds the state
//! directory's exclusive lock for its whole life, so no other process can be driving any run, and one thread lock
//! admits one drive per run in this process: every whole-session obligation Holdpoint records (an expired lease, a
//! failed release) therefore names a session that already ended, and is proven ended. The Brain registers no
//! activity, so no attempt obligation can exist; one would be answered "not yet" (fail closed). A live lease is never
//! proven: Holdpoint refuses to delete or drive a run whose lease has not expired, and the service waits for it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use holdpoint::{Authority, Credentials, Host, Obligation, Proof, RunInfo};

use crate::attachments::Attachment;
use crate::provider::{ProviderConfig, Scope};

/// What one request lends the run it drives.
#[derive(Debug)]
pub struct TurnSession {
    /// The provider and key.
    pub provider: ProviderConfig,
    /// The prepared attachments.
    pub attachments: Vec<Attachment>,
    /// The request's cancellation and usage.
    pub scope: Scope,
}

/// The Brain's Holdpoint host.
#[derive(Debug, Default)]
pub struct BrainHost {
    sessions: Mutex<HashMap<String, Arc<TurnSession>>>,
}

impl BrainHost {
    /// Installs the session of the request about to drive `run_id`.
    pub fn install(&self, run_id: &str, session: TurnSession) {
        self.sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(run_id.to_owned(), Arc::new(session));
    }

    /// Removes `run_id`'s session; its key is zeroized once the last call holding it returns.
    pub fn remove(&self, run_id: &str) {
        self.sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(run_id);
    }
}

impl Host for BrainHost {
    fn authorize(&self, run: &RunInfo<'_>) -> Authority {
        let sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        sessions.get(run.id).map_or(Authority::Wait, |session| {
            Authority::Granted(Credentials::new(Arc::clone(session)))
        })
    }

    fn prove_ended(&self, _run: &RunInfo<'_>, obligation: &Obligation) -> Proof {
        if obligation.attempt.is_none() {
            Proof::Ended
        } else {
            Proof::NotYet
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderId;
    use serde_json::Value;
    use zeroize::Zeroizing;

    fn info(id: &str) -> RunInfo<'_> {
        static NULL: Value = Value::Null;
        RunInfo {
            id,
            definition: "brain-turn",
            version: 1,
            input: &NULL,
        }
    }

    #[test]
    fn only_an_installed_session_grants_authority() {
        let host = BrainHost::default();
        assert!(matches!(host.authorize(&info("run")), Authority::Wait));
        let provider = ProviderConfig {
            provider: ProviderId::OpenAi,
            model: "m".into(),
            api_key: Zeroizing::new("k".into()),
        };
        host.install(
            "run",
            TurnSession {
                provider,
                attachments: vec![],
                scope: Scope::default(),
            },
        );
        let Authority::Granted(credentials) = host.authorize(&info("run")) else {
            panic!("granted")
        };
        assert!(credentials.get::<Arc<TurnSession>>().is_some());
        assert!(matches!(host.authorize(&info("other")), Authority::Wait));
        host.remove("run");
        assert!(matches!(host.authorize(&info("run")), Authority::Wait));
    }

    #[test]
    fn whole_sessions_are_proven_and_attempts_never() {
        let host = BrainHost::default();
        let session = Obligation {
            id: 1,
            fence: 1,
            attempt: None,
            incarnation: String::new(),
        };
        let attempt = Obligation {
            id: 2,
            fence: 1,
            attempt: Some((1, 1)),
            incarnation: String::new(),
        };
        assert_eq!(host.prove_ended(&info("run"), &session), Proof::Ended);
        assert_eq!(host.prove_ended(&info("run"), &attempt), Proof::NotYet);
    }
}

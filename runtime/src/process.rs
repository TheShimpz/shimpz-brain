//! The serving process: its command line, the exclusive lock on its state directory, its stores, and its worker.

use std::fs::File;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use holdpoint::{
    Authority, Host, Limits, ModelOptions, Obligation, Proof, RetryPolicy, RunInfo, Store,
    Timeouts, Worker, WorkerOptions,
};

use crate::provider::{Providers, Route, Transport};
use crate::server::{App, routes::Lanes};
use crate::turn::definition::BrainTurn;
use crate::turn::host::BrainHost;
use crate::turn::model::ProviderModel;
use crate::turn::run;
use crate::turn::service::TurnService;
use crate::turn::threads::Threads;

/// Team's runtime bearer in the image.
pub const TOKEN_FILE: &str = "/run/shimpz-brain-runtime/token";
/// The Brain's state volume in the image.
pub const STATE_DIR: &str = "/var/lib/shimpz-brain-runtime";
/// Where the image listens.
pub const LISTEN: &str = "0.0.0.0:8080";
/// A drive session's lease: a crashed session's runs wait at most this long before a new request settles them.
pub const LEASE: Duration = Duration::from_secs(6);

/// The `serve` command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// The listening address.
    pub listen: SocketAddr,
    /// Team's bearer file.
    pub token_file: PathBuf,
    /// The private state directory.
    pub state_dir: PathBuf,
    /// Where provider requests go.
    pub route: Route,
}

/// Reads `serve`'s arguments (after the subcommand); `None` for any other line.
#[must_use]
pub fn options(arguments: &[String]) -> Option<Options> {
    let mut listen = LISTEN.parse().ok()?;
    let mut token_file = PathBuf::from(TOKEN_FILE);
    let mut state_dir = PathBuf::from(STATE_DIR);
    let mut route = None;
    let mut pairs = arguments.chunks(2);
    for pair in &mut pairs {
        let [flag, value] = pair else { return None };
        match flag.as_str() {
            "--listen" => listen = value.parse().ok()?,
            "--token-file" => token_file = PathBuf::from(value),
            "--state-dir" => state_dir = PathBuf::from(value),
            "--egress" if route.is_none() => route = Some(Route::egress(value)?),
            "--provider-origin" if route.is_none() => route = Some(Route::origin(value)?),
            _ => return None,
        }
    }
    Some(Options {
        listen,
        token_file,
        state_dir,
        route: route?,
    })
}

/// Holds the state directory's exclusive lock for the life of the process.
///
/// # Errors
///
/// When the lock file cannot be opened or another process holds the lock.
pub fn lock_state(directory: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(directory.join("lock"))?;
    rustix::fs::flock(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive)?;
    Ok(file)
}

/// Shares the host between Holdpoint and the turn service.
struct SharedHost(Arc<BrainHost>);

impl Host for SharedHost {
    fn authorize(&self, run: &RunInfo<'_>) -> Authority {
        self.0.authorize(run)
    }

    fn prove_ended(&self, run: &RunInfo<'_>, obligation: &Obligation) -> Proof {
        self.0.prove_ended(run, obligation)
    }
}

/// The ledger's bounds: a turn's input holds every Assistant contract and its kept history.
#[must_use]
pub fn limits() -> Limits {
    Limits {
        max_value_bytes: 16 << 20,
        max_run_bytes: 256 << 20,
        ..Limits::default()
    }
}

/// Opens the stores and builds the application; must run inside the Tokio runtime.
///
/// # Errors
///
/// When a store cannot be opened.
pub fn app(options: &Options) -> Result<App, String> {
    let ledger = Store::open(&options.state_dir.join("ledger.sqlite3"), limits())
        .map_err(|_| "the ledger cannot be opened")?;
    let threads = Threads::open(&options.state_dir.join("threads.sqlite3"))
        .map_err(|_| "the thread store cannot be opened")?;
    let providers = Providers::new(Arc::new(Transport::new(&options.route)));
    let host = Arc::new(BrainHost::default());
    let model = ModelOptions {
        retry: RetryPolicy::never(),
        timeouts: Timeouts {
            start_to_close_ms: Some(300_000),
            ..Timeouts::default()
        },
    };
    let worker = Worker::builder(Arc::new(ledger), "brain", SharedHost(Arc::clone(&host)))
        .definition(BrainTurn)
        .model(run::MODEL, model, ProviderModel::new(providers.clone()))
        .options(WorkerOptions {
            lease_ms: u64::try_from(LEASE.as_millis()).unwrap_or(u64::MAX),
            inline_wait_ms: 1_000,
            poll_ms: 100,
            abandon_grace_ms: 2_000,
            max_abandoned: 16,
            proof_retry_ms: 200,
            concurrency: 1,
        })
        .build();
    let service = TurnService::new(Arc::new(worker), host, threads, providers, LEASE);
    Ok(App {
        token_file: options.token_file.clone(),
        service: Arc::new(service),
        lanes: Lanes::default(),
        memory: crate::server::boundary::MemoryBudget::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn serve_takes_exactly_one_provider_route() {
        let egress = options(&line("--egress shimpz-brain-egress:8888")).unwrap();
        assert_eq!(egress.listen, LISTEN.parse().unwrap());
        assert_eq!(
            egress.route,
            Route::Egress {
                host: "shimpz-brain-egress".into(),
                port: 8888
            }
        );
        let origin = options(&line("--listen 127.0.0.1:9 --state-dir /s --token-file /t --provider-origin http://127.0.0.1:7")).unwrap();
        assert_eq!(origin.state_dir, PathBuf::from("/s"));
        assert_eq!(origin.route, Route::Origin("127.0.0.1:7".parse().unwrap()));
        assert!(
            options(&line("--listen 127.0.0.1:9")).is_none(),
            "a route is required"
        );
        assert!(
            options(&line("--egress a:1 --provider-origin http://127.0.0.1:7")).is_none(),
            "only one route"
        );
        assert!(
            options(&line("--provider-origin http://10.0.0.1:7")).is_none(),
            "the origin is loopback only"
        );
        assert!(options(&line("--egress a:1 --other x")).is_none());
        assert!(options(&line("--egress")).is_none());
    }

    #[test]
    fn the_state_lock_is_exclusive() {
        let directory = tempfile::tempdir().unwrap();
        let held = lock_state(directory.path()).unwrap();
        assert!(lock_state(directory.path()).is_err());
        drop(held);
        assert!(lock_state(directory.path()).is_ok());
    }
}

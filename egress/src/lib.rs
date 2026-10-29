//! The only internet route for the shared Space Brain (`shimpz-egress brain`).
//!
//! The Brain reaches this proxy over its dedicated `brain_egress` network. The proxy is never attached to a Team
//! core, Assistant, or data plane, and its own `brain_egress_out` network is its only internet route, so internal
//! datastores stay unreachable and every outbound destination is audited. The exact `:443` allowlist is derived from
//! the packaged neutral model catalog plus the fixed decision host; a missing, invalid, or wildcard catalog policy
//! prevents the proxy from binding.
//!
//! Deliberately minimal: network-gated, not token-gated; CONNECT-only, so a plain-HTTP forward request is refused
//! (`405`) and the tunnel stays opaque TLS end to end; allowlisted by hostname, which this proxy resolves, so the
//! Brain, with no default route, cannot even resolve external names itself; and fail-closed. The neutral transport
//! (`egress-transport`, ADR-0104) resolves, records, connects, admits the TLS `ClientHello`, and splices; this
//! profile owns only its policy, audit stream, and resource envelope.

pub mod policy;

use std::collections::BTreeSet;
use std::fmt;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

use egress_transport::text::python_list;
use egress_transport::{
    AuditError, AuditWriter, Decision, Envelope, Fields, Limits, Profile, Server, Target,
    Transport, refuse,
};

/// The port every Brain `HTTPS_PROXY` address and the healthcheck name.
pub const PORT: u16 = 8888;
/// The security-relevant record of which public host the Brain was allowed to reach or refused.
pub const AUDIT_PATH: &str = "/var/log/brain-egress/audit.jsonl";
/// The service name of every audit line.
pub const SERVICE: &str = "egress-proxy";
/// HTTPS only: every legitimate Brain destination is TLS.
pub const ALLOWED_PORT: u16 = 443;
/// The concurrency, per-source concurrency, and backlog.
pub const LIMITS: Limits = Limits {
    concurrency: 64,
    source_concurrency: 8,
    backlog: 16,
};

/// The Brain profile: its exact host allowlist and its audit stream.
pub struct Brain {
    allowed: BTreeSet<String>,
    audit: AuditWriter,
}

impl Brain {
    /// A profile over a non-empty allowlist without a wildcard.
    ///
    /// # Errors
    ///
    /// The allowlist is empty or holds `*`.
    pub fn new(allowed: BTreeSet<String>, audit: AuditWriter) -> Result<Self, Startup> {
        if allowed.is_empty() || allowed.contains("*") {
            return Err(Startup::Policy);
        }
        Ok(Self { allowed, audit })
    }

    /// Whether one CONNECT exactly matches the `:443` policy: the host lowercased without one trailing dot.
    #[must_use]
    pub fn permitted(&self, host: &str, port: u16) -> bool {
        let canonical = host.to_ascii_lowercase();
        port == ALLOWED_PORT
            && self
                .allowed
                .contains(canonical.strip_suffix('.').unwrap_or(&canonical))
    }
}

impl Profile for Brain {
    fn envelope(&self) -> Envelope {
        Envelope::default()
    }

    /// Network-gated: forward only an exactly allowlisted provider or decision host.
    fn admit(&self, request: &[u8], _client: IpAddr) -> Result<Target, Decision> {
        let target = egress_transport::request::parse_connect(request)?;
        if !self.permitted(&target.host, target.port) {
            return Err(refuse(
                403,
                "target-rejected",
                target.subject(),
                Fields::new(),
            ));
        }
        Ok(target)
    }

    fn record(&self, decision: &Decision) -> Result<(), AuditError> {
        self.audit.log(decision)
    }
}

/// Why the profile refuses to start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Startup {
    /// The provider policy is unavailable or invalid.
    Policy,
    /// The audit directory's custody is unsafe or unavailable.
    Custody,
    /// The listener could not be bound.
    Listen,
}

impl fmt::Display for Startup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Policy => "provider policy is unavailable; refusing to start",
            Self::Custody => "audit custody is unavailable; refusing to start",
            Self::Listen => "the listener is unavailable; refusing to start",
        })
    }
}

/// Where the profile reads its policy, writes its audit, and listens.
pub struct Setup {
    /// The packaged model catalog.
    pub catalog: PathBuf,
    /// The audit stream.
    pub audit: AuditWriter,
    /// The listening port.
    pub port: u16,
    /// The system interfaces.
    pub transport: Transport,
}

impl Setup {
    /// The image's paths, port, and system interfaces.
    #[must_use]
    pub fn production() -> Self {
        Self {
            catalog: PathBuf::from(policy::CATALOG_PATH),
            audit: AuditWriter::new(AUDIT_PATH, SERVICE),
            port: PORT,
            transport: Transport::system(),
        }
    }
}

/// Load the policy, check audit custody, and bind, in that order; nothing listens unless all three hold.
///
/// # Errors
///
/// The first startup step that fails.
pub fn start(setup: Setup) -> Result<Server, Startup> {
    let providers = policy::load_provider_hosts(&setup.catalog).map_err(|_| Startup::Policy)?;
    setup.audit.ensure_custody().map_err(|_| Startup::Custody)?;
    let mut allowed = providers.clone();
    allowed.extend(policy::DECISION_HOSTS.map(str::to_owned));
    let profile = Arc::new(Brain::new(allowed, setup.audit)?);
    let server = Server::bind(setup.port, LIMITS, profile, Arc::new(setup.transport))
        .map_err(|_| Startup::Listen)?;
    let providers: Vec<&str> = providers.iter().map(String::as_str).collect();
    eprintln!(
        "brain-egress listening on :{}; providers={} decisions={}",
        setup.port,
        python_list(&providers),
        python_list(&policy::DECISION_HOSTS)
    );
    Ok(server)
}

/// Start and serve forever; on a startup failure, say why and return 1.
pub fn run(setup: Setup) -> u8 {
    match start(setup) {
        Ok(server) => server.serve(),
        Err(startup) => {
            eprintln!("brain-egress: {startup}");
            1
        }
    }
}

/// The image's entry: [`run`] with the production setup.
#[must_use]
pub fn main() -> u8 {
    run(Setup::production())
}

#[cfg(test)]
mod tests;

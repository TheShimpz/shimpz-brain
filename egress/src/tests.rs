use std::collections::BTreeSet;
use std::fs;
use std::net::{SocketAddr, TcpListener};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use egress_transport::fixtures::{FixedStamp, exchange, public, transport};
use egress_transport::{AuditWriter, Profile};

use super::{Brain, LIMITS, PORT, SERVICE, Setup, Startup, run, start};

/// A private 0700 directory removed at drop, holding the audit and the catalog.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("brain-egress-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).expect("scratch");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).expect("mode");
        fs::write(
            path.join("catalog.json"),
            include_bytes!("../../../.standards/model_catalog.json"),
        )
        .expect("catalog");
        Self(path)
    }

    fn writer(&self) -> AuditWriter {
        AuditWriter::with_parts(
            self.0.join("audit.jsonl"),
            SERVICE,
            Box::new(FixedStamp::default()),
            Box::new(egress_transport::audit::SystemDisk),
        )
    }

    fn lines(&self) -> Vec<String> {
        fs::read_to_string(self.0.join("audit.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn allowed(hosts: &[&str]) -> BTreeSet<String> {
    hosts.iter().map(|host| (*host).to_owned()).collect()
}

fn brain(scratch: &Scratch) -> Brain {
    Brain::new(allowed(&["api.openai.com"]), scratch.writer()).expect("valid policy")
}

fn closed() -> SocketAddr {
    public("127.0.0.1:9")
}

#[test]
fn only_an_exact_catalog_host_on_443_is_permitted() {
    let scratch = Scratch::new("permit");
    let brain = Brain::new(
        allowed(&["api.openai.com", "api.typesafe.ai"]),
        scratch.writer(),
    )
    .expect("valid");
    assert!(brain.permitted("api.openai.com", 443));
    assert!(brain.permitted("API.OPENAI.COM.", 443));
    assert!(brain.permitted("api.typesafe.ai", 443));
    for (host, port) in [
        ("sub.api.openai.com", 443),
        ("api.openai.com", 80),
        ("*", 443),
        ("api.openai.com..", 443),
        ("typesafe.ai", 443),
        ("docs.typesafe.ai", 443),
    ] {
        assert!(!brain.permitted(host, port), "{host}:{port}");
    }
    for policy in [allowed(&[]), allowed(&["*"])] {
        assert_eq!(
            Brain::new(policy, scratch.writer()).err(),
            Some(Startup::Policy)
        );
    }
    assert_eq!(brain.envelope(), egress_transport::Envelope::default());
}

#[test]
fn an_exact_host_is_tunneled_and_recorded_with_its_admitted_addresses() {
    let scratch = Scratch::new("tunnel");
    let brain = brain(&scratch);
    let upstream = TcpListener::bind("127.0.0.1:0").expect("bind");
    let (transport, _) = transport(
        vec![public("1.1.1.1:443")],
        upstream.local_addr().expect("address"),
        Vec::new(),
    );
    let request = b"CONNECT API.OpenAI.com.:443 HTTP/1.1\r\n\r\n";
    let response = exchange(&brain, &transport, request, public("10.0.0.2:1234"));
    assert_eq!(response, b"HTTP/1.1 200 Connection established\r\n\r\n");
    let lines = scratch.lines();
    assert!(lines[1].contains("\"reason\":\"tls-incomplete\",\"result\":\"denied\""));
    assert_eq!(
        lines[0],
        format!(
            "{{\"addresses\":[\"1.1.1.1\"],\"code\":200,\"level\":\"info\",\"msg\":\"connect API.OpenAI.com.:443: ok\",\
             \"op\":\"connect\",\"prev_sha256\":\"{}\",\"reason\":\"allowed\",\"result\":\"ok\",\
             \"service\":\"egress-proxy\",\"subject\":\"API.OpenAI.com.:443\",\
             \"trace_id\":\"00000000000000000000000000000000\",\"ts\":\"2026-10-29T12:00:00Z\"}}",
            "0".repeat(64)
        )
    );
}

#[test]
fn non_connect_malformed_and_unlisted_targets_are_refused_and_recorded() {
    for (request, status, reason) in [
        (&b"GET / HTTP/1.1\r\n\r\n"[..], 405, "method"),
        (b"CONNECT api.openai.com HTTP/1.1\r\n\r\n", 400, "target"),
        (
            b"CONNECT sub.api.openai.com:443 HTTP/1.1\r\n\r\n",
            403,
            "target-rejected",
        ),
        (
            b"CONNECT api.openai.com:80 HTTP/1.1\r\n\r\n",
            403,
            "target-rejected",
        ),
    ] {
        let scratch = Scratch::new("refuse");
        let brain = brain(&scratch);
        let (transport, connector) = transport(vec![public("1.1.1.1:443")], closed(), Vec::new());
        let response = exchange(&brain, &transport, request, public("10.0.0.2:1234"));
        assert!(response.starts_with(format!("HTTP/1.1 {status} ").as_bytes()));
        let lines = scratch.lines();
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].contains(&format!("\"code\":{status},\"level\":\"warn\"")),
            "{}",
            lines[0]
        );
        assert!(lines[0].contains(&format!("\"reason\":\"{reason}\",\"result\":\"denied\"")));
        assert_eq!(connector.attempts.lock().expect("attempts").len(), 0);
    }
}

#[test]
fn the_loopback_healthcheck_probe_is_a_distinct_non_warning_refusal() {
    let scratch = Scratch::new("probe");
    let brain = brain(&scratch);
    let (transport, _) = transport(Vec::new(), closed(), Vec::new());
    let response = exchange(
        &brain,
        &transport,
        b"GET / HTTP/1.1\r\n\r\n",
        public("127.0.0.1:40000"),
    );
    assert!(response.starts_with(b"HTTP/1.1 405 "));
    let line = &scratch.lines()[0];
    assert!(
        line.contains("\"level\":\"info\"") && line.contains("\"source\":\"loopback-probe\""),
        "{line}"
    );
}

fn setup(scratch: &Scratch, catalog: &str, port: u16) -> Setup {
    let (transport, _) = transport(Vec::new(), closed(), Vec::new());
    Setup {
        catalog: scratch.0.join(catalog),
        audit: scratch.writer(),
        port,
        transport,
    }
}

#[test]
fn startup_checks_policy_then_custody_then_binds_with_the_profile_s_envelope() {
    let scratch = Scratch::new("startup");
    let server = start(setup(&scratch, "catalog.json", 0)).expect("starts");
    assert_eq!(server.admission.held().0, 0);
    assert_eq!(
        (
            LIMITS.concurrency,
            LIMITS.source_concurrency,
            LIMITS.backlog
        ),
        (64, 8, 16)
    );
    assert_eq!(
        start(setup(&scratch, "missing.json", 0)).err(),
        Some(Startup::Policy)
    );
    let busy = server.listener.local_addr().expect("address").port();
    assert_eq!(
        start(setup(&scratch, "catalog.json", busy)).err(),
        Some(Startup::Listen)
    );
    fs::set_permissions(&scratch.0, fs::Permissions::from_mode(0o750)).expect("mode");
    assert_eq!(
        start(setup(&scratch, "catalog.json", 0)).err(),
        Some(Startup::Custody)
    );
    fs::set_permissions(&scratch.0, fs::Permissions::from_mode(0o700)).expect("mode");
    assert_eq!(run(setup(&scratch, "missing.json", 0)), 1);
    for (startup, message) in [
        (
            Startup::Policy,
            "provider policy is unavailable; refusing to start",
        ),
        (
            Startup::Custody,
            "audit custody is unavailable; refusing to start",
        ),
        (
            Startup::Listen,
            "the listener is unavailable; refusing to start",
        ),
    ] {
        assert_eq!(startup.to_string(), message);
    }
}

#[test]
fn a_started_profile_serves_until_the_process_ends() {
    let scratch = Scratch::new("serve");
    let port = {
        let probe = TcpListener::bind("127.0.0.1:0").expect("bind");
        probe.local_addr().expect("address").port()
    };
    let serving = setup(&scratch, "catalog.json", port);
    std::thread::spawn(move || run(serving));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !egress_transport::health::refuses_get(port, std::time::Duration::from_secs(1)) {
        assert!(
            std::time::Instant::now() < deadline,
            "the profile never listened"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(PORT, 8888);
    assert_eq!(super::main(), 1);
}

//! Running the built `shimpz-brain` binary for black-box tests: a private state directory and token, a loopback
//! listener, and a minimal HTTP/1.1 client.

#![allow(
    dead_code,
    reason = "each test binary uses a part of the shared support"
)]

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The test bearer.
pub const TOKEN: &str = "test-runtime-token";

/// One running Brain.
pub struct Brain {
    /// Where it listens.
    pub address: SocketAddr,
    /// Its state directory.
    pub state: PathBuf,
    child: Child,
}

/// A free loopback port.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

impl Brain {
    /// Starts the binary over `directory` (its token file and `state/`), sending provider calls to `origin`.
    pub fn start(directory: &Path, origin: SocketAddr, environment: &[(&str, &str)]) -> Self {
        let state = directory.join("state");
        std::fs::create_dir_all(&state).unwrap();
        let token = directory.join("token");
        if !token.exists() {
            std::fs::write(&token, TOKEN).unwrap();
        }
        let address: SocketAddr = format!("127.0.0.1:{}", free_port()).parse().unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_shimpz-brain"));
        command
            .args(["serve", "--listen", &address.to_string(), "--token-file"])
            .arg(&token)
            .arg("--state-dir")
            .arg(&state)
            .args(["--provider-origin", &format!("http://{origin}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for (name, value) in environment {
            command.env(name, value);
        }
        let child = command.spawn().unwrap();
        let brain = Self {
            address,
            state,
            child,
        };
        brain.wait_healthy();
        brain
    }

    fn wait_healthy(&self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if request(self.address, "GET", "/health", &[], None)
                .is_ok_and(|response| response.status == 200)
            {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("the Brain did not become healthy");
    }

    /// Kills the process at once, as a crash would.
    pub fn kill(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Waits for the process to exit by itself (a failpoint abort); its exit status.
    pub fn wait_exit(&mut self) -> std::process::ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the Brain did not exit");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// A POST with the test bearer and a JSON body.
    pub fn post(&self, path: &str, body: &str) -> Response {
        let bearer = format!("Bearer {TOKEN}");
        request(
            self.address,
            "POST",
            path,
            &[("authorization", &bearer)],
            Some(body.as_bytes()),
        )
        .unwrap()
    }
}

impl Drop for Brain {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One HTTP answer.
#[derive(Debug)]
pub struct Response {
    /// The status code.
    pub status: u16,
    /// The raw head.
    pub head: String,
    /// The body text.
    pub body: String,
}

/// Sends one request with `Connection: close` and reads the whole answer.
pub fn request(
    address: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> std::io::Result<Response> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))?;
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: brain\r\nConnection: close\r\n");
    for (name, value) in headers {
        let _ = write!(head, "{name}: {value}\r\n");
    }
    if let Some(body) = body {
        let _ = write!(
            head,
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        );
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    if let Some(body) = body {
        stream.write_all(body)?;
    }
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status = head
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    Ok(Response {
        status,
        head: head.to_owned(),
        body: body.to_owned(),
    })
}

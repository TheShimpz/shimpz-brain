//! A scripted provider origin on loopback: the Brain sends it the provider requests it would send through its
//! egress, and it answers each with the next scripted behaviour, recording what it received.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

/// What the origin does with one request.
#[derive(Debug, Clone)]
pub enum Behaviour {
    /// Answers with this status and JSON body.
    Answer(u16, String),
    /// Reads the request, then holds the connection open until the client closes it.
    Hang,
}

/// One request the origin received.
#[derive(Debug, Clone)]
pub struct Received {
    /// The request path.
    pub path: String,
    /// The request body.
    pub body: String,
}

#[derive(Debug, Default)]
struct State {
    script: VecDeque<Behaviour>,
    received: Vec<Received>,
    closed_by_client: usize,
}

/// The running origin.
#[derive(Debug, Clone)]
pub struct Origin {
    /// Where it listens.
    pub address: SocketAddr,
    state: Arc<Mutex<State>>,
}

fn read_request(stream: &mut TcpStream) -> Option<Received> {
    let mut raw = Vec::new();
    let mut byte = [0_u8; 1];
    while !raw.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).ok()? == 0 {
            return None;
        }
        raw.push(byte[0]);
    }
    let head = String::from_utf8_lossy(&raw).into_owned();
    let path = head.split(' ').nth(1)?.to_owned();
    let length = head
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|value| value.trim().to_owned())
        })
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0_u8; length];
    stream.read_exact(&mut body).ok()?;
    Some(Received {
        path,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

impl Origin {
    /// Starts an origin answering with `script`, in order; once it is empty every request gets a 500.
    #[must_use]
    pub fn start(script: Vec<Behaviour>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let state = Arc::new(Mutex::new(State {
            script: script.into(),
            ..State::default()
        }));
        let shared = Arc::clone(&state);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let state = Arc::clone(&shared);
                std::thread::spawn(move || serve(stream, &state));
            }
        });
        Self { address, state }
    }

    /// Every request received so far.
    #[must_use]
    pub fn received(&self) -> Vec<Received> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .received
            .clone()
    }

    /// How many hanging connections the client closed.
    #[must_use]
    pub fn closed_by_client(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .closed_by_client
    }

    /// Waits until the origin received `count` requests.
    pub fn wait_received(&self, count: usize) {
        for _ in 0..1_000 {
            if self.received().len() >= count {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("the origin did not receive {count} requests");
    }

    /// Waits until the client closed `count` hanging connections, within `seconds`.
    pub fn wait_closed(&self, count: usize, seconds: u64) -> bool {
        for _ in 0..seconds * 100 {
            if self.closed_by_client() >= count {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }
}

fn serve(mut stream: TcpStream, state: &Arc<Mutex<State>>) {
    let Some(received) = read_request(&mut stream) else {
        return;
    };
    let behaviour = {
        let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
        state.received.push(received);
        state.script.pop_front().unwrap_or_else(|| {
            Behaviour::Answer(500, r#"{"error":{"type":"server_error"}}"#.into())
        })
    };
    match behaviour {
        Behaviour::Answer(status, body) => {
            let head = format!(
                "HTTP/1.1 {status} OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body.as_bytes());
        }
        Behaviour::Hang => {
            let mut buffer = [0_u8; 64];
            while stream.read(&mut buffer).is_ok_and(|read| read > 0) {}
            state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .closed_by_client += 1;
        }
    }
}

/// An OpenAI Responses answer whose output is `output` (a JSON array text).
#[must_use]
pub fn openai_answer(output: &str) -> Behaviour {
    Behaviour::Answer(
        200,
        format!(
            r#"{{"id":"resp_test","object":"response","status":"completed","output":{output},"usage":{{"input_tokens":12,"output_tokens":3,"total_tokens":15,"input_tokens_details":{{"cached_tokens":0}},"output_tokens_details":{{"reasoning_tokens":0}}}}}}"#
        ),
    )
}

/// An OpenAI Responses answer with one plain reply.
#[must_use]
pub fn reply(text: &str) -> Behaviour {
    openai_answer(&format!(
        r#"[{{"type":"message","id":"msg_test","role":"assistant","status":"completed","content":[{{"type":"output_text","text":"{text}","annotations":[]}}]}}]"#
    ))
}

/// An OpenAI Responses answer with one function call.
#[must_use]
pub fn call(name: &str, arguments: &str) -> Behaviour {
    let arguments = arguments.replace('"', "\\\"");
    openai_answer(&format!(
        r#"[{{"type":"function_call","id":"fc_test","call_id":"call_test","name":"{name}","arguments":"{arguments}","status":"completed"}}]"#
    ))
}

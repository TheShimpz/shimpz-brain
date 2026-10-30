//! The `shimpz-brain` binary: `serve` the Brain, or `healthcheck` the one serving in this container.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use brain_runtime::process;

fn healthcheck() -> ExitCode {
    let checked = (|| -> std::io::Result<bool> {
        let mut stream = TcpStream::connect_timeout(
            &"127.0.0.1:8080".parse().map_err(std::io::Error::other)?,
            Duration::from_secs(2),
        )?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        stream.write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
        let mut head = [0_u8; 128];
        let read = stream.read(&mut head)?;
        let status = head[..read]
            .split(|byte| *byte == b'\r')
            .next()
            .unwrap_or_default();
        Ok(status == b"HTTP/1.0 200 OK" || status == b"HTTP/1.1 200 OK")
    })();
    if matches!(checked, Ok(true)) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

async fn stopped() {
    let Ok(mut terminate) =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
    else {
        return;
    };
    tokio::select! {
        _ = terminate.recv() => {}
        _ = tokio::signal::ctrl_c() => {}
    }
}

fn serve(arguments: &[String]) -> ExitCode {
    let Some(options) = process::options(arguments) else {
        return ExitCode::from(2);
    };
    let Ok(_lock) = process::lock_state(&options.state_dir) else {
        return ExitCode::from(3);
    };
    let Ok(runtime) = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    else {
        return ExitCode::FAILURE;
    };
    runtime.block_on(async {
        let Ok(app) = process::app(&options) else {
            return ExitCode::FAILURE;
        };
        let Ok(listener) = tokio::net::TcpListener::bind(options.listen).await else {
            return ExitCode::FAILURE;
        };
        brain_runtime::server::serve(listener, Arc::new(app), stopped()).await;
        ExitCode::SUCCESS
    })
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.split_first() {
        Some((command, rest)) if command == "serve" => serve(rest),
        Some((command, rest)) if command == "healthcheck" && rest.is_empty() => healthcheck(),
        _ => ExitCode::from(2),
    }
}

//! Fuzzes the chat turn requests over arbitrary bodies: strict JSON, then the closed start and resume readers, which
//! must admit a request or refuse it, never panic.
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    if let Ok(body) = strict_json::parse(data) {
        let _ = brain_runtime::server::turn_input::start(&body);
        let _ = brain_runtime::server::turn_input::resume(&body);
    }
});

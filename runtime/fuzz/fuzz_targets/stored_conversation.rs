//! Fuzzes the stored conversation and turn context decoders over arbitrary JSON: a tampered thread row or run input
//! must decode exactly or be refused.
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    if let Ok(value) = strict_json::parse(data) {
        let _ = brain_runtime::conversation::from_value(&value);
        let _ = brain_runtime::turn::context::TurnContext::from_value(&value);
    }
});

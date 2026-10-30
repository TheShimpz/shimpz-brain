//! Fuzzes both provider adapters' answer parsers over arbitrary provider bodies: one model turn or a refusal.
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let _ = brain_runtime::provider::openai::response(data);
    let _ = brain_runtime::provider::anthropic::response(data);
});

//! The stateless operations Team asks the Brain: Action purposes, capability plans, intent routes (with the Jev fast
//! path), and the Browser agent's step, fold, completeness check, and final answer. Each reads its closed request,
//! makes its provider calls on the request's [`crate::provider::Scope`], and returns its answer without `usage`, which
//! the server adds.

pub mod browser;
pub mod capability;
pub mod intent;
pub mod jev;
pub mod purpose;

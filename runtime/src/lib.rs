//! The Shimpz Brain: the provider-neutral reasoning runtime Team's chat turns, decisions, and Browser steps call
//! (ADR-0124). A chat turn is a durable Holdpoint run; every other operation is one stateless provider call. The
//! Brain requests Actions and never executes one: Team owns authorization, execution, and audit.

pub mod attachments;
pub mod budget;
pub mod cancel;
pub mod catalog;
pub mod conversation;
pub mod decisions;
pub mod errors;
pub mod failpoint;
pub mod json;
pub mod process;
pub mod prompt;
pub mod provider;
pub mod server;
pub mod text;
pub mod tools;
pub mod turn;

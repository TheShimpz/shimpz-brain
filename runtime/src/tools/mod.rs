//! The chat agent's own tools and the guards that review every model response before any of its calls runs
//! (`action_tool.py`, `clarification.py`, `memory.py`, `routine.py`, `browser_tool.py`, `tool_refusal.py`).

pub mod action;
pub mod browser;
pub mod clarification;
pub mod memory;
pub mod refusal;
pub mod routine;

use crate::errors::BrainError;

/// A guard's verdict on the latest model response: `Ok(None)` lets its calls run; `Ok(Some(correction))` refuses the
/// whole response, answering every call of it with `correction`; `Err` ends the turn (an unanswerable call).
pub type Review = Result<Option<String>, BrainError>;

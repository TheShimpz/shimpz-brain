//! The Brain's closed error kinds and the exact HTTP answer each one has (the Python runtime's mapping).

use std::fmt;

/// Why a Brain operation failed. No variant carries provider text, request content, or a credential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrainError {
    /// Trusted orchestration input or persisted output violated the closed contract: 400 with this fixed text.
    Contract(String),
    /// A provider request failed, or its answer broke a closed contract: 502.
    Provider,
    /// A state operation failed without exposing what is stored: 503.
    State,
    /// Team closed the request while it ran: 409 with this fixed text.
    Cancelled(&'static str),
    /// The operation's own capacity lane is full: 429 with this fixed text.
    Capacity(&'static str),
}

impl BrainError {
    /// A contract error with `detail`.
    #[must_use]
    pub fn contract(detail: impl Into<String>) -> Self {
        Self::Contract(detail.into())
    }

    /// The HTTP status and public detail of this error.
    #[must_use]
    pub fn answer(&self) -> (u16, &str) {
        match self {
            Self::Contract(detail) => (400, detail),
            Self::Provider => (502, "Model provider request failed"),
            Self::State => (503, "Brain runtime state operation failed"),
            Self::Cancelled(detail) => (409, detail),
            Self::Capacity(detail) => (429, detail),
        }
    }
}

impl fmt::Display for BrainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (status, detail) = self.answer();
        write!(formatter, "{status} {detail}")
    }
}

impl std::error::Error for BrainError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_has_its_fixed_answer() {
        assert_eq!(
            BrainError::contract("invalid chat message").answer(),
            (400, "invalid chat message")
        );
        assert_eq!(
            BrainError::Provider.answer(),
            (502, "Model provider request failed")
        );
        assert_eq!(
            BrainError::State.answer(),
            (503, "Brain runtime state operation failed")
        );
        assert_eq!(
            BrainError::Cancelled("Chat turn cancelled").answer(),
            (409, "Chat turn cancelled")
        );
        assert_eq!(
            BrainError::Capacity("Intent route capacity reached").to_string(),
            "429 Intent route capacity reached"
        );
    }
}

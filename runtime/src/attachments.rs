//! Request-local chat attachment content (`attachments.py`, ADR-0093): admitted from Team's closed field, committed
//! by digest, projected only into each provider call of the turn, and charged against the turn's token ceilings.

use crate::conversation::{Block, Message};
use crate::errors::BrainError;
use crate::json::Value;
use crate::provider::{Projection, ProviderConfig, Providers, Scope};
use crate::turn::context::AssistantDefinition;

/// The id prefix of a turn that carries attachments, which the next new turn forgets whole.
pub const ATTACHED_TURN_PREFIX: &str = "shimpz-attached-";
/// The provider attempts one attachment call may make, each reserving its charge.
pub const MAX_ATTEMPTS: u32 = 3;

/// One attachment's content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Content {
    /// Extracted text; `pdf` marks text extracted from a PDF.
    Text {
        /// The text.
        text: String,
        /// Whether it came from a PDF.
        pdf: bool,
    },
    /// A re-encoded JPEG or PNG.
    Image {
        /// Its media type.
        media_type: String,
        /// Its base64 encoding.
        base64: String,
    },
    /// A file that cannot be read here.
    Opaque {
        /// Why.
        reason: String,
    },
}

/// One prepared file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attachment {
    /// Its file id.
    pub id: String,
    /// Its name.
    pub name: String,
    /// Its media type.
    pub media_type: String,
    /// Its size in bytes.
    pub size: u64,
    /// Its SHA-256.
    pub sha256: String,
    /// Its content.
    pub content: Content,
}

/// Admits Team's `attachments` field (`admit`).
///
/// # Errors
///
/// [`BrainError::Contract`] "invalid attachments".
pub fn admit(_raw: &Value) -> Result<Vec<Attachment>, BrainError> {
    unimplemented!("ported by the attachments microtask")
}

/// The digest a turn records at its start (`commitment`).
#[must_use]
pub fn commitment(_attachments: &[Attachment]) -> String {
    unimplemented!("ported by the attachments microtask")
}

/// Each attachment's content kind: `text`, `image`, or `opaque`.
#[must_use]
pub fn kinds(_attachments: &[Attachment]) -> Vec<String> {
    unimplemented!("ported by the attachments microtask")
}

/// One attachment as the blocks a provider call carries (`blocks`).
#[must_use]
pub fn blocks(_attachment: &Attachment) -> Vec<Block> {
    unimplemented!("ported by the attachments microtask")
}

/// The projection of the turn's attachments onto its turn message; `None` without attachments.
#[must_use]
pub fn projection(_turn_message_id: &str, _attachments: &[Attachment]) -> Option<Projection> {
    unimplemented!("ported by the attachments microtask")
}

/// Each attachment's token charge, counted by the provider within one deadline or estimated (`charges`), admitted
/// against the per-file and per-call ceilings (`admit_charges`): the per-call charge.
///
/// # Errors
///
/// [`BrainError::Contract`] with the Python texts; [`BrainError::Cancelled`] when the turn was cancelled.
pub fn charge(
    _providers: &Providers,
    _scope: &Scope,
    _config: &ProviderConfig,
    _attachments: &[Attachment],
) -> Result<u64, BrainError> {
    unimplemented!("ported by the attachments microtask")
}

/// Refuses a provider call that would carry the attachments past the turn's cumulative ceiling (`admit_call`).
///
/// # Errors
///
/// "attachments exceed the turn's token ceiling".
pub fn admit_call(_charge: u64, _calls_so_far: u32) -> Result<(), String> {
    unimplemented!("ported by the attachments microtask")
}

/// The provider attempts the logical turn already reserved, as its model turns record them (`dispatched`).
#[must_use]
pub fn dispatched(_messages: &[Message], _turn_message_id: &str) -> u32 {
    unimplemented!("ported by the attachments microtask")
}

/// Whether a user message started a turn that read attachments (`read`).
#[must_use]
pub fn read(_message: &Message) -> bool {
    unimplemented!("ported by the attachments microtask")
}

/// The Assistants a turn offers: all, or only authorizing Actions while attachment content is in it (`exposed`).
#[must_use]
pub fn exposed(
    _assistants: &[AssistantDefinition],
    _reads_content: bool,
) -> Vec<AssistantDefinition> {
    unimplemented!("ported by the attachments microtask")
}

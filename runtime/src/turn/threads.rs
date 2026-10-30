//! The Brain's thread store: each conversation's kept history, the Assistant scope it belongs to, and the one run a
//! thread may have open (`threads.sqlite3` beside the Holdpoint ledger).
//!
//! A row is strict JSON text decoded with the strict reader; a row that does not decode is a state error, never
//! repaired. Deletions overwrite their pages (`secure_delete`) and empty the write-ahead log, checked, so a deleted
//! thread's text survives in neither file. Two-store ordering (the debate's revision 4): a turn records `open_run`
//! before it submits its run; a completed turn appends its exchange together with `committed_run` (the same run id)
//! while keeping `open_run`, then the service deletes the run and clears `open_run`; a later request finding
//! `open_run` set deletes that run and appends nothing unless `committed_run` already names it.

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension, params};

use crate::conversation::{self, Message};
use crate::errors::BrainError;
use crate::json;

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    history TEXT NOT NULL,
    open_run TEXT,
    committed_run TEXT
) STRICT";

/// One stored thread.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Thread {
    /// The digest of the exact Assistant contract the history belongs to.
    pub scope: String,
    /// The kept history.
    pub history: Vec<Message>,
    /// The run the thread has open, if any.
    pub open_run: Option<String>,
    /// The run whose exchange `history` already holds.
    pub committed_run: Option<String>,
}

/// The thread store.
#[derive(Debug)]
pub struct Threads {
    connection: Mutex<Connection>,
}

fn state<T>(_error: T) -> BrainError {
    BrainError::State
}

impl Threads {
    /// Opens (creating with mode 0600) the store at `path`.
    ///
    /// # Errors
    ///
    /// [`BrainError::State`].
    pub fn open(path: &Path) -> Result<Self, BrainError> {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(path)
            .map_err(state)?;
        let connection = Connection::open(path).map_err(state)?;
        connection
            .busy_timeout(std::time::Duration::from_secs(10))
            .map_err(state)?;
        for (name, value) in [
            ("journal_mode", "WAL"),
            ("synchronous", "FULL"),
            ("secure_delete", "ON"),
            ("trusted_schema", "OFF"),
        ] {
            connection.pragma_update(None, name, value).map_err(state)?;
        }
        connection.execute_batch(SCHEMA).map_err(state)?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, BrainError> {
        self.connection.lock().map_err(state)
    }

    /// The stored thread, or `None`.
    ///
    /// # Errors
    ///
    /// [`BrainError::State`] when the store fails or the row does not decode.
    pub fn load(&self, thread_id: &str) -> Result<Option<Thread>, BrainError> {
        let connection = self.lock()?;
        let row: Option<(String, String, Option<String>, Option<String>)> = connection
            .query_row(
                "SELECT scope, history, open_run, committed_run FROM threads WHERE thread_id = ?1",
                [thread_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()
            .map_err(state)?;
        let Some((scope, history, open_run, committed_run)) = row else {
            return Ok(None);
        };
        let history = json::parse(history.as_bytes())
            .ok()
            .and_then(|value| conversation::from_value(&value))
            .ok_or(BrainError::State)?;
        Ok(Some(Thread {
            scope,
            history,
            open_run,
            committed_run,
        }))
    }

    /// Stores `thread` under `thread_id`.
    ///
    /// # Errors
    ///
    /// [`BrainError::State`].
    pub fn save(&self, thread_id: &str, thread: &Thread) -> Result<(), BrainError> {
        let history = json::compact(&conversation::to_value(&thread.history));
        self.lock()?
            .execute(
                "INSERT INTO threads (thread_id, scope, history, open_run, committed_run) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT (thread_id) DO UPDATE SET scope = excluded.scope, history = excluded.history,
                 open_run = excluded.open_run, committed_run = excluded.committed_run",
                params![thread_id, thread.scope, history, thread.open_run, thread.committed_run],
            )
            .map_err(state)?;
        Ok(())
    }

    /// Deletes the thread (whether or not it exists) and empties the log.
    ///
    /// # Errors
    ///
    /// [`BrainError::State`] when the store fails or the log cannot be emptied.
    pub fn delete(&self, thread_id: &str) -> Result<(), BrainError> {
        self.lock()?
            .execute("DELETE FROM threads WHERE thread_id = ?1", [thread_id])
            .map_err(state)?;
        self.truncate_log()
    }

    /// Empties the write-ahead log, refusing when a reader keeps it.
    ///
    /// # Errors
    ///
    /// [`BrainError::State`].
    pub fn truncate_log(&self) -> Result<(), BrainError> {
        let busy: i64 = self
            .lock()?
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))
            .map_err(state)?;
        if busy == 0 {
            Ok(())
        } else {
            Err(BrainError::State)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, Threads) {
        let directory = tempfile::tempdir().unwrap();
        let threads = Threads::open(&directory.path().join("threads.sqlite3")).unwrap();
        (directory, threads)
    }

    #[test]
    fn a_thread_round_trips_and_is_deleted_without_trace() {
        let (directory, threads) = store();
        let marker = "thread-erasure-marker";
        let thread = Thread {
            scope: "scope".into(),
            history: vec![Message::User {
                id: "u1".into(),
                text: marker.into(),
            }],
            open_run: Some("run".into()),
            committed_run: None,
        };
        assert_eq!(threads.load("t1").unwrap(), None);
        threads.save("t1", &thread).unwrap();
        assert_eq!(threads.load("t1").unwrap(), Some(thread));
        threads.delete("t1").unwrap();
        threads.delete("t1").unwrap();
        assert_eq!(threads.load("t1").unwrap(), None);
        let path = directory.path().join("threads.sqlite3");
        let wal = std::fs::metadata(path.with_extension("sqlite3-wal"))
            .map_or(0, |metadata| metadata.len());
        assert_eq!(wal, 0);
        let bytes = std::fs::read(path).unwrap();
        assert!(
            !bytes
                .windows(marker.len())
                .any(|window| window == marker.as_bytes())
        );
    }

    #[test]
    fn a_row_that_does_not_decode_is_a_state_error() {
        let (directory, threads) = store();
        threads.save("t1", &Thread::default()).unwrap();
        let tamper = Connection::open(directory.path().join("threads.sqlite3")).unwrap();
        tamper
            .execute("UPDATE threads SET history = '[{\"role\":\"user\"}]'", [])
            .unwrap();
        assert_eq!(threads.load("t1"), Err(BrainError::State));
        tamper
            .execute("UPDATE threads SET history = 'not json'", [])
            .unwrap();
        assert_eq!(threads.load("t1"), Err(BrainError::State));
    }

    #[test]
    fn a_kept_log_refuses_erasure() {
        let (directory, threads) = store();
        threads.save("t1", &Thread::default()).unwrap();
        let reader = Connection::open(directory.path().join("threads.sqlite3")).unwrap();
        reader
            .execute_batch("BEGIN; SELECT count(*) FROM threads;")
            .unwrap();
        assert_eq!(threads.delete("t1"), Err(BrainError::State));
        reader.execute_batch("COMMIT;").unwrap();
        assert_eq!(threads.delete("t1"), Ok(()));
    }
}

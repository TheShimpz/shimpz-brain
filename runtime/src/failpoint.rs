//! Named crash points for the kill tests (the `failpoints` feature, never built into the image): the process aborts
//! at the point `SHIMPZ_BRAIN_FAILPOINT` names, leaving its stores exactly as a crash there would.

/// Aborts the process when the `failpoints` feature is on and `SHIMPZ_BRAIN_FAILPOINT` names `point`.
#[cfg(feature = "failpoints")]
pub fn hit(point: &str) {
    if std::env::var("SHIMPZ_BRAIN_FAILPOINT").is_ok_and(|selected| selected == point) {
        std::process::abort();
    }
}

/// Does nothing: crash points exist only in test builds.
#[cfg(not(feature = "failpoints"))]
pub fn hit(_point: &str) {}

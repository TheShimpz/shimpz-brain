//! Black-box tests of the built `shimpz-brain` binary over HTTP.

mod agent;
mod boundary;
#[cfg(feature = "failpoints")]
mod crash;
mod origin;
mod support;
mod team;
mod turns;

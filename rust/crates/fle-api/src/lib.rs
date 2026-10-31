//! The `/api` that FLE Studio's web page talks to, in Rust. Stage 0 grows route by route; the Python Studio is the
//! specification and `tests/` compare both servers request by request.
#![allow(clippy::manual_is_multiple_of)] // `is_multiple_of` needs a newer Rust than we promise
pub mod catalog;
pub mod chats;
pub mod downloads;
pub mod engine;
pub mod engines;
pub mod gguf;
pub mod hf;
pub mod launch;
pub mod library;
pub mod live;
pub mod report_time;
pub mod run;
pub mod server;
pub mod settings;
pub mod system;

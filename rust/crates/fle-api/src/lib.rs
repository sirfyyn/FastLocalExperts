//! The `/api` that FLE Studio's web page talks to, in Rust. Stage 0 grows route by route; the Python Studio is the
//! specification and `tests/` compare both servers request by request.
pub mod catalog;
pub mod chats;
pub mod downloads;
pub mod engines;
pub mod gguf;
pub mod hf;
pub mod launch;
pub mod library;
pub mod server;
pub mod settings;
pub mod system;

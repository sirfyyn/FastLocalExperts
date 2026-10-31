//! fle-core: everything `fle` (terminal) and, later, the Tauri app need to take a person from "nothing installed"
//! to "model answering": hardware, engines, model catalog, resumable downloads, start plan, server, chat.
//! No Python, no uv, no git. One static binary on top of this.

pub mod catalog;
pub mod chat;
pub mod download;
pub mod engines;
pub mod plan;
pub mod policy;
pub mod server;
pub mod system;

use std::path::PathBuf;

/// Where FLE keeps engines, models and logs. `--home` / FLE_HOME override it.
pub fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("FLE_HOME") {
        return PathBuf::from(p);
    }
    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .unwrap_or_default()
            .join("Library/Application Support/FLE")
    }
    #[cfg(target_os = "windows")]
    {
        dirs::data_local_dir().unwrap_or_default().join("FLE")
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        dirs::data_local_dir().unwrap_or_default().join("fle")
    }
}

/// Progress events every long step reports, so a TUI, a CLI and a GUI can show them their own way.
#[derive(Debug, Clone)]
pub enum Event {
    Step(String),
    Progress {
        label: String,
        done: u64,
        total: u64,
    },
    Log(String),
    Ready {
        base: String,
        engine: String,
        model: String,
        command: String,
    },
}

pub mod flow;

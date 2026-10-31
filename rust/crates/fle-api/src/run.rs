//! Start FLE Studio's server: `fle studio`. Port of `fle_studio/__main__.py` for the local mode (the network mode with TLS,
//! pairing and updates comes with the remote stage).
use crate::server::{new_studio, serve_listener, Options};
use serde_json::Value;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// The version the page is told (the Python Studio this server replaces).
pub const STUDIO_VERSION: &str = "0.2.0";

#[derive(Default, Clone)]
pub struct RunOptions {
    pub home: Option<PathBuf>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub no_browser: bool,
    /// use an already running llama.cpp / FLE server
    pub attach: String,
    /// serve the page from this folder instead of the copy inside the binary
    pub web_dir: Option<PathBuf>,
}

/// Where FLE Studio keeps its files on each operating system (`paths.data_dir`).
pub fn data_dir() -> PathBuf {
    if let Ok(o) = std::env::var("FLE_STUDIO_HOME") {
        if !o.is_empty() {
            return match o.strip_prefix("~/") {
                Some(r) => dirs::home_dir().unwrap_or_default().join(r),
                None => PathBuf::from(o),
            };
        }
    }
    let home = dirs::home_dir().unwrap_or_default();
    if cfg!(target_os = "macos") {
        home.join("Library")
            .join("Application Support")
            .join("FLE Studio")
    } else if cfg!(windows) {
        PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_else(|_| {
            home.join("AppData")
                .join("Local")
                .to_string_lossy()
                .to_string()
        }))
        .join("FLE Studio")
    } else {
        PathBuf::from(std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
            home.join(".local")
                .join("share")
                .to_string_lossy()
                .to_string()
        }))
        .join("fle-studio")
    }
}

/// Is an FLE Studio already answering on this port?
fn running(port: u16) -> bool {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(1500))
        .build();
    match agent
        .get(&format!("http://127.0.0.1:{port}/api/ping"))
        .call()
    {
        Ok(r) => r
            .into_string()
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .map(|v| v["app"] == "fle-studio")
            .unwrap_or(false),
        Err(_) => false,
    }
}

pub fn open_browser(url: &str) {
    let (cmd, args): (&str, Vec<&str>) = if cfg!(target_os = "macos") {
        ("open", vec![url])
    } else if cfg!(windows) {
        ("cmd", vec!["/c", "start", "", url])
    } else {
        ("xdg-open", vec![url])
    };
    let _ = std::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Run the server until Ctrl+C or SIGTERM. Returns the exit code.
pub fn run(o: RunOptions) -> i32 {
    let home = o
        .home
        .clone()
        .map(|h| match h.to_string_lossy().strip_prefix("~/") {
            Some(r) => dirs::home_dir().unwrap_or_default().join(r),
            None => h.clone(),
        });
    let home = home.unwrap_or_else(data_dir);
    let _ = std::fs::create_dir_all(&home);
    let st = new_studio(Options {
        home: home.clone(),
        web_dir: o.web_dir.clone(),
        version: STUDIO_VERSION.into(),
        github_api: crate::engines::GITHUB_API.into(),
    });
    let cfg = |k: &str| st.config.lock().unwrap().get(k).cloned();
    let host = o.host.clone().unwrap_or_else(|| {
        cfg("server.host")
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "127.0.0.1".into())
    });
    let port = o
        .port
        .filter(|p| *p != 0)
        .unwrap_or_else(|| cfg("server.port").and_then(|v| v.as_u64()).unwrap_or(8642) as u16);
    let local = matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1");
    if !local {
        eprintln!("FLE Studio: listening beyond this machine (host mode with an access key, TLS and pairing) is not in the Rust server yet. Use the Python FLE Studio for that, or an SSH tunnel to this one.");
        return 1;
    }
    if running(port) {
        let url = format!("http://127.0.0.1:{port}/");
        println!("FLE Studio is already running: {url}");
        if !o.no_browser {
            open_browser(&url);
        }
        return 0;
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let result: Result<(), String> = rt.block_on(async {
        // the port, or the next free one above it (up to 20 tries); 0 asks the system
        let mut bound = None;
        let mut last = None;
        let tries: Vec<u16> = if port == 0 {
            vec![0]
        } else {
            (port..port.saturating_add(20)).collect()
        };
        for p in tries {
            match tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], p))).await {
                Ok(l) => {
                    bound = Some(l);
                    break;
                }
                Err(e) => last = Some(e),
            }
        }
        let Some(listener) = bound else {
            return Err(format!(
                "FLE Studio could not start its web server: {}",
                last.map(|e| e.to_string()).unwrap_or_default()
            ));
        };
        let actual = listener.local_addr().map(|a| a.port()).unwrap_or(port);
        let url = format!("http://127.0.0.1:{actual}/");
        println!("FLE Studio {STUDIO_VERSION} — {url}");
        println!("Data folder: {}", home.display());
        println!("Press Ctrl+C to stop (this also stops a model FLE Studio started).");
        let key = cfg("attach_key")
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();
        let (mut attach_url, mut attach_unit) = (o.attach.clone(), String::new());
        if attach_url.is_empty()
            && cfg("attach_auto")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        {
            // connected when FLE Studio last ran
            attach_url = cfg("attach_url")
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
            attach_unit = cfg("attach_service")
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
        }
        if !attach_url.is_empty() || !attach_unit.is_empty() {
            let shown = if attach_unit.is_empty() {
                attach_url.clone()
            } else {
                attach_unit.clone()
            };
            match st.engine.attach(&attach_url, &key, &attach_unit) {
                Ok(_) => println!("Connected to {shown}"),
                Err(e) => eprintln!(
                    "Could not connect to {shown}: {}",
                    match e {
                        crate::engine::EngineErr::Busy(m)
                        | crate::engine::EngineErr::Invalid(m) => m,
                    }
                ),
            }
        } else if cfg("launch.autostart")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            && cfg("last_model")
                .and_then(|v| v.as_str().map(|s| !s.is_empty()))
                .unwrap_or(false)
        {
            let st2 = st.clone();
            std::thread::spawn(move || {
                let model = st2
                    .config
                    .lock()
                    .unwrap()
                    .get("last_model")
                    .and_then(|v| v.as_str().map(String::from))
                    .unwrap_or_default();
                let profile = st2.config.lock().unwrap().get("launch.profile").cloned();
                match st2.make_plan(&model, profile.as_ref()) {
                    Ok(plan) => {
                        let name = plan.model["name"].as_str().unwrap_or("").to_string();
                        match st2.engine.start(plan) {
                            Ok(()) => println!("Loading {name} …"),
                            Err(e) => eprintln!(
                                "Autostart failed: {}",
                                match e {
                                    crate::engine::EngineErr::Busy(m)
                                    | crate::engine::EngineErr::Invalid(m) => m,
                                }
                            ),
                        }
                    }
                    Err(e) => eprintln!("Autostart failed: {}", e.1),
                }
            });
        }
        let open = cfg("server.open_browser")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
            && !o.no_browser;
        if open {
            let u = url.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(800));
                open_browser(&u);
            });
        }
        let stop = async {
            #[cfg(unix)]
            {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("signal");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        };
        serve_listener(st.clone(), listener, stop)
            .await
            .map_err(|e| e.to_string())?;
        println!("Stopping FLE Studio …");
        st.engine.shutdown();
        Ok(())
    });
    match result {
        Ok(()) => 0,
        Err(m) => {
            eprintln!("{m}");
            1
        }
    }
}

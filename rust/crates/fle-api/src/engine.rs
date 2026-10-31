//! The engine: one llama-server process that FLE Studio starts and watches, or a server it attaches to. Port of
//! `fle_studio/engine.py` and `procutil.py` without the systemd part (a unit to follow comes with the Linux service work).
//!
//! All process starts happen on one long-lived supervisor thread; on Linux the engine is told to exit when that thread ends
//! (`PR_SET_PDEATHSIG`), so that thread must not be a short-lived request thread.
use crate::launch::Plan;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Log lines that mark a loading phase (checked in order, first match wins).
const PHASES: [(&str, &str); 9] = [
    ("listening on", "Ready"),
    ("model loaded", "Starting the server"),
    ("warming up", "Warming up"),
    (
        "moe expert cache enabled",
        "Setting up the GPU expert cache",
    ),
    ("llama_kv_cache", "Allocating the KV cache"),
    ("llama_context:", "Creating the context"),
    ("load_tensors:", "Loading weights"),
    ("llama_params_fit", "Fitting the model into GPU memory"),
    ("llama_model_loader: loaded meta data", "Reading the model"),
];

/// Error lines and what to tell the user about them.
const HINTS: [(&str, &str); 6] = [
    (
        r"(?i)out of memory|cudaMalloc failed|failed to allocate .*buffer|ErrorOutOfDeviceMemory",
        "The GPU ran out of memory. Lower the context or the GPU expert cache on the Engine page.",
    ),
    (
        r"(?i)error while loading shared libraries|cannot open shared object|\.dll was not found|Library not loaded|image not found",
        "The engine misses a library. Reinstall the engine on the Engine page.",
    ),
    (
        r"(?i)CUDA driver version is insufficient|no CUDA-capable device|cudaGetDeviceCount",
        "CUDA could not start. Update the NVIDIA driver, or use the Vulkan or CPU engine.",
    ),
    (
        r"(?i)failed to load model|error loading model|unable to load model",
        "The engine could not load the model. Check that the download finished (Models page).",
    ),
    (
        r"(?i)couldn't bind|address already in use|bind failed|only one usage of each socket address",
        "The engine port is taken. Set the port to 0 (automatic) on the Engine page.",
    ),
    (
        r"(?i)error: (invalid|unknown) argument|unknown option|invalid value",
        "The engine does not know one of the arguments. Is this profile meant for another engine?",
    ),
];

struct Rx {
    phases: Vec<(&'static str, &'static str)>,
    hints: Vec<(Regex, &'static str)>,
    cache: Regex,
    cache_on: Regex,
    accept: Regex,
}

fn rx() -> &'static Rx {
    static R: OnceLock<Rx> = OnceLock::new();
    R.get_or_init(|| Rx {
        phases: PHASES.to_vec(),
        hints: HINTS
            .iter()
            .map(|(r, h)| (Regex::new(r).unwrap(), *h))
            .collect(),
        cache: Regex::new(
            r"moe-cache:.*?(?:steps=(\d+).*?)?hits=(\d+).*?misses=(\d+).*?hit-rate=([\d.]+)\s*%",
        )
        .unwrap(),
        cache_on: Regex::new(r"(?i)MoE expert cache enabled[:,]?\s*(.*)").unwrap(),
        accept: Regex::new(r"draft acceptance(?: rate)?\s*=\s*([\d.]+)").unwrap(),
    })
}

/// What the page is told for an engine problem: `Busy` is a 409, `Invalid` (a `ValueError`) a 400.
#[derive(Debug, Clone)]
pub enum EngineErr {
    Busy(String),
    Invalid(String),
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// True when something already accepts connections on this local port. A second llama-server started there would take its
/// share of RAM and GPU memory next to the first one, so Studio refuses the start instead.
pub fn port_taken(port: i64) -> bool {
    let Ok(p) = u16::try_from(port) else {
        return false;
    };
    std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], p)),
        Duration::from_millis(500),
    )
    .is_ok()
}

type Listener = Box<dyn Fn(&str) + Send + Sync>;

struct State {
    state: String, // stopped | starting | ready | stopping | crashed | attached
    plan: Option<(u64, Plan)>,
    base_url: String,
    api_key: String,
    reachable: bool,
    lines: VecDeque<(u64, f64, String)>,
    seq: u64,
    props: Value,
    phase: String,
    error: String,
    hint: String,
    exit_code: Option<i32>,
    started: f64,
    ready_at: f64,
    load_total: u64,
    load_read: u64,
    rss: Option<u64>,
    cache: Map<String, Value>,
    accept_logged: Option<f64>,
    restart_on_crash: bool,
    crashes: VecDeque<f64>,
    stop_requested: bool,
    log_file: Option<std::fs::File>,
    /// the running process: (generation, pid, handle)
    proc: Option<(u64, u32, Arc<Mutex<Child>>)>,
    next_gen: u64,
}

enum Op {
    Start(u64, Box<Plan>),
    Stop,
}

#[derive(Clone)]
pub struct Engine {
    st: Arc<Mutex<State>>,
    ops: Sender<Op>,
    log_dir: PathBuf,
    listeners: Arc<Mutex<Vec<Listener>>>,
}

fn is_running(c: &Arc<Mutex<Child>>) -> bool {
    matches!(c.lock().unwrap().try_wait(), Ok(None))
}

impl Engine {
    pub fn new(data_dir: &Path) -> Engine {
        let log_dir = data_dir.join("logs");
        let _ = std::fs::create_dir_all(&log_dir);
        let st = Arc::new(Mutex::new(State {
            state: "stopped".into(),
            plan: None,
            base_url: String::new(),
            api_key: String::new(),
            reachable: false,
            lines: VecDeque::new(),
            seq: 0,
            props: json!({}),
            phase: String::new(),
            error: String::new(),
            hint: String::new(),
            exit_code: None,
            started: 0.0,
            ready_at: 0.0,
            load_total: 0,
            load_read: 0,
            rss: None,
            cache: Map::new(),
            accept_logged: None,
            restart_on_crash: false,
            crashes: VecDeque::new(),
            stop_requested: false,
            log_file: None,
            proc: None,
            next_gen: 1,
        }));
        let (tx, rx) = channel::<Op>();
        let e = Engine {
            st,
            ops: tx,
            log_dir,
            listeners: Arc::new(Mutex::new(vec![])),
        };
        let sup = e.clone();
        std::thread::Builder::new()
            .name("engine-supervisor".into())
            .spawn(move || sup.supervise(rx))
            .expect("supervisor thread");
        e
    }

    pub fn add_listener(&self, l: Listener) {
        self.listeners.lock().unwrap().push(l);
    }

    pub fn set_restart_on_crash(&self, on: bool) {
        self.st.lock().unwrap().restart_on_crash = on;
    }

    // ------------------------------------------------------------------ control

    fn refuse_start(s: &State) -> Result<(), EngineErr> {
        if matches!(s.state.as_str(), "starting" | "ready" | "stopping") {
            return Err(EngineErr::Busy(
                "An engine is already running. Stop it first.".into(),
            ));
        }
        Ok(())
    }

    pub fn start(&self, plan: Plan) -> Result<(), EngineErr> {
        Self::refuse_start(&self.st.lock().unwrap())?;
        if port_taken(plan.port) {
            return Err(EngineErr::Busy(format!("Port {} is already in use: another server answers there. Stop that server first, or give this start another port.", plan.port)));
        }
        let gen;
        {
            let mut s = self.st.lock().unwrap();
            Self::refuse_start(&s)?;
            gen = s.next_gen;
            s.next_gen += 1;
            s.reachable = false;
            s.state = "starting".into();
            s.phase = "Starting the engine".into();
            s.base_url = format!("http://127.0.0.1:{}", plan.port);
            s.plan = Some((gen, plan.clone()));
            s.error.clear();
            s.hint.clear();
            s.exit_code = None;
            s.cache = Map::new();
            s.accept_logged = None;
            s.props = json!({});
            s.stop_requested = false;
            s.api_key.clear();
        }
        let _ = self.ops.send(Op::Start(gen, Box::new(plan)));
        Ok(())
    }

    /// Stop the engine FLE Studio started, or disconnect from an attached server.
    pub fn stop(&self) {
        let mut s = self.st.lock().unwrap();
        if s.state == "attached" {
            Self::detach_locked(&mut s);
            return;
        }
        if s.proc.is_none() {
            if s.state != "crashed" {
                s.state = "stopped".into();
            }
            return;
        }
        s.stop_requested = true;
        s.state = "stopping".into();
        s.phase = "Stopping".into();
        drop(s);
        let _ = self.ops.send(Op::Stop);
    }

    fn detach_locked(s: &mut State) {
        s.state = "stopped".into();
        s.base_url.clear();
        s.api_key.clear();
        s.reachable = false;
        s.props = json!({});
        s.error.clear();
        s.hint.clear();
        s.phase.clear();
        s.load_total = 0;
        s.load_read = 0;
        s.rss = None;
    }

    /// Use a server that runs already. A systemd unit to follow comes with the Linux service work.
    pub fn attach(&self, url: &str, key: &str, unit: &str) -> Result<Value, EngineErr> {
        if !unit.trim().is_empty() {
            return Err(EngineErr::Invalid(
                "Services can be used on Linux with systemd only.".into(),
            ));
        }
        let mut url = url.trim().trim_end_matches('/').to_string();
        if url.is_empty() {
            return Err(EngineErr::Invalid(
                "Enter the server address, e.g. http://127.0.0.1:8080".into(),
            ));
        }
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            url = format!("http://{url}");
        }
        if let Some(u) = url.strip_suffix("/v1") {
            url = u.to_string();
        }
        {
            let mut s = self.st.lock().unwrap();
            if s.proc.is_some() {
                return Err(EngineErr::Busy(
                    "Stop the engine FLE Studio started before attaching to another server.".into(),
                ));
            }
            s.state = "attached".into();
            s.base_url = url;
            s.api_key = key.to_string();
            s.plan = None;
            s.error.clear();
            s.hint.clear();
            s.phase.clear();
            s.cache = Map::new();
            s.accept_logged = None;
            s.props = json!({});
            s.exit_code = None;
            s.rss = None;
            s.load_total = 0;
            s.load_read = 0;
        }
        self.refresh_attached();
        Ok(self.status())
    }

    /// "url" (attached) or "".
    pub fn bound(&self) -> &'static str {
        if self.st.lock().unwrap().state == "attached" {
            "url"
        } else {
            ""
        }
    }

    /// Read the attached server's health again.
    pub fn refresh_attached(&self) {
        if self.st.lock().unwrap().state != "attached" {
            return;
        }
        let code = self.health();
        let info = if code == 200 {
            self.get_json("/props")
        } else {
            None
        };
        let mut s = self.st.lock().unwrap();
        if s.state == "attached" {
            s.reachable = code == 200;
            if let Some(i) = info {
                s.props = i;
            }
            s.error = if code == 200 {
                String::new()
            } else if code == 503 {
                "The server is loading its model".into()
            } else {
                "The server does not answer".into()
            };
        }
    }

    /// Stop a running engine and wait for it; used when Studio exits.
    pub fn shutdown(&self) {
        let proc = {
            let mut s = self.st.lock().unwrap();
            s.stop_requested = true;
            s.proc.clone()
        };
        if let Some((_, pid, c)) = proc {
            if is_running(&c) {
                ask_to_stop(pid, &c);
                let t = Instant::now();
                while is_running(&c) && t.elapsed() < Duration::from_secs(15) {
                    std::thread::sleep(Duration::from_millis(50));
                }
                if is_running(&c) {
                    let _ = c.lock().unwrap().kill();
                }
            }
        }
    }

    // ------------------------------------------------------------------ supervisor

    fn supervise(&self, rx: std::sync::mpsc::Receiver<Op>) {
        for op in rx {
            match op {
                Op::Start(gen, plan) => {
                    if let Err(e) = self.spawn(gen, *plan) {
                        let msg = format!("Could not start the engine: {e}");
                        {
                            let mut s = self.st.lock().unwrap();
                            s.proc = None;
                            s.state = "crashed".into();
                            s.error = msg.clone();
                            s.phase.clear();
                        }
                        self.append(&format!("FLE Studio: {msg}"));
                    }
                }
                Op::Stop => self.terminate(),
            }
        }
    }

    fn spawn(&self, gen: u64, plan: Plan) -> Result<(), String> {
        {
            let s = self.st.lock().unwrap();
            if s.state != "starting" || s.plan.as_ref().map(|p| p.0) != Some(gen) {
                return Ok(()); // stopped again before we got here
            }
        }
        let binary = PathBuf::from(&plan.binary);
        if !binary.is_file() {
            return Err(format!("no engine at {}", binary.display()));
        }
        let dir = binary.parent().unwrap_or(Path::new(".")).to_path_buf();
        let mut cmd = Command::new(&binary);
        cmd.args(&plan.args).current_dir(&dir).stdin(Stdio::null());
        for (k, v) in &plan.env {
            cmd.env(k, v);
        }
        if cfg!(target_os = "linux") {
            let old = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let mine = plan
                .env
                .iter()
                .find(|(k, _)| k == "LD_LIBRARY_PATH")
                .map(|(_, v)| v.clone())
                .unwrap_or(old);
            cmd.env(
                "LD_LIBRARY_PATH",
                if mine.is_empty() {
                    dir.display().to_string()
                } else {
                    format!("{}:{mine}", dir.display())
                },
            );
        } else if cfg!(windows) {
            let path = plan
                .env
                .iter()
                .find(|(k, _)| k == "PATH")
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| std::env::var("PATH").unwrap_or_default());
            cmd.env("PATH", format!("{};{path}", dir.display()));
        }
        self.open_log();
        let start_line = format!("$ {}", plan.command_line().replace('\n', " "));
        self.append(&start_line);
        for l in self.listeners.lock().unwrap().iter() {
            l(&start_line);
        }
        let mut total = 0u64;
        let model_path = plan.model["path"].as_str().unwrap_or("");
        let mut files: Vec<PathBuf> = crate::gguf::split_parts(Path::new(model_path));
        if let Some(d) = &plan.draft {
            files.push(PathBuf::from(d));
        }
        for f in files {
            total += std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        }
        // stdout and stderr go into one pipe, like `stderr=STDOUT`
        let (reader, writer) = std::io::pipe().map_err(|e| e.to_string())?;
        cmd.stdout(writer.try_clone().map_err(|e| e.to_string())?)
            .stderr(writer);
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            // SAFETY: only async-signal-safe calls between fork and exec
            unsafe {
                cmd.pre_exec(|| {
                    libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                    Ok(())
                });
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000 | 0x0000_0200);
        }
        let child = cmd.spawn().map_err(|e| e.to_string())?;
        drop(cmd); // the parent's copy of the write end of the pipe must go, or the reader never sees the end
        let pid = child.id();
        let handle = Arc::new(Mutex::new(child));
        {
            let mut s = self.st.lock().unwrap();
            s.proc = Some((gen, pid, handle.clone()));
            s.started = now();
            s.ready_at = 0.0;
            s.load_total = total;
            s.load_read = 0;
            s.rss = None;
        }
        let me = self.clone();
        let h2 = handle.clone();
        std::thread::Builder::new()
            .name("engine-log".into())
            .spawn(move || me.read_output(gen, reader, h2))
            .map_err(|e| e.to_string())?;
        let me = self.clone();
        std::thread::Builder::new()
            .name("engine-ready".into())
            .spawn(move || me.wait_ready(gen, pid, handle))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn terminate(&self) {
        let proc = self.st.lock().unwrap().proc.clone();
        let Some((_, pid, c)) = proc else { return };
        if !is_running(&c) {
            return;
        }
        ask_to_stop(pid, &c);
        let t = Instant::now();
        while is_running(&c) && t.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(50));
        }
        if is_running(&c) {
            self.append("FLE Studio: the engine did not stop within 20 s; killing it");
            let _ = c.lock().unwrap().kill();
            let t = Instant::now();
            while is_running(&c) && t.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    fn read_output(&self, gen: u64, reader: std::io::PipeReader, child: Arc<Mutex<Child>>) {
        let mut r = std::io::BufReader::new(reader);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match r.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf)
                        .trim_end_matches(['\r', '\n'])
                        .to_string();
                    if !line.trim().is_empty() {
                        self.on_line(&line, true);
                    }
                }
            }
        }
        let code = loop {
            match child.lock().unwrap().try_wait() {
                Ok(Some(st)) => break exit_code(&st),
                Ok(None) => {}
                Err(_) => break -1,
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        self.on_exit(gen, code);
    }

    fn wait_ready(&self, gen: u64, pid: u32, child: Arc<Mutex<Child>>) {
        while is_running(&child) {
            {
                let s = self.st.lock().unwrap();
                if s.proc.as_ref().map(|p| p.0) != Some(gen) || s.state != "starting" {
                    return;
                }
            }
            let (read, rss) = read_and_memory(pid);
            {
                let mut s = self.st.lock().unwrap();
                if let Some(r) = read {
                    s.load_read = r;
                }
                s.rss = rss;
            }
            if self.health() == 200 {
                let props = self.get_json("/props").unwrap_or(json!({}));
                let mut s = self.st.lock().unwrap();
                if s.proc.as_ref().map(|p| p.0) == Some(gen) && s.state == "starting" {
                    s.state = "ready".into();
                    s.phase = "Ready".into();
                    s.ready_at = now();
                    s.props = props;
                    s.reachable = true;
                }
                return;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    fn on_exit(&self, gen: u64, rc: i32) {
        let mut again: Option<Plan> = None;
        {
            let mut s = self.st.lock().unwrap();
            if s.proc.as_ref().map(|p| p.0) != Some(gen) {
                return;
            }
            s.proc = None;
            s.exit_code = Some(rc);
            s.reachable = false;
            if s.stop_requested {
                s.state = "stopped".into();
                s.phase.clear();
            } else {
                s.state = "crashed".into();
                s.phase.clear();
                if s.error.is_empty() {
                    let last = s
                        .lines
                        .iter()
                        .rev()
                        .map(|(_, _, t)| t.clone())
                        .find(|t| !t.starts_with("FLE Studio:"))
                        .unwrap_or_default();
                    s.error = format!("The engine exited (code {rc})")
                        + &if last.is_empty() {
                            String::new()
                        } else {
                            format!(": {last}")
                        };
                }
                let t = now();
                s.crashes.push_back(t);
                if s.crashes.len() > 10 {
                    s.crashes.pop_front();
                }
                let recent = s.crashes.iter().filter(|c| t - **c < 600.0).count();
                if s.restart_on_crash && s.ready_at != 0.0 && recent <= 3 {
                    again = s.plan.as_ref().map(|p| p.1.clone());
                }
            }
        }
        self.append(&format!("FLE Studio: engine exited with code {rc}"));
        self.close_log();
        if let Some(plan) = again {
            self.append("FLE Studio: restart on crash is on; starting the engine again in 5 s");
            let me = self.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(5));
                if let Err(EngineErr::Busy(m) | EngineErr::Invalid(m)) = me.start(plan) {
                    me.append(&format!("FLE Studio: not restarted. {m}"));
                }
            });
        }
    }

    // ------------------------------------------------------------------ log

    fn open_log(&self) {
        self.close_log();
        let mut logs: Vec<PathBuf> = std::fs::read_dir(&self.log_dir)
            .map(|r| {
                r.flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name().is_some_and(|n| {
                            n.to_string_lossy().starts_with("engine-")
                                && n.to_string_lossy().ends_with(".log")
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        logs.sort();
        let keep_from = logs.len().saturating_sub(9);
        for old in &logs[..keep_from] {
            let _ = std::fs::remove_file(old);
        }
        let stamp = crate::report_time::stamp();
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_dir.join(format!("engine-{stamp}.log")))
            .ok();
        self.st.lock().unwrap().log_file = f;
    }

    fn close_log(&self) {
        self.st.lock().unwrap().log_file = None;
    }

    pub fn append(&self, text: &str) {
        let mut s = self.st.lock().unwrap();
        s.seq += 1;
        let (seq, t) = (s.seq, now());
        s.lines.push_back((seq, t, text.to_string()));
        if s.lines.len() > 5000 {
            s.lines.pop_front();
        }
        if let Some(f) = s.log_file.as_mut() {
            let _ = writeln!(f, "{text}");
            let _ = f.flush();
        }
    }

    fn on_line(&self, line: &str, parse: bool) {
        self.append(line);
        if !parse {
            return;
        }
        for l in self.listeners.lock().unwrap().iter() {
            l(line);
        }
        let low = line.to_lowercase();
        let rx = rx();
        let mut s = self.st.lock().unwrap();
        if s.state == "starting" {
            if let Some((_, phase)) = rx.phases.iter().find(|(n, _)| low.contains(n)) {
                s.phase = phase.to_string();
            }
        }
        if let Some(m) = rx.cache.captures(line) {
            let g = |i: usize| m.get(i).map(|x| x.as_str());
            s.cache.insert(
                "steps".into(),
                json!(g(1).and_then(|x| x.parse::<i64>().ok()).unwrap_or(0)),
            );
            s.cache.insert(
                "hits".into(),
                json!(g(2).and_then(|x| x.parse::<i64>().ok()).unwrap_or(0)),
            );
            s.cache.insert(
                "misses".into(),
                json!(g(3).and_then(|x| x.parse::<i64>().ok()).unwrap_or(0)),
            );
            s.cache.insert(
                "hit_rate".into(),
                json!(g(4).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0)),
            );
            s.cache.insert("at".into(), json!(now()));
        } else if let Some(m) = rx.cache_on.captures(line) {
            let v = m[1].trim();
            s.cache
                .insert("enabled".into(), json!(if v.is_empty() { "on" } else { v }));
        }
        if let Some(m) = rx.accept.captures(line) {
            s.accept_logged = m[1].parse::<f64>().ok();
        }
        if s.hint.is_empty() {
            if let Some((_, hint)) = rx.hints.iter().find(|(r, _)| r.is_match(line)) {
                s.hint = hint.to_string();
                s.error = line.trim().chars().take(400).collect();
            }
        }
    }

    pub fn log_since(&self, seq: u64, limit: usize) -> Value {
        let s = self.st.lock().unwrap();
        let out: Vec<Value> = s
            .lines
            .iter()
            .filter(|(n, _, _)| *n > seq)
            .map(|(n, t, x)| json!({"seq": n, "t": t, "text": x}))
            .collect();
        let skip = out.len().saturating_sub(limit);
        json!({"lines": out[skip..], "seq": s.seq})
    }

    // ------------------------------------------------------------------ HTTP

    fn agent(timeout: u64) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout))
            .build()
    }

    fn authed(&self, req: ureq::Request) -> ureq::Request {
        let key = self.st.lock().unwrap().api_key.clone();
        if key.is_empty() {
            req
        } else {
            req.set("Authorization", &format!("Bearer {key}"))
        }
    }

    pub fn health(&self) -> i64 {
        let base = self.st.lock().unwrap().base_url.clone();
        match self
            .authed(Self::agent(2).get(&format!("{base}/health")))
            .call()
        {
            Ok(r) => i64::from(r.status()),
            Err(ureq::Error::Status(c, _)) => i64::from(c),
            Err(_) => 0,
        }
    }

    pub fn get_json(&self, path: &str) -> Option<Value> {
        let base = self.st.lock().unwrap().base_url.clone();
        if base.is_empty() {
            return None;
        }
        let r = self
            .authed(Self::agent(3).get(&format!("{base}{path}")))
            .call()
            .ok()?;
        serde_json::from_str(&r.into_string().ok()?).ok()
    }

    // ------------------------------------------------------------------ status

    pub fn status(&self) -> Value {
        let s = self.st.lock().unwrap();
        let starting = s.state == "starting";
        let mut progress = if matches!(s.state.as_str(), "ready" | "attached") {
            1.0
        } else {
            0.0
        };
        if starting && s.load_total != 0 {
            progress = (s.load_read as f64 / s.load_total as f64).min(0.99);
        }
        let props = if s.props.is_object() {
            s.props.clone()
        } else {
            json!({})
        };
        let dgs = &props["default_generation_settings"];
        let plan = s.plan.as_ref().map(|p| &p.1);
        let mut model: Value = plan
            .map(|p| p.model.clone())
            .filter(|m| m.as_object().is_some_and(|o| !o.is_empty()))
            .unwrap_or(json!({}));
        if model.as_object().is_none_or(|o| o.is_empty())
            && props.as_object().is_some_and(|o| !o.is_empty())
        {
            let path = props["model_path"].as_str().unwrap_or("").to_string();
            let name = props["model_alias"]
                .as_str()
                .filter(|n| !n.is_empty())
                .map(String::from)
                .unwrap_or_else(|| {
                    Path::new(&path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .filter(|n| !n.is_empty())
                        .unwrap_or_else(|| "model".into())
                });
            model = json!({"name": name, "path": path});
        }
        let pick: Map<String, Value> = ["id", "name", "path", "catalog_id", "size"]
            .iter()
            .map(|k| (k.to_string(), model.get(*k).cloned().unwrap_or(Value::Null)))
            .collect();
        let truthy = |v: &Value| !(v.is_null() || v == &json!(0) || v == &json!(""));
        let n_ctx = if truthy(&dgs["n_ctx"]) {
            dgs["n_ctx"].clone()
        } else {
            props["n_ctx"].clone()
        };
        json!({
            "state": s.state, "phase": s.phase, "error": s.error, "hint": s.hint, "exit_code": s.exit_code, "reachable": s.reachable,
            "attached": s.state == "attached", "service": Value::Null,
            "base_url": if matches!(s.state.as_str(), "starting" | "ready" | "attached") { s.base_url.clone() } else { String::new() },
            "pid": s.proc.as_ref().map(|p| p.1), "started": s.started, "ready_at": s.ready_at,
            "load": {"read": s.load_read, "total": s.load_total, "progress": progress, "elapsed": if starting { json!(now() - s.started) } else { Value::Null }},
            "plan": plan.map(|p| p.public()), "model": pick, "n_ctx": n_ctx, "slots": props["total_slots"], "build": props.get("build_info").cloned().unwrap_or(json!("")),
            "cache": s.cache, "accept_logged": s.accept_logged, "rss": s.rss, "log_seq": s.seq,
        })
    }

    /// The pid of the engine Studio started, if one runs.
    pub fn proc_pid(&self) -> Option<u32> {
        self.st.lock().unwrap().proc.as_ref().map(|p| p.1)
    }

    pub fn cache_snapshot(&self) -> Map<String, Value> {
        self.st.lock().unwrap().cache.clone()
    }

    pub fn accept_logged(&self) -> Option<f64> {
        self.st.lock().unwrap().accept_logged
    }

    /// (state, base url, key) for the samplers and the chat proxy.
    pub fn target(&self) -> (String, String, String) {
        let s = self.st.lock().unwrap();
        (s.state.clone(), s.base_url.clone(), s.api_key.clone())
    }

    pub fn note_memory(&self, rss: Option<u64>) {
        self.st.lock().unwrap().rss = rss;
    }
}

/// Like Python's `returncode`: the exit code, or minus the signal number when a signal ended the process.
fn exit_code(st: &std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = st.signal() {
            return -sig;
        }
    }
    st.code().unwrap_or(-1)
}

/// The polite stop: SIGTERM lets llama-server finish its cleanup (and FLE save its expert profile).
fn ask_to_stop(pid: u32, child: &Arc<Mutex<Child>>) {
    #[cfg(unix)]
    {
        let _ = child; // SAFETY: plain signal to a child of ours
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        let _ = child.lock().unwrap().kill();
    }
}

/// (bytes the process has read so far, memory it holds) or `None` where this OS does not tell.
pub fn read_and_memory(pid: u32) -> (Option<u64>, Option<u64>) {
    #[cfg(target_os = "linux")]
    {
        let mut read = None;
        let mut rss = None;
        if let Ok(io) = std::fs::read_to_string(format!("/proc/{pid}/io")) {
            let v = |k: &str| {
                io.lines()
                    .find_map(|l| {
                        l.strip_prefix(&format!("{k}: "))
                            .and_then(|x| x.trim().parse::<u64>().ok())
                    })
                    .unwrap_or(0)
            };
            read = Some(v("rchar").max(v("read_bytes")));
        }
        if let Ok(st) = std::fs::read_to_string(format!("/proc/{pid}/status")) {
            rss = st
                .lines()
                .find_map(|l| {
                    l.strip_prefix("VmRSS:")
                        .and_then(|x| x.split_whitespace().next())
                        .and_then(|x| x.parse::<u64>().ok())
                })
                .map(|k| k * 1024);
        }
        (read, rss)
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: proc_pid_rusage fills the buffer we hand it
        unsafe {
            let mut info: libc::rusage_info_v2 = std::mem::zeroed();
            let r = libc::proc_pid_rusage(
                pid as i32,
                libc::RUSAGE_INFO_V2,
                &mut info as *mut _ as *mut libc::rusage_info_t,
            );
            if r == 0 {
                (Some(info.ri_diskio_bytesread), Some(info.ri_phys_footprint))
            } else {
                (None, None)
            }
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = pid;
        (None, None)
    }
}

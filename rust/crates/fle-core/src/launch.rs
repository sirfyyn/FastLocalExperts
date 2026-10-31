//! Turn a model, a profile and this machine into one llama-server command.
//! Port of `tools/fle/plan.py` and the engine part of `tools/fle/hardware.py`; the Python helper is the specification and the
//! tests compare both. The profile names are the same everywhere: off, auto, balanced, max, nvme, debug.
use crate::config::{self, as_bool, as_int};
use crate::gguf::{self, human_bytes, Model};
use crate::FleError;
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

pub fn profile_help(p: &str) -> &'static str {
    match p {
        "off" => "FLE features off; a plain llama.cpp baseline for comparisons.",
        "auto" => "Size the GPU expert cache and memory from this machine.",
        "balanced" => "Leave more free memory; a little slower, less likely to run out.",
        "max" => "Use most free GPU memory for the expert cache; needs a quiet machine.",
        "nvme" => "For a model larger than RAM: memory-map from disk and keep the cache small.",
        "debug" => "Auto settings plus timing and reuse diagnostics.",
        _ => "",
    }
}
fn profile_context(p: &str) -> i64 {
    match p {
        "max" => 65536,
        "nvme" => 16384,
        _ => 32768,
    }
}
fn cache_factor(p: &str) -> f64 {
    match p {
        "off" | "nvme" => 0.0,
        "balanced" => 0.70,
        _ => 1.0,
    }
}
/// GPU memory besides weights, KV, cache and margin: CUDA context, compute buffers, the MTP head (measured on the reference machine).
pub const ENGINE_OVERHEAD_MIB: u64 = 6749;
pub const DEFAULT_FIT_TARGET_MIB: i64 = 1115;

// ---------- what the machine has ----------

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: i64,
    pub name: String,
    pub memory_total: u64,
    pub memory_used: u64,
    pub memory_free: u64,
}

/// Is this port on this host free to listen on?
pub type PortCheck = Box<dyn Fn(&str, i64) -> bool>;

pub struct Host {
    pub cpu_physical: i64,
    pub cpu_logical: i64,
    pub mem_total: u64,
    pub mem_available: u64,
    pub gpus: Vec<GpuInfo>,
    pub windows: bool,
    pub linux: bool,
    pub port_free: PortCheck,
}

pub fn run_capture(
    exe: &Path,
    args: &[&str],
    timeout: u64,
    cwd: Option<&Path>,
    ld: Option<&Path>,
) -> (i32, String) {
    let mut c = Command::new(exe);
    c.args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());
    if let Some(d) = cwd {
        c.current_dir(d);
    }
    if let Some(dir) = ld {
        let old = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        c.env(
            "LD_LIBRARY_PATH",
            if old.is_empty() {
                dir.display().to_string()
            } else {
                format!("{}:{old}", dir.display())
            },
        );
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    let Ok(mut child) = c.spawn() else {
        return (127, String::new());
    };
    let (mut so, mut se) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let t1 = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = std::io::Read::read_to_end(&mut so, &mut b);
        b
    });
    let t2 = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = std::io::Read::read_to_end(&mut se, &mut b);
        b
    });
    let start = Instant::now();
    let code = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st.code().unwrap_or(1),
            Ok(None) if start.elapsed() > Duration::from_secs(timeout) => {
                let _ = child.kill();
                let _ = child.wait();
                break 127;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => break 127,
        }
    };
    let mut out = String::from_utf8_lossy(&t1.join().unwrap_or_default()).to_string();
    out.push_str(&String::from_utf8_lossy(&t2.join().unwrap_or_default()));
    (code, out)
}

impl Host {
    /// What `hardware.py` reads: /proc on Linux, sysctl on macOS, nvidia-smi for GPUs.
    pub fn detect() -> Host {
        let (mut total, mut avail) = (0u64, 0u64);
        let logical = std::thread::available_parallelism()
            .map(|n| n.get() as i64)
            .unwrap_or(1);
        let mut physical = 0i64;
        if cfg!(target_os = "macos") {
            let sysctl = |k: &str| {
                run_capture(Path::new("sysctl"), &["-n", k], 3, None, None)
                    .1
                    .trim()
                    .parse::<i64>()
                    .unwrap_or(0)
            };
            total = sysctl("hw.memsize") as u64;
            avail = total;
            physical = sysctl("hw.physicalcpu");
        } else if cfg!(target_os = "linux") {
            if let Ok(t) = std::fs::read_to_string("/proc/meminfo") {
                let (mut free, mut a) = (0u64, None);
                for l in t.lines() {
                    let kb = |r: &str| {
                        r.split_whitespace()
                            .next()
                            .and_then(|x| x.parse::<u64>().ok())
                            .unwrap_or(0)
                            * 1024
                    };
                    if let Some(r) = l.strip_prefix("MemTotal:") {
                        total = kb(r)
                    } else if let Some(r) = l.strip_prefix("MemAvailable:") {
                        a = Some(kb(r))
                    } else if let Some(r) = l.strip_prefix("MemFree:") {
                        free = kb(r)
                    }
                }
                avail = a.unwrap_or(free);
            }
            if let Ok(t) = std::fs::read_to_string("/proc/cpuinfo") {
                let mut pairs = std::collections::BTreeSet::new();
                let (mut ph, mut co): (Option<String>, Option<String>) = (None, None);
                for l in t.lines() {
                    if l.starts_with("physical id") {
                        ph = l.split_once(':').map(|x| x.1.trim().to_string());
                    } else if l.starts_with("core id") {
                        co = l.split_once(':').map(|x| x.1.trim().to_string());
                    } else if l.trim().is_empty() && ph.is_some() && co.is_some() {
                        pairs.insert((ph.take().unwrap(), co.take().unwrap()));
                    }
                }
                physical = pairs.len() as i64;
            }
        }
        if physical == 0 {
            physical = logical;
        }
        Host {
            cpu_physical: physical,
            cpu_logical: logical,
            mem_total: total,
            mem_available: avail,
            gpus: nvidia_gpus(),
            windows: cfg!(windows),
            linux: cfg!(target_os = "linux"),
            port_free: Box::new(port_free),
        }
    }
}

/// Output of a short command, only when it succeeded (`system._run` in Studio); empty when it is missing, fails or hangs.
pub fn run_ok(exe: &str, args: &[&str], timeout: u64) -> String {
    let (code, out) = run_capture(Path::new(exe), args, timeout, None, None);
    if code == 0 {
        out
    } else {
        String::new()
    }
}

/// stdout and stderr of a short command, trimmed; empty when it cannot run.
pub fn run_text(exe: &str, args: &[&str]) -> String {
    run_capture(Path::new(exe), args, 5, None, None)
        .1
        .trim()
        .to_string()
}

pub fn nvidia_gpus() -> Vec<GpuInfo> {
    let Some(smi) = which("nvidia-smi") else {
        return vec![];
    };
    let (code, out) = run_capture(
        &smi,
        &[
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        8,
        None,
        None,
    );
    if code != 0 {
        return vec![];
    }
    out.lines()
        .filter_map(|line| {
            let p: Vec<&str> = line.split(',').map(str::trim).collect();
            if p.len() < 5 {
                return None;
            }
            let mib = |s: &str| s.parse::<f64>().ok().map(|f| f as u64 * MIB);
            Some(GpuInfo {
                index: p[0].parse().ok()?,
                name: p[1].to_string(),
                memory_total: mib(p[2])?,
                memory_used: mib(p[3])?,
                memory_free: mib(p[4])?,
            })
        })
        .collect()
}

pub fn port_free(host: &str, port: i64) -> bool {
    let Ok(port) = u16::try_from(port) else {
        return false;
    };
    std::net::TcpListener::bind((host, port)).is_ok()
}

pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        vec![".exe".into(), ".cmd".into(), ".bat".into(), String::new()]
    } else {
        vec![String::new()]
    };
    std::env::split_paths(&path).find_map(|d| {
        exts.iter()
            .map(|e| d.join(format!("{name}{e}")))
            .find(|p| p.is_file())
    })
}

// ---------- the engine ----------

/// `hardware.detect_engine`: run the binary three times and read what it says. Keys and order as in Python.
pub fn detect_engine(path: &str) -> Map<String, Value> {
    let p = expand(path);
    let mut m = Map::new();
    if !p.is_file() {
        m.insert("ok".into(), json!(false));
        m.insert("path".into(), json!(p.to_string_lossy()));
        m.insert(
            "problem".into(),
            json!(format!("No file at {}", p.display())),
        );
        return m;
    }
    let dir = p.parent().map(Path::to_path_buf).unwrap_or_default();
    let ld = if cfg!(windows) {
        None
    } else {
        Some(dir.as_path())
    };
    let (vc, vtext) = run_capture(&p, &["--version"], 60, Some(&dir), ld);
    let (_, help) = run_capture(&p, &["--help"], 60, Some(&dir), ld);
    let (_, devs) = run_capture(&p, &["--list-devices"], 15, Some(&dir), ld);
    let ver = parse_version(&vtext);
    let flavor = if help.contains("--moe-expert-cache-inserts") {
        "fle"
    } else {
        "upstream"
    };
    let blob = format!("{vtext} {help} {devs}").to_lowercase();
    let backend = if blob.contains("cuda") {
        "cuda"
    } else if blob.contains("metal") {
        "metal"
    } else if blob.contains("vulkan") {
        "vulkan"
    } else if blob.contains("rocm") || blob.contains("hip") {
        "rocm"
    } else {
        "cpu"
    };
    let ok = vc == 0 && ver.is_some();
    m.insert("ok".into(), json!(ok));
    m.insert("path".into(), json!(p.to_string_lossy()));
    m.insert("version".into(), json!(ver.unwrap_or_default()));
    m.insert("backend".into(), json!(backend));
    m.insert("flavor".into(), json!(flavor));
    m.insert(
        "has_fit_target".into(),
        json!(help.contains("--fit-target")),
    );
    m.insert("has_load_mode".into(), json!(help.contains("--load-mode")));
    m.insert(
        "has_cmoe".into(),
        json!(help.contains("-cmoe") || help.contains("--cpu-moe")),
    );
    m.insert("has_spec_mtp".into(), json!(help.contains("--spec-type")));
    let problem = if ok {
        String::new()
    } else {
        vtext
            .trim()
            .lines()
            .last()
            .unwrap_or("no output")
            .chars()
            .take(300)
            .collect()
    };
    m.insert("problem".into(), json!(problem));
    m
}

/// `version:\s*(\S+)(?:\s*\(([0-9a-f]+)\))?` -> "6000" or "6000 (abc123)".
fn parse_version(text: &str) -> Option<String> {
    let at = text.find("version:")?;
    let rest = text[at + 8..].trim_start();
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let v = &rest[..end];
    if v.is_empty() {
        return None;
    }
    let after = rest[end..].trim_start();
    if let Some(r) = after.strip_prefix('(') {
        if let Some(close) = r.find(')') {
            let h = &r[..close];
            if !h.is_empty() && h.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
                return Some(format!("{v} ({h})"));
            }
        }
    }
    Some(v.to_string())
}

fn expand(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(r) => dirs::home_dir().unwrap_or_default().join(r),
        None => PathBuf::from(p),
    }
}

/// `hardware.engine_candidates`, in the order of the Python helper (the explicit one is added by the caller).
pub fn engine_candidates(installed: &[PathBuf]) -> Vec<String> {
    let mut c: Vec<String> = vec![];
    if let Ok(e) = std::env::var("FLE_ENGINE") {
        if !e.trim().is_empty() {
            c.push(e.trim().to_string());
        }
    }
    if let Some(p) = which("llama-server") {
        c.push(p.to_string_lossy().to_string());
    }
    c.extend(installed.iter().map(|p| p.to_string_lossy().to_string()));
    let mut seen = vec![];
    c.into_iter()
        .map(|s| expand(&s).to_string_lossy().to_string())
        .filter(|s| {
            if seen.contains(s) {
                false
            } else {
                seen.push(s.clone());
                true
            }
        })
        .collect()
}

/// The places `hardware.find_engine` looks, plus the engines `fle` installed itself.
pub fn find_engine(explicit: &str, installed: &[PathBuf]) -> String {
    let mut c: Vec<String> = vec![];
    if !explicit.is_empty() {
        c.push(explicit.to_string());
    }
    if let Ok(e) = std::env::var("FLE_ENGINE") {
        if !e.trim().is_empty() {
            c.push(e.trim().to_string());
        }
    }
    if let Some(p) = which("llama-server") {
        c.push(p.to_string_lossy().to_string());
    }
    c.extend(installed.iter().map(|p| p.to_string_lossy().to_string()));
    c.into_iter()
        .map(|s| expand(&s))
        .find(|p| p.is_file())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
}

// ---------- text helpers with the exact behaviour of the Python ones ----------

#[allow(clippy::manual_is_multiple_of)] // `is_multiple_of` needs a newer Rust than we promise
pub fn human_int(n: i64) -> String {
    let s = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// `shlex.quote`
pub fn shquote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    if s.chars().all(|c| {
        c.is_alphanumeric()
            || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
    }) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

fn is_number(t: &str) -> bool {
    let t = t.strip_prefix('-').unwrap_or(t);
    let (a, b) = match t.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (t, None),
    };
    let digits = |x: &str| !x.is_empty() && x.chars().all(|c| c.is_ascii_digit());
    digits(a) && b.is_none_or(digits)
}
fn is_flag(t: &str) -> bool {
    t.starts_with('-') && t.len() > 1 && !is_number(t)
}
/// A flag together with its value, or a flag alone; order is kept.
pub fn flag_groups(args: &[String]) -> Vec<Vec<String>> {
    let mut g = vec![];
    let mut i = 0;
    while i < args.len() {
        let nxt = args.get(i + 1);
        if is_flag(&args[i]) && nxt.is_some_and(|n| !is_flag(n)) {
            g.push(vec![args[i].clone(), args[i + 1].clone()]);
            i += 2;
        } else {
            g.push(vec![args[i].clone()]);
            i += 1;
        }
    }
    g
}

fn generic_name(n: &str) -> bool {
    let l = n.to_lowercase();
    let digit_after = |p: &str| {
        l.strip_prefix(p)
            .and_then(|r| r.chars().next())
            .is_some_and(|c| c.is_ascii_digit())
    };
    matches!(l.as_str(), "bf16" | "f16" | "f32")
        || digit_after("q")
        || digit_after("iq")
        || digit_after("mxfp")
}

fn stable_alias(model: &Model, model_path: &str) -> String {
    let name = model.name.trim();
    if !name.is_empty() && !generic_name(name) {
        return name.to_string();
    }
    let p = if model_path.is_empty() {
        model.first.clone()
    } else {
        PathBuf::from(model_path)
    };
    let stem = p
        .file_stem()
        .map(|s| s.to_string_lossy().trim().to_string())
        .unwrap_or_default();
    if stem.is_empty() {
        "model".into()
    } else {
        stem
    }
}

pub fn profile_path(model: &Model, model_path: &str) -> PathBuf {
    use sha1::{Digest, Sha1};
    let alias = stable_alias(model, model_path);
    let mut safe = String::new();
    let mut gap = false;
    for c in alias.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
            if gap && !safe.is_empty() {
                safe.push('-');
            }
            gap = false;
            safe.push(c);
        } else {
            gap = true;
        }
    }
    if gap {
        safe.push('-');
    }
    let mut safe = safe.trim_matches(|c| c == '-' || c == '.').to_string();
    if safe.is_empty() {
        safe = "model".into();
    }
    let name = model.name.trim();
    if name.is_empty() || generic_name(name) {
        let p = if model_path.is_empty() {
            model.first.clone()
        } else {
            expand(model_path)
        };
        let resolved = std::fs::canonicalize(&p)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string();
        let resolved = resolved
            .strip_prefix(r"\\?\")
            .unwrap_or(&resolved)
            .to_string();
        safe = format!(
            "{safe}-{}",
            &hex::encode(Sha1::digest(resolved.as_bytes()))[..8]
        );
    }
    config::config_path(None)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default()
        .join("profiles")
        .join(format!("{safe}.bin"))
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => true, // settings.get("mtp", True)
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(_) => true,
    }
}

// ---------- the plan ----------

pub struct Plan {
    pub profile: String,
    pub engine: Map<String, Value>,
    pub binary: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub host: String,
    pub port: i64,
    pub model: Model,
    pub sidecars: [Option<String>; 3], // mtp_head, mtp_vocab, mmproj
    pub context: i64,
    pub slots: i64,
    pub kv_unified: bool,
    pub context_per_slot: i64,
    pub spec: String,
    pub alias: String,
    pub threads: i64,
    pub fit_target_mib: i64,
    pub cache_mib: i64,
    pub fallback: bool,
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
    pub estimate: Map<String, Value>,
}

impl Plan {
    pub fn command(&self) -> String {
        if cfg!(windows) {
            let mut lines: Vec<String> = self
                .env
                .iter()
                .map(|(k, v)| format!("set {k}={v}"))
                .collect();
            let mut all = vec![self.binary.clone()];
            all.extend(self.args.iter().cloned());
            lines.push(all.join(" "));
            return lines.join("\r\n");
        }
        let env = self
            .env
            .iter()
            .map(|(k, v)| format!("{k}={}", shquote(v)))
            .collect::<Vec<_>>()
            .join(" ");
        let mut all = vec![self.binary.clone()];
        all.extend(self.args.iter().cloned());
        let cmd = all.iter().map(|x| shquote(x)).collect::<Vec<_>>().join(" ");
        if env.is_empty() {
            cmd
        } else {
            format!("{env} \\\n  {cmd}")
        }
    }

    /// One environment variable per line, then the binary and one flag (with its value) per line.
    pub fn command_block(&self, oneline: bool) -> String {
        if oneline || cfg!(windows) {
            return self.command();
        }
        let mut lines: Vec<String> = self
            .env
            .iter()
            .map(|(k, v)| format!("{k}={}", shquote(v)))
            .collect();
        lines.push(shquote(&self.binary));
        for g in flag_groups(&self.args) {
            lines.push(format!(
                "  {}",
                g.iter().map(|x| shquote(x)).collect::<Vec<_>>().join(" ")
            ));
        }
        lines.join(" \\\n")
    }

    pub fn as_json(&self) -> Value {
        // the Python `Model.summary()` has no sidecars; the CLI adds them for `inspect` only
        let mut model = self.model.summary();
        if let Some(m) = model.as_object_mut() {
            m.shift_remove("sidecars");
        }
        let env: Map<String, Value> = self
            .env
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect();
        let [head, vocab, mmproj] = &self.sidecars;
        json!({
            "profile": self.profile, "profile_help": profile_help(&self.profile), "engine": self.engine, "binary": self.binary,
            "args": self.args, "env": env, "command": self.command(), "host": self.host, "port": self.port, "context": self.context,
            "slots": self.slots, "kv_unified": self.kv_unified, "context_per_slot": self.context_per_slot, "spec": self.spec,
            "alias": self.alias, "fallback": self.fallback, "threads": self.threads, "fit_target_mib": self.fit_target_mib,
            "cache_mib": self.cache_mib, "model": model,
            "sidecars": {"mtp_head": head, "mtp_vocab": vocab, "mmproj": mmproj},
            "notes": self.notes, "warnings": self.warnings, "estimate": self.estimate,
        })
    }

    /// The `fle plan` text (`_print_plan` in cli.py), without colour.
    pub fn text(&self, oneline: bool) -> String {
        let mut s = format!(
            "profile:  {}  ({})\n",
            self.profile,
            profile_help(&self.profile)
        );
        s += &format!(
            "model:    {} ({})\n",
            self.model.name,
            human_bytes(self.model.size)
        );
        s += &format!("alias:    {}\n", self.alias);
        let f = |k: &str| {
            self.engine
                .get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("None")
                .to_string()
        };
        s += &format!(
            "engine:   {} ({}, {})\n",
            self.binary,
            f("flavor"),
            f("backend")
        );
        let note = format!(
            "{} slot(s), {}",
            self.slots,
            if self.kv_unified {
                "unified KV"
            } else {
                "separate KV"
            }
        );
        s += &format!(
            "context:  {} tokens total ({} per slot; {note})\n",
            human_int(self.context),
            human_int(self.context_per_slot)
        );
        s += &format!(
            "spec:     {}\nthreads:  {}\ncache:    {} MiB\nfit:      {} MiB\nhost:     {}:{}\n\n",
            self.spec,
            self.threads,
            human_int(self.cache_mib),
            human_int(self.fit_target_mib),
            self.host,
            self.port
        );
        s += &self.command_block(oneline);
        s += "\n";
        if !self.notes.is_empty() {
            s += "\nnotes:\n";
            for n in &self.notes {
                s += &format!("  - {n}\n");
            }
        }
        if !self.warnings.is_empty() {
            s += "\nwarnings:\n";
            for w in &self.warnings {
                s += &format!("  - {w}\n");
            }
        }
        s
    }
}

/// What `build` gets besides settings and model: the flags that are not settings.
#[derive(Default, Clone)]
pub struct Options {
    pub engine_path: String,
    pub model_path: String,
    pub extra_args: Vec<String>,
    pub env_override: Vec<(String, String)>,
    pub gpu_ids: String,
    pub no_mtp: bool,
    pub mtp_head: String,
    pub mtp_vocab: String,
    pub spec: String,
    pub installed_engines: Vec<PathBuf>,
}

fn threads(requested: i64, h: &Host) -> i64 {
    if requested > 0 {
        return requested;
    }
    let physical = if h.cpu_physical > 0 {
        h.cpu_physical
    } else {
        1
    };
    let logical = if h.cpu_logical > 0 {
        h.cpu_logical
    } else {
        physical
    };
    if logical > physical {
        physical.max(1)
    } else {
        (physical - 1).max(1)
    }
}

pub fn context(profile: &str, model: &Model, requested: i64) -> i64 {
    let mut ctx = if requested > 0 {
        requested
    } else {
        profile_context(profile)
    };
    let cap = model.context_length().unwrap_or(0) as i64;
    if cap > 0 {
        ctx = ctx.min(cap);
    }
    ctx.max(1024)
}

fn pinned(profile: &str, setting: &str, model: &Model, h: &Host) -> bool {
    if profile == "off" || profile == "nvme" {
        return false;
    }
    match setting {
        "on" => true,
        "off" => false,
        _ => {
            (h.windows || h.linux)
                && h.mem_total as i128 - model.expert_bytes as i128 >= GIB as i128 * 24
        }
    }
}

fn extra_parallel(args: &[String]) -> i64 {
    for (i, a) in args.iter().enumerate() {
        if a == "--parallel" && i + 1 < args.len() {
            return as_int(&json!(args[i + 1]), 0);
        }
        if let Some(v) = a.strip_prefix("--parallel=") {
            return as_int(&json!(v), 0);
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn cache_budget(
    profile: &str,
    requested: i64,
    model: &Model,
    ctx: i64,
    fit_target: i64,
    gpus: &[&GpuInfo],
    notes: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> (i64, Map<String, Value>) {
    let free: u64 = gpus.iter().map(|g| g.memory_free).sum();
    let gpu_weights = model.gpu_bytes;
    let kv = (model.kv_bytes_per_token("q8_0", "q8_0") * ctx as f64) as i64;
    let overhead = (ENGINE_OVERHEAD_MIB * MIB) as i64;
    let available =
        (free as i64 - gpu_weights as i64 - kv - overhead - fit_target * MIB as i64).max(0);
    let base_mib = available / MIB as i64;
    let auto_mib = (base_mib as f64 * cache_factor(profile)) as i64;
    let has_experts = model.experts().unwrap_or(0) > 0;
    let mut cache = if profile == "off" {
        0
    } else if requested > 0 {
        if requested > base_mib {
            warnings.push(format!("The requested cache of {} MiB is larger than the {} MiB left after weights and KV cache; loading may fail.", human_int(requested), human_int(base_mib)));
        }
        requested
    } else if profile == "nvme" {
        2048.min(auto_mib)
    } else {
        auto_mib
    };
    if !has_experts && cache > 0 {
        if requested > 0 {
            warnings.push(format!(
                "--cache-mib {} is ignored: this is a dense model with no experts to cache.",
                human_int(requested)
            ));
        }
        cache = 0;
    }
    if profile != "off" && has_experts && base_mib <= 0 && requested <= 0 {
        warnings.push(
            "No GPU memory is left for the expert cache; experts will run on the CPU.".into(),
        );
    }
    if profile != "off" && profile != "nvme" && cache > 0 {
        notes.push(format!(
            "GPU expert cache {cache} MiB: {} free, minus {} weights, {} KV at {} tokens, {} overhead and {} MiB margin.",
            human_bytes(free), human_bytes(gpu_weights), human_bytes(kv as u64), human_int(ctx), human_bytes(overhead as u64), human_int(fit_target)
        ));
    }
    if profile == "nvme" {
        notes.push(if has_experts {
            "NVMe profile: the model is memory-mapped, so the filesystem can page experts in on demand.".into()
        } else {
            "NVMe profile: nvme is meant for models larger than RAM. The model is memory-mapped and starts without warm-up, so the filesystem pages weights in on demand. A dense model has no experts to stream: the layers that stay off the GPU are read in full for every token, so one larger than RAM plus VRAM will be slow, and one that fits is better off with auto.".into()
        });
    }
    let mut e = Map::new();
    for (k, v) in [
        ("vram_free", free as i64),
        ("gpu_weights", gpu_weights as i64),
        ("kv_cache", kv),
        ("overhead", overhead),
        ("fit_target", fit_target * MIB as i64),
        ("available", available),
        ("base_mib", base_mib),
    ] {
        e.insert(k.into(), json!(v));
    }
    (cache.max(0), e)
}

fn err(m: impl Into<String>, h: impl Into<String>) -> anyhow::Error {
    FleError::new(m, h).into()
}

pub fn build(settings: &Map<String, Value>, model: &Model, o: &Options, h: &Host) -> Result<Plan> {
    let sget = |k: &str| settings.get(k);
    let sstr = |k: &str| match sget(k) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(v) => v.to_string(),
    };
    let profile = {
        let p = sstr("profile").to_lowercase();
        if p.is_empty() {
            "auto".to_string()
        } else {
            p
        }
    };
    if !config::PROFILES.contains(&profile.as_str()) {
        return Err(err(
            format!("Unknown profile '{profile}'"),
            format!("Use one of: {}.", config::PROFILES.join(", ")),
        ));
    }
    let mut path = if o.engine_path.is_empty() {
        sstr("engine")
    } else {
        o.engine_path.clone()
    };
    if path.is_empty() {
        path = find_engine("", &o.installed_engines);
    }
    if path.is_empty() {
        return Err(err("No llama-server found.", "Pass --engine /path/to/llama-server, set engine in `fle config`, or build this checkout."));
    }
    let engine = detect_engine(&path);
    let eb = |k: &str| engine.get(k).and_then(Value::as_bool).unwrap_or(false);
    let es = |k: &str| {
        engine
            .get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    if !eb("ok") {
        let prob = es("problem");
        return Err(err(
            format!(
                "The engine does not start: {}",
                if prob.is_empty() { path.clone() } else { prob }
            ),
            "Check the path and its libraries.",
        ));
    }
    let (mut notes, mut warnings): (Vec<String>, Vec<String>) = (vec![], vec![]);
    if profile == "off" {
        notes.push("FLE features are off. This is the reproducibility baseline.".into());
    }
    let fle = es("flavor") == "fle";
    if !fle && profile != "off" {
        return Err(err(
            "This is not an FLE engine.",
            "Choose an FLE build, or use --profile off for a plain llama.cpp baseline.",
        ));
    }
    if fle && profile == "off" {
        notes.push(
            "The binary is FLE, but every FLE_* setting is left out for the baseline.".into(),
        );
    }
    let host = {
        let s = sstr("host");
        if s.is_empty() {
            "127.0.0.1".to_string()
        } else {
            s
        }
    };
    let port = sget("port").map(|v| as_int(v, 8080)).unwrap_or(8080);
    let req_ctx = sget("context").map(|v| as_int(v, 0)).unwrap_or(0);
    let ctx = context(&profile, model, req_ctx);
    if model
        .context_length()
        .is_some_and(|c| c > 0 && ctx == c as i64)
        && req_ctx > ctx
    {
        notes.push(format!(
            "Context capped at the model maximum of {} tokens.",
            human_int(ctx)
        ));
    }
    let nthreads = threads(sget("threads").map(|v| as_int(v, 0)).unwrap_or(0), h);
    let mut fit_target = {
        let f = sget("fit_target_mib").map(|v| as_int(v, 0)).unwrap_or(0);
        if f == 0 {
            DEFAULT_FIT_TARGET_MIB
        } else {
            f
        }
    };
    let has_experts = model.experts().unwrap_or(0) > 0;
    if profile == "balanced" && fit_target == DEFAULT_FIT_TARGET_MIB {
        fit_target = 1536;
        notes.push(if has_experts { "Balanced keeps a 1536 MiB fit margin (1115 MiB in auto) and 30 percent of the cache free.".into() } else { "Balanced keeps a 1536 MiB fit margin (1115 MiB in auto); a dense model has no expert cache.".into() });
    } else if profile == "max" && fit_target == DEFAULT_FIT_TARGET_MIB {
        fit_target = 512;
        notes.push(
            "Max uses a 512 MiB fit margin instead of 1115 MiB; stop other GPU work first.".into(),
        );
    } else if profile == "nvme" && fit_target == DEFAULT_FIT_TARGET_MIB {
        fit_target = 1024;
    }
    if profile == "off" {
        fit_target = 0;
    }
    if h.gpus.is_empty() && fle {
        return Err(err(
            "The FLE engine needs an NVIDIA GPU, and none was found.",
            "Install the driver, or use an upstream CPU build with --profile off.",
        ));
    }
    let wanted: Vec<String> = o.gpu_ids.split(',').map(|x| x.trim().to_string()).collect();
    let chosen: Vec<&GpuInfo> = h
        .gpus
        .iter()
        .filter(|g| o.gpu_ids.is_empty() || wanted.contains(&g.index.to_string()))
        .collect();
    if !o.gpu_ids.is_empty() && chosen.is_empty() {
        return Err(err(
            format!("No GPU with index {}.", o.gpu_ids),
            "Run nvidia-smi to see the indices.",
        ));
    }
    let requested_cache = sget("gpu_cache_mib").map(|v| as_int(v, 0)).unwrap_or(0);
    let (cache_mib, mut estimate) = cache_budget(
        &profile,
        requested_cache,
        model,
        ctx,
        fit_target,
        &chosen,
        &mut notes,
        &mut warnings,
    );
    for (k, v) in [
        ("model_size", model.size as i64),
        ("model_total", model.size as i64),
        ("expert_bytes", model.expert_bytes as i64),
        ("input_bytes", model.input_bytes as i64),
        ("ram_total", h.mem_total as i64),
        ("ram_available", h.mem_available as i64),
        ("context", ctx),
        ("threads", nthreads),
        ("cache_mib", cache_mib),
    ] {
        estimate.insert(k.into(), json!(v));
    }
    if !model.missing.is_empty() || !model.incomplete.is_empty() {
        warnings.push(format!("{} model part(s) are missing or incomplete; the size and cache estimate are incomplete.", model.missing.len() + model.incomplete.len()));
    }
    let gpu_total: u64 = chosen.iter().map(|g| g.memory_total).sum();
    if model.size > h.mem_total + gpu_total {
        warnings.push(if has_experts {
            "The model is larger than RAM plus VRAM. Use --profile nvme to stream experts from disk, or add memory.".into()
        } else {
            "The model is larger than RAM plus VRAM. A dense model has no experts to stream; --profile nvme memory-maps it from disk, but the layers that stay off the GPU are read in full for every token, so expect it to be slow. A smaller quantisation or more memory is the real fix.".into()
        });
    }
    if model.expert_bytes > 0 && h.mem_available > 0 && model.expert_bytes > h.mem_available {
        warnings.push(format!(
            "Experts need {} in RAM, but only {} is free.",
            human_bytes(model.expert_bytes),
            human_bytes(h.mem_available)
        ));
    }

    let side = gguf::sidecars(&model.first);
    let sc = |k: &str| {
        side.get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let head = if o.mtp_head.is_empty() {
        sc("mtp_head")
    } else {
        o.mtp_head.clone()
    };
    let vocab = if o.mtp_vocab.is_empty() {
        sc("mtp_vocab")
    } else {
        o.mtp_vocab.clone()
    };
    let requested_spec = {
        let s = if !o.spec.is_empty() {
            o.spec.clone()
        } else if !sstr("spec").is_empty() {
            sstr("spec")
        } else {
            "auto".into()
        };
        s.to_lowercase()
    };
    if !config::SPECS.contains(&requested_spec.as_str()) {
        return Err(err(
            format!("Unknown speculation mode '{requested_spec}'"),
            format!("Use one of: {}.", config::SPECS.join(", ")),
        ));
    }
    let mtp_setting = sget("mtp");
    let mtp_switched_off = o.no_mtp || !as_bool(mtp_setting.unwrap_or(&json!(true)), true);
    if mtp_switched_off && requested_spec == "mtp" {
        return Err(err("--spec mtp and a switched-off MTP (--no-mtp, or mtp = false in the config or FLE_MTP) contradict each other.", "Pick one: drop --no-mtp / set mtp = true, or use --spec auto."));
    }
    if (requested_spec == "ngram-mod" || requested_spec == "ngram-simple") && profile == "off" {
        return Err(err(
            "The off profile is the reproducibility baseline; it does not add speculation.",
            format!("Use --profile auto with --spec {requested_spec}, or keep --profile off."),
        ));
    }
    if head.is_empty() != vocab.is_empty() {
        warnings.push("MTP needs both a head file and a vocabulary file; MTP stays off.".into());
    }
    let (mut use_mtp, mut ngram) = (false, String::new());
    let (has_pair, setting_on) = (!head.is_empty() && !vocab.is_empty(), truthy(mtp_setting));
    match requested_spec.as_str() {
        "none" => notes.push("Speculation is off.".into()),
        "mtp" => {
            if !has_pair {
                return Err(err("MTP was requested, but no MTP sidecar files (head and vocabulary) were found next to the model.", format!("Run `fle fetch-sidecars {}` to download them when a source is known, or use --spec auto / --spec ngram-mod.", shquote(&model.first.to_string_lossy()))));
            }
            if profile == "off" || profile == "nvme" {
                warnings.push(format!("MTP is not used in the {profile} profile."));
            } else {
                use_mtp = true;
            }
        }
        "ngram-mod" | "ngram-simple" => {
            if !eb("has_spec_mtp") {
                return Err(err(format!("This engine does not support --spec-type, so {requested_spec} cannot be used."), "Use an FLE build or an upstream llama.cpp with speculative decoding."));
            }
            ngram = requested_spec.clone();
            if profile == "nvme" {
                notes.push(
                    "Ngram speculation is on in the NVMe profile; it needs no draft model.".into(),
                );
            }
        }
        _ => {
            use_mtp = setting_on && !o.no_mtp && profile != "off" && profile != "nvme" && has_pair;
            if profile == "nvme" && !head.is_empty() {
                notes.push("MTP stays off in the NVMe profile; it needs memory that the expert tier can use.".into());
            } else if setting_on
                && !o.no_mtp
                && head.is_empty()
                && profile != "off"
                && profile != "nvme"
            {
                notes.push("MTP is off: no MTP head and vocabulary file were found next to the model. Use --spec ngram-mod for self-speculation.".into());
            }
        }
    }
    let effective_spec = if use_mtp {
        "mtp".to_string()
    } else if !ngram.is_empty() {
        ngram.clone()
    } else {
        "none".to_string()
    };

    let model_arg = if o.model_path.is_empty() {
        model.first.to_string_lossy().to_string()
    } else {
        o.model_path.clone()
    };
    let mut args: Vec<String> = ["-m", &model_arg, "-c", &ctx.to_string(), "--jinja"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let add = |a: &mut Vec<String>, xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    let moe = has_experts;
    if fle {
        add(&mut args, &["-ngl", "99"]);
        if moe {
            add(&mut args, &["-cmoe"]);
        }
    } else if model.is_moe() {
        add(&mut args, &["--cpu-moe"]);
    }
    add(
        &mut args,
        &[
            "--host",
            &host,
            "--port",
            &port.to_string(),
            "--metrics",
            "-t",
            &nthreads.to_string(),
        ],
    );
    if profile != "off" {
        add(&mut args, &["-ctk", "q8_0", "-ctv", "q8_0"]);
    }
    if fle && profile != "off" {
        add(
            &mut args,
            &["-fit", "on", "--fit-target", &fit_target.to_string()],
        );
        if moe {
            add(&mut args, &["--moe-expert-cache-inserts", "1"]);
        }
    }
    let log_verbosity = engine
        .get("log_verbosity")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if profile == "debug" && log_verbosity {
        add(&mut args, &["--log-verbosity", "4"]);
    }
    if use_mtp {
        add(
            &mut args,
            &[
                "-md",
                &head,
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "2",
                "--spec-draft-p-min",
                "0.5",
                "--no-spec-draft-backend-sampling",
            ],
        );
        if moe {
            add(&mut args, &["-cmoed"]);
        }
    }
    if !ngram.is_empty() {
        add(&mut args, &["--spec-type", &ngram]);
        notes.push(format!("Self-speculation: {ngram}. It drafts from the prompt/context; acceptance is prompt-dependent."));
    }
    let pin = pinned(
        &profile,
        &{
            let s = sstr("pinned_load");
            if s.is_empty() {
                "auto".into()
            } else {
                s
            }
        },
        model,
        h,
    );
    if pin {
        add(&mut args, &["--load-mode", "none"]);
        notes.push(if moe { "Experts are read into pinned host memory (--load-mode none); this needs the expert bytes as free RAM.".into() } else { "Model weights are read into pinned host memory (--load-mode none); this needs the model size as free RAM.".into() });
    }
    if profile == "nvme" {
        add(&mut args, &["--no-warmup"]);
    }

    let mut env: Vec<(String, String)> = vec![];
    let set = |env: &mut Vec<(String, String)>, k: &str, v: String| match env
        .iter_mut()
        .find(|(a, _)| a == k)
    {
        Some(e) => e.1 = v,
        None => env.push((k.to_string(), v)),
    };
    if !o.gpu_ids.is_empty() {
        set(&mut env, "CUDA_VISIBLE_DEVICES", o.gpu_ids.clone());
    }
    if fle && profile != "off" {
        for (k, v) in [
            ("FLE_PREFILL_LEAN", "1"),
            ("FLE_HC_FUSED", "1"),
            ("FLE_SLOT_EARLY_SAVE", "1"),
        ] {
            set(&mut env, k, v.into());
        }
        if moe {
            set(&mut env, "FLE_MOE_CACHE_BUDGET_FIT", "1".into());
            set(
                &mut env,
                "FLE_MOE_PROFILE",
                profile_path(model, &model_arg)
                    .to_string_lossy()
                    .to_string(),
            );
            set(&mut env, "FLE_MOE_PROFILE_SAVE_SEC", "600".into());
            if cache_mib > 0 {
                set(&mut env, "FLE_MOE_CACHE_BUDGET_MIB", cache_mib.to_string());
                set(&mut env, "FLE_MC_OVERLAP", "2".into());
            }
            if matches!(profile.as_str(), "auto" | "max" | "debug") {
                set(&mut env, "FLE_MOE_CACHE_BIAS", "0.001".into());
            }
            if profile == "balanced" || profile == "max" {
                set(&mut env, "FLE_MOE_ADMIT_FREQ", "2000".into());
                set(&mut env, "FLE_MOE_EVICT_OLDEST", "32".into());
            }
        }
        if profile == "debug" {
            set(&mut env, "FLE_TIME_PROFILE", "20000".into());
            set(&mut env, "FLE_REUSE_REASON", "1".into());
            set(&mut env, "FLE_NAN_GUARD", "1".into());
        }
        if use_mtp {
            set(&mut env, "FLE_MTP_VOCAB", vocab.clone());
            set(&mut env, "FLE_MTP_VOCAB_N", "32768".into());
            set(&mut env, "FLE_MTP_PENALTIES", "1".into());
        }
    }
    if let Some(Value::Object(e)) = sget("env") {
        for (k, v) in e {
            set(
                &mut env,
                k,
                match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                },
            );
        }
    }
    for (k, v) in &o.env_override {
        set(&mut env, k, v.clone());
    }
    args.extend(o.extra_args.iter().cloned());
    let alias = stable_alias(model, &model_arg);
    add(&mut args, &["--alias", &alias]);

    let slots = extra_parallel(&args).max(1);
    let kv_unified = args.iter().any(|a| a == "--kv-unified");
    let per_slot = if slots == 1 || kv_unified {
        ctx
    } else {
        (ctx / slots).max(1)
    };
    if slots > 1 && kv_unified {
        notes.push(format!(
            "{slots} slots share one unified KV cache; the context is total, not per slot."
        ));
    } else if slots > 1 {
        warnings.push(format!("Explicit --parallel {slots} without --kv-unified: each slot gets about {} tokens and unified KV is off. Add --arg=--kv-unified, or use one slot.", human_int(per_slot)));
    }
    if !has_experts && matches!(profile.as_str(), "auto" | "balanced" | "max" | "debug") {
        notes.push("Dense model: no expert cache and no MoE flags; full GPU offload, fit, KV cache and speculation still apply.".into());
    }
    if profile == "max" && !has_experts {
        notes.push(format!("Max does not add an expert cache for a dense model; it only raises the default context to {} tokens and lowers the fit margin. For speed, try --spec ngram-mod.", human_int(profile_context("max"))));
    }
    if cache_mib > 0 && profile != "off" && profile != "nvme" {
        if profile == "balanced" {
            notes.push("Balanced keeps 30 percent of the possible cache free for other GPU work and spikes.".into());
        } else if profile == "max" {
            notes.push(
                "Max gives the cache all of the possible free memory; other GPU work may not fit."
                    .into(),
            );
        }
    }
    if profile == "debug" {
        notes.push("Debug adds timing and graph-reuse counters; it is not a speed profile.".into());
    }
    if !(h.port_free)(&host, port) {
        warnings.push(format!(
            "Port {port} on {host} is already in use. Stop that server or choose --port."
        ));
    }
    let opt = |s: &str| {
        if s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    };
    Ok(Plan {
        profile,
        engine,
        binary: path,
        args,
        env,
        host,
        port,
        model: model.clone(),
        sidecars: [opt(&head), opt(&vocab), opt(&sc("mmproj"))],
        context: ctx,
        slots,
        kv_unified,
        context_per_slot: per_slot,
        spec: effective_spec,
        alias,
        threads: nthreads,
        fit_target_mib: fit_target,
        cache_mib,
        fallback: false,
        notes,
        warnings,
        estimate,
    })
}

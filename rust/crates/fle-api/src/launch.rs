//! How an engine is started: its command line and environment. Port of `fle_studio/launch.py` (and the one function of
//! `qol.py` it uses).
//!
//! Two ways to get one: **auto** (derived from this machine, the model and a few settings) and **a profile** (a saved,
//! complete command with placeholders, pasted as shell text and run exactly as written plus what Studio needs to talk to
//! the engine). Placeholders: {model} {mtp_head} {mtp_vocab} {expert_profile} {port} {host} {threads} {context} {data}.
use crate::settings::{py_repr, py_str, py_truthy};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: f64 = 1024.0 * 1024.0;
const GIB_I: i64 = 1024 * 1024 * 1024;
/// free GPU memory next to the expert cache, from the reference command
pub const FIT_TARGET_MIB: i64 = 1115;
/// GPU memory besides weights, KV, cache and margin (measured with the reference command)
pub const ENGINE_OVERHEAD_MIB: i64 = 6749;

/// The command from the model card (section "FLE engine"), as a profile.
pub const REFERENCE_ENV: [(&str, &str); 20] = [
    ("FLE_MOE_CACHE_BUDGET_MIB", "17035"),
    ("FLE_MOE_CACHE_BUDGET_FIT", "1"),
    ("FLE_MOE_PROFILE", "{expert_profile}"),
    ("FLE_MOE_PROFILE_SAVE_SEC", "600"),
    ("FLE_MOE_ADMIT_FREQ", "2000"),
    ("FLE_MOE_EVICT_OLDEST", "32"),
    ("FLE_MOE_CACHE_BIAS", "0.001"),
    ("FLE_MOE_OBS_MAXTOK", "8"),
    ("FLE_MC_OVERLAP", "2"),
    ("FLE_HC_FUSED", "1"),
    ("FLE_CKPT_TAIL", "512"),
    ("FLE_MTP_VOCAB", "{mtp_vocab}"),
    ("FLE_MTP_VOCAB_N", "32768"),
    ("FLE_MTP_PENALTIES", "1"),
    ("FLE_PREFILL_LEAN", "1"),
    ("FLE_SLOT_EARLY_SAVE", "1"),
    ("FLE_CACHE_GUARD_SLOT", "0"),
    ("FLE_UBATCH_LEND", "1"),
    ("FLE_UBATCH_PREFETCH", "1"),
    ("FLE_UBATCH_PREFETCH_SLOTS", "3"),
];

pub const REFERENCE_ARGS: &[&str] = &[
    "-m",
    "{model}",
    "-md",
    "{mtp_head}",
    "--spec-type",
    "draft-mtp",
    "--spec-draft-n-max",
    "2",
    "--spec-draft-p-min",
    "0.5",
    "--no-spec-draft-backend-sampling",
    "-cmoed",
    "-c",
    "262144",
    "--parallel",
    "2",
    "--kv-unified",
    "--jinja",
    "-b",
    "4096",
    "-ub",
    "4096",
    "-ctk",
    "q8_0",
    "-ctv",
    "q8_0",
    "-fit",
    "on",
    "--fit-target",
    "1115",
    "--moe-expert-cache-inserts",
    "1",
    "--cache-ram",
    "32768",
    "--load-mode",
    "none",
    "-t",
    "16",
    "-C",
    "0x0000FFFF",
    "--cpu-strict",
    "1",
    "--temp",
    "0.6",
    "--top-p",
    "0.95",
    "--top-k",
    "20",
    "--min-p",
    "0",
    "--presence-penalty",
    "1.5",
];

/// The reference arguments without the three empty fillers that give the constant a fixed length.
fn reference_args() -> Vec<String> {
    REFERENCE_ARGS
        .iter()
        .filter(|a| !a.is_empty())
        .map(|a| a.to_string())
        .collect()
}

/// Cache policy and compute switches of the reference command that do not depend on the machine.
fn fle_policy_env() -> Vec<(String, String)> {
    REFERENCE_ENV
        .iter()
        .filter(|(k, _)| {
            !matches!(
                *k,
                "FLE_MOE_CACHE_BUDGET_MIB" | "FLE_MOE_PROFILE" | "FLE_MTP_VOCAB"
            )
        })
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// `PlanError`: the message is what the page shows. Two arguments (as the Python code sometimes passes) print as a tuple.
#[derive(Debug, Clone)]
pub struct PlanError(pub String);
/// A message with its hint. The Python Studio printed the pair as a tuple (`('…', '…')`) by mistake; here it reads as
/// two sentences (deliberate difference D1, see DEVIATIONS.md).
fn plan_err2(a: &str, b: &str) -> PlanError {
    PlanError(format!("{a} {b}"))
}
type R<T> = Result<T, PlanError>;
/// `KEY=VALUE` pairs in the order they were written.
pub type EnvPairs = Vec<(String, String)>;
/// environment, binary, arguments of a pasted command
pub type Command = (EnvPairs, String, Vec<String>);
/// chosen GPUs, their environment, a note
type GpuPick = (Vec<Value>, EnvPairs, String);
fn perr<T>(m: impl Into<String>) -> R<T> {
    Err(PlanError(m.into()))
}

// ---------------------------------------------------------------- text helpers

/// `shlex.quote`
pub fn shlex_quote(s: &str) -> String {
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

/// `shlex.quote` that leaves placeholders such as `{model}` readable (`{a,b}` and the like still get quoted).
pub fn shell_quote(word: &str) -> String {
    // replace every `{[a-z_]+}` by "x", then test the safe set
    let mut out = String::new();
    let b: Vec<char> = word.chars().collect();
    let mut i = 0;
    while i < b.len() {
        if b[i] == '{' {
            let mut j = i + 1;
            while j < b.len() && (b[j].is_ascii_lowercase() || b[j] == '_') {
                j += 1;
            }
            if j > i + 1 && j < b.len() && b[j] == '}' {
                out.push('x');
                i = j + 1;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    if !word.is_empty()
        && out.chars().all(|c| {
            c.is_alphanumeric()
                || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
        })
    {
        word.to_string()
    } else {
        shlex_quote(word)
    }
}

/// `shlex.split` (POSIX): quotes group, a backslash escapes, no comments.
pub fn shlex_split(text: &str) -> Result<Vec<String>, String> {
    let mut words: Vec<String> = vec![];
    let mut cur = String::new();
    let mut has = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {
                if has {
                    words.push(std::mem::take(&mut cur));
                    has = false;
                }
            }
            '\\' => match chars.next() {
                Some(n) => {
                    cur.push(n);
                    has = true;
                }
                None => return Err("No escaped character".into()),
            },
            '\'' => {
                has = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(n) => cur.push(n),
                        None => return Err("No closing quotation".into()),
                    }
                }
            }
            '"' => {
                has = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(n) if n == '"' || n == '\\' => cur.push(n),
                            Some(n) => {
                                cur.push('\\');
                                cur.push(n);
                            }
                            None => return Err("No closing quotation".into()),
                        },
                        Some(n) => cur.push(n),
                        None => return Err("No closing quotation".into()),
                    }
                }
            }
            other => {
                cur.push(other);
                has = true;
            }
        }
    }
    if has {
        words.push(cur);
    }
    Ok(words)
}

fn split_words(text: &str) -> R<Vec<String>> {
    if !cfg!(windows) {
        return shlex_split(text)
            .map_err(|e| PlanError(format!("Could not read the command: {e}")));
    }
    // Windows: double quotes group, backslashes are path separators.
    let (mut words, mut cur, mut quoted, mut has) = (vec![], String::new(), false, false);
    for ch in text.chars() {
        if ch == '"' {
            quoted = !quoted;
            has = true;
        } else if ch.is_whitespace() && !quoted {
            if !cur.is_empty() || has {
                words.push(std::mem::take(&mut cur));
            }
            has = false;
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() || has {
        words.push(cur);
    }
    Ok(words)
}

fn is_env_assign(w: &str) -> bool {
    let Some(eq) = w.find('=') else { return false };
    let k = &w[..eq];
    !k.is_empty()
        && !k.as_bytes()[0].is_ascii_digit()
        && k.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// Parse pasted shell text into (environment, binary, arguments). Accepts backslash, caret or backtick line continuations,
/// a leading `env`, `KEY=VALUE` words before the binary and a systemd-style `ExecStart=` prefix.
pub fn split_command(text: &str) -> R<Command> {
    let t = text.trim();
    // \\\r?\n | ^\r?\n | `\r?\n  ->  one space
    let mut s = String::new();
    let cs: Vec<char> = t.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        if matches!(cs[i], '\\' | '^' | '`') {
            let mut j = i + 1;
            if j < cs.len() && cs[j] == '\r' {
                j += 1;
            }
            if j < cs.len() && cs[j] == '\n' {
                s.push(' ');
                i = j + 1;
                continue;
            }
        }
        s.push(cs[i]);
        i += 1;
    }
    let s = match s.strip_prefix("ExecStart=") {
        Some(r) => r.trim_start_matches(['-', '@', '+', '!', ':']).to_string(),
        None => s,
    };
    let mut words = split_words(&s)?;
    let mut env: Vec<(String, String)> = vec![];
    while words
        .first()
        .is_some_and(|w| w == "env" || is_env_assign(w))
    {
        let w = words.remove(0);
        if w != "env" {
            let (k, v) = w.split_once('=').unwrap();
            match env.iter_mut().find(|(a, _)| a == k) {
                Some(e) => e.1 = v.to_string(),
                None => env.push((k.to_string(), v.to_string())),
            }
        }
    }
    let mut binary = String::new();
    if words.first().is_some_and(|w| !w.starts_with('-')) {
        binary = words.remove(0);
    }
    Ok((env, binary, words))
}

pub fn parse_env_lines(text: &str) -> R<Vec<(String, String)>> {
    let mut env: Vec<(String, String)> = vec![];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let ok = line.split_once('=').filter(|(k, _)| {
            let k = k.trim();
            !k.is_empty()
                && !k.as_bytes()[0].is_ascii_digit()
                && k.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        });
        let Some((k, v)) = ok else {
            return perr(format!("Not a KEY=VALUE line: {line}"));
        };
        set_pair(&mut env, k.trim(), v.trim().to_string());
    }
    Ok(env)
}

fn set_pair(env: &mut Vec<(String, String)>, k: &str, v: String) {
    match env.iter_mut().find(|(a, _)| a == k) {
        Some(e) => e.1 = v,
        None => env.push((k.to_string(), v)),
    }
}

fn has(args: &[String], names: &[&str]) -> bool {
    args.iter().any(|a| {
        names.contains(&a.as_str())
            || names
                .iter()
                .any(|n| n.starts_with("--") && a.starts_with(&format!("{n}=")))
    })
}

fn arg_value(args: &[String], names: &[&str]) -> Option<String> {
    for (i, a) in args.iter().enumerate() {
        if names.contains(&a.as_str()) && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        for n in names {
            if n.starts_with("--") {
                if let Some(v) = a.strip_prefix(&format!("{n}=")) {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// Python's `f"{n:,}"`
pub fn commas(n: i64) -> String {
    fle_core::launch::human_int(n)
}

fn num(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn int_of(v: Option<&Value>) -> i64 {
    match v {
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(0),
        Some(Value::String(s)) => s.trim().parse::<f64>().map(|f| f as i64).unwrap_or(0),
        Some(Value::Bool(b)) => i64::from(*b),
        _ => 0,
    }
}

fn round_mib(n_bytes: f64, step: i64) -> i64 {
    let whole = (n_bytes / MIB).floor() as i64;
    (whole.div_euclid(step) * step).max(0)
}

/// `(context, per slot)` like `qol.effective_context`.
pub fn effective_context(context: i64, parallel: i64, kv_unified: bool) -> (i64, i64) {
    let parallel = parallel.max(1);
    let context = context.max(1);
    if kv_unified || parallel == 1 {
        (context, context)
    } else {
        (context, (context / parallel).max(1))
    }
}

/// One-line note about the effective context (`qol.effective_context(...)["note"]`).
pub fn effective_context_note(context: i64, parallel: i64, kv_unified: bool) -> String {
    let parallel = parallel.max(1);
    let context = context.max(1);
    if kv_unified || parallel == 1 {
        if parallel > 1 {
            format!(
                "{} tokens total, shared by all {parallel} slots through one unified KV cache",
                commas(context)
            )
        } else {
            format!("{} tokens total", commas(context))
        }
    } else {
        let per = (context / parallel).max(1);
        if context % parallel != 0 {
            format!(
                "about {} tokens per slot ({} tokens total, separate KV per slot)",
                commas(per),
                commas(context)
            )
        } else {
            format!(
                "{} tokens per slot ({} tokens total, separate KV per slot)",
                commas(per),
                commas(context)
            )
        }
    }
}

fn ensure(p: &Path) -> PathBuf {
    let _ = std::fs::create_dir_all(p);
    p.to_path_buf()
}

pub fn free_port(preferred: u16) -> u16 {
    if preferred != 0 && std::net::TcpListener::bind(("127.0.0.1", preferred)).is_ok() {
        return preferred;
    }
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- plan

#[derive(Debug, Clone)]
pub struct Plan {
    /// fle | upstream | custom
    pub engine: String,
    pub binary: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub port: i64,
    pub model: Value,
    pub profile: String,
    pub notes: Vec<String>,
    /// path of the draft/MTP model, for the load progress
    pub draft: Option<String>,
    /// set when this plan replaces one that failed to load or could not be built: why (it is not tried a second time)
    pub fallback: String,
}

impl Plan {
    /// The command as one would type it in this OS's shell, for display and copying.
    pub fn command_line(&self) -> String {
        if cfg!(windows) {
            let mut lines: Vec<String> = self
                .env
                .iter()
                .map(|(k, v)| format!("set {k}={v}"))
                .collect();
            let all: Vec<&String> = std::iter::once(&self.binary)
                .chain(self.args.iter())
                .collect();
            lines.push(
                all.iter()
                    .map(|a| win_quote(a))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            return lines.join("\r\n");
        }
        let env = self
            .env
            .iter()
            .map(|(k, v)| format!("{k}={}", shell_quote(v)))
            .collect::<Vec<_>>()
            .join(" ");
        let cmd = std::iter::once(&self.binary)
            .chain(self.args.iter())
            .map(|a| shell_quote(a))
            .collect::<Vec<_>>()
            .join(" ");
        if env.is_empty() {
            cmd
        } else {
            format!("{env} \\\n  {cmd}")
        }
    }

    pub fn public(&self) -> Value {
        let env: Map<String, Value> = self
            .env
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect();
        json!({"engine": self.engine, "binary": self.binary, "args": self.args, "env": env, "port": self.port, "model": self.model, "profile": self.profile,
            "notes": self.notes, "draft": self.draft, "fallback": self.fallback, "command": self.command_line()})
    }
}

/// `subprocess.list2cmdline` for one word.
fn win_quote(a: &str) -> String {
    let need = a.is_empty() || a.contains([' ', '\t']);
    let mut out = String::new();
    let mut bs = 0;
    for c in a.chars() {
        if c == '\\' {
            bs += 1;
        } else if c == '"' {
            out.push_str(&"\\".repeat(bs * 2 + 1));
            out.push('"');
            bs = 0;
        } else {
            out.push_str(&"\\".repeat(bs));
            bs = 0;
            out.push(c);
        }
    }
    out.push_str(&"\\".repeat(bs));
    if need {
        format!("\"{}{}\"", out, "\\".repeat(bs))
    } else {
        out
    }
}

/// Which NVIDIA GPUs the engine sees: (chosen GPUs, environment, note).
fn gpu_selection(setting: &str, nvidia: &[Value]) -> R<GpuPick> {
    if nvidia.is_empty() {
        return Ok((vec![], vec![], String::new()));
    }
    let s = if setting.trim().is_empty() {
        "auto".to_string()
    } else {
        setting.trim().to_lowercase()
    };
    if s == "all" || (nvidia.len() == 1 && s == "auto") {
        return Ok((nvidia.to_vec(), vec![], String::new()));
    }
    let pick: Vec<Value> = if s == "auto" {
        // max() keeps the first of equal values
        let mut best = &nvidia[0];
        for g in nvidia {
            if g["memory_total"].as_i64().unwrap_or(0) > best["memory_total"].as_i64().unwrap_or(0)
            {
                best = g;
            }
        }
        vec![best.clone()]
    } else {
        let want: Vec<String> = s
            .split(',')
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect();
        let pick: Vec<Value> = nvidia
            .iter()
            .filter(|g| want.contains(&g["index"].to_string()))
            .cloned()
            .collect();
        if pick.is_empty() {
            let have = nvidia
                .iter()
                .map(|g| g["index"].to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return perr(format!(
                "No NVIDIA GPU with number {s}; this machine has {have}"
            ));
        }
        pick
    };
    let env = vec![
        ("CUDA_DEVICE_ORDER".to_string(), "PCI_BUS_ID".to_string()),
        (
            "CUDA_VISIBLE_DEVICES".to_string(),
            pick.iter()
                .map(|g| g["index"].to_string())
                .collect::<Vec<_>>()
                .join(","),
        ),
    ];
    let names = pick
        .iter()
        .map(|g| format!("GPU {} ({})", g["index"], g["name"].as_str().unwrap_or("")))
        .collect::<Vec<_>>()
        .join(", ");
    Ok((
        pick,
        env,
        format!("Engine uses {names}; the other GPUs stay free."),
    ))
}

/// `{x:g}` for a float, `str(x)` otherwise.
fn sampling_text(v: &Value) -> String {
    match v {
        Value::Number(n) if !(n.is_i64() || n.is_u64()) => {
            let f = n.as_f64().unwrap_or(0.0);
            fmt_g(f)
        }
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Python's `format(x, "g")`: six significant digits, no trailing zeros.
pub fn fmt_g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-5..6).contains(&exp) {
        let s = format!("{:.5e}", x);
        let (m, e) = s.split_once('e').unwrap();
        let m = if m.contains('.') {
            m.trim_end_matches('0').trim_end_matches('.')
        } else {
            m
        };
        let ev: i32 = e.parse().unwrap();
        return format!("{m}e{}{:02}", if ev < 0 { '-' } else { '+' }, ev.abs());
    }
    let decimals = (5 - exp).max(0) as usize;
    let s = format!("{x:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

pub struct AutoInput<'a> {
    pub model: &'a Value,
    pub engine: &'a Value,
    pub settings: &'a Value,
    pub sysinfo: &'a Value,
    pub port: i64,
    pub data_dir: &'a Path,
}

/// The command FLE Studio picks for this model on this machine.
pub fn auto_plan(i: &AutoInput) -> R<Plan> {
    let (launch, chat) = (&i.settings["launch"], &i.settings["chat"]);
    let (model, engine, sys) = (i.model, i.engine, i.sysinfo);
    let kind = engine["kind"].as_str().unwrap_or("").to_string();
    let flavor = match engine.get("flavor") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => {
            if kind == "fle" {
                "fle".into()
            } else {
                "upstream".into()
            }
        }
    };
    let info = if model["info"].is_object() {
        model["info"].clone()
    } else {
        json!({})
    };
    let mut notes: Vec<String> = vec![];
    let mut ctx = match int_of(launch.get("context")) {
        0 => 32768,
        c => c,
    };
    if let Some(cl) = info["context_length"].as_i64().filter(|c| *c != 0) {
        if ctx > cl {
            ctx = cl;
            notes.push(format!(
                "Context capped at the model's maximum of {} tokens.",
                commas(ctx)
            ));
        }
    }
    let parallel = int_of(launch.get("parallel")).max(1);
    let mut args: Vec<String> = [
        "-m",
        model["path"].as_str().unwrap_or(""),
        "-c",
        &ctx.to_string(),
        "--jinja",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if parallel > 1 {
        args.extend([
            "--parallel".to_string(),
            parallel.to_string(),
            "--kv-unified".to_string(),
        ]);
        notes.push(effective_context_note(ctx, parallel, true));
    }
    let mut env: Vec<(String, String)> = vec![];
    let nvidia: Vec<Value> = sys["gpus"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|g| g["vendor"] == "nvidia")
        .cloned()
        .collect();
    let (chosen, gpu_env, note) =
        gpu_selection(launch["gpus"].as_str().unwrap_or("auto"), &nvidia)?;
    for (k, v) in gpu_env {
        set_pair(&mut env, &k, v);
    }
    if !note.is_empty() {
        notes.push(note);
    }
    let mem = &sys["memory"];
    let mem_total = mem["total"].as_f64().unwrap_or(0.0);
    let physical = match int_of(sys["cpu"].get("physical")) {
        0 => 1,
        p => p,
    };
    let mut threads = match int_of(launch.get("threads")) {
        0 => physical,
        t => t,
    };
    let mut draft: Option<String> = None;
    let pinned_bytes;

    if flavor == "fle" {
        if chosen.is_empty() {
            return perr(
                "The FLE engine needs an NVIDIA GPU. Pick the upstream engine on the Engine page.",
            );
        }
        let vram_total: f64 = chosen.iter().map(|g| num(g, "memory_total")).sum();
        let vram_free: f64 = chosen
            .iter()
            .map(|g| (num(g, "memory_total") - num(g, "memory_used")).max(0.0))
            .sum();
        let ub = if vram_total >= 20.0 * GIB { 4096 } else { 2048 };
        let kv = num(&info, "kv_bytes_per_token_q8") * ctx as f64;
        let overhead = ENGINE_OVERHEAD_MIB as f64 * MIB;
        let gpu_weights = num(&info, "gpu_bytes");
        let mut budget_mib = int_of(launch.get("gpu_cache_mib"));
        if budget_mib <= 0 {
            // The budget is added to the fit margin (FLE_MOE_CACHE_BUDGET_FIT=1): too large a budget makes the fit step
            // move weights to the CPU, so it errs on the small side.
            budget_mib = round_mib(
                vram_free - gpu_weights - kv - overhead - FIT_TARGET_MIB as f64 * MIB,
                1,
            );
            notes.push(format!(
                "GPU expert cache {:.1} GiB: {:.1} GiB free GPU memory minus {:.1} GiB of always-on weights, {:.1} GiB KV cache, {:.1} GiB for compute buffers and the CUDA context (measured with the reference command) and {} MiB margin. Set it by hand on the Engine page if loading fails or memory is left over.",
                budget_mib as f64 / 1024.0, vram_free / GIB, gpu_weights / GIB, kv / GIB, overhead / GIB, FIT_TARGET_MIB
            ));
        }
        if budget_mib < 1024 {
            notes.push(
                "Little GPU memory is left for the expert cache, so most experts run on the CPU."
                    .into(),
            );
        }
        let logical = match int_of(sys["cpu"].get("logical")) {
            0 => physical,
            l => l,
        };
        if int_of(launch.get("threads")) == 0 && physical > 1 && logical <= physical {
            threads = physical - 1;
            notes.push(format!("{threads} threads: this processor runs one thread per core, so one core stays free for the engine's GPU and expert-cache threads."));
        }
        for (k, v) in fle_policy_env() {
            set_pair(&mut env, &k, v);
        }
        set_pair(&mut env, "FLE_MOE_CACHE_BUDGET_MIB", budget_mib.to_string());
        let profile_dir = ensure(&i.data_dir.join("expert-profiles"));
        set_pair(
            &mut env,
            "FLE_MOE_PROFILE",
            profile_dir
                .join(format!("{}.bin", py_str(&model["id"])))
                .to_string_lossy()
                .to_string(),
        );
        for a in [
            "-b",
            &ub.to_string(),
            "-ub",
            &ub.to_string(),
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "-fit",
            "on",
            "--fit-target",
            &FIT_TARGET_MIB.to_string(),
            "--moe-expert-cache-inserts",
            "1",
            "-t",
            &threads.to_string(),
        ] {
            args.push(a.to_string());
        }
        let expert = num(&info, "expert_bytes");
        let mut pinned = launch["pinned_load"]
            .as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or(if launch.get("pinned_load").is_none() {
                "auto"
            } else {
                ""
            })
            .to_string();
        if pinned == "auto" {
            pinned = if cfg!(target_os = "linux") || cfg!(windows) {
                "on".into()
            } else {
                "off".into()
            };
            if mem_total - expert < 24.0 * GIB {
                pinned = "off".into();
                notes.push("Experts stay memory-mapped (not pinned): pinning them would leave less than 24 GiB RAM.".into());
            }
        }
        if pinned == "on" {
            args.extend(["--load-mode".to_string(), "none".to_string()]);
        }
        let spec = match launch.get("spec") {
            Some(v) if py_truthy(v) => py_str(v).to_lowercase(),
            _ => "auto".into(),
        };
        if !["auto", "none", "mtp", "ngram-mod", "ngram-simple"].contains(&spec.as_str()) {
            return Err(plan_err2(
                &format!("Unknown speculation mode {}.", py_repr(&json!(spec))),
                "Use auto, none, mtp, ngram-mod or ngram-simple.",
            ));
        }
        let mtp = if model["mtp"].is_object() {
            model["mtp"].clone()
        } else {
            json!({})
        };
        let head = mtp["head"].as_str().unwrap_or("").to_string();
        let vocab = mtp["vocab"].as_str().unwrap_or("").to_string();
        let (mut use_mtp, mut ngram) = (false, String::new());
        let mtp_on = launch.get("mtp").map(py_truthy).unwrap_or(true);
        // Only an engine whose help does not list --spec-type is known to lack it; entries without the key are recent.
        let spec_type_ok = engine.get("spec_type").map(py_truthy).unwrap_or(true);
        if matches!(spec.as_str(), "mtp" | "ngram-mod" | "ngram-simple") && !spec_type_ok {
            return Err(plan_err2(
                &format!("Speculation mode {} needs an engine that knows --spec-type, and this one does not.", py_repr(&json!(spec))),
                "Update the engine, or set speculation to none.",
            ));
        }
        match spec.as_str() {
            "none" => notes.push("Speculation is off.".into()),
            "mtp" => {
                if !mtp_on {
                    return Err(plan_err2(
                        "Speculation is set to mtp, but MTP is switched off (launch.mtp); the two settings contradict each other.",
                        "Switch MTP on, or set speculation to auto or none.",
                    ));
                }
                if head.is_empty() || vocab.is_empty() {
                    return Err(plan_err2("MTP was requested, but this model has no MTP head and vocabulary file.", "Download the MTP sidecars on the Models page, or set speculation to auto or ngram-mod."));
                }
                use_mtp = true;
            }
            "ngram-mod" | "ngram-simple" => {
                ngram = spec.clone();
                notes.push(format!("Self-speculation: {spec}. It drafts from the prompt/context; acceptance is prompt-dependent."));
            }
            _ => {
                if mtp_on && !head.is_empty() && !vocab.is_empty() && !spec_type_ok {
                    notes.push(
                        "MTP is off: this engine does not know --spec-type. Update the engine."
                            .into(),
                    );
                } else if mtp_on && !head.is_empty() && !vocab.is_empty() {
                    use_mtp = true;
                } else if mtp_on
                    && mtp["supported"].as_bool() == Some(true)
                    && (head.is_empty() || vocab.is_empty())
                {
                    notes.push("MTP is off: the MTP head or its vocabulary file is not downloaded (Models page).".into());
                }
            }
        }
        if use_mtp {
            draft = Some(head.clone());
            for a in [
                "-md",
                &head,
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "2",
                "--spec-draft-p-min",
                "0.5",
                "--no-spec-draft-backend-sampling",
                "-cmoed",
            ] {
                args.push(a.to_string());
            }
            set_pair(&mut env, "FLE_MTP_VOCAB", vocab);
            set_pair(&mut env, "FLE_MTP_VOCAB_N", "32768".into());
            set_pair(&mut env, "FLE_MTP_PENALTIES", "1".into());
        }
        if !ngram.is_empty() {
            args.extend(["--spec-type".to_string(), ngram]);
        }
        pinned_bytes = if pinned == "on" { expert } else { 0.0 };
    } else {
        args.extend(["-fit".to_string(), "on".to_string()]);
        if int_of(launch.get("threads")) != 0 {
            args.extend(["-t".to_string(), threads.to_string()]);
        }
        pinned_bytes = 0.0;
        if engine["backend"] == "cpu" && info["moe"].as_bool() == Some(true) {
            notes.push("CPU engine: every expert runs on the CPU.".into());
        }
    }

    let mut cram = launch
        .get("prompt_cache_mib")
        .map(|v| int_of(Some(v)))
        .unwrap_or(-1);
    if cram < 0 {
        let headroom = mem_total - pinned_bytes - 16.0 * GIB;
        cram = 32768.min(round_mib(headroom / 4.0, 1024));
    }
    args.extend(["--cache-ram".to_string(), cram.to_string()]);
    if let Some(m) = launch["cpu_mask"].as_str().filter(|m| !m.is_empty()) {
        args.extend([
            "-C".to_string(),
            m.to_string(),
            "--cpu-strict".to_string(),
            "1".to_string(),
        ]);
    }
    let mut sampling: Map<String, Value> =
        model["sampling"].as_object().cloned().unwrap_or_default();
    if chat["sampling"] == "custom" {
        sampling = ["temperature", "top_p", "top_k", "min_p", "presence_penalty"]
            .iter()
            .map(|k| (k.to_string(), chat[k].clone()))
            .collect();
    }
    for (key, value) in &sampling {
        let flag = match key.as_str() {
            "temperature" => "--temp",
            "top_p" => "--top-p",
            "top_k" => "--top-k",
            "min_p" => "--min-p",
            "presence_penalty" => "--presence-penalty",
            _ => continue,
        };
        args.extend([flag.to_string(), sampling_text(value)]);
    }
    args.extend(split_words(launch["extra_args"].as_str().unwrap_or(""))?);
    for (k, v) in parse_env_lines(launch["extra_env"].as_str().unwrap_or(""))? {
        set_pair(&mut env, &k, v);
    }
    finish(
        Plan {
            engine: kind,
            binary: engine["path"].as_str().unwrap_or("").to_string(),
            args,
            env,
            port: i.port,
            model: model.clone(),
            profile: "auto".into(),
            notes,
            draft,
            fallback: String::new(),
        },
        engine
            .get("log_verbosity")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    )
}

/// Add what FLE Studio needs to reach and follow the engine, unless the command sets it itself. `log_verbosity`: the engine
/// knows `-lv` (llama.cpp logs its library lines at level 4, one above the default).
fn finish(mut plan: Plan, log_verbosity: bool) -> R<Plan> {
    if log_verbosity
        && !plan.env.iter().any(|(k, _)| k == "LLAMA_ARG_LOG_VERBOSITY")
        && !has(
            &plan.args,
            &[
                "-v",
                "--verbose",
                "--log-verbose",
                "-lv",
                "--verbosity",
                "--log-verbosity",
            ],
        )
    {
        plan.args.extend(["-lv".to_string(), "4".to_string()]);
    }
    if !has(&plan.args, &["--host"]) {
        plan.args
            .extend(["--host".to_string(), "127.0.0.1".to_string()]);
    }
    if has(&plan.args, &["--port"]) {
        if let Some(v) = arg_value(&plan.args, &["--port"]).filter(|v| !v.is_empty()) {
            plan.port = v
                .trim()
                .parse::<i64>()
                .map_err(|_| PlanError("--port needs a number".into()))?;
        }
    } else {
        plan.args
            .extend(["--port".to_string(), plan.port.to_string()]);
    }
    if !has(&plan.args, &["--metrics"]) {
        plan.args.push("--metrics".to_string());
    }
    if !has(&plan.args, &["-a", "--alias"]) {
        let name = match plan.model.get("name") {
            Some(v) if py_truthy(v) => py_str(v),
            _ => "model".into(),
        };
        plan.args.extend(["-a".to_string(), name]);
    }
    Ok(plan)
}

// ---------------------------------------------------------------- profiles

#[derive(Debug, Clone)]
pub struct Profile {
    pub id: String,
    pub name: String,
    /// fle | upstream | custom (custom: `binary` is the engine)
    pub engine: Value,
    pub binary: Value,
    pub args: Value,
    pub env: Value,
    pub notes: Value,
    pub builtin: bool,
    pub meta: Value,
}

impl Profile {
    fn strs(v: &Value) -> Vec<String> {
        v.as_array()
            .map(|a| a.iter().map(py_str).collect())
            .unwrap_or_default()
    }
    fn env_pairs(&self) -> Vec<(String, String)> {
        self.env
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), py_str(v))).collect())
            .unwrap_or_default()
    }
    pub fn command_text(&self) -> String {
        let env = self
            .env_pairs()
            .iter()
            .map(|(k, v)| format!("{k}={}", shell_quote(v)))
            .collect::<Vec<_>>()
            .join(" ");
        let bin = match &self.binary {
            v if py_truthy(v) => py_str(v),
            _ => "{engine}".into(),
        };
        let cmd = std::iter::once(bin)
            .chain(Self::strs(&self.args))
            .map(|a| shell_quote(&a))
            .collect::<Vec<_>>()
            .join(" ");
        if env.is_empty() {
            cmd
        } else {
            format!("{env} \\\n{cmd}")
        }
    }
    pub fn public(&self) -> Value {
        json!({"id": self.id, "name": self.name, "engine": self.engine, "binary": self.binary, "args": self.args, "env": self.env, "notes": self.notes,
            "builtin": self.builtin, "meta": self.meta, "command": self.command_text()})
    }
    fn stored(&self) -> Value {
        json!({"id": self.id, "name": self.name, "engine": self.engine, "binary": self.binary, "args": self.args, "env": self.env, "notes": self.notes,
            "builtin": self.builtin, "meta": self.meta})
    }
}

pub fn builtin() -> Vec<Profile> {
    let env: Map<String, Value> = REFERENCE_ENV
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    vec![Profile {
        id: "fle-reference".into(),
        name: "FLE reference (32 GB GPU)".into(),
        engine: json!("fle"),
        binary: json!(""),
        args: json!(reference_args()),
        env: Value::Object(env),
        notes: json!("The exact command from the model card: 262,144-token context shared by two slots, 16 threads pinned to cores 0-15, a 17 GiB GPU expert cache for a 32 GB card. Needs the MTP head and about 54 GiB of free RAM for FLE-IQ3_M. Measured only on an RTX PRO 4500 32 GB with 120 GB RAM."),
        builtin: true,
        meta: json!({}),
    }]
}

pub struct Profiles {
    file: PathBuf,
    pub items: Vec<Profile>,
}

impl Profiles {
    pub fn new(data_dir: &Path) -> Profiles {
        let file = data_dir.join("profiles.json");
        let mut items = vec![];
        if let Some(Value::Array(a)) = std::fs::read_to_string(&file)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        {
            for raw in a {
                // Profile(**raw): a field it does not know, or a non-object, ends the loading (what was read stays)
                let Value::Object(mut o) = raw else { break };
                o.shift_remove("builtin");
                o.shift_remove("command");
                let known = [
                    "id", "name", "engine", "binary", "args", "env", "notes", "meta",
                ];
                if o.keys().any(|k| !known.contains(&k.as_str()))
                    || !o.contains_key("id")
                    || !o.contains_key("name")
                {
                    break;
                }
                items.push(Profile {
                    id: py_str(&o["id"]),
                    name: py_str(&o["name"]),
                    engine: o.get("engine").cloned().unwrap_or(json!("fle")),
                    binary: o.get("binary").cloned().unwrap_or(json!("")),
                    args: o.get("args").cloned().unwrap_or(json!([])),
                    env: o.get("env").cloned().unwrap_or(json!({})),
                    notes: o.get("notes").cloned().unwrap_or(json!("")),
                    builtin: false,
                    meta: o.get("meta").cloned().unwrap_or(json!({})),
                });
            }
        }
        Profiles { file, items }
    }

    pub fn all(&self) -> Vec<Profile> {
        builtin()
            .into_iter()
            .chain(self.items.iter().cloned())
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<Profile> {
        self.all().into_iter().find(|p| p.id == id)
    }

    /// Create or replace a profile from `{id?, name, engine, command | binary+args+env, notes}`.
    pub fn save(&mut self, data: &Map<String, Value>) -> R<Profile> {
        let name = data
            .get("name")
            .filter(|v| py_truthy(v))
            .map(py_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if name.is_empty() {
            return perr("A profile needs a name");
        }
        let (env, mut binary, args): (Value, String, Value);
        if let Some(cmd) = data.get("command") {
            let (e, b, a) = split_command(&py_str(cmd))?;
            env = Value::Object(e.into_iter().map(|(k, v)| (k, json!(v))).collect());
            binary = b;
            args = json!(a);
            if ["{engine}", "llama-server", "llama-server.exe"].contains(&binary.as_str()) {
                binary = String::new();
            }
        } else {
            env = data
                .get("env")
                .filter(|v| py_truthy(v))
                .cloned()
                .unwrap_or(json!({}));
            binary = data
                .get("binary")
                .filter(|v| py_truthy(v))
                .map(py_str)
                .unwrap_or_default();
            args = data
                .get("args")
                .filter(|v| py_truthy(v))
                .cloned()
                .unwrap_or(json!([]));
        }
        let engine = match data.get("engine") {
            Some(v) if py_truthy(v) => py_str(v),
            _ => {
                if binary.is_empty() {
                    "fle".into()
                } else {
                    "custom".into()
                }
            }
        };
        if !["fle", "upstream", "custom"].contains(&engine.as_str()) {
            return perr("engine must be fle, upstream or custom");
        }
        if engine == "custom" && binary.is_empty() {
            return perr("A custom-engine profile needs the path of its llama-server at the start of the command");
        }
        let mut ident = data
            .get("id")
            .filter(|v| py_truthy(v))
            .map(py_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if ident.is_empty() || builtin().iter().any(|b| b.id == ident) {
            let mut slug = String::new();
            let mut gap = false;
            for c in name.to_lowercase().chars() {
                if c.is_ascii_lowercase() || c.is_ascii_digit() {
                    if gap && !slug.is_empty() {
                        slug.push('-');
                    }
                    gap = false;
                    slug.push(c);
                } else {
                    gap = true;
                }
            }
            // strip the dashes first, then cut to 40 characters (the cut may leave a dash at the end, as in Python)
            let slug: String = slug.trim_matches('-').chars().take(40).collect();
            let slug = if slug.is_empty() {
                "profile".to_string()
            } else {
                slug
            };
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            ident = format!("{slug}-{}", t % 100000);
        }
        let old = self.get(&ident);
        let meta = match data.get("meta") {
            Some(m) if m.is_object() => m.clone(),
            _ => old.map(|o| o.meta).unwrap_or(json!({})),
        };
        let notes = data
            .get("notes")
            .filter(|v| py_truthy(v))
            .map(py_str)
            .unwrap_or_default();
        let prof = Profile {
            id: ident.clone(),
            name,
            engine: json!(engine),
            binary: json!(binary),
            args,
            env,
            notes: json!(notes),
            builtin: false,
            meta,
        };
        self.items.retain(|p| p.id != ident);
        self.items.push(prof.clone());
        self.write();
        Ok(prof)
    }

    pub fn delete(&mut self, id: &str) {
        self.items.retain(|p| p.id != id);
        self.write();
    }

    fn write(&self) {
        if let Some(d) = self.file.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let tmp = self.file.with_extension("tmp");
        let body: Vec<Value> = self.items.iter().map(Profile::stored).collect();
        let _ = std::fs::write(&tmp, serde_json::to_string_pretty(&body).unwrap())
            .and_then(|_| std::fs::rename(&tmp, &self.file));
    }
}

pub struct ProfileInput<'a> {
    pub profile: &'a Profile,
    pub model: Option<&'a Value>,
    pub engine: Option<&'a Value>,
    pub settings: &'a Value,
    pub sysinfo: &'a Value,
    pub port: i64,
    pub data_dir: &'a Path,
}

/// A profile's command with its placeholders filled in.
pub fn profile_plan(i: &ProfileInput) -> R<Plan> {
    let empty = json!({});
    let model = i
        .model
        .cloned()
        .filter(|m| m.is_object())
        .unwrap_or_else(|| empty.clone());
    let profile = i.profile;
    let mtp = if model["mtp"].is_object() {
        model["mtp"].clone()
    } else {
        json!({})
    };
    let launch = &i.settings["launch"];
    let threads = match int_of(launch.get("threads")) {
        0 => match &i.sysinfo["cpu"]["physical"] {
            v if py_truthy(v) => py_str(v),
            _ => "1".into(),
        },
        t => t.to_string(),
    };
    let model_id = if model.get("id").is_some() {
        py_str(&model["id"])
    } else {
        profile.id.clone()
    };
    let profile_dir = ensure(&i.data_dir.join("expert-profiles"));
    let values: Vec<(&str, String)> = vec![
        ("model", model["path"].as_str().unwrap_or("").to_string()),
        ("mtp_head", mtp["head"].as_str().unwrap_or("").to_string()),
        ("mtp_vocab", mtp["vocab"].as_str().unwrap_or("").to_string()),
        (
            "expert_profile",
            profile_dir
                .join(format!("{model_id}.bin"))
                .to_string_lossy()
                .to_string(),
        ),
        ("port", i.port.to_string()),
        ("host", "127.0.0.1".into()),
        ("threads", threads),
        ("context", py_str(&launch["context"])),
        ("data", i.data_dir.to_string_lossy().to_string()),
    ];
    let needed = |k: &str| match k {
        "model" => Some("a model"),
        "mtp_head" => Some("the MTP head (Models page)"),
        "mtp_vocab" => Some("the MTP vocabulary file"),
        _ => None,
    };
    let fill = |text: &str| -> R<String> {
        // re.sub(r"\{(\w+)\}", ...)
        let cs: Vec<char> = text.chars().collect();
        let mut out = String::new();
        let mut k = 0;
        while k < cs.len() {
            if cs[k] == '{' {
                let mut j = k + 1;
                while j < cs.len() && (cs[j].is_alphanumeric() || cs[j] == '_') {
                    j += 1;
                }
                if j > k + 1 && j < cs.len() && cs[j] == '}' {
                    let key: String = cs[k + 1..j].iter().collect();
                    if let Some((_, v)) = values.iter().find(|(n, _)| *n == key) {
                        if v.is_empty() {
                            if let Some(what) = needed(&key) {
                                return perr(format!("Profile “{}” needs {what}", profile.name));
                            }
                        }
                        out.push_str(v);
                        k = j + 1;
                        continue;
                    }
                }
            }
            out.push(cs[k]);
            k += 1;
        }
        Ok(out)
    };
    let mut args: Vec<String> = Profile::strs(&profile.args)
        .iter()
        .map(|a| fill(a))
        .collect::<R<_>>()?;
    let mut env: Vec<(String, String)> = vec![];
    for (k, v) in profile.env_pairs() {
        set_pair(&mut env, &k, fill(&v)?);
    }
    let model_path = model["path"].as_str().unwrap_or("");
    if !has(&args, &["-m", "--model"]) && !model_path.is_empty() {
        args.insert(0, model_path.to_string());
        args.insert(0, "-m".to_string());
    }
    if !has(&args, &["-m", "--model"]) {
        return perr("Choose a model first");
    }
    let mut notes: Vec<String> = match &profile.notes {
        v if py_truthy(v) => vec![py_str(v)],
        _ => vec![],
    };
    // The GPU and core choice on the Engine page also steers a profile, unless its command already decides.
    let gpus = match launch.get("gpus") {
        Some(v) if py_truthy(v) => py_str(v),
        _ => "auto".into(),
    };
    if gpus != "auto" && !env.iter().any(|(k, _)| k == "CUDA_VISIBLE_DEVICES") {
        let nvidia: Vec<Value> = i.sysinfo["gpus"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|g| g["vendor"] == "nvidia")
            .cloned()
            .collect();
        let (_, gpu_env, note) = gpu_selection(&gpus, &nvidia)?;
        let mut merged = gpu_env;
        for (k, v) in env {
            set_pair(&mut merged, &k, v);
        }
        env = merged;
        if !note.is_empty() {
            notes.push(note);
        }
    }
    if let Some(m) = launch["cpu_mask"].as_str().filter(|m| !m.is_empty()) {
        if !has(&args, &["-C", "--cpu-mask"]) {
            args.extend([
                "-C".to_string(),
                m.to_string(),
                "--cpu-strict".to_string(),
                "1".to_string(),
            ]);
        }
    }
    let pengine = py_str(&profile.engine);
    let (binary, kind) = if pengine == "custom" {
        (fill(&py_str(&profile.binary))?, "custom".to_string())
    } else {
        let Some(e) = i.engine.filter(|e| py_truthy(e)) else {
            return perr(format!(
                "Install the {} engine first (Engine page)",
                if pengine == "fle" { "FLE" } else { "upstream" }
            ));
        };
        (
            e["path"].as_str().unwrap_or("").to_string(),
            e["kind"].as_str().unwrap_or("").to_string(),
        )
    };
    let draft = arg_value(&args, &["-md", "--model-draft"]);
    let plan = Plan {
        engine: kind,
        binary,
        args,
        env,
        port: i.port,
        model: model.clone(),
        profile: profile.id.clone(),
        notes,
        draft,
        fallback: String::new(),
    };
    // For a custom profile `engine` is the check of its own binary (None if that could not run): no -lv unless known.
    let lv = i
        .engine
        .and_then(|e| e.get("log_verbosity"))
        .and_then(Value::as_bool)
        .unwrap_or(pengine != "custom");
    finish(plan, lv)
}

pub fn gib() -> i64 {
    GIB_I
}

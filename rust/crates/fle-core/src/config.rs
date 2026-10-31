//! Global fle settings: one TOML file. Port of `tools/fle/config.py`; the Python helper is the specification.
//!
//! CLI flags beat environment variables, environment variables beat the file, and the file beats the built-in defaults.
use crate::FleError;
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const PROFILES: [&str; 6] = ["off", "auto", "balanced", "max", "nvme", "debug"];
pub const SPECS: [&str; 5] = ["auto", "none", "mtp", "ngram-mod", "ngram-simple"];

/// In the order of `DEFAULTS` in config.py (it decides the order of `config show` and of the saved file).
pub fn defaults() -> Map<String, Value> {
    let Value::Object(m) = json!({
        "profile": "auto", "spec": "auto", "engine": "", "model": "", "host": "127.0.0.1", "port": 8080, "context": 0, "threads": 0,
        "gpu_cache_mib": 0, "fit_target_mib": 0, "pinned_load": "auto", "mtp": true, "env": {}
    }) else {
        unreachable!()
    };
    m
}

const ENV_KEYS: [(&str, &str); 11] = [
    ("FLE_PROFILE", "profile"),
    ("FLE_SPEC", "spec"),
    ("FLE_ENGINE", "engine"),
    ("FLE_MODEL", "model"),
    ("FLE_HOST", "host"),
    ("FLE_PORT", "port"),
    ("FLE_CONTEXT", "context"),
    ("FLE_THREADS", "threads"),
    ("FLE_GPU_CACHE_MIB", "gpu_cache_mib"),
    ("FLE_FIT_TARGET_MIB", "fit_target_mib"),
    ("FLE_MTP", "mtp"),
];

/// Python's `int(value)` with the fallback `int(float(str(value)))`, and `default` for booleans and junk.
pub fn as_int(v: &Value, default: i64) -> i64 {
    match v {
        Value::Bool(_) => default,
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(default),
        Value::String(s) => s
            .trim()
            .parse::<i64>()
            .ok()
            .or_else(|| s.trim().parse::<f64>().ok().map(|f| f as i64))
            .unwrap_or(default),
        _ => default,
    }
}

pub fn as_bool(v: &Value, default: bool) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::String(s) => ["1", "true", "yes", "on"].contains(&s.trim().to_lowercase().as_str()),
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        _ => default,
    }
}

fn expand_home(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => dirs::home_dir().unwrap_or_default().join(rest),
        None => PathBuf::from(p),
    }
}

pub fn config_path(explicit: Option<&str>) -> PathBuf {
    if let Some(e) = explicit.filter(|e| !e.is_empty()) {
        return expand_home(e);
    }
    if let Some(e) = std::env::var("FLE_CONFIG")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        return expand_home(&e);
    }
    let root = match std::env::var("XDG_CONFIG_HOME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        Some(b) => expand_home(&b),
        None => dirs::home_dir().unwrap_or_default().join(".config"),
    };
    root.join("fle").join("config.toml")
}

fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        other => other.to_string(),
    }
}

pub fn load(explicit: Option<&str>) -> Result<(Map<String, Value>, PathBuf)> {
    let path = config_path(explicit);
    let mut data = defaults();
    if !path.exists() {
        return Ok((data, path));
    }
    let text = std::fs::read_to_string(&path)?;
    let raw: Map<String, Value> = toml::from_str::<Value>(&text)
        .map_err(|e| {
            FleError::new(
                format!("Cannot read {}: {}", path.display(), e.message()),
                "Fix the TOML or delete the file to start from defaults.",
            )
        })?
        .as_object()
        .cloned()
        .unwrap_or_default();
    for (key, default) in defaults() {
        let Some(value) = raw.get(&key) else { continue };
        match &default {
            Value::Object(_) => {
                if let Value::Object(o) = value {
                    data.insert(
                        key,
                        Value::Object(
                            o.iter()
                                .map(|(k, v)| (k.clone(), Value::String(py_str(v))))
                                .collect(),
                        ),
                    );
                }
            }
            Value::Bool(d) => {
                data.insert(key, Value::Bool(as_bool(value, *d)));
            }
            Value::Number(d) => {
                data.insert(key, json!(as_int(value, d.as_i64().unwrap_or(0))));
            }
            _ => {
                data.insert(key, Value::String(py_str(value)));
            }
        }
    }
    let profile = py_str(&data["profile"]).to_lowercase();
    let profile = if profile.is_empty() {
        "auto".to_string()
    } else {
        profile
    };
    if !PROFILES.contains(&profile.as_str()) {
        return Err(FleError::new(
            format!("Unknown profile '{profile}' in {}", path.display()),
            format!("Use one of: {}.", PROFILES.join(", ")),
        )
        .into());
    }
    Ok((data, path))
}

/// Python's `json.dumps(str(value), ensure_ascii=True)`.
fn ascii_json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) < 0x7f => out.push(c),
            c => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
        }
    }
    out.push('"');
    out
}

fn toml_value(v: &Value) -> String {
    match v {
        Value::Bool(b) => b.to_string(),
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => format!("{:?}", n.as_f64().unwrap_or(0.0)),
        other => ascii_json_string(&py_str(other)),
    }
}

pub fn save(data: &Map<String, Value>, explicit: Option<&str>) -> Result<PathBuf> {
    let path = config_path(explicit);
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut lines = vec![
        "# fle settings. CLI flags > environment > this file > profile defaults.".to_string(),
        "# Run `fle config show` to see the values in effect.".to_string(),
        String::new(),
    ];
    let d = defaults();
    for (key, default) in &d {
        if key == "env" {
            continue;
        }
        lines.push(format!(
            "{key} = {}",
            toml_value(data.get(key).unwrap_or(default))
        ));
    }
    if let Some(Value::Object(env)) = data.get("env") {
        if !env.is_empty() {
            lines.push(String::new());
            lines.push("[env]".into());
            let mut keys: Vec<_> = env.keys().collect();
            keys.sort();
            for k in keys {
                lines.push(format!("{k} = {}", toml_value(&env[k])));
            }
        }
    }
    std::fs::write(&path, lines.join("\n") + "\n")?;
    Ok(path)
}

pub fn set_dotted(data: &mut Map<String, Value>, key: &str, value: Value) {
    let parts: Vec<&str> = key.split('.').collect();
    let mut node = data;
    for part in &parts[..parts.len() - 1] {
        if !matches!(node.get(*part), Some(Value::Object(_))) {
            node.insert((*part).to_string(), Value::Object(Map::new()));
        }
        node = node.get_mut(*part).unwrap().as_object_mut().unwrap();
    }
    node.insert(parts[parts.len() - 1].to_string(), value);
}

pub fn unset_dotted(data: &mut Map<String, Value>, key: &str) {
    let parts: Vec<&str> = key.split('.').collect();
    let mut node = data;
    for part in &parts[..parts.len() - 1] {
        match node.get_mut(*part) {
            Some(Value::Object(o)) => node = o,
            _ => return,
        }
    }
    node.shift_remove(parts[parts.len() - 1]);
}

/// The few FLE_* variables that set the helper itself; every other FLE_* value goes to the engine.
pub fn env_settings() -> Map<String, Value> {
    let mut out = Map::new();
    for (env_key, key) in ENV_KEYS {
        let Ok(v) = std::env::var(env_key) else {
            continue;
        };
        let val = Value::String(v);
        out.insert(
            key.to_string(),
            match key {
                "port" | "context" | "threads" | "gpu_cache_mib" | "fit_target_mib" => {
                    json!(as_int(&val, 0))
                }
                "mtp" => Value::Bool(as_bool(&val, true)),
                _ => val,
            },
        );
    }
    out
}

/// CLI > environment > file > defaults. `cli` holds only what was given.
pub fn effective(
    cli: &Map<String, Value>,
    data: &Map<String, Value>,
) -> Result<Map<String, Value>> {
    let mut out = defaults();
    for layer in [data.clone(), env_settings(), cli.clone()] {
        for (key, value) in layer {
            if value.is_null() {
                continue;
            }
            if key == "env" {
                if let Value::Object(o) = &value {
                    let mut merged = out
                        .get("env")
                        .and_then(|e| e.as_object().cloned())
                        .unwrap_or_default();
                    for (k, v) in o {
                        merged.insert(k.clone(), Value::String(py_str(v)));
                    }
                    out.insert("env".into(), Value::Object(merged));
                    continue;
                }
            }
            out.insert(key, value);
        }
    }
    let p = py_str(&out["profile"]).to_lowercase();
    let p = if p.is_empty() { "auto".to_string() } else { p };
    if !PROFILES.contains(&p.as_str()) {
        return Err(FleError::new(
            format!("Unknown profile '{p}'"),
            format!("Use one of: {}.", PROFILES.join(", ")),
        )
        .into());
    }
    out.insert("profile".into(), Value::String(p));
    Ok(out)
}

/// `config set KEY VALUE` converts the text like `_parse_value` in cli.py: bool, int, float, else the text.
pub fn parse_value(text: &str) -> Value {
    let low = text.trim().to_lowercase();
    if low == "true" || low == "false" {
        return Value::Bool(low == "true");
    }
    if let Ok(i) = text.parse::<i64>() {
        return json!(i);
    }
    if let Ok(f) = text.parse::<f64>() {
        return json!(f);
    }
    Value::String(text.to_string())
}

pub fn show_text(path: &Path, settings: &Map<String, Value>) -> String {
    let mut s = format!("path: {}\n", path.display());
    for (k, v) in settings {
        if k != "env" {
            s += &format!("{k} = {}\n", py_str(v));
        }
    }
    if let Some(Value::Object(e)) = settings.get("env") {
        if !e.is_empty() {
            s += "[env]\n";
            for (k, v) in e {
                s += &format!("{k} = {}\n", py_str(v));
            }
        }
    }
    s
}

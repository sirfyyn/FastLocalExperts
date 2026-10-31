//! Settings: one JSON file in the data folder, merged over the defaults. Port of `fle_studio/config.py`.
//!
//! Only keys that exist in the defaults can be set, and every value is coerced to the type of its default, so a
//! hand-edited or outdated file cannot put the app into a state the page does not know. Secrets never leave the
//! server: `public()` replaces them with flags.
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const SECRETS: [&str; 3] = ["hf_token", "access_key", "attach_key"];

pub fn defaults() -> Map<String, Value> {
    let Value::Object(m) = json!({
        "models_dir": "",
        "server": {"host": "127.0.0.1", "port": 8642, "open_browser": true, "allowed_hosts": "", "tls": true},
        "remote": {"name": ""},
        "access_key": "",
        "hf_token": "",
        "engine": {"choice": "auto", "custom_path": "", "upstream_tag": "b11540", "fle_tag": "latest"},
        "launch": {
            "profile": "auto", "context": 32768, "parallel": 1, "mtp": true, "spec": "auto", "gpu_cache_mib": 0, "gpus": "auto",
            "threads": 0, "cpu_mask": "", "pinned_load": "auto", "prompt_cache_mib": -1, "port": 0, "extra_args": "", "extra_env": "",
            "restart_on_crash": false, "autostart": false
        },
        "chat": {
            "sampling": "model", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "max_tokens": 0, "thinking": true, "system_prompt": "", "max_upload_mb": 25
        },
        "tools": {
            "enabled": true, "max_rounds": 8, "call_timeout": 60, "calculator": true, "datetime": true, "files": false, "files_dir": "",
            "web_search": false, "web_search_confirm": true, "web_search_url": "", "mcp_enabled": true, "mcp_json": "", "mcp_trusted": []
        },
        "forge": {"tools_dir": ""},
        "attach_url": "",
        "attach_service": "",
        "attach_key": "",
        "attach_auto": false,
        "last_model": "",
        "theme": "system",
        "updates": {"check": true, "prereleases": false, "dismissed": "", "dismissed_engines": ""}
    }) else {
        unreachable!()
    };
    m
}

/// Python's `repr()` for the values that can come out of a JSON body (used in the error text the page shows).
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if s.contains('\'') && !s.contains('"') {
                format!("\"{}\"", s.replace('\\', "\\\\"))
            } else {
                format!(
                    "'{}'",
                    s.replace('\\', "\\\\")
                        .replace('\'', "\\'")
                        .replace('\n', "\\n")
                        .replace('\r', "\\r")
                        .replace('\t', "\\t")
                )
            }
        }
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, x)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(x)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `str(value)` in Python.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Coerce to the type of the default; `Err(())` where Python would raise ValueError or TypeError.
fn coerce(value: &Value, default: &Value) -> Result<Value, ()> {
    match default {
        Value::Bool(_) => Ok(Value::Bool(match value {
            Value::String(s) => {
                ["1", "true", "yes", "on"].contains(&s.trim().to_lowercase().as_str())
            }
            other => py_truthy(other),
        })),
        Value::Number(d) if d.is_i64() || d.is_u64() => {
            // int(float(value))
            let f = match value {
                Value::Bool(b) => f64::from(u8::from(*b)),
                Value::Number(n) => n.as_f64().ok_or(())?,
                Value::String(s) => s.trim().parse::<f64>().map_err(|_| ())?,
                _ => return Err(()),
            };
            if !f.is_finite() {
                return Err(());
            }
            Ok(json!(f as i64))
        }
        Value::Number(_) => {
            let f = match value {
                Value::Bool(b) => f64::from(u8::from(*b)),
                Value::Number(n) => n.as_f64().ok_or(())?,
                Value::String(s) => s.trim().parse::<f64>().map_err(|_| ())?,
                _ => return Err(()),
            };
            Ok(json!(f))
        }
        Value::String(_) => Ok(Value::String(if value.is_null() {
            String::new()
        } else {
            py_str(value)
        })),
        _ => Ok(value.clone()),
    }
}

/// `base` with `patch` applied. Unknown keys are dropped; with `strict` a bad value is an error with Python's wording.
pub fn merge(
    base: &Map<String, Value>,
    patch: &Map<String, Value>,
    strict: bool,
) -> Result<Map<String, Value>, String> {
    let mut out = base.clone();
    for (key, value) in patch {
        let Some(b) = base.get(key) else { continue };
        if let Value::Object(bm) = b {
            if let Value::Object(pm) = value {
                out.insert(key.clone(), Value::Object(merge(bm, pm, strict)?));
            }
            continue;
        }
        match coerce(value, b) {
            Ok(v) => {
                out.insert(key.clone(), v);
            }
            Err(()) if strict => {
                return Err(format!("invalid value for {key}: {}", py_repr(value)))
            }
            Err(()) => {}
        }
    }
    Ok(out)
}

pub struct Config {
    pub path: PathBuf,
    pub data: Map<String, Value>,
}

impl Config {
    pub fn load(path: &Path) -> Config {
        let mut data = defaults();
        if path.exists() {
            let parsed = std::fs::read_to_string(path)
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok());
            match parsed {
                Some(Value::Object(p)) => {
                    data = merge(&defaults(), &p, false).unwrap_or_else(|_| defaults())
                }
                // A broken file must not stop the app; keep it for the user and start from defaults.
                _ => {
                    let _ = std::fs::rename(path, path.with_extension("broken.json"));
                }
            }
        }
        Config {
            path: path.to_path_buf(),
            data,
        }
    }

    pub fn get(&self, dotted: &str) -> Option<&Value> {
        let mut node: &Value = &Value::Null;
        let mut first = true;
        for part in dotted.split('.') {
            node = if first {
                self.data.get(part)?
            } else {
                node.get(part)?
            };
            first = false;
        }
        Some(node)
    }

    pub fn update(&mut self, patch: &Map<String, Value>) -> Result<Value, String> {
        self.data = merge(&self.data, patch, true)?;
        self.save().map_err(|e| e.to_string())?;
        Ok(self.public())
    }

    fn save(&self) -> std::io::Result<()> {
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = self.path.with_extension("tmp");
        std::fs::write(
            &tmp,
            serde_json::to_string_pretty(&Value::Object(self.data.clone())).unwrap(),
        )?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
            // the file can hold a Hugging Face token
        }
        std::fs::rename(&tmp, &self.path)
    }

    pub fn models_dir(&self) -> PathBuf {
        match self.data.get("models_dir").and_then(Value::as_str) {
            Some(s) if !s.is_empty() => match s.strip_prefix("~/") {
                Some(r) => dirs::home_dir().unwrap_or_default().join(r),
                None => PathBuf::from(s),
            },
            _ => self.path.parent().unwrap_or(Path::new(".")).join("models"),
        }
    }

    pub fn public(&self) -> Value {
        let mut out = self.data.clone();
        for key in SECRETS {
            let has = out
                .shift_remove(key)
                .as_ref()
                .map(py_truthy)
                .unwrap_or(false);
            out.insert(format!("has_{key}"), Value::Bool(has));
        }
        out.insert(
            "models_dir".into(),
            Value::String(self.models_dir().to_string_lossy().to_string()),
        );
        Value::Object(out)
    }
}

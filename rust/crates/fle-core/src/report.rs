//! A diagnostic report a user can attach to a bug report without leaking keys. Port of `tools/fle/report.py`.
use crate::config;
use crate::FleError;
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::io::Write;
use std::path::PathBuf;

pub const TOOL_VERSION: &str = "0.1";

/// Remove secrets: any key that names a token, key, secret, password or credential.
pub fn redact(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| {
                    let up = k.to_uppercase();
                    if ["TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL"]
                        .iter()
                        .any(|w| up.contains(w))
                    {
                        (k.clone(), json!("<redacted>"))
                    } else {
                        (k.clone(), redact(x))
                    }
                })
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(redact).collect()),
        other => other.clone(),
    }
}

/// `2026-10-31T09:41:23+00:00`, like `datetime.isoformat(timespec="seconds")` in UTC.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// `doctor` plus the tool, the time and the redacted settings. `doctor` is the result of `doctor::run`.
pub fn build(mut report: Value, settings: &Map<String, Value>) -> Result<Value> {
    let (raw, path) = config::load(None)?;
    let r = report.as_object_mut().unwrap();
    r.insert("tool".into(), json!("fle"));
    r.insert("tool_version".into(), json!(TOOL_VERSION));
    r.insert("generated".into(), json!(utc_now()));
    r.insert("config_file".into(), json!(path.to_string_lossy()));
    let mut shown: Map<String, Value> = settings
        .iter()
        .filter(|(k, _)| *k != "env")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    shown = redact(&Value::Object(shown))
        .as_object()
        .cloned()
        .unwrap_or_default();
    shown.insert(
        "env".into(),
        redact(settings.get("env").unwrap_or(&json!({}))),
    );
    r.insert("settings".into(), Value::Object(shown));
    r.insert("file_settings".into(), redact(&Value::Object(raw)));
    Ok(report)
}

fn checks_text(r: &Value) -> String {
    let mut lines = vec![format!(
        "fle doctor: {}",
        r["status"].as_str().unwrap_or("None")
    )];
    for c in r["checks"].as_array().into_iter().flatten() {
        lines.push(format!(
            "[{}] {}: {}",
            c["status"].as_str().unwrap_or(""),
            c["name"].as_str().unwrap_or(""),
            c["detail"].as_str().unwrap_or("")
        ));
        if let Some(h) = c["hint"].as_str().filter(|h| !h.is_empty()) {
            lines.push(format!("  -> {h}"));
        }
    }
    lines.join("\n") + "\n"
}

/// Write the report. `-` prints it (returns "-"); a `.zip` path (or `as_zip`) gets report.json, plan.sh and checks.txt.
pub fn write(report: &Value, out: &str, as_zip: bool) -> Result<String> {
    let text = crate::gguf::py_json(report);
    if out == "-" {
        println!("{text}");
        return Ok("-".into());
    }
    let path = match out.strip_prefix("~/") {
        Some(r) => dirs::home_dir().unwrap_or_default().join(r),
        None => PathBuf::from(out),
    };
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| {
            FleError::new(
                format!("Cannot write {}: {e}", path.display()),
                "Check the path.",
            )
        })?;
    }
    let is_zip = as_zip
        || path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("zip"));
    if is_zip {
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&path)?);
        let opt = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zw.start_file("report.json", opt)?;
        zw.write_all(text.as_bytes())?;
        if let Some(cmd) = report["plan"]["command"].as_str().filter(|c| !c.is_empty()) {
            zw.start_file("plan.sh", opt)?;
            zw.write_all(format!("#!/usr/bin/env sh\n{cmd}\n").as_bytes())?;
        }
        zw.start_file("checks.txt", opt)?;
        zw.write_all(checks_text(report).as_bytes())?;
        zw.finish()?;
    } else {
        std::fs::write(&path, format!("{text}\n"))?;
    }
    Ok(path.to_string_lossy().to_string())
}

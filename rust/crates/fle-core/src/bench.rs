//! A small OpenAI-compatible benchmark: tokens per second and latency through /v1. Port of `tools/fle/bench.py`.
use crate::FleError;
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::time::{Duration, Instant};

pub const DEFAULT_MAX_REGRESSION: f64 = 10.0;

fn request(url: &str, payload: Option<&Value>, timeout: f64) -> Result<Value> {
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout.max(0.001)))
        .build()?;
    let req = match payload {
        Some(p) => c.post(url).json(p),
        None => c.get(url),
    };
    let resp = req.send().map_err(|e| {
        FleError::new(
            format!("Cannot reach {url}: {e}"),
            "Start the server first, or pass --url.",
        )
    })?;
    let status = resp.status();
    let body = resp.text().map_err(|e| {
        FleError::new(
            format!("Cannot reach {url}: {e}"),
            "Start the server first, or pass --url.",
        )
    })?;
    if !status.is_success() {
        return Err(FleError::new(
            format!(
                "{url} returned HTTP {}: {}",
                status.as_u16(),
                body.chars().take(500).collect::<String>()
            ),
            "Check the server log.",
        )
        .into());
    }
    serde_json::from_str(&body).map_err(|e| {
        FleError::new(
            format!("Cannot reach {url}: {e}"),
            "Start the server first, or pass --url.",
        )
        .into()
    })
}

fn model_name(base: &str, wanted: &str) -> Result<String> {
    if !wanted.is_empty() {
        return Ok(wanted.into());
    }
    let body = request(
        &format!("{}/v1/models", base.trim_end_matches('/')),
        None,
        600.0,
    )?;
    let items = body["data"].as_array().cloned().unwrap_or_default();
    let Some(first) = items.first() else {
        return Err(FleError::new(
            format!("The server at {base} named no model."),
            "Pass --model NAME.",
        )
        .into());
    };
    let name = first["id"].as_str().unwrap_or("").trim().to_string();
    if name.is_empty() {
        return Err(FleError::new(
            format!("The server at {base} returned an empty model name."),
            "Pass --model NAME.",
        )
        .into());
    }
    Ok(name)
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}
fn int(v: &Value) -> i64 {
    v.as_i64()
        .or_else(|| v.as_f64().map(|f| f as i64))
        .unwrap_or(0)
}
fn or0(a: &Value, b: &Value) -> Value {
    if num(a) != 0.0 {
        a.clone()
    } else {
        b.clone()
    }
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
/// Python's `round(x, 4)`.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

pub struct Params {
    pub url: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: i64,
    pub runs: i64,
    pub warmup: i64,
    pub timeout: f64,
}
impl Params {
    pub fn new(url: &str) -> Self {
        Params {
            url: url.into(),
            model: String::new(),
            prompt: "Write a short paragraph about why local language models are useful.".into(),
            max_tokens: 128,
            runs: 3,
            warmup: 1,
            timeout: 600.0,
        }
    }
}

pub fn run(p: &Params) -> Result<Value> {
    let mut base = p.url.trim_end_matches('/').to_string();
    if !base.starts_with("http://") && !base.starts_with("https://") {
        base = format!("http://{base}");
    }
    let name = model_name(&base, &p.model)?;
    let payload = json!({"model": name, "messages": [{"role": "user", "content": p.prompt}], "max_tokens": p.max_tokens.max(1), "temperature": 0, "stream": false});
    let mut samples: Vec<Value> = vec![];
    for index in 0..(p.warmup.max(0) + p.runs.max(1)) {
        let started = Instant::now();
        let body = request(
            &format!("{base}/v1/chat/completions"),
            Some(&payload),
            p.timeout,
        )?;
        let elapsed = started.elapsed().as_secs_f64();
        let (usage, timings) = (&body["usage"], &body["timings"]);
        let completion = int(&or0(&usage["completion_tokens"], &timings["predicted_n"]));
        let mut tok_s = num(&timings["predicted_per_second"]);
        if tok_s == 0.0 && completion > 0 {
            tok_s = if elapsed > 0.0 {
                completion as f64 / elapsed
            } else {
                0.0
            };
        }
        let draft_n = int(&timings["draft_n"]);
        let warm = index < p.warmup;
        let sample = json!({
            "warmup": warm, "latency_ms": elapsed * 1000.0, "completion_tokens": completion,
            "prompt_tokens": int(&or0(&usage["prompt_tokens"], &timings["prompt_n"])), "tok_s": tok_s,
            "predicted_ms": num(&timings["predicted_ms"]), "prompt_tok_s": num(&timings["prompt_per_second"]),
            "finish_reason": body["choices"][0]["finish_reason"].as_str().unwrap_or(""),
            "accept": if draft_n != 0 { json!(round4(int(&timings["draft_n_accepted"]) as f64 / draft_n as f64)) } else { Value::Null },
        });
        if !warm {
            samples.push(sample);
        }
    }
    if samples.is_empty() {
        return Err(FleError::new("No measured run finished.", "Check the server.").into());
    }
    let mut toks: Vec<f64> = samples
        .iter()
        .map(|s| num(&s["tok_s"]))
        .filter(|t| *t > 0.0)
        .collect();
    let mut lat: Vec<f64> = samples.iter().map(|s| num(&s["latency_ms"])).collect();
    let (mn, mx) = (
        toks.iter().cloned().fold(f64::INFINITY, f64::min),
        toks.iter().cloned().fold(0.0, f64::max),
    );
    let truncated = samples
        .iter()
        .filter(|s| s["finish_reason"] == "length")
        .count();
    Ok(json!({
        "url": base, "model": name, "prompt": p.prompt, "max_tokens": p.max_tokens, "runs": samples,
        "median_tok_s": if toks.is_empty() { 0.0 } else { median(&mut toks) }, "median_latency_ms": median(&mut lat),
        "min_tok_s": if toks.is_empty() { 0.0 } else { mn }, "max_tok_s": if toks.is_empty() { 0.0 } else { mx }, "truncated": truncated,
    }))
}

/// 'http-local', 'https-remote' and so on: runs are only comparable when the kind matches.
pub fn url_kind(url: &str) -> String {
    let full = if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    };
    let (scheme, rest) = full.split_once("://").unwrap();
    let auth = rest.split(['/', '?', '#']).next().unwrap_or("");
    let auth = auth.rsplit('@').next().unwrap_or("");
    let host = if let Some(r) = auth.strip_prefix('[') {
        r.split(']').next().unwrap_or("").to_string()
    } else {
        auth.split(':').next().unwrap_or("").to_string()
    };
    let host = host.to_lowercase();
    let scheme = if scheme.is_empty() {
        "http".to_string()
    } else {
        scheme.to_lowercase()
    };
    let local = ["127.0.0.1", "localhost", "::1", "0.0.0.0"].contains(&host.as_str());
    format!("{scheme}-{}", if local { "local" } else { "remote" })
}

pub fn load_baseline(path: &str) -> Result<Value> {
    let p = match path.strip_prefix("~/") {
        Some(r) => dirs::home_dir().unwrap_or_default().join(r),
        None => path.into(),
    };
    let text = std::fs::read_to_string(&p).map_err(|e| {
        FleError::new(
            format!("Cannot read baseline {path}: {e}"),
            "Check the path.",
        )
    })?;
    let mut data: Value = serde_json::from_str(&text).map_err(|e| {
        FleError::new(
            format!("{path} is not valid JSON: {e}"),
            "Use a file written by `fle bench --output`.",
        )
    })?;
    if data["primary"].is_object() {
        data = data["primary"].clone();
    }
    if !data.is_object() || !data["median_tok_s"].is_number() {
        return Err(FleError::new(
            format!("{path} has no bench result (median_tok_s is missing)."),
            "Use a file written by `fle bench --output`.",
        )
        .into());
    }
    Ok(data)
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
fn repr(v: &Value) -> String {
    match v {
        Value::String(s) => {
            if s.contains('\'') && !s.contains('"') {
                format!("\"{s}\"")
            } else {
                format!("'{}'", s.replace('\'', "\\'"))
            }
        }
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

/// Compare the median tok/s of two results: "ok", "regression" or "skipped" (the runs cannot be compared).
pub fn compare_baseline(current: &Value, baseline: &Value, max_regression: f64) -> Value {
    let (now, before) = (
        num(&current["median_tok_s"]),
        num(&baseline["median_tok_s"]),
    );
    let mut o = Map::new();
    let (mut status, mut reason, mut delta) = ("skipped".to_string(), String::new(), Value::Null);
    let (uc, ub) = (
        url_kind(current["url"].as_str().unwrap_or("")),
        url_kind(baseline["url"].as_str().unwrap_or("")),
    );
    if truthy(&current["truncated"]) {
        reason = "the current run hit the token limit (truncated), so its speed is not comparable"
            .into();
    } else if truthy(&baseline["truncated"]) {
        reason = "the baseline run hit the token limit (truncated), so its speed is not comparable"
            .into();
    } else if current["model"] != baseline["model"] {
        reason = format!(
            "different model ({} in the baseline, {} now)",
            repr(&baseline["model"]),
            repr(&current["model"])
        );
    } else if uc != ub {
        reason = format!("different kind of URL ({ub} in the baseline, {uc} now)");
    } else if before <= 0.0 || now <= 0.0 {
        reason = "one side has no tokens per second".into();
    } else {
        let d = (now - before) / before * 100.0;
        status = if d < -max_regression.abs() {
            "regression"
        } else {
            "ok"
        }
        .into();
        delta = json!(d);
    }
    o.insert("status".into(), json!(status));
    o.insert("reason".into(), json!(reason));
    o.insert("baseline_tok_s".into(), json!(before));
    o.insert("current_tok_s".into(), json!(now));
    o.insert("delta_pct".into(), delta);
    o.insert("max_regression_pct".into(), json!(max_regression));
    Value::Object(o)
}

/// Python's `:g` for a float (shortest form, no trailing zeros).
fn fmt_g(x: f64) -> String {
    let s = format!("{x}");
    s
}

pub fn comparison_text(r: &Value) -> String {
    if r["status"] == "skipped" {
        return format!(
            "baseline: not compared, {}.\n",
            r["reason"].as_str().unwrap_or("")
        );
    }
    format!(
        "baseline: {:.2} -> {:.2} tok/s ({:+.1} %, allowed drop {} %): {}\n",
        num(&r["baseline_tok_s"]),
        num(&r["current_tok_s"]),
        num(&r["delta_pct"]),
        fmt_g(num(&r["max_regression_pct"]).abs()),
        if r["status"] == "ok" {
            "ok"
        } else {
            "REGRESSION"
        }
    )
}

pub fn result_text(r: &Value) -> String {
    use crate::launch::human_int;
    let mut s = format!(
        "model: {}  url: {}\n",
        r["model"].as_str().unwrap_or(""),
        r["url"].as_str().unwrap_or("")
    );
    s += &format!(
        "{:>3}  {:>7}  {:>8}  {:>10}  {:<8}  {:>7}\n",
        "run", "tokens", "tok/s", "latency", "finish", "accept"
    );
    for (i, x) in r["runs"].as_array().into_iter().flatten().enumerate() {
        let accept = if x["accept"].is_null() {
            "-".to_string()
        } else {
            format!("{:.3}", num(&x["accept"]))
        };
        let fin = x["finish_reason"]
            .as_str()
            .filter(|f| !f.is_empty())
            .unwrap_or("-");
        s += &format!(
            "{:>3}  {:>7}  {:>8.2}  {:>8.0} ms  {:<8}  {:>7}\n",
            i + 1,
            human_int(int(&x["completion_tokens"])),
            num(&x["tok_s"]),
            num(&x["latency_ms"]),
            fin,
            accept
        );
    }
    s += &format!(
        "median: {:.2} tok/s, {:.0} ms (min {:.2}, max {:.2})\n",
        num(&r["median_tok_s"]),
        num(&r["median_latency_ms"]),
        num(&r["min_tok_s"]),
        num(&r["max_tok_s"])
    );
    if truthy(&r["truncated"]) {
        s += &format!(
            "warning: {} run(s) hit the token limit; the answers may be cut off.\n",
            r["truncated"]
        );
    }
    s
}

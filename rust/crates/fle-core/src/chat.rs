use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub decode_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub tokens: u64,
}

/// Stream one chat answer from an OpenAI-compatible server. `on_text(is_reasoning, text)` gets every piece.
pub fn stream(
    base: &str,
    model: &str,
    history: &[(String, String)],
    on_text: &mut dyn FnMut(bool, &str),
) -> Result<Stats> {
    let msgs: Vec<Value> = history
        .iter()
        .map(|(r, c)| json!({"role": r, "content": c}))
        .collect();
    let body = json!({"model": model, "messages": msgs, "stream": true, "stream_options": {"include_usage": true}});
    let c = reqwest::blocking::Client::builder().timeout(None).build()?;
    let r = c
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()?;
    if !r.status().is_success() {
        bail!("the engine answered HTTP {}", r.status());
    }
    let mut st = Stats::default();
    for line in BufReader::new(r).lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(d) = v["choices"][0]["delta"].as_object() {
            if let Some(t) = d
                .get("reasoning_content")
                .and_then(|x| x.as_str())
                .filter(|t| !t.is_empty())
            {
                on_text(true, t);
            }
            if let Some(t) = d
                .get("content")
                .and_then(|x| x.as_str())
                .filter(|t| !t.is_empty())
            {
                on_text(false, t);
            }
        }
        if let Some(t) = v.get("timings") {
            st.decode_tps = t["predicted_per_second"].as_f64().or(st.decode_tps);
            st.prefill_tps = t["prompt_per_second"].as_f64().or(st.prefill_tps);
            st.tokens = t["predicted_n"].as_u64().unwrap_or(st.tokens);
        }
    }
    Ok(st)
}

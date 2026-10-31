//! Live inference data, sampled about once a second. Port of `fle_studio/metrics.py`.
//!
//! Speeds come from the counters in llama-server's /metrics: tokens generated per second of generation time, prompt tokens
//! per second of prompt time, and the share of drafted (MTP) tokens that were accepted, each over the last interval. A server
//! started without --metrics has no such counters; then the timings of the last answer stand in.
use crate::engine::{read_and_memory, Engine};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const P: &str = "llamacpp:";

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Sum of the values per metric name (labels are dropped).
pub fn parse_prometheus(text: &str) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line.rsplit_once(' ') else {
            continue;
        };
        let name = name.split('{').next().unwrap_or("").trim().to_string();
        if let Ok(v) = value.trim().parse::<f64>() {
            *out.entry(name).or_insert(0.0) += v;
        }
    }
    out
}

fn round(x: f64, digits: i32) -> f64 {
    format!("{:.*}", digits as usize, x).parse().unwrap_or(x)
}

/// What changed between two readings of the counters (`_rates`). `None` for `prev` is a first reading.
pub fn rates(
    prev: Option<&HashMap<String, f64>>,
    cur: &HashMap<String, f64>,
) -> Map<String, Value> {
    let g = |m: &HashMap<String, f64>, k: &str| m.get(&format!("{P}{k}")).copied().unwrap_or(0.0);
    let mut out = Map::new();
    out.insert("busy".into(), json!(g(cur, "requests_processing") > 0.0));
    out.insert("deferred".into(), json!(g(cur, "requests_deferred") as i64));
    let Some(prev) = prev else { return out };
    let d = |k: &str| g(cur, k) - g(prev, k);
    if d("tokens_predicted_total") < 0.0 || d("prompt_tokens_total") < 0.0 {
        return out; // the server restarted
    }
    let (gen_n, gen_s) = (
        d("tokens_predicted_total"),
        d("tokens_predicted_seconds_total"),
    );
    let (pp_n, pp_s) = (d("prompt_tokens_total"), d("prompt_seconds_total"));
    let (drafted, accepted) = (
        d("spec_decode_num_draft_tokens_total"),
        d("spec_decode_num_accepted_tokens_total"),
    );
    if gen_n > 0.0 && gen_s > 0.0 {
        out.insert("decode".into(), json!(round(gen_n / gen_s, 2)));
        out.insert("generated".into(), json!(gen_n as i64));
    }
    if pp_n > 0.0 && pp_s > 0.0 {
        out.insert("prefill".into(), json!(round(pp_n / pp_s, 1)));
        out.insert("prompt_tokens".into(), json!(pp_n as i64));
    }
    if drafted > 0.0 {
        out.insert("accept".into(), json!(round(accepted / drafted, 4)));
    }
    out
}

struct Inner {
    history: VecDeque<Value>,
    responses: VecDeque<Value>,
    metrics_available: Option<bool>,
    prev: Option<HashMap<String, f64>>,
    prev_url: String,
    gpus: Vec<Value>,
    totals: Value,
}

#[derive(Clone)]
pub struct Live {
    engine: Engine,
    inner: Arc<Mutex<Inner>>,
}

impl Live {
    pub fn new(engine: Engine, interval: f64) -> Live {
        let l = Live {
            engine,
            inner: Arc::new(Mutex::new(Inner {
                history: VecDeque::new(),
                responses: VecDeque::new(),
                metrics_available: None,
                prev: None,
                prev_url: String::new(),
                gpus: vec![],
                totals: json!({}),
            })),
        };
        let me = l.clone();
        std::thread::Builder::new()
            .name("live-sampler".into())
            .spawn(move || {
                let has_nvidia = !crate::system::nvidia().is_empty();
                let mut tick = 0u64;
                loop {
                    let t0 = Instant::now();
                    me.sample(tick, has_nvidia);
                    tick += 1;
                    let spent = t0.elapsed().as_secs_f64();
                    std::thread::sleep(Duration::from_secs_f64((interval - spent).max(0.2)));
                }
            })
            .expect("live thread");
        l
    }

    /// Called by the chat proxy with the `timings` of every finished answer.
    pub fn record_response(&self, timings: &Value, usage: Option<&Value>) {
        let mut e = Map::new();
        e.insert("t".into(), json!(now()));
        for k in [
            "prompt_n",
            "prompt_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_second",
            "draft_n",
            "draft_n_accepted",
            "cache_n",
        ] {
            e.insert(k.into(), timings.get(k).cloned().unwrap_or(Value::Null));
        }
        if let Some(u) = usage.filter(|u| crate::settings::py_truthy(u)) {
            e.insert("usage".into(), u.clone());
        }
        let mut i = self.inner.lock().unwrap();
        i.responses.push_back(Value::Object(e));
        if i.responses.len() > 100 {
            i.responses.pop_front();
        }
    }

    fn get(&self, base: &str, path: &str, key: &str) -> (i64, String) {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(2))
            .build();
        let mut req = agent.get(&format!("{base}{path}"));
        if !key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        match req.call() {
            Ok(r) => {
                let c = i64::from(r.status());
                (c, r.into_string().unwrap_or_default())
            }
            Err(ureq::Error::Status(c, _)) => (i64::from(c), String::new()),
            Err(_) => (0, String::new()),
        }
    }

    fn sample(&self, tick: u64, has_nvidia: bool) {
        let bound = self.engine.bound();
        if bound == "url" && tick % 5 == 0 {
            self.engine.refresh_attached();
        }
        let (state, base, key) = self.engine.target();
        let mut s = Map::new();
        s.insert("t".into(), json!(round(now(), 2)));
        s.insert("state".into(), json!(state));
        let mem = crate::system::memory();
        let (total, avail) = (
            mem["total"].as_u64().unwrap_or(0),
            mem["available"].as_u64().unwrap_or(0),
        );
        s.insert("ram_used".into(), json!(total as i64 - avail as i64));
        s.insert("ram_total".into(), json!(total));
        if has_nvidia && (tick % 2 == 0 || self.inner.lock().unwrap().gpus.is_empty()) {
            let gpus: Vec<Value> = crate::system::nvidia()
                .into_iter()
                .map(|g| {
                    Value::Object(
                        [
                            "index",
                            "name",
                            "utilization",
                            "memory_used",
                            "memory_total",
                            "power_w",
                            "temperature_c",
                        ]
                        .iter()
                        .map(|k| (k.to_string(), g.get(*k).cloned().unwrap_or(Value::Null)))
                        .collect(),
                    )
                })
                .collect();
            self.inner.lock().unwrap().gpus = gpus;
        }
        s.insert(
            "gpus".into(),
            Value::Array(self.inner.lock().unwrap().gpus.clone()),
        );
        if let Some(pid) = self.engine.proc_pid() {
            let rss = read_and_memory(pid).1;
            s.insert("engine_mem".into(), json!(rss));
            self.engine.note_memory(rss);
        }
        let live = state == "ready" || state == "attached";
        if live && !base.is_empty() {
            {
                let mut i = self.inner.lock().unwrap();
                if base != i.prev_url {
                    // An engine Studio started counts from zero, so even an answer that finished before the first sample shows
                    // up. A server it attached to has run before: its first sample is the baseline.
                    i.prev = if state == "ready" {
                        Some(HashMap::new())
                    } else {
                        None
                    };
                    i.prev_url = base.clone();
                    i.metrics_available = None;
                }
            }
            let (code, text) = self.get(&base, "/metrics", &key);
            if code == 200 && !text.is_empty() {
                let cur = parse_prometheus(&text);
                let mut i = self.inner.lock().unwrap();
                i.metrics_available = Some(true);
                let r = rates(i.prev.as_ref(), &cur);
                i.prev = Some(cur.clone());
                for (k, v) in r {
                    s.insert(k, v);
                }
                let g = |k: &str| cur.get(&format!("{P}{k}")).copied().unwrap_or(0.0);
                i.totals = json!({"generated": g("tokens_predicted_total"), "prompt": g("prompt_tokens_total"), "decode_calls": g("n_decode_total")});
            } else if code == 404 || code == 501 {
                self.inner.lock().unwrap().metrics_available = Some(false);
            }
            if tick % 2 == 0 {
                let (code, text) = self.get(&base, "/slots", &key);
                if code == 200 && text.starts_with('[') {
                    if let Ok(Value::Array(slots)) = serde_json::from_str::<Value>(&text) {
                        s.insert(
                            "slots_busy".into(),
                            json!(slots
                                .iter()
                                .filter(|x| crate::settings::py_truthy(&x["is_processing"]))
                                .count()),
                        );
                        s.insert("slots".into(), json!(slots.len()));
                    }
                }
            }
        } else {
            let mut i = self.inner.lock().unwrap();
            i.prev = None; // the next start (same port or not) begins a new count
            i.prev_url.clear();
        }
        if let Some(h) = self
            .engine
            .cache_snapshot()
            .get("hit_rate")
            .filter(|h| !h.is_null())
        {
            s.insert("cache_hit".into(), h.clone());
        }
        let mut i = self.inner.lock().unwrap();
        if i.metrics_available == Some(false) {
            if let Some(last) = i.responses.back().cloned() {
                s.insert("last_answer".into(), json!(true));
                s.insert("decode".into(), last["predicted_per_second"].clone());
                s.insert("prefill".into(), last["prompt_per_second"].clone());
                if crate::settings::py_truthy(&last["draft_n"]) {
                    s.insert(
                        "accept".into(),
                        json!(
                            last["draft_n_accepted"].as_f64().unwrap_or(0.0)
                                / last["draft_n"].as_f64().unwrap_or(1.0)
                        ),
                    );
                }
            }
        }
        i.history.push_back(Value::Object(s));
        if i.history.len() > 900 {
            i.history.pop_front();
        }
    }

    pub fn snapshot(&self, since: f64) -> Value {
        let i = self.inner.lock().unwrap();
        let hist: Vec<Value> = if since != 0.0 {
            i.history
                .iter()
                .filter(|h| h["t"].as_f64().unwrap_or(0.0) > since)
                .cloned()
                .collect()
        } else {
            i.history.iter().cloned().collect()
        };
        json!({"now": now(), "history": hist, "metrics_available": i.metrics_available, "last_response": i.responses.back().cloned(),
            "totals": i.totals, "cache": self.engine.cache_snapshot(), "accept_logged": self.engine.accept_logged()})
    }
}

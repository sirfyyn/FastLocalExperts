//! The /v1 pass-through and the live numbers against the Python Studio, with a small fake engine server of our own. Also the
//! counter arithmetic (Prometheus text, rates between two readings) against the Python functions.
#![cfg(unix)]
mod harness;
use fle_api::live::{parse_prometheus, rates};
use harness::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const DRIVER: &str = r#"
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fle_studio import metrics
from fle_studio.engine import Engine
sc = json.load(open(sys.argv[2]))
out = {"prom": [metrics.parse_prometheus(t) for t in sc["texts"]], "rates": []}
live = metrics.Live(Engine(Path(tempfile.mkdtemp())), interval=3600)
for prev, cur in sc["pairs"]:
    live._prev = prev
    out["rates"].append(live._rates(cur))
print(json.dumps(out))
"#;

#[test]
fn counters_and_rates_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found: skipped");
        return;
    };
    let texts = vec![
        "# HELP x y\n# TYPE x counter\nllamacpp:tokens_predicted_total 120\nllamacpp:prompt_tokens_total 40\nllamacpp:n_decode_total 7\n".to_string(),
        "llamacpp:requests_processing{slot=\"0\"} 1\nllamacpp:requests_processing{slot=\"1\"} 2\nllamacpp:requests_deferred 3\nbad line\nname notanumber\n\n".to_string(),
        "a 1\na 2.5\nb{x=\"y z\"} 4\n".to_string(),
        "".to_string(),
    ];
    let m = |pairs: &[(&str, f64)]| -> Value {
        Value::Object(
            pairs
                .iter()
                .map(|(k, v)| (format!("llamacpp:{k}"), json!(v)))
                .collect(),
        )
    };
    let base = m(&[
        ("tokens_predicted_total", 100.0),
        ("tokens_predicted_seconds_total", 4.0),
        ("prompt_tokens_total", 50.0),
        ("prompt_seconds_total", 1.0),
        ("spec_decode_num_draft_tokens_total", 20.0),
        ("spec_decode_num_accepted_tokens_total", 10.0),
    ]);
    let more = m(&[
        ("tokens_predicted_total", 220.0),
        ("tokens_predicted_seconds_total", 7.0),
        ("prompt_tokens_total", 90.0),
        ("prompt_seconds_total", 2.0),
        ("spec_decode_num_draft_tokens_total", 50.0),
        ("spec_decode_num_accepted_tokens_total", 31.0),
        ("requests_processing", 1.0),
        ("requests_deferred", 2.0),
    ]);
    let restarted = m(&[
        ("tokens_predicted_total", 5.0),
        ("prompt_tokens_total", 1.0),
    ]);
    let idle = m(&[
        ("tokens_predicted_total", 100.0),
        ("tokens_predicted_seconds_total", 4.0),
        ("prompt_tokens_total", 50.0),
    ]);
    let thirds = m(&[
        ("tokens_predicted_total", 103.0),
        ("tokens_predicted_seconds_total", 4.3),
        ("prompt_tokens_total", 51.0),
        ("prompt_seconds_total", 1.0003),
    ]);
    let pairs: Vec<(Value, Value)> = vec![
        (Value::Null, more.clone()),
        (base.clone(), more.clone()),
        (more.clone(), restarted.clone()),
        (base.clone(), idle.clone()),
        (base.clone(), thirds),
        (json!({}), more.clone()),
        (json!({}), json!({})),
        (base.clone(), base.clone()),
    ];
    let d = tmp("proxy-pure");
    std::fs::write(d.join("driver.py"), DRIVER).unwrap();
    std::fs::write(
        d.join("in.json"),
        json!({"texts": texts, "pairs": pairs}).to_string(),
    )
    .unwrap();
    let out = std::process::Command::new(
        std::env::var("FLE_STUDIO_PYTHON").unwrap_or_else(|_| "python3".into()),
    )
    .arg(d.join("driver.py"))
    .arg(&src)
    .arg(d.join("in.json"))
    .output()
    .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py: Value = serde_json::from_slice(&out.stdout).unwrap();
    for (i, t) in texts.iter().enumerate() {
        let got: Value = Value::Object(
            parse_prometheus(t)
                .into_iter()
                .map(|(k, v)| (k, json!(v)))
                .collect(),
        );
        assert_eq!(got, py["prom"][i], "parse_prometheus #{i}");
    }
    let to_map = |v: &Value| -> HashMap<String, f64> {
        v.as_object()
            .unwrap()
            .iter()
            .map(|(k, x)| (k.clone(), x.as_f64().unwrap()))
            .collect()
    };
    for (i, (prev, cur)) in pairs.iter().enumerate() {
        let p = if prev.is_null() {
            None
        } else {
            Some(to_map(prev))
        };
        let got = Value::Object(rates(p.as_ref(), &to_map(cur)));
        assert_eq!(got, py["rates"][i], "rates #{i}");
    }
    eprintln!(
        "compared with the Python Studio: {} metric texts and {} rate readings identical",
        texts.len(),
        pairs.len()
    );
}

// ------------------------------------------------------------------ the fake engine

struct Counters {
    predicted: AtomicI64,
    prompt: AtomicI64,
}

fn fake_engine() -> (SocketAddr, Arc<Counters>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let c = Arc::new(Counters {
        predicted: AtomicI64::new(0),
        prompt: AtomicI64::new(0),
    });
    let c2 = c.clone();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let c = c2.clone();
            std::thread::spawn(move || {
                let mut s = s.unwrap();
                let mut raw = vec![];
                let mut buf = vec![0u8; 65536];
                let (head, body) = loop {
                    let n = s.read(&mut buf).unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    raw.extend_from_slice(&buf[..n]);
                    if let Some(p) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&raw[..p]).to_string();
                        let want: usize = head
                            .lines()
                            .find_map(|l| {
                                l.to_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        if raw.len() >= p + 4 + want {
                            break (head, raw[p + 4..p + 4 + want].to_vec());
                        }
                    }
                };
                let mut lines = head.lines();
                let first = lines.next().unwrap_or("").to_string();
                let (method, target) = (
                    first.split(' ').next().unwrap_or("").to_string(),
                    first.split(' ').nth(1).unwrap_or("/").to_string(),
                );
                let hv = |name: &str| {
                    head.lines()
                        .find_map(|l| {
                            l.to_lowercase()
                                .strip_prefix(&format!("{name}:"))
                                .map(|_| l.split_once(':').unwrap().1.trim().to_string())
                        })
                        .unwrap_or_default()
                };
                let send = |s: &mut std::net::TcpStream,
                            code: &str,
                            ctype: &str,
                            body: &[u8],
                            extra: &str| {
                    let _ = s.write_all(format!("HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n", body.len()).as_bytes());
                    let _ = s.write_all(body);
                };
                let path = target.split('?').next().unwrap_or("").to_string();
                match path.as_str() {
                    "/health" => send(&mut s, "200 OK", "application/json", b"{\"status\":\"ok\"}", ""),
                    "/props" => send(&mut s, "200 OK", "application/json", json!({"model_path": "/m/x.gguf", "model_alias": "fake", "total_slots": 2, "build_info": "b1", "default_generation_settings": {"n_ctx": 4096}}).to_string().as_bytes(), ""),
                    "/slots" => send(&mut s, "200 OK", "application/json", b"[{\"is_processing\": false}, {\"is_processing\": true}]", ""),
                    "/metrics" => {
                        let t = format!("# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total {}\nllamacpp:prompt_tokens_total {}\nllamacpp:n_decode_total 3\nllamacpp:requests_processing 0\nllamacpp:requests_deferred 0\n", c.predicted.load(Ordering::SeqCst), c.prompt.load(Ordering::SeqCst));
                        send(&mut s, "200 OK", "text/plain; version=0.0.4", t.as_bytes(), "");
                    }
                    "/v1/models" => send(&mut s, "200 OK", "application/json", b"{\"data\":[{\"id\":\"fake\"}]}", ""),
                    "/v1/chat/completions" => {
                        c.predicted.fetch_add(12, Ordering::SeqCst);
                        c.prompt.fetch_add(5, Ordering::SeqCst);
                        let timings = json!({"prompt_n": 5, "prompt_ms": 10.0, "prompt_per_second": 500.0, "predicted_n": 12, "predicted_ms": 300.0, "predicted_per_second": 40.0, "draft_n": 8, "draft_n_accepted": 6, "cache_n": 0});
                        let usage = json!({"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17});
                        if String::from_utf8_lossy(&body).contains("\"stream\": true") || String::from_utf8_lossy(&body).contains("\"stream\":true") {
                            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
                            for part in ["Hel", "lo"] {
                                let _ = s.write_all(format!("data: {}\n\n", json!({"choices": [{"delta": {"content": part}}]})).as_bytes());
                                let _ = s.flush();
                                std::thread::sleep(Duration::from_millis(80));
                            }
                            let _ = s.write_all(format!("data: {}\n\ndata: [DONE]\n\n", json!({"choices": [{"finish_reason": "stop"}], "usage": usage, "timings": timings})).as_bytes());
                        } else {
                            send(&mut s, "200 OK", "application/json", json!({"choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}], "usage": usage, "timings": timings}).to_string().as_bytes(), "");
                        }
                    }
                    "/v1/echo" => send(&mut s, "200 OK", "application/json", json!({"method": method, "target": target, "content_type": hv("content-type"), "accept": hv("accept"), "authorization": hv("authorization"), "body": String::from_utf8_lossy(&body)}).to_string().as_bytes(), ""),
                    "/v1/fail" => send(&mut s, "500 Internal Server Error", "application/json", b"{\"error\":{\"message\":\"boom\"}}", ""),
                    "/v1/missing" => send(&mut s, "404 Not Found", "application/json", b"{\"error\":{\"message\":\"nope\"}}", ""),
                    "/v1/plain" => send(&mut s, "200 OK", "text/plain", b"hello there", ""),
                    "/v1/redirect" => send(&mut s, "302 Found", "text/plain", b"", "Location: /v1/plain\r\n"),
                    _ => send(&mut s, "404 Not Found", "application/json", b"{}", ""),
                }
            });
        }
    });
    (addr, c)
}

#[test]
fn the_proxy_and_the_live_numbers_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let (hp, hr) = (tmp("proxy-py"), tmp("proxy-rs"));
    let py = start_python(&src, &hp);
    let rs: SocketAddr = start_rust(&src, &hr);
    let (tr, tp) = (
        token_of(&http(rs, "GET", "/", &[], None)),
        token_of(&http(py.addr, "GET", "/", &[], None)),
    );
    let (er, ep) = (fake_engine(), fake_engine());
    let n = std::cell::Cell::new(0);
    let req = |addr: SocketAddr,
               tok: &str,
               method: &str,
               path: &str,
               hdrs: &[(&str, &str)],
               body: Option<&str>| {
        let mut h: Vec<(&str, &str)> = hdrs.to_vec();
        h.push(("X-FLE-Token", tok));
        http(addr, method, path, &h, body.map(str::as_bytes))
    };
    let both = |label: &str,
                method: &str,
                path: &str,
                hdrs: &[(&str, &str)],
                body: Option<&str>|
     -> (Resp, Resp) {
        let (a, b) = (
            req(rs, &tr, method, path, hdrs, body),
            req(py.addr, &tp, method, path, hdrs, body),
        );
        compare(label, &a, &b, &tr, &tp);
        n.set(n.get() + 1);
        (a, b)
    };
    // nothing loaded yet
    both("no model", "POST", "/v1/chat/completions", &[], Some("{}"));
    both("no model models", "GET", "/v1/models", &[], None);
    both("live empty", "GET", "/api/live?since=abc", &[], None);
    // attach to the fake engines (with an API key, so the pass-through can be seen to add it)
    for (addr, tok, e) in [(rs, &tr, &er), (py.addr, &tp, &ep)] {
        let r = req(
            addr,
            tok,
            "POST",
            "/api/engine/attach",
            &[],
            Some(
                &json!({"url": format!("http://127.0.0.1:{}", e.0.port()), "key": "k1"})
                    .to_string(),
            ),
        );
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    }
    both("models", "GET", "/v1/models", &[], None);
    both(
        "echo get with query",
        "GET",
        "/v1/echo?a=1&b=x%20y",
        &[],
        None,
    );
    both(
        "echo post json",
        "POST",
        "/v1/echo",
        &[("Content-Type", "application/json")],
        Some("{\"x\": 1}"),
    );
    both(
        "echo post other type",
        "POST",
        "/v1/echo",
        &[("Content-Type", "text/plain"), ("Accept", "text/plain")],
        Some("plain body"),
    );
    both("echo post empty", "POST", "/v1/echo", &[], None);
    both("echo put", "PUT", "/v1/echo", &[], Some("{}"));
    both("echo delete", "DELETE", "/v1/echo", &[], None);
    both("error status passes through", "GET", "/v1/fail", &[], None);
    both("404 passes through", "GET", "/v1/missing", &[], None);
    both("not json", "GET", "/v1/plain", &[], None);
    both("redirect is not followed", "GET", "/v1/redirect", &[], None);
    for (i, body) in [
        "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}",
        "{\"messages\": [], \"stream\": false}",
        "not json at all",
    ]
    .iter()
    .enumerate()
    {
        both(
            &format!("chat {i}"),
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            Some(body),
        );
    }
    // tools together with a forced answer format are refused before they reach the engine
    let clash = |extra: &str| {
        format!("{{\"messages\": [], \"tools\": [{{\"type\": \"function\"}}]{extra}}}")
    };
    for (i, extra) in [
        ", \"response_format\": {\"type\": \"json_object\"}",
        ", \"json_schema\": {\"a\": 1}",
        ", \"grammar\": \"root ::= x\"",
        ", \"response_format\": {\"type\": \"text\"}",
        ", \"response_format\": {}",
        ", \"grammar\": \"\"",
        ", \"tool_choice\": \"none\", \"grammar\": \"x\"",
        ", \"response_format\": {\"type\": \"json_schema\"}, \"grammar\": \"y\"",
    ]
    .iter()
    .enumerate()
    {
        both(
            &format!("clash {i}"),
            "POST",
            "/v1/chat/completions",
            &[],
            Some(&clash(extra)),
        );
    }
    both(
        "tools only",
        "POST",
        "/v1/chat/completions",
        &[],
        Some("{\"messages\": [], \"tools\": [{\"type\": \"function\"}]}"),
    );
    both("tools word elsewhere", "POST", "/v1/chat/completions", &[], Some("{\"messages\": [{\"role\": \"user\", \"content\": \"say \\\"tools\\\"\"}], \"grammar\": \"x\"}"));
    both(
        "clash on another path",
        "POST",
        "/v1/completions",
        &[],
        Some(&clash(", \"grammar\": \"x\"")),
    );
    both(
        "clash trailing slash",
        "POST",
        "/v1/chat/completions/",
        &[],
        Some(&clash(", \"grammar\": \"x\"")),
    );
    // streams: the same chunks, the same headers
    for (label, body) in [
        ("stream", "{\"stream\": true, \"messages\": []}"),
        ("stream compact", "{\"stream\":true,\"messages\":[]}"),
    ] {
        let (a, b) = (
            req(rs, &tr, "POST", "/v1/chat/completions", &[], Some(body)),
            req(
                py.addr,
                &tp,
                "POST",
                "/v1/chat/completions",
                &[],
                Some(body),
            ),
        );
        assert_eq!(a.status, b.status, "{label}");
        for k in [
            "content-type",
            "cache-control",
            "x-accel-buffering",
            "connection",
        ] {
            assert_eq!(a.h(k), b.h(k), "[{label}] header {k}");
        }
        assert_eq!(
            String::from_utf8_lossy(&a.body),
            String::from_utf8_lossy(&b.body),
            "[{label}] body"
        );
        n.set(n.get() + 1);
    }
    // the answers feed the live numbers; wait until the sampler has seen the counters twice
    std::thread::sleep(Duration::from_millis(3500));
    let live = |addr: SocketAddr| -> Value {
        let r = http(addr, "GET", "/api/live", &[], None);
        assert_eq!(r.status, 200);
        serde_json::from_slice(&r.body).unwrap()
    };
    let (a, b) = (live(rs), live(py.addr));
    assert_eq!(a["metrics_available"], json!(true));
    assert_eq!(a["metrics_available"], b["metrics_available"]);
    assert_eq!(a["totals"], b["totals"], "the counters the engine reports");
    let scrub = |v: &Value| {
        let mut v = v.clone();
        v["t"] = json!(0);
        v
    };
    assert_eq!(
        scrub(&a["last_response"]),
        scrub(&b["last_response"]),
        "the last answer"
    );
    assert_eq!(
        (&a["cache"], &a["accept_logged"]),
        (&b["cache"], &b["accept_logged"])
    );
    let last = |v: &Value| v["history"].as_array().unwrap().last().cloned().unwrap();
    let must = [
        "t",
        "state",
        "ram_used",
        "ram_total",
        "gpus",
        "busy",
        "deferred",
    ];
    for k in must {
        assert!(
            last(&a).get(k).is_some() && last(&b).get(k).is_some(),
            "history entry lacks {k}: {} / {}",
            last(&a),
            last(&b)
        );
    }
    assert_eq!(last(&a)["state"], last(&b)["state"]);
    assert_eq!(last(&a)["busy"], last(&b)["busy"]);
    // the first samples after attaching count from the attach: no speed at all
    let first_speeds = |v: &Value| {
        v["history"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|h| h.get("decode").is_some())
            .count()
    };
    assert!(
        first_speeds(&a) <= 1 && first_speeds(&b) <= 1,
        "{} / {}",
        first_speeds(&a),
        first_speeds(&b)
    );
    // `since`
    let t_mid = a["history"][0]["t"].as_f64().unwrap();
    let (x, y) = (
        http(
            rs,
            "GET",
            &format!("/api/live?since={}", t_mid + 1.0),
            &[],
            None,
        ),
        http(
            py.addr,
            "GET",
            &format!("/api/live?since={}", t_mid + 1.0),
            &[],
            None,
        ),
    );
    let cnt = |r: &Resp| {
        serde_json::from_slice::<Value>(&r.body).unwrap()["history"]
            .as_array()
            .unwrap()
            .len()
    };
    assert!(
        cnt(&x) > 0 && cnt(&y) > 0,
        "the history after a time is not empty"
    );
    // the engine goes away: a 502 with the words around the reason
    drop((er, ep));
    let dead_r = req(
        rs,
        &tr,
        "POST",
        "/api/engine/attach",
        &[],
        Some(&json!({"url": "http://127.0.0.1:9"}).to_string()),
    );
    let dead_p = req(
        py.addr,
        &tp,
        "POST",
        "/api/engine/attach",
        &[],
        Some(&json!({"url": "http://127.0.0.1:9"}).to_string()),
    );
    assert_eq!((dead_r.status, dead_p.status), (200, 200));
    let (a, b) = (
        req(rs, &tr, "GET", "/v1/models", &[], None),
        req(py.addr, &tp, "GET", "/v1/models", &[], None),
    );
    assert_eq!((a.status, b.status), (502, 502));
    let msg = |r: &Resp| {
        serde_json::from_slice::<Value>(&r.body).unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .split(':')
            .next()
            .unwrap()
            .to_string()
    };
    assert_eq!(msg(&a), msg(&b));
    assert_eq!(
        serde_json::from_slice::<Value>(&a.body).unwrap()["error"]["type"],
        "bad_gateway"
    );
    eprintln!("compared with the Python Studio: {} proxy requests (and streams) and the live numbers identical", n.get());
}

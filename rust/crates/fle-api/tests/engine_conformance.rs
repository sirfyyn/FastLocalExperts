//! Starting, watching, stopping and attaching an engine, against the Python Studio, with a fake llama-server of our own
//! (a small script that logs like the real one, answers /health and /props, and can crash on request).
#![cfg(unix)]
mod common;
mod harness;
use common::{kv, tensors, write};
use harness::*;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::Path;
use std::time::{Duration, Instant};

const FAKE: &str = r#"#!/usr/bin/env python3
import sys, os, time, json, signal, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv[1:]
if "--version" in a:
    print("version: 1234 (abcdef0)"); sys.exit(0)
if "--help" in a:
    print("--log-verbosity N\n--spec-type T"); sys.exit(0)
port = int(a[a.index("--port") + 1])
mode = os.environ.get("FAKE_MODE", "ok")
model = a[a.index("-m") + 1] if "-m" in a else ""
def say(t):
    print(t, flush=True)
if mode == "exit0":
    say("srv main: nothing to do"); sys.exit(0)
ready_at = time.time() + (2.0 if mode == "slow" else 0.3)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            self._send(200 if time.time() >= ready_at else 503, {"status": "ok" if time.time() >= ready_at else "loading model"})
        elif self.path == "/props":
            self._send(200, {"model_path": model, "model_alias": "fake", "total_slots": 2, "build_info": "b1234-abcdef0", "default_generation_settings": {"n_ctx": 32768}})
        else:
            self._send(404, {})
srv = HTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
def term(*_):
    say("srv main: stopping"); os._exit(0)
signal.signal(signal.SIGTERM, term)
say("llama_model_loader: loaded meta data with 12 key-value pairs")
time.sleep(0.1)
say("llama_params_fit: fitting")
say("load_tensors: loading")
if mode == "crash":
    say("CUDA error: out of memory")
    time.sleep(0.3); sys.exit(3)
if mode == "badarg":
    say("error: unknown argument: --nope"); time.sleep(0.2); sys.exit(1)
say("llama_context: creating")
say("llama_kv_cache: allocating")
say("MoE expert cache enabled: 12 GiB")
say("warming up the model with an empty run")
say("srv main: model loaded")
while time.time() < ready_at:
    time.sleep(0.05)
say("srv main: listening on http://127.0.0.1:%d" % port)
time.sleep(0.4)
say("moe-cache: steps=100 hits=900 misses=100 hit-rate=90.0 %")
say("draft acceptance rate = 0.75")
if mode == "late":
    time.sleep(0.8); say("something broke"); sys.exit(9)
while True:
    time.sleep(0.2)
"#;

fn seed(home: &Path, fake: &Path) {
    let m = home.join("models");
    std::fs::create_dir_all(&m).unwrap();
    write(
        &m.join("tiny.gguf"),
        &kv("qwen3", "Tiny", false),
        &tensors(false),
    );
    write(
        &m.join("part-00001-of-00002.gguf"),
        &kv("qwen3", "Part", false),
        &tensors(false),
    );
    std::fs::write(fake, FAKE).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(fake, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn status(addr: SocketAddr) -> Value {
    serde_json::from_slice(&http(addr, "GET", "/api/engine", &[], None).body).unwrap()
}

fn wait_for(addr: SocketAddr, what: &str, f: &dyn Fn(&Value) -> bool) -> Value {
    let t = Instant::now();
    loop {
        let s = status(addr);
        if f(&s) {
            return s;
        }
        assert!(
            t.elapsed() < Duration::from_secs(40),
            "{what}: still {} / {} on {addr}",
            s["state"],
            s["phase"]
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn log_texts(addr: SocketAddr) -> Vec<String> {
    let v: Value =
        serde_json::from_slice(&http(addr, "GET", "/api/engine/log?since=0", &[], None).body)
            .unwrap();
    v["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["text"].as_str().unwrap().to_string())
        .collect()
}

/// What stands still in a status: the volatile numbers become yes/no, the port and the folder become names.
fn norm(v: &Value, port: i64, home: &Path, fake: &Path) -> Value {
    let mut v = v.clone();
    for k in ["started", "ready_at"] {
        v[k] = json!(v[k].as_f64().unwrap_or(0.0) > 0.0);
    }
    v["pid"] = json!(!v["pid"].is_null());
    if v["plan"].is_object() {
        v["plan"]["port"] = json!("PORT");
    }
    v["rss"] = json!(!v["rss"].is_null());
    v["load"]["read"] = json!(0);
    v["load"]["elapsed"] = json!(!v["load"]["elapsed"].is_null());
    if v["state"] == "starting" {
        // the moment a status is taken decides whether the first log line is in yet
        v["load"]["progress"] = json!(0);
        v["log_seq"] = json!(0);
    }
    if !v["cache"].is_null() {
        v["cache"].as_object_mut().map(|c| c.remove("at"));
    }
    let t = v
        .to_string()
        .replace(&home.to_string_lossy().to_string(), "HOME")
        .replace(&fake.to_string_lossy().to_string(), "FAKE")
        .replace(&format!("127.0.0.1:{port}"), "127.0.0.1:PORT")
        .replace(&format!("--port {port}"), "--port PORT")
        .replace(&format!("\"--port\",\"{port}\""), "\"--port\",\"PORT\"")
        .replace(&format!("\"{port}\""), "\"PORT\"");
    canon(serde_json::from_str::<Value>(&t.replace("-rs", "-X").replace("-py", "-X")).unwrap())
}

fn port_of(s: &Value) -> i64 {
    s["plan"]["port"].as_i64().unwrap_or(0)
}

#[test]
fn the_engine_matches_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let (hp, hr) = (tmp("eng-py"), tmp("eng-rs"));
    let (fp, fr) = (hp.join("fake-llama-server"), hr.join("fake-llama-server"));
    seed(&hp, &fp);
    seed(&hr, &fr);
    let py = start_python(&src, &hp);
    let rs: SocketAddr = start_rust(&src, &hr);
    let (tr, tp) = (
        token_of(&http(rs, "GET", "/", &[], None)),
        token_of(&http(py.addr, "GET", "/", &[], None)),
    );
    let n = std::cell::Cell::new(0);
    let post = |addr: SocketAddr, tok: &str, path: &str, body: &Value| {
        http(
            addr,
            "POST",
            path,
            &[("X-FLE-Token", tok)],
            Some(body.to_string().as_bytes()),
        )
    };
    let call = |path: &str, body: Value| -> (Resp, Resp) {
        (post(rs, &tr, path, &body), post(py.addr, &tp, path, &body))
    };
    let cmp_resp = |label: &str, a: &Resp, b: &Resp| {
        assert_eq!(
            a.status,
            b.status,
            "[{label}] status: {} / {}",
            String::from_utf8_lossy(&a.body),
            String::from_utf8_lossy(&b.body)
        );
        let text = |r: &Resp, h: &Path, f: &Path| {
            String::from_utf8_lossy(&r.body)
                .replace(&h.to_string_lossy().to_string(), "HOME")
                .replace(&f.to_string_lossy().to_string(), "FAKE")
                .replace("-rs", "-X")
                .replace("-py", "-X")
        };
        let (x, y) = (
            serde_json::from_str::<Value>(&text(a, &hr, &fr)).unwrap_or(Value::Null),
            serde_json::from_str::<Value>(&text(b, &hp, &fp)).unwrap_or(Value::Null),
        );
        if x.get("error").is_some() {
            // a port number inside an error text differs; the words around it do not
            let strip = |v: &Value| {
                v["error"]["message"]
                    .as_str()
                    .unwrap_or("")
                    .split(|c: char| c.is_ascii_digit())
                    .collect::<String>()
            };
            assert_eq!(strip(&x), strip(&y), "[{label}] error");
        }
        n.set(n.get() + 1);
    };
    let config = |settings: Value| {
        for (addr, tok) in [(rs, &tr), (py.addr, &tp)] {
            assert_eq!(post(addr, tok, "/api/settings", &settings).status, 200);
        }
    };
    // the fake is a custom engine
    config(
        json!({"engine": {"choice": "custom", "custom_path": "{FAKE}"}})
            .to_string()
            .replace("{FAKE}", "")
            .parse()
            .unwrap(),
    );
    for (addr, tok, fake) in [(rs, &tr, &fr), (py.addr, &tp, &fp)] {
        assert_eq!(
            post(
                addr,
                tok,
                "/api/settings",
                &json!({"engine": {"choice": "custom", "custom_path": fake.to_string_lossy()}})
            )
            .status,
            200
        );
    }
    let model_id = |addr: SocketAddr| -> String {
        let v: Value =
            serde_json::from_slice(&http(addr, "GET", "/api/models", &[], None).body).unwrap();
        v["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["file"] == "tiny.gguf")
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let (mr, mp) = (model_id(rs), model_id(py.addr));
    let part_id = |addr: SocketAddr| -> String {
        let v: Value =
            serde_json::from_slice(&http(addr, "GET", "/api/models", &[], None).body).unwrap();
        v["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["file"] == "part-00001-of-00002.gguf")
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let start = |label: &str, extra: &str| -> (Value, Value) {
        config(json!({"launch": {"extra_env": extra}}));
        let (a, b) = (
            post(rs, &tr, "/api/engine/start", &json!({"model": mr})),
            post(py.addr, &tp, "/api/engine/start", &json!({"model": mp})),
        );
        cmp_resp(&format!("{label} start"), &a, &b);
        assert_eq!(
            a.status,
            200,
            "{label}: {}",
            String::from_utf8_lossy(&a.body)
        );
        (
            serde_json::from_slice(&a.body).unwrap(),
            serde_json::from_slice(&b.body).unwrap(),
        )
    };
    let settle = |label: &str, states: &[&str], marker: &str| -> (Value, Value) {
        let f = |s: &Value| {
            states.contains(&s["state"].as_str().unwrap_or(""))
                && (marker.is_empty() || log_has(marker))
        };
        fn log_has(_m: &str) -> bool {
            true
        }
        let _ = f;
        let g = |addr: SocketAddr| {
            wait_for(addr, label, &|s| {
                states.contains(&s["state"].as_str().unwrap_or(""))
            });
            if !marker.is_empty() {
                let t = Instant::now();
                while !log_texts(addr).iter().any(|l| l.contains(marker)) {
                    assert!(
                        t.elapsed() < Duration::from_secs(20),
                        "{label}: no log line with {marker}"
                    );
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
            status(addr)
        };
        (g(rs), g(py.addr))
    };
    let same = |label: &str, a: &Value, b: &Value| {
        let (pa, pb) = (port_of(a), port_of(b));
        assert_eq!(
            norm(a, pa, &hr, &fr),
            norm(b, pb, &hp, &fp),
            "[{label}] status"
        );
        n.set(n.get() + 1);
    };
    let same_log = |label: &str, a: &Value, b: &Value| {
        let clean =
            |addr_texts: Vec<String>, _port: i64, home: &Path, fake: &Path| -> Vec<String> {
                // the log keeps every run: each has its own port
                let ports = regex::Regex::new(r"(--port |127\.0\.0\.1:)\d+").unwrap();
                addr_texts
                    .into_iter()
                    .map(|t| {
                        ports
                            .replace_all(
                                &t.replace(&home.to_string_lossy().to_string(), "HOME")
                                    .replace(&fake.to_string_lossy().to_string(), "FAKE")
                                    .replace("-rs", "-X")
                                    .replace("-py", "-X"),
                                "${1}PORT",
                            )
                            .to_string()
                    })
                    .collect()
            };
        let (la, lb) = (
            clean(log_texts(rs), port_of(a), &hr, &fr),
            clean(log_texts(py.addr), port_of(b), &hp, &fp),
        );
        assert_eq!(la, lb, "[{label}] log");
        n.set(n.get() + 1);
    };

    // 1. a good start, then a stop
    let (a0, b0) = start("ok", "FAKE_MODE=ok");
    same("ok just started", &a0, &b0);
    let (a, b) = settle("ok ready", &["ready"], "draft acceptance");
    same("ok ready", &a, &b);
    assert_eq!(a["state"], "ready");
    assert_eq!(a["cache"]["hit_rate"], json!(90.0));
    assert_eq!(a["accept_logged"], json!(0.75));
    assert_eq!(a["slots"], json!(2));
    let (x, y) = call("/api/engine/start", json!({"model": mr}));
    let (_, _) = (&x, &y);
    cmp_resp(
        "start while running",
        &post(rs, &tr, "/api/engine/start", &json!({"model": mr})),
        &post(py.addr, &tp, "/api/engine/start", &json!({"model": mp})),
    );
    let (a, b) = (
        post(rs, &tr, "/api/engine/stop", &json!({})),
        post(py.addr, &tp, "/api/engine/stop", &json!({})),
    );
    cmp_resp("stop", &a, &b);
    let (a, b) = settle("ok stopped", &["stopped"], "engine exited");
    same("ok stopped", &a, &b);
    same_log("ok", &a, &b);
    cmp_resp(
        "stop again",
        &post(rs, &tr, "/api/engine/stop", &json!({})),
        &post(py.addr, &tp, "/api/engine/stop", &json!({})),
    );
    // 2. crashes and what is said about them
    for (label, mode, marker) in [
        ("crash", "crash", "engine exited"),
        ("exit0", "exit0", "engine exited"),
        ("late crash", "late", "engine exited"),
        ("bad argument", "badarg", "engine exited"),
    ] {
        start(label, &format!("FAKE_MODE={mode}"));
        let (a, b) = settle(label, &["crashed"], marker);
        same(label, &a, &b);
        same_log(label, &a, &b);
        assert_eq!(a["state"], "crashed");
    }
    // 3. a slow load shows its phases and still ends ready
    start("slow", "FAKE_MODE=slow");
    let (a, b) = settle("slow ready", &["ready"], "draft acceptance");
    same("slow ready", &a, &b);
    same_log("slow", &a, &b);
    cmp_resp(
        "stop slow",
        &post(rs, &tr, "/api/engine/stop", &json!({})),
        &post(py.addr, &tp, "/api/engine/stop", &json!({})),
    );
    settle("slow stopped", &["stopped"], "engine exited");
    // 4. refused starts
    config(json!({"launch": {"extra_env": "FAKE_MODE=ok"}}));
    cmp_resp(
        "no model",
        &post(rs, &tr, "/api/engine/start", &json!({"model": ""})),
        &post(py.addr, &tp, "/api/engine/start", &json!({"model": ""})),
    );
    cmp_resp(
        "unknown model",
        &post(
            rs,
            &tr,
            "/api/engine/start",
            &json!({"model": "ffffffffffff"}),
        ),
        &post(
            py.addr,
            &tp,
            "/api/engine/start",
            &json!({"model": "ffffffffffff"}),
        ),
    );
    cmp_resp(
        "incomplete model",
        &post(rs, &tr, "/api/engine/start", &json!({"model": part_id(rs)})),
        &post(
            py.addr,
            &tp,
            "/api/engine/start",
            &json!({"model": part_id(py.addr)}),
        ),
    );
    cmp_resp(
        "unknown profile",
        &post(
            rs,
            &tr,
            "/api/engine/start",
            &json!({"model": mr, "profile": "nope"}),
        ),
        &post(
            py.addr,
            &tp,
            "/api/engine/start",
            &json!({"model": mp, "profile": "nope"}),
        ),
    );
    let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    config(
        json!({"launch": {"extra_args": format!("--port {}", busy.local_addr().unwrap().port())}}),
    );
    cmp_resp(
        "port taken",
        &post(rs, &tr, "/api/engine/start", &json!({"model": mr})),
        &post(py.addr, &tp, "/api/engine/start", &json!({"model": mp})),
    );
    config(json!({"launch": {"extra_args": ""}}));
    // 5. previews
    for (label, body_r, body_p) in [
        ("preview auto", json!({"model": mr}), json!({"model": mp})),
        (
            "preview reference",
            json!({"model": mr, "profile": "fle-reference"}),
            json!({"model": mp, "profile": "fle-reference"}),
        ),
        ("preview nothing", json!({}), json!({})),
    ] {
        let (a, b) = (
            post(rs, &tr, "/api/engine/preview", &body_r),
            post(py.addr, &tp, "/api/engine/preview", &body_p),
        );
        assert_eq!(a.status, b.status, "[{label}]");
        if a.status == 200 {
            let (x, y): (Value, Value) = (
                serde_json::from_slice(&a.body).unwrap(),
                serde_json::from_slice(&b.body).unwrap(),
            );
            let (pa, pb) = (x["port"].as_i64().unwrap(), y["port"].as_i64().unwrap());
            let t = |v: &Value, p: i64, h: &Path, f: &Path| {
                let mut v = v.clone();
                v["port"] = json!("PORT");
                {
                    serde_json::from_str::<Value>(
                        &v.to_string()
                            .replace(&h.to_string_lossy().to_string(), "HOME")
                            .replace(&f.to_string_lossy().to_string(), "FAKE")
                            .replace(&p.to_string(), "PORT")
                            .replace("-rs", "-X")
                            .replace("-py", "-X"),
                    )
                    .unwrap()
                }
            };
            let (x, y) = (canon(t(&x, pa, &hr, &fr)), canon(t(&y, pb, &hp, &fp)));
            assert_eq!(x, y, "[{label}] plan");
        } else {
            cmp_resp(label, &a, &b);
        }
        n.set(n.get() + 1);
    }
    // 6. attaching to a server that runs already
    #[allow(clippy::zombie_processes)] // the test kills and waits for both fakes below
    let spawn_fake = |fake: &Path| -> (std::process::Child, u16) {
        let port = free_port();
        let c = std::process::Command::new(fake)
            .args(["--port", &port.to_string()])
            .env("FAKE_MODE", "ok")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        for _ in 0..100 {
            if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
                std::thread::sleep(Duration::from_millis(500));
                return (c, port);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("the fake did not start");
    };
    // the start without a model name used the last model, so an engine runs now: attaching is refused while it does
    cmp_resp(
        "attach while running",
        &post(
            rs,
            &tr,
            "/api/engine/attach",
            &json!({"url": "http://127.0.0.1:9"}),
        ),
        &post(
            py.addr,
            &tp,
            "/api/engine/attach",
            &json!({"url": "http://127.0.0.1:9"}),
        ),
    );
    cmp_resp(
        "stop it",
        &post(rs, &tr, "/api/engine/stop", &json!({})),
        &post(py.addr, &tp, "/api/engine/stop", &json!({})),
    );
    settle("stopped before attach", &["stopped"], "engine exited");
    let (mut c1, p1) = spawn_fake(&fr);
    let (mut c2, p2) = spawn_fake(&fp);
    let (a, b) = (
        post(
            rs,
            &tr,
            "/api/engine/attach",
            &json!({"url": format!("127.0.0.1:{p1}/v1/"), "key": "secret"}),
        ),
        post(
            py.addr,
            &tp,
            "/api/engine/attach",
            &json!({"url": format!("127.0.0.1:{p2}/v1/"), "key": "secret"}),
        ),
    );
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    assert_eq!(b.status, 200);
    let (sa, sb): (Value, Value) = (
        serde_json::from_slice(&a.body).unwrap(),
        serde_json::from_slice(&b.body).unwrap(),
    );
    for (x, p) in [(&sa, p1), (&sb, p2)] {
        assert_eq!(
            (x["state"].as_str(), x["reachable"].as_bool()),
            (Some("attached"), Some(true)),
            "{x}"
        );
        let _ = p;
    }
    let strip_ports = |v: &Value, p: u16, h: &Path, f: &Path| {
        serde_json::from_str::<Value>(
            &norm(v, 0, h, f)
                .to_string()
                .replace(&format!("127.0.0.1:{p}"), "127.0.0.1:PORT"),
        )
        .unwrap()
    };
    assert_eq!(
        strip_ports(&sa, p1, &hr, &fr),
        strip_ports(&sb, p2, &hp, &fp),
        "attached status"
    );
    let (a, b) = (
        http(rs, "GET", "/api/settings", &[], None),
        http(py.addr, "GET", "/api/settings", &[], None),
    );
    let flags = |r: &Resp| {
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        json!([
            v["attach_auto"],
            v["has_attach_key"],
            v["attach_url"]
                .as_str()
                .unwrap()
                .replace(&p1.to_string(), "P")
                .replace(&p2.to_string(), "P")
        ])
    };
    assert_eq!(flags(&a), flags(&b), "the settings after attach");
    // Deliberate difference D2: the Python Studio starts a second engine next to an attached server, this one refuses.
    let refused = post(rs, &tr, "/api/engine/start", &json!({"model": mr}));
    assert_eq!(refused.status, 409);
    assert!(
        String::from_utf8_lossy(&refused.body).contains("A server is attached"),
        "{}",
        String::from_utf8_lossy(&refused.body)
    );
    let (a, b) = (
        post(rs, &tr, "/api/engine/stop", &json!({})),
        post(py.addr, &tp, "/api/engine/stop", &json!({})),
    );
    cmp_resp("detach", &a, &b);
    settle("stopped after detach", &["stopped"], "engine exited");
    same("detached", &status(rs), &status(py.addr));
    let (a, b) = (
        http(rs, "GET", "/api/settings", &[], None),
        http(py.addr, "GET", "/api/settings", &[], None),
    );
    assert_eq!(flags(&a), flags(&b), "the settings after detach");
    let _ = (c1.kill(), c2.kill(), c1.wait(), c2.wait());
    // attach to nothing, with a bad address, and to a unit (Linux has systemd; the unit comes later)
    let dead = free_port();
    let (a, b) = (
        post(
            rs,
            &tr,
            "/api/engine/attach",
            &json!({"url": format!("http://127.0.0.1:{dead}")}),
        ),
        post(
            py.addr,
            &tp,
            "/api/engine/attach",
            &json!({"url": format!("http://127.0.0.1:{dead}")}),
        ),
    );
    assert_eq!(a.status, b.status);
    let (sa, sb): (Value, Value) = (
        serde_json::from_slice(&a.body).unwrap(),
        serde_json::from_slice(&b.body).unwrap(),
    );
    assert_eq!(
        (
            sa["state"].clone(),
            sa["reachable"].clone(),
            sa["error"].clone()
        ),
        (
            sb["state"].clone(),
            sb["reachable"].clone(),
            sb["error"].clone()
        ),
        "attach to a dead port"
    );
    cmp_resp(
        "detach dead",
        &post(rs, &tr, "/api/engine/stop", &json!({})),
        &post(py.addr, &tp, "/api/engine/stop", &json!({})),
    );
    cmp_resp(
        "attach without address",
        &post(rs, &tr, "/api/engine/attach", &json!({"url": ""})),
        &post(py.addr, &tp, "/api/engine/attach", &json!({"url": ""})),
    );
    cmp_resp(
        "service action",
        &post(rs, &tr, "/api/engine/service/restart", &json!({})),
        &post(py.addr, &tp, "/api/engine/service/restart", &json!({})),
    );
    #[cfg(target_os = "macos")]
    cmp_resp(
        "attach a unit on a Mac",
        &post(
            rs,
            &tr,
            "/api/engine/attach",
            &json!({"url": "", "service": "fle"}),
        ),
        &post(
            py.addr,
            &tp,
            "/api/engine/attach",
            &json!({"url": "", "service": "fle"}),
        ),
    );
    // 7. the log after a given line
    let (a, b) = (
        http(rs, "GET", "/api/engine/log?since=3", &[], None),
        http(py.addr, "GET", "/api/engine/log?since=3", &[], None),
    );
    let (x, y): (Value, Value) = (
        serde_json::from_slice(&a.body).unwrap(),
        serde_json::from_slice(&b.body).unwrap(),
    );
    assert_eq!(x["lines"][0]["seq"], y["lines"][0]["seq"], "log since");
    assert_eq!(x["seq"], y["seq"], "log sequence");
    let _ = call;
    eprintln!("compared with the Python Studio: {} checks (start, ready, stop, crashes, slow load, refusals, previews, attach, log) identical", n.get());
}

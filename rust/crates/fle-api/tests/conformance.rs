//! Request-by-request comparison with the Python Studio: the same requests go to both servers, and status, the headers
//! the page depends on, and the JSON (or page) that comes back must agree. Skipped when the Studio sources are absent.
//! Point `FLE_STUDIO_SRC` at the `src` folder of a fle-studio checkout.
#![cfg(unix)]
use fle_api::server::{new_studio, serve, Options};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

fn studio_src() -> Option<PathBuf> {
    let c = std::env::var("FLE_STUDIO_SRC")
        .map(PathBuf::from)
        .ok()
        .into_iter()
        .chain([
            dirs_home().join("fle-studio-020/src"),
            dirs_home().join("fle-studio/src"),
        ]);
    c.into_iter()
        .find(|p| p.join("fle_studio/server.py").exists())
}
fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}
fn tmp(n: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fle-api-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}
impl Resp {
    fn h(&self, k: &str) -> Vec<&str> {
        self.headers
            .iter()
            .filter(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
            .collect()
    }
}

/// A raw HTTP/1.1 request, so the test controls every header (Host, Origin, token) exactly.
fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(std::time::Duration::from_secs(20)))
        .unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\n");
    if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host")) {
        req += &format!("Host: 127.0.0.1:{}\r\n", addr.port());
    }
    for (k, v) in headers {
        req += &format!("{k}: {v}\r\n");
    }
    if let Some(b) = body {
        req += &format!("Content-Length: {}\r\n", b.len());
    }
    req += "Connection: close\r\n\r\n";
    s.write_all(req.as_bytes()).unwrap();
    if let Some(b) = body {
        // a server may answer (413) and hang up before the whole body is sent; the answer is still there to read
        let _ = s.write_all(b);
    }
    let mut raw = vec![];
    let _ = s.read_to_end(&mut raw);
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("no header end");
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let mut lines = head.lines();
    let status: u16 = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect();
    let mut body = raw[split + 4..].to_vec();
    if let Some(i) = head.to_lowercase().find("transfer-encoding: chunked") {
        let _ = i;
        let mut out = vec![];
        let mut rest = &body[..];
        loop {
            let nl = rest.windows(2).position(|w| w == b"\r\n").unwrap();
            let n = usize::from_str_radix(std::str::from_utf8(&rest[..nl]).unwrap().trim(), 16)
                .unwrap();
            if n == 0 {
                break;
            }
            out.extend(&rest[nl + 2..nl + 2 + n]);
            rest = &rest[nl + 2 + n + 2..];
        }
        body = out;
    }
    Resp {
        status,
        headers,
        body,
    }
}

struct Py {
    child: Child,
    addr: SocketAddr,
}
impl Drop for Py {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
#[allow(clippy::zombie_processes)] // `Py` kills and waits for the child when it is dropped
fn start_python(src: &Path, home: &Path) -> Py {
    let port = free_port();
    let child =
        Command::new(std::env::var("FLE_STUDIO_PYTHON").unwrap_or_else(|_| "python3".into()))
            .args(["-m", "fle_studio", "--home"])
            .arg(home)
            .args(["--port", &port.to_string(), "--no-browser"])
            .env("PYTHONPATH", src)
            .env("FLE_STUDIO_HOME", home)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    for _ in 0..100 {
        if TcpStream::connect(addr).is_ok() {
            std::thread::sleep(std::time::Duration::from_millis(200));
            return Py { child, addr };
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!("the Python Studio did not start");
}
fn start_rust(src: &Path, home: &Path) -> SocketAddr {
    let st = new_studio(Options {
        home: home.to_path_buf(),
        web_dir: Some(src.join("fle_studio/web")),
        version: python_version(src),
    });
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(serve(st, "127.0.0.1:0".parse().unwrap(), Some(tx)))
            .unwrap();
    });
    rx.recv().unwrap()
}
fn python_version(src: &Path) -> String {
    let t = std::fs::read_to_string(src.join("fle_studio/__init__.py")).unwrap();
    t.split("__version__ = \"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .to_string()
}
fn token_of(r: &Resp) -> String {
    let t = String::from_utf8_lossy(&r.body).to_string();
    t.split("name=\"fle-token\" content=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .to_string()
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Ids and times that the servers make up (random ids, "now") become fixed names, so the rest can be compared.
fn canon(v: Value) -> Value {
    fn collect(v: &Value, map: &mut Vec<String>) {
        match v {
            Value::Object(o) => {
                if let Some(Value::String(id)) = o.get("id") {
                    if (is_hex(id, 12) || is_hex(id, 16)) && !map.contains(id) {
                        map.push(id.clone());
                    }
                }
                o.values().for_each(|x| collect(x, map));
            }
            Value::Array(a) => a.iter().for_each(|x| collect(x, map)),
            _ => {}
        }
    }
    fn apply(v: Value, map: &[String], now: f64) -> Value {
        match v {
            Value::String(s) => match map.iter().position(|m| *m == s) {
                Some(i) => Value::String(format!("<id{i}>")),
                None => Value::String(s),
            },
            Value::Array(a) => Value::Array(a.into_iter().map(|x| apply(x, map, now)).collect()),
            Value::Object(o) => Value::Object(
                o.into_iter()
                    .map(|(k, x)| {
                        let near_now = (k == "created" || k == "updated")
                            && x.as_f64().is_some_and(|f| (f - now).abs() < 120.0);
                        (
                            k,
                            if near_now {
                                json!("<now>")
                            } else {
                                apply(x, map, now)
                            },
                        )
                    })
                    .collect(),
            ),
            other => other,
        }
    }
    let mut map = vec![];
    collect(&v, &mut map);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    apply(v, &map, now)
}

const KEEP: [&str; 7] = [
    "content-type",
    "cache-control",
    "x-content-type-options",
    "referrer-policy",
    "content-security-policy",
    "x-frame-options",
    "x-fle-session",
];

fn compare(label: &str, a: &Resp, b: &Resp, ta: &str, tb: &str) {
    assert_eq!(
        a.status, b.status,
        "[{label}] status: rust {} python {}",
        a.status, b.status
    );
    for k in KEEP {
        assert_eq!(a.h(k), b.h(k), "[{label}] header {k}");
    }
    let ct = a.h("content-type").first().copied().unwrap_or("");
    if ct.starts_with("application/json") && !a.body.is_empty() {
        // the two servers run in folders that differ only in their suffix
        let norm = |b: &[u8]| {
            serde_json::from_str::<Value>(
                &String::from_utf8_lossy(b)
                    .replace("-rs", "-X")
                    .replace("-py", "-X"),
            )
            .unwrap()
        };
        let (x, y): (Value, Value) = (canon(norm(&a.body)), canon(norm(&b.body)));
        assert_eq!(x, y, "[{label}] JSON body differs");
    } else {
        let (x, y) = (
            String::from_utf8_lossy(&a.body).replace(ta, "TOKEN"),
            String::from_utf8_lossy(&b.body).replace(tb, "TOKEN"),
        );
        assert_eq!(x, y, "[{label}] body differs");
    }
}

fn web_files(dir: &Path, base: &Path, out: &mut Vec<String>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            web_files(&p, base, out);
        } else {
            out.push(format!(
                "/{}",
                p.strip_prefix(base).unwrap().to_string_lossy()
            ));
        }
    }
}

#[test]
fn stage_zero_plumbing_matches_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let mut checked = 0;
    for variant in ["fresh", "junk", "broken"] {
        let (hp, hr) = (tmp(&format!("{variant}-py")), tmp(&format!("{variant}-rs")));
        for h in [&hp, &hr] {
            match variant {
                "junk" => std::fs::write(h.join("settings.json"), r#"{"launch": {"context": "4096", "parallel": "two", "mtp": "no"}, "theme": 5, "nope": 1, "hf_token": "secret", "server": {"port": "9001"}, "chat": {"temperature": "0.2"}, "tools": {"mcp_trusted": ["a"]}}"#).unwrap(),
                "broken" => std::fs::write(h.join("settings.json"), "{ this is not json").unwrap(),
                _ => {}
            }
        }
        for h in [&hp, &hr] {
            let c = h.join("chats");
            std::fs::create_dir_all(&c).unwrap();
            let seeds: [(&str, &str); 7] = [
                (
                    "1111111111111111.json",
                    r#"{"id": "1111111111111111", "title": "Old list", "updated": 1000, "model": "m1", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}"#,
                ),
                (
                    "2222222222222222.json",
                    r#"{"id": "2222222222222222", "title": "Branchy", "updated": 3000, "model": "m2", "leaf": "b2", "created": 5, "messages": [{"id": "a", "parent": null, "role": "user", "content": "q"}, {"id": "b1", "parent": "a", "role": "assistant", "content": "x"}, {"id": "b2", "parent": "a", "role": "assistant", "content": "y"}, {"id": "c", "parent": "b1", "role": "user", "content": "z"}]}"#,
                ),
                (
                    "3333333333333333.json",
                    r#"{"id": "3333333333333333", "updated": -5, "messages": "nope"}"#,
                ),
                ("4444444444444444.json", "[]"),
                ("5555555555555555.json", "{broken"),
                (
                    "6666666666666666.json",
                    r#"{"title": null, "model": null, "updated": 2000, "messages": [{"id": "p", "parent": "q", "role": "user", "content": "1"}, {"id": "q", "parent": "p", "role": "assistant", "content": [{"text": "two"}, "three", {"nope": 1}, 7]}, {"id": "p", "parent": "zz", "role": "user", "content": "dup"}]}"#,
                ),
                ("not-a-chat.json", r#"{"id": "x"}"#),
            ];
            for (n, t) in seeds {
                std::fs::write(c.join(n), t).unwrap();
            }
        }
        let py = start_python(&src, &hp);
        let rs = start_rust(&src, &hr);
        let (ia, ib) = (
            http(rs, "GET", "/", &[], None),
            http(py.addr, "GET", "/", &[], None),
        );
        let (tr, tp) = (token_of(&ia), token_of(&ib));
        compare(&format!("{variant} index"), &ia, &ib, &tr, &tp);
        let ids: std::cell::RefCell<(Vec<String>, Vec<String>)> = Default::default();
        let mut both = |label: &str,
                        method: &str,
                        path: &str,
                        hdrs: &[(&str, &str)],
                        body: Option<&[u8]>,
                        tokened: bool| {
            let run = |addr: SocketAddr, tok: &str, mine: bool| {
                let mut h: Vec<(&str, &str)> = hdrs.to_vec();
                if tokened {
                    h.push(("X-FLE-Token", tok));
                }
                let mut p = path.to_string();
                for i in 0..10 {
                    let list = ids.borrow();
                    let known = if mine { &list.0 } else { &list.1 };
                    if let Some(id) = known.get(i) {
                        p = p.replace(&format!("{{{i}}}"), id);
                    }
                }
                http(addr, method, &p, &h, body)
            };
            let (a, b) = (run(rs, &tr, true), run(py.addr, &tp, false));
            if method == "POST" && path == "/api/chats" && a.status == 200 {
                let id = |r: &Resp| {
                    serde_json::from_slice::<Value>(&r.body).unwrap()["id"]
                        .as_str()
                        .unwrap()
                        .to_string()
                };
                let mut l = ids.borrow_mut();
                l.0.push(id(&a));
                l.1.push(id(&b));
            }
            compare(&format!("{variant} {label}"), &a, &b, &tr, &tp);
            checked += 1;
        };
        both("ping", "GET", "/api/ping", &[], None, false);
        both("head ping", "HEAD", "/api/ping", &[], None, false);
        both("settings", "GET", "/api/settings", &[], None, false);
        for (i, body) in [
            json!({"launch": {"context": "8192", "parallel": 3, "mtp": "off", "extra_args": "--x 1"}}),
            json!({"chat": {"temperature": "0.25", "top_k": 40.9, "thinking": 0, "system_prompt": null}, "theme": "dark"}),
            json!({"hf_token": "abc", "access_key": "k", "attach_key": ""}),
            json!({"server": {"port": "9100", "open_browser": "false"}, "unknown": {"x": 1}, "tools": {"mcp_trusted": ["a", "b"], "max_rounds": "12"}}),
            json!({"launch": {"context": "abc"}}),
            json!({"chat": {"temperature": "hot"}}),
            json!({"launch": {"parallel": null}}),
            json!({"launch": {"parallel": [1]}}),
            json!({"models_dir": "~/m", "last_model": 7, "engine": {"choice": true}}),
            json!({"hf_token": null}),
            json!({}),
        ]
        .iter()
        .enumerate()
        {
            both(&format!("settings update {i}"), "POST", "/api/settings", &[("Content-Type", "application/json")], Some(body.to_string().as_bytes()), true);
            both(&format!("settings after {i}"), "GET", "/api/settings", &[], None, false);
        }
        for (i, raw) in [&b"[]"[..], b"not json", b"", b"\"x\"", b"123"]
            .iter()
            .enumerate()
        {
            both(
                &format!("settings raw body {i}"),
                "POST",
                "/api/settings",
                &[],
                Some(raw),
                true,
            );
        }
        both("chats list", "GET", "/api/chats", &[], None, false);
        for id in [
            "1111111111111111",
            "2222222222222222",
            "3333333333333333",
            "4444444444444444",
            "5555555555555555",
            "6666666666666666",
            "7777777777777777",
        ] {
            both(
                &format!("seeded chat {id}"),
                "GET",
                &format!("/api/chats/{id}"),
                &[],
                None,
                false,
            );
        }
        for bad in [
            "/api/chats/short",
            "/api/chats/ZZZZZZZZZZZZZZZZ",
            "/api/chats/0123456789abcdef0",
            "/api/chats/not-a-chat",
        ] {
            both(&format!("bad chat id {bad}"), "GET", bad, &[], None, false);
            both(
                &format!("bad chat id put {bad}"),
                "PUT",
                bad,
                &[],
                Some(b"{}"),
                true,
            );
            both(
                &format!("bad chat id delete {bad}"),
                "DELETE",
                bad,
                &[],
                None,
                true,
            );
        }
        let long = "x".repeat(200);
        let creates: Vec<Value> = vec![
            json!({"messages": [{"role": "user", "content": "Hello    world  \n again"}, {"role": "assistant", "content": "Hi"}]}),
            json!({"title": "  Mine  ", "model": "m", "system": "be brief", "created": 42, "leaf": "n2", "messages": [{"id": "n1", "role": "user", "content": "a", "parent": null}, {"id": "n2", "role": "assistant", "content": "b", "parent": "n1"}, {"id": "n3", "role": "assistant", "content": "c", "parent": "n1"}]}),
            json!({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "part one"}, "part two", {"x": 1}]}]}),
            json!({"messages": [{"role": "user", "content": "ünï çödé — ".repeat(12)}]}),
            json!({"title": 5, "messages": []}),
            json!({"title": long, "messages": [{"role": "user", "content": "t"}]}),
            json!({"title": null, "model": 7, "system": ["x"], "messages": [{"role": "assistant", "content": "only assistant"}]}),
            json!({"messages": [{"id": "a", "parent": "b", "role": "user", "content": "1"}, {"id": "b", "parent": "a", "role": "assistant", "content": "2"}, {"id": "c", "parent": "c", "role": "user", "content": "3"}]}),
            json!({"messages": [{"id": "dup", "role": "user", "content": "1", "parent": null}, {"id": "dup", "role": "assistant", "content": "2", "parent": "dup"}, {"id": "bad id!", "role": "user", "content": "3", "parent": "nope"}], "leaf": "missing"}),
            json!({"messages": [1, "x", null, {"role": "user", "content": "kept"}, [1]]}),
            json!({"messages": []}),
            json!({"messages": "text"}),
            json!({"model": "no messages"}),
            json!({"messages": [{"role": "user", "content": "x", "tool_calls": [{"name": "calc"}], "attachments": [{"name": "f.txt", "kind": "text", "chars": 3}]}], "extra": "dropped"}),
        ];
        for (i, body) in creates.iter().enumerate() {
            both(
                &format!("chat create {i}"),
                "POST",
                "/api/chats",
                &[],
                Some(body.to_string().as_bytes()),
                true,
            );
        }
        for i in 0..8 {
            both(
                &format!("chat get created {i}"),
                "GET",
                &format!("/api/chats/{{{i}}}"),
                &[],
                None,
                false,
            );
        }
        both(
            "chat put update",
            "PUT",
            "/api/chats/{0}",
            &[],
            Some(
                json!({"title": "Renamed", "messages": [{"role": "user", "content": "new"}]})
                    .to_string()
                    .as_bytes(),
            ),
            true,
        );
        both("chat put keeps created", "PUT", "/api/chats/{1}", &[], Some(json!({"created": 7, "messages": [{"id": "z", "role": "user", "content": "new", "parent": null}]}).to_string().as_bytes()), true);
        both(
            "chat put invalid",
            "PUT",
            "/api/chats/{2}",
            &[],
            Some(b"{\"messages\": 1}"),
            true,
        );
        both(
            "chat put not found id",
            "PUT",
            "/api/chats/aaaaaaaaaaaaaaaa",
            &[],
            Some(json!({"messages": []}).to_string().as_bytes()),
            true,
        );
        both(
            "chat get not found",
            "GET",
            "/api/chats/bbbbbbbbbbbbbbbb",
            &[],
            None,
            false,
        );
        both(
            "chat create bad json",
            "POST",
            "/api/chats",
            &[],
            Some(b"{nope"),
            true,
        );
        both(
            "chat create array",
            "POST",
            "/api/chats",
            &[],
            Some(b"[]"),
            true,
        );
        both(
            "chat create no token",
            "POST",
            "/api/chats",
            &[],
            Some(b"{\"messages\": []}"),
            false,
        );
        both("chats list after", "GET", "/api/chats", &[], None, false);
        both("chat delete", "DELETE", "/api/chats/{0}", &[], None, true);
        both(
            "chat delete again",
            "DELETE",
            "/api/chats/{0}",
            &[],
            None,
            true,
        );
        both(
            "chat get deleted",
            "GET",
            "/api/chats/{0}",
            &[],
            None,
            false,
        );
        both(
            "chat delete seeded",
            "DELETE",
            "/api/chats/1111111111111111",
            &[],
            None,
            true,
        );
        both("chats list end", "GET", "/api/chats", &[], None, false);
        let big = format!(
            "{{\"messages\": [{{\"role\": \"user\", \"content\": \"{}\"}}]}}",
            "a".repeat(5 * 1024 * 1024)
        );
        both(
            "chat create 5 MB is fine",
            "POST",
            "/api/chats",
            &[],
            Some(big.as_bytes()),
            true,
        );
        let settings_big = format!("{{\"theme\": \"{}\"}}", "a".repeat(5 * 1024 * 1024));
        both(
            "settings 5 MB is too large",
            "POST",
            "/api/settings",
            &[],
            Some(settings_big.as_bytes()),
            true,
        );
        both("no token", "POST", "/api/settings", &[], Some(b"{}"), false);
        both(
            "wrong token",
            "POST",
            "/api/settings",
            &[("X-FLE-Token", "nope")],
            Some(b"{}"),
            false,
        );
        both(
            "other origin",
            "POST",
            "/api/settings",
            &[("Origin", "http://evil.example")],
            Some(b"{}"),
            true,
        );
        both(
            "null origin",
            "POST",
            "/api/settings",
            &[("Origin", "null")],
            Some(b"{}"),
            true,
        );
        both(
            "delete no token",
            "DELETE",
            "/api/chats/0123456789abcdef",
            &[],
            None,
            false,
        );
        both(
            "foreign host",
            "GET",
            "/api/ping",
            &[("Host", "evil.example")],
            None,
            false,
        );
        both(
            "localhost host",
            "GET",
            "/api/ping",
            &[("Host", "localhost:1234")],
            None,
            false,
        );
        both(
            "ipv6 host",
            "GET",
            "/api/ping",
            &[("Host", "[::1]:1234")],
            None,
            false,
        );
        for p in [
            "/nope",
            "/api/nope",
            "/api/",
            "/some/client/route",
            "/x.png",
            "/js/missing.js",
            "/..%2f..%2fetc/passwd",
            "/.hidden",
        ] {
            both(&format!("get {p}"), "GET", p, &[], None, false);
        }
        both("post unknown", "POST", "/api/nope", &[], Some(b"{}"), true);
        both("post ping", "POST", "/api/ping", &[], Some(b"{}"), true);
        let mut files = vec![];
        let web = src.join("fle_studio/web");
        web_files(&web, &web, &mut files);
        files.sort();
        for f in &files {
            both(&format!("static {f}"), "GET", f, &[], None, false);
        }
        both("head static", "HEAD", "/app.css", &[], None, false);
        let (fa, fb) = (
            std::fs::read_to_string(hr.join("settings.json")),
            std::fs::read_to_string(hp.join("settings.json")),
        );
        let (fa, fb) = (
            serde_json::from_str::<Value>(&fa.unwrap()).unwrap(),
            serde_json::from_str::<Value>(&fb.unwrap()).unwrap(),
        );
        assert_eq!(fa, fb, "[{variant}] settings.json on disk differs");
        assert_eq!(
            fa.as_object().unwrap().keys().collect::<Vec<_>>(),
            fb.as_object().unwrap().keys().collect::<Vec<_>>(),
            "[{variant}] key order on disk"
        );
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |h: &Path| {
                std::fs::metadata(h.join("settings.json"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777
            };
            assert_eq!(mode(&hr), mode(&hp), "[{variant}] file mode");
        }
        assert_eq!(
            hr.join("settings.broken.json").exists(),
            hp.join("settings.broken.json").exists(),
            "[{variant}] broken file kept"
        );
    }
    eprintln!(
        "compared with the Python Studio: {checked} requests identical in 3 starting situations"
    );
}

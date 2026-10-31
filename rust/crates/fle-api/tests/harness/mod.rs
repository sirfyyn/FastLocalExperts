//! Shared by the Studio conformance tests: start the Python Studio and the Rust server side by side, speak raw HTTP to both,
//! and compare what comes back (ids and times that a server makes up are replaced by fixed names first).
#![allow(dead_code)]
use fle_api::server::{new_studio, serve, Options};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

pub fn studio_src() -> Option<PathBuf> {
    let c = std::env::var("FLE_STUDIO_SRC")
        .map(PathBuf::from)
        .ok()
        .into_iter()
        .chain([
            dirs_home().join("fle-studio-main/src"),
            dirs_home().join("fle-studio/src"),
        ]);
    c.into_iter()
        .find(|p| p.join("fle_studio/server.py").exists())
}
pub fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}
/// A scratch folder that is removed when the test ends, also when it fails.
pub struct Tmp(PathBuf);
impl std::ops::Deref for Tmp {
    type Target = PathBuf;
    fn deref(&self) -> &PathBuf {
        &self.0
    }
}
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
pub fn tmp(n: &str) -> Tmp {
    // without symlinks in the path (macOS keeps /var behind /private/var): the Python Studio compares resolved with
    // unresolved folders in places, and the two only agree when there is nothing to resolve
    let base = std::fs::canonicalize(std::env::temp_dir()).unwrap();
    let d = base.join(format!("fle-api-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

pub struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}
impl Resp {
    pub fn h(&self, k: &str) -> Vec<&str> {
        self.headers
            .iter()
            .filter(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
            .collect()
    }
}

/// A raw HTTP/1.1 request, so the test controls every header (Host, Origin, token) exactly.
pub fn http(
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

pub struct Py {
    pub child: Child,
    pub addr: SocketAddr,
}
impl Drop for Py {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
pub fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
pub fn start_python(src: &Path, home: &Path) -> Py {
    start_python_with(src, home, "")
}

/// Like `start_python`, with a line of Python run first (it can change a module constant, for example).
#[allow(clippy::zombie_processes)] // `Py` kills and waits for the child when it is dropped
pub fn start_python_with(src: &Path, home: &Path, prelude: &str) -> Py {
    let port = free_port();
    let code = format!(
        "import sys\n{prelude}\nfrom fle_studio.__main__ import main\nsys.exit(main(sys.argv[1:]))"
    );
    let child =
        Command::new(std::env::var("FLE_STUDIO_PYTHON").unwrap_or_else(|_| "python3".into()))
            .args(["-c", &code, "--home"])
            .arg(home)
            .args(["--port", &port.to_string(), "--no-browser"])
            .env("PYTHONPATH", src)
            // the core only: the plugins of the package (forge, updates, remote access, ...) are not what this server replaces
            .env("FLE_STUDIO_NO_BUNDLED", "1")
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
pub fn start_rust(src: &Path, home: &Path) -> SocketAddr {
    start_rust_with(src, home, fle_api::engines::GITHUB_API)
}

pub fn start_rust_with(src: &Path, home: &Path, github_api: &str) -> SocketAddr {
    start_rust_full(src, home, github_api, true)
}

/// `with_web_dir`: serve the page from the Studio checkout; without it the copy inside the binary is served.
pub fn start_rust_full(
    src: &Path,
    home: &Path,
    github_api: &str,
    with_web_dir: bool,
) -> SocketAddr {
    let st = new_studio(Options {
        home: home.to_path_buf(),
        web_dir: with_web_dir.then(|| src.join("fle_studio/web")),
        version: python_version(src),
        github_api: github_api.to_string(),
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
pub fn python_version(src: &Path) -> String {
    let t = std::fs::read_to_string(src.join("fle_studio/__init__.py")).unwrap();
    t.split("__version__ = \"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .to_string()
}
pub fn token_of(r: &Resp) -> String {
    let t = String::from_utf8_lossy(&r.body).to_string();
    t.split("name=\"fle-token\" content=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .to_string()
}

pub fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Ids and times that the servers make up (random ids, "now") become fixed names, so the rest can be compared.
pub fn canon(v: Value) -> Value {
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

pub const KEEP: [&str; 7] = [
    "content-type",
    "cache-control",
    "x-content-type-options",
    "referrer-policy",
    "content-security-policy",
    "x-frame-options",
    "x-fle-session",
];

pub fn compare(label: &str, a: &Resp, b: &Resp, ta: &str, tb: &str) {
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

//! The download worker against the Python Studio: the same queued jobs, pointed at a local server of our own, must end in the
//! same states, with the same files on disk and the same messages. Nothing here touches the real network.
#![cfg(unix)]
mod common;
mod harness;
use fle_api::server::Studio;
use harness::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn content(n: usize, salt: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[2] ^ salt)
        .collect()
}
fn sha(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

/// A tiny HTTP server with the behaviours a download has to survive.
fn fake_server() -> (String, Arc<AtomicUsize>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", l.local_addr().unwrap().port());
    let flaky = Arc::new(AtomicUsize::new(0));
    let (f2, short) = (flaky.clone(), Arc::new(AtomicUsize::new(0)));
    std::thread::spawn(move || {
        for s in l.incoming() {
            let (f2, short) = (f2.clone(), short.clone());
            std::thread::spawn(move || {
                let mut s = s.unwrap();
                let mut buf = vec![0u8; 8192];
                let n = s.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = req.split(' ').nth(1).unwrap_or("/").to_string();
                let range = req.lines().find_map(|l| {
                    l.to_lowercase()
                        .strip_prefix("range: bytes=")
                        .map(|r| r.trim_end_matches('-').trim().parse::<usize>().unwrap_or(0))
                });
                let (name, body): (&str, Vec<u8>) = match path.as_str() {
                    "/ok.bin" | "/redir-target.bin" => ("ok", content(3 * 1024 * 1024 + 17, 1)),
                    "/norange.bin" => ("norange", content(1_500_000, 2)),
                    "/flaky.bin" => ("flaky", content(900_000, 3)),
                    "/short.bin" => ("short", content(1_200_000, 4)),
                    "/slow.bin" => ("slow", content(6 * 1024 * 1024, 5)),
                    "/bad.bin" => ("bad", content(500_000, 6)),
                    "/unknown.bin" => ("unknown", content(777_777, 7)),
                    "/resume.bin" => ("resume", content(2_500_000, 8)),
                    "/redirect.bin" => {
                        let _ = s.write_all(format!("HTTP/1.1 302 Found\r\nLocation: {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", "/redir-target.bin").as_bytes());
                        return;
                    }
                    "/forbidden.bin" => {
                        let _ = s.write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                        return;
                    }
                    _ => {
                        let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                        return;
                    }
                };
                if name == "flaky" && f2.fetch_add(1, Ordering::SeqCst) == 0 {
                    let _ = s.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    return;
                }
                let honour = name != "norange";
                let start = if honour { range.unwrap_or(0) } else { 0 };
                if start >= body.len() && start > 0 {
                    let _ = s.write_all(b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    return;
                }
                let part = &body[start..];
                let head = if start > 0 {
                    format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n", part.len(), start, body.len() - 1, body.len())
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        part.len()
                    )
                };
                let _ = s.write_all(head.as_bytes());
                if name == "short" && short.fetch_add(1, Ordering::SeqCst) == 0 {
                    let _ = s.write_all(&part[..part.len() / 2]); // hang up half way
                    return;
                }
                if name == "slow" {
                    for c in part.chunks(1024 * 1024) {
                        if s.write_all(c).is_err() {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(500));
                    }
                    return;
                }
                let _ = s.write_all(part);
            });
        }
    });
    (base, flaky)
}

fn seed(home: &Path, base: &str) {
    let h = home.to_string_lossy();
    let m = home.join("models");
    std::fs::create_dir_all(m.join("pre")).unwrap();
    // an interrupted download (the first 900 kB are there) and a finished one, and a part that is too big
    let resume = content(2_500_000, 8);
    std::fs::write(m.join("resume.bin.part"), &resume[..900_000]).unwrap();
    std::fs::write(m.join("have.bin"), content(1000, 9)).unwrap();
    std::fs::write(m.join("toobig.bin.part"), content(9_000_000, 9)).unwrap();
    let ok = content(3 * 1024 * 1024 + 17, 1);
    let f = |name: &str, url: &str, size: usize, sha: Option<String>| json!({"url": format!("{base}{url}"), "dest": format!("{h}/models/{name}"), "size": size, "sha256": sha, "done": 0});
    let job = |id: &str, files: Vec<Value>| json!({"id": id, "title": format!("job {id}"), "kind": "model", "dedupe": format!("t:{id}"), "meta": {}, "state": "queued", "error": "", "created": 1000.5, "finished": null, "files": files, "speed": 5.0});
    let jobs = vec![
        job(
            "000000000001",
            vec![f("ok.bin", "/ok.bin", ok.len(), Some(sha(&ok)))],
        ),
        job(
            "000000000002",
            vec![f("bad.bin", "/bad.bin", 500_000, Some("0".repeat(64)))],
        ),
        job(
            "000000000003",
            vec![f("missing.bin", "/missing.bin", 10, None)],
        ),
        job(
            "000000000004",
            vec![f("forbidden.bin", "/forbidden.bin", 10, None)],
        ),
        job(
            "000000000005",
            vec![f(
                "resume.bin",
                "/resume.bin",
                2_500_000,
                Some(sha(&resume)),
            )],
        ),
        job(
            "000000000006",
            vec![f("norange.bin", "/norange.bin", 1_500_000, None)],
        ),
        job(
            "000000000007",
            vec![f("unknown.bin", "/unknown.bin", 0, None)],
        ),
        job(
            "000000000008",
            vec![
                f("have.bin", "/ok.bin", 1000, None),
                f(
                    "redirected.bin",
                    "/redirect.bin",
                    3 * 1024 * 1024 + 17,
                    None,
                ),
            ],
        ),
        job(
            "000000000009",
            vec![f("toobig.bin", "/bad.bin", 500_000, None)],
        ),
        job(
            "00000000000a",
            vec![f("flaky.bin", "/flaky.bin", 900_000, None)],
        ),
        job(
            "00000000000b",
            vec![f("short.bin", "/short.bin", 1_200_000, None)],
        ),
        job(
            "00000000000c",
            vec![f("slow.bin", "/slow.bin", 6 * 1024 * 1024, None)],
        ),
    ];
    std::fs::write(home.join("downloads.json"), Value::Array(jobs).to_string()).unwrap();
}

fn list(addr: SocketAddr, base: &str) -> Vec<Value> {
    let r = http(addr, "GET", "/api/downloads", &[], None);
    let v: Value = serde_json::from_slice(&r.body).unwrap();
    let mut out = v["downloads"].as_array().unwrap().clone();
    for j in out.iter_mut() {
        j["speed"] = json!(0);
        j["finished"] = json!(!j["finished"].is_null());
        j["created"] = json!(0);
        if j["id"] == "00000000000c" && j["state"] == "paused" {
            // where exactly the pause caught the slow transfer depends on the moment
            j["files"][0]["done"] = json!(0);
            j["bytes_done"] = json!(0);
            j["progress"] = json!(0);
        }
    }
    out.iter()
        .map(|j| {
            serde_json::from_str(
                &j.to_string()
                    .replace(base, "BASE")
                    .replace("-rs", "-X")
                    .replace("-py", "-X"),
            )
            .unwrap()
        })
        .collect()
}

/// Equal lists, or a message that names the first job and field that differ.
fn same(label: &str, a: &[Value], b: &[Value]) {
    assert_eq!(a.len(), b.len(), "{label}: number of jobs");
    for (x, y) in a.iter().zip(b) {
        for (k, v) in x.as_object().unwrap() {
            if k == "files" {
                for (i, f) in v.as_array().unwrap().iter().enumerate() {
                    assert_eq!(
                        f, &y["files"][i],
                        "{label}: job {} file {i}\n rust  : {}\n python: {}",
                        x["id"], f, y["files"][i]
                    );
                }
            }
            assert_eq!(v, &y[k], "{label}: job {} field {k}", x["id"]);
        }
    }
}

fn wait(addr: SocketAddr, id: &str, states: &[&str], secs: u64) -> Value {
    let t = Instant::now();
    loop {
        let r = http(addr, "GET", "/api/downloads", &[], None);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        if let Some(j) = v["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|j| j["id"] == id)
        {
            if states.contains(&j["state"].as_str().unwrap()) {
                return j.clone();
            }
        }
        assert!(
            t.elapsed() < Duration::from_secs(secs),
            "job {id} did not reach {states:?}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn files_under(root: &Path) -> Vec<(String, u64, String)> {
    fn walk(d: &Path, root: &Path, out: &mut Vec<(String, u64, String)>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else {
                let b = std::fs::read(&p).unwrap();
                out.push((
                    p.strip_prefix(root).unwrap().to_string_lossy().to_string(),
                    b.len() as u64,
                    sha(&b),
                ));
            }
        }
    }
    let mut out = vec![];
    walk(&root.join("models"), root, &mut out);
    out.sort();
    out
}

#[test]
fn the_download_worker_matches_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let (hp, hr) = (tmp("dl-py"), tmp("dl-rs"));
    let (bp, _fp) = fake_server();
    let (br, _fr) = fake_server();
    seed(&hp, &bp);
    seed(&hr, &br);
    let py = start_python(&src, &hp);
    let rs = start_rust(&src, &hr);
    let (tr, tp) = (
        token_of(&http(rs, "GET", "/", &[], None)),
        token_of(&http(py.addr, "GET", "/", &[], None)),
    );
    let pair = |label: &str, method: &str, path: &str| {
        let a = http(rs, method, path, &[("X-FLE-Token", &tr)], Some(b"{}"));
        let b = http(py.addr, method, path, &[("X-FLE-Token", &tp)], Some(b"{}"));
        assert_eq!(a.status, b.status, "[{label}] status");
        let norm = |r: &Resp, base: &str| {
            let mut v: Value = serde_json::from_slice(&r.body).unwrap();
            if v.get("speed").is_some() {
                if v["state"] == "downloading"
                    || (v["id"] == "00000000000c" && v["state"] != "done")
                {
                    // how far a running transfer has come at this moment is not the same on two servers
                    v["files"][0]["done"] = json!(0);
                    v["bytes_done"] = json!(0);
                    v["progress"] = json!(0);
                }
                v["speed"] = json!(0);
                v["finished"] = json!(!v["finished"].is_null());
                v["created"] = json!(0);
            }
            serde_json::from_str::<Value>(
                &v.to_string()
                    .replace(base, "BASE")
                    .replace("-rs", "-X")
                    .replace("-py", "-X"),
            )
            .unwrap()
        };
        assert_eq!(norm(&a, &br), norm(&b, &bp), "[{label}] body");
    };
    // the first eleven jobs run one after the other; the slow one is left for the pause test
    for id in [
        "000000000001",
        "000000000002",
        "000000000003",
        "000000000004",
        "000000000005",
        "000000000006",
        "000000000007",
        "000000000008",
        "000000000009",
        "00000000000a",
        "00000000000b",
    ] {
        let finish = ["done", "failed", "cancelled", "paused"];
        let (a, b) = (wait(rs, id, &finish, 90), wait(py.addr, id, &finish, 90));
        assert_eq!(
            a["state"], b["state"],
            "job {id}: rust {} python {}",
            a["error"], b["error"]
        );
    }
    // the slow job is running by now: pause it half way
    wait(rs, "00000000000c", &["downloading"], 30);
    wait(py.addr, "00000000000c", &["downloading"], 30);
    std::thread::sleep(Duration::from_millis(700));
    pair("pause running", "POST", "/api/downloads/00000000000c/pause");
    let (a, b) = (
        wait(rs, "00000000000c", &["paused", "done"], 30),
        wait(py.addr, "00000000000c", &["paused", "done"], 30),
    );
    assert_eq!(
        (&a["state"], &b["state"]),
        (&json!("paused"), &json!("paused"))
    );
    same("after the first round", &list(rs, &br), &list(py.addr, &bp));
    // the half-written slow file is where each server happened to be paused: leave it out here, the next check covers the whole file
    let settled = |h: &Path| {
        files_under(h)
            .into_iter()
            .filter(|f| f.0 != "models/slow.bin.part")
            .collect::<Vec<_>>()
    };
    assert_eq!(
        settled(&hr),
        settled(&hp),
        "files on disk after the first round"
    );
    // resume it and let it finish from where it stopped
    pair("resume", "POST", "/api/downloads/00000000000c/resume");
    wait(rs, "00000000000c", &["done", "failed"], 60);
    wait(py.addr, "00000000000c", &["done", "failed"], 60);
    same("after resume", &list(rs, &br), &list(py.addr, &bp));
    assert_eq!(
        files_under(&hr),
        files_under(&hp),
        "files on disk after resume"
    );
    // the actions on finished and failed jobs
    for (i, (id, what)) in [
        ("000000000003", "resume"),
        ("000000000001", "pause"),
        ("000000000001", "cancel"),
        ("000000000002", "cancel"),
        ("000000000009", "remove"),
        ("000000000001", "remove"),
        ("ffffffffffff", "remove"),
        ("000000000004", "remove"),
    ]
    .iter()
    .enumerate()
    {
        pair(
            &format!("action {i} {what}"),
            "POST",
            &format!("/api/downloads/{id}/{what}"),
        );
    }
    wait(rs, "000000000003", &["failed"], 30);
    wait(py.addr, "000000000003", &["failed"], 30);
    same("after the actions", &list(rs, &br), &list(py.addr, &bp));
    assert_eq!(
        files_under(&hr),
        files_under(&hp),
        "files on disk at the end"
    );
    for bad in [
        "/api/downloads/zz/remove",
        "/api/downloads/000000000001/explode",
        "/api/downloads/000000000001",
    ] {
        let (a, b) = (
            http(rs, "POST", bad, &[("X-FLE-Token", &tr)], Some(b"{}")),
            http(py.addr, "POST", bad, &[("X-FLE-Token", &tp)], Some(b"{}")),
        );
        assert_eq!(a.status, b.status, "{bad}");
    }
    let _: Option<&Studio> = None;
    eprintln!("compared with the Python Studio: 12 jobs, resume, pause, resume and 8 actions identical, files on disk identical");
}

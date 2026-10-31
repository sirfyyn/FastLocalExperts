//! fetch() against a local HTTP server: the same cases as FLE Studio's tests/test_downloads.py.
use fle_core::download::fetch;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::{Arc, Mutex};

fn data() -> Vec<u8> {
    (0..1_048_576u32).map(|i| (i % 251) as u8).collect()
}
fn sha(d: &[u8]) -> String {
    hex::encode(Sha256::digest(d))
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Range,
    NoRange,
    CutFirst,
}

struct Srv {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
}

fn serve(mode: Mode) -> Srv {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::<String>::new()));
    let reqs = requests.clone();
    let body = data();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
            let range = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("range: bytes=")
                        .map(|r| r.trim_end_matches('-').to_string())
                })
                .unwrap_or_default();
            let nth = {
                let mut r = reqs.lock().unwrap();
                r.push(range.clone());
                r.len()
            };
            let reply = |s: &mut std::net::TcpStream, code: &str, extra: &str, b: &[u8]| {
                let _ = write!(
                    s,
                    "HTTP/1.1 {code}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                    b.len()
                );
                let _ = s.write_all(b);
            };
            if path == "/missing" {
                reply(&mut s, "404 Not Found", "", b"");
                continue;
            }
            if path == "/gated" {
                reply(&mut s, "403 Forbidden", "", b"");
                continue;
            }
            let start: usize = if mode == Mode::NoRange {
                0
            } else {
                range.parse().unwrap_or(0)
            };
            if start >= body.len() && !range.is_empty() && mode != Mode::NoRange {
                reply(&mut s, "416 Range Not Satisfiable", "", b"");
                continue;
            }
            let part = &body[start..];
            let code = if start > 0 {
                "206 Partial Content"
            } else {
                "200 OK"
            };
            if mode == Mode::CutFirst && nth == 1 {
                let _ = write!(
                    s,
                    "HTTP/1.1 {code}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    part.len()
                );
                let _ = s.write_all(&part[..part.len() / 3]); // closes with bytes missing
                continue;
            }
            reply(&mut s, code, "", part);
        }
    });
    Srv { url, requests }
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("fle-test-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn part_of(dest: &Path) -> std::path::PathBuf {
    dest.with_extension(format!(
        "{}part",
        dest.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ))
}
fn go(url: &str, dest: &Path, s: Option<&str>) -> anyhow::Result<()> {
    fetch(url, dest, s, &mut |_, _| {})
}

#[test]
fn fresh_download_with_checksum() {
    let srv = serve(Mode::Range);
    let dest = tmp("fresh").join("m").join("a.gguf");
    go(&format!("{}/file", srv.url), &dest, Some(&sha(&data()))).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data());
    assert!(!part_of(&dest).exists());
}

#[test]
fn resume_continues_from_the_part_file_and_verifies_it() {
    let srv = serve(Mode::Range);
    let dest = tmp("resume").join("a.gguf");
    std::fs::write(part_of(&dest), &data()[..300_000]).unwrap();
    go(&format!("{}/file", srv.url), &dest, Some(&sha(&data()))).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data());
    assert_eq!(srv.requests.lock().unwrap().as_slice(), ["300000"]);
}

#[test]
fn server_without_range_restarts_the_file() {
    let srv = serve(Mode::NoRange);
    let dest = tmp("norange").join("a.gguf");
    std::fs::write(part_of(&dest), &data()[..1000]).unwrap();
    go(&format!("{}/file", srv.url), &dest, Some(&sha(&data()))).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data()); // one copy, not the old 1000 bytes plus a whole new file
}

#[test]
fn complete_part_file_answered_with_416_is_accepted() {
    let srv = serve(Mode::Range);
    let dest = tmp("416").join("a.gguf");
    std::fs::write(part_of(&dest), data()).unwrap();
    go(&format!("{}/file", srv.url), &dest, None).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data());
}

#[test]
fn existing_file_is_not_fetched_again() {
    let srv = serve(Mode::Range);
    let dest = tmp("exists").join("a.gguf");
    std::fs::write(&dest, data()).unwrap();
    go(&format!("{}/file", srv.url), &dest, Some(&sha(&data()))).unwrap();
    assert!(srv.requests.lock().unwrap().is_empty());
}

#[test]
fn checksum_mismatch_moves_the_file_aside() {
    let srv = serve(Mode::Range);
    let dest = tmp("bad").join("a.gguf");
    let e = go(&format!("{}/file", srv.url), &dest, Some(&"0".repeat(64)))
        .unwrap_err()
        .to_string();
    assert!(e.contains("checksum mismatch"), "{e}");
    assert!(!dest.exists());
    assert_eq!(std::fs::read(dest.with_extension("bad")).unwrap(), data());
}

#[test]
fn http_404_and_403_become_readable_errors() {
    let srv = serve(Mode::Range);
    let d = tmp("err");
    let e = go(&format!("{}/missing", srv.url), &d.join("a.gguf"), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("404"), "{e}");
    let e = go(&format!("{}/gated", srv.url), &d.join("b.gguf"), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("access denied"), "{e}");
}

#[test]
fn connection_cut_mid_file_is_resumed_by_running_again() {
    let srv = serve(Mode::CutFirst);
    let dest = tmp("cut").join("a.gguf");
    let url = format!("{}/file", srv.url);
    let e = go(&url, &dest, Some(&sha(&data())))
        .unwrap_err()
        .to_string();
    assert!(e.contains("ended early") || e.contains("error"), "{e}");
    assert!(
        part_of(&dest).exists(),
        "the .part file stays for the resume"
    );
    go(&url, &dest, Some(&sha(&data()))).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data());
    let r = srv.requests.lock().unwrap();
    assert_eq!(r.len(), 2);
    assert!(!r[1].is_empty(), "the second request asked for the rest");
}

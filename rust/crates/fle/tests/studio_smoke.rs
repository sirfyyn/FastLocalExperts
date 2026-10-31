//! `fle studio` as a user starts it: the program, a folder of its own, the page from inside the binary.
#![cfg(unix)]
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn get(port: u16, path: &str) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.write_all(
        format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
            .as_bytes(),
    )
    .unwrap();
    let mut raw = String::new();
    let _ = s.read_to_string(&mut raw);
    let status = raw
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    (
        status,
        raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string(),
    )
}

#[test]
fn fle_studio_serves_the_page_and_the_api_and_stops_on_a_signal() {
    let home = std::env::temp_dir().join(format!("fle-smoke-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    // a port that is free now
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_fle"))
        .args(["studio", "--no-browser", "--home"])
        .arg(&home)
        .args(["--port", &port.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let t = Instant::now();
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            t.elapsed() < Duration::from_secs(20),
            "the server did not start"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    let (code, body) = get(port, "/api/ping");
    assert_eq!(code, 200);
    assert!(body.contains("\"app\":\"fle-studio\""), "{body}");
    let (code, page) = get(port, "/");
    assert_eq!(code, 200);
    assert!(
        page.contains("<meta name=\"fle-token\" content=\"") && !page.contains("__FLE_TOKEN__"),
        "the page carries a session token"
    );
    let (code, state) = get(port, "/api/state");
    assert_eq!(code, 200);
    assert!(
        state.contains("\"data_dir\""),
        "{}",
        &state[..state.len().min(200)]
    );
    assert!(home.join("chats").exists(), "the data folder is made");
    // a second start on the same port says so and ends
    let again = Command::new(env!("CARGO_BIN_EXE_fle"))
        .args(["studio", "--no-browser", "--home"])
        .arg(&home)
        .args(["--port", &port.to_string()])
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&again.stdout).contains("already running"),
        "{}",
        String::from_utf8_lossy(&again.stdout)
    );
    // SIGTERM ends it cleanly
    unsafe {
        libc_kill(child.id() as i32);
    }
    let t = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(
            t.elapsed() < Duration::from_secs(20),
            "the server did not stop"
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    assert!(status.success(), "{status:?}");
    let mut out = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    assert!(out.contains("Stopping FLE Studio"), "{out}");
    let _ = std::fs::remove_dir_all(&home);
}

extern "C" {
    #[link_name = "kill"]
    fn libc_kill_raw(pid: i32, sig: i32) -> i32;
}
unsafe fn libc_kill(pid: i32) {
    libc_kill_raw(pid, 15);
}

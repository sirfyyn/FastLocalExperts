//! Launch profiles over HTTP against the Python Studio: list, save (from pasted text and from fields), the errors, delete and
//! the reset of the chosen profile, and the file on disk.
#![cfg(unix)]
mod harness;
use harness::*;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::Path;

fn seed(home: &Path) {
    let items = json!([
        {"id": "mine", "name": "Mine", "engine": "fle", "binary": "", "args": ["-c", "4096"], "env": {"A": "1"}, "notes": "n", "builtin": false, "meta": {"tuned": {"x": 1}}, "command": "ignored"},
        {"id": "plain", "name": "Plain"},
        {"id": "weird", "name": "Weird", "args": ["--x", "a b"], "env": {"K": "v w"}, "binary": "/bin/x", "engine": "custom"},
        {"id": "bad", "name": "Bad", "surprise": true},
        {"id": "after-bad", "name": "Never loaded"}
    ]);
    std::fs::write(home.join("profiles.json"), items.to_string()).unwrap();
}

/// Profile ids made from a name end in the time of the day, which differs between the two servers.
fn unstamp(v: &mut Value) {
    match v {
        Value::Object(o) => {
            for (k, x) in o.iter_mut() {
                if k == "id" {
                    if let Some(s) = x.as_str() {
                        if let Some((head, tail)) = s.rsplit_once('-') {
                            if !tail.is_empty()
                                && tail.len() <= 5
                                && tail.bytes().all(|b| b.is_ascii_digit())
                            {
                                *x = json!(format!("{head}-<time>"));
                            }
                        }
                    }
                } else {
                    unstamp(x);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(unstamp),
        _ => {}
    }
}

#[test]
fn profiles_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let (hp, hr) = (tmp("prof-py"), tmp("prof-rs"));
    seed(&hp);
    seed(&hr);
    let py = start_python(&src, &hp);
    let rs: SocketAddr = start_rust(&src, &hr);
    let (tr, tp) = (
        token_of(&http(rs, "GET", "/", &[], None)),
        token_of(&http(py.addr, "GET", "/", &[], None)),
    );
    let n = std::cell::Cell::new(0);
    let both = |label: &str, method: &str, path: &str, body: Option<&str>| -> (Resp, Resp) {
        let (a, b) = (
            http(
                rs,
                method,
                path,
                &[("X-FLE-Token", &tr)],
                body.map(str::as_bytes),
            ),
            http(
                py.addr,
                method,
                path,
                &[("X-FLE-Token", &tp)],
                body.map(str::as_bytes),
            ),
        );
        assert_eq!(
            a.status,
            b.status,
            "[{label}] status: {}",
            String::from_utf8_lossy(&a.body)
        );
        let norm = |r: &Resp| {
            let text = String::from_utf8_lossy(&r.body)
                .replace("-rs", "-X")
                .replace("-py", "-X");
            let mut v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            unstamp(&mut v);
            canon(v)
        };
        assert_eq!(norm(&a), norm(&b), "[{label}] body");
        n.set(n.get() + 1);
        (a, b)
    };
    both("list", "GET", "/api/profiles", None);
    let save = |label: &str, body: Value| {
        both(label, "POST", "/api/profiles", Some(&body.to_string()));
    };
    save(
        "save pasted",
        json!({"name": "Pasted", "command": "FLE_A=1 FLE_B='x y' /opt/llama-server -m /m/a.gguf \\\n  -c 8192 --jinja", "notes": "from the card"}),
    );
    save(
        "save pasted without binary",
        json!({"name": "No Binary", "command": "FLE_A=1 -c 4096 --jinja"}),
    );
    save(
        "save pasted llama-server",
        json!({"name": "Plain Name", "command": "llama-server -c 1000"}),
    );
    save(
        "save pasted engine placeholder",
        json!({"name": "Placeholder", "command": "{engine} -c 1000", "engine": "upstream"}),
    );
    save(
        "save pasted systemd",
        json!({"name": "Unit", "command": "ExecStart=/usr/bin/llama-server -m x --port 1", "engine": "custom"}),
    );
    save(
        "save fields",
        json!({"name": "Fields", "engine": "upstream", "binary": "", "args": ["-c", "1"], "env": {"X": "y"}}),
    );
    save(
        "save fields custom",
        json!({"name": "Custom", "binary": "/bin/my-server", "args": []}),
    );
    save(
        "save keep id",
        json!({"id": "mine", "name": "Mine renamed", "args": ["-c", "2"]}),
    );
    save(
        "save keep id keeps meta",
        json!({"id": "mine", "name": "Mine again", "command": "bin -x"}),
    );
    save(
        "save with meta",
        json!({"id": "tuned", "name": "Tuned", "meta": {"tuned": {"speed": 12.5}}, "args": []}),
    );
    save(
        "save builtin id gets a new one",
        json!({"id": "fle-reference", "name": "Copy of the reference", "command": "-c 1"}),
    );
    save(
        "save long name",
        json!({"name": "A very long profile name with — odd chars & symbols! that goes past forty characters", "args": []}),
    );
    save("save symbols only name", json!({"name": "!!!", "args": []}));
    save("save unicode name", json!({"name": "Größe", "args": []}));
    save("no name", json!({"command": "bin"}));
    save("blank name", json!({"name": "   "}));
    save("bad engine", json!({"name": "X", "engine": "turbo"}));
    save(
        "custom without binary",
        json!({"name": "X", "engine": "custom"}),
    );
    save(
        "unclosed quote",
        json!({"name": "X", "command": "bin 'oops"}),
    );
    both("bad body", "POST", "/api/profiles", Some("nope"));
    both("array body", "POST", "/api/profiles", Some("[]"));
    both("list after", "GET", "/api/profiles", None);
    // choose a profile, delete it: the choice goes back to auto
    for (addr, tok) in [(rs, &tr), (py.addr, &tp)] {
        let r = http(
            addr,
            "POST",
            "/api/settings",
            &[("X-FLE-Token", tok)],
            Some(
                json!({"launch": {"profile": "weird"}})
                    .to_string()
                    .as_bytes(),
            ),
        );
        assert_eq!(r.status, 200);
    }
    both(
        "delete chosen",
        "POST",
        "/api/profiles/weird/delete",
        Some("{}"),
    );
    both("settings after delete", "GET", "/api/settings", None);
    both(
        "delete other",
        "POST",
        "/api/profiles/plain/delete",
        Some("{}"),
    );
    both(
        "delete unknown",
        "POST",
        "/api/profiles/nope/delete",
        Some("{}"),
    );
    both(
        "delete builtin",
        "POST",
        "/api/profiles/fle-reference/delete",
        Some("{}"),
    );
    both("list at the end", "GET", "/api/profiles", None);
    let disk = |h: &Path| {
        let mut v: Value =
            serde_json::from_str(&std::fs::read_to_string(h.join("profiles.json")).unwrap())
                .unwrap();
        unstamp(&mut v);
        v
    };
    assert_eq!(disk(&hr), disk(&hp), "profiles.json on disk");
    eprintln!(
        "compared with the Python Studio: {} requests and profiles.json identical",
        n.get()
    );
}

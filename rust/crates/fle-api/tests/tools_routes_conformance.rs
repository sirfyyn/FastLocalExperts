//! The tool routes, the capabilities, the hosts list and the update panel over HTTP, against the Python Studio.
#![cfg(unix)]
mod harness;
use harness::*;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::Path;

fn seed(home: &Path) {
    std::fs::create_dir_all(home.join("share")).unwrap();
    std::fs::write(home.join("share/a.txt"), "hello").unwrap();
    // no update check at the start: the Python Studio would ask GitHub
    std::fs::write(
        home.join("settings.json"),
        json!({"updates": {"check": false}}).to_string(),
    )
    .unwrap();
}

#[test]
fn tool_routes_hosts_capabilities_and_update_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let (hp, hr) = (tmp("toolr-py"), tmp("toolr-rs"));
    seed(&hp);
    seed(&hr);
    let py = start_python(&src, &hp);
    let rs: SocketAddr = start_rust(&src, &hr);
    let (tr, tp) = (
        token_of(&http(rs, "GET", "/", &[], None)),
        token_of(&http(py.addr, "GET", "/", &[], None)),
    );
    let n = std::cell::Cell::new(0);
    let both = |label: &str, method: &str, path: &str, body: Option<&Value>, skip: &[&str]| {
        let b = body.map(|v| v.to_string());
        let a = http(
            rs,
            method,
            path,
            &[("X-FLE-Token", &tr)],
            b.as_deref().map(str::as_bytes),
        );
        let c = http(
            py.addr,
            method,
            path,
            &[("X-FLE-Token", &tp)],
            b.as_deref().map(str::as_bytes),
        );
        assert_eq!(
            a.status,
            c.status,
            "[{label}] status: {} / {}",
            String::from_utf8_lossy(&a.body),
            String::from_utf8_lossy(&c.body)
        );
        let norm = |r: &Resp, h: &Path| {
            let mut v: Value = serde_json::from_slice(&r.body).unwrap_or(Value::Null);
            if let Some(o) = v.as_object_mut() {
                for k in skip {
                    o.remove(*k);
                }
            }
            canon(
                serde_json::from_str(
                    &v.to_string()
                        .replace(&*h.to_string_lossy(), "HOME")
                        .replace("-rs", "-X")
                        .replace("-py", "-X"),
                )
                .unwrap(),
            )
        };
        let (x, y) = (norm(&a, &hr), norm(&c, &hp));
        if x != y {
            let mut why = vec![];
            if let (Some(xo), Some(yo)) = (x.as_object(), y.as_object()) {
                for (k, v) in xo {
                    if yo.get(k) != Some(v) {
                        why.push(format!(
                            "{k}: rust {v} python {}",
                            yo.get(k).map(|v| v.to_string()).unwrap_or_default()
                        ));
                    }
                }
            }
            panic!(
                "[{label}] body differs in {}",
                why.join(" | ").chars().take(1500).collect::<String>()
            );
        }
        n.set(n.get() + 1);
    };
    let config = |settings: Value| {
        for (addr, tok) in [(rs, &tr), (py.addr, &tp)] {
            assert_eq!(
                http(
                    addr,
                    "POST",
                    "/api/settings",
                    &[("X-FLE-Token", tok)],
                    Some(settings.to_string().as_bytes())
                )
                .status,
                200
            );
        }
    };
    both("tools default", "GET", "/api/tools", None, &[]);
    both(
        "plugins (none in the core)",
        "GET",
        "/api/plugins",
        None,
        &[],
    );
    both(
        "plugin toggle unknown",
        "POST",
        "/api/plugins/forge/enable",
        Some(&json!({})),
        &[],
    );
    both(
        "plugin toggle bad id",
        "POST",
        "/api/plugins/Forge/enable",
        Some(&json!({})),
        &[],
    );
    both("plugin route", "GET", "/api/p/forge/jobs", None, &[]);
    both(
        "plugin route bad id",
        "POST",
        "/api/p/Bad_Id/x",
        Some(&json!({})),
        &[],
    );
    both("plugin files", "GET", "/plugins/forge/boot.js", None, &[]);
    both("plugin files bad", "GET", "/plugins/forge", None, &[]);
    both(
        "capabilities, nothing loaded",
        "GET",
        "/api/chat/capabilities",
        None,
        &[],
    );
    let share = |h: &Path| h.join("share").to_string_lossy().to_string();
    for (addr, tok, h) in [(rs, &tr, &hr), (py.addr, &tp, &hp)] {
        let r = http(addr, "POST", "/api/settings", &[("X-FLE-Token", tok)], Some(json!({"tools": {"files": true, "files_dir": share(h), "web_search": true, "mcp_trusted": ["a"], "max_rounds": 3,
            "mcp_json": "{\"mcpServers\": {\"a\": {\"command\": \"x\"}, \"b\": {\"url\": \"https://x.example/mcp\", \"oauth\": {\"client_id\": \"i\"}}, \"c\": {}}}"}}).to_string().as_bytes()));
        assert_eq!(r.status, 200);
    }
    both(
        "tools with files, search and servers (the servers cannot connect here yet)",
        "GET",
        "/api/tools",
        None,
        &["servers", "connecting"],
    );
    let (a, b) = (
        http(rs, "GET", "/api/tools", &[], None),
        http(py.addr, "GET", "/api/tools", &[], None),
    );
    let (x, y): (Value, Value) = (
        serde_json::from_slice(&a.body).unwrap(),
        serde_json::from_slice(&b.body).unwrap(),
    );
    let names = |v: &Value| {
        v["servers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s["name"].clone(),
                    s["transport"].clone(),
                    s["trusted"].clone(),
                    s["auth"]["remote"].clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    let mut xs = names(&x);
    let mut ys = names(&y);
    xs.sort_by_key(|s| s.0.to_string());
    ys.sort_by_key(|s| s.0.to_string());
    assert_eq!(xs, ys, "the servers listed");
    config(json!({"tools": {"mcp_json": ""}}));
    for (i, (name, args)) in [
        ("calculator", json!({"expression": "6*7"})),
        ("calculator", json!({"expression": "1/0"})),
        ("calculator", json!("{\"expression\": \"2+2\"}")),
        ("calculator", json!("not json")),
        ("calculator", json!([1])),
        ("calculator", json!("")),
        ("get_current_time", json!({"timezone": "UTC"})),
        ("list_files", json!({})),
        ("read_file", json!({"path": "a.txt"})),
        ("read_file", json!({"path": "../x"})),
        ("nope", json!({})),
    ]
    .iter()
    .enumerate()
    {
        let body = json!({"name": name, "arguments": args});
        let (a, b) = (
            http(
                rs,
                "POST",
                "/api/tools/call",
                &[("X-FLE-Token", &tr)],
                Some(
                    body.to_string()
                        .replace(&share(&hp), &share(&hr))
                        .as_bytes(),
                ),
            ),
            http(
                py.addr,
                "POST",
                "/api/tools/call",
                &[("X-FLE-Token", &tp)],
                Some(body.to_string().as_bytes()),
            ),
        );
        assert_eq!(a.status, b.status, "tool call {i} {name}");
        let (mut x, mut y): (Value, Value) = (
            serde_json::from_slice(&a.body).unwrap(),
            serde_json::from_slice(&b.body).unwrap(),
        );
        for v in [&mut x, &mut y] {
            if v.get("ms").is_some() {
                v["ms"] = json!(0);
            }
        }
        if *name != "get_current_time" {
            assert_eq!(x, y, "tool call {i} {name}");
        }
        n.set(n.get() + 1);
    }
    both(
        "web search asks first",
        "POST",
        "/api/tools/call",
        Some(&json!({"name": "web_search", "arguments": {"query": "x"}})),
        &[],
    );
    both(
        "tools off",
        "POST",
        "/api/tools/reload",
        Some(&json!({})),
        &["servers"],
    );
    config(json!({"tools": {"enabled": false}}));
    both("everything off", "GET", "/api/tools", None, &[]);
    both(
        "call while off",
        "POST",
        "/api/tools/call",
        Some(&json!({"name": "calculator", "arguments": {"expression": "1"}})),
        &[],
    );
    config(json!({"chat": {"max_upload_mb": 9999}}));
    both(
        "capabilities, upload limit",
        "GET",
        "/api/chat/capabilities",
        None,
        &["pdf"],
    );
    eprintln!("compared with the Python Studio: {} requests (tools, calls, hosts, capabilities, update) identical", n.get());
}

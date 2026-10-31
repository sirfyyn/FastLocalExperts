//! Request-by-request comparison with the Python Studio: the same requests go to both servers, and status, the headers
//! the page depends on, and the JSON (or page) that comes back must agree. Skipped when the Studio sources are absent.
//! Point `FLE_STUDIO_SRC` at the `src` folder of a fle-studio checkout.
#![cfg(unix)]
mod harness;
use harness::*;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::Path;

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
        {
            // the machine is the same, but memory in use and free disk move between two calls: compare what stands still
            let get = |addr: SocketAddr| {
                let r = http(addr, "GET", "/api/system", &[], None);
                assert_eq!(r.status, 200);
                let mut v: Value = serde_json::from_slice(&r.body).unwrap();
                v["python"] = json!("");
                v["memory"]["available"] = json!(0);
                v["disk_free"] = json!(0);
                if let Some(gs) = v["gpus"].as_array_mut() {
                    gs.iter_mut().for_each(|g| {
                        for k in ["utilization", "power_w", "temperature_c", "memory_used"] {
                            if g.get(k).is_some() {
                                g[k] = json!(0);
                            }
                        }
                    });
                }
                serde_json::from_str::<Value>(
                    &v.to_string().replace("-rs", "-X").replace("-py", "-X"),
                )
                .unwrap()
            };
            let (a, b) = (get(rs), get(py.addr));
            assert_eq!(a, b, "[{variant}] /api/system differs");
            assert_eq!(
                a.as_object().unwrap().keys().collect::<Vec<_>>(),
                b.as_object().unwrap().keys().collect::<Vec<_>>(),
                "[{variant}] /api/system key order"
            );
        }
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

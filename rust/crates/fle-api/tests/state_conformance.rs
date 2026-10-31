//! `/api/state` (everything the page loads at once) and the page served from the copy inside the binary, against the
//! Python Studio.
#![cfg(unix)]
mod common;
mod harness;
use common::{kv, tensors, write};
use harness::*;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::Path;

fn seed(home: &Path) {
    let m = home.join("models");
    std::fs::create_dir_all(&m).unwrap();
    write(
        &m.join("moe.gguf"),
        &kv("qwen3moe", "Own MoE", true),
        &tensors(true),
    );
    write(
        &m.join("dense.gguf"),
        &kv("qwen3", "Own Dense", false),
        &tensors(false),
    );
    std::fs::write(
        home.join("profiles.json"),
        json!([{"id": "mine", "name": "Mine", "args": ["-c", "1"]}]).to_string(),
    )
    .unwrap();
    let h = home.to_string_lossy();
    std::fs::write(home.join("downloads.json"), json!([{"id": "aaaaaaaaaaaa", "title": "done one", "kind": "model", "dedupe": "catalog:fle-iq3_m", "meta": {}, "state": "failed", "error": "boom", "created": 1.5, "finished": null,
        "files": [{"url": "https://example.invalid/a", "dest": format!("{h}/models/a"), "size": 100, "sha256": null, "done": 30}], "speed": 0.0}]).to_string()).unwrap();
    std::fs::write(
        home.join("settings.json"),
        json!({"theme": "dark", "hf_token": "secret", "launch": {"context": 8192}}).to_string(),
    )
    .unwrap();
}

fn scrub(mut v: Value) -> Value {
    v["system"]["python"] = json!("");
    v["system"]["memory"]["available"] = json!(0);
    v["system"]["disk_free"] = json!(0);
    if let Some(gs) = v["system"]["gpus"].as_array_mut() {
        gs.iter_mut().for_each(|g| {
            for k in ["utilization", "power_w", "temperature_c", "memory_used"] {
                if g.get(k).is_some() {
                    g[k] = json!(0);
                }
            }
        });
    }
    let t = v.to_string().replace("-rs", "-X").replace("-py", "-X");
    canon(serde_json::from_str(&t).unwrap())
}

#[test]
fn state_and_the_embedded_page_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let (hp, hr) = (tmp("state-py"), tmp("state-rs"));
    seed(&hp);
    seed(&hr);
    let py = start_python(&src, &hp);
    let rs: SocketAddr = start_rust_full(&src, &hr, fle_api::engines::GITHUB_API, false); // the page comes from the binary
    let (ia, ib) = (
        http(rs, "GET", "/", &[], None),
        http(py.addr, "GET", "/", &[], None),
    );
    let (tr, tp) = (token_of(&ia), token_of(&ib));
    compare("index from the binary", &ia, &ib, &tr, &tp);
    let (a, b) = (
        http(rs, "GET", "/api/state", &[], None),
        http(py.addr, "GET", "/api/state", &[], None),
    );
    assert_eq!((a.status, b.status), (200, 200));
    let (x, y): (Value, Value) = (
        serde_json::from_slice(&a.body).unwrap(),
        serde_json::from_slice(&b.body).unwrap(),
    );
    assert_eq!(
        x.as_object().unwrap().keys().collect::<Vec<_>>(),
        y.as_object().unwrap().keys().collect::<Vec<_>>(),
        "state keys and their order"
    );
    let (x, y) = (scrub(x), scrub(y));
    for k in x.as_object().unwrap().keys() {
        assert_eq!(x[k], y[k], "state section {k}");
    }
    // the same page, file by file
    fn files(dir: &Path, base: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                files(&p, base, out);
            } else {
                out.push(format!(
                    "/{}",
                    p.strip_prefix(base).unwrap().to_string_lossy()
                ));
            }
        }
    }
    let web = src.join("fle_studio/web");
    let mut list = vec![];
    files(&web, &web, &mut list);
    list.sort();
    for f in &list {
        let (a, b) = (
            http(rs, "GET", f, &[], None),
            http(py.addr, "GET", f, &[], None),
        );
        compare(&format!("page file {f}"), &a, &b, &tr, &tp);
    }
    // and the repository's own copy is what the Studio checkout has
    let own = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../web/studio");
    for f in &list {
        let (a, b) = (
            std::fs::read(web.join(f.trim_start_matches('/'))).unwrap(),
            std::fs::read(own.join(f.trim_start_matches('/'))).unwrap_or_default(),
        );
        assert_eq!(a, b, "web/studio{f} differs from the Studio checkout");
    }
    eprintln!(
        "compared with the Python Studio: /api/state ({} sections) and {} page files identical",
        x.as_object().unwrap().len(),
        list.len()
    );
}

//! Models, catalog, extras and downloads against the Python Studio: the same folder of models (and the same
//! `library.json` and `downloads.json`) is given to both servers, and the same requests must get the same answers.
#![cfg(unix)]
mod common;
mod harness;
use common::{kv, tensors, write};
use harness::*;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::Path;

const FLE: &str = "sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF";

fn seed(home: &Path) {
    let m = home.join("models");
    let mk = |rel: &str, arch: &'static str, name: &'static str, experts: bool| {
        let p = m.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        write(&p, &kv(arch, name, experts), &tensors(experts));
    };
    // a catalog model with one of its two parts present, and its MTP files under the repo folders
    mk(
        &format!("{FLE}/IQ3_M/Qwen3.8-Flash-Next-FLE-IQ3_M-00001-of-00002.gguf"),
        "qwen3moe",
        "Whatever",
        true,
    );
    mk(
        &format!("{FLE}/mtp/mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf"),
        "qwen3",
        "Head",
        false,
    );
    std::fs::write(m.join(FLE).join("mtp/fle-mtp-vocab-32k.txt"), b"vocab").unwrap();
    // a catalog model with both parts (tiny files): complete, supports MTP, but its head and vocabulary are not there
    mk(
        &format!("{FLE}/IQ3_M/Qwen3.8-Flash-Next-FLE-IQ3_M-00002-of-00002.gguf"),
        "qwen3moe",
        "Whatever",
        true,
    );
    // a catalog model whose second part is being downloaded
    mk(
        &format!("{FLE}/IQ2_M/Qwen3.8-Flash-Next-FLE-IQ2_M-00001-of-00002.gguf"),
        "qwen3moe",
        "Two",
        true,
    );
    std::fs::write(
        m.join(FLE)
            .join("IQ2_M/Qwen3.8-Flash-Next-FLE-IQ2_M-00002-of-00002.gguf.part"),
        vec![0u8; 777],
    )
    .unwrap();
    mk("own/moe.gguf", "qwen3moe", "Own MoE", true);
    mk("own/sub/dense.gguf", "qwen3", "Own Dense", false);
    mk("own/noname.gguf", "qwen3", "", false);
    mk("big-00001-of-00002.gguf", "qwen3moe", "Big", true);
    mk("big-00002-of-00002.gguf", "qwen3moe", "Big", true);
    mk("orphan-00002-of-00002.gguf", "qwen3moe", "Orphan", true);
    mk("mmproj-foo.gguf", "clip", "Vision", false);
    mk("Draft-mtp-x.gguf", "qwen3", "Draft", false);
    mk("UP.GGUF", "qwen3", "Upper", false);
    mk("_downloads/hidden-by-name.gguf", "qwen3", "No", false);
    mk(".hidden/dot.gguf", "qwen3", "No", false);
    std::fs::write(m.join("garbage.gguf"), b"not a gguf").unwrap();
    std::fs::write(m.join("notes.txt"), b"hello").unwrap();
    // an imported file outside the folder, one that is gone, and the MTP files next to a neighbour
    let ext = home.join("ext");
    std::fs::create_dir_all(ext.join("mtp")).unwrap();
    write(
        &ext.join("outside.gguf"),
        &kv("qwen3moe", "Outside", true),
        &tensors(true),
    );
    write(
        &ext.join("outside-two-00001-of-00002.gguf"),
        &kv("qwen3moe", "Outside Two", true),
        &tensors(true),
    );
    std::fs::write(ext.join("mtp/fle-mtp-vocab-32k.txt"), b"v").unwrap();
    let h = home.to_string_lossy();
    std::fs::write(
        home.join("library.json"),
        json!({"imports": [{"path": format!("{h}/ext/outside.gguf"), "added": 1}, {"path": format!("{h}/ext/gone.gguf"), "added": 2},
            {"path": format!("{h}/ext/outside-two-00001-of-00002.gguf"), "added": 3}, {"nope": 1}, "junk"]})
        .to_string(),
    )
    .unwrap();
    let file = |dest: &str, size: i64, done: i64| json!({"url": "https://example.invalid/x", "dest": format!("{h}/{dest}"), "size": size, "sha256": null, "done": done});
    let job = |id: &str, state: &str, dedupe: &str, files: Vec<Value>| {
        json!({"id": id, "title": format!("job {id}"), "kind": "model", "dedupe": dedupe, "meta": {"catalog_id": "x"}, "state": state, "error": if state == "failed" { "boom" } else { "" },
            "created": 1000.5, "finished": if state == "done" { json!(2000.5) } else { Value::Null }, "files": files, "speed": 99.0, "_flag": "pause"})
    };
    std::fs::write(
        home.join("downloads.json"),
        json!([
            job(
                "aaaaaaaaaaaa",
                "done",
                "catalog:fle-iq3_m",
                vec![file("models/a", 100, 100)]
            ),
            job(
                "bbbbbbbbbbbb",
                "failed",
                "catalog:fle-iq3_m",
                vec![file("models/b", 100, 40), file("models/c", 50, 0)]
            ),
            job(
                "cccccccccccc",
                "cancelled",
                "catalog:fle-iq2_m",
                vec![file("models/d", 10, 0)]
            ),
            job(
                "dddddddddddd",
                "paused",
                "catalog:gpt-oss-20b",
                vec![file("models/e", 0, 0)]
            ),
            job("eeeeeeeeeeee", "done", "hf:x", vec![]),
        ])
        .to_string(),
    )
    .unwrap();
}

#[test]
fn models_catalog_and_downloads_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let (hp, hr) = (tmp("lib-py"), tmp("lib-rs"));
    seed(&hp);
    seed(&hr);
    let py = start_python(&src, &hp);
    let rs: SocketAddr = start_rust(&src, &hr);
    let (ia, ib) = (
        http(rs, "GET", "/", &[], None),
        http(py.addr, "GET", "/", &[], None),
    );
    let (tr, tp) = (token_of(&ia), token_of(&ib));
    let mut n = 0;
    let mut both = |label: &str, method: &str, path: &str, body: Option<&str>| {
        let run = |addr: SocketAddr, tok: &str, home: &Path| {
            let sub = |t: &str| {
                let mut t = t.replace("{HOME}", &home.to_string_lossy());
                // {model:<file name>} is the id this server gave that file
                while let Some(i) = t.find("{model:") {
                    let j = t[i..].find('}').unwrap() + i;
                    let file = t[i + 7..j].to_string();
                    let r = http(addr, "GET", "/api/models", &[], None);
                    let v: Value = serde_json::from_slice(&r.body).unwrap();
                    let id = v["models"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|m| m["file"] == file.as_str())
                        .map(|m| m["id"].as_str().unwrap().to_string())
                        .unwrap_or_else(|| "000000000000".into());
                    t = format!("{}{}{}", &t[..i], id, &t[j + 1..]);
                }
                t
            };
            let (p, b) = (sub(path), body.map(sub));
            http(
                addr,
                method,
                &p,
                &[("X-FLE-Token", tok)],
                b.as_deref().map(str::as_bytes),
            )
        };
        let (a, b) = (run(rs, &tr, &hr), run(py.addr, &tp, &hp));
        compare(label, &a, &b, &tr, &tp);
        n += 1;
    };
    both("models", "GET", "/api/models", None);
    both("catalog", "GET", "/api/catalog", None);
    both("downloads", "GET", "/api/downloads", None);
    for (i, body) in [
        r#"{"path": "{HOME}/ext/outside.gguf"}"#,
        r#"{"path": "{HOME}/ext/nope.gguf"}"#,
        r#"{"path": "{HOME}/models/notes.txt"}"#,
        r#"{"path": "{HOME}/models/garbage.gguf"}"#,
        r#"{"path": "\"{HOME}/models/own/moe.gguf\""}"#,
        r#"{"path": "  {HOME}/ext/outside-two-00002-of-00002.gguf  "}"#,
        r#"{"path": ""}"#,
        r#"{}"#,
        r#"{"path": null}"#,
        r#"[]"#,
        r#"nope"#,
    ]
    .iter()
    .enumerate()
    {
        both(&format!("add {i}"), "POST", "/api/models/add", Some(body));
        both(&format!("models after add {i}"), "GET", "/api/models", None);
    }
    // the plan preview (no engine is installed here: an answer with the fix, not a failure)
    let plan_body = |m: &str, extra: &str| format!("{{\"model\": \"{m}\"{extra}}}");
    let iq3 = "{model:Qwen3.8-Flash-Next-FLE-IQ3_M-00001-of-00002.gguf}";
    both(
        "plan, a catalog model needing MTP files",
        "POST",
        "/api/engine/plan",
        Some(&plan_body(iq3, "")),
    );
    both(
        "plan, a model with no catalog entry",
        "POST",
        "/api/engine/plan",
        Some(&plan_body("{model:UP.GGUF}", "")),
    );
    both(
        "plan, nothing chosen",
        "POST",
        "/api/engine/plan",
        Some("{}"),
    );
    both(
        "plan, unknown model",
        "POST",
        "/api/engine/plan",
        Some(&plan_body("ffffffffffff", "")),
    );
    both(
        "plan, incomplete model",
        "POST",
        "/api/engine/plan",
        Some(&plan_body(
            "{model:Qwen3.8-Flash-Next-FLE-IQ2_M-00001-of-00002.gguf}",
            "",
        )),
    );
    both(
        "plan, a profile",
        "POST",
        "/api/engine/plan",
        Some(&plan_body(
            "{model:UP.GGUF}",
            ", \"profile\": \"fle-reference\"",
        )),
    );
    both(
        "plan, unknown profile",
        "POST",
        "/api/engine/plan",
        Some(&plan_body("{model:UP.GGUF}", ", \"profile\": \"nope\"")),
    );
    both("plan bad body", "POST", "/api/engine/plan", Some("nope"));
    both(
        "preview, a catalog model without an engine",
        "POST",
        "/api/engine/preview",
        Some(&plan_body(iq3, "")),
    );
    both(
        "remove import",
        "POST",
        "/api/models/{model:outside.gguf}/remove",
        Some("{}"),
    );
    both(
        "remove import with delete is refused",
        "POST",
        "/api/models/{model:outside-two-00001-of-00002.gguf}/remove",
        Some(r#"{"delete": true}"#),
    );
    both(
        "remove unknown",
        "POST",
        "/api/models/ffffffffffff/remove",
        Some("{}"),
    );
    both(
        "remove bad id",
        "POST",
        "/api/models/xyz/remove",
        Some("{}"),
    );
    both(
        "remove bad body",
        "POST",
        "/api/models/{model:UP.GGUF}/remove",
        Some("nope"),
    );
    both(
        "remove folder model keeps files",
        "POST",
        "/api/models/{model:UP.GGUF}/remove",
        Some("{}"),
    );
    both(
        "remove folder model with delete",
        "POST",
        "/api/models/{model:moe.gguf}/remove",
        Some(r#"{"delete": true}"#),
    );
    both(
        "remove split with delete",
        "POST",
        "/api/models/{model:big-00001-of-00002.gguf}/remove",
        Some(r#"{"delete": 1}"#),
    );
    both("models at end", "GET", "/api/models", None);
    both("catalog at end", "GET", "/api/catalog", None);
    // the models folder can be moved in the settings
    let other = |h: &Path| format!(r#"{{"models_dir": "{}/ext"}}"#, h.display());
    for (addr, tok, h) in [(rs, &tr, &hr), (py.addr, &tp, &hp)] {
        let r = http(
            addr,
            "POST",
            "/api/settings",
            &[("X-FLE-Token", tok)],
            Some(other(h).as_bytes()),
        );
        assert_eq!(r.status, 200);
    }
    both("models in another folder", "GET", "/api/models", None);
    both("catalog in another folder", "GET", "/api/catalog", None);
    // library.json on disk: the same imports (the time they were added aside)
    let imports = |h: &Path| {
        let mut v: Value =
            serde_json::from_str(&std::fs::read_to_string(h.join("library.json")).unwrap())
                .unwrap();
        for i in v["imports"].as_array_mut().unwrap() {
            i["added"] = json!(0);
        }
        serde_json::from_str::<Value>(&v.to_string().replace("-rs", "-X").replace("-py", "-X"))
            .unwrap()
    };
    assert_eq!(imports(&hr), imports(&hp), "library.json differs");
    for f in [
        "moe.gguf",
        "big-00001-of-00002.gguf",
        "big-00002-of-00002.gguf",
    ] {
        let find = |h: &Path| walkdir_has(&h.join("models"), f);
        assert_eq!(find(&hr), find(&hp), "{f} on disk");
    }
    eprintln!(
        "compared with the Python Studio: {n} requests identical (models, catalog, downloads)"
    );
}

fn walkdir_has(dir: &Path, name: &str) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                e.file_name() == name || (e.path().is_dir() && walkdir_has(&e.path(), name))
            })
        })
        .unwrap_or(false)
}

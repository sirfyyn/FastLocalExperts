//! sidecars, bench and serve against the Python helper (or, for the supervisor, against the behaviour its tests describe).
#![cfg(unix)]
mod common;
use common::*;
use fle_core::gguf::{self, py_json};
use fle_core::launch::{Plan, *};
use fle_core::serve::{Process, State, Supervisor};
use fle_core::{bench, config, sidecars as sc, FleError};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

fn tmp(n: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fle-tools-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn tools_dir() -> Option<PathBuf> {
    let t = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tools");
    t.join("fle/bench.py").exists().then_some(t)
}
fn python(driver: &str, input: &Value, dir: &Path, envs: &[(&str, &str)]) -> Option<Value> {
    let t = tools_dir()?;
    std::fs::write(dir.join("driver.py"), driver).unwrap();
    std::fs::write(dir.join("in.json"), serde_json::to_string(input).unwrap()).unwrap();
    let mut c = std::process::Command::new("python3");
    c.arg(dir.join("driver.py"))
        .arg(&t)
        .arg(dir.join("in.json"));
    for (k, v) in envs {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    assert!(
        o.status.success(),
        "python driver failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    Some(serde_json::from_slice(&o.stdout).unwrap())
}

// ---------------------------------------------------------------- sidecars

const SIDECAR_DRIVER: &str = r#"
import json, sys, hashlib, os
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fle import gguf, sidecars
from fle.util import FleError
out = []
for sc in json.load(open(sys.argv[2])):
    model = gguf.read_model(sc["model"])
    if sc["op"] == "items":
        src = sidecars.find_source(model)
        if src is None:
            out.append({"source": None}); continue
        items = sidecars.plan_items(model, src, force=sc["force"])
        out.append({"source": src.family, "items": [i.as_dict() for i in items]})
    else:
        import re
        files = tuple(sidecars.SidecarFile(f["kind"], "u/r", f["path"], f["size"], f["sha256"]) for f in sc["files"])
        src = sidecars.Source("T", re.compile("^T"), files)
        items = sidecars.plan_items(model, src, force=sc["force"])
        def fetcher(url, dest, headers, sc=sc):
            data = sc["bodies"].get(url.rsplit("/", 1)[-1], "").encode()
            dest.write_bytes(data)
        says = []
        try:
            written = sidecars.fetch(items, force=sc["force"], fetcher=fetcher, say=says.append)
            out.append({"written": [str(i.dest) for i in written], "say": [s.replace(str(Path(sc["model"]).parent), "D") for s in says]})
        except FleError as e:
            out.append({"error": str(e).replace(str(Path(sc["model"]).parent), "D"), "hint": e.hint})
        leftovers = sorted(p.name for p in Path(sc["model"]).parent.iterdir() if ".part-" in p.name)
        out[-1]["leftovers"] = leftovers
print(json.dumps(out))
"#;

fn sha(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

#[test]
fn sidecars_match_the_python_helper() {
    let d = tmp("sc");
    let known = d.join("known");
    std::fs::create_dir_all(&known).unwrap();
    let kmodel = known.join("Qwen3.8-Flash-Next-FLE-IQ3_M.gguf");
    write(&kmodel, &kv("qwen3moe", "FLE", true), &tensors(true));
    let kpresent = d.join("kp");
    std::fs::create_dir_all(&kpresent).unwrap();
    let kpm = kpresent.join("qwen3.8-flash-next-fle-x.gguf");
    write(&kpm, &kv("qwen3moe", "FLE", true), &tensors(true));
    std::fs::write(kpresent.join("mtp-head.gguf"), b"x").unwrap();
    std::fs::write(kpresent.join("fle-mtp-vocab-32k.txt"), b"x").unwrap();
    let other = d.join("other.gguf");
    write(&other, &kv("qwen3", "Other", false), &tensors(false));

    let mut input = vec![];
    for (m, f) in [
        (&kmodel, false),
        (&kmodel, true),
        (&kpm, false),
        (&kpm, true),
        (&other, false),
    ] {
        input.push(json!({"op": "items", "model": m.to_string_lossy(), "force": f}));
    }
    // small custom sources for the download logic
    let files = json!([{"kind": "mtp_head", "path": "x/head.gguf", "size": 5, "sha256": sha("HEAD!")}, {"kind": "mtp_vocab", "path": "x/vocab.txt", "size": 4, "sha256": sha("voc\n")}]);
    let dl = |name: &str, bodies: Value, force: bool, pre: &[(&str, &str)], side: &str| -> Value {
        let dir = d.join(format!("{name}-{side}"));
        std::fs::create_dir_all(&dir).unwrap();
        let m = dir.join("T-model.gguf");
        write(&m, &kv("qwen3moe", "T", true), &tensors(true));
        for (n, c) in pre {
            std::fs::write(dir.join(n), c).unwrap();
        }
        json!({"op": "fetch", "model": m.to_string_lossy(), "force": force, "files": files, "bodies": bodies})
    };
    let good = json!({"head.gguf": "HEAD!", "vocab.txt": "voc\n"});
    type Case<'a> = (&'a str, Value, bool, Vec<(&'a str, &'a str)>);
    let cases: Vec<Case> = vec![
        ("ok", good.clone(), false, vec![]),
        (
            "short",
            json!({"head.gguf": "HEAD", "vocab.txt": "voc\n"}),
            false,
            vec![],
        ),
        (
            "badhash",
            json!({"head.gguf": "HEAD?", "vocab.txt": "voc\n"}),
            false,
            vec![],
        ),
        (
            "present",
            good.clone(),
            false,
            vec![("mtp-head.gguf", "x"), ("fle-mtp-vocab.txt", "x")],
        ),
        (
            "force",
            good.clone(),
            true,
            vec![("head.gguf", "OLD!!"), ("vocab.txt", "old\n")],
        ),
        ("half", good.clone(), false, vec![("head.gguf", "OLD!!")]),
    ];
    // Python and Rust each work on their own copy of the folder, because a download changes it
    let mut rs_input = input.clone();
    for (name, bodies, force, pre) in &cases {
        input.push(dl(name, bodies.clone(), *force, pre, "py"));
        rs_input.push(dl(name, bodies.clone(), *force, pre, "rs"));
    }

    let Some(py) = python(SIDECAR_DRIVER, &Value::Array(input.clone()), &d, &[]) else {
        eprintln!("python helper not found: skipped");
        return;
    };
    let fake_sources = |files: &Value| sc::Source {
        family: "T".into(),
        prefix: "T".into(),
        files: files
            .as_array()
            .unwrap()
            .iter()
            .map(|f| sc::SidecarFile {
                kind: if f["kind"] == "mtp_head" {
                    "mtp_head"
                } else {
                    "mtp_vocab"
                },
                repo: "u/r".into(),
                path: f["path"].as_str().unwrap().into(),
                size: f["size"].as_u64().unwrap(),
                sha256: f["sha256"].as_str().unwrap().into(),
            })
            .collect(),
    };
    for (sc_in, want) in rs_input.iter().zip(py.as_array().unwrap()) {
        let model = gguf::read_model(Path::new(sc_in["model"].as_str().unwrap())).unwrap();
        let force = sc_in["force"].as_bool().unwrap();
        if sc_in["op"] == "items" {
            match sc::find_source(&model, &sc::known_sources()) {
                None => assert!(want["source"].is_null(), "{sc_in}"),
                Some(s) => {
                    let items: Vec<Value> = sc::plan_items(&model, &s, force)
                        .iter()
                        .map(|i| i.as_json())
                        .collect();
                    assert_eq!(want["source"], s.family);
                    assert_eq!(py_json(&json!(items)), py_json(&want["items"]), "{sc_in}");
                }
            }
            continue;
        }
        let src = fake_sources(&sc_in["files"]);
        let items = sc::plan_items(&model, &src, force);
        let bodies = sc_in["bodies"].clone();
        let fetcher = move |url: &str, dest: &Path, _t: Option<&str>| -> anyhow::Result<()> {
            std::fs::write(
                dest,
                bodies[url.rsplit('/').next().unwrap()]
                    .as_str()
                    .unwrap_or("")
                    .as_bytes(),
            )?;
            Ok(())
        };
        let said = RefCell::new(vec![]);
        let dir = Path::new(sc_in["model"].as_str().unwrap())
            .parent()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let r = sc::fetch(
            &items,
            force,
            &fetcher,
            &|t| said.borrow_mut().push(t.replace(&dir, "D")),
            Some(u64::MAX / 2),
        );
        let leftovers: Vec<String> = {
            let mut v: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
                .filter(|n| n.contains(".part-"))
                .collect();
            v.sort();
            v
        };
        assert_eq!(json!(leftovers), want["leftovers"], "{sc_in}");
        match (r, want.get("error")) {
            (Ok(w), None) => {
                assert_eq!(
                    json!(w
                        .iter()
                        .map(|i| i.dest.to_string_lossy().replace("-rs/", "-py/"))
                        .collect::<Vec<_>>()),
                    want["written"],
                    "{sc_in}"
                );
                assert_eq!(json!(said.borrow().clone()), want["say"], "{sc_in}");
            }
            (Err(e), Some(m)) => {
                let fe = e.downcast_ref::<FleError>().unwrap();
                assert_eq!(json!(fe.message.replace(&dir, "D")), *m, "{sc_in}");
                assert_eq!(json!(fe.hint), want["hint"], "{sc_in}");
            }
            (a, b) => panic!(
                "outcome differs: rust {:?} python {b:?} for {sc_in}",
                a.map(|x| x.len())
            ),
        }
    }
    eprintln!(
        "compared with the Python helper: {} sidecar scenarios identical",
        input.len()
    );
}

#[test]
fn sidecar_download_is_refused_without_disk_space() {
    let d = tmp("space");
    let m = d.join("T-m.gguf");
    write(&m, &kv("qwen3moe", "T", true), &tensors(true));
    let model = gguf::read_model(&m).unwrap();
    let s = sc::Source {
        family: "T".into(),
        prefix: "T".into(),
        files: vec![sc::SidecarFile {
            kind: "mtp_head",
            repo: "u/r".into(),
            path: "a/h.gguf".into(),
            size: 100 << 20,
            sha256: "0".repeat(64),
        }],
    };
    let items = sc::plan_items(&model, &s, false);
    let e = sc::fetch(
        &items,
        false,
        &|_, _, _| panic!("must not download"),
        &|_| {},
        Some(10 << 20),
    )
    .unwrap_err();
    assert!(
        e.to_string().starts_with("Not enough disk space in "),
        "{e}"
    );
}

// ---------------------------------------------------------------- bench

fn fake_server(chat: Value, models: Value) -> (String, std::thread::JoinHandle<()>, Rc<()>) {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", l.local_addr().unwrap().port());
    let h = std::thread::spawn(move || {
        l.set_nonblocking(false).unwrap();
        for stream in l.incoming() {
            let mut st = stream.unwrap();
            let mut buf = vec![0u8; 65536];
            let n = st.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            if req.starts_with("GET /quit") {
                let _ = st.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                break;
            }
            let (code, body) = if req.starts_with("GET /v1/models") {
                (200, models.to_string())
            } else if req.starts_with("POST /v1/chat/completions") {
                (
                    if req.contains("FAILME") { 500 } else { 200 },
                    if req.contains("FAILME") {
                        "boom".to_string()
                    } else {
                        chat.to_string()
                    },
                )
            } else {
                (404, "{}".into())
            };
            let _ = st.write_all(format!("HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    (url, h, Rc::new(()))
}

const BENCH_DRIVER: &str = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
from fle import bench
from fle.util import FleError
out = []
for sc in json.load(open(sys.argv[2])):
    try:
        if sc["op"] == "run":
            r = bench.run(sc["url"], model=sc["model"], prompt=sc["prompt"], max_tokens=sc["max_tokens"], runs=sc["runs"], warmup=sc["warmup"])
            for s in r["runs"]:
                s["latency_ms"] = 0
            r["median_latency_ms"] = 0
            out.append({"result": r})
        elif sc["op"] == "compare":
            out.append({"result": bench.compare_baseline(sc["cur"], sc["base"], sc["max"])})
        elif sc["op"] == "kind":
            out.append({"result": bench.url_kind(sc["url"])})
        elif sc["op"] == "load":
            out.append({"result": bench.load_baseline(sc["path"])})
        elif sc["op"] == "text":
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bench.print_result(sc["res"])
                if sc.get("cmp"):
                    bench.print_comparison(sc["cmp"])
            out.append({"result": buf.getvalue()})
    except FleError as e:
        out.append({"error": str(e), "hint": e.hint})
print(json.dumps(out))
"#;

fn res(model: &str, url: &str, med: f64, trunc: i64) -> Value {
    json!({"url": url, "model": model, "prompt": "p", "max_tokens": 8, "median_tok_s": med, "median_latency_ms": 12.5, "min_tok_s": med - 1.0, "max_tok_s": med + 1.0, "truncated": trunc,
        "runs": [{"warmup": false, "latency_ms": 1234.5678, "completion_tokens": 12345, "prompt_tokens": 5, "tok_s": med, "predicted_ms": 1.0, "prompt_tok_s": 2.0, "finish_reason": "length", "accept": 0.5}, {"warmup": false, "latency_ms": 99.0, "completion_tokens": 7, "prompt_tokens": 5, "tok_s": 0.5, "predicted_ms": 1.0, "prompt_tok_s": 2.0, "finish_reason": "", "accept": null}]})
}

#[test]
fn bench_matches_the_python_helper() {
    let d = tmp("bench");
    let chat = json!({"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 9, "prompt_tokens": 4},
        "timings": {"predicted_per_second": 33.25, "predicted_ms": 270.0, "prompt_per_second": 120.5, "draft_n": 7, "draft_n_accepted": 3, "predicted_n": 9}});
    let (url, h, _) = fake_server(chat.clone(), json!({"data": [{"id": "my-model"}]}));
    let mut input = vec![
        json!({"op": "run", "url": url, "model": "", "prompt": "hi", "max_tokens": 16, "runs": 2, "warmup": 1}),
        json!({"op": "run", "url": url.trim_start_matches("http://"), "model": "named", "prompt": "hi", "max_tokens": 0, "runs": 0, "warmup": -3}),
    ];
    for u in [
        "127.0.0.1:1/x",
        "http://localhost:80",
        "https://example.com/v1",
        "[::1]:8080",
        "0.0.0.0",
        "HTTP://LOCALHOST:1",
        "http://user@127.0.0.1:9",
        "10.0.0.5:8080",
    ] {
        input.push(json!({"op": "kind", "url": u}));
    }
    let base = |model: &str, url: &str, med: f64, trunc: i64| res(model, url, med, trunc);
    for (cur, bas, mx) in [
        (
            base("m", "http://127.0.0.1:1", 100.0, 0),
            base("m", "http://localhost:2", 95.0, 0),
            10.0,
        ),
        (
            base("m", "http://127.0.0.1:1", 80.0, 0),
            base("m", "http://localhost:2", 100.0, 0),
            10.0,
        ),
        (
            base("m", "http://127.0.0.1:1", 80.0, 0),
            base("m", "http://localhost:2", 100.0, 0),
            -25.0,
        ),
        (
            base("m", "http://127.0.0.1:1", 100.0, 1),
            base("m", "http://127.0.0.1:1", 100.0, 0),
            10.0,
        ),
        (
            base("m", "http://127.0.0.1:1", 100.0, 0),
            base("m", "http://127.0.0.1:1", 100.0, 2),
            10.0,
        ),
        (
            base("a", "http://127.0.0.1:1", 100.0, 0),
            base("b'x", "http://127.0.0.1:1", 100.0, 0),
            10.0,
        ),
        (
            base("m", "http://127.0.0.1:1", 100.0, 0),
            base("m", "https://example.com", 100.0, 0),
            10.0,
        ),
        (
            base("m", "http://127.0.0.1:1", 0.0, 0),
            base("m", "http://127.0.0.1:1", 100.0, 0),
            10.0,
        ),
        (
            base("m", "http://127.0.0.1:1", 120.0, 0),
            base("m", "http://127.0.0.1:1", 100.0, 0),
            2.5,
        ),
    ] {
        input.push(json!({"op": "compare", "cur": cur, "base": bas, "max": mx}));
        let cmp = bench::compare_baseline(&cur, &bas, mx);
        input.push(json!({"op": "text", "res": cur, "cmp": cmp}));
    }
    let bad = d.join("bad.json");
    std::fs::write(&bad, "{not json").unwrap();
    let nores = d.join("nores.json");
    std::fs::write(&nores, "{\"x\": 1}").unwrap();
    let prim = d.join("prim.json");
    std::fs::write(
        &prim,
        serde_json::to_string(
            &json!({"primary": res("m", "http://127.0.0.1:1", 50.0, 0), "compare": {}}),
        )
        .unwrap(),
    )
    .unwrap();
    let plain = d.join("plain.json");
    std::fs::write(
        &plain,
        serde_json::to_string(&res("m", "http://127.0.0.1:1", 60.0, 0)).unwrap(),
    )
    .unwrap();
    for p in [&bad, &nores, &prim, &plain] {
        input.push(json!({"op": "load", "path": p.to_string_lossy()}));
    }
    let py = python(BENCH_DRIVER, &Value::Array(input.clone()), &d, &[]);
    let Some(py) = py else {
        eprintln!("python helper not found: skipped");
        return;
    };
    let mut n = 0;
    for (inp, want) in input.iter().zip(py.as_array().unwrap()) {
        let got: Result<Value, FleError> = match inp["op"].as_str().unwrap() {
            "run" => {
                let mut p = bench::Params::new(inp["url"].as_str().unwrap());
                p.model = inp["model"].as_str().unwrap().into();
                p.prompt = inp["prompt"].as_str().unwrap().into();
                p.max_tokens = inp["max_tokens"].as_i64().unwrap();
                p.runs = inp["runs"].as_i64().unwrap();
                p.warmup = inp["warmup"].as_i64().unwrap();
                bench::run(&p)
                    .map(|mut r| {
                        for s in r["runs"].as_array_mut().unwrap() {
                            s["latency_ms"] = json!(0);
                        }
                        r["median_latency_ms"] = json!(0);
                        r
                    })
                    .map_err(|e| e.downcast::<FleError>().unwrap())
            }
            "compare" => Ok(bench::compare_baseline(
                &inp["cur"],
                &inp["base"],
                inp["max"].as_f64().unwrap(),
            )),
            "kind" => Ok(json!(bench::url_kind(inp["url"].as_str().unwrap()))),
            "load" => bench::load_baseline(inp["path"].as_str().unwrap())
                .map_err(|e| e.downcast::<FleError>().unwrap()),
            "text" => {
                let mut t = bench::result_text(&inp["res"]);
                if !inp["cmp"].is_null() {
                    t += &bench::comparison_text(&inp["cmp"]);
                }
                Ok(json!(t))
            }
            _ => unreachable!(),
        };
        match (got, want.get("error")) {
            (Ok(g), None) => assert_eq!(
                py_json(&g),
                py_json(&want["result"]),
                "{}",
                inp.to_string().chars().take(200).collect::<String>()
            ),
            (Err(e), Some(m)) => {
                // the OS wording of an I/O or JSON error is not ours to copy; the explanation and the hint are
                let first = |s: &str| s.split(':').next().unwrap_or("").to_string();
                assert_eq!(first(&e.message), first(m.as_str().unwrap()), "{inp}");
                assert_eq!(e.hint, want["hint"].as_str().unwrap(), "{inp}");
            }
            (a, b) => panic!("outcome differs: rust {a:?} python {b:?} for {inp}"),
        }
        n += 1;
    }
    let _ = reqwest::blocking::get(format!("{url}/quit"));
    h.join().unwrap();
    eprintln!("compared with the Python helper: {n} bench scenarios identical");
}

#[test]
fn bench_reports_http_errors_and_empty_model_lists() {
    let (url, h, _) = fake_server(json!({}), json!({"data": []}));
    let e = bench::run(&bench::Params::new(&url))
        .unwrap_err()
        .downcast::<FleError>()
        .unwrap();
    assert!(
        e.message.contains("named no model") && e.hint == "Pass --model NAME.",
        "{}",
        e.message
    );
    let mut p = bench::Params::new(&url);
    p.model = "m".into();
    p.prompt = "FAILME".into();
    let e = bench::run(&p).unwrap_err().downcast::<FleError>().unwrap();
    assert!(
        e.message.contains("returned HTTP 500: ") && e.hint == "Check the server log.",
        "{}",
        e.message
    );
    let e = bench::run(&bench::Params::new("127.0.0.1:1"))
        .unwrap_err()
        .downcast::<FleError>()
        .unwrap();
    assert!(
        e.message
            .starts_with("Cannot reach http://127.0.0.1:1/v1/models")
            && e.hint == "Start the server first, or pass --url."
    );
    let _ = reqwest::blocking::get(format!("{url}/quit"));
    h.join().unwrap();
}

// ---------------------------------------------------------------- serve

struct Fake {
    polls: Vec<Option<i32>>,
    log: Rc<RefCell<Vec<String>>>,
    name: &'static str,
    wait_code: i32,
}
impl Process for Fake {
    fn poll(&mut self) -> Option<i32> {
        if self.polls.len() > 1 {
            self.polls.remove(0)
        } else {
            self.polls
                .first()
                .copied()
                .flatten()
                .map(Some)
                .unwrap_or(None)
        }
    }
    fn wait(&mut self, _t: Option<std::time::Duration>) -> Option<i32> {
        self.log.borrow_mut().push(format!("wait {}", self.name));
        Some(self.wait_code)
    }
    fn terminate(&mut self) {}
    fn kill(&mut self) {}
}

fn a_plan(binary: &str) -> Plan {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let d = tmp(&format!(
        "plan{}",
        N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    let m = d.join("m.gguf");
    write(&m, &kv("qwen3", "T", false), &tensors(false));
    let e = engine(&d, "up", false, true);
    let mut st = config::defaults();
    st.insert("profile".into(), json!("off"));
    let h = Host {
        cpu_physical: 2,
        cpu_logical: 2,
        mem_total: 1 << 34,
        mem_available: 1 << 34,
        gpus: vec![],
        windows: false,
        linux: true,
        port_free: Box::new(|_, _| true),
    };
    let mut p = build(
        &st,
        &gguf::read_model(&m).unwrap(),
        &Options {
            engine_path: e,
            ..Default::default()
        },
        &h,
    )
    .unwrap();
    p.binary = binary.into();
    p
}

fn sup<'a>(
    spawned: Rc<RefCell<Vec<String>>>,
    health: Vec<bool>,
    procs: Vec<Fake>,
    said: Rc<RefCell<Vec<String>>>,
    timeout: f64,
    progress: bool,
) -> Supervisor<'a> {
    let now = Rc::new(RefCell::new(0.0f64));
    let (n1, n2) = (now.clone(), now.clone());
    let procs = Rc::new(RefCell::new(procs));
    let health = Rc::new(RefCell::new(health));
    Supervisor {
        spawn: Box::new(move |p: &Plan| {
            spawned.borrow_mut().push(p.binary.clone());
            let mut f = procs.borrow_mut().remove(0);
            f.log = spawned.clone();
            Ok(Box::new(f) as Box<dyn Process>)
        }),
        health: Box::new(move |_| {
            let mut h = health.borrow_mut();
            if h.len() > 1 {
                h.remove(0)
            } else {
                h[0]
            }
        }),
        ready_timeout: timeout,
        poll_interval: 1.0,
        sleep: Box::new(move |s| *n1.borrow_mut() += s),
        clock: Box::new(move || *n2.borrow()),
        say: Box::new(move |t| said.borrow_mut().push(t.to_string())),
        progress,
        progress_interval: 3.0,
        waited: 0.0,
    }
}
fn fake(polls: Vec<Option<i32>>, name: &'static str, wait: i32) -> Fake {
    Fake {
        polls,
        log: Rc::new(RefCell::new(vec![])),
        name,
        wait_code: wait,
    }
}

#[test]
fn supervisor_waits_for_ready_then_follows_the_server() {
    let (sp, said) = (Rc::new(RefCell::new(vec![])), Rc::new(RefCell::new(vec![])));
    let mut s = sup(
        sp.clone(),
        vec![false, false, false, false, false, true],
        vec![fake(vec![None], "A", 7)],
        said.clone(),
        100.0,
        true,
    );
    let code = s
        .run(&a_plan("A"), &mut |_| panic!("no fallback needed"))
        .unwrap();
    assert_eq!(code, 7);
    assert_eq!(*sp.borrow(), ["A", "wait A"]);
    assert_eq!(said.borrow().last().unwrap(), "fle: ready after 5 s");
    assert!(
        said.borrow()
            .iter()
            .any(|l| l.starts_with("fle: still loading ... 3 s")),
        "{:?}",
        said.borrow()
    );
}

#[test]
fn supervisor_ready_message_without_a_terminal_and_clean_exit() {
    let (sp, said) = (Rc::new(RefCell::new(vec![])), Rc::new(RefCell::new(vec![])));
    let mut s = sup(
        sp.clone(),
        vec![true],
        vec![fake(vec![None], "A", 0)],
        said.clone(),
        100.0,
        false,
    );
    assert_eq!(s.run(&a_plan("A"), &mut |_| panic!()).unwrap(), 0);
    assert_eq!(*said.borrow(), ["fle: the server is ready."]);
    let (sp2, said2) = (Rc::new(RefCell::new(vec![])), Rc::new(RefCell::new(vec![])));
    let mut s = sup(
        sp2.clone(),
        vec![false],
        vec![fake(vec![Some(0)], "A", 5)],
        said2,
        100.0,
        false,
    );
    assert_eq!(s.run(&a_plan("A"), &mut |_| panic!()).unwrap(), 0);
    assert_eq!(
        *sp2.borrow(),
        ["A"],
        "a server that ended cleanly is not waited for again"
    );
}

#[test]
fn supervisor_falls_back_once_when_the_server_dies_while_loading() {
    let (sp, said) = (Rc::new(RefCell::new(vec![])), Rc::new(RefCell::new(vec![])));
    let mut s = sup(
        sp.clone(),
        vec![false],
        vec![
            fake(vec![None, Some(139)], "A", 0),
            fake(vec![None], "B", 3),
        ],
        said.clone(),
        100.0,
        false,
    );
    let mut reasons = vec![];
    let code = s
        .run(&a_plan("A"), &mut |r| {
            reasons.push(r.to_string());
            Ok(a_plan("B"))
        })
        .unwrap();
    assert_eq!(code, 3);
    assert_eq!(
        reasons,
        ["the server exited with code 139 before it was ready"]
    );
    assert_eq!(*sp.borrow(), ["A", "B", "wait B"]);
    assert_eq!(
        *said.borrow(),
        [
            "fle: the server exited with code 139 before it was ready.",
            "fle: restarting once with upstream llama.cpp (B, profile off, no FLE features)."
        ]
    );
}

#[test]
fn supervisor_explains_when_no_fallback_is_possible_and_leaves_a_slow_server_alone() {
    let (sp, said) = (Rc::new(RefCell::new(vec![])), Rc::new(RefCell::new(vec![])));
    let mut s = sup(
        sp,
        vec![false],
        vec![fake(vec![Some(1)], "A", 0)],
        said,
        100.0,
        false,
    );
    let e = s
        .run(&a_plan("A"), &mut |_| {
            Err(FleError::new("nothing installed", "install one").into())
        })
        .unwrap_err()
        .downcast::<FleError>()
        .unwrap();
    assert_eq!(e.message, "The server failed to start (the server exited with code 1 before it was ready), and no fallback is possible: nothing installed");
    assert_eq!(e.hint, "install one");
    let (sp, said) = (Rc::new(RefCell::new(vec![])), Rc::new(RefCell::new(vec![])));
    let mut s = sup(
        sp.clone(),
        vec![false],
        vec![fake(vec![None], "A", 4)],
        said.clone(),
        5.0,
        false,
    );
    let (st, _) = s.wait_ready(&mut fake(vec![None], "X", 0), &a_plan("A"));
    assert_eq!(st, State::Timeout);
    assert_eq!(s.run(&a_plan("A"), &mut |_| panic!()).unwrap(), 4);
    assert_eq!(
        said.borrow()[0],
        "fle: still loading after the wait limit; leaving the server running without fallback."
    );
}

//! Engines against the Python Studio: which builds a machine is offered, which release assets belong to them, and a whole
//! install from a release server of our own (download, unpack, register, check, remove). Nothing here touches the network.
#![cfg(unix)]
mod harness;
use fle_api::engines::{asset_patterns, cuda_ok, cudart_pattern, driver_major, variants};
use harness::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::time::{Duration, Instant};

const DRIVER: &str = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
from fle_studio import engines
sc = json.load(open(sys.argv[2]))
out = {"variants": [], "patterns": [], "cuda": [], "driver": []}
for s in sc["systems"]:
    for os_, arch in sc["osarch"]:
        engines._os_arch = lambda os_=os_, arch=arch: (os_, arch)
        out["variants"].append(engines.variants(s))
for v in sc["variant_ids"]:
    for os_, arch in sc["osarch"]:
        try:
            pats = engines._asset_patterns(v, os_, arch)
        except ValueError as e:
            out["patterns"].append({"error": str(e)}); continue
        import re
        row = {}
        for name in sc["names"]:
            m = re.match(pats[0], name)
            row[name] = None if not m else (m.groupdict().get("cuda") or "")
        for cu in ("12.4", "13.0", "12.8"):
            pat = engines._cudart_pattern(os_, arch, cu)
            row["cudart:" + cu] = [n for n in sc["names"] if re.match(pat, n)]
        out["patterns"].append(row)
for v in sc["cuda_versions"]:
    for d in sc["drivers"]:
        out["cuda"].append(engines._cuda_ok(v, d))
for s in sc["systems"]:
    out["driver"].append(engines._driver_major(s))
print(json.dumps(out))
"#;

fn names() -> Vec<String> {
    let mut v: Vec<String> = vec![];
    for plat in ["ubuntu", "win", "macos"] {
        for arch in ["x64", "arm64"] {
            for ext in [".tar.gz", ".zip"] {
                v.push(format!("llama-b6000-bin-{plat}-{arch}{ext}"));
                v.push(format!("llama-b6000-bin-{plat}-cpu-{arch}{ext}"));
                v.push(format!("llama-b6000-bin-{plat}-vulkan-{arch}{ext}"));
                for cu in ["12.4", "12.8", "13.0", "12.6", "11.8"] {
                    v.push(format!("llama-b6000-bin-{plat}-cuda-{cu}-{arch}{ext}"));
                    v.push(format!("fle-v1.2-bin-{plat}-cuda-{cu}-{arch}{ext}"));
                    v.push(format!(
                        "cudart-llama-b6000-bin-{plat}-cuda-{cu}-{arch}{ext}"
                    ));
                    v.push(format!("cudart-llama-bin-{plat}-cuda-{cu}-{arch}{ext}"));
                }
            }
        }
    }
    v.extend(
        [
            "llama-bX-bin-macos-arm64.tar.gz",
            "llama-b-bin-macos-arm64.tar.gz",
            "fle--bin-ubuntu-cuda-12.4-x64.tar.gz",
            "fle-x-bin-ubuntu-cuda--x64.tar.gz",
            "llama-b6000-bin-macos-arm64.tar.gz.sha256",
            "source.zip",
        ]
        .map(String::from),
    );
    v
}

#[test]
fn offers_and_asset_patterns_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found: skipped");
        return;
    };
    let gpu = |v: &str, d: &str| json!({"vendor": v, "driver": d});
    let systems = vec![
        json!({"gpus": [], "cpu": {"avx2": true}, "other_gpus": []}),
        json!({"gpus": [gpu("nvidia", "570.12")], "cpu": {"avx2": true}, "other_gpus": []}),
        json!({"gpus": [gpu("nvidia", "550.1")], "cpu": {"avx2": false}, "other_gpus": []}),
        json!({"gpus": [gpu("nvidia", "")], "cpu": {"avx2": true}, "other_gpus": []}),
        json!({"gpus": [gpu("nvidia", "abc")], "cpu": {"avx2": true}, "other_gpus": [{"vendor": "amd"}]}),
        json!({"gpus": [], "cpu": {"avx2": false}, "other_gpus": [{"vendor": "amd"}, {"vendor": "intel"}]}),
        json!({"gpus": [{"vendor": "apple", "unified": true}], "cpu": {"avx2": null}, "other_gpus": []}),
        json!({}),
    ];
    let osarch = [
        ("linux", "x64"),
        ("linux", "arm64"),
        ("win", "x64"),
        ("macos", "arm64"),
        ("macos", "x64"),
    ];
    let variant_ids = [
        "upstream-metal",
        "upstream-macos-x64",
        "upstream-cpu",
        "upstream-vulkan",
        "upstream-cuda",
        "fle-cuda",
        "bogus",
    ];
    let cuda_versions = [
        "13", "13.0", "12.8", "12.9", "12.6", "12.4", "12.2", "12", "11.8", "7",
    ];
    let drivers = [0i64, 535, 550, 560, 570, 575, 580, 999];
    let n = names();
    let d = tmp("eng-pure");
    std::fs::write(d.join("driver.py"), DRIVER).unwrap();
    std::fs::write(d.join("in.json"), json!({"systems": systems, "osarch": osarch, "variant_ids": variant_ids, "names": n, "cuda_versions": cuda_versions, "drivers": drivers}).to_string()).unwrap();
    let out = std::process::Command::new(
        std::env::var("FLE_STUDIO_PYTHON").unwrap_or_else(|_| "python3".into()),
    )
    .arg(d.join("driver.py"))
    .arg(&src)
    .arg(d.join("in.json"))
    .output()
    .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py: Value = serde_json::from_slice(&out.stdout).unwrap();
    let mut k = 0;
    for s in &systems {
        for (os, arch) in osarch {
            assert_eq!(
                Value::Array(variants(s, os, arch)),
                py["variants"][k],
                "variants {s} {os}-{arch}"
            );
            k += 1;
        }
    }
    let mut k = 0;
    for v in variant_ids {
        for (os, arch) in osarch {
            let want = &py["patterns"][k];
            k += 1;
            match asset_patterns(v, os, arch) {
                Err(e) => assert_eq!(want["error"], json!(e), "{v}"),
                Ok(pats) => {
                    for name in &n {
                        let got = pats[0].matches(name).map(|c| c.unwrap_or_default());
                        assert_eq!(
                            got.map(Value::from).unwrap_or(Value::Null),
                            want[name],
                            "{v} {os}-{arch} {name}"
                        );
                    }
                    for cu in ["12.4", "13.0", "12.8"] {
                        let p = cudart_pattern(os, arch, cu);
                        let hits: Vec<&String> =
                            n.iter().filter(|x| p.matches(x).is_some()).collect();
                        assert_eq!(
                            json!(hits),
                            want[format!("cudart:{cu}")],
                            "cudart {os}-{arch} {cu}"
                        );
                    }
                }
            }
        }
    }
    let mut k = 0;
    for v in cuda_versions {
        for dr in drivers {
            assert_eq!(json!(cuda_ok(v, dr)), py["cuda"][k], "cuda_ok({v}, {dr})");
            k += 1;
        }
    }
    for (i, s) in systems.iter().enumerate() {
        assert_eq!(json!(driver_major(s)), py["driver"][i], "driver_major {s}");
    }
    eprintln!(
        "compared with the Python Studio: {} offers, {} name checks, {} CUDA verdicts identical",
        systems.len() * osarch.len(),
        n.len() * variant_ids.len() * osarch.len(),
        cuda_versions.len() * drivers.len()
    );
}

// ------------------------------------------------------------------ a whole install

fn sha(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

const SCRIPT: &str = "#!/bin/sh\ncase \"$1\" in\n  --version) echo 'version: 5555 (abc1234)' ;;\n  --help) printf -- '--log-verbosity N\\n--spec-type T\\n' ;;\nesac\n";
const BROKEN: &str = "#!/bin/sh\necho 'error: this build is broken' >&2\nexit 1\n";

fn tar_gz(files: &[(&str, &[u8], u32)]) -> Vec<u8> {
    let enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut t = tar::Builder::new(enc);
    for (name, data, mode) in files {
        // the name goes straight into the header: the tar library would refuse to write a path with `..` in it
        let mut h = tar::Header::new_old();
        h.as_old_mut().name[..name.len()].copy_from_slice(name.as_bytes());
        h.set_size(data.len() as u64);
        h.set_mode(*mode);
        h.set_entry_type(tar::EntryType::Regular);
        h.set_cksum();
        t.append(&h, *data).unwrap();
    }
    t.into_inner().unwrap().finish().unwrap()
}

/// The release a tag stands for: asset name -> bytes (none: the tag is unknown).
fn release(tag: &str) -> Option<Vec<(String, Vec<u8>)>> {
    let good = tar_gz(&[
        ("build/bin/llama-server", SCRIPT.as_bytes(), 0o644),
        ("build/bin/libggml.so", b"lib", 0o644),
        ("build/README", b"hi", 0o644),
        ("libcudart.so.12", b"cuda", 0o644),
        ("docs/libcublas.so.12", b"cublas", 0o644),
    ]);
    let name = "llama-b6000-bin-macos-arm64.tar.gz";
    let cpu = "llama-b6000-bin-ubuntu-x64.tar.gz";
    let both = |data: Vec<u8>| vec![(name.to_string(), data.clone()), (cpu.to_string(), data)];
    Some(match tag {
        "bOK" => both(good),
        "bNoServer" => both(tar_gz(&[("build/bin/other", b"x", 0o755)])),
        "bBadServer" => both(tar_gz(&[("llama-server", BROKEN.as_bytes(), 0o755)])),
        "bUnsafe" => both(tar_gz(&[
            ("../evil.txt", b"x", 0o644),
            ("llama-server", SCRIPT.as_bytes(), 0o755),
        ])),
        "bBroken" => both(b"this is not an archive".to_vec()),
        "bNoMatch" => vec![("something-else.zip".to_string(), b"x".to_vec())],
        _ => return None,
    })
}

fn release_server() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", l.local_addr().unwrap().port());
    let b2 = base.clone();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let b2 = b2.clone();
            std::thread::spawn(move || {
                let mut s = s.unwrap();
                let mut buf = vec![0u8; 8192];
                let n = s.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = req.split(' ').nth(1).unwrap_or("/").to_string();
                let send = |s: &mut std::net::TcpStream, code: &str, body: &[u8]| {
                    let _ = s.write_all(
                        format!(
                            "HTTP/1.1 {code}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .as_bytes(),
                    );
                    let _ = s.write_all(body);
                };
                if let Some(tag) = path.split("/releases/tags/").nth(1) {
                    match release(tag) {
                        Some(assets) => {
                            let a: Vec<Value> = assets.iter().map(|(n, d)| json!({"name": n, "browser_download_url": format!("{b2}/dl/{tag}/{n}"), "size": d.len(), "digest": format!("sha256:{}", sha(d))})).collect();
                            send(
                                &mut s,
                                "200 OK",
                                json!({"tag_name": tag, "assets": a}).to_string().as_bytes(),
                            );
                        }
                        None => send(&mut s, "404 Not Found", b"{}"),
                    }
                } else if let Some(rest) = path.strip_prefix("/dl/") {
                    let (tag, name) = rest.split_once('/').unwrap_or(("", ""));
                    match release(tag).and_then(|a| a.into_iter().find(|(n, _)| n == name)) {
                        Some((_, d)) => send(&mut s, "200 OK", &d),
                        None => send(&mut s, "404 Not Found", b""),
                    }
                } else {
                    send(&mut s, "404 Not Found", b"{}");
                }
            });
        }
    });
    base
}

fn tree(dir: &Path) -> Vec<(String, u32)> {
    use std::os::unix::fs::PermissionsExt;
    fn walk(d: &Path, root: &Path, out: &mut Vec<(String, u32)>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                out.push((format!("{rel}/"), 0));
                walk(&p, root, out);
            } else {
                out.push((
                    rel,
                    std::fs::metadata(&p).unwrap().permissions().mode() & 0o111,
                ));
            }
        }
    }
    let mut out = vec![];
    walk(dir, dir, &mut out);
    out.sort();
    out
}

#[test]
fn an_engine_install_matches_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found (set FLE_STUDIO_SRC): skipped");
        return;
    };
    let (hp, hr) = (tmp("eng-py"), tmp("eng-rs"));
    let (bp, br) = (release_server(), release_server());
    let py = start_python_with(
        &src,
        &hp,
        &format!("import fle_studio.engines as e\ne.GITHUB_API = '{bp}'"),
    );
    let rs: SocketAddr = start_rust_with(&src, &hr, &br);
    let (tr, tp) = (
        token_of(&http(rs, "GET", "/", &[], None)),
        token_of(&http(py.addr, "GET", "/", &[], None)),
    );
    let norm = |v: Value, base: &str, home: &Path| -> Value {
        let mut t = v
            .to_string()
            .replace(base, "BASE")
            .replace(&*home.to_string_lossy(), "HOME");
        t = t.replace("-rs", "-X").replace("-py", "-X");
        let mut v: Value = serde_json::from_str(&t).unwrap();
        fn scrub(v: &mut Value) {
            match v {
                Value::Object(o) => {
                    // the wording of an archive library's complaint is its own
                    if o.get("title")
                        .and_then(Value::as_str)
                        .is_some_and(|t| t.ends_with("bUnsafe") || t.ends_with("bBroken"))
                    {
                        o.insert("error".into(), json!("<from the archive library>"));
                    }
                    for (k, x) in o.iter_mut() {
                        if matches!(k.as_str(), "created" | "installed" | "speed") {
                            *x = json!(0);
                        } else if k == "finished" {
                            *x = json!(!x.is_null());
                        } else {
                            scrub(x);
                        }
                    }
                }
                Value::Array(a) => a.iter_mut().for_each(scrub),
                _ => {}
            }
        }
        scrub(&mut v);
        canon(v)
    };
    let call = |addr: SocketAddr, tok: &str, method: &str, path: &str, body: Option<&str>| {
        http(
            addr,
            method,
            path,
            &[("X-FLE-Token", tok)],
            body.map(str::as_bytes),
        )
    };
    let n = std::cell::Cell::new(0);
    let both = |label: &str, method: &str, path: &str, body: Option<&str>| -> (Resp, Resp) {
        let (a, b) = (
            call(rs, &tr, method, path, body),
            call(py.addr, &tp, method, path, body),
        );
        assert_eq!(
            a.status,
            b.status,
            "[{label}] status {} vs {}: {}",
            a.status,
            b.status,
            String::from_utf8_lossy(&a.body)
        );
        let (x, y): (Value, Value) = (
            serde_json::from_slice(&a.body).unwrap_or(Value::Null),
            serde_json::from_slice(&b.body).unwrap_or(Value::Null),
        );
        assert_eq!(norm(x, &br, &hr), norm(y, &bp, &hp), "[{label}] body");
        n.set(n.get() + 1);
        (a, b)
    };
    let finish = |label: &str, id_of: &dyn Fn(&Resp) -> String, a: &Resp, b: &Resp| {
        let (ia, ib) = (id_of(a), id_of(b));
        let t = Instant::now();
        let state = |addr: SocketAddr, id: &str| -> Value {
            let r = http(addr, "GET", "/api/downloads", &[], None);
            let v: Value = serde_json::from_slice(&r.body).unwrap();
            v["downloads"]
                .as_array()
                .unwrap()
                .iter()
                .find(|j| j["id"] == id)
                .cloned()
                .unwrap_or(Value::Null)
        };
        loop {
            let (x, y) = (state(rs, &ia), state(py.addr, &ib));
            let fin =
                |v: &Value| matches!(v["state"].as_str(), Some("done" | "failed" | "cancelled"));
            if fin(&x) && fin(&y) {
                return (x, y);
            }
            assert!(
                t.elapsed() < Duration::from_secs(60),
                "{label}: the install did not finish"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    };
    let job_id = |r: &Resp| {
        serde_json::from_slice::<Value>(&r.body).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string()
    };
    both("engines at the start", "GET", "/api/engines", None);
    both(
        "install nothing",
        "POST",
        "/api/engines/install",
        Some("{}"),
    );
    both(
        "install unknown",
        "POST",
        "/api/engines/install",
        Some(r#"{"variant": "nope"}"#),
    );
    both(
        "install bad body",
        "POST",
        "/api/engines/install",
        Some("nope"),
    );
    let offered: Value =
        serde_json::from_slice(&http(rs, "GET", "/api/engines", &[], None).body).unwrap();
    let variant = offered["variants"][0]["id"].as_str().unwrap().to_string();
    let set = |tag: &str| {
        for (addr, tok) in [(rs, &tr), (py.addr, &tp)] {
            let r = http(
                addr,
                "POST",
                "/api/settings",
                &[("X-FLE-Token", tok)],
                Some(
                    json!({"engine": {"upstream_tag": tag}})
                        .to_string()
                        .as_bytes(),
                ),
            );
            assert_eq!(r.status, 200);
        }
    };
    for tag in ["bGone", "bNoMatch"] {
        set(tag);
        both(
            &format!("install {tag}"),
            "POST",
            "/api/engines/install",
            Some(&json!({"variant": variant}).to_string()),
        );
    }
    let body = json!({"variant": variant}).to_string();
    // the failing archives: the job fails, nothing is registered
    for tag in ["bNoServer", "bBadServer", "bUnsafe", "bBroken"] {
        set(tag);
        let (a, b) = both(
            &format!("install {tag}"),
            "POST",
            "/api/engines/install",
            Some(&body),
        );
        let (x, y) = finish(tag, &job_id, &a, &b);
        assert_eq!(
            (&x["state"], &y["state"]),
            (&json!("failed"), &json!("failed")),
            "{tag}: {} / {}",
            x["error"],
            y["error"]
        );
        if tag != "bUnsafe" && tag != "bBroken" {
            // these two texts come from the archive library of each language; the others are our own words
            assert_eq!(x["error"], y["error"], "{tag}");
        }
        both(&format!("engines after {tag}"), "GET", "/api/engines", None);
    }
    assert!(!hr.join("engines/evil.txt").exists() && !hp.join("engines/evil.txt").exists());
    // the good one
    set("bOK");
    let (a, b) = both("install bOK", "POST", "/api/engines/install", Some(&body));
    let (x, y) = finish("bOK", &job_id, &a, &b);
    assert_eq!(
        (&x["state"], &y["state"]),
        (&json!("done"), &json!("done")),
        "{} / {}",
        x["error"],
        y["error"]
    );
    both("engines after the install", "GET", "/api/engines", None);
    both("downloads after the install", "GET", "/api/downloads", None);
    let id = format!("{variant}-bOK");
    let strip = |h: &Path| {
        tree(&h.join("engines").join(&id))
            .into_iter()
            .collect::<Vec<_>>()
    };
    assert_eq!(strip(&hr), strip(&hp), "the unpacked engine folder");
    assert!(
        strip(&hr)
            .iter()
            .any(|(n, m)| n == "build/bin/llama-server" && *m != 0),
        "llama-server is executable"
    );
    assert!(
        strip(&hr)
            .iter()
            .any(|(n, _)| n == "build/bin/libcudart.so.12")
            && strip(&hr)
                .iter()
                .any(|(n, _)| n == "build/bin/libcublas.so.12"),
        "CUDA libraries gather beside the server"
    );
    let leftovers = |h: &Path| tree(&h.join("engines/_downloads")).len();
    assert_eq!(
        leftovers(&hr),
        leftovers(&hp),
        "archives are removed after the install"
    );
    both("install again", "POST", "/api/engines/install", Some(&body));
    // the check of a binary
    let bin = |h: &Path| format!("{}/engines/{id}/build/bin/llama-server", h.display());
    for (what, path_of) in [
        ("installed", 0),
        ("missing", 1),
        ("folder", 2),
        ("empty", 3),
    ] {
        let p = |h: &Path| match path_of {
            0 => bin(h),
            1 => format!("{}/nope", h.display()),
            2 => format!("{}/engines", h.display()),
            _ => String::new(),
        };
        let (a, b) = (
            call(
                rs,
                &tr,
                "POST",
                "/api/engines/check",
                Some(&json!({"path": p(&hr)}).to_string()),
            ),
            call(
                py.addr,
                &tp,
                "POST",
                "/api/engines/check",
                Some(&json!({"path": p(&hp)}).to_string()),
            ),
        );
        assert_eq!(a.status, b.status, "check {what}");
        let (x, y): (Value, Value) = (
            serde_json::from_slice(&a.body).unwrap(),
            serde_json::from_slice(&b.body).unwrap(),
        );
        assert_eq!(norm(x, &br, &hr), norm(y, &bp, &hp), "check {what}");
        n.set(n.get() + 1);
    }
    // a binary that does not start
    for h in [&hr, &hp] {
        let f = h.join("broken-server");
        std::fs::write(&f, BROKEN).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let (a, b) = (
        call(
            rs,
            &tr,
            "POST",
            "/api/engines/check",
            Some(&json!({"path": hr.join("broken-server")}).to_string()),
        ),
        call(
            py.addr,
            &tp,
            "POST",
            "/api/engines/check",
            Some(&json!({"path": hp.join("broken-server")}).to_string()),
        ),
    );
    assert_eq!(
        norm(serde_json::from_slice(&a.body).unwrap(), &br, &hr),
        norm(serde_json::from_slice(&b.body).unwrap(), &bp, &hp),
        "check broken"
    );
    // removal
    both(
        "remove unknown",
        "POST",
        "/api/engines/nope/remove",
        Some("{}"),
    );
    both(
        "remove installed",
        "POST",
        &format!("/api/engines/{id}/remove"),
        Some("{}"),
    );
    both("engines at the end", "GET", "/api/engines", None);
    assert_eq!(
        hr.join("engines").join(&id).exists(),
        hp.join("engines").join(&id).exists(),
        "the folder is gone"
    );
    eprintln!("compared with the Python Studio: {} requests and the whole install (unpack, libraries, modes, check, remove) identical", n.get());
}

//! `doctor`, `report` and `sidecars` against the Python helper: same machine, model and settings, same result.
#![cfg(unix)]
mod common;
use common::*;
use fle_core::launch::{GpuInfo, Host};
use fle_core::{config, doctor, gguf, gguf::py_json, report};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

const DRIVER: &str = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
from fle import hardware, plan, gguf, config, doctor, report
out = []
for sc in json.load(open(sys.argv[2])):
    h = sc["host"]
    hardware.cpu = lambda h=h: {"name": "x", "physical": h["cpu_physical"], "logical": h["cpu_logical"]}
    hardware.memory = lambda h=h: {"total": h["mem_total"], "available": h["mem_available"]}
    hardware.gpus = lambda h=h: [dict(g, vendor="nvidia") for g in h["gpus"]]
    hardware.port_free = lambda host, port, h=h: h["port_free"]
    hardware.summary = lambda model_path=None, s=sc["system"]: s
    hardware.find_engine = lambda explicit="": ""
    plan.os.name = "posix"
    plan.sys.platform = "linux"
    r = doctor.run(sc["model"], sc["settings"])
    entry = {"report": r}
    if sc.get("redact"):
        entry["redacted"] = report.redact(sc["redact"])
    out.append(entry)
print(json.dumps(out, ensure_ascii=True, indent=2))
"#;

fn tmp(n: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fle-doctor-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[derive(Clone)]
struct Sc {
    name: String,
    model: String,
    settings: Map<String, Value>,
    gpus: Vec<(i64, u64, u64)>,
    mem: (u64, u64),
    port_free: bool,
    disk: u64,
}

fn host_of(sc: &Sc) -> Host {
    let free = sc.port_free;
    Host {
        cpu_physical: 8,
        cpu_logical: 16,
        mem_total: sc.mem.0 << 20,
        mem_available: sc.mem.1 << 20,
        gpus: sc
            .gpus
            .iter()
            .map(|(i, t, f)| GpuInfo {
                index: *i,
                name: format!("GPU{i}"),
                memory_total: t << 20,
                memory_used: (t - f) << 20,
                memory_free: f << 20,
            })
            .collect(),
        windows: false,
        linux: true,
        port_free: Box::new(move |_, _| free),
    }
}
fn system_of(sc: &Sc) -> Value {
    let gpus: Vec<Value> = sc.gpus.iter().map(|(i, t, f)| json!({"index": i, "name": format!("GPU{i}"), "memory_total": t << 20, "memory_used": (t - f) << 20, "memory_free": f << 20, "vendor": "nvidia"})).collect();
    json!({"os": "Test OS", "platform": "linux", "cpu": {"name": "x", "physical": 8, "logical": 16}, "memory": {"total": sc.mem.0 << 20, "available": sc.mem.1 << 20},
        "gpus": gpus, "disk_free": sc.disk})
}

#[test]
fn doctor_matches_the_python_helper() {
    let d = tmp("m");
    let moe = d.join("moe.gguf");
    write(&moe, &kv("qwen3moe", "Tiny MoE", true), &tensors(true));
    let dense = d.join("dense.gguf");
    write(&dense, &kv("qwen3", "Tiny Dense", false), &tensors(false));
    let with_head = d.join("h");
    std::fs::create_dir_all(&with_head).unwrap();
    write(
        &with_head.join("m.gguf"),
        &kv("qwen3moe", "Head Only", true),
        &tensors(true),
    );
    std::fs::write(with_head.join("mtp-head.gguf"), b"x").unwrap();
    let with_both = d.join("b");
    std::fs::create_dir_all(&with_both).unwrap();
    write(
        &with_both.join("m.gguf"),
        &kv("qwen3moe", "Both", true),
        &tensors(true),
    );
    std::fs::write(with_both.join("mtp-head.gguf"), b"x").unwrap();
    std::fs::write(with_both.join("fle-mtp-vocab-32k.txt"), b"x").unwrap();
    let garbage = d.join("garbage.gguf");
    std::fs::write(&garbage, b"this is not a gguf file at all").unwrap();
    let fle = engine(&d, "fle-server", true, true);
    let up = engine(&d, "up-server", false, true);
    let old = engine(&d, "old-server", true, false);

    let base = |name: &str, model: &str, eng: &str, profile: &str| -> Sc {
        let mut st = config::defaults();
        st.insert("profile".into(), json!(profile));
        st.insert("engine".into(), json!(eng));
        Sc {
            name: name.into(),
            model: model.into(),
            settings: st,
            gpus: vec![(0, 32768, 30000), (1, 16384, 8000)],
            mem: (122880, 100000),
            port_free: true,
            disk: 500 << 30,
        }
    };
    let m = |p: &Path| p.to_string_lossy().to_string();
    let mut all: Vec<Sc> = vec![];
    for model in [&moe, &dense] {
        for profile in config::PROFILES {
            for e in [&fle, &up] {
                all.push(base(
                    &format!(
                        "{} {profile} {}",
                        model.display(),
                        e.rsplit('/').next().unwrap()
                    ),
                    &m(model),
                    e,
                    profile,
                ));
            }
        }
    }
    all.push(base("old engine", &m(&moe), &old, "auto"));
    all.push(base("old engine off", &m(&moe), &old, "off"));
    all.push(base(
        "head only",
        &m(&with_head.join("m.gguf")),
        &fle,
        "auto",
    ));
    all.push(base(
        "head and vocab",
        &m(&with_both.join("m.gguf")),
        &fle,
        "auto",
    ));
    all.push(base("no model", "", &fle, "auto"));
    all.push(base(
        "missing model",
        &m(&d.join("nope.gguf")),
        &fle,
        "auto",
    ));
    all.push(base("garbage model", &m(&garbage), &fle, "auto"));
    all.push(base(
        "missing engine",
        &m(&moe),
        "/nonexistent/llama-server",
        "auto",
    ));
    all.push(base("no engine at all", &m(&moe), "", "auto"));
    let mut c = base("no gpu", &m(&moe), &fle, "auto");
    c.gpus = vec![];
    all.push(c);
    let mut c = base("no gpu off", &m(&moe), &up, "off");
    c.gpus = vec![];
    all.push(c);
    let mut c = base("tiny gpu moe", &m(&moe), &fle, "auto");
    c.gpus = vec![(0, 4096, 1000)];
    all.push(c);
    let mut c = base("tiny gpu dense", &m(&dense), &fle, "max");
    c.gpus = vec![(0, 4096, 1000)];
    all.push(c);
    let mut c = base("tiny machine", &m(&moe), &fle, "auto");
    c.mem = (50, 10);
    c.gpus = vec![(0, 1, 1)];
    c.disk = 1 << 20;
    all.push(c);
    let mut c = base("tiny machine dense", &m(&dense), &fle, "balanced");
    c.mem = (50, 10);
    c.gpus = vec![(0, 1, 1)];
    all.push(c);
    let mut c = base("port busy", &m(&moe), &fle, "auto");
    c.port_free = false;
    all.push(c);
    let mut c = base("few ram", &m(&moe), &fle, "auto");
    c.mem = (4096, 20);
    all.push(c);
    let mut c = base("profile with capitals", &m(&moe), &fle, "auto");
    c.settings.insert("profile".into(), json!("MAX"));
    all.push(c);
    let mut c = base("unknown profile", &m(&moe), &fle, "auto");
    c.settings.insert("profile".into(), json!("turbo"));
    all.push(c);

    let input: Vec<Value> = all.iter().map(|c| json!({"model": c.model, "settings": c.settings, "system": system_of(c),
        "host": {"cpu_physical": 8, "cpu_logical": 16, "mem_total": c.mem.0 << 20, "mem_available": c.mem.1 << 20,
        "gpus": c.gpus.iter().map(|(i, t, f)| json!({"index": i, "name": format!("GPU{i}"), "memory_total": t << 20, "memory_used": (t - f) << 20, "memory_free": f << 20})).collect::<Vec<_>>(), "port_free": c.port_free}})).collect();
    let tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tools");
    if !tools.join("fle/doctor.py").exists() {
        eprintln!("python helper not found: skipped");
        return;
    }
    std::fs::write(d.join("driver.py"), DRIVER).unwrap();
    std::fs::write(d.join("in.json"), serde_json::to_string(&input).unwrap()).unwrap();
    let cfg_home = d.join("xdg");
    let out = std::process::Command::new("python3")
        .arg(d.join("driver.py"))
        .arg(&tools)
        .arg(d.join("in.json"))
        .env("XDG_CONFIG_HOME", &cfg_home)
        .env_remove("FLE_CONFIG")
        .env_remove("FLE_ENGINE")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "python driver failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap();
    std::env::set_var("XDG_CONFIG_HOME", &cfg_home);
    std::env::remove_var("FLE_CONFIG");
    std::env::remove_var("FLE_ENGINE");
    for (sc, want) in all.iter().zip(&py) {
        let got = doctor::run(&sc.model, &sc.settings, &host_of(sc), &system_of(sc), &[]);
        let theirs = &want["report"];
        if sc.name == "no engine at all" && fle_core::launch::which("llama-server").is_some() {
            continue; // a llama-server on PATH is found by Rust and (patched out) not by the Python driver
        }
        for (k, v) in got.as_object().unwrap() {
            assert_eq!(v, &theirs[k], "[{}] field {k} differs", sc.name);
        }
        assert_eq!(py_json(&got), py_json(theirs), "[{}] JSON differs", sc.name);
    }
    eprintln!(
        "compared with the Python helper: {} doctor reports identical",
        all.len()
    );
}

#[test]
fn doctor_text_and_status_logic() {
    let checks = vec![
        json!({"name": "RAM", "status": "warn", "detail": "x", "hint": ""}),
        json!({"name": "GPU", "status": "fail", "detail": "y", "hint": ""}),
    ];
    assert_eq!(doctor::worst(&checks), "fail");
    assert_eq!(doctor::worst(&[]), "ok");
    let s = doctor::suggest(&checks, "auto", "/m/a b.gguf", false);
    let cmds: Vec<_> = s
        .iter()
        .map(|i| i["command"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        cmds,
        [
            "fle plan '/m/a b.gguf' --profile nvme",
            "fle plan '/m/a b.gguf' --profile off"
        ]
    );
    assert!(doctor::suggest(&checks, "auto", "", false).is_empty());
}

#[test]
fn report_redacts_secrets_and_writes_zip() {
    let v = json!({"HF_TOKEN": "x", "nested": {"api_key": "y", "ok": 1, "list": [{"Password": "z", "n": 2}]}, "env": {"MY_SECRET": "s", "FLE_X": "1"}});
    let r = report::redact(&v);
    assert_eq!(r["HF_TOKEN"], "<redacted>");
    assert_eq!(r["nested"]["api_key"], "<redacted>");
    assert_eq!(r["nested"]["ok"], 1);
    assert_eq!(r["nested"]["list"][0]["Password"], "<redacted>");
    assert_eq!(r["env"]["MY_SECRET"], "<redacted>");
    assert_eq!(r["env"]["FLE_X"], "1");
    let t = report::utc_now();
    assert!(
        t.len() == 25 && t.ends_with("+00:00") && &t[4..5] == "-" && &t[10..11] == "T",
        "{t}"
    );
    let d = tmp("zip");
    let rep = json!({"status": "ok", "checks": [{"name": "a", "status": "ok", "detail": "d", "hint": "h"}], "plan": {"command": "llama-server -m x"}});
    let p = report::write(&rep, d.join("r.zip").to_str().unwrap(), false).unwrap();
    let z = zip::ZipArchive::new(std::fs::File::open(&p).unwrap()).unwrap();
    let mut names: Vec<_> = z.file_names().map(String::from).collect();
    names.sort();
    assert_eq!(names, ["checks.txt", "plan.sh", "report.json"]);
    let p2 = report::write(&rep, d.join("r.json").to_str().unwrap(), false).unwrap();
    assert_eq!(
        std::fs::read_to_string(p2).unwrap(),
        format!("{}\n", py_json(&rep))
    );
    let _ = gguf::human_bytes(1);
}

//! The start plan: Rust and the Python helper (`tools/fle/plan.py`) get the same machine, model and settings and must
//! produce the same plan, byte for byte in JSON, or the same error and hint.
#![cfg(unix)] // the fake engines are shell scripts
use fle_core::{config, gguf, gguf::py_json, launch::*};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

mod common;
use common::*;

const DRIVER: &str = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
from fle import hardware, plan, gguf, config
from fle.util import FleError
out = []
for sc in json.load(open(sys.argv[2])):
    h = sc["host"]
    hardware.cpu = lambda h=h: {"name": "x", "physical": h["cpu_physical"], "logical": h["cpu_logical"]}
    hardware.memory = lambda h=h: {"total": h["mem_total"], "available": h["mem_available"]}
    hardware.gpus = lambda h=h: [dict(g, vendor="nvidia") for g in h["gpus"]]
    hardware.port_free = lambda host, port, h=h: h["port_free"]
    plan.os.name = "posix"
    plan.sys.platform = "linux"
    o = sc["opts"]
    try:
        model = gguf.read_model(sc["model"])
        p = plan.build(sc["settings"], model, engine_path=o["engine_path"], model_path=sc["model"], extra_args=o["extra_args"],
                       env_override=dict(o["env_override"]), gpu_ids=o["gpu_ids"], no_mtp=o["no_mtp"], mtp_head=o["mtp_head"],
                       mtp_vocab=o["mtp_vocab"], spec=o["spec"])
        d = p.as_dict()
        out.append({"plan": d, "block": p.command_block(), "oneline": p.command_block(oneline=True)})
    except FleError as e:
        out.append({"error": str(e), "hint": e.hint})
print(json.dumps(out, ensure_ascii=True, indent=2))
"#;

struct Sc {
    name: String,
    model: PathBuf,
    settings: Map<String, Value>,
    opts: Options,
    gpus: Vec<(i64, u64, u64)>, // index, total MiB, free MiB
    mem: (u64, u64),
    port_free: bool,
}

fn host_json(sc: &Sc) -> Value {
    let gpus: Vec<Value> = sc.gpus.iter().map(|(i, t, f)| json!({"index": i, "name": format!("GPU{i}"), "memory_total": t << 20, "memory_used": (t - f) << 20, "memory_free": f << 20})).collect();
    json!({"cpu_physical": 8, "cpu_logical": 16, "mem_total": sc.mem.0 << 20, "mem_available": sc.mem.1 << 20, "gpus": gpus, "port_free": sc.port_free})
}

fn rust_host(sc: &Sc) -> Host {
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

#[test]
fn text_helpers_behave_like_python() {
    assert_eq!(shquote(""), "''");
    assert_eq!(shquote("a-b_c/d.e:f=g,h@i%j+k"), "a-b_c/d.e:f=g,h@i%j+k");
    assert_eq!(shquote("it's"), "'it'\"'\"'s'");
    assert_eq!(shquote("a b"), "'a b'");
    assert_eq!(human_int(32768), "32,768");
    assert_eq!(human_int(999), "999");
    assert_eq!(human_int(1_000_000), "1,000,000");
    let a: Vec<String> = ["-ngl", "99", "--jinja", "-c", "-5", "--x", "-t", "8"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let g = flag_groups(&a);
    assert_eq!(
        g,
        vec![
            vec!["-ngl", "99"],
            vec!["--jinja"],
            vec!["-c", "-5"],
            vec!["--x"],
            vec!["-t", "8"]
        ]
    );
}

#[test]
fn plans_match_the_python_helper() {
    let d = tmp("m");
    let moe = d.join("moe.gguf");
    write(&moe, &kv("qwen3moe", "Tiny MoE", true), &tensors(true));
    let dense = d.join("dense.gguf");
    write(&dense, &kv("qwen3", "Tiny Dense", false), &tensors(false));
    let generic = d.join("q4_k_m.gguf");
    write(&generic, &kv("qwen3moe", "Q4_K_M", true), &tensors(true));
    let fle = engine(&d, "fle-server", true, true);
    let up = engine(&d, "up-server", false, true);
    let nospec = engine(&d, "nospec-server", true, false);
    let head = d.join("head.gguf");
    std::fs::write(&head, b"x").unwrap();
    let vocab = d.join("vocab.txt");
    std::fs::write(&vocab, b"x").unwrap();

    let two = vec![(0, 32768, 30000), (1, 16384, 8000)];
    let base = |name: &str, model: &Path, engine: &str, profile: &str| -> Sc {
        let mut st = config::defaults();
        st.insert("profile".into(), json!(profile));
        Sc {
            name: name.to_string(),
            model: model.to_path_buf(),
            settings: st,
            opts: Options {
                engine_path: engine.to_string(),
                model_path: model.to_string_lossy().to_string(),
                ..Default::default()
            },
            gpus: two.clone(),
            mem: (122880, 100000),
            port_free: true,
        }
    };
    let mut all: Vec<Sc> = vec![];
    for model in [&moe, &dense] {
        for profile in config::PROFILES {
            for eng in [&fle, &up] {
                all.push(base(
                    &format!(
                        "{} {profile} {}",
                        model.display(),
                        eng.rsplit('/').next().unwrap()
                    ),
                    model,
                    eng,
                    profile,
                ));
            }
        }
    }
    let mut c = base("generic name", &generic, &fle, "auto");
    c.settings.insert("context".into(), json!(999999));
    all.push(c);
    for spec in ["none", "mtp", "ngram-mod", "ngram-simple", "bogus"] {
        for profile in ["auto", "off", "nvme"] {
            let mut c = base(&format!("spec {spec} {profile}"), &moe, &fle, profile);
            c.opts.spec = spec.into();
            all.push(c);
            let mut c = base(
                &format!("spec {spec} {profile} with sidecars"),
                &moe,
                &fle,
                profile,
            );
            c.opts.spec = spec.into();
            c.opts.mtp_head = head.to_string_lossy().to_string();
            c.opts.mtp_vocab = vocab.to_string_lossy().to_string();
            all.push(c);
        }
    }
    for (i, (h, v)) in [("", ""), ("h", ""), ("", "v"), ("h", "v")]
        .iter()
        .enumerate()
    {
        let mut c = base(&format!("sidecar pair {i}"), &moe, &fle, "auto");
        if !h.is_empty() {
            c.opts.mtp_head = head.to_string_lossy().to_string();
        }
        if !v.is_empty() {
            c.opts.mtp_vocab = vocab.to_string_lossy().to_string();
        }
        all.push(c.clone_sc());
        c.opts.no_mtp = true;
        all.push(c);
    }
    let mut c = base("ngram on engine without spec-type", &moe, &nospec, "auto");
    c.opts.spec = "ngram-mod".into();
    all.push(c);
    let mut c = base("mtp contradiction", &moe, &fle, "auto");
    c.opts.spec = "mtp".into();
    c.opts.no_mtp = true;
    all.push(c);
    let mut c = base("mtp setting off", &moe, &fle, "auto");
    c.settings.insert("mtp".into(), json!(false));
    all.push(c);
    let mut c = base("cache request too big", &moe, &fle, "auto");
    c.settings.insert("gpu_cache_mib".into(), json!(99999));
    all.push(c);
    let mut c = base("cache request on dense", &dense, &fle, "auto");
    c.settings.insert("gpu_cache_mib".into(), json!(500));
    all.push(c);
    let mut c = base("no gpu fle", &moe, &fle, "auto");
    c.gpus = vec![];
    all.push(c);
    let mut c = base("no gpu upstream off", &moe, &up, "off");
    c.gpus = vec![];
    all.push(c);
    let mut c = base("tiny gpu", &moe, &fle, "auto");
    c.gpus = vec![(0, 4096, 1000)];
    all.push(c);
    let mut c = base("tiny machine", &moe, &fle, "nvme");
    c.gpus = vec![(0, 1, 1)];
    c.mem = (1, 1);
    all.push(c);
    let mut c = base("gpu ids", &moe, &fle, "auto");
    c.opts.gpu_ids = "1".into();
    all.push(c.clone_sc());
    c.opts.gpu_ids = "7".into();
    all.push(c);
    let mut c = base("port busy", &moe, &fle, "auto");
    c.port_free = false;
    all.push(c);
    let mut c = base("few ram, pinned auto", &moe, &fle, "auto");
    c.mem = (4096, 2048);
    all.push(c);
    for pin in ["on", "off", "auto"] {
        let mut c = base(&format!("pinned {pin}"), &moe, &fle, "max");
        c.settings.insert("pinned_load".into(), json!(pin));
        all.push(c);
    }
    let mut c = base("parallel without unified", &moe, &fle, "auto");
    c.opts.extra_args = vec!["--parallel".into(), "4".into()];
    all.push(c.clone_sc());
    c.opts.extra_args = vec!["--parallel=2".into(), "--kv-unified".into()];
    all.push(c);
    let mut c = base("env layers", &moe, &fle, "auto");
    c.settings.insert(
        "env".into(),
        json!({"FLE_MOE_CACHE_BIAS": "0.5", "FLE_X": "1"}),
    );
    c.opts.env_override = vec![
        ("FLE_X".into(), "cli".into()),
        ("A B".into(), "it's".into()),
    ];
    c.opts.extra_args = vec![
        "--no-mmap".into(),
        "-b".into(),
        "2048".into(),
        "--chat template".into(),
    ];
    all.push(c);
    let mut c = base("threads, fit, host", &dense, &fle, "balanced");
    c.settings.insert("threads".into(), json!(6));
    c.settings.insert("fit_target_mib".into(), json!(2000));
    c.settings.insert("host".into(), json!("0.0.0.0"));
    c.settings.insert("port".into(), json!(9090));
    all.push(c);
    let mut c = base("unknown profile", &moe, &fle, "turbo");
    c.settings.insert("profile".into(), json!("turbo"));
    all.push(c);
    let mut c = base("missing engine", &moe, "/nonexistent/llama-server", "auto");
    c.opts.engine_path = "/nonexistent/llama-server".into();
    all.push(c);

    let input: Vec<Value> = all.iter().map(|c| json!({"model": c.model.to_string_lossy(), "settings": c.settings, "opts": {
        "engine_path": c.opts.engine_path, "extra_args": c.opts.extra_args, "env_override": c.opts.env_override, "gpu_ids": c.opts.gpu_ids,
        "no_mtp": c.opts.no_mtp, "mtp_head": c.opts.mtp_head, "mtp_vocab": c.opts.mtp_vocab, "spec": c.opts.spec}, "host": host_json(c)})).collect();
    let repo_tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tools");
    if !repo_tools.join("fle/plan.py").exists() {
        eprintln!("python helper not found: skipped");
        return;
    }
    std::fs::write(d.join("driver.py"), DRIVER).unwrap();
    std::fs::write(d.join("in.json"), serde_json::to_string(&input).unwrap()).unwrap();
    let cfg_home = d.join("xdg");
    let out = std::process::Command::new("python3")
        .arg(d.join("driver.py"))
        .arg(&repo_tools)
        .arg(d.join("in.json"))
        .env("XDG_CONFIG_HOME", &cfg_home)
        .env_remove("FLE_CONFIG")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "python driver failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(py.len(), all.len());

    std::env::set_var("XDG_CONFIG_HOME", &cfg_home);
    std::env::remove_var("FLE_CONFIG");
    let (mut plans, mut errors) = (0, 0);
    for (sc, want) in all.iter().zip(&py) {
        let got = build(
            &sc.settings,
            &gguf::read_model(&sc.model).unwrap(),
            &sc.opts,
            &rust_host(sc),
        );
        match (got, want.get("error")) {
            (Ok(p), None) => {
                let theirs = &want["plan"];
                let mine: Value = serde_json::from_str(&py_json(&p.as_json())).unwrap();
                for (k, v) in mine.as_object().unwrap() {
                    assert_eq!(v, &theirs[k], "[{}] field {k} differs", sc.name);
                }
                assert_eq!(
                    py_json(&mine),
                    py_json(theirs),
                    "[{}] JSON differs",
                    sc.name
                );
                assert_eq!(
                    p.command_block(false),
                    want["block"].as_str().unwrap(),
                    "[{}] command block differs",
                    sc.name
                );
                assert_eq!(
                    p.command_block(true),
                    want["oneline"].as_str().unwrap(),
                    "[{}] one-line command differs",
                    sc.name
                );
                plans += 1;
            }
            (Err(e), Some(msg)) => {
                let fe = e
                    .downcast_ref::<fle_core::FleError>()
                    .unwrap_or_else(|| panic!("[{}] not a FleError: {e}", sc.name));
                assert_eq!(
                    &fe.message,
                    msg.as_str().unwrap(),
                    "[{}] error text differs",
                    sc.name
                );
                assert_eq!(
                    &fe.hint,
                    want["hint"].as_str().unwrap(),
                    "[{}] hint differs",
                    sc.name
                );
                errors += 1;
            }
            (Ok(_), Some(msg)) => panic!("[{}] Rust built a plan, Python said: {msg}", sc.name),
            (Err(e), None) => panic!("[{}] Rust failed ({e}), Python built a plan", sc.name),
        }
    }
    eprintln!("compared with the Python helper: {plans} plans and {errors} errors identical");
    assert!(plans > 40 && errors > 10);
}

trait CloneSc {
    fn clone_sc(&self) -> Sc;
}
impl CloneSc for Sc {
    fn clone_sc(&self) -> Sc {
        Sc {
            name: self.name.clone(),
            model: self.model.clone(),
            settings: self.settings.clone(),
            opts: self.opts.clone(),
            gpus: self.gpus.clone(),
            mem: self.mem,
            port_free: self.port_free,
        }
    }
}

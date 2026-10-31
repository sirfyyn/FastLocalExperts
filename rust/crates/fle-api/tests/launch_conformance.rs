//! The engine start command (auto plan, profile plan, pasted-command parsing) against the Python Studio: the same inputs
//! must give the same command, environment and notes, or the same error text.
#![cfg(unix)]
mod harness;
use fle_api::launch::*;
use fle_api::settings::defaults;
use harness::*;
use serde_json::{json, Map, Value};

const DRIVER: &str = r#"
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fle_studio import launch
sc = json.load(open(sys.argv[2]))
data = Path(sc["data_dir"])
out = {"quote": [launch.shell_quote(w) for w in sc["words"]], "split": [], "envlines": [], "auto": [], "profile": []}
for t in sc["commands"]:
    try:
        env, b, a = launch.split_command(t)
        out["split"].append({"env": env, "binary": b, "args": a})
    except launch.PlanError as e:
        out["split"].append({"error": str(e)})
for t in sc["envtexts"]:
    try:
        out["envlines"].append(launch.parse_env_lines(t))
    except launch.PlanError as e:
        out["envlines"].append({"error": str(e)})
for c in sc["auto"]:
    try:
        p = launch.auto_plan(model=c["model"], engine=c["engine"], settings=c["settings"], sysinfo=c["sysinfo"], port=c["port"], data_dir=data)
        out["auto"].append(p.public())
    except launch.PlanError as e:
        out["auto"].append({"error": str(e)})
for c in sc["profile"]:
    try:
        prof = launch.Profile(**c["profile"])
        p = launch.profile_plan(profile=prof, model=c["model"], engine=c["engine"], settings=c["settings"], sysinfo=c["sysinfo"], port=c["port"], data_dir=data)
        out["profile"].append(p.public())
    except launch.PlanError as e:
        out["profile"].append({"error": str(e)})
out["builtin"] = [p.public() for p in launch.BUILTIN]
print(json.dumps(out))
"#;

fn settings(f: impl FnOnce(&mut Value)) -> Value {
    let mut s = Value::Object(defaults());
    f(&mut s);
    s
}

fn gpu(index: i64, name: &str, total_gib: f64, used_gib: f64) -> Value {
    let g = 1073741824.0;
    json!({"vendor": "nvidia", "index": index, "name": name, "memory_total": (total_gib * g) as i64, "memory_used": (used_gib * g) as i64, "driver": "570.1"})
}

fn sys(gpus: Vec<Value>, mem_gib: f64, physical: i64, logical: i64) -> Value {
    json!({"gpus": gpus, "memory": {"total": (mem_gib * 1073741824.0) as i64, "available": 1}, "cpu": {"physical": physical, "logical": logical}})
}

fn model(moe: bool, mtp: bool, head: bool) -> Value {
    json!({"id": "0123456789ab", "name": "Tiny", "path": "/models/tiny.gguf",
        "info": {"arch": "qwen3moe", "name": "Tiny", "context_length": 40960, "layers": 4, "experts": if moe { json!(64) } else { Value::Null }, "experts_used": 4,
                 "expert_bytes": if moe { 30_000_000_000i64 } else { 0 }, "input_bytes": 1_000_000_000i64, "gpu_bytes": 4_500_000_000i64, "moe": moe, "chat_template": true, "kv_bytes_per_token_q8": 69632},
        "mtp": {"supported": mtp, "head": if head { json!("/m/head.gguf") } else { Value::Null }, "vocab": if head { json!("/m/vocab.txt") } else { Value::Null }},
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}})
}

fn eng(kind: &str, flavor: Value, backend: &str, lv: Option<bool>) -> Value {
    let mut e = json!({"kind": kind, "flavor": flavor, "backend": backend, "path": "/engines/llama-server", "version": "1"});
    if let Some(l) = lv {
        e["log_verbosity"] = json!(l);
    }
    e
}

#[test]
fn start_commands_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found: skipped");
        return;
    };
    let d = tmp("launch");
    let data = d.join("data");
    let words: Vec<&str> = vec![
        "",
        "plain",
        "a b",
        "{model}",
        "{mtp_head}x",
        "{a,b}",
        "{Model}",
        "{}",
        "x{model}y",
        "it's",
        "--flag=1",
        "/p/a-b_c.d",
        "ü",
        "$HOME",
        "{model} {port}",
        "a{port}b{host}",
    ];
    let commands: Vec<&str> = vec![
        "FLE_A=1 FLE_B='two words' /bin/llama-server -m /m/x.gguf --port 8080",
        "env FLE_A=1 llama-server -c 4096 \\\n  --jinja \\\n  -ngl 99",
        "ExecStart=/usr/bin/llama-server -m x",
        "ExecStart=-@/usr/bin/llama-server -m x",
        "-m x --host 0.0.0.0",
        "FLE_X=1 -m x",
        "A=1 A=2 bin",
        "bin 'unclosed",
        "bin trailing\\",
        "bin \"dq \\\" and \\\\ and \\n\" 'sq \\n'",
        "  \n  bin   a    b  \n",
        "",
        "env",
        "1BAD=1 bin",
        "bin ^\r\n arg",
        "bin `\n arg",
        "A= bin",
        "A==b bin",
    ];
    let envtexts: Vec<&str> = vec![
        "A=1\nB = 2 \n# comment\n\n  C=x=y\n",
        "A=1\nnot a line\n",
        "=x",
        "1A=x",
        "A-B=x",
        "A=1\nA=2",
        "",
    ];
    let mut auto: Vec<Value> = vec![];
    let big = sys(vec![gpu(0, "RTX Big", 32.0, 1.0)], 120.0, 16, 32);
    let two = sys(
        vec![gpu(0, "RTX A", 16.0, 8.0), gpu(1, "RTX B", 32.0, 2.0)],
        64.0,
        8,
        8,
    );
    let small = sys(vec![gpu(0, "RTX Small", 8.0, 7.5)], 16.0, 4, 8);
    let none = sys(vec![], 16.0, 8, 8);
    let apple = json!({"gpus": [{"vendor": "apple", "index": 0, "unified": true, "memory_total": 17179869184i64}], "memory": {"total": 17179869184i64, "available": 1}, "cpu": {"physical": 8, "logical": 8}});
    let mut add = |m: Value, e: Value, s: Value, y: Value| {
        auto.push(json!({"model": m, "engine": e, "settings": s, "sysinfo": y, "port": 18080}))
    };
    let fle = || eng("fle", json!("fle"), "cuda", Some(true));
    for y in [&big, &two, &small, &none, &apple] {
        for (moe, mtp, head) in [
            (true, true, true),
            (true, true, false),
            (false, false, false),
            (true, false, false),
        ] {
            add(model(moe, mtp, head), fle(), settings(|_| {}), y.clone());
            add(
                model(moe, mtp, head),
                eng("upstream", json!("upstream"), "cuda", Some(true)),
                settings(|_| {}),
                y.clone(),
            );
        }
    }
    let m = || model(true, true, true);
    let tweak = |f: &dyn Fn(&mut Value)| settings(|s| f(s));
    let cases: Vec<(&str, Value)> = vec![
        ("ctx", tweak(&|s| s["launch"]["context"] = json!(1_000_000))),
        ("ctx0", tweak(&|s| s["launch"]["context"] = json!(0))),
        ("parallel", tweak(&|s| s["launch"]["parallel"] = json!(4))),
        ("threads", tweak(&|s| s["launch"]["threads"] = json!(6))),
        (
            "budget",
            tweak(&|s| s["launch"]["gpu_cache_mib"] = json!(12000)),
        ),
        ("gpus all", tweak(&|s| s["launch"]["gpus"] = json!("all"))),
        ("gpus 1", tweak(&|s| s["launch"]["gpus"] = json!("1"))),
        ("gpus 0,1", tweak(&|s| s["launch"]["gpus"] = json!("0, 1"))),
        ("gpus 7", tweak(&|s| s["launch"]["gpus"] = json!("7"))),
        (
            "pinned on",
            tweak(&|s| s["launch"]["pinned_load"] = json!("on")),
        ),
        (
            "pinned off",
            tweak(&|s| s["launch"]["pinned_load"] = json!("off")),
        ),
        ("spec none", tweak(&|s| s["launch"]["spec"] = json!("none"))),
        ("spec mtp", tweak(&|s| s["launch"]["spec"] = json!("mtp"))),
        (
            "spec ngram-mod",
            tweak(&|s| s["launch"]["spec"] = json!("ngram-mod")),
        ),
        (
            "spec ngram-simple",
            tweak(&|s| s["launch"]["spec"] = json!("NGRAM-SIMPLE")),
        ),
        (
            "spec bogus",
            tweak(&|s| s["launch"]["spec"] = json!("bogus")),
        ),
        ("mtp off", tweak(&|s| s["launch"]["mtp"] = json!(false))),
        (
            "cram 0",
            tweak(&|s| s["launch"]["prompt_cache_mib"] = json!(0)),
        ),
        (
            "cram 4096",
            tweak(&|s| s["launch"]["prompt_cache_mib"] = json!(4096)),
        ),
        ("mask", tweak(&|s| s["launch"]["cpu_mask"] = json!("0xFF"))),
        (
            "extra",
            tweak(&|s| {
                s["launch"]["extra_args"] =
                    json!("--no-mmap -b 2048 --chat-template-file '/a b/t.jinja'");
                s["launch"]["extra_env"] = json!("FLE_X=1\n# c\nFLE_Y = 2");
            }),
        ),
        (
            "extra port",
            tweak(&|s| {
                s["launch"]["extra_args"] =
                    json!("--port 9999 --host 0.0.0.0 -a custom --metrics -v")
            }),
        ),
        (
            "extra bad port",
            tweak(&|s| s["launch"]["extra_args"] = json!("--port abc")),
        ),
        (
            "extra bad env",
            tweak(&|s| s["launch"]["extra_env"] = json!("not a line")),
        ),
        (
            "extra bad quote",
            tweak(&|s| s["launch"]["extra_args"] = json!("'oops")),
        ),
        (
            "custom sampling",
            tweak(&|s| {
                s["chat"]["sampling"] = json!("custom");
                s["chat"]["temperature"] = json!(0.25);
                s["chat"]["top_k"] = json!(40);
                s["chat"]["min_p"] = json!(0.05);
                s["chat"]["presence_penalty"] = json!(0.0);
                s["chat"]["top_p"] = json!(0.9);
            }),
        ),
        (
            "lv env",
            tweak(&|s| s["launch"]["extra_env"] = json!("LLAMA_ARG_LOG_VERBOSITY=1")),
        ),
    ];
    for (_, s) in &cases {
        for y in [&big, &two, &small] {
            add(m(), fle(), s.clone(), y.clone());
        }
        add(
            m(),
            eng("upstream", json!("upstream"), "cpu", Some(true)),
            s.clone(),
            none.clone(),
        );
    }
    // engines of other kinds: custom binary (flavor known or not), no -lv, a model with no info and no sampling
    add(
        m(),
        eng("custom", json!("fle"), "cuda", Some(false)),
        settings(|_| {}),
        big.clone(),
    );
    add(
        m(),
        eng("custom", Value::Null, "cuda", None),
        settings(|_| {}),
        none.clone(),
    );
    add(
        json!({"id": "x", "path": "/m/y.gguf"}),
        eng("fle", json!("fle"), "cuda", None),
        settings(|_| {}),
        big.clone(),
    );
    add(
        json!({"id": "x", "path": "/m/y.gguf", "info": null, "mtp": null, "sampling": null}),
        eng("upstream", json!("upstream"), "cpu", Some(true)),
        settings(|_| {}),
        none.clone(),
    );

    // profiles
    let reference = json!({"id": "fle-reference", "name": "FLE reference (32 GB GPU)", "engine": "fle", "binary": "", "args": reference_args_json(), "env": reference_env_json(), "notes": "n", "builtin": true, "meta": {}});
    let p = |id: &str,
             name: &str,
             engine: &str,
             binary: &str,
             args: Value,
             env: Value,
             notes: &str| json!({"id": id, "name": name, "engine": engine, "binary": binary, "args": args, "env": env, "notes": notes, "builtin": false, "meta": {}});
    let profiles = vec![
        reference,
        p(
            "a",
            "plain",
            "fle",
            "",
            json!(["--jinja", "-c", "{context}", "--parallel", "2"]),
            json!({}),
            "",
        ),
        p(
            "b",
            "with model",
            "upstream",
            "",
            json!(["-m", "/own/model.gguf", "--port", "{port}"]),
            json!({"X": "{threads}"}),
            "notes here",
        ),
        p(
            "c",
            "custom",
            "custom",
            "/opt/my/llama-server",
            json!(["-c", "{context}", "--host", "{host}", "--alias=mine"]),
            json!({"CUDA_VISIBLE_DEVICES": "0"}),
            "",
        ),
        p(
            "d",
            "unknown",
            "fle",
            "",
            json!([
                "--x",
                "{unknown}",
                "{data}/y",
                "{expert_profile}",
                "{mtp_head}"
            ]),
            json!({}),
            "",
        ),
        p(
            "e",
            "needs vocab",
            "fle",
            "",
            json!(["-m", "{model}"]),
            json!({"V": "{mtp_vocab}"}),
            "",
        ),
        p(
            "f",
            "mask in args",
            "fle",
            "",
            json!(["-C", "0x1", "-md", "/d.gguf", "--model-draft=/e.gguf"]),
            json!({}),
            "",
        ),
        p(
            "g",
            "lv in args",
            "fle",
            "",
            json!([
                "--log-verbosity",
                "1",
                "-a",
                "x",
                "--metrics",
                "--host",
                "1.2.3.4"
            ]),
            json!({}),
            "",
        ),
        p(
            "h",
            "bad port",
            "fle",
            "",
            json!(["--port=abc"]),
            json!({}),
            "",
        ),
        p(
            "i",
            "custom braces",
            "custom",
            "{data}/bin",
            json!([]),
            json!({}),
            "",
        ),
    ];
    let mut prof: Vec<Value> = vec![];
    for pr in &profiles {
        for (m_, e_, s_, y_) in [
            (
                model(true, true, true),
                eng("fle", json!("fle"), "cuda", Some(true)),
                settings(|_| {}),
                big.clone(),
            ),
            (
                model(true, true, false),
                eng("upstream", json!("upstream"), "cpu", Some(true)),
                settings(|s| {
                    s["launch"]["gpus"] = json!("1");
                    s["launch"]["cpu_mask"] = json!("0xF");
                }),
                two.clone(),
            ),
            (
                Value::Null,
                Value::Null,
                settings(|s| s["launch"]["threads"] = json!(3)),
                none.clone(),
            ),
            (
                json!({}),
                eng("fle", json!("fle"), "cuda", None),
                settings(|s| s["launch"]["gpus"] = json!("9")),
                big.clone(),
            ),
            (
                model(false, false, false),
                Value::Null,
                settings(|_| {}),
                none.clone(),
            ),
        ] {
            prof.push(json!({"profile": pr, "model": m_, "engine": e_, "settings": s_, "sysinfo": y_, "port": 18080}));
        }
    }
    std::fs::write(d.join("driver.py"), DRIVER).unwrap();
    std::fs::write(d.join("in.json"), json!({"data_dir": data.to_string_lossy(), "words": words, "commands": commands, "envtexts": envtexts, "auto": auto, "profile": prof}).to_string()).unwrap();
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

    for (i, w) in words.iter().enumerate() {
        assert_eq!(json!(shell_quote(w)), py["quote"][i], "shell_quote({w:?})");
    }
    for (i, t) in commands.iter().enumerate() {
        let want = &py["split"][i];
        match split_command(t) {
            Ok((env, b, a)) => {
                let env: Map<String, Value> = env.into_iter().map(|(k, v)| (k, json!(v))).collect();
                assert_eq!(
                    json!({"env": env, "binary": b, "args": a}),
                    *want,
                    "split_command({t:?})"
                );
            }
            Err(e) => assert_eq!(json!({"error": e.0}), *want, "split_command({t:?})"),
        }
    }
    for (i, t) in envtexts.iter().enumerate() {
        let want = &py["envlines"][i];
        match parse_env_lines(t) {
            Ok(e) => {
                let m: Map<String, Value> = e.into_iter().map(|(k, v)| (k, json!(v))).collect();
                assert_eq!(Value::Object(m), *want, "parse_env_lines({t:?})");
            }
            Err(e) => assert_eq!(json!({"error": e.0}), *want, "parse_env_lines({t:?})"),
        }
    }
    let same = |label: &str, i: usize, got: Result<Plan, PlanError>, want: &Value| match got {
        Ok(p) => assert_eq!(p.public(), *want, "{label} #{i}"),
        Err(e) => assert_eq!(json!({"error": e.0}), untuple(want), "{label} #{i}"),
    };
    for (i, c) in auto.iter().enumerate() {
        let got = auto_plan(&AutoInput {
            model: &c["model"],
            engine: &c["engine"],
            settings: &c["settings"],
            sysinfo: &c["sysinfo"],
            port: 18080,
            data_dir: &data,
        });
        same("auto", i, got, &py["auto"][i]);
    }
    let all_profiles: Vec<Profile> = builtin().into_iter().chain(std::iter::empty()).collect();
    assert_eq!(
        all_profiles[0].public(),
        py["builtin"][0],
        "the built-in profile"
    );
    for (i, c) in prof.iter().enumerate() {
        let mut o = c["profile"].as_object().unwrap().clone();
        o.insert("id".into(), c["profile"]["id"].clone());
        let profile = profile_of(&c["profile"]);
        let nullable = |v: &Value| if v.is_null() { None } else { Some(v.clone()) };
        let (m, e) = (nullable(&c["model"]), nullable(&c["engine"]));
        let got = profile_plan(&ProfileInput {
            profile: &profile,
            model: m.as_ref(),
            engine: e.as_ref(),
            settings: &c["settings"],
            sysinfo: &c["sysinfo"],
            port: 18080,
            data_dir: &data,
        });
        same("profile", i, got, &py["profile"][i]);
    }
    let errors = |k: &str| {
        py[k]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v.get("error").is_some())
            .count()
    };
    assert!(
        errors("auto") >= 8 && errors("profile") >= 8 && errors("split") >= 2,
        "the cases must include real errors: {} {} {}",
        errors("auto"),
        errors("profile"),
        errors("split")
    );
    eprintln!("compared with the Python Studio: {} quotes, {} commands, {} env texts, {} auto plans, {} profile plans identical", words.len(), commands.len(), envtexts.len(), auto.len(), prof.len());
}

fn profile_of(v: &Value) -> Profile {
    Profile {
        id: v["id"].as_str().unwrap().into(),
        name: v["name"].as_str().unwrap().into(),
        engine: v["engine"].clone(),
        binary: v["binary"].clone(),
        args: v["args"].clone(),
        env: v["env"].clone(),
        notes: v["notes"].clone(),
        builtin: v["builtin"].as_bool().unwrap_or(false),
        meta: v["meta"].clone(),
    }
}
fn reference_args_json() -> Value {
    json!(REFERENCE_ARGS)
}
fn reference_env_json() -> Value {
    Value::Object(
        REFERENCE_ENV
            .iter()
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect(),
    )
}

/// Deliberate difference D1: Python printed a message with its hint as a tuple, `('message', 'hint')`; here they are two
/// sentences. Turn the Python text into ours before comparing.
fn untuple(v: &Value) -> Value {
    let Some(t) = v["error"].as_str() else {
        return v.clone();
    };
    let Some(inner) = t.strip_prefix('(').and_then(|x| x.strip_suffix(')')) else {
        return v.clone();
    };
    let q = match inner.chars().next() {
        Some(c @ ('\'' | '"')) => c,
        _ => return v.clone(),
    };
    let sep = format!("{q}, ");
    let Some(at) = inner.find(&sep).filter(|a| *a > 0) else {
        return v.clone();
    };
    let (first, rest) = (&inner[1..at], &inner[at + 3..]);
    let q2 = match rest.chars().next() {
        Some(c @ ('\'' | '"')) if rest.ends_with(c) => c,
        _ => return v.clone(),
    };
    let second = &rest[1..rest.len() - 1];
    let _ = q2;
    json!({"error": format!("{} {}", first.replace("\\'", "'"), second.replace("\\'", "'"))})
}

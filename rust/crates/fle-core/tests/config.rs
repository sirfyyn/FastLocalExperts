//! Settings file and precedence; the Python helper (`tools/fle/config.py`) is the specification.
use fle_core::config::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(()); // the tests below change process-wide environment variables

fn tmp(n: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fle-cfg-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn clear_env() {
    for k in [
        "FLE_PROFILE",
        "FLE_SPEC",
        "FLE_ENGINE",
        "FLE_MODEL",
        "FLE_HOST",
        "FLE_PORT",
        "FLE_CONTEXT",
        "FLE_THREADS",
        "FLE_GPU_CACHE_MIB",
        "FLE_FIT_TARGET_MIB",
        "FLE_MTP",
        "FLE_CONFIG",
    ] {
        std::env::remove_var(k);
    }
}
fn load_at(p: &Path) -> serde_json::Map<String, Value> {
    load(Some(p.to_str().unwrap())).unwrap().0
}

#[test]
fn defaults_come_in_the_documented_order() {
    let keys: Vec<_> = defaults().keys().cloned().collect();
    assert_eq!(
        keys,
        [
            "profile",
            "spec",
            "engine",
            "model",
            "host",
            "port",
            "context",
            "threads",
            "gpu_cache_mib",
            "fit_target_mib",
            "pinned_load",
            "mtp",
            "env"
        ]
    );
}

#[test]
fn missing_file_gives_the_defaults() {
    let _g = ENV.lock().unwrap();
    clear_env();
    let p = tmp("none").join("c.toml");
    assert_eq!(load_at(&p), defaults());
}

#[test]
fn file_values_are_coerced_like_python() {
    let _g = ENV.lock().unwrap();
    clear_env();
    let p = tmp("coerce").join("c.toml");
    std::fs::write(&p, "profile = \"MAX\"\nport = \"9000\"\ncontext = 4096.7\nmtp = \"no\"\nthreads = true\nhost = 7\n[env]\nA = 1\nB = true\n").unwrap();
    let d = load_at(&p);
    assert_eq!(d["profile"], "MAX"); // load only validates; `effective` lower-cases
    assert_eq!(
        effective(&Default::default(), &d).unwrap()["profile"],
        "max"
    );
    assert_eq!(d["port"], 9000); // text -> int
    assert_eq!(d["context"], 4096); // float -> int
    assert_eq!(d["mtp"], false); // "no" is not a true word
    assert_eq!(d["threads"], 0); // a boolean is not a number: the default
    assert_eq!(d["host"], "7"); // anything else is text
    assert_eq!(d["env"], json!({"A": "1", "B": "True"}));
}

#[test]
fn unknown_profile_is_refused_with_a_hint() {
    let _g = ENV.lock().unwrap();
    clear_env();
    let p = tmp("badprof").join("c.toml");
    std::fs::write(&p, "profile = \"turbo\"\n").unwrap();
    let e = load(Some(p.to_str().unwrap())).unwrap_err();
    let fe = e.downcast_ref::<fle_core::FleError>().unwrap();
    assert!(
        fe.message.contains("Unknown profile 'turbo'")
            && fe.hint.contains("off, auto, balanced, max, nvme, debug")
    );
}

#[test]
fn precedence_is_cli_then_environment_then_file_then_defaults() {
    let _g = ENV.lock().unwrap();
    clear_env();
    let p = tmp("prec").join("c.toml");
    std::fs::write(&p, "port = 9001\nprofile = \"balanced\"\nthreads = 4\n[env]\nFLE_X = \"file\"\nFLE_Y = \"file\"\n").unwrap();
    let data = load_at(&p);
    let e = effective(&Default::default(), &data).unwrap();
    assert_eq!(
        (
            e["port"].as_i64(),
            e["profile"].as_str(),
            e["threads"].as_i64()
        ),
        (Some(9001), Some("balanced"), Some(4))
    );
    std::env::set_var("FLE_PORT", "9002");
    std::env::set_var("FLE_PROFILE", "max");
    std::env::set_var("FLE_MTP", "0");
    let e = effective(&Default::default(), &data).unwrap();
    assert_eq!(
        (
            e["port"].as_i64(),
            e["profile"].as_str(),
            e["mtp"].as_bool(),
            e["threads"].as_i64()
        ),
        (Some(9002), Some("max"), Some(false), Some(4))
    );
    let mut cli = serde_json::Map::new();
    cli.insert("port".into(), json!(9003));
    cli.insert("env".into(), json!({"FLE_Y": "cli", "FLE_Z": "cli"}));
    let e = effective(&cli, &data).unwrap();
    assert_eq!(e["port"], 9003);
    assert_eq!(
        e["env"],
        json!({"FLE_X": "file", "FLE_Y": "cli", "FLE_Z": "cli"})
    ); // env tables merge, later layers win
    clear_env();
}

#[test]
fn bad_profile_from_the_environment_is_refused() {
    let _g = ENV.lock().unwrap();
    clear_env();
    std::env::set_var("FLE_PROFILE", "bogus");
    let e = effective(&Default::default(), &defaults()).unwrap_err();
    assert!(e.to_string().contains("Unknown profile 'bogus'"));
    clear_env();
}

#[test]
fn save_and_load_round_trip_and_dotted_keys_work() {
    let _g = ENV.lock().unwrap();
    clear_env();
    let p = tmp("rt").join("sub").join("c.toml");
    let mut d = defaults();
    set_dotted(&mut d, "port", json!(9100));
    set_dotted(&mut d, "model", json!("/mödel/ü.gguf"));
    set_dotted(&mut d, "env.FLE_B", json!("2"));
    set_dotted(&mut d, "env.FLE_A", json!("1"));
    save(&d, Some(p.to_str().unwrap())).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert!(
        text.contains("model = \"/m\\u00f6del/\\u00fc.gguf\""),
        "{text}"
    );
    assert!(
        text.find("FLE_A").unwrap() < text.find("FLE_B").unwrap(),
        "env keys are sorted"
    );
    assert_eq!(load_at(&p)["model"], "/mödel/ü.gguf");
    unset_dotted(&mut d, "env.FLE_A");
    unset_dotted(&mut d, "nothing.here"); // not an error
    assert_eq!(d["env"], json!({"FLE_B": "2"}));
}

#[test]
fn set_value_conversion_matches_the_cli() {
    assert_eq!(parse_value("TRUE"), json!(true));
    assert_eq!(parse_value("8080"), json!(8080));
    assert_eq!(parse_value("0.5"), json!(0.5));
    assert_eq!(parse_value("auto"), json!("auto"));
}

fn py(args: &[&str], envs: &[(&str, &str)]) -> Option<(String, bool)> {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../fle");
    if !helper.exists() {
        return None;
    }
    let mut c = std::process::Command::new("python3");
    c.arg(&helper).arg("config").args(args);
    for (k, v) in envs {
        c.env(k, v);
    }
    let o = c.output().ok()?;
    Some((
        String::from_utf8_lossy(&o.stdout).replace("\r\n", "\n"),
        o.status.success(),
    ))
}

/// Same operations through the Python helper and through this library: the saved files and the effective settings agree.
#[test]
fn matches_the_python_helper() {
    let _g = ENV.lock().unwrap();
    clear_env();
    let (d1, d2) = (tmp("py-rs"), tmp("py-py"));
    let (rs, pf) = (d1.join("c.toml"), d2.join("c.toml"));
    let mut data = defaults();
    for (k, v) in [
        ("profile", "max"),
        ("port", "9000"),
        ("mtp", "false"),
        ("model", "/mödel.gguf"),
        ("context", "4096.7"),
        ("env.FLE_MC_OVERLAP", "3"),
        ("env.A", "b"),
    ] {
        let val = if k.starts_with("env.") {
            json!(v)
        } else {
            parse_value(v)
        };
        set_dotted(&mut data, k, val);
        if py(&["--config", pf.to_str().unwrap(), "set", k, v], &[]).is_none() {
            eprintln!("python helper not found: skipped");
            return;
        }
    }
    // Python re-reads the file for every set, so context 4096.7 became 4096 there; do the same here before saving
    let (data_rs, _) = {
        save(&data, Some(rs.to_str().unwrap())).unwrap();
        load(Some(rs.to_str().unwrap())).unwrap()
    };
    save(&data_rs, Some(rs.to_str().unwrap())).unwrap();
    // Python's write_text writes CRLF on Windows; both are valid TOML
    let text = |p: &Path| std::fs::read_to_string(p).unwrap().replace("\r\n", "\n");
    assert_eq!(text(&rs), text(&pf), "the saved files differ");
    for envs in [
        vec![],
        vec![
            ("FLE_PORT", "9111"),
            ("FLE_PROFILE", "max"),
            ("FLE_MTP", "0"),
            ("FLE_THREADS", "3.9"),
        ],
    ] {
        clear_env();
        for (k, v) in &envs {
            std::env::set_var(k, v);
        }
        let eff = effective(&Default::default(), &load_at(&rs)).unwrap();
        let mine =
            fle_core::gguf::py_json(&json!({"path": rs.to_string_lossy(), "settings": eff})) + "\n";
        let (theirs, ok) =
            py(&["--config", pf.to_str().unwrap(), "--json", "show"], &envs).unwrap();
        assert!(ok);
        assert_eq!(
            mine.replace(rs.to_str().unwrap(), "X"),
            theirs.replace(pf.to_str().unwrap(), "X"),
            "show --json differs (env {envs:?})"
        );
    }
    clear_env();
}

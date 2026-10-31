//! One read-only setup check: model, engine, memory, disk and the command that would start.
//! Port of `tools/fle/doctor.py`; the Python helper is the specification.
use crate::gguf::{self, human_bytes};
use crate::launch::{self, human_int, shquote, Host, Options, MIB};
use crate::FleError;
use serde_json::{json, Map, Value};
use std::path::PathBuf;

fn check(name: &str, status: &str, detail: impl Into<String>, hint: &str) -> Value {
    json!({"name": name, "status": status, "detail": detail.into(), "hint": hint})
}

fn rank(s: &str) -> u8 {
    match s {
        "warn" => 1,
        "fail" => 2,
        _ => 0,
    }
}

pub fn worst(checks: &[Value]) -> &'static str {
    let mut w = 0;
    for c in checks {
        w = w.max(rank(c["status"].as_str().unwrap_or("ok")));
    }
    ["ok", "warn", "fail"][w as usize]
}

/// Another profile that would resolve a warning or failure, as a ready command. At most one per profile.
pub fn suggest(checks: &[Value], profile: &str, model_path: &str, dense: bool) -> Vec<Value> {
    if model_path.is_empty() {
        return vec![];
    }
    let mut found: Vec<(String, String)> = vec![];
    let mut add = |target: &str, reason: &str| {
        if target != profile && !found.iter().any(|(t, _)| t == target) {
            found.push((target.into(), reason.into()));
        }
    };
    for c in checks {
        let (name, status, detail) = (
            c["name"].as_str().unwrap_or(""),
            c["status"].as_str().unwrap_or(""),
            c["detail"].as_str().unwrap_or(""),
        );
        let hint = c["hint"].as_str().unwrap_or("");
        if name == "RAM" && status == "warn" {
            add(
                "nvme",
                "the experts are larger than the free RAM; nvme memory-maps them from disk",
            );
        } else if name == "plan warning" && detail.contains("larger than RAM plus VRAM") {
            if dense {
                add("nvme", "the model is larger than RAM plus VRAM; nvme memory-maps it from disk (slow for a dense model)");
            } else {
                add(
                    "nvme",
                    "the model is larger than RAM plus VRAM; nvme streams experts from disk",
                );
            }
        } else if name == "GPU memory"
            && status == "warn"
            && matches!(profile, "auto" | "max" | "debug")
        {
            add(
                "balanced",
                "little VRAM is free; balanced keeps a bigger margin and a smaller cache",
            );
        } else if name == "GPU" && status == "fail" {
            add(
                "off",
                "no NVIDIA GPU was found; the off profile runs a plain upstream llama.cpp",
            );
        } else if (name == "engine flags" || name == "plan")
            && (status == "warn" || status == "fail")
            && format!("{detail}{hint}").contains("not an FLE engine")
        {
            add(
                "off",
                "the engine is not an FLE build; the off profile needs no FLE features",
            );
        }
    }
    let q = shquote(model_path);
    found.into_iter().map(|(t, r)| json!({"profile": t, "reason": r, "command": format!("fle plan {q} --profile {t}")})).collect()
}

/// `hardware.check_engine_flags`: problems with the engine for this profile.
pub fn check_engine_flags(
    engine: &Map<String, Value>,
    profile: &str,
    moe: Option<bool>,
) -> Vec<String> {
    let b = |k: &str| engine.get(k).and_then(Value::as_bool).unwrap_or(false);
    if !b("ok") {
        let p = engine.get("problem").and_then(Value::as_str).unwrap_or("");
        return vec![format!(
            "The engine does not start: {}.",
            if p.is_empty() { "unknown error" } else { p }
        )];
    }
    let mut out = vec![];
    if profile != "off" {
        if engine.get("flavor").and_then(Value::as_str) != Some("fle") {
            out.push("This is not an FLE engine; the FLE cache and profile settings will be ignored or rejected.".into());
        }
        if !b("has_fit_target") {
            out.push("This engine has no --fit-target; use a newer FLE build.".into());
        }
        if !b("has_cmoe") && moe != Some(false) {
            out.push("This engine has no -cmoe; experts would not stay on the CPU.".into());
        }
    }
    out
}

fn py_list(v: &[u32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn fle_msg(e: &anyhow::Error) -> (String, String) {
    match e.downcast_ref::<FleError>() {
        Some(f) => (f.message.clone(), f.hint.clone()),
        None => (e.to_string(), String::new()),
    }
}

pub fn run(
    model_path: &str,
    settings: &Map<String, Value>,
    host: &Host,
    system: &Value,
    installed: &[PathBuf],
) -> Value {
    let mut checks: Vec<Value> = vec![];
    let (mut model, mut verification, mut parsed): (
        Option<gguf::Model>,
        Option<Value>,
        Option<launch::Plan>,
    ) = (None, None, None);
    let profile = match settings.get("profile") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => "auto".to_string(),
    };
    if model_path.is_empty() {
        checks.push(check(
            "model",
            "warn",
            "no model selected",
            "Pass a model, or set model in `fle config`.",
        ));
    } else {
        match gguf::read_model(std::path::Path::new(model_path)) {
            Ok(m) => {
                let v = gguf::verify_model(&m);
                let ok = v["ok"].as_bool().unwrap_or(false);
                checks.push(check(
                    "model",
                    if ok { "ok" } else { "fail" },
                    format!("{} ({}, {})", m.name, m.arch, human_bytes(m.size)),
                    if ok {
                        ""
                    } else {
                        v["errors"][0].as_str().unwrap_or("")
                    },
                ));
                if !m.complete() {
                    let mut d = format!("{} part(s) missing", m.missing.len());
                    if !m.incomplete.is_empty() {
                        d += &format!(", {} incomplete", m.incomplete.len());
                    }
                    checks.push(check(
                        "model parts",
                        "fail",
                        d,
                        "Wait for the download to finish.",
                    ));
                } else {
                    checks.push(check(
                        "model parts",
                        "ok",
                        format!("{} part(s) present", m.parts.len()),
                        "",
                    ));
                }
                for w in v["warnings"].as_array().into_iter().flatten() {
                    checks.push(check("model warning", "warn", w.as_str().unwrap_or(""), ""));
                }
                if !m.unknown_types.is_empty() {
                    checks.push(check(
                        "tensor types",
                        "fail",
                        format!("unknown type ids {}", py_list(&m.unknown_types)),
                        "This engine cannot load the file; use the engine that wrote it.",
                    ));
                } else {
                    checks.push(check(
                        "tensor types",
                        "ok",
                        format!("{} known types", m.type_histogram.len()),
                        "",
                    ));
                }
                let side = &v["sidecars"];
                let (h, o) = (side["mtp_head"].is_string(), side["mtp_vocab"].is_string());
                if h && o {
                    checks.push(check("MTP", "ok", "head and vocabulary found", ""));
                } else if h || o {
                    checks.push(check(
                        "MTP",
                        "warn",
                        "head or vocabulary is missing",
                        "Download both files, or turn MTP off.",
                    ));
                } else {
                    checks.push(check("MTP", "ok", "off (no MTP files)", ""));
                }
                verification = Some(v);
                model = Some(m);
            }
            Err(e) => {
                let (m, h) = fle_msg(&e);
                checks.push(check("model", "fail", m, &h));
            }
        }
    }

    let eng = settings.get("engine").and_then(Value::as_str).unwrap_or("");
    let engine_path = if eng.is_empty() {
        launch::find_engine("", installed)
    } else {
        eng.to_string()
    };
    let mut engine: Option<Map<String, Value>> = None;
    if engine_path.is_empty() {
        checks.push(check(
            "engine",
            "fail",
            "no llama-server found",
            "Set engine in `fle config`, or pass --engine.",
        ));
    } else {
        let e = launch::detect_engine(&engine_path);
        if e.get("ok") == Some(&json!(true)) {
            checks.push(check(
                "engine",
                "ok",
                format!(
                    "{} ({}, {})",
                    e["path"].as_str().unwrap_or(""),
                    e["flavor"].as_str().unwrap_or(""),
                    e["backend"].as_str().unwrap_or("")
                ),
                "",
            ));
            let moe = model.as_ref().map(|m| m.experts().unwrap_or(0) > 0);
            for p in check_engine_flags(&e, &profile, moe) {
                checks.push(check(
                    "engine flags",
                    if profile != "off" { "fail" } else { "warn" },
                    p,
                    "",
                ));
            }
        } else {
            let p = e.get("problem").and_then(Value::as_str).unwrap_or("");
            checks.push(check(
                "engine",
                "fail",
                if p.is_empty() { "does not start" } else { p },
                "Check the path.",
            ));
        }
        engine = Some(e);
    }

    let free: u64 = host.gpus.iter().map(|g| g.memory_free).sum();
    if profile != "off" {
        if host.gpus.is_empty() {
            checks.push(check(
                "GPU",
                "fail",
                "no NVIDIA GPU found",
                "Use an upstream CPU build with --profile off.",
            ));
        } else {
            checks.push(check(
                "GPU",
                "ok",
                format!("{} GPU(s), {} free", host.gpus.len(), human_bytes(free)),
                "",
            ));
            if let Some(m) = &model {
                let kv = (m.kv_bytes_per_token("q8_0", "q8_0")
                    * launch::context(&profile, m, 0) as f64) as i64;
                let need = m.gpu_bytes as i64 + kv + (launch::ENGINE_OVERHEAD_MIB * MIB) as i64;
                if need > free as i64 {
                    let has = m.experts().unwrap_or(0) > 0;
                    let (d, h) = if has {
                        (format!("non-expert weights and KV need about {}, {} free", human_bytes(need as u64), human_bytes(free)), "Stop other GPU work, lower the context, or use --profile balanced/nvme.")
                    } else {
                        (
                            format!(
                                "weights and KV need about {}, {} free",
                                human_bytes(need as u64),
                                human_bytes(free)
                            ),
                            "Stop other GPU work, lower the context, or use --profile balanced.",
                        )
                    };
                    checks.push(check("GPU memory", "warn", d, h));
                } else {
                    checks.push(check(
                        "GPU memory",
                        "ok",
                        format!(
                            "{} left for the cache",
                            human_bytes((free as i64 - need) as u64)
                        ),
                        "",
                    ));
                }
            }
        }
    }
    if let Some(m) = &model {
        if m.expert_bytes > 0 {
            let warn = m.expert_bytes > host.mem_available;
            checks.push(check(
                "RAM",
                if warn { "warn" } else { "ok" },
                format!(
                    "experts {}, available {}",
                    human_bytes(m.expert_bytes),
                    human_bytes(host.mem_available)
                ),
                if warn {
                    "Close other programs, or use --profile nvme."
                } else {
                    ""
                },
            ));
        }
        let disk = system["disk_free"].as_u64().unwrap_or(0);
        if disk < m.size {
            checks.push(check(
                "disk",
                "warn",
                format!(
                    "{} free for a {} model",
                    human_bytes(disk),
                    human_bytes(m.size)
                ),
                "",
            ));
        }
        let o = Options {
            installed_engines: installed.to_vec(),
            ..Default::default()
        };
        match launch::build(settings, m, &o, host) {
            Ok(p) => {
                checks.push(check(
                    "plan",
                    "ok",
                    format!(
                        "{}, spec {}, {} MiB cache, {} tokens",
                        p.profile,
                        p.spec,
                        p.cache_mib,
                        human_int(p.context)
                    ),
                    "",
                ));
                for w in &p.warnings {
                    checks.push(check("plan warning", "warn", w.clone(), ""));
                }
                parsed = Some(p);
            }
            Err(e) => {
                let (m, h) = fle_msg(&e);
                checks.push(check("plan", "fail", m, &h));
            }
        }
    } else {
        checks.push(check(
            "plan",
            "warn",
            "no model to plan for",
            "Pass a model or set model in `fle config`.",
        ));
    }
    let status = worst(&checks);
    let dense = model
        .as_ref()
        .is_some_and(|m| m.experts().unwrap_or(0) == 0);
    let suggestions = suggest(&checks, &profile, model_path, dense);
    json!({
        "ok": status != "fail", "status": status, "profile": profile, "config": crate::config::config_path(None).to_string_lossy(),
        "model": model.as_ref().map(|m| {
            let mut s = m.summary();
            if let Some(o) = s.as_object_mut() { o.shift_remove("sidecars"); }
            s
        }),
        "verification": verification, "engine": engine, "plan": parsed.as_ref().map(|p| p.as_json()), "system": system,
        "checks": checks, "suggestions": suggestions,
    })
}

pub fn print_report(r: &Value) -> String {
    let st = r["status"].as_str().unwrap_or("");
    let mut s = format!("fle doctor: {st}\n");
    for c in r["checks"].as_array().into_iter().flatten() {
        let mark = match c["status"].as_str().unwrap_or("") {
            "ok" => "ok  ",
            "warn" => "warn",
            _ => "fail",
        };
        s += &format!(
            "[{mark}] {}: {}\n",
            c["name"].as_str().unwrap_or(""),
            c["detail"].as_str().unwrap_or("")
        );
        if let Some(h) = c["hint"].as_str().filter(|h| !h.is_empty()) {
            s += &format!("       -> {h}\n");
        }
    }
    let sys = &r["system"];
    let bytes = |v: &Value| v.as_u64().map(human_bytes).unwrap_or_else(|| "?".into());
    s += &format!(
        "\nsystem: {} / {}\n",
        sys["os"].as_str().unwrap_or(""),
        sys["runtime"].as_str().unwrap_or("fle")
    );
    let cpu = &sys["cpu"];
    s += &format!(
        "cpu:    {} ({} cores, {} threads)\n",
        cpu["name"].as_str().unwrap_or(""),
        cpu["physical"],
        cpu["logical"]
    );
    s += &format!("ram:    {}\n", bytes(&sys["memory"]["total"]));
    for g in sys["gpus"].as_array().into_iter().flatten() {
        s += &format!(
            "gpu {}: {} {} free of {}\n",
            g["index"],
            g["name"].as_str().unwrap_or(""),
            bytes(&g["memory_free"]),
            bytes(&g["memory_total"])
        );
    }
    s += &format!("disk:   {} free\n", bytes(&sys["disk_free"]));
    if let Some(m) = r["model"].as_object() {
        let kind = match m["experts"].as_u64() {
            Some(n) if n > 0 => format!("{} experts, ", human_int(n as i64)),
            _ => "dense, ".into(),
        };
        s += &format!(
            "model:  {} ({}, {}{}, {} tensors)\n",
            m["name"].as_str().unwrap_or(""),
            m["arch"].as_str().unwrap_or(""),
            kind,
            m["size_human"].as_str().unwrap_or(""),
            m["tensor_count"]
        );
    }
    s += &format!("config: {}\n", r["config"].as_str().unwrap_or(""));
    let sg = r["suggestions"].as_array().cloned().unwrap_or_default();
    if !sg.is_empty() {
        s += &format!(
            "\n{}\n",
            if sg.len() == 1 {
                "suggestion:"
            } else {
                "suggestions:"
            }
        );
        for i in &sg {
            s += &format!(
                "  {}\n      {}\n",
                i["command"].as_str().unwrap_or(""),
                i["reason"].as_str().unwrap_or("")
            );
        }
    }
    s
}

/// What `hardware.summary` reports, read from the real machine.
pub fn system_summary(model_path: &str, host: &Host) -> Value {
    let probe = {
        let mut p = if model_path.is_empty() {
            dirs::home_dir().unwrap_or_default()
        } else {
            PathBuf::from(model_path)
        };
        while !p.exists() && p.parent().is_some_and(|q| q != p) {
            p = p.parent().unwrap().to_path_buf();
        }
        p
    };
    let dir = if probe.is_dir() {
        probe.clone()
    } else {
        probe
            .parent()
            .map(|q| q.to_path_buf())
            .unwrap_or(probe.clone())
    };
    let info = crate::system::detect(&dir);
    let os = match std::env::consts::OS {
        "macos" => format!(
            "macOS {}",
            launch::run_text("sw_vers", &["-productVersion"])
        ),
        "windows" => "Windows".to_string(),
        _ => std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|t| {
                t.lines().find_map(|l| {
                    l.strip_prefix("PRETTY_NAME=")
                        .map(|v| v.trim_matches('"').to_string())
                })
            })
            .unwrap_or_else(|| "Linux".into()),
    };
    let gpus: Vec<Value> = host.gpus.iter().map(|g| json!({"index": g.index, "name": g.name, "memory_total": g.memory_total, "memory_used": g.memory_used, "memory_free": g.memory_free, "vendor": "nvidia"})).collect();
    json!({
        "os": os, "platform": match std::env::consts::OS { "macos" => "macos", "windows" => "windows", _ => "linux" },
        "runtime": format!("fle {}", env!("CARGO_PKG_VERSION")),
        "cpu": {"name": if info.cpu.is_empty() { "CPU".to_string() } else { info.cpu.clone() }, "physical": host.cpu_physical, "logical": host.cpu_logical},
        "memory": {"total": host.mem_total, "available": host.mem_available}, "memory_human": human_bytes(host.mem_total),
        "gpus": gpus, "disk_free": info.disk_free_bytes, "disk_free_human": human_bytes(info.disk_free_bytes),
    })
}

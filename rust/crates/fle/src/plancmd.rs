//! `fle plan`: the settings merge and the plan, as `_settings`, `_plan_for` and `_cmd_plan` in tools/fle/cli.py.
use anyhow::Result;
use clap::Args;
use fle_core::config;
use fle_core::launch::{self, Host, Options, Plan};
use fle_core::{gguf, FleError};
use serde_json::{json, Map, Value};
use std::path::PathBuf;

#[derive(Args, Clone, Default)]
pub struct Common {
    /// model file, first shard, or folder
    pub model: Option<String>,
    /// machine-readable output
    #[arg(long)]
    pub json: bool,
    /// use another config file
    #[arg(long, value_name = "FILE")]
    pub config: Option<String>,
    /// path to llama-server
    #[arg(long, value_name = "FILE")]
    pub engine: Option<String>,
    /// off, auto, balanced, max, nvme, debug
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(config::PROFILES))]
    pub profile: Option<String>,
    /// accepted for compatibility; this program prints no colour
    #[arg(long, value_name = "WHEN", hide = true)]
    pub color: Option<String>,
    #[arg(long, hide = true)]
    pub debug: bool,
}

#[derive(Args, Clone, Default)]
pub struct PlanArgs {
    #[command(flatten)]
    pub common: Common,
    /// print the start command as before (environment line, then one long command line)
    #[arg(long)]
    pub oneline: bool,
    /// listen address (default 127.0.0.1)
    #[arg(long)]
    pub host: Option<String>,
    /// listen port (default 8080)
    #[arg(long)]
    pub port: Option<i64>,
    /// context tokens (0 = profile default)
    #[arg(long)]
    pub context: Option<i64>,
    /// CPU threads (0 = profile default)
    #[arg(long)]
    pub threads: Option<i64>,
    /// GPU expert cache in MiB (0 = automatic)
    #[arg(long = "cache-mib")]
    pub gpu_cache_mib: Option<i64>,
    /// free VRAM beside the cache in MiB
    #[arg(long = "fit-target")]
    pub fit_target_mib: Option<i64>,
    /// experts in pinned RAM
    #[arg(long = "pinned", value_parser = ["auto", "on", "off"])]
    pub pinned_load: Option<String>,
    /// speculation: auto, none, mtp, ngram-mod or ngram-simple (default: auto)
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(config::SPECS))]
    pub spec: Option<String>,
    /// do not use an MTP head even when it is there
    #[arg(long)]
    pub no_mtp: bool,
    /// MTP head GGUF
    #[arg(long, value_name = "FILE")]
    pub mtp_head: Option<String>,
    /// MTP vocabulary text file
    #[arg(long, value_name = "FILE")]
    pub mtp_vocab: Option<String>,
    /// CUDA_VISIBLE_DEVICES, for example 0 or 0,1
    #[arg(long = "gpu", value_name = "IDS")]
    pub gpu_ids: Option<String>,
    /// extra llama-server argument
    #[arg(long = "arg", value_name = "ARG", allow_hyphen_values = true)]
    pub extra_arg: Vec<String>,
    /// extra engine environment
    #[arg(long, value_name = "KEY=VALUE")]
    pub env: Vec<String>,
    /// if the FLE plan cannot be built, retry with an installed upstream llama.cpp and profile off
    #[arg(long)]
    pub fallback: bool,
}

/// `_settings`: config file < environment < the flags that were given.
pub fn settings(a: &PlanArgs) -> Result<Map<String, Value>> {
    let (data, path) = config::load(a.common.config.as_deref())?;
    let mut cli = Map::new();
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            cli.insert(k.into(), v);
        }
    };
    put("profile", a.common.profile.clone().map(Value::from));
    put("spec", a.spec.clone().map(Value::from));
    put("engine", a.common.engine.clone().map(Value::from));
    put("host", a.host.clone().map(Value::from));
    put("port", a.port.map(Value::from));
    put("context", a.context.map(Value::from));
    put("threads", a.threads.map(Value::from));
    put("gpu_cache_mib", a.gpu_cache_mib.map(Value::from));
    put("fit_target_mib", a.fit_target_mib.map(Value::from));
    put("pinned_load", a.pinned_load.clone().map(Value::from));
    if a.no_mtp {
        put("mtp", Some(json!(false)));
    }
    let mut s = config::effective(&cli, &data)?;
    s.insert("config_path".into(), json!(path.to_string_lossy()));
    Ok(s)
}

pub fn model_path(
    model: &Option<String>,
    s: &Map<String, Value>,
    required: bool,
) -> Result<String> {
    let p = model.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| {
        s.get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    });
    if p.is_empty() && required {
        return Err(FleError::new(
            "No model given.",
            "Pass MODEL, or run `fle config set model /path/to/model.gguf`.",
        )
        .into());
    }
    Ok(p)
}

fn env_pairs(v: &[String]) -> Result<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = vec![];
    for item in v {
        match item.split_once('=') {
            Some((k, val)) if !k.trim().is_empty() => {
                match out.iter_mut().find(|(a, _)| a == k.trim()) {
                    Some(e) => e.1 = val.to_string(),
                    None => out.push((k.trim().to_string(), val.to_string())),
                }
            }
            _ => {
                return Err(
                    FleError::new(format!("--env needs KEY=VALUE, got '{item}'"), "").into(),
                )
            }
        }
    }
    Ok(out)
}

fn installed(home: &std::path::Path) -> Vec<PathBuf> {
    fle_core::engines::installed(home)
        .into_iter()
        .filter_map(|(_, d)| fle_core::engines::find_server(&d))
        .collect()
}

fn options(a: &PlanArgs, model: &gguf::Model, home: &std::path::Path) -> Result<Options> {
    Ok(Options {
        engine_path: String::new(),
        model_path: model.first.to_string_lossy().to_string(),
        extra_args: a.extra_arg.clone(),
        env_override: env_pairs(&a.env)?,
        gpu_ids: a.gpu_ids.clone().unwrap_or_default(),
        no_mtp: a.no_mtp,
        mtp_head: a.mtp_head.clone().unwrap_or_default(),
        mtp_vocab: a.mtp_vocab.clone().unwrap_or_default(),
        spec: a.spec.clone().unwrap_or_default(),
        installed_engines: installed(home),
    })
}

fn upstream_engine(explicit: &str, home: &std::path::Path) -> String {
    let mut c: Vec<String> = vec![];
    if !explicit.is_empty() {
        c.push(explicit.into());
    }
    if let Ok(e) = std::env::var("FLE_UPSTREAM_ENGINE") {
        if !e.trim().is_empty() {
            c.push(e.trim().into());
        }
    }
    if let Ok(e) = std::env::var("FLE_ENGINE") {
        if !e.trim().is_empty() {
            c.push(e.trim().into());
        }
    }
    if let Some(p) = launch::which("llama-server") {
        c.push(p.to_string_lossy().into());
    }
    c.extend(
        installed(home)
            .iter()
            .map(|p| p.to_string_lossy().to_string()),
    );
    for cand in c {
        if !std::path::Path::new(&cand).is_file() {
            continue;
        }
        let i = launch::detect_engine(&cand);
        if i.get("ok") == Some(&json!(true)) && i.get("flavor") == Some(&json!("upstream")) {
            return cand;
        }
    }
    String::new()
}

/// `_plan_for`: read the model, build the plan, and with --fallback retry on a plain llama.cpp.
pub fn plan_for(
    a: &PlanArgs,
    s: &Map<String, Value>,
    path: &str,
    home: &std::path::Path,
    host: &Host,
) -> Result<Plan> {
    let model = gguf::read_model(std::path::Path::new(path))?;
    let mut o = options(a, &model, home)?;
    o.engine_path = s
        .get("engine")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    match launch::build(s, &model, &o, host) {
        Ok(p) => Ok(p),
        Err(e) => {
            let Some(fe) = e.downcast_ref::<FleError>().cloned() else {
                return Err(e);
            };
            let spec = a
                .spec
                .clone()
                .filter(|x| !x.is_empty())
                .or_else(|| {
                    s.get("spec")
                        .and_then(Value::as_str)
                        .filter(|x| !x.is_empty())
                        .map(String::from)
                })
                .unwrap_or_else(|| "auto".into())
                .to_lowercase();
            if !a.fallback || (spec != "auto" && spec != "none") {
                return Err(e);
            }
            let reason = format!("FLE plan failed: {fe}");
            let up = upstream_engine(&o.engine_path, home);
            if up.is_empty() {
                return Err(FleError::new(reason, "No upstream llama.cpp was found. Install one, set FLE_UPSTREAM_ENGINE, or pass --engine /path/to/llama-server.").into());
            }
            let mut fb = s.clone();
            for (k, v) in [
                ("profile", json!("off")),
                ("spec", json!("none")),
                ("engine", json!(up)),
                ("mtp", json!(false)),
            ] {
                fb.insert(k.into(), v);
            }
            let mut o2 = o.clone();
            o2.engine_path = up;
            o2.no_mtp = true;
            o2.spec = "none".into();
            let mut p = launch::build(&fb, &model, &o2, host)?;
            p.fallback = true;
            p.notes.insert(0, reason);
            p.notes.insert(
                1,
                "Falling back to upstream llama.cpp with profile off; FLE features are disabled."
                    .into(),
            );
            Ok(p)
        }
    }
}

pub fn run(a: PlanArgs, home: &std::path::Path) -> Result<()> {
    let s = settings(&a)?;
    let path = model_path(&a.common.model, &s, true)?;
    let plan = plan_for(&a, &s, &path, home, &Host::detect())?;
    if a.common.json {
        println!("{}", gguf::py_json(&plan.as_json()));
    } else {
        print!("{}", plan.text(a.oneline));
    }
    Ok(())
}

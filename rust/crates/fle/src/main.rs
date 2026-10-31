mod tui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use fle_core::{catalog, chat, data_dir, engines, flow, system, Event};
use std::io::Write;

mod plancmd;

#[derive(Parser)]
#[command(
    name = "fle",
    version,
    about = "Run large language models locally. No flags, no variables: just `fle`."
)]
struct Cli {
    /// Keep engines, models and logs in this folder (default: per-user app data)
    #[arg(long, global = true)]
    home: Option<std::path::PathBuf>,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// print the config path
    Path,
    /// write a default config
    Init {
        #[arg(long)]
        force: bool,
    },
    /// show the effective settings
    Show,
    /// set KEY VALUE
    Set { key: String, value: String },
    /// remove KEY
    Unset { key: String },
    /// open the file in $EDITOR
    Edit,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show or change the global settings (~/.config/fle/config.toml)
    Config {
        /// use another config file
        #[arg(long, global = true, value_name = "FILE")]
        config: Option<String>,
        #[arg(long, global = true)]
        json: bool,
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// What this machine can do, and which engine fits it
    Doctor,
    /// The models FLE knows, and whether each fits this machine
    Models,
    /// Load a model and keep serving it (OpenAI-compatible API) until Ctrl+C
    Run { model: String },
    /// Read a model's header: architecture, parts, expert bytes, KV cache (no weights are loaded)
    Inspect {
        /// model file, first shard, or the folder that contains the shards
        model: std::path::PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Check parts, tensor count, offsets and tensor types of a model file
    Verify {
        model: std::path::PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Show the llama-server command and the numbers behind it
    Plan(Box<plancmd::PlanArgs>),
    /// Ask one question and print the answer (loads the model for the duration)
    Ask {
        model: String,
        question: Vec<String>,
    },
}

fn print_event(e: &Event) {
    match e {
        Event::Step(s) => eprintln!("\n==> {s}"),
        Event::Progress { label, done, total } => {
            let pct = if *total > 0 {
                *done as f64 * 100.0 / *total as f64
            } else {
                0.0
            };
            eprint!(
                "\r    {label}: {pct:5.1} %  ({} / {})   ",
                system::gib(*done),
                system::gib(*total)
            );
            let _ = std::io::stderr().flush();
        }
        Event::Log(l) => eprint!("\r    {l:<110}"),
        Event::Ready {
            base,
            engine,
            model,
            command,
        } => {
            eprintln!("\n\nReady: {model} on {engine}\n  API: {base}/v1   (OpenAI-compatible, no key needed)\n  Command (shown, never needed):\n    {command}");
        }
    }
}

fn main() {
    if let Err(e) = run() {
        // same words as the Python helper: "fle: message", then "hint: ..."; exit code 2
        if let Some(fe) = e.downcast_ref::<fle_core::FleError>() {
            eprintln!("fle: {fe}");
            if !fe.hint.is_empty() {
                eprintln!("hint: {}", fe.hint);
            }
        } else {
            eprintln!("fle: {e:#}");
        }
        std::process::exit(2);
    }
}

fn run_config(config: Option<String>, json: bool, action: ConfigAction) -> Result<()> {
    use fle_core::{config, FleError};
    let path = config::config_path(config.as_deref());
    match action {
        ConfigAction::Path => println!("{}", path.display()),
        ConfigAction::Init { force } => {
            if path.exists() && !force {
                return Err(FleError::new(
                    format!("{} already exists", path.display()),
                    "Use --force to replace it.",
                )
                .into());
            }
            let (data, _) = config::load(config.as_deref())?;
            println!("{}", config::save(&data, config.as_deref())?.display());
        }
        ConfigAction::Edit => {
            if !path.exists() {
                let (data, _) = config::load(config.as_deref())?;
                config::save(&data, config.as_deref())?;
            }
            let editor = std::env::var("VISUAL")
                .or_else(|_| std::env::var("EDITOR"))
                .unwrap_or_else(|_| "vi".into());
            let st = std::process::Command::new(&editor).arg(&path).status()?;
            std::process::exit(st.code().unwrap_or(1));
        }
        ConfigAction::Show => {
            let (data, resolved) = config::load(config.as_deref())?;
            let settings = config::effective(&serde_json::Map::new(), &data)?;
            if json {
                let v =
                    serde_json::json!({"path": resolved.to_string_lossy(), "settings": settings});
                println!("{}", fle_core::gguf::py_json(&v));
            } else {
                print!("{}", config::show_text(&resolved, &settings));
            }
        }
        ConfigAction::Set { key, value } => {
            let (mut data, _) = config::load(config.as_deref())?;
            if key == "profile" && !config::PROFILES.contains(&value.as_str()) {
                return Err(FleError::new(
                    format!("Unknown profile '{value}'"),
                    format!("Use one of: {}.", config::PROFILES.join(", ")),
                )
                .into());
            }
            let v = if key.starts_with("env.") {
                serde_json::Value::String(value)
            } else {
                config::parse_value(&value)
            };
            config::set_dotted(&mut data, &key, v);
            println!("{}", config::save(&data, config.as_deref())?.display());
        }
        ConfigAction::Unset { key } => {
            let (mut data, _) = config::load(config.as_deref())?;
            config::unset_dotted(&mut data, &key);
            println!("{}", config::save(&data, config.as_deref())?.display());
        }
    }
    Ok(())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let home = cli.home.unwrap_or_else(data_dir);
    match cli.cmd {
        None => tui::run(&home),
        Some(Cmd::Config {
            config,
            json,
            action,
        }) => run_config(config, json, action),
        Some(Cmd::Plan(a)) => plancmd::run(*a, &home),
        Some(Cmd::Doctor) => {
            let s = system::detect(&home);
            println!("OS        {} {}\nCPU       {} ({} cores)\nMemory    {}\nGPU       {}\nDisk free {} (in {})", s.os, s.arch, s.cpu, s.cores, system::gib(s.ram_bytes),
                if s.gpus.is_empty() { "none".into() } else { s.gpus.iter().map(|g| g.name.clone()).collect::<Vec<_>>().join(", ") }, system::gib(s.disk_free_bytes), home.display());
            if s.gpu_usable_bytes > 0 {
                println!("GPU usable {}", system::gib(s.gpu_usable_bytes));
            }
            println!("\nEngines that fit (best first):");
            for v in engines::variants(&s) {
                println!("  {:<42} {}", v.label, v.why);
            }
            let inst = engines::installed(&home);
            println!(
                "\nInstalled: {}",
                if inst.is_empty() {
                    "nothing yet (the first run installs the best one)".into()
                } else {
                    inst.iter()
                        .map(|(n, _)| n.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            );
            Ok(())
        }
        Some(Cmd::Models) => {
            let s = system::detect(&home);
            for m in catalog::builtin() {
                let f = catalog::fit(&m, &s);
                let have = flow::model_path(&home, &m).exists();
                println!(
                    "{:<18} {:>9}  {:<8} {}{}\n{:<18} {}",
                    m.id,
                    system::gib(m.size),
                    f.label(),
                    m.title,
                    if have { "  [downloaded]" } else { "" },
                    "",
                    m.note
                );
            }
            Ok(())
        }
        Some(Cmd::Inspect { model, json }) => {
            let m = fle_core::gguf::read_model(&model)?;
            let sm = m.summary();
            if json {
                println!("{}", fle_core::gguf::py_json(&sm));
                return Ok(());
            }
            let n = |k: &str| sm[k].as_u64().map(|v| v.to_string()).unwrap_or("-".into());
            println!("{}  ({})", m.name, m.arch);
            println!(
                "  size        {} in {} part(s){}",
                system::gib(m.size),
                m.parts.len(),
                if m.complete() { "" } else { "  INCOMPLETE" }
            );
            println!(
                "  layers      {}   context {}   experts {} (used {})",
                n("layers"),
                n("context_length"),
                n("experts"),
                n("experts_used")
            );
            println!(
                "  expert data {}   always-needed {}   input tables {}",
                system::gib(m.expert_bytes),
                system::gib(m.gpu_bytes),
                system::gib(m.input_bytes)
            );
            println!(
                "  KV cache    {} bytes/token (q8_0)   {} (f16)",
                n("kv_bytes_per_token_q8"),
                n("kv_bytes_per_token_f16")
            );
            println!("  quant types:");
            for (t, (c, b)) in &m.type_histogram {
                println!("    {t:<8} {c:>5} tensors  {}", system::gib(*b));
            }
            Ok(())
        }
        Some(Cmd::Verify { model, json }) => {
            let m = fle_core::gguf::read_model(&model)?;
            let r = fle_core::gguf::verify_model(&m);
            let ok = r["ok"].as_bool().unwrap_or(false);
            if json {
                println!("{}", fle_core::gguf::py_json(&r));
            } else {
                println!("verify: {}", if ok { "ok" } else { "failed" });
                for c in r["checks"].as_array().unwrap() {
                    println!(
                        "[{}] {}: {}",
                        c["status"].as_str().unwrap(),
                        c["name"].as_str().unwrap(),
                        c["detail"].as_str().unwrap()
                    );
                    if let Some(h) = c["hint"].as_str().filter(|h| !h.is_empty()) {
                        println!("       -> {h}");
                    }
                }
                for w in r["warnings"].as_array().unwrap() {
                    println!("[warn] {}", w.as_str().unwrap());
                }
                for e in r["errors"].as_array().unwrap() {
                    println!("[fail] {}", e.as_str().unwrap());
                }
            }
            std::process::exit(if ok { 0 } else { 1 });
        }
        Some(Cmd::Run { model }) => {
            let srv = flow::ensure_and_start(&home, &model, &mut |e| print_event(&e))?;
            eprintln!("\nServing. Press Ctrl+C to stop.");
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            let _ = ctrlc_wait(tx);
            let _ = rx.recv();
            drop(srv);
            Ok(())
        }
        Some(Cmd::Ask { model, question }) => {
            let srv = flow::ensure_and_start(&home, &model, &mut |e| print_event(&e))?;
            let q = question.join(" ");
            eprintln!("\n> {q}\n");
            let st = chat::stream(
                &srv.base,
                &model,
                &[("user".into(), q)],
                &mut |reasoning, t| {
                    if reasoning {
                        eprint!("{t}")
                    } else {
                        print!("{t}");
                        let _ = std::io::stdout().flush();
                    }
                },
            )?;
            println!(
                "\n\n[{} tokens, {} tok/s decode, {} tok/s prompt]",
                st.tokens,
                st.decode_tps
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or("-".into()),
                st.prefill_tps
                    .map(|v| format!("{v:.0}"))
                    .unwrap_or("-".into())
            );
            Ok(())
        }
    }
}

// Wait for Ctrl+C without a dependency: block on a signal via a tiny thread reading stdin close is not enough; use libc-free approach.
fn ctrlc_wait(tx: std::sync::mpsc::Sender<()>) -> std::io::Result<()> {
    std::thread::spawn(move || {
        // `fle run` is meant to be stopped with Ctrl+C, which ends the process and its child (same process group).
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
            let _ = tx.send(());
        }
    });
    Ok(())
}

mod tui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use fle_core::{catalog, chat, data_dir, engines, flow, system, Event};
use std::io::Write;

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
enum Cmd {
    /// What this machine can do, and which engine fits it
    Doctor,
    /// The models FLE knows, and whether each fits this machine
    Models,
    /// Load a model and keep serving it (OpenAI-compatible API) until Ctrl+C
    Run { model: String },
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let home = cli.home.unwrap_or_else(data_dir);
    match cli.cmd {
        None => tui::run(&home),
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

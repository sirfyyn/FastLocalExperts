//! The whole way from nothing to an answering model, as one function that reports events.
use crate::{catalog, engines, plan, server::Server, system, Event};
use anyhow::{bail, Context, Result};
use std::path::Path;
use std::time::Duration;

pub fn model_path(home: &Path, m: &catalog::Model) -> std::path::PathBuf {
    home.join("models")
        .join(m.repo.replace('/', "__"))
        .join(Path::new(m.file).file_name().unwrap())
}

/// Install the engine for this machine if needed, download the model if needed (resumable, checked), start the engine
/// with automatic settings and wait until it answers.
pub fn ensure_and_start(
    home: &Path,
    model_id: &str,
    emit: &mut dyn FnMut(Event),
) -> Result<Server> {
    let sys = system::detect(home);
    let m = catalog::builtin()
        .into_iter()
        .find(|m| m.id == model_id)
        .with_context(|| format!("unknown model '{model_id}' (try: fle models)"))?;
    if catalog::fit(&m, &sys) == catalog::Fit::TooBig {
        bail!(
            "{} does not fit this machine ({} RAM, {} free on disk)",
            m.title,
            system::gib(sys.ram_bytes),
            system::gib(sys.disk_free_bytes)
        );
    }
    let v = engines::variants(&sys)
        .into_iter()
        .find(|v| !(m.fle_only && v.kind != "fle"))
        .context("no engine fits this machine")?;
    emit(Event::Step(format!("Engine: {} - {}", v.label, v.why)));
    let existing = engines::installed(home)
        .into_iter()
        .find(|(n, _)| n.starts_with(v.id));
    let server_bin = match existing {
        Some((_, p)) => p,
        None => {
            emit(Event::Step(
                "Installing the engine (checked against the SHA-256 GitHub publishes)".into(),
            ));
            engines::install(home, &v, &sys, &mut |l, d, t| {
                emit(Event::Progress {
                    label: l.into(),
                    done: d,
                    total: t,
                })
            })?
        }
    };
    let path = model_path(home, &m);
    if !path.exists() {
        emit(Event::Step(format!(
            "Downloading {} ({})",
            m.title,
            system::gib(m.size)
        )));
        let url = format!("https://huggingface.co/{}/resolve/main/{}", m.repo, m.file);
        crate::download::fetch(&url, &path, m.sha256, &mut |d, t| {
            emit(Event::Progress {
                label: format!("model {}", m.file),
                done: d,
                total: t,
            })
        })?;
    }
    emit(Event::Step("Choosing settings for this machine".into()));
    let pl = plan::auto_plan(&sys, v.kind, &path, m.size, m.id);
    let command = pl.command_line(&server_bin);
    emit(Event::Step(format!(
        "Starting the engine (context {} tokens)",
        pl.ctx
    )));
    let mut srv = Server::start(&server_bin, &pl, &home.join("logs"))?;
    srv.wait_ready(Duration::from_secs(900), &mut |l| {
        emit(Event::Log(l.to_string()))
    })?;
    emit(Event::Ready {
        base: srv.base.clone(),
        engine: v.label.into(),
        model: m.id.into(),
        command,
    });
    Ok(srv)
}

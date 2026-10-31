use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

pub fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("fle/0.1")
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(None)
        .build()?)
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN").ok().or_else(|| {
        let p = dirs::home_dir()?.join(".cache/huggingface/token");
        fs::read_to_string(p).ok().map(|s| s.trim().to_string())
    })
}

/// Download `url` to `dest`, resuming from `dest.part`, verifying SHA-256 when known. `progress(done, total)`.
pub fn fetch(
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    fs::create_dir_all(dest.parent().context("no parent")?)?;
    let part = dest.with_extension(format!(
        "{}part",
        dest.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));
    let mut have = part.metadata().map(|m| m.len()).unwrap_or(0);
    let mut hasher = Sha256::new();
    if have > 0 && sha256.is_some() {
        let mut f = File::open(&part)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }
    let c = client()?;
    let mut req = c.get(url);
    if let Some(t) = hf_token().filter(|_| url.contains("huggingface.co")) {
        req = req.bearer_auth(t);
    }
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }
    let mut resp = req.send()?;
    let status = resp.status().as_u16();
    if status == 416 {
        fs::rename(&part, dest)?;
        return Ok(());
    }
    if status == 401 || status == 403 {
        bail!("access denied ({status}): gated repo? Accept its terms on Hugging Face and set HF_TOKEN");
    }
    if !resp.status().is_success() {
        bail!("download failed: HTTP {status} for {url}");
    }
    if have > 0 && status == 200 {
        // server ignored Range: start over
        have = 0;
        hasher = Sha256::new();
        fs::remove_file(&part).ok();
    }
    let total = have + resp.content_length().unwrap_or(0);
    let mut out = OpenOptions::new().create(true).append(true).open(&part)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut done = have;
    let mut last = std::time::Instant::now();
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        if sha256.is_some() {
            hasher.update(&buf[..n]);
        }
        done += n as u64;
        if last.elapsed().as_millis() > 150 {
            progress(done, total);
            last = std::time::Instant::now();
        }
    }
    out.flush()?;
    progress(done, total);
    if total > 0 && done < total {
        bail!("connection ended early ({done} of {total} bytes); run again to resume");
    }
    if let Some(want) = sha256 {
        let got = hex::encode(hasher.finalize());
        if !got.eq_ignore_ascii_case(want) {
            let bad = dest.with_extension("bad");
            fs::rename(&part, &bad).ok();
            bail!(
                "checksum mismatch (got {got}); file moved to {}",
                bad.display()
            );
        }
    }
    fs::rename(&part, dest)?;
    Ok(())
}

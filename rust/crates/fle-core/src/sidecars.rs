//! `fle fetch-sidecars`: download the MTP head and vocabulary that belong to a known model, next to the model.
//! Port of `tools/fle/sidecars.py`. Only models with a known source are served; a model that is not in the table gets
//! "no known source", never a guessed URL. The download goes through an injected fetcher, so tests never touch the network.
use crate::gguf::{self, human_bytes, Model};
use crate::FleError;
use anyhow::Result;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const HF: &str = "https://huggingface.co";
pub const FLE_REPO: &str = "sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF";

#[derive(Clone, Debug)]
pub struct SidecarFile {
    pub kind: &'static str, // "mtp_head" or "mtp_vocab": the keys `gguf::sidecars` uses
    pub repo: String,
    pub path: String,
    pub size: u64,
    pub sha256: String,
}
impl SidecarFile {
    pub fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
    /// `quote(path)` keeps `/` and the unreserved characters.
    pub fn url(&self) -> String {
        let q: String = self
            .path
            .bytes()
            .map(|b| {
                if b.is_ascii_alphanumeric() || b"_.-~/".contains(&b) {
                    (b as char).to_string()
                } else {
                    format!("%{b:02X}")
                }
            })
            .collect();
        format!("{HF}/{}/resolve/main/{q}", self.repo)
    }
}

#[derive(Clone, Debug)]
pub struct Source {
    pub family: String,
    /// matched case-insensitively against the start of the model file name
    pub prefix: String,
    pub files: Vec<SidecarFile>,
}

pub fn known_sources() -> Vec<Source> {
    vec![Source {
        family: "Qwen3.8-Flash-Next-FLE".into(),
        prefix: "Qwen3.8-Flash-Next-FLE".into(),
        files: vec![
            SidecarFile {
                kind: "mtp_head",
                repo: FLE_REPO.into(),
                path: "mtp/mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf".into(),
                size: 1528277056,
                sha256: "25217613e5e4659186db0b81ce2be77ab5955d01817732e4597b8f2bf0e292fa".into(),
            },
            SidecarFile {
                kind: "mtp_vocab",
                repo: FLE_REPO.into(),
                path: "mtp/fle-mtp-vocab-32k.txt".into(),
                size: 281159,
                sha256: "8385f0457511fe600986bbd0ecb0b32766cbe31859c12e78a033685c28eef0d8".into(),
            },
        ],
    }]
}

pub fn find_source(model: &Model, sources: &[Source]) -> Option<Source> {
    let name = model.first.file_name()?.to_string_lossy().to_lowercase();
    sources
        .iter()
        .find(|s| name.starts_with(&s.prefix.to_lowercase()))
        .cloned()
}

#[derive(Clone, Debug)]
pub struct Item {
    pub file: SidecarFile,
    pub dest: PathBuf,
    pub action: &'static str, // "download", "present" or "replace"
}
impl Item {
    pub fn as_json(&self) -> Value {
        json!({"kind": self.file.kind, "name": self.file.name(), "url": self.file.url(), "size": self.file.size, "sha256": self.file.sha256, "dest": self.dest.to_string_lossy(), "action": self.action})
    }
}

pub fn plan_items(model: &Model, source: &Source, force: bool) -> Vec<Item> {
    let present = gguf::sidecars(&model.first);
    let folder = model.first.parent().unwrap_or(Path::new(".")).to_path_buf();
    source
        .files
        .iter()
        .map(|f| {
            let dest = folder.join(f.name());
            if force {
                Item {
                    file: f.clone(),
                    action: if dest.exists() { "replace" } else { "download" },
                    dest,
                }
            } else if let Some(p) = present
                .get(f.kind)
                .and_then(Value::as_str)
                .filter(|p| !p.is_empty())
            {
                Item {
                    file: f.clone(),
                    dest: PathBuf::from(p),
                    action: "present",
                }
            } else {
                Item {
                    file: f.clone(),
                    dest,
                    action: "download",
                }
            }
        })
        .collect()
}

pub fn sha256_of(p: &Path) -> Result<String> {
    let mut h = Sha256::new();
    let mut f = std::fs::File::open(p)?;
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

pub type Fetcher<'a> = &'a dyn Fn(&str, &Path, Option<&str>) -> Result<()>;

/// Stream `url` into `dest`. The only place that uses the network.
pub fn http_fetcher(url: &str, dest: &Path, bearer: Option<&str>) -> Result<()> {
    let fail = |e: String| {
        FleError::new(
            format!("Download failed for {url}: {e}"),
            "Check the network, or download the file by hand.",
        )
    };
    let c = reqwest::blocking::Client::builder()
        .user_agent("fle-helper")
        .timeout(None)
        .build()
        .map_err(|e| fail(e.to_string()))?;
    let mut req = c.get(url);
    if let Some(t) = bearer {
        req = req.bearer_auth(t);
    }
    let mut resp = req
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| fail(e.to_string()))?;
    let mut out = std::fs::File::create(dest).map_err(|e| fail(e.to_string()))?;
    resp.copy_to(&mut out).map_err(|e| fail(e.to_string()))?;
    Ok(())
}

fn free_bytes(folder: &Path) -> u64 {
    crate::system::detect(folder).disk_free_bytes
}

/// Download what is not present; returns the items that were written. Errors are `FleError`s.
pub fn fetch(
    items: &[Item],
    force: bool,
    fetcher: Fetcher,
    say: &dyn Fn(&str),
    free: Option<u64>,
) -> Result<Vec<Item>> {
    let todo: Vec<&Item> = items.iter().filter(|i| i.action != "present").collect();
    if todo.is_empty() {
        return Ok(vec![]);
    }
    let folder = todo[0]
        .dest
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let needed: u64 = todo.iter().map(|i| i.file.size).sum();
    let free = free.unwrap_or_else(|| free_bytes(&folder));
    if free < needed + 64 * 1024 * 1024 {
        return Err(FleError::new(
            format!(
                "Not enough disk space in {}: {} needed, {} free.",
                folder.display(),
                human_bytes(needed),
                human_bytes(free)
            ),
            "Free some space or move the model.",
        )
        .into());
    }
    let token = std::env::var("HF_TOKEN")
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    let mut written = vec![];
    for item in todo {
        let name = item.dest.file_name().unwrap().to_string_lossy().to_string();
        if item.dest.exists() && !force {
            say(&format!(
                "{name}: already there, left alone (use --force to replace)"
            ));
            continue;
        }
        let temp = item
            .dest
            .with_file_name(format!(".{name}.part-{}", std::process::id()));
        say(&format!(
            "{name}: downloading {}",
            human_bytes(item.file.size)
        ));
        let res = (|| -> Result<()> {
            fetcher(&item.file.url(), &temp, token.as_deref())?;
            let size = std::fs::metadata(&temp)
                .map(|m| m.len() as i64)
                .unwrap_or(-1);
            if size != item.file.size as i64 {
                return Err(FleError::new(
                    format!("{name}: got {size} bytes, expected {}.", item.file.size),
                    "The download was cut short or the file changed upstream. Nothing was written.",
                )
                .into());
            }
            let digest = sha256_of(&temp)?;
            if digest != item.file.sha256 {
                return Err(FleError::new(
                    format!(
                        "{name}: SHA-256 mismatch ({}... instead of {}...).",
                        &digest[..12],
                        &item.file.sha256[..12]
                    ),
                    "Nothing was written. Try again, or download the file by hand.",
                )
                .into());
            }
            if item.dest.exists() && !force {
                return Err(FleError::new(
                    format!("{} appeared during the download.", item.dest.display()),
                    "It was left alone; use --force to replace it.",
                )
                .into());
            }
            std::fs::rename(&temp, &item.dest)?;
            Ok(())
        })();
        let _ = std::fs::remove_file(&temp);
        res?;
        written.push(item.clone());
    }
    Ok(written)
}

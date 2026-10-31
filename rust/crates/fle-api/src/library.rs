//! The local model library: GGUF files in the models folder, plus files elsewhere that the user added by path. Port of
//! `fle_studio/library.py`. Split models are one entry, listed under their first part; a model whose parts are still
//! downloading shows as incomplete.
use crate::catalog;
use crate::gguf::{inspect_cached, read_header, split_match, split_parts};
use serde_json::{json, Value};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};

const SKIP_DIRS: [&str; 4] = ["_downloads", "_engines", ".cache", ".git"];

/// `Path.resolve()`: symlinks resolved as far as the path exists, the rest appended as written.
pub fn py_resolve(p: &Path) -> PathBuf {
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    };
    let mut tail: Vec<std::ffi::OsString> = vec![];
    let mut cur = abs.as_path();
    loop {
        if let Ok(c) = std::fs::canonicalize(cur) {
            let mut out = c;
            for t in tail.iter().rev() {
                out.push(t);
            }
            return normalize_dots(&out);
        }
        match (cur.file_name(), cur.parent()) {
            (Some(n), Some(par)) => {
                tail.push(n.to_os_string());
                cur = par;
            }
            _ => return abs,
        }
    }
}

fn normalize_dots(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// `Path(text)` as Python prints it: empty is `.`, doubled and trailing slashes and `.` parts are gone, `..` stays.
pub fn py_path(text: &str) -> PathBuf {
    let mut out = PathBuf::new();
    for c in Path::new(text).components() {
        if !matches!(c, std::path::Component::CurDir) {
            out.push(c.as_os_str());
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

pub fn model_id(path: &Path) -> String {
    let key = py_resolve(path).to_string_lossy().to_string();
    let key = if cfg!(windows) {
        key.to_lowercase()
    } else {
        key
    };
    hex::encode(Sha1::digest(key.as_bytes()))[..12].to_string()
}

fn kind(name: &str) -> &'static str {
    let low = name.to_lowercase();
    if low.starts_with("mtp-") || low.contains("-mtp-") {
        "draft"
    } else if low.contains("mmproj") {
        "mmproj"
    } else {
        "model"
    }
}

pub enum LibError {
    NotFound,
    Invalid(String),
}

pub struct Library {
    file: PathBuf,
    pub imports: Vec<Value>,
}

/// `os.walk` with the pruning `scan` does: files of every folder below `root`, symlinked folders are listed but not entered.
fn walk(root: &Path, out: &mut Vec<(PathBuf, String)>) {
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    let (mut dirs, mut files) = (vec![], vec![]);
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let is_link = e.file_type().map(|t| t.is_symlink()).unwrap_or(false);
        let is_dir = std::fs::metadata(e.path())
            .map(|m| m.is_dir())
            .unwrap_or(false);
        if is_dir {
            dirs.push((name, e.path(), is_link));
        } else {
            files.push(name);
        }
    }
    for f in files {
        out.push((root.to_path_buf(), f));
    }
    for (name, path, is_link) in dirs {
        if SKIP_DIRS.contains(&name.as_str()) || name.starts_with('.') || is_link {
            continue;
        }
        walk(&path, out);
    }
}

impl Library {
    pub fn new(data_dir: &Path) -> Library {
        let file = data_dir.join("library.json");
        let imports = std::fs::read_to_string(&file)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|d| d.get("imports").and_then(Value::as_array).cloned())
            .map(|a| {
                a.into_iter()
                    .filter(|i| {
                        i.as_object()
                            .is_some_and(|o| o.get("path").is_some_and(crate::settings::py_truthy))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Library { file, imports }
    }

    fn save(&self) {
        if let Some(d) = self.file.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let tmp = self.file.with_extension("tmp");
        let _ = std::fs::write(
            &tmp,
            serde_json::to_string_pretty(&json!({"imports": self.imports})).unwrap(),
        )
        .and_then(|_| std::fs::rename(&tmp, &self.file));
    }

    /// Every model on disk.
    pub fn scan(&self, models_dir: &Path) -> Vec<Value> {
        let head_name = catalog::mtp_head()["name"].as_str().unwrap().to_string();
        let vocab_name = catalog::mtp_vocab()["name"].as_str().unwrap().to_string();
        let mut firsts: Vec<(String, PathBuf, &'static str)> = vec![]; // insertion order like a dict
        let mut extras: Vec<(String, PathBuf)> = vec![];
        let set_extra = |extras: &mut Vec<(String, PathBuf)>, k: &str, v: PathBuf| match extras
            .iter_mut()
            .find(|(n, _)| n == k)
        {
            Some(e) => e.1 = v,
            None => extras.push((k.to_string(), v)),
        };
        let put = |firsts: &mut Vec<(String, PathBuf, &'static str)>,
                   p: PathBuf,
                   src: &'static str,
                   overwrite: bool| {
            let key = p.to_string_lossy().to_string();
            match firsts.iter_mut().find(|(k, _, _)| *k == key) {
                Some(e) if overwrite => *e = (key, p, src),
                Some(_) => {}
                None => firsts.push((key, p, src)),
            }
        };
        if models_dir.exists() {
            let mut all = vec![];
            walk(models_dir, &mut all);
            for (dir, fname) in all {
                let p = dir.join(&fname);
                if fname == head_name || fname == vocab_name {
                    set_extra(&mut extras, &fname, p.clone());
                }
                if !fname.to_lowercase().ends_with(".gguf") {
                    continue;
                }
                if let Some((base, no, count)) = split_match(&fname) {
                    if no != "00001" {
                        let first = p.with_file_name(format!("{base}-00001-of-{count}.gguf"));
                        if !first.exists() {
                            put(&mut firsts, first, "folder", false);
                        }
                        continue;
                    }
                }
                put(&mut firsts, p, "folder", true);
            }
        }
        for imp in &self.imports {
            let p = PathBuf::from(imp["path"].as_str().unwrap_or(""));
            put(&mut firsts, p.clone(), "import", false);
            for extra in [&head_name, &vocab_name] {
                let parent = p.parent().unwrap_or(Path::new("")).to_path_buf();
                let grand = parent.parent().unwrap_or(Path::new("")).to_path_buf();
                for cand in [
                    parent.join(extra),
                    grand.join("mtp").join(extra),
                    parent.join("mtp").join(extra),
                ] {
                    if !extras.iter().any(|(n, _)| n == extra) && cand.exists() {
                        set_extra(&mut extras, extra, cand);
                    }
                }
            }
        }
        let mut out: Vec<Value> = firsts
            .iter()
            .map(|(_, p, s)| self.describe(p, s, &extras, &head_name, &vocab_name))
            .collect();
        out.sort_by_key(|m| {
            (
                m["kind"] != "model",
                m["name"].as_str().unwrap_or("").to_lowercase(),
            )
        });
        out
    }

    fn describe(
        &self,
        first: &Path,
        source: &str,
        extras: &[(String, PathBuf)],
        head_name: &str,
        vocab_name: &str,
    ) -> Value {
        let fname = first.file_name().unwrap().to_string_lossy().to_string();
        let mut part_info = vec![];
        for p in split_parts(first) {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            let partial = p.with_file_name(format!("{name}.part"));
            if p.exists() {
                part_info.push(json!({"name": name, "size": std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), "state": "present"}));
            } else if partial.exists() {
                part_info.push(json!({"name": name, "size": std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0), "state": "downloading"}));
            } else {
                part_info.push(json!({"name": name, "size": 0, "state": "missing"}));
            }
        }
        let complete = part_info.iter().all(|p| p["state"] == "present");
        let cat = catalog::by_filename(&fname);
        let (mut info, mut error) = (Value::Null, String::new());
        if complete {
            match inspect_cached(first) {
                Ok(i) => info = i.public(),
                Err(e) => error = e.0,
            }
        }
        let name = match cat {
            Some(c) => c["name"].as_str().unwrap_or("").to_string(),
            None => match info
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
            {
                Some(n) => n.to_string(),
                None => split_match(&fname)
                    .map(|(b, _, _)| b)
                    .unwrap_or_else(|| fname.clone()),
            },
        };
        let name = name.strip_suffix(".gguf").unwrap_or(&name).to_string();
        let find = |n: &str| {
            extras
                .iter()
                .find(|(k, _)| k == n)
                .map(|(_, p)| p.to_string_lossy().to_string())
        };
        let mtp_ok = cat.is_some_and(|c| c["mtp"].as_bool() == Some(true));
        let size: u64 = part_info
            .iter()
            .map(|p| p["size"].as_u64().unwrap_or(0))
            .sum();
        json!({
            "id": model_id(first), "name": name, "file": fname, "path": first.to_string_lossy(),
            "folder": first.parent().unwrap_or(Path::new("")).to_string_lossy(), "parts": part_info, "size": size, "complete": complete,
            "source": source, "catalog_id": cat.map(|c| c["id"].clone()).unwrap_or(Value::Null), "kind": kind(&fname), "info": info, "error": error,
            "mtp": {"supported": mtp_ok, "head": if mtp_ok { find(head_name).map(Value::from).unwrap_or(Value::Null) } else { Value::Null },
                    "vocab": if mtp_ok { find(vocab_name).map(Value::from).unwrap_or(Value::Null) } else { Value::Null }},
            "sampling": cat.map(|c| c["sampling"].clone()).unwrap_or(json!({})),
        })
    }

    /// A model by id or by path.
    pub fn get(&self, models_dir: &Path, ident: &str) -> Option<Value> {
        self.scan(models_dir)
            .into_iter()
            .find(|m| m["id"] == ident || m["path"] == ident)
    }

    /// Add a GGUF file that lives outside the models folder (nothing is copied).
    pub fn add(&mut self, models_dir: &Path, path_text: &str) -> Result<Value, String> {
        let t = path_text.trim().trim_matches('"');
        let mut p = match t.strip_prefix("~/") {
            Some(r) => dirs::home_dir().unwrap_or_default().join(py_path(r)),
            None => py_path(t),
        };
        if !p.is_file() {
            return Err(format!("No file at {}", p.display()));
        }
        if p.extension().map(|e| e.to_string_lossy().to_lowercase()) != Some("gguf".into()) {
            return Err("That is not a .gguf file".into());
        }
        let fname = p.file_name().unwrap().to_string_lossy().to_string();
        if let Some((base, no, count)) = split_match(&fname) {
            if no != "00001" {
                p = p.with_file_name(format!("{base}-00001-of-{count}.gguf"));
            }
        }
        read_header(&p, 0).map_err(|e| e.0)?;
        let ps = p.to_string_lossy().to_string();
        if !self.imports.iter().any(|i| i["path"] == ps) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            self.imports.push(json!({"path": ps, "added": now}));
            self.save();
        }
        self.get(models_dir, &ps)
            .ok_or_else(|| "The file was added but could not be read back".to_string())
    }

    pub fn remove(
        &mut self,
        models_dir: &Path,
        ident: &str,
        delete_files: bool,
    ) -> Result<(), LibError> {
        let m = self.get(models_dir, ident).ok_or(LibError::NotFound)?;
        let mp = m["path"].as_str().unwrap_or("").to_string();
        let before = self.imports.len();
        self.imports.retain(|i| i["path"] != mp);
        if self.imports.len() != before {
            self.save();
        }
        if !delete_files {
            return Ok(());
        }
        let root = py_resolve(models_dir);
        let first = py_resolve(Path::new(&mp));
        if !first.ancestors().skip(1).any(|a| a == root) {
            return Err(LibError::Invalid(
                "Only files inside the models folder are deleted; files you added by path stay"
                    .into(),
            ));
        }
        for p in split_parts(&first) {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            for c in [p.clone(), p.with_file_name(format!("{n}.part"))] {
                if c.exists() {
                    let _ = std::fs::remove_file(c);
                }
            }
        }
        Ok(())
    }
}

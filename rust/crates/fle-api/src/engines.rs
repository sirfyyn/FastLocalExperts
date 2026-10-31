//! Engines: which llama-server builds fit this machine, installing them from GitHub releases, and checking a llama-server
//! of your own. Port of `fle_studio/engines.py`.
//!
//! - FLE engine (sirfyyn/FastLocalExperts releases): NVIDIA GPUs on Linux and Windows x64 with AVX2.
//! - Upstream llama.cpp (ggml-org/llama.cpp, the tag in the settings): Metal on Apple Silicon, Vulkan for AMD and Intel GPUs,
//!   CUDA as the alternative on NVIDIA, CPU everywhere.
//!
//! Asset names are matched by pattern against the release's own asset list, and each download is checked against the SHA-256
//! digest GitHub publishes for it.
use crate::downloads::Downloads;
use crate::settings::Config;
use fle_core::launch::run_capture;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const UPSTREAM_REPO: &str = "ggml-org/llama.cpp";
pub const FLE_REPO: &str = "sirfyyn/FastLocalExperts";
pub const GITHUB_API: &str = "https://api.github.com/repos";

/// An error with the HTTP status the page gets for it.
pub enum EngineError {
    /// `RuntimeError`: 409
    Runtime(String),
    /// `OSError`: 502, "Could not reach GitHub: ..."
    Network(String),
    /// `KeyError`: 404
    NotFound,
}
use EngineError::*;

/// (os, arch) like `_os_arch`: macos | win | linux, arm64 | x64 | other.
pub fn os_arch() -> (&'static str, String) {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(windows) {
        "win"
    } else {
        "linux"
    };
    let arch = match std::env::consts::ARCH {
        "aarch64" | "arm64" => "arm64".to_string(),
        "x86_64" | "amd64" => "x64".to_string(),
        a => a.to_string(),
    };
    (os, arch)
}

/// CUDA major.minor -> lowest NVIDIA driver branch that runs it.
fn cuda_driver(version: &str) -> Option<i64> {
    match version {
        "13" => Some(580),
        "12.8" => Some(570),
        "12.9" => Some(575),
        "12.6" => Some(560),
        "12.4" => Some(550),
        _ => None,
    }
}

pub fn driver_major(sys: &Value) -> i64 {
    for g in sys["gpus"].as_array().into_iter().flatten() {
        if g["vendor"] == "nvidia" {
            if let Some(d) = g["driver"].as_str().filter(|d| !d.is_empty()) {
                if let Ok(n) = d.split('.').next().unwrap_or("").parse::<i64>() {
                    return n;
                }
            }
        }
    }
    0
}

pub fn cuda_ok(version: &str, driver: i64) -> bool {
    let need = cuda_driver(version)
        .or_else(|| cuda_driver(version.split('.').next().unwrap_or("")))
        .unwrap_or(999);
    driver >= need
}

/// Engine choices for this machine, best first, each with the reason it is offered.
pub fn variants(sys: &Value, os: &str, arch: &str) -> Vec<Value> {
    let gpus = sys["gpus"].as_array().cloned().unwrap_or_default();
    let has_nvidia = gpus.iter().any(|g| g["vendor"] == "nvidia");
    let avx2 = sys["cpu"]["avx2"].as_bool().unwrap_or(false);
    let others = sys["other_gpus"]
        .as_array()
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    let mut out = vec![];
    let mut add = |id: &str, kind: &str, backend: &str, label: &str, why: &str| {
        out.push(json!({"id": id, "kind": kind, "backend": backend, "label": label, "why": why}))
    };
    if os == "macos" {
        if arch == "arm64" {
            add(
                "upstream-metal",
                "upstream",
                "metal",
                "llama.cpp for Apple Silicon (Metal)",
                "Runs the model on the Apple GPU with unified memory.",
            );
        } else {
            add(
                "upstream-macos-x64",
                "upstream",
                "cpu",
                "llama.cpp for Intel Macs",
                "CPU build for Intel Macs.",
            );
        }
        return out;
    }
    if has_nvidia && arch == "x64" {
        if avx2 {
            add(
                "fle-cuda",
                "fle",
                "cuda",
                "FLE engine (NVIDIA CUDA)",
                "The FLE build: GPU expert cache, MTP drafting, pinned expert loading.",
            );
        }
        add(
            "upstream-cuda",
            "upstream",
            "cuda",
            "llama.cpp (NVIDIA CUDA)",
            "Upstream llama.cpp on the NVIDIA GPU.",
        );
    }
    if others || has_nvidia {
        add(
            "upstream-vulkan",
            "upstream",
            "vulkan",
            "llama.cpp (Vulkan)",
            if others {
                "Works on AMD, Intel and NVIDIA GPUs."
            } else {
                "Fallback for GPUs without a CUDA build."
            },
        );
    }
    add(
        "upstream-cpu",
        "upstream",
        "cpu",
        "llama.cpp (CPU only)",
        "Works everywhere; slow for large models.",
    );
    out
}

/// A release asset name against one of the small patterns below. Each pattern is a template: `{n}` is digits, `{cuda}` a
/// CUDA version (digits and dots); everything else must match exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern(Vec<Part>);
#[derive(Debug, Clone, PartialEq)]
enum Part {
    Lit(String),
    Digits,
    AnyRun, // `.+`
    Cuda,
}

impl Pattern {
    /// Returns the CUDA version the name carries, `Some(None)` for a match without one, `None` for no match.
    pub fn matches(&self, name: &str) -> Option<Option<String>> {
        fn go(parts: &[Part], s: &str, cuda: &mut Option<String>) -> bool {
            match parts.split_first() {
                None => s.is_empty(),
                Some((Part::Lit(l), rest)) => s
                    .strip_prefix(l.as_str())
                    .is_some_and(|r| go(rest, r, cuda)),
                Some((Part::Digits, rest)) => {
                    let n = s.bytes().take_while(u8::is_ascii_digit).count();
                    (1..=n).rev().any(|k| go(rest, &s[k..], cuda))
                }
                Some((Part::AnyRun, rest)) => {
                    (1..=s.len()).any(|k| s.is_char_boundary(k) && go(rest, &s[k..], cuda))
                }
                Some((Part::Cuda, rest)) => {
                    let n = s
                        .bytes()
                        .take_while(|b| b.is_ascii_digit() || *b == b'.')
                        .count();
                    (1..=n).rev().any(|k| {
                        let mut c2 = Some(s[..k].to_string());
                        if go(rest, &s[k..], &mut c2) {
                            *cuda = c2;
                            true
                        } else {
                            false
                        }
                    })
                }
            }
        }
        let mut cuda = None;
        go(&self.0, name, &mut cuda).then_some(cuda)
    }
}

fn pat(parts: Vec<Part>) -> Pattern {
    Pattern(parts)
}
fn lit(s: &str) -> Part {
    Part::Lit(s.to_string())
}

/// The release assets of a variant; the first is the engine, CUDA builds add their runtime.
pub fn asset_patterns(variant: &str, os: &str, arch: &str) -> Result<Vec<Pattern>, String> {
    let ext = if os == "win" { ".zip" } else { ".tar.gz" };
    let plat = match os {
        "linux" => "ubuntu",
        "win" => "win",
        _ => "macos",
    };
    Ok(vec![match variant {
        "upstream-metal" | "upstream-macos-x64" => pat(vec![
            lit("llama-b"),
            Part::Digits,
            lit(&format!("-bin-macos-{arch}{ext}")),
        ]),
        "upstream-cpu" => {
            let cpu = if os == "win" { "cpu-" } else { "" };
            pat(vec![
                lit("llama-b"),
                Part::Digits,
                lit(&format!("-bin-{plat}-{cpu}{arch}{ext}")),
            ])
        }
        "upstream-vulkan" => pat(vec![
            lit("llama-b"),
            Part::Digits,
            lit(&format!("-bin-{plat}-vulkan-{arch}{ext}")),
        ]),
        "upstream-cuda" => pat(vec![
            lit("llama-b"),
            Part::Digits,
            lit(&format!("-bin-{plat}-cuda-")),
            Part::Cuda,
            lit(&format!("-{arch}{ext}")),
        ]),
        "fle-cuda" => pat(vec![
            lit("fle-"),
            Part::AnyRun,
            lit(&format!("-bin-{plat}-cuda-")),
            Part::Cuda,
            lit(&format!("-{arch}{ext}")),
        ]),
        other => return Err(format!("unknown engine {other}")),
    }])
}

pub fn cudart_pattern(os: &str, arch: &str, cuda: &str) -> Pattern {
    if os == "win" {
        pat(vec![lit(&format!(
            "cudart-llama-bin-win-cuda-{cuda}-{arch}.zip"
        ))])
    } else {
        pat(vec![
            lit("cudart-llama-b"),
            Part::Digits,
            lit(&format!("-bin-ubuntu-cuda-{cuda}-{arch}.tar.gz")),
        ])
    }
}

pub struct Engines {
    inner: Arc<Inner>,
}

struct Inner {
    dir: PathBuf,
    file: PathBuf,
    items: Mutex<Vec<Value>>,
    settings: Arc<Mutex<Config>>,
    downloads: Downloads,
    github_api: String,
    detect_cache: Mutex<HashMap<(String, i64), Value>>,
}

fn http_json(url: &str) -> Result<Value, EngineError> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(30))
        .user_agent(&format!("fle-studio/{}", env!("CARGO_PKG_VERSION")))
        .build();
    match agent
        .get(url)
        .set("Accept", "application/vnd.github+json")
        .call()
    {
        Ok(r) => {
            serde_json::from_str::<Value>(&r.into_string().map_err(|e| Network(e.to_string()))?)
                .map_err(|e| Network(e.to_string()))
        }
        Err(ureq::Error::Status(c, _)) => Err(Network(match c {
            403 => "Forbidden".into(),
            404 => "Not Found".into(),
            429 => "Too Many Requests".into(),
            c => format!("HTTP {c}"),
        })),
        Err(e) => {
            use std::error::Error;
            let mut cur: &dyn Error = &e;
            while let Some(s) = cur.source() {
                cur = s;
            }
            Err(Network(cur.to_string()))
        }
    }
}

fn mtime(p: &Path) -> i64 {
    std::fs::metadata(p)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn py_name(s: &str) -> String {
    // re.sub(r"[^\w.-]+", "_", s)
    let mut out = String::new();
    let mut gap = false;
    for c in s.chars() {
        if c.is_alphanumeric() || matches!(c, '_' | '.' | '-') {
            if gap {
                out.push('_');
                gap = false;
            }
            out.push(c);
        } else {
            gap = true;
        }
    }
    if gap {
        out.push('_');
    }
    out
}

impl Engines {
    pub fn new(
        data_dir: &Path,
        downloads: Downloads,
        settings: Arc<Mutex<Config>>,
        github_api: &str,
    ) -> Engines {
        let dir = data_dir.join("engines");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("engines.json");
        let items = match std::fs::read_to_string(&file)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        {
            Some(Value::Array(a)) => a,
            _ => vec![],
        };
        let e = Engines {
            inner: Arc::new(Inner {
                dir,
                file,
                items: Mutex::new(items),
                settings,
                downloads: downloads.clone(),
                github_api: github_api.to_string(),
                detect_cache: Mutex::new(HashMap::new()),
            }),
        };
        let me = Engines {
            inner: e.inner.clone(),
        };
        downloads.set_hook("engine", Box::new(move |job| me.install_downloaded(job)));
        e
    }

    fn save(&self, items: &[Value]) {
        let tmp = self.inner.file.with_extension("tmp");
        let _ = std::fs::write(&tmp, serde_json::to_string_pretty(items).unwrap())
            .and_then(|_| std::fs::rename(&tmp, &self.inner.file));
    }

    pub fn installed(&self) -> Vec<Value> {
        self.inner
            .items
            .lock()
            .unwrap()
            .iter()
            .filter(|e| Path::new(e["path"].as_str().unwrap_or("")).exists())
            .cloned()
            .collect()
    }

    fn setting(&self, key: &str) -> String {
        let c = self.inner.settings.lock().unwrap();
        c.get("engine")
            .and_then(|e| e.get(key))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }

    fn release(&self, repo: &str, tag: &str) -> Result<Value, EngineError> {
        let tail = if tag.is_empty() || tag == "latest" {
            "latest".to_string()
        } else {
            format!("tags/{tag}")
        };
        http_json(&format!("{}/{repo}/releases/{tail}", self.inner.github_api))
    }

    /// Which release assets an install would download (a readable reason if none fits).
    pub fn plan_install(&self, variant: &str, sys: &Value) -> Result<Value, EngineError> {
        let (os, arch) = os_arch();
        let fle_tag = {
            let t = self.setting("fle_tag");
            if t.is_empty() {
                "latest".to_string()
            } else {
                t
            }
        };
        let up_tag = {
            let t = self.setting("upstream_tag");
            if t.is_empty() {
                "b11540".to_string()
            } else {
                t
            }
        };
        let (rel, source) = if variant.starts_with("fle") {
            (self.release(FLE_REPO, &fle_tag)?, FLE_REPO)
        } else {
            (self.release(UPSTREAM_REPO, &up_tag)?, UPSTREAM_REPO)
        };
        let assets = rel["assets"].as_array().cloned().unwrap_or_default();
        let pattern = asset_patterns(variant, os, &arch)
            .map_err(Runtime)?
            .remove(0);
        let driver = driver_major(sys);
        let mut matches: Vec<(Vec<i64>, Value, Option<String>)> = vec![];
        for a in &assets {
            if let Some(cuda) = pattern.matches(a["name"].as_str().unwrap_or("")) {
                if let Some(c) = &cuda {
                    if !cuda_ok(c, driver) {
                        continue;
                    }
                }
                let key = cuda
                    .clone()
                    .unwrap_or_else(|| "0".into())
                    .split('.')
                    .map(|x| x.parse::<i64>().unwrap_or(0))
                    .collect();
                matches.push((key, a.clone(), cuda));
            }
        }
        if matches.is_empty() {
            let hint = if variant.contains("cuda") && driver != 0 {
                format!(" (NVIDIA driver {driver} is too old for the CUDA builds in it)")
            } else {
                String::new()
            };
            return Err(Runtime(format!(
                "No {variant} build for {os}-{arch} in {source} {}{hint}",
                rel["tag_name"].as_str().unwrap_or("")
            )));
        }
        matches.sort_by(|a, b| b.0.cmp(&a.0));
        let (_, main, cuda) = matches.remove(0);
        let mut files = vec![main];
        if let Some(c) = &cuda {
            let p = cudart_pattern(os, &arch, c);
            let mut runtime = assets
                .iter()
                .find(|a| p.matches(a["name"].as_str().unwrap_or("")).is_some())
                .cloned();
            if runtime.is_none() && source == FLE_REPO {
                // FLE builds use upstream's CUDA runtime package
                let up = self.release(UPSTREAM_REPO, &up_tag)?;
                runtime = up["assets"].as_array().and_then(|a| {
                    a.iter()
                        .find(|x| p.matches(x["name"].as_str().unwrap_or("")).is_some())
                        .cloned()
                });
            }
            if runtime.is_none() && os == "win" {
                return Err(Runtime(format!(
                    "The CUDA {c} runtime package is missing from the release"
                )));
            }
            files.extend(runtime);
        }
        let assets_out: Vec<Value> = files
            .iter()
            .map(|a| {
                let digest = a["digest"].as_str().unwrap_or("");
                let sha = digest.strip_prefix("sha256:").unwrap_or(digest);
                json!({"name": a["name"], "url": a["browser_download_url"], "size": a.get("size").cloned().filter(|s| !s.is_null()).unwrap_or(json!(0)), "sha256": if sha.is_empty() { Value::Null } else { json!(sha) }})
            })
            .collect();
        Ok(
            json!({"variant": variant, "tag": rel["tag_name"].as_str().unwrap_or(""), "source": source, "cuda": cuda, "assets": assets_out}),
        )
    }

    pub fn install(&self, variant: &str, sys: &Value) -> Result<Value, EngineError> {
        let plan = self.plan_install(variant, sys)?;
        let stage = self.inner.dir.join("_downloads");
        let _ = std::fs::create_dir_all(&stage);
        let files: Vec<Value> = plan["assets"].as_array().unwrap().iter().map(|a| json!({"url": a["url"], "dest": stage.join(a["name"].as_str().unwrap_or("")).to_string_lossy(), "size": a["size"], "sha256": a["sha256"]})).collect();
        let (os, arch) = os_arch();
        let label = variants(sys, os, &arch)
            .into_iter()
            .find(|v| v["id"] == variant)
            .map(|v| v["label"].as_str().unwrap_or("").to_string())
            .unwrap_or_else(|| variant.to_string());
        let tag = plan["tag"].as_str().unwrap_or("");
        Ok(self.inner.downloads.add(
            &format!("{label} {tag}"),
            &files,
            "engine",
            json!({"variant": variant, "tag": tag, "cuda": plan["cuda"]}),
            &format!("engine:{variant}:{tag}"),
        ))
    }

    /// Download hook: unpack the archives into `engines/<variant>-<tag>/` and register the engine.
    fn install_downloaded(&self, job: &Map<String, Value>) -> Result<(), String> {
        let meta = &job["meta"];
        let name = py_name(&format!(
            "{}-{}",
            meta["variant"].as_str().unwrap_or(""),
            meta["tag"].as_str().unwrap_or("")
        ));
        let target = self.inner.dir.join(&name);
        let tmp = self.inner.dir.join(format!(".{name}.tmp"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
        let files = job["files"].as_array().cloned().unwrap_or_default();
        for f in &files {
            extract(Path::new(f["dest"].as_str().unwrap_or("")), &tmp)?;
        }
        let server = find_server(&tmp).ok_or("The engine archive has no llama-server in it")?;
        let rel = server.strip_prefix(&tmp).unwrap().to_path_buf();
        let bindir = server.parent().unwrap().to_path_buf();
        // CUDA runtime libraries land beside the server, wherever the archive put them.
        for lib in walk_files(&tmp) {
            let n = lib.file_name().unwrap().to_string_lossy().to_string();
            if lib.parent() != Some(bindir.as_path()) && is_cuda_lib(&n) {
                let _ = std::fs::rename(&lib, bindir.join(&n));
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for e in std::fs::read_dir(&bindir)
                .map_err(|e| e.to_string())?
                .flatten()
            {
                let p = e.path();
                let n = e.file_name().to_string_lossy().to_string();
                let suffix = p
                    .extension()
                    .map(|x| format!(".{}", x.to_string_lossy()))
                    .unwrap_or_default();
                if p.is_file() && (suffix.is_empty() || n.contains(".so") || suffix == ".dylib") {
                    let mode = std::fs::metadata(&p)
                        .map_err(|e| e.to_string())?
                        .permissions()
                        .mode();
                    let _ =
                        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode | 0o111));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&target);
        std::fs::rename(&tmp, &target).map_err(|e| e.to_string())?;
        let path = target.join(&rel);
        let info = self.detect(&path.to_string_lossy());
        let variant = meta["variant"].as_str().unwrap_or("");
        let kind = if variant.starts_with("fle") {
            "fle"
        } else {
            "upstream"
        };
        let entry = json!({"id": name, "variant": variant, "kind": kind, "flavor": info["flavor"], "backend": info["backend"], "tag": meta["tag"], "path": path.to_string_lossy(),
            "version": info.get("version").cloned().unwrap_or(json!("")), "installed": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0),
            "ok": info.get("ok").cloned().unwrap_or(json!(false)), "problem": info.get("problem").cloned().unwrap_or(json!(""))});
        {
            let mut items = self.inner.items.lock().unwrap();
            items.retain(|e| e["id"] != name.as_str());
            items.push(entry.clone());
            self.save(&items);
        }
        for f in &files {
            let _ = std::fs::remove_file(f["dest"].as_str().unwrap_or(""));
        }
        if entry["ok"] != json!(true) {
            return Err(format!(
                "Installed, but the engine does not start: {}",
                entry["problem"].as_str().unwrap_or("")
            ));
        }
        Ok(())
    }

    pub fn remove(&self, ident: &str) -> Result<(), EngineError> {
        {
            let mut items = self.inner.items.lock().unwrap();
            if !items.iter().any(|e| e["id"] == ident) {
                return Err(NotFound);
            }
            items.retain(|e| e["id"] != ident);
            self.save(&items);
        }
        let folder = self.inner.dir.join(ident);
        if folder.exists()
            && crate::library::py_resolve(&folder)
                .ancestors()
                .skip(1)
                .any(|a| a == crate::library::py_resolve(&self.inner.dir))
        {
            let _ = std::fs::remove_dir_all(folder);
        }
        Ok(())
    }

    /// Run `llama-server --version` and `--help` once per binary version: does it start, is it FLE, which backend.
    pub fn detect(&self, path: &str) -> Value {
        let p = match path.strip_prefix("~/") {
            Some(r) => dirs::home_dir()
                .unwrap_or_default()
                .join(crate::library::py_path(r)),
            None => crate::library::py_path(path),
        };
        if !p.is_file() {
            return json!({"ok": false, "problem": format!("No file at {}", p.display())});
        }
        let key = (p.to_string_lossy().to_string(), mtime(&p));
        if let Some(hit) = self.inner.detect_cache.lock().unwrap().get(&key) {
            return hit.clone();
        }
        let dir = p.parent().unwrap_or(Path::new(".")).to_path_buf();
        let ld = if cfg!(target_os = "linux") {
            Some(dir.as_path())
        } else {
            None
        };
        let (code, out) = run_capture(&p, &["--version"], 60, Some(&dir), ld);
        let (_, help) = run_capture(&p, &["--help"], 60, Some(&dir), ld);
        let ver = parse_version(&out);
        let siblings: String = std::fs::read_dir(&dir)
            .map(|r| {
                r.flatten()
                    .map(|e| e.file_name().to_string_lossy().to_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        let text = format!("{} {siblings}", out.to_lowercase());
        let backend = if text.contains("cuda") {
            "cuda"
        } else if text.contains("metal") || cfg!(target_os = "macos") {
            "metal"
        } else if text.contains("vulkan") {
            "vulkan"
        } else {
            "cpu"
        };
        let flavor = if help.contains("--moe-expert-cache-inserts") {
            "fle"
        } else {
            "upstream"
        };
        let ok = code == 0 && ver.is_some();
        let problem = if ok {
            String::new()
        } else {
            out.trim()
                .lines()
                .last()
                .unwrap_or("no output")
                .chars()
                .take(300)
                .collect()
        };
        let result = json!({"ok": ok, "version": ver.unwrap_or_default(), "backend": backend, "flavor": flavor, "log_verbosity": help.contains("--log-verbosity"), "problem": problem});
        self.inner
            .detect_cache
            .lock()
            .unwrap()
            .insert(key, result.clone());
        result
    }

    /// The engine a launch should use. `want`: auto | fle | upstream | custom.
    pub fn resolve(&self, want: &str, sys: &Value) -> Result<Value, EngineError> {
        let choice = self.setting("choice");
        if want == "custom" || (want == "auto" && choice == "custom") {
            let path = self.setting("custom_path");
            if path.is_empty() {
                return Err(Runtime(
                    "Set the path of your llama-server on the Engine page".into(),
                ));
            }
            let info = self.detect(&path);
            if info["ok"] != json!(true) {
                return Err(Runtime(format!(
                    "Your llama-server does not start: {}",
                    info["problem"].as_str().unwrap_or("")
                )));
            }
            let shown = match path.strip_prefix("~/") {
                Some(r) => dirs::home_dir().unwrap_or_default().join(r),
                None => PathBuf::from(&path),
            };
            return Ok(
                json!({"kind": "custom", "flavor": info["flavor"], "backend": info["backend"], "path": shown.to_string_lossy(), "version": info.get("version").cloned().unwrap_or(json!("")),
                "log_verbosity": info.get("log_verbosity").cloned().unwrap_or(json!(false))}),
            );
        }
        let want = if want == "auto" {
            if choice.is_empty() {
                "auto".to_string()
            } else {
                choice
            }
        } else {
            want.to_string()
        };
        let (os, arch) = os_arch();
        let order: Vec<String> = variants(sys, os, &arch)
            .iter()
            .map(|v| v["id"].as_str().unwrap_or("").to_string())
            .collect();
        let mut have = self.installed();
        have.sort_by(|a, b| {
            let rank = |e: &Value| {
                order
                    .iter()
                    .position(|o| e["variant"] == o.as_str())
                    .unwrap_or(99)
            };
            rank(a).cmp(&rank(b)).then(
                b["installed"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .partial_cmp(&a["installed"].as_f64().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        if want == "fle" || want == "upstream" {
            have.retain(|e| e["kind"] == want.as_str());
        }
        let Some(e) = have.first() else {
            return Err(Runtime("Install an engine first (Engine page)".into()));
        };
        // Engines installed here are recent releases, which all know -lv.
        Ok(
            json!({"kind": e["kind"], "flavor": if e["flavor"].is_null() || e["flavor"] == "" { e["kind"].clone() } else { e["flavor"].clone() }, "backend": e["backend"], "path": e["path"],
            "version": e.get("version").cloned().unwrap_or(json!("")), "id": e["id"], "log_verbosity": true}),
        )
    }
}

/// `version:\s*(\S+)(?:\s*\(([0-9a-f]+)\))?`
fn parse_version(text: &str) -> Option<String> {
    let at = text.find("version:")?;
    let rest = text[at + 8..].trim_start();
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let v = &rest[..end];
    if v.is_empty() {
        return None;
    }
    let after = rest[end..].trim_start();
    if let Some(r) = after.strip_prefix('(') {
        if let Some(close) = r.find(')') {
            let h = &r[..close];
            if !h.is_empty() && h.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
                return Some(format!("{v} ({h})"));
            }
        }
    }
    Some(v.to_string())
}

fn is_cuda_lib(name: &str) -> bool {
    let l = name.to_lowercase();
    let l = l.strip_prefix("lib").unwrap_or(&l);
    // (lib)?cu(da|blas|blasLt|rt)
    let l2 = name.to_lowercase();
    [l, l2.as_str()].iter().any(|s| {
        s.starts_with("cuda")
            || s.starts_with("cublas")
            || s.starts_with("cublaslt")
            || s.starts_with("curt")
    })
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    if let Ok(rd) = std::fs::read_dir(root) {
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                out.extend(walk_files(&p));
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

fn find_server(root: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let mut hits: Vec<PathBuf> = walk_files(root)
        .into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == name))
        .collect();
    hits.sort_by_key(|p| p.components().count());
    hits.into_iter().next()
}

fn lexical(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            o => out.push(o.as_os_str()),
        }
    }
    out
}

/// Unpack a .zip or .tar.gz safely: nothing may land outside `dest`.
fn extract(archive: &Path, dest: &Path) -> Result<(), String> {
    let root = lexical(&crate::library::py_resolve(dest));
    let inside = |p: &Path| {
        let t = lexical(&root.join(p));
        t.starts_with(&root)
    };
    let name = archive.file_name().unwrap().to_string_lossy().to_string();
    if name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(std::fs::File::open(archive).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        for i in 0..z.len() {
            let n = z.by_index(i).map_err(|e| e.to_string())?.name().to_string();
            if !inside(Path::new(&n)) {
                return Err(format!("unsafe path in archive: {n}"));
            }
        }
        return z.extract(dest).map_err(|e| e.to_string());
    }
    let open = || -> Result<tar::Archive<flate2::read::GzDecoder<std::fs::File>>, String> {
        Ok(tar::Archive::new(flate2::read::GzDecoder::new(
            std::fs::File::open(archive).map_err(|e| e.to_string())?,
        )))
    };
    for entry in open()?.entries().map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path().map_err(|e| e.to_string())?.to_path_buf();
        let shown = path.to_string_lossy().to_string();
        if !inside(&path) {
            return Err(format!("unsafe path in archive: {shown}"));
        }
        let t = entry.header().entry_type();
        if !(t.is_file() || t.is_dir() || t.is_symlink()) {
            return Err(format!("unsupported entry in archive: {shown}"));
        }
        if t.is_symlink() {
            let link = entry
                .link_name()
                .map_err(|e| e.to_string())?
                .unwrap_or_default();
            let parent = path.parent().unwrap_or(Path::new(""));
            if !inside(&parent.join(&link)) {
                return Err(format!("unsafe link in archive: {shown}"));
            }
        }
    }
    open()?.unpack(dest).map_err(|e| e.to_string())
}

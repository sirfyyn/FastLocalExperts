use crate::download::{client, fetch};
use crate::system::SysInfo;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The llama.cpp release FLE is tested with (same pin as FLE Studio).
pub const LLAMA_TAG: &str = "b11540";
const UPSTREAM: &str = "ggml-org/llama.cpp";
const FLE_REPO: &str = "sirfyyn/FastLocalExperts";

#[derive(Debug, Clone)]
pub struct Variant {
    pub id: &'static str,
    pub kind: &'static str, // fle | upstream
    pub backend: &'static str,
    pub label: &'static str,
    pub why: &'static str,
}

/// Engine choices for this machine, best first. Same rules as FLE Studio's engines.py.
pub fn variants(s: &SysInfo) -> Vec<Variant> {
    let mut v = Vec::new();
    let nvidia = s.gpus.iter().any(|g| g.vendor == "nvidia");
    if s.os == "macos" {
        if s.arch == "arm64" {
            v.push(Variant {
                id: "upstream-metal",
                kind: "upstream",
                backend: "metal",
                label: "llama.cpp for Apple Silicon (Metal)",
                why: "Runs the model on the Apple GPU with unified memory.",
            });
        } else {
            v.push(Variant {
                id: "upstream-macos-x64",
                kind: "upstream",
                backend: "cpu",
                label: "llama.cpp for Intel Macs",
                why: "CPU build for Intel Macs.",
            });
        }
        return v;
    }
    if nvidia && s.arch == "x64" {
        if s.avx2 {
            v.push(Variant {
                id: "fle-cuda",
                kind: "fle",
                backend: "cuda",
                label: "FLE engine (NVIDIA CUDA)",
                why: "The FLE build: GPU expert cache, MTP drafting, pinned expert loading.",
            });
        }
        v.push(Variant {
            id: "upstream-cuda",
            kind: "upstream",
            backend: "cuda",
            label: "llama.cpp (NVIDIA CUDA)",
            why: "Upstream llama.cpp on the NVIDIA GPU.",
        });
    }
    if nvidia || s.gpus.iter().any(|g| g.vendor == "other") {
        v.push(Variant {
            id: "upstream-vulkan",
            kind: "upstream",
            backend: "vulkan",
            label: "llama.cpp (Vulkan)",
            why: "Works on AMD, Intel and NVIDIA GPUs.",
        });
    }
    v.push(Variant {
        id: "upstream-cpu",
        kind: "upstream",
        backend: "cpu",
        label: "llama.cpp (CPU)",
        why: "Works everywhere; slow for big models.",
    });
    v
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
    digest: Option<String>,
}
#[derive(Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

fn release(repo: &str, tag: Option<&str>) -> Result<Release> {
    let url = match tag {
        Some(t) => format!("https://api.github.com/repos/{repo}/releases/tags/{t}"),
        None => format!("https://api.github.com/repos/{repo}/releases/latest"),
    };
    let r = client()?
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()?;
    if !r.status().is_success() {
        bail!("GitHub answered {} for {url}", r.status());
    }
    Ok(r.json()?)
}

fn pick<'a>(assets: &'a [Asset], v: &Variant, os: &str, arch: &str) -> Option<&'a Asset> {
    let want: Vec<&str> = match (v.backend, os, arch) {
        ("metal", "macos", "arm64") => vec!["macos-arm64"],
        ("cpu", "macos", "x64") => vec!["macos-x64"],
        ("cuda", "linux", _) => vec!["ubuntu", "cuda"],
        ("cuda", "windows", _) => vec!["win", "cuda"],
        ("vulkan", "linux", _) => vec!["ubuntu", "vulkan"],
        ("vulkan", "windows", _) => vec!["win", "vulkan"],
        ("cpu", "linux", _) => vec!["ubuntu-x64"],
        ("cpu", "windows", _) => vec!["win-cpu"],
        _ => return None,
    };
    assets
        .iter()
        .filter(|a| {
            (a.name.ends_with(".tar.gz") || a.name.ends_with(".zip"))
                && want.iter().all(|w| a.name.contains(w))
        })
        .filter(|a| !a.name.contains("cudart"))
        .max_by_key(|a| a.name.clone())
}

/// Where an installed engine's llama-server lives, if there is one.
pub fn installed(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir.join("engines")) {
        for e in rd.flatten() {
            if let Some(p) = find_server(&e.path()) {
                out.push((e.file_name().to_string_lossy().to_string(), p));
            }
        }
    }
    out.sort();
    out
}

pub fn find_server(root: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == name) {
                return Some(p);
            }
        }
    }
    None
}

/// Download and unpack an engine; returns the path of its llama-server. Checks the SHA-256 GitHub publishes.
pub fn install(
    dir: &Path,
    v: &Variant,
    s: &SysInfo,
    progress: &mut dyn FnMut(&str, u64, u64),
) -> Result<PathBuf> {
    let (repo, tag) = if v.kind == "fle" {
        (FLE_REPO, None)
    } else {
        (UPSTREAM, Some(LLAMA_TAG))
    };
    let rel = release(repo, tag)?;
    let asset = pick(&rel.assets, v, &s.os, &s.arch).with_context(|| {
        format!(
            "no {} build for {}-{} in {repo} {}",
            v.id, s.os, s.arch, rel.tag_name
        )
    })?;
    let target = dir
        .join("engines")
        .join(format!("{}-{}", v.id, rel.tag_name));
    if let Some(p) = find_server(&target) {
        return Ok(p);
    }
    let archive = dir.join("downloads").join(&asset.name);
    let sha = asset
        .digest
        .as_deref()
        .and_then(|d| d.strip_prefix("sha256:"));
    let label = format!("engine {}", asset.name);
    fetch(&asset.browser_download_url, &archive, sha, &mut |d, t| {
        progress(&label, d, t)
    })?;
    std::fs::create_dir_all(&target)?;
    extract(&archive, &target)?;
    let server = find_server(&target).context("llama-server not found in the archive")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&server, std::fs::Permissions::from_mode(0o755)).ok();
    }
    Ok(server)
}

fn safe(dest: &Path, rel: &Path) -> Result<PathBuf> {
    if rel.components().any(|c| {
        matches!(
            c,
            std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_)
        )
    }) {
        bail!("unsafe path in archive: {}", rel.display());
    }
    Ok(dest.join(rel))
}

fn extract(archive: &Path, dest: &Path) -> Result<()> {
    let name = archive.to_string_lossy();
    if name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(std::fs::File::open(archive)?)?;
        for i in 0..z.len() {
            let mut f = z.by_index(i)?;
            let out = safe(dest, Path::new(f.name()))?;
            if f.is_dir() {
                std::fs::create_dir_all(&out)?;
            } else {
                std::fs::create_dir_all(out.parent().unwrap())?;
                std::io::copy(&mut f, &mut std::fs::File::create(&out)?)?;
            }
        }
        return Ok(());
    }
    let mut t = tar::Archive::new(flate2::read::GzDecoder::new(std::fs::File::open(archive)?));
    for e in t.entries()? {
        let mut e = e?;
        let rel = e.path()?.to_path_buf();
        let out = safe(dest, &rel)?;
        if e.header().entry_type().is_symlink() || e.header().entry_type().is_hard_link() {
            let link = e.link_name()?.map(|l| l.to_path_buf()).unwrap_or_default();
            safe(out.parent().unwrap_or(dest), &link)?; // the link must stay inside too
        }
        if let Some(p) = out.parent() {
            std::fs::create_dir_all(p)?;
        }
        e.unpack(&out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{Gpu, SysInfo};

    fn s(os: &str, arch: &str, nvidia: bool, avx2: bool) -> SysInfo {
        SysInfo {
            os: os.into(),
            arch: arch.into(),
            cpu: "x".into(),
            cores: 8,
            ram_bytes: 64 << 30,
            disk_free_bytes: 100 << 30,
            avx2,
            gpus: if nvidia {
                vec![Gpu {
                    vendor: "nvidia".into(),
                    name: "g".into(),
                    vram_bytes: Some(32 << 30),
                }]
            } else {
                vec![]
            },
            gpu_usable_bytes: 0,
        }
    }
    #[test]
    fn apple_silicon_gets_metal_and_only_that() {
        let v = variants(&s("macos", "arm64", false, false));
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].id, "upstream-metal");
    }
    #[test]
    fn nvidia_with_avx2_gets_the_fle_engine_first() {
        let v = variants(&s("linux", "x64", true, true));
        assert_eq!(v[0].id, "fle-cuda");
        assert!(
            v.iter().any(|x| x.id == "upstream-cuda")
                && v.iter().any(|x| x.id == "upstream-vulkan")
        );
    }
    #[test]
    fn nvidia_without_avx2_skips_the_fle_engine() {
        assert_eq!(
            variants(&s("linux", "x64", true, false))[0].id,
            "upstream-cuda"
        );
    }
    #[test]
    fn no_gpu_gets_the_cpu_build() {
        assert_eq!(
            variants(&s("linux", "x64", false, true))[0].id,
            "upstream-cpu"
        );
    }
    #[test]
    fn archive_paths_must_stay_inside() {
        let d = Path::new("/tmp/x");
        assert!(safe(d, Path::new("llama/bin/llama-server")).is_ok());
        assert!(safe(d, Path::new("../evil")).is_err());
        assert!(safe(d, Path::new("a/../../evil")).is_err());
        assert!(safe(d, Path::new("/etc/passwd")).is_err());
    }
    #[test]
    fn asset_choice_matches_the_platform_and_skips_cudart() {
        let a = |n: &str| Asset {
            name: n.into(),
            browser_download_url: String::new(),
            digest: None,
        };
        let assets = vec![
            a("llama-b1-bin-macos-arm64.tar.gz"),
            a("llama-b1-bin-macos-x64.tar.gz"),
            a("llama-b1-bin-ubuntu-x64.tar.gz"),
            a("llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz"),
            a("cudart-llama-bin-win-cuda-12.4-x64.zip"),
            a("llama-b1-bin-win-cuda-12.4-x64.zip"),
        ];
        let m = variants(&s("macos", "arm64", false, false)).remove(0);
        assert_eq!(
            pick(&assets, &m, "macos", "arm64").unwrap().name,
            "llama-b1-bin-macos-arm64.tar.gz"
        );
        let c = variants(&s("windows", "x64", true, false))
            .into_iter()
            .find(|v| v.id == "upstream-cuda")
            .unwrap();
        assert_eq!(
            pick(&assets, &c, "windows", "x64").unwrap().name,
            "llama-b1-bin-win-cuda-12.4-x64.zip"
        );
    }
}

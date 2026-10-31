use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone)]
pub struct Gpu {
    pub vendor: String, // nvidia | apple | other
    pub name: String,
    pub vram_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SysInfo {
    pub os: String,   // macos | linux | windows
    pub arch: String, // arm64 | x64
    pub cpu: String,
    pub cores: usize,
    pub ram_bytes: u64,
    pub disk_free_bytes: u64,
    pub avx2: bool,
    pub gpus: Vec<Gpu>,
    /// Memory the GPU may use: unified memory limit on Apple Silicon, VRAM of the biggest NVIDIA card otherwise.
    pub gpu_usable_bytes: u64,
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).output().ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

pub fn detect(dir: &Path) -> SysInfo {
    let os = std::env::consts::OS.to_string();
    let arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        "x86_64" => "x64",
        a => a,
    }
    .to_string();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let (mut cpu, mut ram, mut avx2) = (String::new(), 0u64, false);
    if os == "macos" {
        cpu = run("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
        ram = run("sysctl", &["-n", "hw.memsize"])
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        avx2 = run("sysctl", &["-n", "machdep.cpu.leaf7_features"])
            .is_some_and(|s| s.contains("AVX2"));
    } else if os == "linux" {
        let mi = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        ram = mi
            .lines()
            .find(|l| l.starts_with("MemTotal"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024;
        let ci = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        cpu = ci
            .lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
            .unwrap_or("")
            .trim()
            .to_string();
        avx2 = ci
            .lines()
            .any(|l| l.starts_with("flags") && l.contains(" avx2"));
    }
    let mut gpus = Vec::new();
    if let Some(out) = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
    ) {
        for line in out.lines() {
            let mut p = line.split(',');
            let name = p.next().unwrap_or("").trim().to_string();
            let mib: u64 = p.next().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
            gpus.push(Gpu {
                vendor: "nvidia".into(),
                name,
                vram_bytes: Some(mib * 1024 * 1024),
            });
        }
    }
    let apple = os == "macos" && arch == "arm64";
    if apple {
        gpus.push(Gpu {
            vendor: "apple".into(),
            name: format!("{} GPU", cpu),
            vram_bytes: None,
        });
    }
    let gpu_usable_bytes = if apple {
        (ram as f64 * 0.667) as u64 // what Metal lets one process use (recommendedMaxWorkingSetSize is about 2/3)
    } else {
        gpus.iter().filter_map(|g| g.vram_bytes).max().unwrap_or(0)
    };
    let _ = std::fs::create_dir_all(dir);
    let disk_free_bytes = run("df", &["-k", &dir.to_string_lossy()])
        .and_then(|o| {
            o.lines().nth(1).and_then(|l| {
                l.split_whitespace()
                    .nth(3)
                    .and_then(|v| v.parse::<u64>().ok())
            })
        })
        .map(|k| k * 1024)
        .unwrap_or(0);
    SysInfo {
        os,
        arch,
        cpu,
        cores,
        ram_bytes: ram,
        disk_free_bytes,
        avx2,
        gpus,
        gpu_usable_bytes,
    }
}

pub fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / 1073741824.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gib_formats() {
        assert_eq!(gib(1 << 30), "1.0 GiB");
        assert_eq!(gib(3 << 29), "1.5 GiB");
    }
    #[test]
    fn detect_here_finds_memory_and_a_writable_disk() {
        let d = std::env::temp_dir().join("fle-test-system");
        let s = detect(&d);
        assert!(s.ram_bytes > 0 && s.cores > 0 && s.disk_free_bytes > 0);
        assert!(!s.os.is_empty() && !s.arch.is_empty());
    }
}

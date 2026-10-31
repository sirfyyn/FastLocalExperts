//! What this machine has: OS, CPU, memory, GPUs and disk. Port of `fle_studio/system.py`; macOS, Linux and Windows.
use fle_core::launch::{run_ok, which};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const MAC: bool = cfg!(target_os = "macos");
const WIN: bool = cfg!(windows);
const LINUX: bool = cfg!(target_os = "linux");

fn ps(script: &str, timeout: u64) -> String {
    run_ok("powershell", &["-NoProfile", "-Command", script], timeout)
}

fn first_int(s: &str) -> u64 {
    s.trim().parse().unwrap_or(0)
}

/// `{"total", "available"}` in bytes: physical memory, and what can be used without swapping.
pub fn memory() -> Value {
    let (total, avail) = if LINUX {
        let mut info: std::collections::HashMap<String, u64> = Default::default();
        match std::fs::read_to_string("/proc/meminfo") {
            Ok(t) => {
                for l in t.lines() {
                    if let Some((k, rest)) = l.split_once(':') {
                        if let Some(v) = rest
                            .split_whitespace()
                            .next()
                            .and_then(|v| v.parse::<u64>().ok())
                        {
                            info.insert(k.to_string(), v * 1024);
                        }
                    }
                }
                let g = |k: &str| info.get(k).copied();
                (
                    g("MemTotal").unwrap_or(0),
                    g("MemAvailable").or(g("MemFree")).unwrap_or(0),
                )
            }
            Err(_) => (0, 0),
        }
    } else if MAC {
        let total = first_int(&run_ok("sysctl", &["-n", "hw.memsize"], 5));
        let vm = run_ok("vm_stat", &[], 5);
        let page = vm
            .split("page size of ")
            .nth(1)
            .and_then(|r| r.split_whitespace().next())
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(16384);
        let pages = |name: &str| -> u64 {
            vm.lines()
                .find_map(|l| {
                    l.trim_start()
                        .strip_prefix(&format!("{name}:"))
                        .map(|r| r.trim().trim_end_matches('.').parse::<u64>().unwrap_or(0))
                })
                .unwrap_or(0)
        };
        let free = pages("Pages free")
            + pages("Pages inactive")
            + pages("Pages speculative")
            + pages("Pages purgeable");
        (total, total.min(free * page))
    } else if WIN {
        let out = ps("$o = Get-CimInstance Win32_OperatingSystem; \"$($o.TotalVisibleMemorySize) $($o.FreePhysicalMemory)\"", 8);
        let mut it = out
            .split_whitespace()
            .map(|x| x.parse::<u64>().unwrap_or(0));
        (it.next().unwrap_or(0) * 1024, it.next().unwrap_or(0) * 1024)
    } else {
        (0, 0)
    };
    json!({"total": total, "available": avail})
}

fn linux_physical_cores(text: &str) -> Option<u64> {
    let mut cores = std::collections::BTreeSet::new();
    for block in text.split("\n\n") {
        let get = |key: &str| {
            block.lines().find_map(|l| {
                l.strip_prefix(key)
                    .and_then(|r| r.trim_start().strip_prefix(':'))
                    .map(|v| v.trim().to_string())
            })
        };
        if let (Some(p), Some(c)) = (get("physical id"), get("core id")) {
            if p.parse::<u64>().is_ok() && c.parse::<u64>().is_ok() {
                cores.insert((p, c));
            }
        }
    }
    (!cores.is_empty()).then_some(cores.len() as u64)
}

/// `platform.machine().lower()`
fn machine() -> String {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "arm64".into(),
        ("windows", "x86_64") => "amd64".into(),
        ("windows", "aarch64") => "arm64".into(),
        (_, a) => a.to_string(),
    }
}

pub fn cpu() -> Value {
    static C: OnceLock<Value> = OnceLock::new();
    C.get_or_init(|| {
        let logical = std::thread::available_parallelism().map(|n| n.get() as u64).unwrap_or(1);
        let mut name = if MAC && cfg!(target_arch = "aarch64") { "arm".to_string() } else { machine() };
        let (mut physical, mut avx2): (Option<u64>, Option<bool>) = (None, None);
        if LINUX {
            let text = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            if let Some(n) = text.lines().find_map(|l| l.strip_prefix("model name").and_then(|r| r.trim_start().strip_prefix(':')).map(|v| v.trim().to_string())) {
                name = n;
            }
            physical = linux_physical_cores(&text);
            avx2 = Some(text.lines().find_map(|l| l.strip_prefix("flags").and_then(|r| r.trim_start().strip_prefix(':')).map(|v| format!(" {v}").contains(" avx2"))).unwrap_or(false));
        } else if MAC {
            let n = run_ok("sysctl", &["-n", "machdep.cpu.brand_string"], 5).trim().to_string();
            if !n.is_empty() {
                name = n;
            }
            let p = first_int(&run_ok("sysctl", &["-n", "hw.physicalcpu"], 5));
            physical = (p > 0).then_some(p);
            if machine() == "x86_64" {
                avx2 = Some(run_ok("sysctl", &["-n", "machdep.cpu.leaf7_features"], 5).contains("AVX2"));
            }
        } else if WIN {
            let n = ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name", 8).trim().to_string();
            if !n.is_empty() {
                name = n;
            }
            let p = first_int(&ps("(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum", 8));
            physical = (p > 0).then_some(p);
            #[cfg(target_arch = "x86_64")]
            {
                avx2 = Some(std::arch::is_x86_feature_detected!("avx2"));
            }
        }
        json!({"name": name, "arch": machine(), "logical": logical, "physical": physical.unwrap_or_else(|| (logical / 2).max(1)), "avx2": avx2})
    })
    .clone()
}

fn nvidia_smi() -> Option<PathBuf> {
    if let Some(p) = which("nvidia-smi") {
        return Some(p);
    }
    if WIN {
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let pf = std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".into());
        return [
            PathBuf::from(root).join("System32").join("nvidia-smi.exe"),
            PathBuf::from(pf)
                .join("NVIDIA Corporation")
                .join("NVSMI")
                .join("nvidia-smi.exe"),
        ]
        .into_iter()
        .find(|c| c.exists());
    }
    None
}

const NV_FIELDS: &str = "index,name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap,power.draw,temperature.gpu";

/// One line of `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`.
pub fn parse_nvidia(out: &str, fields: &[&str]) -> Vec<Value> {
    out.trim()
        .lines()
        .map(|line| {
            let row: Vec<(&str, String)> = fields.iter().cloned().zip(line.split(',').map(|v| v.trim().to_string())).collect();
            let get = |k: &str| row.iter().find(|(f, _)| *f == k).map(|(_, v)| v.clone());
            let num = |k: &str| get(k).and_then(|v| v.parse::<f64>().ok());
            let (total, used) = (num("memory.total").unwrap_or(0.0), num("memory.used").unwrap_or(0.0));
            let numv = |k: &str| num(k).map(|f| json!(f)).unwrap_or(Value::Null);
            json!({
                "vendor": "nvidia", "index": num("index").filter(|f| *f != 0.0).map(|f| f as i64).unwrap_or(0),
                "name": get("name").unwrap_or_else(|| "NVIDIA GPU".into()), "memory_total": (total * MIB as f64) as u64, "memory_used": (used * MIB as f64) as u64,
                "utilization": numv("utilization.gpu"), "power_w": numv("power.draw"), "temperature_c": numv("temperature.gpu"),
                "driver": get("driver_version").unwrap_or_default(), "compute_cap": get("compute_cap").unwrap_or_default(),
            })
        })
        .collect()
}

pub fn nvidia() -> Vec<Value> {
    let Some(smi) = nvidia_smi() else {
        return vec![];
    };
    let smi = smi.to_string_lossy().to_string();
    let mut fields: Vec<&str> = NV_FIELDS.split(',').collect();
    let mut out = run_ok(
        &smi,
        &[
            &format!("--query-gpu={NV_FIELDS}"),
            "--format=csv,noheader,nounits",
        ],
        5,
    );
    if out.is_empty() {
        // old drivers do not know compute_cap
        fields.retain(|f| *f != "compute_cap");
        out = run_ok(
            &smi,
            &[
                &format!("--query-gpu={}", fields.join(",")),
                "--format=csv,noheader,nounits",
            ],
            5,
        );
    }
    parse_nvidia(&out, &fields)
}

fn metal_limit(total: u64) -> u64 {
    let o = run_ok("sysctl", &["-n", "iogpu.wired_limit_mb"], 5)
        .trim()
        .to_string();
    if let Ok(n) = o.parse::<u64>() {
        if n > 0 {
            return n * MIB;
        }
    }
    (total as f64 * if total >= 64 * GIB { 0.75 } else { 0.67 }) as u64
}

pub fn apple(mem_total: u64) -> Vec<Value> {
    if !(MAC && machine() == "arm64") {
        return vec![];
    }
    let chip = run_ok("sysctl", &["-n", "machdep.cpu.brand_string"], 5)
        .trim()
        .to_string();
    let chip = if chip.is_empty() {
        "Apple Silicon".to_string()
    } else {
        chip
    };
    vec![
        json!({"vendor": "apple", "index": 0, "name": format!("{chip} GPU"), "unified": true, "memory_total": mem_total, "metal_limit": metal_limit(mem_total)}),
    ]
}

/// AMD and Intel GPUs, for the Vulkan engine. Names only; cached because the hardware does not change.
pub fn other_gpus() -> Vec<Value> {
    static C: OnceLock<Vec<Value>> = OnceLock::new();
    C.get_or_init(|| {
        let mut found: Vec<Value> = vec![];
        if LINUX {
            let mut cards: Vec<PathBuf> = std::fs::read_dir("/sys/class/drm").map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
            cards.sort();
            for card in cards {
                let n = card.file_name().unwrap().to_string_lossy().to_string();
                if !(n.starts_with("card") && n.len() == 5 && n.as_bytes()[4].is_ascii_digit()) {
                    continue;
                }
                let Ok(vendor) = std::fs::read_to_string(card.join("device/vendor")) else { continue };
                let v = match vendor.trim() {
                    "0x1002" => "amd",
                    "0x8086" => "intel",
                    _ => continue,
                };
                let total = std::fs::read_to_string(card.join("device/mem_info_vram_total")).ok().and_then(|t| t.trim().parse::<u64>().ok()).unwrap_or(0);
                found.push(json!({"vendor": v, "index": found.len(), "name": format!("{} GPU", v.to_uppercase()), "memory_total": total}));
            }
        } else if WIN {
            for name in ps("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }", 8).lines() {
                let name = name.trim();
                let low = name.to_lowercase();
                if !name.is_empty() && !low.starts_with("nvidia") && !low.starts_with("microsoft") {
                    let v = if low.contains("amd") || low.contains("radeon") { "amd" } else { "intel" };
                    found.push(json!({"vendor": v, "index": found.len(), "name": name, "memory_total": 0}));
                }
            }
        }
        found
    })
    .clone()
}

pub fn gpus(mem_total: u64) -> Vec<Value> {
    let n = nvidia();
    if n.is_empty() {
        apple(mem_total)
    } else {
        n
    }
}

pub fn disk_free(path: &Path) -> u64 {
    let mut probe = path.to_path_buf();
    while !probe.exists() && probe.parent().is_some_and(|p| p != probe) {
        probe = probe.parent().unwrap().to_path_buf();
    }
    if WIN {
        let letter = probe.to_string_lossy().chars().next().unwrap_or('C');
        return first_int(&ps(&format!("(Get-PSDrive -Name {letter}).Free"), 8));
    }
    run_ok("df", &["-Pk", &probe.to_string_lossy()], 5)
        .lines()
        .nth(1)
        .and_then(|l| {
            l.split_whitespace()
                .nth(3)
                .and_then(|v| v.parse::<u64>().ok())
        })
        .map(|k| k * 1024)
        .unwrap_or(0)
}

fn os_name() -> String {
    static C: OnceLock<String> = OnceLock::new();
    C.get_or_init(|| {
        if MAC {
            return format!(
                "macOS {}",
                run_ok("sw_vers", &["-productVersion"], 5).trim()
            );
        }
        if WIN {
            let v = ps("[Environment]::OSVersion.Version.ToString()", 8)
                .trim()
                .to_string();
            let build: u64 = v
                .split('.')
                .nth(2)
                .and_then(|b| b.parse().ok())
                .unwrap_or(0);
            return format!("Windows {} ({v})", if build >= 22000 { "11" } else { "10" });
        }
        std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|t| {
                t.lines().find_map(|l| {
                    l.strip_prefix("PRETTY_NAME=")
                        .map(|v| v.trim_matches('"').to_string())
                })
            })
            .unwrap_or_else(|| "Linux".into())
    })
    .clone()
}

/// The whole picture the page shows (`system.info`). `python` is empty: this server is not Python.
pub fn info(models_dir: &Path) -> Value {
    let mem = memory();
    let g = gpus(mem["total"].as_u64().unwrap_or(0));
    let other = if g.is_empty() { other_gpus() } else { vec![] };
    let mut m = Map::new();
    m.insert("os".into(), json!(os_name()));
    m.insert(
        "platform".into(),
        json!(if MAC {
            "macos"
        } else if WIN {
            "windows"
        } else {
            "linux"
        }),
    );
    m.insert("python".into(), json!(""));
    m.insert("cpu".into(), cpu());
    m.insert("memory".into(), mem);
    m.insert("gpus".into(), Value::Array(g));
    m.insert("other_gpus".into(), Value::Array(other));
    m.insert("disk_free".into(), json!(disk_free(models_dir)));
    m.insert("models_dir".into(), json!(models_dir.to_string_lossy()));
    Value::Object(m)
}

use crate::policy::FLE_POLICY_ENV;
use crate::system::SysInfo;
use std::path::Path;

/// A complete start: environment, arguments, and the one-line command a person could paste into a terminal.
/// Nobody has to type this; it is shown so nothing is hidden.
#[derive(Debug, Clone)]
pub struct Plan {
    pub env: Vec<(String, String)>,
    pub args: Vec<String>,
    pub port: u16,
    pub ctx: u32,
}

impl Plan {
    pub fn command_line(&self, server: &Path) -> String {
        let q = |s: &str| {
            if s.chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_./=:,".contains(c))
            {
                s.to_string()
            } else {
                format!("'{s}'")
            }
        };
        let mut out: Vec<String> = self
            .env
            .iter()
            .map(|(k, v)| format!("{k}={}", q(v)))
            .collect();
        out.push(q(&server.to_string_lossy()));
        out.extend(self.args.iter().map(|a| q(a)));
        out.join(" ")
    }
}

pub fn free_port(preferred: u16) -> u16 {
    if preferred != 0 && std::net::TcpListener::bind(("127.0.0.1", preferred)).is_ok() {
        return preferred;
    }
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(preferred)
}

/// Automatic settings for this machine. The context is sized from the memory the model leaves free; on the FLE engine
/// the policy variables are set for the user (the same table FLE Studio uses).
pub fn auto_plan(
    s: &SysInfo,
    engine_kind: &str,
    model: &Path,
    model_bytes: u64,
    alias: &str,
) -> Plan {
    let port = free_port(8080);
    let gpu = !s.gpus.is_empty();
    let have = if gpu && s.gpu_usable_bytes > 0 {
        s.gpu_usable_bytes
    } else {
        s.ram_bytes / 2
    };
    // about 150 MB of KV cache per 1k tokens is a safe upper bound for small dense models; MoE/linear-attention models need less
    let spare = have.saturating_sub((model_bytes as f64 * 1.15) as u64);
    let ctx = ((spare as f64 / 150e6) * 1024.0).clamp(2048.0, 32768.0) as u32;
    let ctx = ctx / 1024 * 1024;
    let mut args: Vec<String> = vec![
        "-m".into(),
        model.to_string_lossy().into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "-c".into(),
        ctx.to_string(),
        "--jinja".into(),
        "--metrics".into(),
        "-a".into(),
        alias.into(),
    ];
    if gpu {
        args.extend(["-ngl".into(), "99".into()]);
    }
    let mut env = Vec::new();
    if engine_kind == "fle" {
        env.extend(
            FLE_POLICY_ENV
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string())),
        );
        args.extend(["-fit".into(), "on".into()]);
    }
    Plan {
        env,
        args,
        port,
        ctx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{Gpu, SysInfo};
    use std::path::Path;

    fn sys(gpu: bool, ram_gib: u64, usable_gib: u64) -> SysInfo {
        SysInfo {
            os: "linux".into(),
            arch: "x64".into(),
            cpu: "x".into(),
            cores: 8,
            ram_bytes: ram_gib << 30,
            disk_free_bytes: 100 << 30,
            avx2: true,
            gpus: if gpu {
                vec![Gpu {
                    vendor: "nvidia".into(),
                    name: "g".into(),
                    vram_bytes: Some(usable_gib << 30),
                }]
            } else {
                vec![]
            },
            gpu_usable_bytes: usable_gib << 30,
        }
    }

    #[test]
    fn gpu_machine_offloads_everything_and_sets_no_fle_variables_for_upstream() {
        let p = auto_plan(
            &sys(true, 16, 8),
            "upstream",
            Path::new("/m/a.gguf"),
            800_000_000,
            "a",
        );
        assert!(p.args.windows(2).any(|w| w[0] == "-ngl" && w[1] == "99"));
        assert!(p.env.is_empty());
        assert!(p.args.iter().any(|a| a == "--jinja"));
    }
    #[test]
    fn cpu_machine_does_not_ask_for_gpu_layers() {
        let p = auto_plan(
            &sys(false, 16, 0),
            "upstream",
            Path::new("/m/a.gguf"),
            800_000_000,
            "a",
        );
        assert!(!p.args.iter().any(|a| a == "-ngl"));
    }
    #[test]
    fn the_fle_engine_gets_its_policy_variables_without_the_user_typing_them() {
        let p = auto_plan(
            &sys(true, 128, 32),
            "fle",
            Path::new("/m/a.gguf"),
            88_000_000_000,
            "fle",
        );
        assert!(p.env.iter().any(|(k, v)| k == "FLE_MC_OVERLAP" && v == "2"));
        assert!(p.env.iter().any(|(k, _)| k == "FLE_MOE_CACHE_BUDGET_FIT"));
        assert!(p.args.windows(2).any(|w| w[0] == "-fit" && w[1] == "on"));
    }
    #[test]
    fn context_is_clamped_and_a_multiple_of_1024() {
        for (ram, usable, size) in [
            (8, 5, 800_000_000u64),
            (8, 5, 5_000_000_000),
            (192, 32, 1_000_000_000),
        ] {
            let p = auto_plan(
                &sys(true, ram, usable),
                "upstream",
                Path::new("/m/a.gguf"),
                size,
                "a",
            );
            assert!((2048..=32768).contains(&p.ctx), "{}", p.ctx);
            assert_eq!(p.ctx % 1024, 0);
        }
    }
    #[test]
    fn command_line_quotes_what_needs_quoting() {
        let p = Plan {
            env: vec![("A".into(), "x y".into())],
            args: vec!["-m".into(), "/My Models/a.gguf".into()],
            port: 1,
            ctx: 2048,
        };
        let c = p.command_line(Path::new("/opt/llama-server"));
        assert_eq!(c, "A='x y' /opt/llama-server -m '/My Models/a.gguf'");
    }
    #[test]
    fn free_port_returns_something_bindable() {
        let p = free_port(0);
        assert!(p > 0);
    }
}

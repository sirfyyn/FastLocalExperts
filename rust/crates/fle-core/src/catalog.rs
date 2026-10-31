use crate::system::SysInfo;

#[derive(Debug, Clone)]
pub struct Model {
    pub id: &'static str,
    pub title: &'static str,
    pub note: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub sha256: Option<&'static str>,
    /// Needs the FLE engine (experts in RAM, hot ones cached on the GPU).
    pub fle_only: bool,
}

pub fn builtin() -> Vec<Model> {
    vec![
        Model {
            id: "llama-3.2-1b",
            title: "Llama 3.2 1B Instruct (Q4_K_M)",
            note: "Tiny, runs on any laptop. Good for trying FLE out.",
            repo: "bartowski/Llama-3.2-1B-Instruct-GGUF",
            file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            size: 807_694_464,
            sha256: Some("6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"),
            fle_only: false,
        },
        Model {
            id: "llama-3.2-1b-q8",
            title: "Llama 3.2 1B Instruct (Q8_0)",
            note: "Same model, higher precision.",
            repo: "bartowski/Llama-3.2-1B-Instruct-GGUF",
            file: "Llama-3.2-1B-Instruct-Q8_0.gguf",
            size: 1_321_083_008,
            sha256: Some("432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"),
            fle_only: false,
        },
        Model {
            id: "fle-iq3-m",
            title: "Qwen3.8-Flash-Next FLE-IQ3_M (125B MoE)",
            note: "The FLE showcase: ~90 tok/s on one 32 GB GPU. Needs ~120 GB RAM.",
            repo: "sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF",
            file: "IQ3_M/Qwen3.8-Flash-Next-FLE-IQ3_M-00001-of-00002.gguf",
            size: 88_870_000_000,
            sha256: None,
            fle_only: true,
        },
    ]
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fit {
    Fits,
    Tight,
    TooBig,
}

impl Fit {
    pub fn label(self) -> &'static str {
        match self {
            Fit::Fits => "fits",
            Fit::Tight => "tight",
            Fit::TooBig => "too big",
        }
    }
}

/// Fit of a model on this machine, before anything is downloaded. Dense models must fit the memory the GPU may use
/// (unified memory on a Mac) with room for the KV cache; FLE-only models keep their experts in RAM.
pub fn fit(m: &Model, s: &SysInfo) -> Fit {
    let need = (m.size as f64 * 1.25) as u64; // weights plus KV cache and compute buffers
    let have = if m.fle_only {
        s.ram_bytes
    } else {
        s.gpu_usable_bytes.max(s.ram_bytes / 2)
    };
    if m.size + 512 * 1024 * 1024 > s.disk_free_bytes && s.disk_free_bytes > 0 {
        return Fit::TooBig;
    }
    if need <= have * 8 / 10 {
        Fit::Fits
    } else if need <= have {
        Fit::Tight
    } else {
        Fit::TooBig
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{Gpu, SysInfo};

    fn mac_m2() -> SysInfo {
        SysInfo {
            os: "macos".into(),
            arch: "arm64".into(),
            cpu: "Apple M2".into(),
            cores: 8,
            ram_bytes: 8 << 30,
            disk_free_bytes: 18 << 30,
            avx2: false,
            gpus: vec![Gpu {
                vendor: "apple".into(),
                name: "Apple M2 GPU".into(),
                vram_bytes: None,
            }],
            gpu_usable_bytes: (8u64 << 30) * 2 / 3,
        }
    }
    fn workstation(ram_gib: u64) -> SysInfo {
        SysInfo {
            os: "linux".into(),
            arch: "x64".into(),
            cpu: "x".into(),
            cores: 16,
            ram_bytes: ram_gib << 30,
            disk_free_bytes: 500 << 30,
            avx2: true,
            gpus: vec![Gpu {
                vendor: "nvidia".into(),
                name: "RTX".into(),
                vram_bytes: Some(32 << 30),
            }],
            gpu_usable_bytes: 32 << 30,
        }
    }
    fn get(id: &str) -> Model {
        builtin().into_iter().find(|m| m.id == id).unwrap()
    }

    #[test]
    fn small_model_fits_a_laptop_and_the_big_one_does_not() {
        assert_eq!(fit(&get("llama-3.2-1b"), &mac_m2()), Fit::Fits);
        assert_eq!(fit(&get("fle-iq3-m"), &mac_m2()), Fit::TooBig);
    }
    #[test]
    fn fle_model_needs_the_ram_not_the_vram() {
        assert_eq!(fit(&get("fle-iq3-m"), &workstation(64)), Fit::TooBig);
        assert_eq!(fit(&get("fle-iq3-m"), &workstation(128)), Fit::Tight);
        assert_eq!(fit(&get("fle-iq3-m"), &workstation(192)), Fit::Fits);
    }
    #[test]
    fn a_full_disk_is_too_big_whatever_the_memory() {
        let mut s = workstation(192);
        s.disk_free_bytes = 10 << 30;
        assert_eq!(fit(&get("fle-iq3-m"), &s), Fit::TooBig);
    }
    #[test]
    fn catalog_has_unique_ids_and_known_checksums_are_64_hex() {
        let c = builtin();
        let mut ids: Vec<_> = c.iter().map(|m| m.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), c.len());
        for m in c.iter().filter_map(|m| m.sha256) {
            assert_eq!(m.len(), 64);
            assert!(m.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}

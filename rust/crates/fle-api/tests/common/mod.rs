//! Shared by the Python-comparison tests: tiny GGUF files and fake llama-server scripts.
#![allow(dead_code)]
use std::path::{Path, PathBuf};

pub enum V {
    U32(u32),
    Str(&'static str),
    U32List(Vec<u32>),
    F32(f32),
    Bool(bool),
}
fn s(b: &mut Vec<u8>, t: &str) {
    b.extend((t.len() as u64).to_le_bytes());
    b.extend(t.as_bytes());
}
pub fn write(path: &Path, kv: &[(&str, V)], tensors: &[(String, Vec<u64>)]) {
    let mut h = Vec::new();
    h.extend(b"GGUF");
    h.extend(3u32.to_le_bytes());
    h.extend((tensors.len() as u64).to_le_bytes());
    h.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        s(&mut h, k);
        match v {
            V::U32(x) => {
                h.extend(4u32.to_le_bytes());
                h.extend(x.to_le_bytes());
            }
            V::Str(t) => {
                h.extend(8u32.to_le_bytes());
                s(&mut h, t);
            }
            V::U32List(l) => {
                h.extend(9u32.to_le_bytes());
                h.extend(4u32.to_le_bytes());
                h.extend((l.len() as u64).to_le_bytes());
                for x in l {
                    h.extend(x.to_le_bytes());
                }
            }
            V::F32(x) => {
                h.extend(6u32.to_le_bytes());
                h.extend(x.to_le_bytes());
            }
            V::Bool(b) => {
                h.extend(7u32.to_le_bytes());
                h.push(u8::from(*b));
            }
        }
    }
    let mut off = 0u64;
    for (n, dims) in tensors {
        s(&mut h, n);
        h.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            h.extend(d.to_le_bytes());
        }
        h.extend(1u32.to_le_bytes());
        h.extend(off.to_le_bytes());
        off += (2 * dims.iter().product::<u64>()).div_ceil(32) * 32;
    }
    while h.len() % 32 != 0 {
        h.push(0);
    }
    h.extend(std::iter::repeat_n(0u8, off as usize));
    std::fs::write(path, h).unwrap();
}
pub fn kv(arch: &'static str, name: &'static str, experts: bool) -> Vec<(&'static str, V)> {
    // keys carry the architecture; leak the formatted names so they live long enough
    let k = |t: &str| -> &'static str { Box::leak(format!("{arch}.{t}").into_boxed_str()) };
    let mut v = vec![
        ("general.architecture", V::Str(arch)),
        ("general.name", V::Str(name)),
        (k("block_count"), V::U32(4)),
        (k("context_length"), V::U32(40960)),
        (k("attention.head_count"), V::U32(16)),
        (k("attention.head_count_kv"), V::U32(4)),
        (k("embedding_length"), V::U32(1024)),
        ("general.file_type", V::U32(1)),
    ];
    if experts {
        v.push((k("expert_count"), V::U32(64)));
        v.push((k("expert_used_count"), V::U32(4)));
    }
    v
}
pub fn tensors(experts: bool) -> Vec<(String, Vec<u64>)> {
    let mut t = vec![("token_embd.weight".to_string(), vec![64, 32])];
    for l in 0..4 {
        t.push((format!("blk.{l}.attn_q.weight"), vec![64, 64]));
        if experts {
            t.push((format!("blk.{l}.ffn_up_exps.weight"), vec![16, 32, 8]));
            t.push((format!("blk.{l}.ffn_down_exps.weight"), vec![32, 16, 8]));
        } else {
            t.push((format!("blk.{l}.ffn_up.weight"), vec![64, 128]));
        }
    }
    t.push(("output.weight".to_string(), vec![64, 32]));
    t
}

/// A scratch folder that is removed when the test ends, also when it fails.
pub struct Tmp(pub PathBuf);
impl std::ops::Deref for Tmp {
    type Target = PathBuf;
    fn deref(&self) -> &PathBuf {
        &self.0
    }
}
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
pub fn tmp(n: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("fle-launch-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

pub fn engine(dir: &Path, name: &str, fle: bool, spec: bool) -> String {
    let p = dir.join(name);
    let mut help = String::from("-ngl N\n--jinja\n-cmoe, --cpu-moe\nCUDA devices\n");
    if fle {
        help += "--moe-expert-cache-inserts N\n--fit-target MiB\n--load-mode MODE\n";
    }
    if spec {
        help += "--spec-type TYPE\n";
    }
    let script = format!("#!/bin/sh\ncase \"$1\" in\n  --version) echo 'version: 6001 (abc123f)' ;;\n  --help) cat <<'EOT'\n{help}EOT\n  ;;\n  --list-devices) echo 'ggml_cuda_init: found 1 CUDA devices' ;;\nesac\n");
    std::fs::write(&p, script).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p.to_string_lossy().to_string()
}

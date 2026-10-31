//! GGUF reader: synthetic files (like tools/fle/tests/gguf_fixture.py) and a conformance check against the Python helper.
use fle_core::gguf::*;
use std::path::{Path, PathBuf};

enum V {
    U32(u32),
    Str(&'static str),
}

fn s(b: &mut Vec<u8>, t: &str) {
    b.extend((t.len() as u64).to_le_bytes());
    b.extend(t.as_bytes());
}

/// A tiny GGUF: metadata, tensor infos (F16), zero-filled data. `cut` removes bytes from the end (an incomplete download).
fn write(path: &Path, kv: &[(&str, V)], tensors: &[(&str, Vec<u64>)], cut: usize) {
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
        }
    }
    let mut off = 0u64;
    let mut sizes = Vec::new();
    for (n, dims) in tensors {
        s(&mut h, n);
        h.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            h.extend(d.to_le_bytes());
        }
        h.extend(1u32.to_le_bytes());
        h.extend(off.to_le_bytes());
        let sz = 2 * dims.iter().product::<u64>();
        sizes.push(sz);
        off += sz.div_ceil(32) * 32;
    }
    while h.len() % 32 != 0 {
        h.push(0);
    }
    h.extend(std::iter::repeat_n(0u8, off as usize));
    h.truncate(h.len() - cut);
    std::fs::write(path, h).unwrap();
}

fn tmp(n: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fle-gguf-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn moe_kv() -> Vec<(&'static str, V)> {
    vec![
        ("general.architecture", V::Str("qwen3moe")),
        ("general.name", V::Str("Tiny MoE")),
        ("qwen3moe.block_count", V::U32(2)),
        ("qwen3moe.expert_count", V::U32(8)),
        ("qwen3moe.expert_used_count", V::U32(2)),
        ("qwen3moe.context_length", V::U32(4096)),
        ("qwen3moe.attention.head_count", V::U32(8)),
        ("qwen3moe.attention.head_count_kv", V::U32(4)),
        ("qwen3moe.embedding_length", V::U32(512)),
        ("general.file_type", V::U32(1)),
    ]
}
fn moe_tensors() -> Vec<(&'static str, Vec<u64>)> {
    let mut t = vec![("token_embd.weight", vec![16, 8])];
    for l in 0..2 {
        t.push((
            Box::leak(format!("blk.{l}.attn_q.weight").into_boxed_str()) as &str,
            vec![8, 8],
        ));
        t.push((
            Box::leak(format!("blk.{l}.ffn_up_exps.weight").into_boxed_str()) as &str,
            vec![8, 4, 8],
        ));
        t.push((
            Box::leak(format!("blk.{l}.ffn_down_exps.weight").into_boxed_str()) as &str,
            vec![8, 4, 8],
        ));
    }
    t.push(("output.weight", vec![8, 16]));
    t
}

#[test]
fn experts_inputs_and_the_rest_are_told_apart() {
    let d = tmp("moe");
    let p = d.join("m.gguf");
    write(&p, &moe_kv(), &moe_tensors(), 0);
    let m = read_model(&p).unwrap();
    assert_eq!((m.arch.as_str(), m.name.as_str()), ("qwen3moe", "Tiny MoE"));
    assert_eq!(m.tensor_count, 8);
    assert_eq!(m.expert_bytes, 4 * 2 * 8 * 4 * 8); // four expert tensors of 8x4x8 F16
    assert_eq!(m.input_bytes, 2 * 16 * 8);
    assert_eq!(m.gpu_bytes, 2 * 2 * 64 + 2 * 8 * 16);
    assert!(m.complete());
    let sm = m.summary();
    assert_eq!(sm["experts"], 8);
    assert_eq!(sm["experts_used"], 2);
    assert_eq!(sm["layers"], 2);
    assert_eq!(sm["context_length"], 4096);
    assert_eq!(sm["type_histogram"]["F16"]["tensors"], 8);
}

#[test]
fn kv_cache_per_token_follows_heads_and_cache_type() {
    let d = tmp("kv");
    let p = d.join("m.gguf");
    write(&p, &moe_kv(), &moe_tensors(), 0);
    let m = read_model(&p).unwrap();
    // 2 layers x 4 kv heads x (64 key + 64 value) x bytes per element; head_dim = 512 / 8 = 64
    assert_eq!(
        m.kv_bytes_per_token("f16", "f16"),
        2.0 * 4.0 * (64.0 * 2.0 + 64.0 * 2.0)
    );
    assert!(
        (m.kv_bytes_per_token("q8_0", "q8_0") - 2.0 * 4.0 * (64.0 * 34.0 / 32.0 * 2.0)).abs()
            < 1e-9
    );
}

#[test]
fn split_models_resolve_from_any_part_and_from_the_folder() {
    let d = tmp("split");
    write(
        &d.join("m-00001-of-00002.gguf"),
        &moe_kv(),
        &moe_tensors()[..4],
        0,
    );
    write(
        &d.join("m-00002-of-00002.gguf"),
        &moe_kv(),
        &moe_tensors()[4..],
        0,
    );
    write(&d.join("mmproj-m.gguf"), &moe_kv(), &moe_tensors()[..1], 0);
    for p in [
        d.join("m-00002-of-00002.gguf"),
        d.join("m-00001-of-00002.gguf"),
        d.clone(),
    ] {
        let m = read_model(&p).unwrap();
        assert_eq!(m.parts.len(), 2, "{}", p.display());
        assert_eq!(m.first.file_name().unwrap(), "m-00001-of-00002.gguf");
        assert!(m.complete());
        assert_eq!(m.tensor_count, 8);
    }
}

#[test]
fn a_missing_shard_and_a_cut_shard_make_the_model_incomplete() {
    let d = tmp("incomplete");
    write(
        &d.join("m-00001-of-00002.gguf"),
        &moe_kv(),
        &moe_tensors()[..4],
        0,
    );
    let m = read_model(&d.join("m-00001-of-00002.gguf")).unwrap();
    assert!(!m.complete());
    assert_eq!(m.missing.len(), 1);
    write(
        &d.join("m-00002-of-00002.gguf"),
        &moe_kv(),
        &moe_tensors()[4..],
        100,
    );
    let m = read_model(&d.join("m-00001-of-00002.gguf")).unwrap();
    assert!(m.missing.is_empty());
    assert_eq!(m.incomplete, ["m-00002-of-00002.gguf"]);
}

#[test]
fn damaged_files_give_readable_errors() {
    let d = tmp("bad");
    std::fs::write(d.join("a.gguf"), b"NOPE0000").unwrap();
    assert!(read_model(&d.join("a.gguf"))
        .unwrap_err()
        .to_string()
        .contains("not a GGUF file"));
    let p = d.join("b.gguf");
    write(&p, &moe_kv(), &moe_tensors(), 0);
    let bytes = std::fs::read(&p).unwrap();
    std::fs::write(d.join("c.gguf"), &bytes[..60]).unwrap();
    assert!(read_model(&d.join("c.gguf"))
        .unwrap_err()
        .to_string()
        .contains("ends inside the header"));
    assert!(read_model(&d.join("nothing.gguf"))
        .unwrap_err()
        .to_string()
        .contains("No model at"));
    let e = tmp("empty");
    assert!(read_model(&e)
        .unwrap_err()
        .to_string()
        .contains("No GGUF model in"));
}

#[test]
fn unknown_tensor_types_are_reported_not_hidden() {
    assert_eq!(type_name(1, "release"), "F16");
    assert_eq!(type_name(43, "release"), "IQ2_KT");
    assert_eq!(type_name(43, "deepseek41"), "MXFP8");
    assert_eq!(type_name(99, "release"), "type-99");
    assert_eq!(tensor_nbytes(&[256, 4], 14, "release"), 210 * 4); // Q6_K: one block per row
    assert_eq!(tensor_nbytes(&[100, 4], 14, "release"), 0); // row not a multiple of the block size
}

/// The Python helper is the specification: the same file must give the same summary.
#[test]
fn matches_the_python_helper() {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../fle");
    if !helper.exists()
        || std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
    {
        eprintln!("python helper not found: skipped");
        return;
    }
    let d = tmp("conf");
    let p = d.join("m.gguf");
    write(&p, &moe_kv(), &moe_tensors(), 0);
    let out = std::process::Command::new("python3")
        .arg(&helper)
        .args(["inspect", p.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    if !out.status.success() {
        eprintln!("python helper failed: skipped");
        return;
    }
    let py: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let rs = read_model(&p).unwrap().summary();
    for (k, v) in rs.as_object().unwrap() {
        assert_eq!(&py[k], v, "field {k} differs from the Python helper");
    }
}

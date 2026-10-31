//! Studio's GGUF reader against the Python one (`fle_studio/gguf.py`): same files, same numbers, same error texts.
#![cfg(unix)]
mod common;
use common::*;
use fle_api::gguf::{inspect, inspect_cached, split_match, split_parts};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const DRIVER: &str = r#"
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fle_studio import gguf
sc = json.load(open(sys.argv[2]))
out = {"files": [], "split": []}
for p in sc["files"]:
    try:
        i = gguf.inspect_cached(p)
        out["files"].append({"public": i.public(), "size": i.size, "missing": i.missing, "parts": [Path(x).name for x in i.parts]})
    except (gguf.GGUFError, OSError) as e:
        out["files"].append({"error": str(e) if isinstance(e, gguf.GGUFError) else "OSError"})
for n in sc["names"]:
    m = gguf.SPLIT_RE.match(n)
    out["split"].append(None if not m else [m.group("base"), m.group("no"), m.group("count")])
print(json.dumps(out))
"#;

/// A scratch folder that is removed when the test ends, also when it fails.
struct Tmp(PathBuf);
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
fn tmp(n: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("fle-gguf-api-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

fn src() -> Option<PathBuf> {
    let h = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    std::env::var("FLE_STUDIO_SRC")
        .map(PathBuf::from)
        .ok()
        .into_iter()
        .chain([h.join("fle-studio-020/src"), h.join("fle-studio/src")])
        .find(|p| p.join("fle_studio/gguf.py").exists())
}

#[test]
fn studio_gguf_matches_the_python_reader() {
    let d = tmp("m");
    let p = |n: &str| d.join(n);
    write(
        &p("moe.gguf"),
        &kv("qwen3moe", "Tiny MoE", true),
        &tensors(true),
    );
    write(
        &p("dense.gguf"),
        &kv("qwen3", "Tiny Dense", false),
        &tensors(false),
    );
    write(&p("noname.gguf"), &kv("qwen3", "", false), &tensors(false));
    // per-layer kv heads (a list), a float and a bool among the scalars, hybrid attention
    let mut hyb = kv("qwen3next", "Hybrid", true);
    hyb.retain(|(k, _)| !k.ends_with("head_count_kv"));
    hyb.push((
        "qwen3next.attention.head_count_kv",
        V::U32List(vec![2, 2, 4, 4]),
    ));
    hyb.push(("qwen3next.full_attention_interval", V::U32(2)));
    hyb.push(("qwen3next.rope.freq_base", V::F32(1e-5)));
    hyb.push(("qwen3next.flag", V::Bool(true)));
    hyb.push(("tokenizer.chat_template", V::Str("{{ x }}")));
    write(&p("hybrid.gguf"), &hyb, &tensors(true));
    let mut keyed = kv("llama", "Keyed", false);
    keyed.push(("llama.attention.key_length", V::U32(96)));
    keyed.push(("llama.attention.value_length", V::U32(80)));
    write(&p("keyed.gguf"), &keyed, &tensors(false));
    let mut nohead = kv("llama", "NoHeads", false);
    nohead.retain(|(k, _)| !k.contains("head_count") && !k.contains("embedding_length"));
    write(&p("noheads.gguf"), &nohead, &tensors(false));
    let mut tabled = kv("qwen3moe", "Tables", true);
    tabled.push(("general.alignment", V::U32(64)));
    let mut t = tensors(true);
    t.push(("per_layer_token_embd.weight".into(), vec![64, 64]));
    t.push(("engram.table.weight".into(), vec![32, 32]));
    t.push(("blk.0.ffn_gate_up_exps.weight".into(), vec![16, 16, 8]));
    t.push(("blk.0.ffn_down_shexp.weight".into(), vec![16, 16]));
    write(&p("tables.gguf"), &tabled, &t);
    // a split model: complete, one part missing, first part missing
    for i in 1..=3 {
        write(
            &p(&format!("big-0000{i}-of-00003.gguf")),
            &kv("qwen3moe", "Big", true),
            &tensors(true),
        );
    }
    for i in [1, 3] {
        write(
            &p(&format!("gap-0000{i}-of-00003.gguf")),
            &kv("qwen3moe", "Gap", true),
            &tensors(true),
        );
    }
    write(
        &p("nofirst-00002-of-00002.gguf"),
        &kv("qwen3moe", "NoFirst", true),
        &tensors(true),
    );
    write(
        &p("UPPER-00001-OF-00002.GGUF"),
        &kv("qwen3moe", "", true),
        &tensors(true),
    );
    write(
        &p("UPPER-00002-OF-00002.GGUF"),
        &kv("qwen3moe", "", true),
        &tensors(true),
    );
    // damaged files
    std::fs::write(p("garbage.gguf"), b"this is not a gguf file").unwrap();
    std::fs::write(p("short.gguf"), b"GG").unwrap();
    std::fs::write(p("empty.gguf"), b"").unwrap();
    let whole = std::fs::read(p("moe.gguf")).unwrap();
    std::fs::write(p("cut-header.gguf"), &whole[..60]).unwrap();
    std::fs::write(p("cut-data.gguf"), &whole[..whole.len() - 4096]).unwrap();
    let mut v1 = whole.clone();
    v1[4..8].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(p("v1.gguf"), v1).unwrap();
    let mut huge = whole.clone();
    huge[8..16].copy_from_slice(&5_000_000u64.to_le_bytes());
    std::fs::write(p("huge.gguf"), huge).unwrap();
    let mut v2 = whole.clone();
    v2[4..8].copy_from_slice(&2u32.to_le_bytes());
    std::fs::write(p("v2.gguf"), v2).unwrap();

    let names = [
        "moe.gguf",
        "dense.gguf",
        "noname.gguf",
        "hybrid.gguf",
        "keyed.gguf",
        "noheads.gguf",
        "tables.gguf",
        "big-00001-of-00003.gguf",
        "big-00002-of-00003.gguf",
        "gap-00001-of-00003.gguf",
        "nofirst-00002-of-00002.gguf",
        "nope-00001-of-00002.gguf",
        "UPPER-00001-OF-00002.GGUF",
        "garbage.gguf",
        "short.gguf",
        "empty.gguf",
        "cut-header.gguf",
        "cut-data.gguf",
        "v1.gguf",
        "huge.gguf",
        "v2.gguf",
        "missing.gguf",
    ];
    let files: Vec<String> = names
        .iter()
        .map(|n| p(n).to_string_lossy().to_string())
        .collect();
    let split_names = [
        "a-00001-of-00002.gguf",
        "A-00001-OF-00002.GGUF",
        "a.b-c-00010-of-00012.gguf",
        "a-0001-of-00002.gguf",
        "-00001-of-00002.gguf",
        "x-00001-of-00002.gguf.part",
        "x-00001-of-00002",
        "model-q4.gguf",
        "ü-00001-of-00002.gguf",
        "a-00001-of-000020.gguf",
        ".gguf",
        "0-00001-of-00001.gguf",
    ];
    let Some(srcdir) = src() else {
        eprintln!("FLE Studio sources not found: skipped");
        return;
    };
    std::fs::write(d.join("driver.py"), DRIVER).unwrap();
    std::fs::write(
        d.join("in.json"),
        json!({"files": files, "names": split_names}).to_string(),
    )
    .unwrap();
    let out = std::process::Command::new(
        std::env::var("FLE_STUDIO_PYTHON").unwrap_or_else(|_| "python3".into()),
    )
    .arg(d.join("driver.py"))
    .arg(&srcdir)
    .arg(d.join("in.json"))
    .output()
    .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py: Value = serde_json::from_slice(&out.stdout).unwrap();
    for (i, f) in files.iter().enumerate() {
        let want = &py["files"][i];
        let got = inspect_cached(Path::new(f));
        match (got, want.get("error")) {
            (Ok(m), None) => {
                assert_eq!(m.public(), want["public"], "{f}");
                assert_eq!(json!(m.size), want["size"], "{f} size");
                assert_eq!(json!(m.missing), want["missing"], "{f} missing");
                let parts: Vec<String> = split_parts(Path::new(f))
                    .iter()
                    .map(|x| x.file_name().unwrap().to_string_lossy().to_string())
                    .collect();
                assert_eq!(json!(parts), want["parts"], "{f} parts");
            }
            (Err(e), Some(m)) => {
                if m == "OSError" {
                    continue; // the wording of the operating system is not ours to copy
                }
                assert_eq!(json!(e.0), *m, "{f}");
            }
            (a, b) => panic!("{f}: rust {:?} python {b:?}", a.map(|m| m.name)),
        }
    }
    for (i, n) in split_names.iter().enumerate() {
        let got = split_match(n)
            .map(|(a, b, c)| json!([a, b, c]))
            .unwrap_or(Value::Null);
        assert_eq!(got, py["split"][i], "split_match({n})");
    }
    // the cache must not hide a change
    let again = inspect(Path::new(&files[0])).unwrap();
    assert_eq!(
        again.public(),
        inspect_cached(Path::new(&files[0])).unwrap().public()
    );
    eprintln!(
        "compared with the Python reader: {} files and {} names identical",
        files.len(),
        split_names.len()
    );
}
